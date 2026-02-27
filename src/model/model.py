"""
NeuromorphicLM (v4) — iterative refinement with cortical columns.

Processes N-token segments through R iterative passes.  All B_blocks * C
columns process all N tokens in parallel via a single CorticalColumnGroup
with G = B*C groups.  PM/EM state has explicit B dim: [BS, B, ...].
No Python loop over blocks — only a loop over R refinement passes.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .column import CorticalColumnGroup
from .procedural_memory import ProceduralMemory, PMNeuromodulator
from .episodic_memory import EpisodicMemory, EMNeuromodulator
from .utils import runtime_state_dtype, unit_normalize


class NeuromorphicLM(nn.Module):
    """v4: Iterative refinement with cortical columns (block-batched)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        B = config.B_blocks
        C = config.C
        self.B_C = B * C  # G

        # Embedding + position
        self.embedding = nn.Embedding(config.vocab_size, config.D)
        self.pos_embedding = nn.Embedding(config.N, config.D)

        # Fan-out: D -> G * D_col
        total_col_dim = self.B_C * config.D_col
        self.fan_out = nn.Linear(config.D, total_col_dim)

        # Single merged column group (G = B*C groups)
        self.columns = CorticalColumnGroup(config)

        # Single PM + EM (state: [BS, B, ...])
        self.pm = ProceduralMemory(config.D_mem, config.r, config)
        self.em = EpisodicMemory(config.D_mem, config.M, config)

        # Single neuromodulator pair (shared across blocks)
        self.pm_neuromod = PMNeuromodulator(config)
        self.em_neuromod = EMNeuromodulator(config)

        # Fan-in: G * D_col -> D
        self.fan_in = nn.Linear(total_col_dim, config.D)
        self.ln_final = nn.LayerNorm(config.D)

        # LM head (optionally tied to embedding)
        self.lm_head = nn.Linear(config.D, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Damped mixing parameter (sigmoid -> lambda_mix init)
        # logit(0.5) = 0.0
        init_logit = torch.log(torch.tensor(config.lambda_mix / (1 - config.lambda_mix + 1e-8)))
        self.lambda_logit = nn.Parameter(torch.tensor(float(init_logit)))

    def forward_segment(self, input_ids: Tensor, reset_mask: Tensor | None = None):
        """Process N tokens through R refinement passes.

        input_ids: [BS, N]
        reset_mask: [BS] bool — streams to reset PM/EM before processing

        Returns: (logits [BS, N, vocab], aux_loss scalar)
        """
        if reset_mask is not None:
            self._reset_memory(reset_mask)

        BS, N = input_ids.shape
        B, C, G = self.config.B_blocks, self.config.C, self.B_C
        device = input_ids.device

        x = self.embedding(input_ids)  # [BS, N, D]
        positions = torch.arange(N, device=device)
        x = x + self.pos_embedding(positions)

        # Fan-out to column space
        x_flat = self.fan_out(x)  # [BS, N, G*D_col]
        x_cols = x_flat.view(BS, N, G, self.config.D_col)

        z_hat_prev = None
        aux_loss = torch.tensor(0.0, device=device)
        lam = torch.sigmoid(self.lambda_logit)

        for r in range(self.config.R):
            # All G columns, all N tokens, one batched call
            x_out, z, z_hat, surprise, elig_info, nov_info = \
                self.columns.forward(x_cols, self.pm, self.em, z_hat_prev)

            # PCM prediction loss
            if self.config.pcm_enabled and z_hat_prev is not None and z is not None:
                aux_loss = aux_loss + self.columns.pcm.prediction_loss(z_hat_prev, z) * self.config.pcm_pred_weight

            # PM eligibility routing + commit, EM novelty + write
            self._process_and_commit(elig_info, nov_info, x_out)

            # Damped mixing
            if r > 0:
                x_cols = (1 - lam) * x_cols + lam * x_out
            else:
                x_cols = x_out

            z_hat_prev = z_hat

        # Fan-in
        x = x_cols.reshape(BS, N, -1)   # [BS, N, G*D_col]
        x = self.fan_in(x)              # [BS, N, D]
        x = self.ln_final(x)
        logits = self.lm_head(x)        # [BS, N, vocab]

        return logits, aux_loss

    # ------------------------------------------------------------------
    # PM eligibility routing + commit, EM novelty + write
    # ------------------------------------------------------------------

    def _process_and_commit(self, elig_info, nov_info, x_out):
        """Route eligibility to PM slots, score EM novelty, then commit/write.

        elig_info: (k_cand, v_cand, gate) — all [BS, N, B, C, D_mem] or [BS, N, B, C]
        nov_info: (q_nov, v_nov, w_nov, surprise) — same shapes
        x_out: [BS, N, G, D_col] (unused directly, info extracted in column)
        """
        k_cand, v_cand, gate = elig_info
        BS, N, B, C, D_mem = k_cand.shape

        # --- PM eligibility routing (5D einsums) ---
        if self.config.pm_enabled:
            k_norm = unit_normalize(k_cand)
            route_scores = torch.einsum(
                "snbcd, sbrd -> snbcr", k_norm, self.pm.pm_K
            )  # [BS, N, B, C, r]
            route_w = torch.softmax(
                route_scores / self.config.tau_route_pm, dim=-1
            )  # [BS, N, B, C, r]
            gated_routed = gate.unsqueeze(-1) * route_w  # [BS, N, B, C, r]
            elig_K = torch.einsum("snbcr, snbcd -> sbrd", gated_routed, k_cand)
            elig_V = torch.einsum("snbcr, snbcd -> sbrd", gated_routed, v_cand)
        else:
            elig_K = torch.zeros(
                BS, B, self.config.r, D_mem,
                device=k_cand.device, dtype=k_cand.dtype,
            )
            elig_V = torch.zeros_like(elig_K)

        # --- EM novelty scoring + candidate selection ---
        q_nov, v_nov, w_nov, surp = nov_info

        if self.config.em_enabled:
            novelty = self.em.score_novelty(q_nov, surp, w_nov)
            em_cands = self.em.select_top_candidates(
                q_nov, v_nov, novelty, self.config.C_em
            )
        else:
            em_cands = (
                torch.zeros(BS, B, self.config.C_em, D_mem, device=k_cand.device, dtype=k_cand.dtype),
                torch.zeros(BS, B, self.config.C_em, D_mem, device=k_cand.device, dtype=k_cand.dtype),
                torch.zeros(BS, B, self.config.C_em, device=k_cand.device, dtype=k_cand.dtype),
            )

        # --- PM commit ---
        if self.config.pm_enabled:
            self.pm.base_decay()
            # Neuromod: flatten [BS, B] -> [BS*B] for MLP, reshape back
            BSB = BS * B
            elig_summary = elig_K.norm(dim=-1).mean(dim=-1)  # [BS, B]
            pm_usage = self.pm.pm_a.sum(dim=-1)              # [BS, B]
            content_emb = elig_K.mean(dim=2)                 # [BS, B, D_mem]
            g, slot_logits, tau = self.pm_neuromod(
                elig_summary.reshape(BSB).detach(),
                pm_usage.reshape(BSB).detach(),
                content_emb.reshape(BSB, D_mem).detach(),
            )
            # Reshape neuromod outputs back to [BS, B, ...]
            g = g.reshape(BS, B)
            slot_logits = slot_logits.reshape(BS, B, -1)
            tau = tau.reshape(BS, B)
            self.pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # --- EM write ---
        cand_K, cand_V, cand_scores = em_cands
        if self.config.em_enabled:
            em_usage = self.em.em_S.sum(dim=-1)              # [BS, B]
            novelty_mean = cand_scores.mean(dim=-1)          # [BS, B]
            content_emb = cand_K.mean(dim=2)                 # [BS, B, D_mem]
            BSB = BS * B
            g_em, tau_em, decay = self.em_neuromod(
                novelty_mean.reshape(BSB).detach(),
                em_usage.reshape(BSB).detach(),
                content_emb.reshape(BSB, D_mem).detach(),
            )
            # Reshape back to [BS, B]
            g_em = g_em.reshape(BS, B)
            tau_em = tau_em.reshape(BS, B)
            decay = decay.reshape(BS, B)
            self.em.write(cand_K, cand_V, cand_scores, g_em, tau_em, decay)
            self.em.age_tick(self.config.N)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_memory(self, mask: Tensor):
        """Reset PM/EM for masked streams at doc boundary.

        mask: [BS] bool.  PM/EM handle expansion internally.
        """
        if not self.config.lifelong_mode:
            self.pm.reset_content(mask)
            self.em.reset_states(mask)

    def initialize_states(self, BS: int, device: torch.device):
        """Pre-allocate runtime state tensors (with explicit B dim)."""
        dtype = runtime_state_dtype(device)
        self.pm.initialize(BS, device, dtype)
        self.em.initialize(BS, device, dtype)

    def detach_states(self):
        """TBPTT boundary: detach all PM/EM state."""
        self.pm.detach_states()
        self.em.detach_states()

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        return self.train(False)
