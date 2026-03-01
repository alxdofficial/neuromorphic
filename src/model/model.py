"""
NeuromorphicLM (v5) — scan-memory-scan with cortical columns.

Three-stage cycle per segment of N tokens:
  Stage 1: Fast causal scan (C independent columns, element-wise recurrence)
  Stage 2: Memory ops (write-before-read with causal prefix sums)
  Stage 3: Integration scan (second causal scan integrating memory reads)

All C columns process every token. Each column sees a D_col = D/C feature slice.
B memory banks, each addressed by column D_col slices.
NTP training — causal scans are inherently autoregressive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .scan import ScanLayer
from .predictive_coding import GroupedLinear, WithinScanPCM
from .procedural_memory import ProceduralMemory
from .episodic_memory import EpisodicMemory, EMNeuromodulator
from .utils import runtime_state_dtype


class NeuromorphicLM(nn.Module):
    """v5: Scan-memory-scan with cortical columns."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        B = config.B
        C = config.C
        D = config.D
        D_col = config.D_col
        D_embed = config.D_embed
        L = config.L_scan
        expansion = config.scan_expansion

        # Embedding + position
        self.embedding = nn.Embedding(config.vocab_size, D_embed)
        if D_embed != D:
            self.proj_up = nn.Linear(D_embed, D)
            self.proj_down = nn.Linear(D, D_embed)
        else:
            self.proj_up = None
            self.proj_down = None
        self.pos_embed = nn.Parameter(torch.randn(config.N, D) * 0.02)

        # Stage 1: L_scan layers
        self.stage1 = nn.ModuleList([
            ScanLayer(C, D_col, expansion) for _ in range(L)
        ])

        # Stage 1 projections (shared across banks) — fused into single matmul
        self.W_seed_w = GroupedLinear(C, D_col, 2 * D_col)  # -> (seed, w_cand)

        # PCM (within-scan prediction)
        if config.pcm_enabled:
            self.pcm = WithinScanPCM(C, D_col)
        else:
            self.pcm = None

        # Novelty blend weight: produces per-bank blend for surprise vs recon_error
        self.W_nov = nn.Linear(D, B)
        nn.init.zeros_(self.W_nov.weight)
        nn.init.zeros_(self.W_nov.bias)

        # Stage 2: Memory systems
        self.pm = ProceduralMemory(B, D, config.decay_pm)
        self.em = EpisodicMemory(
            B, config.M, D, config.n_trail_steps,
            S_max=config.S_max, budget=config.budget_em,
            decay=config.decay_em,
        )
        self.em_neuromod = EMNeuromodulator(
            hidden=config.neuromod_hidden,
        )

        # Stage 3: L_scan layers
        self.stage3 = nn.ModuleList([
            ScanLayer(C, D_col, expansion) for _ in range(L)
        ])

        # Output
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward_segment(
        self, input_ids: Tensor, reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Process one N-token segment through the three-stage cycle.

        Args:
            input_ids: [BS, N]
            reset_mask: [BS] bool — streams to reset PM/EM before processing

        Returns:
            logits: [BS, N, vocab]
            aux_loss: scalar
        """
        if reset_mask is not None:
            self._reset_memory(reset_mask)

        BS, N = input_ids.shape
        C = self.config.C
        D_col = self.config.D_col
        D = self.config.D
        device = input_ids.device

        # --- Stage 1: Fast Scan ---
        x = self.embedding(input_ids)               # [BS, N, D_embed]
        if self.proj_up is not None:
            x = self.proj_up(x)                     # [BS, N, D]
        x = x + self.pos_embed[:N]                  # [BS, N, D]

        x_col = x.view(BS, N, C, D_col)            # [BS, N, C, D_col]

        H = x_col
        for layer in self.stage1:
            if self.config.gradient_checkpointing and self.training:
                H, _ = checkpoint(layer, H, use_reentrant=False)
            else:
                H, _ = layer(H)                     # [BS, N, C, D_col]

        # PCM: within-scan prediction and vector surprise
        # Applied BEFORE projections so seed/w_cand benefit from gain signal
        aux_loss = torch.tensor(0.0, device=device)
        if self.pcm is not None:
            surprise, z_hat, z = self.pcm.compute_surprise(H, x_col)
            H = self.pcm.apply_gain(H, surprise)
            aux_loss = self.pcm.prediction_loss(z_hat, z) * self.config.pcm_pred_weight
            surp_flat = surprise.reshape(BS, N, D)
        else:
            surp_flat = torch.zeros(BS, N, D, device=device, dtype=x.dtype)

        # Projections from gain-modulated scan states (fused single matmul)
        sw = self.W_seed_w(H)                          # [BS, N, C, 2*D_col]
        seed_col, w_col = sw.chunk(2, dim=-1)          # each [BS, N, C, D_col]
        seed = seed_col.reshape(BS, N, D)              # [BS, N, D]
        w_cand = w_col.reshape(BS, N, D)               # [BS, N, D]

        # --- Stage 2: Memory Ops ---
        H_flat = H.reshape(BS, N, D)
        pm_deltas, em_reads, cum_ems = self._memory_ops(
            H_flat, seed, w_cand, surp_flat
        )

        # --- Stage 3: Integration Scan ---
        # Baseline H_flat added once + bank-summed memory deltas
        integrated = H_flat + pm_deltas.sum(2) + em_reads.sum(2) + cum_ems.sum(2)
        H_prime = integrated.view(BS, N, C, D_col)

        for layer in self.stage3:
            if self.config.gradient_checkpointing and self.training:
                H_prime, _ = checkpoint(layer, H_prime, use_reentrant=False)
            else:
                H_prime, _ = layer(H_prime)

        # Output
        out = H_prime.reshape(BS, N, D)
        if self.proj_down is not None:
            out = self.proj_down(out)                # [BS, N, D_embed]
        out = self.ln_final(out)
        logits = self.lm_head(out)                   # [BS, N, vocab]

        return logits, aux_loss

    def _memory_ops(
        self, H_flat: Tensor, seed: Tensor, w_cand: Tensor, surprise: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Stage 2: Write-before-read with causal prefix sums.

        All bank operations are vectorized — no per-bank Python loop.
        PM returns delta only (no baseline H) — baseline is added once
        in forward_segment's integration step.

        Args:
            H_flat: [BS, N, D] — column states (flat)
            seed: [BS, N, D] — EM trail seeds
            w_cand: [BS, N, D] — write candidates
            surprise: [BS, N, D] — vector surprise

        Returns:
            pm_deltas: [BS, N, B, D] — PM modulation deltas (no baseline)
            em_reads: [BS, N, B, D] — EM trail reads
            cum_ems: [BS, N, B, D] — EM causal write buffers
        """
        BS, N, D = H_flat.shape
        B = self.config.B

        # 1. PM write deltas and prefix sums (all banks)
        delta_pm = self.pm.compute_deltas(surprise)       # [BS, N, B, D]
        cum_pm = torch.cumsum(delta_pm, dim=1)            # [BS, N, B, D]

        # 2. EM novelty and write deltas (all banks)
        if self.config.em_enabled:
            w_nov = torch.sigmoid(self.W_nov(H_flat))     # [BS, N, B]
            novelty = self.em.compute_novelty_all(
                w_cand, surprise, w_nov=w_nov
            )                                              # [BS, N, B]
            delta_em = self.em.compute_write_deltas(novelty, w_cand)  # [BS, N, B, D]
        else:
            novelty = torch.zeros(BS, N, B, device=H_flat.device, dtype=H_flat.dtype)
            delta_em = torch.zeros(BS, N, B, D, device=H_flat.device, dtype=H_flat.dtype)

        cum_em = torch.cumsum(delta_em, dim=1)            # [BS, N, B, D]

        # 3. PM read — delta only (baseline H added once in forward_segment)
        if self.config.pm_enabled:
            pm_deltas = self.pm.read_all(H_flat, cum_pm)  # [BS, N, B, D]
        else:
            pm_deltas = torch.zeros(BS, N, B, D, device=H_flat.device, dtype=H_flat.dtype)

        # 4. EM trail read (all banks, from frozen primitives)
        if self.config.em_enabled:
            em_reads = self.em.trail_read_all(seed)       # [BS, N, B, D]
        else:
            em_reads = torch.zeros(BS, N, B, D, device=H_flat.device, dtype=H_flat.dtype)

        # 5. Segment-end commits (all banks)
        if self.config.pm_enabled:
            self.pm.commit(delta_pm.sum(dim=1))           # delta_pm_sum: [BS, B, D]

        if self.config.em_enabled:
            # novelty: [BS, N, B] -> mean over N -> [BS, B]
            g_em = self.em_neuromod(
                novelty.mean(dim=1).detach(),
                self.em.usage_all().detach(),
            )                                              # [BS, B]
            self.em.commit_all(w_cand, novelty, g_em)
            self.em.base_decay()

        return pm_deltas, em_reads, cum_em

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_memory(self, mask: Tensor):
        """Reset PM/EM for masked streams at doc boundary."""
        if not self.config.lifelong_mode:
            self.pm.reset_states(mask)
            self.em.reset_states(mask)

    def initialize_states(self, BS: int, device: torch.device):
        """Pre-allocate runtime state tensors."""
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

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> Tensor:
        """Simple autoregressive generation using N-token windows.

        prompt_ids: [BS, P]
        Returns: [BS, P + max_new_tokens]
        """
        N = self.config.N
        BS = prompt_ids.shape[0]
        device = prompt_ids.device
        sequence = prompt_ids

        if not self.pm.is_initialized() or not self.em.is_initialized():
            self.initialize_states(BS, device)

        for _ in range(max_new_tokens):
            ctx = sequence[:, -N:]
            pad_len = N - ctx.shape[1]
            if pad_len > 0:
                ctx = F.pad(ctx, (pad_len, 0), value=0)

            logits, _ = self.forward_segment(ctx)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, _ = next_logits.topk(top_k, dim=-1)
                next_logits[next_logits < topk_vals[:, -1:]] = -float('inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            sequence = torch.cat([sequence, next_id], dim=1)

        return sequence

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        return self.train(False)
