"""
NeuromorphicLM (v7) — single scan stack, dense scan + grouped PCM.

Single scan stack per segment of N tokens:
  layers[0..L_mem-1]: build representation (H)
  PCM: compute surprise; W_seed_w: produce seed + w_cand
  PM read + EM trail read → additive injection
  layers[L_mem..L_total-1]: integrate with memory context
  output head → logits

Memory commits happen once at segment end (no within-segment writes).
Scan layers are fully dense for GPU efficiency. PCM and W_seed_w remain
grouped (per-feature-group) via free .view() reshaping between [BS,N,D]
and [BS,N,C,D_col].
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
    """v7: Single scan stack with memory injection at L_mem."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        B = config.B
        C = config.C
        D = config.D
        D_col = config.D_col
        D_embed = config.D_embed
        L_total = config.L_total
        d_inner = config.d_inner

        # Embedding + position
        self.embedding = nn.Embedding(config.vocab_size, D_embed)
        if D_embed != D:
            self.proj_up = nn.Linear(D_embed, D)
            self.proj_down = nn.Linear(D, D_embed)
        else:
            self.proj_up = None
            self.proj_down = None
        self.pos_embed = nn.Parameter(torch.randn(config.N, D) * 0.02)

        # Single scan stack: L_total dense layers — [BS, N, D] throughout
        self.layers = nn.ModuleList([
            ScanLayer(D, d_inner, config.dropout, n_layers=L_total,
                      glu_output=config.glu_output) for _ in range(L_total)
        ])

        # Projections: grouped (per-feature-group seeds + write candidates)
        self.W_seed_w = GroupedLinear(C, D_col, 2 * D_col)  # -> (seed, w_cand)

        # PCM (within-scan prediction) — grouped, operates on [BS, N, C, D_col]
        if config.pcm_enabled:
            self.pcm = WithinScanPCM(C, D_col)
        else:
            self.pcm = None

        # Memory systems
        self.pm = ProceduralMemory(B, D, config.D_pm, config.decay_pm)
        self.em = EpisodicMemory(
            B, config.M, D, config.n_trail_steps,
            D_mem=config.D_mem, S_max=config.S_max, budget=config.budget_em,
            decay=config.decay_em, topk=config.em_topk,
        )
        self.em_neuromod = EMNeuromodulator(
            hidden=config.neuromod_hidden,
        )

        # Output
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward_segment(
        self, input_ids: Tensor, reset_mask: Tensor | None = None,
        commit: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Process one N-token segment through single scan stack + memory.

        Args:
            input_ids: [BS, N]
            reset_mask: [BS] bool — streams to reset PM/EM before processing
            commit: whether to commit memory updates at segment end
                    (False for inference with overlapping windows)

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
        L_mem = self.config.L_mem
        device = input_ids.device

        # --- Embedding ---
        x = self.embedding(input_ids)               # [BS, N, D_embed]
        if self.proj_up is not None:
            x = self.proj_up(x)                     # [BS, N, D]
        x = x + self.pos_embed[:N]                  # [BS, N, D]

        # --- Pre-memory layers: layers[0..L_mem-1] ---
        H = x
        for i in range(L_mem):
            layer = self.layers[i]
            carry = self._carries[i] if hasattr(self, '_carries') else None
            if self.config.gradient_checkpointing and self.training:
                H, h_last = checkpoint(layer, H, carry, use_reentrant=False)
            else:
                H, h_last = layer(H, carry)          # [BS, N, D]
            if hasattr(self, '_carries'):
                self._carries[i] = h_last

        # --- PCM: view as grouped for per-feature-group surprise ---
        aux_loss = torch.tensor(0.0, device=device)
        if self.pcm is not None:
            H_col = H.view(BS, N, C, D_col)             # free view
            x_col = x.view(BS, N, C, D_col)             # free view
            surprise_col, z_hat, z = self.pcm.compute_surprise(H_col, x_col)
            H_col = self.pcm.apply_gain(H_col, surprise_col)
            H = H_col.view(BS, N, D)                    # back to dense (free)
            aux_loss = self.pcm.prediction_loss(z_hat, z) * self.config.pcm_pred_weight
            surprise = surprise_col.reshape(BS, N, D)
        else:
            surprise = torch.zeros(BS, N, D, device=device, dtype=x.dtype)

        # --- Projections — grouped (per-feature-group seeds + write candidates) ---
        H_col = H.view(BS, N, C, D_col)                 # free view
        sw = self.W_seed_w(H_col)                        # [BS, N, C, 2*D_col]
        seed_col, w_col = sw.chunk(2, dim=-1)            # each [BS, N, C, D_col]
        seed = seed_col.reshape(BS, N, D)                # [BS, N, D]
        w_cand = w_col.reshape(BS, N, D)                 # [BS, N, D]

        # --- Memory reads (read-only, no within-segment writes) ---
        pm_read_sum, em_read_sum, pm_pre = self._memory_reads(H, seed)

        # Debug: activation + memory diagnostics
        if self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling():
            with torch.no_grad():
                self._dbg_act_norms = {
                    "H": H.norm().item(),
                    "pm": pm_read_sum.norm().item(),
                    "em": em_read_sum.norm().item(),
                }
                stats = {}
                if self.config.pm_enabled:
                    s_gate = torch.sigmoid(surprise.norm(dim=-1))
                    stats["pm_surprise_gate_mean"] = s_gate.mean().item()
                self._dbg_memory_stats = stats

        # --- Additive memory injection ---
        H = H + pm_read_sum + em_read_sum

        # --- Post-memory layers: layers[L_mem..L_total-1] ---
        for i in range(L_mem, self.config.L_total):
            layer = self.layers[i]
            carry = self._carries[i] if hasattr(self, '_carries') else None
            if self.config.gradient_checkpointing and self.training:
                H, h_last = checkpoint(layer, H, carry, use_reentrant=False)
            else:
                H, h_last = layer(H, carry)          # [BS, N, D]
            if hasattr(self, '_carries'):
                self._carries[i] = h_last

        # --- Segment-end memory commits ---
        if commit:
            self._memory_commits(pm_pre, surprise, w_cand)

        # --- Output ---
        out = H
        if self.proj_down is not None:
            out = self.proj_down(out)                # [BS, N, D_embed]
        out = self.ln_final(out)
        logits = self.lm_head(out)                   # [BS, N, vocab]
        # Scale logits for tied embeddings: std ≈ √D_embed at init → normalize to ~1
        logits = logits * (self.config.D_embed ** -0.5)

        return logits, aux_loss

    def _memory_reads(
        self, H: Tensor, seed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Read-only memory operations at the injection point.

        Args:
            H: [BS, N, D] — representation after pre-memory layers
            seed: [BS, N, D] — EM trail seeds

        Returns:
            pm_read_sum: [BS, N, D] — PM fast-weight read, bank-summed
            em_read_sum: [BS, N, D] — EM trail reads, summed over B
            pm_pre: [BS, N, D_pm] or None — PM pre-activations (for commit)
        """
        BS, N, D = H.shape

        # PM: Hebbian fast-weight read (bank-summed via W.sum(B))
        if self.config.pm_enabled:
            pm_read_sum, pm_pre = self.pm.read(H)        # [BS, N, D], [BS, N, D_pm]
        else:
            pm_read_sum = torch.zeros(BS, N, D, device=H.device, dtype=H.dtype)
            pm_pre = None

        # EM trail read (returns [BS, N, D] — summed over B internally)
        if self.config.em_enabled:
            em_read_sum = self.em.trail_read_all(seed)    # [BS, N, D]
        else:
            em_read_sum = torch.zeros(BS, N, D, device=H.device, dtype=H.dtype)

        return pm_read_sum, em_read_sum, pm_pre

    def _memory_commits(
        self, pm_pre: Tensor | None, surprise: Tensor, w_cand: Tensor,
    ):
        """Segment-end memory commits (PM Hebbian + EM neuromodulated).

        Args:
            pm_pre: [BS, N, D_pm] or None — PM pre-activations
            surprise: [BS, N, D] — vector surprise from PCM
            w_cand: [BS, N, D] — write candidates
        """
        # PM commit: Hebbian update
        if self.config.pm_enabled and pm_pre is not None:
            self.pm.commit(pm_pre, surprise, budget=self.config.budget_pm)

        # EM commit: novelty from surprise norm, neuromodulator at segment level
        if self.config.em_enabled:
            BS = w_cand.shape[0]
            B = self.config.B

            # Novelty = write candidate energy + surprise magnitude
            # w_cand.norm() ensures non-zero writes even without PCM (surprise=0)
            # surprise.norm() adds discriminative gating when PCM is active
            novelty = (w_cand.norm(dim=-1, keepdim=True)
                       + surprise.norm(dim=-1, keepdim=True)).expand(BS, -1, B)  # [BS, N, B]

            # Decay existing strengths before computing usage — ensures
            # raw_decay gets gradient through: decay → em_S → usage → neuromod → g_em → alpha → em_K/V
            self.em.base_decay()

            # Segment-level neuromodulator: mean novelty + usage → g_em [BS, B]
            mean_novelty = novelty.mean(dim=1)                    # [BS, B]
            usage = self.em.usage_all()                            # [BS, B]
            g_em = self.em_neuromod(mean_novelty, usage)           # [BS, B]

            # Debug stats
            if self.training and torch.is_grad_enabled() and not torch.compiler.is_compiling():
                with torch.no_grad():
                    if not hasattr(self, '_dbg_memory_stats'):
                        self._dbg_memory_stats = {}
                    self._dbg_memory_stats["em_novelty_mean"] = novelty.mean().item()
                    self._dbg_memory_stats["em_g_em_mean"] = g_em.mean().item()

            self.em.commit_all(w_cand, novelty, g_em)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_memory(self, mask: Tensor):
        """Reset PM/EM and scan carries for masked streams at doc boundary."""
        if not self.config.lifelong_mode:
            self.pm.reset_states(mask)
            self.em.reset_states(mask)
            if hasattr(self, '_carries'):
                mask_f = (~mask).to(dtype=torch.float32).unsqueeze(-1)  # [BS, 1]
                for i, h in enumerate(self._carries):
                    if h is not None:
                        self._carries[i] = h * mask_f

    def initialize_states(self, BS: int, device: torch.device):
        """Pre-allocate runtime state tensors."""
        dtype = runtime_state_dtype(device)
        self.pm.initialize(BS, device, dtype)
        self.em.initialize(BS, device, dtype)
        self._carries = [None] * len(self.layers)

    def detach_states(self):
        """TBPTT boundary: detach all PM/EM and scan carry state."""
        self.pm.detach_states()
        self.em.detach_states()
        if hasattr(self, '_carries'):
            self._carries = [h.detach() if h is not None else None for h in self._carries]

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> Tensor:
        """Autoregressive generation with proper memory handling.

        Prefill: non-overlapping N-token segments with memory commits.
        Decode: sliding window with commit=False (memory frozen after prefill).

        prompt_ids: [BS, P]
        Returns: [BS, P + max_new_tokens]
        """
        N = self.config.N
        BS = prompt_ids.shape[0]
        device = prompt_ids.device

        if not self.pm.is_initialized() or not self.em.is_initialized():
            self.initialize_states(BS, device)

        # --- Prefill: non-overlapping segments WITH memory commits ---
        P = prompt_ids.shape[1]
        logits = None
        for seg_start in range(0, P, N):
            seg = prompt_ids[:, seg_start:min(seg_start + N, P)]
            actual_len = seg.shape[1]
            if actual_len < N:
                seg = F.pad(seg, (0, N - actual_len), value=0)
            logits, _ = self.forward_segment(seg)

        # Position of last real prompt token in the last segment
        last_pos = ((P - 1) % N) if P > 0 else 0

        # --- Decode: sliding window WITHOUT memory commits ---
        sequence = prompt_ids
        for i in range(max_new_tokens):
            if i == 0:
                next_logits = logits[:, last_pos, :]
            else:
                ctx = sequence[:, -N:]
                if ctx.shape[1] < N:
                    ctx = F.pad(ctx, (N - ctx.shape[1], 0), value=0)
                logits, _ = self.forward_segment(ctx, commit=False)
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
