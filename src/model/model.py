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
            ScanLayer(C, D_col, expansion, config.dropout) for _ in range(L)
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
            ScanLayer(C, D_col, expansion, config.dropout) for _ in range(L)
        ])

        # Output
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward_segment(
        self, input_ids: Tensor, reset_mask: Tensor | None = None,
        commit: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Process one N-token segment through the three-stage cycle.

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

        # --- Stage 2: Memory Ops (fused algebra — no [BS,N,B,D] tensors) ---
        H_flat = H.reshape(BS, N, D)
        pm_read_sum, em_read_sum, cum_em_sum = self._memory_ops(
            H_flat, seed, w_cand, surp_flat, commit=commit
        )

        # --- Stage 3: Integration Scan ---
        # Baseline H_flat added once + bank-summed memory deltas (already [BS,N,D])
        integrated = H_flat + pm_read_sum + em_read_sum + cum_em_sum
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
        self, H_flat: Tensor, seed: Tensor, w_cand: Tensor, surprise: Tensor,
        commit: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Stage 2: Write-before-read with causal prefix sums (fused algebra).

        Algebraic fusion: the sum over B banks is computed without ever
        materializing [BS, N, B, D] tensors.  PM deltas and EM write buffers
        share surprise/w_cand across banks (differing only by a per-bank
        scalar), so the bank-sum collapses to scalar sums of per-bank factors.

        Args:
            H_flat: [BS, N, D] — column states (flat)
            seed: [BS, N, D] — EM trail seeds
            w_cand: [BS, N, D] — write candidates
            surprise: [BS, N, D] — vector surprise
            commit: whether to commit memory updates at segment end

        Returns:
            pm_read_sum: [BS, N, D] — PM modulation deltas, summed over B
            em_read_sum: [BS, N, D] — EM trail reads, summed over B
            cum_em_sum:  [BS, N, D] — EM causal write buffers, summed over B
        """
        BS, N, D = H_flat.shape
        B = self.config.B

        # 1. PM: fused read sum — no [BS, N, B, D]
        #    sum_b[H * (bias_b + cumsum(lr_b * surprise))]
        #    = H * (bias.sum(B) + lr.sum() * cumsum(surprise))
        if self.config.pm_enabled:
            lr_pm = self.pm.lr_pm                                     # [B]
            bias_sum = self.pm.pm_bias.sum(dim=1, keepdim=True)       # [BS, 1, D]
            cum_surprise = torch.cumsum(surprise, dim=1)              # [BS, N, D]
            pm_read_sum = H_flat * (bias_sum + lr_pm.sum() * cum_surprise)
        else:
            pm_read_sum = torch.zeros(BS, N, D, device=H_flat.device, dtype=H_flat.dtype)

        # 2. EM novelty (still returns [BS, N, B] — needed for commit)
        if self.config.em_enabled:
            w_nov = torch.sigmoid(self.W_nov(H_flat))                 # [BS, N, B]
            novelty = self.em.compute_novelty_all(
                w_cand, surprise, w_nov=w_nov,
            )                                                          # [BS, N, B]
            # Write buffer sum: fused — no [BS, N, B, D]
            #   sum_b[cumsum(novelty_b * w_cand)] = cumsum(novelty.sum(B) * w_cand)
            nov_sum = novelty.sum(dim=2, keepdim=True)                # [BS, N, 1]
            cum_em_sum = torch.cumsum(nov_sum * w_cand, dim=1)        # [BS, N, D]
        else:
            novelty = torch.zeros(BS, N, B, device=H_flat.device, dtype=H_flat.dtype)
            cum_em_sum = torch.zeros(BS, N, D, device=H_flat.device, dtype=H_flat.dtype)

        # 3. EM trail read (returns [BS, N, D] — summed over B internally)
        if self.config.em_enabled:
            em_read_sum = self.em.trail_read_all(seed)                # [BS, N, D]
        else:
            em_read_sum = torch.zeros(BS, N, D, device=H_flat.device, dtype=H_flat.dtype)

        # 4. Segment-end commits — skipped for commit=False (inference)
        if commit:
            if self.config.pm_enabled:
                # cum_surprise[:, -1] == surprise.sum(1) — reuse existing cumsum
                pm_commit = self.pm.lr_pm[None, :, None] * cum_surprise[:, -1].unsqueeze(1)
                self.pm.commit(pm_commit)

            if self.config.em_enabled:
                g_em = self.em_neuromod(
                    novelty.mean(dim=1).detach(),
                    self.em.usage_all().detach(),
                )                                                      # [BS, B]
                self.em.commit_all(w_cand, novelty, g_em)
                self.em.base_decay()

        return pm_read_sum, em_read_sum, cum_em_sum

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
