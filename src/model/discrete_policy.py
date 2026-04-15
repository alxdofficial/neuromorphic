"""Discrete action policy for the neuromodulator.

Replaces the old encoder→VQ→decoder pipeline with a single integrated module:
the modulator's logit_head emits a categorical distribution over K codes per
cell; a shared codebook maps codes to low-dim embeddings; a shared decoder maps
embeddings to the continuous action space (delta_W, delta_decay).

Key architectural commitments (see docs/design.md + chat log):
- Per-cell logit heads: each of NC cells has its own MLP (mod_in → K logits).
  Cells choose codes independently given shared obs features.
- Shared codebook + decoder: all cells draw from one vocabulary of K
  "memory-update templates." Simpler + prevents under-training per-cell codes.
- Phase 1 (end-to-end backprop): Gumbel-softmax with temperature τ for
  differentiable sampling. τ anneals 1.0→0.3 over the LR schedule (set
  by train.py's step_callback; see `MemoryGraph.gumbel_tau`).
- Phase 2 (GRPO): hard Categorical sampling, log_prob via standard log_softmax.
  Codebook + decoder frozen; only logit_head trains.
- Dead-code reset: wired in train.py bootstrap step_callback only (cycle
  phase 1 + phase 2 have the codebook frozen, so no reset is needed or
  useful there).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiscreteActionPolicy(nn.Module):
    """Policy network: obs → logits → code → embedding → continuous action.

    Shapes (B = batch, NC = cells):
        logits:     [B, NC, K]
        codes:      [B, NC]                  long, index in [0, K)
        embedding:  [B, NC, D]               shared codebook lookup
        action:     [B, NC, action_dim]      shared decoder output

    Parameters:
        logit_head: per-cell 2-layer MLP [NC, mod_in→Hmod→K]  — the "policy"
        codebook:   buffer [K, D]  — frozen after bootstrap
        decoder:    shared 2-layer MLP [D→Hdec→action_dim]  — frozen after bootstrap
    """

    def __init__(
        self,
        n_cells: int,
        mod_in_dim: int,
        action_dim: int,
        num_codes: int = 256,
        code_dim: int = 32,
        logit_hidden: int = 80,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.NC = n_cells
        self.mod_in_dim = mod_in_dim
        self.action_dim = action_dim
        self.K = num_codes
        self.D = code_dim

        # Per-cell logit head: einsum pattern "bni,nih->bnh" then "bnh,nhk->bnk"
        self.logit_w1 = nn.Parameter(
            torch.empty(n_cells, mod_in_dim, logit_hidden))
        self.logit_b1 = nn.Parameter(torch.zeros(n_cells, logit_hidden))
        self.logit_w2 = nn.Parameter(
            torch.empty(n_cells, logit_hidden, num_codes))
        self.logit_b2 = nn.Parameter(torch.zeros(n_cells, num_codes))
        # Xavier init
        nn.init.normal_(self.logit_w1,
                        std=math.sqrt(2.0 / (mod_in_dim + logit_hidden)))
        nn.init.normal_(self.logit_w2,
                        std=math.sqrt(2.0 / (logit_hidden + num_codes)))

        # Shared codebook — learned embedding per code. Small scale so initial
        # decoder outputs aren't saturating.
        codebook = torch.randn(num_codes, code_dim) * (code_dim ** -0.5)
        self.codebook = nn.Parameter(codebook)

        # Shared decoder: code embedding → continuous action
        self.dec_w1 = nn.Parameter(torch.empty(code_dim, decoder_hidden))
        self.dec_b1 = nn.Parameter(torch.zeros(decoder_hidden))
        self.dec_w2 = nn.Parameter(torch.empty(decoder_hidden, action_dim))
        self.dec_b2 = nn.Parameter(torch.zeros(action_dim))
        nn.init.normal_(self.dec_w1,
                        std=math.sqrt(2.0 / (code_dim + decoder_hidden)))
        # Small-std init on final layer so initial actions have small
        # magnitude (memory updates near no-op at init) but not zero —
        # we need non-zero gradient flow for training.
        nn.init.normal_(self.dec_w2, std=1e-3)

        # EMA usage counts for dead-code detection during bootstrap.
        self.register_buffer(
            "usage_count", torch.zeros(num_codes))
        self.register_buffer("usage_total", torch.zeros(1))

    # ------------------------------------------------------------------
    # Forward components
    # ------------------------------------------------------------------

    def compute_logits(self, mod_input: Tensor) -> Tensor:
        """mod_input: [B, NC, mod_in] → logits: [B, NC, K]"""
        dt = mod_input.dtype
        w1 = self.logit_w1.to(dt)
        b1 = self.logit_b1.to(dt)
        w2 = self.logit_w2.to(dt)
        b2 = self.logit_b2.to(dt)
        h = torch.tanh(
            torch.einsum("bni,nih->bnh", mod_input, w1) + b1.unsqueeze(0))
        logits = torch.einsum("bnh,nhk->bnk", h, w2) + b2.unsqueeze(0)
        return logits

    def decode(self, codes: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        """codes: [B, NC] long → action: [B, NC, action_dim]

        Gradient does NOT flow through codes (int); use decode_soft for
        Gumbel-softmax differentiability.
        """
        dt = dtype if dtype is not None else self.codebook.dtype
        emb = self.codebook.to(dt)[codes]  # [B, NC, D]
        return self._decode_from_emb(emb)

    def decode_soft(self, soft_weights: Tensor) -> Tensor:
        """soft_weights: [B, NC, K] (probability-like or Gumbel-softmax output)
        → action: [B, NC, action_dim]

        Soft-mixture over codebook entries — gradient flows through the
        weights back to logits.
        """
        dt = soft_weights.dtype
        codebook = self.codebook.to(dt)
        emb = torch.einsum("bnk,kd->bnd", soft_weights, codebook)
        return self._decode_from_emb(emb)

    def _decode_from_emb(self, emb: Tensor) -> Tensor:
        """emb: [..., D] → action: [..., action_dim]"""
        dt = emb.dtype
        w1 = self.dec_w1.to(dt)
        b1 = self.dec_b1.to(dt)
        w2 = self.dec_w2.to(dt)
        b2 = self.dec_b2.to(dt)
        h = torch.tanh(emb @ w1 + b1)
        out = h @ w2 + b2
        return out

    # ------------------------------------------------------------------
    # Sampling modes
    # ------------------------------------------------------------------

    def sample_discrete(
        self, logits: Tensor, tau: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Hard Categorical sampling (for rollouts / phase 2 RL).

        Returns:
            codes: [B, NC] long
            log_pi: [B, NC] — log-prob of sampled codes under softmax(logits/tau)
        """
        # Use F.softmax directly for multinomial (avoids the log→exp
        # round-trip which loses precision in bf16 for low-prob codes).
        # Keep log_softmax separately for log_pi, which is what the GRPO
        # gradient pass consumes.
        scaled = logits / tau
        probs = F.softmax(scaled, dim=-1)
        log_probs = F.log_softmax(scaled, dim=-1)
        B, NC, K = probs.shape
        codes = torch.multinomial(probs.reshape(-1, K), 1).reshape(B, NC)
        log_pi = log_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        return codes, log_pi

    def sample_gumbel_soft(
        self, logits: Tensor, tau: float = 1.0, hard: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Gumbel-softmax sampling (for phase 1 backprop).

        Returns:
            soft_weights: [B, NC, K] (hard one-hot if hard=True but gradient
                uses the soft version via straight-through)
            codes: [B, NC] long — argmax of soft_weights (for logging /
                action-collection purposes)
        """
        # Use PyTorch's built-in — handles numerical edge cases (U=0 or U=1)
        # and gives correct straight-through gradient for hard=True.
        soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        codes = soft.argmax(dim=-1)
        return soft, codes

    def log_prob(self, logits: Tensor, codes: Tensor, tau: float = 1.0) -> Tensor:
        """Score the log-prob of given codes under the current logits.

        logits: [..., K], codes: [...] long  → log_pi: [...]
        """
        log_probs = F.log_softmax(logits / tau, dim=-1)
        return log_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)

    def entropy(self, logits: Tensor, tau: float = 1.0) -> Tensor:
        """Entropy of the categorical at given logits. logits: [..., K] → [...]"""
        log_probs = F.log_softmax(logits / tau, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    # ------------------------------------------------------------------
    # Convenience: single end-to-end call
    # ------------------------------------------------------------------

    def forward(
        self,
        mod_input: Tensor,
        phase: str = "phase1",
        tau: float = 1.0,
        hard_gumbel: bool = True,
    ) -> dict:
        """Convenience end-to-end forward. Returns dict with all intermediates.

        phase: "phase1" (Gumbel-softmax, differentiable) or
               "phase2" (hard Categorical sampling).

        Eval-mode bypass: when not self.training, both phases take the
        deterministic argmax path — no Gumbel noise, no Categorical sample.
        This keeps eval_ce and phase-2 warmup reproducible across runs.
        """
        logits = self.compute_logits(mod_input)

        if not self.training:
            codes = logits.argmax(dim=-1)
            action = self.decode(codes)
            log_pi = self.log_prob(logits, codes, tau=tau)
        elif phase == "phase1":
            soft, codes = self.sample_gumbel_soft(logits, tau=tau, hard=hard_gumbel)
            action = self.decode_soft(soft)
            # log_prob for logging (not gradient-carrying)
            log_pi = self.log_prob(logits.detach(), codes, tau=tau)
        elif phase == "phase2":
            codes, log_pi = self.sample_discrete(logits, tau=tau)
            action = self.decode(codes)
        else:
            raise ValueError(f"unknown phase: {phase}")

        return {
            "logits": logits,
            "codes": codes,
            "action": action,
            "log_pi": log_pi,
        }

    # ------------------------------------------------------------------
    # Codebook maintenance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_usage(self, codes: Tensor, decay: float = 0.9999):
        """EMA-track code usage counts. Call during bootstrap.

        Default decay=0.9999 is tuned for the per-modulation-event call
        frequency (T/mod_interval ≈ 32 per training step), giving a
        per-update half-life of ~6930 updates ≈ 217 training steps.
        That matches the reset cadence (every 500 steps) so "dead" is
        judged over roughly the last reset cycle, not the last few batches.

        Uses one_hot + sum instead of torch.bincount — the latter has a
        data-dependent output shape (minlength is only a floor, not a
        fixed shape), which breaks torch.compile's static-shape tracing.
        """
        flat = codes.reshape(-1).long()
        # one_hot has static output shape [N, K] when num_classes is given,
        # so the whole thing is safe to trace through torch.compile.
        counts = F.one_hot(flat, num_classes=self.K).sum(dim=0).float()
        self.usage_count.mul_(decay).add_(counts, alpha=1 - decay)
        # scalar .add_ keeps us off a per-call torch.tensor allocation.
        self.usage_total.mul_(decay).add_(float(flat.numel()) * (1 - decay))

    @torch.no_grad()
    def reset_dead_codes(
        self, threshold: float = 0.001, noise_std: float = 0.01,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Reinitialize codes whose usage fraction is below threshold.

        Dead codes are replaced with perturbed copies of randomly-chosen
        active codes. Returns count of codes reset. Call this periodically
        during BOOTSTRAP only (never in cycle phase 1 or 2, once frozen).

        If `optimizer` is provided, the AdamW first/second moments for the
        reset rows are zeroed. Without this, stale momentum would drag the
        freshly-seeded codes right back toward their old identities over
        the next few steps, defeating the reset.
        """
        total = self.usage_total.clamp(min=1.0).item()
        usage_frac = self.usage_count / total
        active = (usage_frac >= threshold).nonzero(as_tuple=True)[0]
        dead = (usage_frac < threshold).nonzero(as_tuple=True)[0]
        if dead.numel() == 0 or active.numel() == 0:
            return 0
        # Sample donor codes from active set
        donor_idx = active[torch.randint(0, active.numel(),
                                          (dead.numel(),), device=active.device)]
        noise = torch.randn_like(self.codebook[dead]) * noise_std
        self.codebook[dead] = self.codebook[donor_idx] + noise
        # Seed reset rows with a small runway above the re-detection
        # threshold: just above the dead cutoff so they don't pretend to be
        # fully-used (which would hide them from re-detection for many
        # steps), but above the threshold so they aren't flagged at the
        # very next reset check. If they accumulate real usage they rise
        # above this seed; if they don't, EMA decay pulls them back below
        # threshold within ~1 reset interval and they get caught again.
        self.usage_count[dead] = threshold * total * 3.0
        # Clear AdamW state for reset rows so momentum doesn't drag them
        # back toward their pre-reset identities.
        if optimizer is not None:
            state = optimizer.state.get(self.codebook)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    moment = state.get(key)
                    if moment is not None:
                        moment[dead] = 0.0
        return int(dead.numel())
