"""Discrete action policy: codebook + sampling primitives.

The encoder (logit production) lives in `attention_modulator.AttentionModulator`,
and the action generator (ΔW, Δdecay) lives in
`attention_modulator.DirectDecoder`. This module is the thin discrete-
bottleneck layer between them: given code logits, sample a code
(Gumbel-softmax or hard Categorical), then look it up in the codebook.

Codebook is SHARED across cells — cells agree on the plasticity vocabulary
but each has its own per-cell decoder to interpret codes into ΔW.

Trains unchanged under Phase 1 (Gumbel backprop) and Phase 2 (GRPO with
log_pi).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiscreteActionPolicy(nn.Module):
    """Codebook lookup + sampling primitives.

    Parameters:
        codebook: buffer [K, D_code] — learned lookup table; frozen in cycle
                  phase 1 and phase 2 GRPO alongside the action decoder.

    Shapes (B = batch):
        logits:     [B, K]
        codes:      [B]                   long, index in [0, K)
        embedding:  [B, D_code]
    """

    def __init__(self, num_codes: int, code_dim: int):
        super().__init__()
        self.K = num_codes
        self.D_code = code_dim

        # Codebook — small-scale init so initial soft-weighted lookup produces
        # unit-ish-scale embeddings.
        codebook = torch.randn(num_codes, code_dim) * (code_dim ** -0.5)
        self.codebook = nn.Parameter(codebook)

        # EMA usage counters for dead-code detection during bootstrap.
        self.register_buffer("usage_count", torch.zeros(num_codes))
        self.register_buffer("usage_total", torch.zeros(1))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_discrete(
        self, logits: Tensor, tau: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Hard Categorical sampling (phase 2 / rollouts).

        logits: [B, K]
        Returns (codes [B], log_pi [B]).
        """
        scaled = logits / tau
        probs = F.softmax(scaled, dim=-1)
        log_probs = F.log_softmax(scaled, dim=-1)
        codes = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_pi = log_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
        return codes, log_pi

    def sample_gumbel_soft(
        self, logits: Tensor, tau: float = 1.0, hard: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Gumbel-softmax sampling (phase 1 backprop).

        logits: [B, K]
        Returns (soft [B, K], codes [B]).

        With hard=True the forward uses a straight-through hard one-hot
        while the backward uses the soft Jacobian. Matches eval behavior
        (argmax) at forward time; gradients flow through the soft path.
        """
        soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        codes = soft.argmax(dim=-1)
        return soft, codes

    def log_prob(self, logits: Tensor, codes: Tensor, tau: float = 1.0) -> Tensor:
        """Score log π(code | logits). logits: [..., K], codes: [...] long → [...]"""
        log_probs = F.log_softmax(logits / tau, dim=-1)
        return log_probs.gather(-1, codes.unsqueeze(-1)).squeeze(-1)

    def entropy(self, logits: Tensor, tau: float = 1.0) -> Tensor:
        """Categorical entropy. logits: [..., K] → [...]"""
        log_probs = F.log_softmax(logits / tau, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    # ------------------------------------------------------------------
    # Codebook lookup
    # ------------------------------------------------------------------

    def lookup(self, codes: Tensor) -> Tensor:
        """Hard lookup: codes [B] long → emb [B, D_code]. No gradient through codes."""
        return self.codebook.to(codes.device)[codes]

    def lookup_soft(self, soft_weights: Tensor) -> Tensor:
        """Differentiable lookup via weighted average over codebook.

        soft_weights: [B, K] (probability-like, typically Gumbel-softmax output).
        Returns: [B, D_code].
        """
        dt = soft_weights.dtype
        codebook = self.codebook.to(dt)
        return soft_weights @ codebook

    # ------------------------------------------------------------------
    # Codebook maintenance (bootstrap only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_usage(self, codes: Tensor, decay: float = 0.9999) -> None:
        """EMA-track code usage counts during bootstrap.

        Uses one_hot + sum (not bincount — bincount has a data-dependent
        output shape that breaks torch.compile tracing).
        """
        flat = codes.reshape(-1).long()
        counts = F.one_hot(flat, num_classes=self.K).sum(dim=0).float()
        self.usage_count.mul_(decay).add_(counts, alpha=1 - decay)
        self.usage_total.mul_(decay).add_(float(flat.numel()) * (1 - decay))

    @torch.no_grad()
    def reset_dead_codes(
        self,
        threshold: float = 0.001,
        noise_std: float = 0.01,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Reinitialize codes whose usage fraction is below `threshold`.

        Dead codes get replaced with perturbed copies of randomly-chosen
        active codes. Clears AdamW momentum rows for reset codes so stale
        momentum doesn't drag them back. Returns count of codes reset.

        Only meaningful during bootstrap (codebook trainable). In cycle
        phase 1 and phase 2 the codebook is frozen; don't call this.
        """
        total = self.usage_total.clamp(min=1.0).item()
        usage_frac = self.usage_count / total
        active = (usage_frac >= threshold).nonzero(as_tuple=True)[0]
        dead = (usage_frac < threshold).nonzero(as_tuple=True)[0]
        if dead.numel() == 0 or active.numel() == 0:
            return 0

        donor_idx = active[torch.randint(
            0, active.numel(), (dead.numel(),), device=active.device)]
        noise = torch.randn_like(self.codebook[dead]) * noise_std
        self.codebook[dead] = self.codebook[donor_idx] + noise

        # Seed reset rows with a small runway above the re-detection threshold
        # so they don't pretend to be fully-used (hiding them for many steps)
        # but also aren't flagged at the next reset check.
        self.usage_count[dead] = threshold * total * 3.0

        # Zero AdamW state for reset rows so stale momentum doesn't drag them
        # back toward their pre-reset identities.
        if optimizer is not None:
            state = optimizer.state.get(self.codebook)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    moment = state.get(key)
                    if moment is not None:
                        moment[dead] = 0.0
        return int(dead.numel())
