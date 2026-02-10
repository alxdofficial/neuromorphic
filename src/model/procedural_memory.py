"""
Procedural Memory — fast low-rank weights with eligibility traces.

One PM instance per (block, layer). B*L total instances.

State per stream:
    pm_K: [BS, r, D_h]     — key bank (unit-normalized)
    pm_V: [BS, r, D_h]     — value bank (unit-normalized)
    pm_a: [BS, r]           — slot strengths (bounded)
    elig_K: [BS, r, D_h]   — eligibility trace for keys
    elig_V: [BS, r, D_h]   — eligibility trace for values

pm_K, pm_V, pm_a are plain tensors (NOT parameters). They are updated at
span boundaries and may carry computation graphs within a TBPTT chunk so
eligibility projections can receive LM gradients from downstream reads.
Eligibility projections (W_k_pre, W_v_post) are trained by backprop.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .scan import parallel_affine_scan
from .utils import StateMixin, unit_normalize, soft_topk, budget_enforce


class ProceduralMemory(nn.Module, StateMixin):
    _state_tensor_names = ["pm_K", "pm_V", "pm_a", "elig_K", "elig_V"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        D_h = config.D_h
        self.r = config.r
        self.D_h = D_h
        self.rho = config.rho
        self.a_max = config.a_max
        self.budget = config.budget_pm
        self.decay = config.decay_pm
        self.commit_top_k = config.commit_top_k
        self.tau = config.tau_pm
        self.weakness_weight = config.weakness_weight_pm

        # Eligibility projection layers (learned parameters)
        self.W_k_pre = nn.Linear(D_h, D_h)
        self.W_v_post = nn.Linear(D_h, D_h)

        # Post-readout MLP: processes the linear lookup result
        if config.pm_readout_ffn:
            self.readout_norm = nn.LayerNorm(D_h)
            self.readout_ffn = nn.Sequential(
                nn.Linear(D_h, D_h * 4),
                nn.GELU(),
                nn.Linear(D_h * 4, D_h),
            )
        else:
            self.readout_norm = None
            self.readout_ffn = None

        # State tensors (lazily initialized)
        self.pm_K: Tensor = None
        self.pm_V: Tensor = None
        self.pm_a: Tensor = None
        self.elig_K: Tensor = None
        self.elig_V: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        """Initialize state on first forward pass."""
        self.pm_K = unit_normalize(torch.randn(BS, self.r, self.D_h, device=device))
        self.pm_V = unit_normalize(torch.randn(BS, self.r, self.D_h, device=device))
        self.pm_a = torch.zeros(BS, self.r, device=device)
        self.elig_K = torch.zeros(BS, self.r, self.D_h, device=device)
        self.elig_V = torch.zeros(BS, self.r, self.D_h, device=device)

    def apply(self, x_block: Tensor) -> Tensor:
        """Read-only PM lookup.

        Args:
            x_block: [BS, D_h] — block input

        Returns:
            y_pm: [BS, D_h]
        """
        if self.pm_K is None:
            self._lazy_init(x_block.shape[0], x_block.device)

        x_q = unit_normalize(x_block)          # [BS, D_h]
        # scores = pm_K @ x_q -> [BS, r]
        scores = torch.einsum("brd, bd -> br", self.pm_K, x_q)
        # y_pm = (pm_a * scores) @ pm_V -> [BS, D_h]
        y_pm = torch.einsum("br, brd -> bd", self.pm_a * scores, self.pm_V)

        # Post-readout processing
        if self.readout_ffn is not None:
            y_pm = y_pm + self.readout_ffn(self.readout_norm(y_pm))

        return y_pm

    def apply_batch(self, x_block_all: Tensor) -> Tensor:
        """Read-only PM lookup for P tokens in parallel.

        Args:
            x_block_all: [BS, P, D_h] — block input for all tokens

        Returns:
            y_pm_all: [BS, P, D_h]
        """
        if self.pm_K is None:
            self._lazy_init(x_block_all.shape[0], x_block_all.device)

        x_q = unit_normalize(x_block_all)                             # [BS, P, D_h]
        scores = torch.einsum("brd, bpd -> bpr", self.pm_K, x_q)     # [BS, P, r]
        weighted = self.pm_a.unsqueeze(1) * scores                    # [BS, P, r]
        y_pm = torch.einsum("bpr, brd -> bpd", weighted, self.pm_V)  # [BS, P, D_h]

        if self.readout_ffn is not None:
            y_pm = y_pm + self.readout_ffn(self.readout_norm(y_pm))

        return y_pm

    def update_eligibility(self, x: Tensor, h: Tensor, surprise: Tensor):
        """Differentiable per-token eligibility accumulation.

        Eligibility is gated by surprise: only tokens the model doesn't
        predict well contribute to the trace. This prevents the trace norm
        from saturating (which would make the commit gate always fire).

        Args:
            x: [BS, D_h] — layer input (pre-synaptic)
            h: [BS, D_h] — layer output (post-synaptic)
            surprise: [BS, 1] — current surprise signal
        """
        if self.elig_K is None:
            self._lazy_init(x.shape[0], x.device)

        k_cand = unit_normalize(self.W_k_pre(x))   # [BS, D_h]
        v_cand = self.W_v_post(h)                   # [BS, D_h]

        # Gate by surprise: low surprise → near-zero accumulation
        # surprise is in ~[0, 5]; map to [0, 1] via clamp + normalize
        gate = (surprise.squeeze(-1) / 5.0).clamp(0.0, 1.0)  # [BS]
        gate = gate.unsqueeze(1).unsqueeze(2)                  # [BS, 1, 1]

        # Accumulate into all r slots (broadcast)
        # elig_K: [BS, r, D_h], k_cand: [BS, 1, D_h]
        self.elig_K = self.rho * self.elig_K + gate * k_cand.unsqueeze(1)
        self.elig_V = self.rho * self.elig_V + gate * v_cand.unsqueeze(1)

    def update_eligibility_batch(self, x_all: Tensor, h_all: Tensor,
                                 surprise_all: Tensor,
                                 reset_mask: Tensor):
        """Batched eligibility accumulation over P tokens using affine scan.

        Equivalent to calling update_eligibility() P times with
        reset_eligibility() at doc boundaries, but batches the projections
        and uses parallel_affine_scan for the recurrence.

        Args:
            x_all: [BS, P, D_h] — layer input (pre-synaptic) for all tokens
            h_all: [BS, P, D_h] — layer output (post-synaptic) for all tokens
            surprise_all: [BS, P, 1] — per-token surprise
            reset_mask: [BS, P] bool — True at doc boundaries
        """
        BS, P, D_h = x_all.shape
        r = self.r

        if self.elig_K is None:
            self._lazy_init(BS, x_all.device)

        # Batched projections (the expensive part — 2 matmuls instead of 2*P)
        k_cand_all = unit_normalize(self.W_k_pre(x_all))  # [BS, P, D_h]
        v_cand_all = self.W_v_post(h_all)                  # [BS, P, D_h]

        # Surprise gating: [BS, P] → [BS, P, 1, 1] for broadcast over [r, D_h]
        gate = (surprise_all.squeeze(-1) / 5.0).clamp(0.0, 1.0)  # [BS, P]
        gate = gate.unsqueeze(-1).unsqueeze(-1)                    # [BS, P, 1, 1]

        # b terms: gate * cand broadcast across r slots → [BS, P, r, D_h]
        b_K = (gate * k_cand_all.unsqueeze(2)).expand(BS, P, r, D_h)
        b_V = (gate * v_cand_all.unsqueeze(2)).expand(BS, P, r, D_h)

        # Carry mask: 0 at reset positions (zeros previous elig), 1 elsewhere
        carry = (~reset_mask).float()  # [BS, P]

        # a = rho * carry, expanded to [BS, P, r, D_h]
        a = (self.rho * carry).unsqueeze(-1).unsqueeze(-1).expand(
            BS, P, r, D_h
        )

        # Flatten [r, D_h] → [r*D_h] for scan (contiguous for reshape)
        a_flat = a.contiguous().reshape(BS, P, r * D_h)
        b_K_flat = b_K.contiguous().reshape(BS, P, r * D_h)
        b_V_flat = b_V.contiguous().reshape(BS, P, r * D_h)
        h_K_init = self.elig_K.reshape(BS, r * D_h)
        h_V_init = self.elig_V.reshape(BS, r * D_h)

        # Affine scan: elig_t = a_t * elig_{t-1} + b_t
        elig_K_all = parallel_affine_scan(a_flat, b_K_flat, h_K_init)
        elig_V_all = parallel_affine_scan(a_flat, b_V_flat, h_V_init)

        # Update state to last token
        self.elig_K = elig_K_all[:, -1].reshape(BS, r, D_h)
        self.elig_V = elig_V_all[:, -1].reshape(BS, r, D_h)

    def base_decay(self):
        """Per-span strength decay applied to ALL streams.

        Called at every span boundary before commit decisions. This ensures
        non-committing streams gradually lose pm_a over time.
        """
        if self.pm_a is not None:
            with torch.no_grad():
                self.pm_a = self.pm_a * self.decay

    def reset_eligibility(self, mask: Tensor):
        """Zero only eligibility traces for masked streams.

        Used in lifelong mode: committed PM weights persist,
        but in-progress eligibility clears at doc boundaries.
        """
        if self.elig_K is not None and mask.any():
            expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
            self.elig_K = self.elig_K * (~expanded).to(self.elig_K.dtype)
            self.elig_V = self.elig_V * (~expanded).to(self.elig_V.dtype)

    def commit(self, commit_mask: Tensor, lambda_vals: Tensor = None,
               g: Tensor = None, slot_logits: Tensor = None):
        """Span-boundary commit. Soft top-k slot selection + EMA update.

        Args:
            commit_mask: [BS] bool — which streams commit
            lambda_vals: [BS] — commit-time decay per stream (default: self.decay)
            g: [BS] — write strength per stream (default: 0.5)
            slot_logits: [BS, r] or None — slot selection logits (default: similarity-based)
        """
        if not commit_mask.any() or self.pm_K is None:
            return

        BS = commit_mask.shape[0]
        device = commit_mask.device

        # Apply defaults for any None args
        if lambda_vals is None:
            lambda_vals = torch.full((BS,), self.decay, device=device)
        if g is None:
            g = torch.full((BS,), 0.5, device=device)

        # Normalized eligibility as commit candidates
        elig_K_norm = unit_normalize(self.elig_K)  # [BS, r, D_h]
        elig_V_norm = self.elig_V                   # [BS, r, D_h]

        # Slot selection: similarity between current keys and eligibility
        # scores: [BS, r] — how well each slot matches the eligibility
        scores = torch.einsum("brd, brd -> br", self.pm_K, elig_K_norm)

        # Weakness bias: prefer overwriting weak slots
        scores = scores - self.weakness_weight * self.pm_a

        # Use controller-provided slot_logits if given, else use similarity
        if slot_logits is not None:
            scores = scores + slot_logits

        # Soft top-k selection
        weights = soft_topk(scores, self.commit_top_k, self.tau)  # [BS, r]

        # Apply commit mask and per-stream write strength
        mask_expanded = commit_mask.float().unsqueeze(-1)  # [BS, 1]
        alpha = weights * g.unsqueeze(-1) * mask_expanded  # [BS, r]

        # Apply commit-time decay on committing streams using controller lambda
        lambda_expanded = lambda_vals.unsqueeze(-1)  # [BS, 1]
        self.pm_a = self.pm_a * (1.0 - mask_expanded * (1.0 - lambda_expanded))

        # EMA update of keys and values
        alpha_3d = alpha.unsqueeze(-1)  # [BS, r, 1]
        self.pm_K = unit_normalize((1 - alpha_3d) * self.pm_K + alpha_3d * elig_K_norm)
        self.pm_V = unit_normalize((1 - alpha_3d) * self.pm_V + alpha_3d * elig_V_norm)

        # Strength update
        self.pm_a = (self.pm_a + alpha).clamp(0.0, self.a_max)

        # Budget enforcement
        self.pm_a = budget_enforce(self.pm_a, self.budget)

        # Reset eligibility for committing streams (keep differentiable path alive)
        reset_3d = commit_mask.float().unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
        self.elig_K = self.elig_K * (1.0 - reset_3d)
        self.elig_V = self.elig_V * (1.0 - reset_3d)


class PMNeuromodulator(nn.Module):
    """Neuromodulator for PM commit decisions.

    Phase A (pm_enabled=False): empty shell, no parameters.
    Phases B–C (pm_enabled=True, rl_enabled=False): learned continuous heads
        (lambda, g, slot_logits) trained via main optimizer backprop,
        with heuristic commit gate (elig_norm > threshold).
    Phase D+ (pm_enabled=True, rl_enabled=True): full learned MLP —
        adds gate_head trained via RL counterfactual.

    Returns:
        commit_mask: [BS] bool
        lambda_vals: [BS]
        g: [BS]
        slot_logits: [BS, r] or None
        p_commit: [BS] or None (only when rl_enabled)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rl_enabled = config.rl_enabled
        self.pm_enabled = config.pm_enabled
        self.threshold = 1.0
        self.default_g = 0.5
        self.default_decay = config.decay_pm

        # Backbone + continuous heads: created when PM is enabled (Phase B+)
        if self.pm_enabled:
            H = config.rl_controller_hidden
            self.backbone = nn.Sequential(
                nn.Linear(3, H),
                nn.ReLU(),
            )
            self.lambda_head = nn.Linear(H, 1)
            self.g_head = nn.Linear(H, 1)
            self.slot_head = nn.Linear(H, config.r)

            # Gate head: only created for RL (Phase D+)
            if self.rl_enabled:
                self.gate_head = nn.Linear(H, 1)

    def forward(self, elig_norm: Tensor, pm_usage: Tensor,
                span_surprise: Tensor) -> tuple:
        if self.pm_enabled:
            if self.rl_enabled:
                return self._forward_learned(elig_norm, pm_usage, span_surprise)
            return self._forward_continuous(elig_norm, pm_usage, span_surprise)
        return self._forward_heuristic(elig_norm, pm_usage, span_surprise)

    def _forward_heuristic(self, elig_norm, pm_usage, span_surprise):
        """Heuristic mode — no learnable params (Phase A fallback)."""
        BS = elig_norm.shape[0]
        commit_mask = elig_norm > self.threshold
        lambda_vals = torch.full((BS,), self.default_decay,
                                 device=elig_norm.device)
        g = torch.full((BS,), self.default_g, device=elig_norm.device)
        return commit_mask, lambda_vals, g, None, None

    def _forward_continuous(self, elig_norm, pm_usage, span_surprise):
        """Heuristic gate + learned continuous heads (Phases B–C)."""
        commit_mask = elig_norm > self.threshold

        features = torch.stack([elig_norm, pm_usage, span_surprise], dim=-1)
        h = self.backbone(features)

        raw_lambda = torch.sigmoid(self.lambda_head(h)).squeeze(-1)
        lambda_vals = self.default_decay + (1.0 - self.default_decay) * raw_lambda

        g = torch.sigmoid(self.g_head(h)).squeeze(-1)
        slot_logits = self.slot_head(h)

        return commit_mask, lambda_vals, g, slot_logits, None

    def _forward_learned(self, elig_norm, pm_usage, span_surprise):
        """Full learned mode — gate + continuous heads (Phase D+)."""
        features = torch.stack([elig_norm, pm_usage, span_surprise], dim=-1)
        h = self.backbone(features)

        p_commit = torch.sigmoid(self.gate_head(h)).squeeze(-1)
        commit_mask = (p_commit > 0.5).detach()

        raw_lambda = torch.sigmoid(self.lambda_head(h)).squeeze(-1)
        lambda_vals = self.default_decay + (1.0 - self.default_decay) * raw_lambda

        g = torch.sigmoid(self.g_head(h)).squeeze(-1)
        slot_logits = self.slot_head(h)

        return commit_mask, lambda_vals, g, slot_logits, p_commit
