"""
Procedural Memory — fast low-rank weights with eligibility traces.

One PM instance per (block, layer). B*L total instances.

State per stream:
    pm_K: [BS, r, D_h]     — key bank (unit-normalized)
    pm_V: [BS, r, D_h]     — value bank (unit-normalized)
    pm_a: [BS, r]           — slot strengths (bounded)
    elig_K: [BS, r, D_h]   — eligibility trace for keys
    elig_V: [BS, r, D_h]   — eligibility trace for values

pm_K, pm_V, pm_a are plain tensors (NOT parameters). Updated under
torch.no_grad() at span boundaries. Eligibility projections (W_k_pre,
W_v_post) ARE parameters trained by backprop through eligibility.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
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
        return y_pm

    def update_eligibility(self, x: Tensor, h: Tensor):
        """Differentiable per-token eligibility accumulation.

        Args:
            x: [BS, D_h] — layer input (pre-synaptic)
            h: [BS, D_h] — layer output (post-synaptic)
        """
        if self.elig_K is None:
            self._lazy_init(x.shape[0], x.device)

        k_cand = unit_normalize(self.W_k_pre(x))   # [BS, D_h]
        v_cand = self.W_v_post(h)                   # [BS, D_h]

        # Accumulate into all r slots (broadcast)
        # elig_K: [BS, r, D_h], k_cand: [BS, 1, D_h]
        self.elig_K = self.rho * self.elig_K + k_cand.unsqueeze(1)
        self.elig_V = self.rho * self.elig_V + v_cand.unsqueeze(1)

    def base_decay(self):
        """Per-span strength decay applied to ALL streams.

        Called at every span boundary before commit decisions. This ensures
        non-committing streams gradually lose pm_a over time.
        """
        if self.pm_a is not None:
            with torch.no_grad():
                self.pm_a = self.pm_a * self.decay

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

        with torch.no_grad():
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


class PMController(nn.Module):
    """Heuristic commit decisions based on eligibility norm + surprise.

    MVP: commits if eligibility norm exceeds a threshold. Returns full
    (commit_mask, lambda, g, slot_logits) tuple with heuristic defaults.
    Future Phase D will use RL-based controllers.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.threshold = 1.0  # heuristic threshold on eligibility norm
        self.default_g = 0.5
        self.default_decay = config.decay_pm

    def forward(self, elig_norm: Tensor, pm_usage: Tensor,
                span_surprise: Tensor) -> tuple:
        """Decide which streams should commit and with what parameters.

        Args:
            elig_norm: [BS] — L2 norm of eligibility traces
            pm_usage: [BS] — sum of pm_a (current usage)
            span_surprise: [BS] — mean surprise over span

        Returns:
            commit_mask: [BS] bool — which streams commit
            lambda_vals: [BS] — per-stream commit decay (heuristic: config.decay_pm)
            g: [BS] — per-stream write strength (heuristic: 0.5)
            slot_logits: None — slot selection deferred to commit() similarity
        """
        BS = elig_norm.shape[0]
        commit_mask = elig_norm > self.threshold
        lambda_vals = torch.full((BS,), self.default_decay,
                                 device=elig_norm.device)
        g = torch.full((BS,), self.default_g, device=elig_norm.device)
        return commit_mask, lambda_vals, g, None
