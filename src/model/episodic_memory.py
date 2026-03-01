"""
Episodic Memory (v5) — trail-based primitive dictionary.

Single instance (batched across B banks via BS,B dims).
Read: trail-based composition (seed navigates primitive space).
Write: novelty-scored soft-routing decomposition with neuromodulated EMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import StateMixin, unit_normalize, budget_enforce


class EpisodicMemory(nn.Module, StateMixin):
    """EM primitive dictionary with trail-based read and soft-routing write.

    State:
        em_K: [BS, B, M, D] — primitive keys (unit-normalized)
        em_V: [BS, B, M, D] — primitive values
        em_S: [BS, B, M] — strengths (0 = inactive)
        em_age: [BS, B, M] — segments since last write
    """

    _state_tensor_names = ["em_K", "em_V", "em_S", "em_age"]

    def __init__(self, B: int, M: int, D: int, n_steps: int = 2,
                 S_max: float = 3.0, budget: float = 32.0,
                 decay: float = 0.999):
        super().__init__()
        self.B = B
        self.M = M
        self.D = D
        self.n_steps = n_steps
        self.S_max = S_max
        self.budget = budget
        self.decay = decay

        # Trail parameters (per bank)
        self.w1 = nn.Parameter(torch.randn(B, D) * 0.02)
        self.w2 = nn.Parameter(torch.randn(B, D) * 0.02)
        self.gate_bias = nn.Parameter(torch.zeros(B, D))
        self.raw_tau = nn.Parameter(torch.zeros(B))        # softplus -> temperature
        self.raw_sigma = nn.Parameter(torch.full([B], -2.0))  # softplus -> noise std
        self.raw_tau_w = nn.Parameter(torch.zeros(B))      # write temperature

        # State (lazily allocated)
        self.em_K: Tensor | None = None
        self.em_V: Tensor | None = None
        self.em_S: Tensor | None = None
        self.em_age: Tensor | None = None

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        """Pre-allocate state tensors."""
        self.em_K = torch.zeros(BS, self.B, self.M, self.D, device=device, dtype=dtype)
        self.em_V = torch.zeros(BS, self.B, self.M, self.D, device=device, dtype=dtype)
        self.em_S = torch.zeros(BS, self.B, self.M, device=device, dtype=dtype)
        self.em_age = torch.zeros(BS, self.B, self.M, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.em_K is not None

    def trail_read(self, seed: Tensor, b: int) -> Tensor:
        """Trail-based read: seed navigates primitive space.

        Args:
            seed: [BS, N, D] — trail starting point
            b: bank index

        Returns:
            y_em: [BS, N, D] — net memory contribution (y - seed)
        """
        s = seed  # [BS, N, D]
        if self.training:
            sigma = F.softplus(self.raw_sigma[b])
            s = s + sigma * torch.randn_like(s)

        y = s
        tau = F.softplus(self.raw_tau[b]) + 0.1

        # Check if any primitives are active (avoid NaN from softmax over all -inf)
        any_active = (self.em_S[:, b] > 0).any(dim=-1)  # [BS]
        if not any_active.any():
            return torch.zeros_like(seed)

        for step in range(self.n_steps):
            # Score all primitives: [BS, N, D] @ [BS, M, D].T -> [BS, N, M]
            scores = torch.matmul(y, self.em_K[:, b].transpose(-2, -1)) / tau
            # Mask inactive primitives
            active = (self.em_S[:, b] > 0).unsqueeze(1)  # [BS, 1, M]
            scores = scores.masked_fill(~active, float('-inf'))
            attn = F.softmax(scores, dim=-1)               # [BS, N, M]
            # Replace NaN from all-inactive rows with zeros
            attn = attn.nan_to_num(0.0)
            delta = torch.matmul(attn, self.em_V[:, b])    # [BS, N, D]

            # Element-wise gate (no matmul — O(D) not O(D^2))
            gate = torch.sigmoid(self.w1[b] * y + self.w2[b] * delta + self.gate_bias[b])
            y = y + gate * delta

        return y - s  # [BS, N, D]

    def compute_novelty(
        self, w_cand: Tensor, surprise: Tensor, b: int,
        w_nov: Tensor | None = None,
    ) -> Tensor:
        """Compute novelty score for bank b.

        Novelty = w_nov * ||surprise|| + (1 - w_nov) * recon_error.

        Args:
            w_cand: [BS, N, D] — write candidates
            surprise: [BS, N, D] — vector surprise
            b: bank index
            w_nov: [BS, N] — learned blend weight ∈ [0,1] (default: 0.5)

        Returns:
            novelty: [BS, N] — scalar novelty per token
        """
        tau = F.softplus(self.raw_tau[b]) + 0.1

        # Reconstruction error: how well can existing primitives explain w_cand?
        w_norm = unit_normalize(w_cand)
        scores = torch.matmul(w_norm, self.em_K[:, b].transpose(-2, -1)) / tau
        active = (self.em_S[:, b] > 0).unsqueeze(1)
        scores = scores.masked_fill(~active, float('-inf'))
        attn = F.softmax(scores, dim=-1)                   # [BS, N, M]
        attn = attn.nan_to_num(0.0)  # handle empty memory
        reconstruction = torch.matmul(attn, self.em_V[:, b])  # [BS, N, D]
        recon_error = (w_cand - reconstruction).norm(dim=-1)   # [BS, N]

        # Surprise magnitude
        surp_mag = surprise.norm(dim=-1)  # [BS, N]

        # Learned blend (defaults to 0.5 if w_nov not provided)
        if w_nov is None:
            w_nov = 0.5
        novelty = w_nov * surp_mag + (1 - w_nov) * recon_error
        return novelty

    def compute_write_deltas(self, novelty: Tensor, w_cand: Tensor) -> Tensor:
        """Per-token EM write delta: novelty * w_cand.

        Args:
            novelty: [BS, N] — scalar novelty per token
            w_cand: [BS, N, D] — write candidates

        Returns:
            delta_em: [BS, N, D] — per-token write delta
        """
        return novelty.unsqueeze(-1) * w_cand  # [BS, N, D]

    def commit(self, w_cand: Tensor, novelty: Tensor, g_em: Tensor, b: int):
        """Segment-end structured write for bank b.

        Soft routing -> aggregate -> neuromodulated EMA.

        Args:
            w_cand: [BS, N, D] — write candidates
            novelty: [BS, N] — novelty scores
            g_em: [BS] — neuromodulated write gate
            b: bank index
        """
        tau_w = F.softplus(self.raw_tau_w[b]) + 0.1

        # Soft routing: which primitives absorb this signal?
        w_norm = unit_normalize(w_cand)
        any_active = (self.em_S[:, b] > 0).any(dim=-1)  # [BS]

        if any_active.any():
            route_scores = torch.matmul(w_norm, self.em_K[:, b].transpose(-2, -1)) / tau_w
            active = (self.em_S[:, b] > 0).unsqueeze(1)
            route_scores = route_scores.masked_fill(~active, float('-inf'))
            route = F.softmax(route_scores, dim=-1)  # [BS, N, M]
            route = route.nan_to_num(1.0 / self.M)  # uniform for streams with no active
        else:
            # No active primitives anywhere — use uniform routing
            route = torch.full(
                (w_cand.shape[0], w_cand.shape[1], self.M),
                1.0 / self.M, device=w_cand.device, dtype=w_cand.dtype
            )

        # Aggregate across N tokens, weighted by novelty
        # novelty: [BS, N] -> [BS, N, 1], route: [BS, N, M]
        weighted_route = novelty.unsqueeze(-1) * route  # [BS, N, M]
        route_agg = weighted_route.mean(dim=1)           # [BS, M]

        # Aggregated updates: [BS, M, D]
        # [BS, M, N] @ [BS, N, D] -> [BS, M, D]
        update_K = torch.bmm(weighted_route.transpose(1, 2), w_norm)
        update_V = torch.bmm(weighted_route.transpose(1, 2), w_cand)

        # Normalize updates per primitive
        denom = weighted_route.sum(dim=1).unsqueeze(-1).clamp(min=1e-8)  # [BS, M, 1]
        update_K = update_K / denom
        update_V = update_V / denom

        # Neuromodulated EMA: alpha = g_em * route_agg (per-primitive)
        alpha = (g_em.unsqueeze(-1) * route_agg).clamp(max=1.0)  # [BS, M]
        alpha_exp = alpha.unsqueeze(-1)  # [BS, M, 1]

        # EMA update (reassignment for autograd)
        new_K = self.em_K.clone()
        new_V = self.em_V.clone()
        new_S = self.em_S.clone()
        new_age = self.em_age.clone()

        new_K[:, b] = (1 - alpha_exp) * self.em_K[:, b] + alpha_exp * unit_normalize(update_K)
        new_V[:, b] = (1 - alpha_exp) * self.em_V[:, b] + alpha_exp * update_V
        new_S[:, b] = (self.em_S[:, b] + alpha).clamp(0, self.S_max)
        new_age[:, b] = self.em_age[:, b] * (1 - alpha)

        self.em_K = new_K
        self.em_V = new_V
        self.em_S = new_S
        self.em_age = new_age

        # Budget enforcement
        self.em_S = budget_enforce(self.em_S.view(-1, self.M), self.budget).view_as(self.em_S)

    def base_decay(self):
        """Apply strength decay and age tick once per segment."""
        if self.em_S is not None:
            self.em_S = self.em_S * self.decay
        if self.em_age is not None:
            active = (self.em_S > 0).to(self.em_age.dtype)
            self.em_age = self.em_age + active

    def usage(self, b: int) -> Tensor:
        """Usage fraction for bank b. Returns [BS]."""
        if self.em_S is None:
            return torch.tensor(0.0)
        return self.em_S[:, b].sum(dim=-1)  # [BS]

    def reset_states(self, mask: Tensor):
        """Full reset for masked streams. mask: [BS] bool."""
        if self.em_S is None:
            return
        expanded = mask[:, None, None]           # [BS, 1, 1]
        expanded_kv = mask[:, None, None, None]  # [BS, 1, 1, 1]
        self.em_S = self.em_S * ~expanded
        self.em_age = self.em_age * ~expanded
        self.em_K = self.em_K * ~expanded_kv
        self.em_V = self.em_V * ~expanded_kv


class EMNeuromodulator(nn.Module):
    """Simplified neuromodulator for EM write decisions.

    Takes novelty_mean and usage, produces g_em scalar per bank.
    """

    def __init__(self, hidden: int = 32, g_floor: float = 0.001, g_ceil: float = 0.95):
        super().__init__()
        self.g_floor = g_floor
        self.g_ceil = g_ceil

        # Input: novelty_mean + usage = 2 features
        self.backbone = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
        )
        self.g_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.backbone[0].bias)

    def forward(self, novelty_mean: Tensor, usage: Tensor) -> Tensor:
        """Produce write gate g_em.

        Args:
            novelty_mean: [BS] — mean novelty across tokens
            usage: [BS] — EM usage (sum of strengths)

        Returns:
            g_em: [BS] — write gate ∈ [g_floor, g_ceil]
        """
        features = torch.stack([novelty_mean, usage], dim=-1)  # [BS, 2]
        h = self.backbone(features)
        g_raw = self.g_head(h).squeeze(-1)
        return self.g_floor + (self.g_ceil - self.g_floor) * torch.sigmoid(g_raw)
