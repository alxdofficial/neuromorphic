"""
Episodic Memory (v5) — trail-based primitive dictionary.

Single instance (batched across B banks via BS,B dims).
Read: trail-based composition (seed navigates primitive space).
Write: novelty-scored soft-routing decomposition with neuromodulated EMA.

All bank operations are vectorized — no per-bank Python loops.
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

    def trail_read_all(self, seed: Tensor) -> Tensor:
        """Trail-based read for all banks simultaneously.

        Args:
            seed: [BS, N, D] — trail starting point (shared across banks)

        Returns:
            y_em: [BS, N, B, D] — net memory contribution (y - seed) per bank
        """
        BS, N, D = seed.shape
        B = self.B

        # Expand seed to [BS, B, N, D]
        s = seed.unsqueeze(1).expand(BS, B, N, D)
        if self.training:
            # sigma: [B] -> [1, B, 1, 1]
            sigma = F.softplus(self.raw_sigma)[None, :, None, None]
            s = s + sigma * torch.randn_like(s)

        y = s
        # tau: [B] -> [1, B, 1, 1]
        tau = (F.softplus(self.raw_tau) + 0.1)[None, :, None, None]

        # active mask: [BS, B, 1, M] — broadcasts over N
        active = (self.em_S > 0).unsqueeze(2)  # [BS, B, 1, M]

        for step in range(self.n_steps):
            # scores: [BS, B, N, D] @ [BS, B, D, M] -> [BS, B, N, M]
            scores = torch.matmul(y, self.em_K.transpose(-2, -1)) / tau
            scores = scores.masked_fill(~active, float('-inf'))
            attn = F.softmax(scores, dim=-1)         # [BS, B, N, M]
            attn = attn.nan_to_num(0.0)
            # delta: [BS, B, N, M] @ [BS, B, M, D] -> [BS, B, N, D]
            delta = torch.matmul(attn, self.em_V)

            # Element-wise gate: w1, w2, gate_bias are [B, D] -> [1, B, 1, D]
            w1 = self.w1[None, :, None, :]
            w2 = self.w2[None, :, None, :]
            gb = self.gate_bias[None, :, None, :]
            gate = torch.sigmoid(w1 * y + w2 * delta + gb)
            y = y + gate * delta

        # result: [BS, B, N, D] -> [BS, N, B, D]
        return (y - s).permute(0, 2, 1, 3)

    def compute_novelty_all(
        self, w_cand: Tensor, surprise: Tensor,
        w_nov: Tensor | None = None,
    ) -> Tensor:
        """Compute novelty score for all banks simultaneously.

        Novelty = w_nov * ||surprise|| + (1 - w_nov) * recon_error.

        Args:
            w_cand: [BS, N, D] — write candidates
            surprise: [BS, N, D] — vector surprise
            w_nov: [BS, N, B] — learned blend weight in [0,1] (default: 0.5)

        Returns:
            novelty: [BS, N, B] — scalar novelty per token per bank
        """
        BS, N, D = w_cand.shape
        B = self.B

        # tau: [B] -> [1, B, 1, 1]
        tau = (F.softplus(self.raw_tau) + 0.1)[None, :, None, None]

        # w_norm: [BS, N, D] -> [BS, 1, N, D] for broadcasting against [BS, B, M, D]
        w_norm = unit_normalize(w_cand).unsqueeze(1)
        # scores: [BS, 1, N, D] @ [BS, B, D, M] -> [BS, B, N, M]
        scores = torch.matmul(w_norm, self.em_K.transpose(-2, -1)) / tau
        active = (self.em_S > 0).unsqueeze(2)  # [BS, B, 1, M]
        scores = scores.masked_fill(~active, float('-inf'))
        attn = F.softmax(scores, dim=-1)       # [BS, B, N, M]
        attn = attn.nan_to_num(0.0)
        # reconstruction: [BS, B, N, M] @ [BS, B, M, D] -> [BS, B, N, D]
        reconstruction = torch.matmul(attn, self.em_V)
        # recon_error: [BS, B, N]
        recon_error = (w_cand.unsqueeze(1) - reconstruction).norm(dim=-1)

        # Surprise magnitude: [BS, N] -> [BS, 1, N]
        surp_mag = surprise.norm(dim=-1).unsqueeze(1)  # [BS, 1, N]

        # Blend: w_nov [BS, N, B] -> [BS, B, N]
        if w_nov is None:
            novelty = 0.5 * surp_mag + 0.5 * recon_error  # [BS, B, N]
        else:
            w = w_nov.permute(0, 2, 1)  # [BS, B, N]
            novelty = w * surp_mag + (1 - w) * recon_error

        # Return as [BS, N, B]
        return novelty.permute(0, 2, 1)

    def compute_write_deltas(self, novelty: Tensor, w_cand: Tensor) -> Tensor:
        """Per-token EM write delta: novelty * w_cand, for all banks.

        Args:
            novelty: [BS, N, B] — scalar novelty per token per bank
            w_cand: [BS, N, D] — write candidates

        Returns:
            delta_em: [BS, N, B, D] — per-token write delta
        """
        return novelty.unsqueeze(-1) * w_cand.unsqueeze(2)  # [BS, N, B, D]

    def commit_all(self, w_cand: Tensor, novelty: Tensor, g_em: Tensor):
        """Segment-end structured write for all banks simultaneously.

        Soft routing -> aggregate -> neuromodulated EMA.

        Args:
            w_cand: [BS, N, D] — write candidates (shared across banks)
            novelty: [BS, N, B] — novelty scores
            g_em: [BS, B] — neuromodulated write gate per bank
        """
        BS, N, D = w_cand.shape
        B = self.B
        M = self.M

        # tau_w: [B] -> [1, B, 1, 1]
        tau_w = (F.softplus(self.raw_tau_w) + 0.1)[None, :, None, None]

        # Soft routing: which primitives absorb this signal?
        # w_norm: [BS, 1, N, D]
        w_norm = unit_normalize(w_cand).unsqueeze(1)
        # route_scores: [BS, 1, N, D] @ [BS, B, D, M] -> [BS, B, N, M]
        route_scores = torch.matmul(w_norm, self.em_K.transpose(-2, -1)) / tau_w
        active = (self.em_S > 0).unsqueeze(2)  # [BS, B, 1, M]
        route_scores = route_scores.masked_fill(~active, float('-inf'))
        route = F.softmax(route_scores, dim=-1)  # [BS, B, N, M]
        route = route.nan_to_num(1.0 / M)

        # Aggregate across N tokens, weighted by novelty
        # novelty: [BS, N, B] -> [BS, B, N, 1]
        nov = novelty.permute(0, 2, 1).unsqueeze(-1)
        weighted_route = nov * route               # [BS, B, N, M]
        route_agg = weighted_route.mean(dim=2)     # [BS, B, M]

        # Aggregated updates: [BS, B, M, N] @ [BS, B, N, D] -> [BS, B, M, D]
        # w_norm: [BS, 1, N, D], w_cand: [BS, 1, N, D] — broadcast across B
        w_norm_b = w_norm.expand(BS, B, N, D)
        w_cand_b = w_cand.unsqueeze(1).expand(BS, B, N, D)
        update_K = torch.matmul(weighted_route.transpose(2, 3), w_norm_b)  # [BS, B, M, D]
        update_V = torch.matmul(weighted_route.transpose(2, 3), w_cand_b)  # [BS, B, M, D]

        # Normalize updates per primitive
        denom = weighted_route.sum(dim=2).unsqueeze(-1).clamp(min=1e-8)  # [BS, B, M, 1]
        update_K = update_K / denom
        update_V = update_V / denom

        # Neuromodulated EMA: alpha = g_em * route_agg (per-primitive)
        # g_em: [BS, B] -> [BS, B, 1], route_agg: [BS, B, M]
        alpha = (g_em.unsqueeze(-1) * route_agg).clamp(max=1.0)  # [BS, B, M]
        alpha_exp = alpha.unsqueeze(-1)  # [BS, B, M, 1]

        # EMA update — arithmetic ops create new tensors for autograd, no clone needed
        self.em_K = (1 - alpha_exp) * self.em_K + alpha_exp * unit_normalize(update_K)
        self.em_V = (1 - alpha_exp) * self.em_V + alpha_exp * update_V
        self.em_S = (self.em_S + alpha).clamp(0, self.S_max)
        self.em_age = self.em_age * (1 - alpha)

        # Budget enforcement
        self.em_S = budget_enforce(self.em_S.view(-1, M), self.budget).view_as(self.em_S)

    def base_decay(self):
        """Apply strength decay and age tick once per segment."""
        if self.em_S is not None:
            self.em_S = self.em_S * self.decay
        if self.em_age is not None:
            active = (self.em_S > 0).to(self.em_age.dtype)
            self.em_age = self.em_age + active

    def usage_all(self) -> Tensor:
        """Usage fraction for all banks. Returns [BS, B]."""
        if self.em_S is None:
            return torch.tensor(0.0)
        return self.em_S.sum(dim=-1)  # [BS, B]

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
    Batched across all banks simultaneously.
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
        """Produce write gate g_em for all banks at once.

        Args:
            novelty_mean: [BS, B] — mean novelty across tokens, per bank
            usage: [BS, B] — EM usage (sum of strengths) per bank

        Returns:
            g_em: [BS, B] — write gate in [g_floor, g_ceil]
        """
        # features: [BS, B, 2]
        features = torch.stack([novelty_mean, usage], dim=-1)
        h = self.backbone(features)       # [BS, B, hidden]
        g_raw = self.g_head(h).squeeze(-1)  # [BS, B]
        return self.g_floor + (self.g_ceil - self.g_floor) * torch.sigmoid(g_raw)
