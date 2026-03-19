"""
Episodic Memory (v5) — trail-based primitive dictionary.

Single instance (batched across B banks via BS,B dims).
Read: trail-based composition (seed navigates primitive space).
Write: novelty-scored soft-routing decomposition with neuromodulated EMA.

All bank operations are vectorized — no per-bank Python loops.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import StateMixin, unit_normalize, budget_enforce


def _logit(p: float) -> float:
    """Inverse sigmoid: logit(p) = log(p / (1-p))."""
    return math.log(p / (1.0 - p))


class EpisodicMemory(nn.Module, StateMixin):
    """EM primitive dictionary with trail-based read and soft-routing write.

    State:
        em_K: [BS, B, M, D] — primitive keys (unit-normalized)
        em_V: [BS, B, M, D] — primitive values
        em_S: [BS, B, M] — strengths (0 = inactive)
    """

    _state_tensor_names = ["em_K", "em_V", "em_S"]

    def __init__(self, B: int, M: int, D: int, n_steps: int = 2,
                 D_mem: int = 0, S_max: float = 3.0, budget: float = 32.0,
                 decay: float = 0.999, topk: int = 0):
        super().__init__()
        self.B = B
        self.M = M
        self.D = D
        self.D_mem = D_mem if D_mem > 0 else D
        self.n_steps = n_steps
        self.S_max = S_max
        self.budget = budget
        self.topk = topk if topk > 0 else M  # 0 means all (no top-k)

        # Latent compression: D → D_mem for memory storage
        if self.D_mem != D:
            self.mem_proj_in = nn.Linear(D, self.D_mem)
            self.mem_proj_out = nn.Linear(self.D_mem, D)
            # Zero-init so EM trail starts silent
            nn.init.zeros_(self.mem_proj_out.weight)
            nn.init.zeros_(self.mem_proj_out.bias)
        else:
            self.mem_proj_in = None
            self.mem_proj_out = None

        # Trail parameters (per bank)
        self.gate_alpha = nn.Parameter(torch.randn(B) * 0.02)  # scalar gate scale
        self.gate_bias = nn.Parameter(torch.zeros(B))           # scalar gate bias
        self.raw_tau = nn.Parameter(torch.zeros(B))             # softplus -> temperature
        self.raw_sigma = nn.Parameter(torch.full([B], -2.0))  # softplus -> noise std
        self.raw_tau_w = nn.Parameter(torch.zeros(B))      # write temperature

        # Per-bank strength decay: sigmoid(raw) → in (0, 1)
        self.raw_decay = nn.Parameter(torch.full([B], _logit(decay)))

        # State (lazily allocated)
        self.em_K: Tensor | None = None
        self.em_V: Tensor | None = None
        self.em_S: Tensor | None = None

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        """Pre-allocate state tensors.

        em_K initialized with small random unit vectors so primitives have distinct
        directions from the start. em_S initialized to a small positive value so all
        primitives are active (avoiding the cold-start masking dead zone where
        em_S=0 means all attention scores are -inf).

        State uses D_mem dimensions (compressed latent space if D_mem < D).
        """
        D_mem = self.D_mem
        self.em_K = unit_normalize(torch.randn(BS, self.B, self.M, D_mem, device=device, dtype=dtype))
        self.em_V = torch.zeros(BS, self.B, self.M, D_mem, device=device, dtype=dtype)
        self.em_S = torch.full((BS, self.B, self.M), 0.01, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.em_K is not None

    def trail_read_all(self, seed: Tensor) -> Tensor:
        """Trail-based read for all banks simultaneously.

        Args:
            seed: [BS, N, D] — trail starting point (shared across banks)

        Returns:
            y_em: [BS, N, D] — net memory contribution (y - seed), summed over banks
        """
        BS, N, D = seed.shape
        B = self.B
        D_mem = self.D_mem

        # Project to memory latent space
        seed_mem = self.mem_proj_in(seed) if self.mem_proj_in is not None else seed

        # Expand seed to [BS, B, N, D_mem]
        s = seed_mem.unsqueeze(1).expand(BS, B, N, D_mem)
        if self.training:
            # sigma: [B] -> [1, B, 1, 1]
            sigma = F.softplus(self.raw_sigma)[None, :, None, None]
            s = s + sigma * torch.randn_like(s)

        y = s
        # tau: [B] -> [1, B, 1, 1]
        tau = (F.softplus(self.raw_tau) + 0.1)[None, :, None, None]

        # active mask: [BS, B, 1, M] — broadcasts over N
        active = (self.em_S > 0).unsqueeze(2)  # [BS, B, 1, M]

        # Pre-compute gate params: [B] -> [1, B, 1, 1]
        g_alpha = self.gate_alpha[None, :, None, None]
        g_bias = self.gate_bias[None, :, None, None]

        for step in range(self.n_steps):
            # scores: [BS, B, N, D_mem] @ [BS, B, D_mem, M] -> [BS, B, N, M]
            scores = torch.matmul(y, self.em_K.transpose(-2, -1)) / tau
            scores = scores.masked_fill(~active, float('-inf'))
            attn = F.softmax(scores, dim=-1)         # [BS, B, N, M]
            attn = attn.nan_to_num(0.0)
            # delta: [BS, B, N, M] @ [BS, B, M, D_mem] -> [BS, B, N, D_mem]
            delta = torch.matmul(attn, self.em_V)

            # Scalar gate: dot-product similarity -> sigmoid -> scale delta
            dot = (y * delta).sum(dim=-1, keepdim=True) / D_mem  # [BS, B, N, 1]
            gate = torch.sigmoid(g_alpha * dot + g_bias)          # [BS, B, N, 1]
            y = y + gate * delta

        # result: [BS, B, N, D_mem] -> sum over B -> [BS, N, D_mem]
        result = (y - s).sum(dim=1)

        # Project back to model dimension
        if self.mem_proj_out is not None:
            result = self.mem_proj_out(result)  # [BS, N, D]
        return result

    def compute_novelty_all(
        self, w_cand: Tensor, surprise: Tensor,
        w_nov: Tensor | None = None,
    ) -> Tensor:
        """Compute novelty score for all banks simultaneously.

        Novelty = w_nov * ||surprise|| + (1 - w_nov) * recon_error.
        Scores and reconstruction computed in D_mem latent space.

        Args:
            w_cand: [BS, N, D] — write candidates
            surprise: [BS, N, D] — vector surprise
            w_nov: [BS, N, B] — learned blend weight in [0,1] (default: 0.5)

        Returns:
            novelty: [BS, N, B] — scalar novelty per token per bank
        """
        BS, N, D = w_cand.shape
        B = self.B

        # Project to memory latent space
        w_cand_mem = self.mem_proj_in(w_cand) if self.mem_proj_in is not None else w_cand

        # tau: [B] -> [1, B, 1, 1]
        tau = (F.softplus(self.raw_tau) + 0.1)[None, :, None, None]

        # w_norm: [BS, N, D_mem] -> [BS, 1, N, D_mem]
        w_norm = unit_normalize(w_cand_mem).unsqueeze(1)
        # scores: [BS, 1, N, D_mem] @ [BS, B, D_mem, M] -> [BS, B, N, M]
        scores = torch.matmul(w_norm, self.em_K.transpose(-2, -1)) / tau
        active = (self.em_S > 0).unsqueeze(2)  # [BS, B, 1, M]
        scores = scores.masked_fill(~active, float('-inf'))
        attn = F.softmax(scores, dim=-1)       # [BS, B, N, M]
        attn = attn.nan_to_num(0.0)
        # reconstruction: [BS, B, N, M] @ [BS, B, M, D_mem] -> [BS, B, N, D_mem]
        reconstruction = torch.matmul(attn, self.em_V)
        # recon_error in D_mem space: [BS, B, N]
        recon_error = (w_cand_mem.unsqueeze(1) - reconstruction).norm(dim=-1)

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
        All operations in D_mem latent space.

        Args:
            w_cand: [BS, N, D] — write candidates (shared across banks)
            novelty: [BS, N, B] — novelty scores
            g_em: [BS, B] — neuromodulated write gate per bank
        """
        BS, N, D = w_cand.shape
        B = self.B
        M = self.M
        D_mem = self.D_mem

        # Project to memory latent space
        w_cand_mem = self.mem_proj_in(w_cand) if self.mem_proj_in is not None else w_cand

        # tau_w: [B] -> [1, B, 1, 1]
        tau_w = (F.softplus(self.raw_tau_w) + 0.1)[None, :, None, None]

        # Soft routing: which primitives absorb this signal?
        # w_norm: [BS, 1, N, D_mem]
        w_norm = unit_normalize(w_cand_mem).unsqueeze(1)
        # route_scores: [BS, 1, N, D_mem] @ [BS, B, D_mem, M] -> [BS, B, N, M]
        route_scores = torch.matmul(w_norm, self.em_K.transpose(-2, -1)) / tau_w
        active = (self.em_S > 0).unsqueeze(2)  # [BS, B, 1, M]
        route_scores = route_scores.masked_fill(~active, float('-inf'))

        # Top-k: concentrate writes on k most-similar primitives
        k = min(self.topk, M)
        if k < M:
            _, topk_idx = route_scores.topk(k, dim=-1)
            topk_mask = torch.zeros_like(route_scores, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            route_scores = route_scores.masked_fill(~topk_mask, float('-inf'))

        route = F.softmax(route_scores, dim=-1)  # [BS, B, N, M]
        route = route.nan_to_num(0.0)

        # Aggregate across N tokens, weighted by novelty
        # novelty: [BS, N, B] -> [BS, B, N, 1]
        nov = novelty.permute(0, 2, 1).unsqueeze(-1)
        weighted_route = nov * route               # [BS, B, N, M]
        wr_sum = weighted_route.sum(dim=2)         # [BS, B, M]
        route_agg = wr_sum * (1.0 / N)            # [BS, B, M] — mean without extra reduction

        # Aggregated updates in D_mem space: [BS, B, M, N] @ [BS, B, N, D_mem]
        w_norm_b = w_norm.expand(BS, B, N, D_mem)
        w_cand_b = w_cand_mem.unsqueeze(1).expand(BS, B, N, D_mem)
        update_K = torch.matmul(weighted_route.transpose(2, 3), w_norm_b)  # [BS, B, M, D_mem]
        update_V = torch.matmul(weighted_route.transpose(2, 3), w_cand_b)  # [BS, B, M, D_mem]

        # Normalize updates per primitive
        denom = wr_sum.unsqueeze(-1).clamp(min=1e-8)  # [BS, B, M, 1]
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

        # Budget enforcement
        self.em_S = budget_enforce(self.em_S.view(-1, M), self.budget).view_as(self.em_S)

    @property
    def decay(self) -> Tensor:
        """Per-bank decay rate [B], in (0, 1)."""
        return torch.sigmoid(self.raw_decay)

    def base_decay(self):
        """Apply per-bank strength decay once per segment."""
        if self.em_S is not None:
            # decay: [B] → [1, B, 1] to broadcast over [BS, B, M]
            self.em_S = self.em_S * self.decay[None, :, None]

    def usage_all(self) -> Tensor:
        """Usage fraction for all banks. Returns [BS, B]."""
        if self.em_S is None:
            return torch.tensor(0.0)
        return self.em_S.sum(dim=-1)  # [BS, B]

    def reset_states(self, mask: Tensor):
        """Re-initialize EM for masked streams (doc boundary).

        Must re-initialize to same state as initialize(), NOT zero —
        em_S=0 makes primitives permanently inactive (active mask = False,
        no routing, no writes, no recovery).

        mask: [BS] bool — True for streams to reset.
        """
        if self.em_S is None:
            return
        device = self.em_S.device
        dtype = self.em_S.dtype
        D_mem = self.D_mem

        # Fresh state matching initialize() — uses D_mem
        fresh_K = unit_normalize(torch.randn(
            1, self.B, self.M, D_mem, device=device, dtype=dtype))
        fresh_V = torch.zeros(
            1, self.B, self.M, D_mem, device=device, dtype=dtype)
        fresh_S = torch.full(
            (1, self.B, self.M), 0.01, device=device, dtype=dtype)

        expanded = mask[:, None, None]           # [BS, 1, 1]
        expanded_kv = mask[:, None, None, None]  # [BS, 1, 1, 1]
        self.em_S = torch.where(expanded, fresh_S, self.em_S)
        self.em_K = torch.where(expanded_kv, fresh_K, self.em_K)
        self.em_V = torch.where(expanded_kv, fresh_V, self.em_V)


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
        nn.init.zeros_(self.g_head.weight)
        nn.init.zeros_(self.g_head.bias)

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
