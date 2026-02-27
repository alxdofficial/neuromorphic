"""
Procedural Memory (v4) — holographic modulation slots.

Per-block (B_blocks instances). Operates in D_mem space.
Read: holographic modulation. Write: neuromodulated EMA commit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .utils import StateMixin, unit_normalize, budget_enforce


class ProceduralMemory(nn.Module, StateMixin):
    """Holographic procedural memory with eligibility-based commit.

    State:
        pm_K: [BS, r, D_mem] — key bank (unit-normalized)
        pm_V: [BS, r, D_mem] — value bank (modulation patterns)
        pm_a: [BS, r] — slot strengths (bounded)
    """

    _state_tensor_names = ["pm_K", "pm_V", "pm_a"]

    def __init__(self, D_mem: int, r_slots: int, config: ModelConfig):
        super().__init__()
        self.D_mem = D_mem
        self.r = r_slots
        self.a_max = config.a_max
        self.budget = config.budget_pm
        self.decay = config.decay_pm
        self.tau_route = config.tau_route_pm

        # State (lazily allocated)
        self.pm_K: Tensor | None = None
        self.pm_V: Tensor | None = None
        self.pm_a: Tensor | None = None

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        """Pre-allocate state tensors."""
        self.pm_K = torch.zeros(BS, self.r, self.D_mem, device=device, dtype=dtype)
        self.pm_V = torch.zeros(BS, self.r, self.D_mem, device=device, dtype=dtype)
        self.pm_a = torch.zeros(BS, self.r, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.pm_K is not None

    def read(self, q: Tensor) -> Tensor:
        """Holographic read. q: [BS,...,D_mem] -> [BS,...,D_mem].

        scores = normalize(q) @ pm_K.T
        weighted = scores * pm_a
        modulation = einsum(weighted, pm_V)
        y = q * modulation  (holographic)
        """
        # q: [BS, N, C, D_mem], pm_K: [BS, r, D_mem]
        q_norm = unit_normalize(q)

        # Expand pm_K for batch matmul: [BS, r, D_mem] -> broadcast over N,C
        scores = torch.einsum("...d, brd -> ...r", q_norm, self.pm_K)  # [BS,N,C,r]

        # Weight by slot strength
        # pm_a: [BS, r] -> [BS, 1, 1, r] for broadcast
        a_expanded = self.pm_a
        for _ in range(scores.dim() - a_expanded.dim()):
            a_expanded = a_expanded.unsqueeze(1)
        weighted = scores * a_expanded  # [BS,N,C,r]

        # Modulation
        modulation = torch.einsum("...r, brd -> ...d", weighted, self.pm_V)  # [BS,N,C,D_mem]

        # Holographic: element-wise multiply
        return q * modulation

    def commit(self, elig_K: Tensor, elig_V: Tensor,
               g: Tensor, slot_logits: Tensor, tau: Tensor):
        """EMA commit of aggregated eligibility.

        elig_K, elig_V: [BS, r, D_mem] — aggregated across N*C
        g: [BS] — write strength
        slot_logits: [BS, r] — slot selection bias
        tau: [BS] — softmax temperature
        """
        # Slot weights
        slot_weights = F.softmax(slot_logits / tau.unsqueeze(-1).clamp(min=0.01), dim=-1)  # [BS, r]

        # Alpha = g * slot_weights
        alpha = (g.unsqueeze(-1) * slot_weights).unsqueeze(-1)  # [BS, r, 1]

        # EMA update
        elig_K_norm = unit_normalize(elig_K)
        elig_V_norm = unit_normalize(elig_V)
        self.pm_K = (1 - alpha) * self.pm_K + alpha * elig_K_norm
        self.pm_V = (1 - alpha) * self.pm_V + alpha * elig_V_norm

        # Strength update
        self.pm_a = (self.pm_a + alpha.squeeze(-1)).clamp(0, self.a_max)

        # Budget enforcement
        self.pm_a = budget_enforce(self.pm_a, self.budget)

    def base_decay(self):
        """pm_a *= decay (called before commit)."""
        if self.pm_a is not None:
            self.pm_a = self.pm_a * self.decay

    def reset_content(self, mask: Tensor):
        """Zero K/V/a for masked streams (doc boundary, non-lifelong)."""
        if self.pm_K is None:
            return
        expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
        self.pm_K = self.pm_K * (~expanded).to(self.pm_K.dtype)
        self.pm_V = self.pm_V * (~expanded).to(self.pm_V.dtype)
        expanded_a = mask.unsqueeze(-1)  # [BS, 1]
        self.pm_a = self.pm_a * (~expanded_a).to(self.pm_a.dtype)


class PMNeuromodulator(nn.Module):
    """Neuromodulator for PM commit decisions.

    Produces fully differentiable outputs for all commit parameters.
    Returns: g [BS], slot_logits [BS, r], tau [BS]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pm_enabled = config.pm_enabled
        self.default_g = config.g_pm_default
        self.tau_floor = config.tau_pm_floor
        self.tau_ceil = config.tau_pm_ceil
        self.default_tau = config.tau_pm

        if self.pm_enabled:
            H = config.neuromod_hidden
            n_scalar = 2  # elig_norm, pm_usage
            n_content = config.content_proj_dim
            n_features = n_scalar + n_content

            self.content_proj = nn.Linear(config.D_mem, n_content)
            self.backbone = nn.Sequential(
                nn.Linear(n_features, H),
                nn.ReLU(),
            )
            self.g_head = nn.Linear(H, 1)
            self.slot_head = nn.Linear(H, config.r)
            self.tau_head = nn.Linear(H, 1)

            nn.init.zeros_(self.backbone[0].bias)
            nn.init.normal_(self.content_proj.weight, std=0.01)
            nn.init.zeros_(self.content_proj.bias)

    def forward(self, elig_summary: Tensor, pm_usage: Tensor,
                content_emb: Tensor | None = None):
        """
        elig_summary: [BS] — eligibility magnitude
        pm_usage: [BS] — sum(pm_a)
        content_emb: [BS, D_mem] — mean eligibility key (optional)
        """
        if not self.pm_enabled:
            BS = elig_summary.shape[0]
            device = elig_summary.device
            return (
                torch.full((BS,), self.default_g, device=device),
                torch.zeros(BS, 8, device=device),  # slot_logits
                torch.full((BS,), self.default_tau, device=device),
            )

        features = [elig_summary.unsqueeze(-1), pm_usage.unsqueeze(-1)]
        if content_emb is not None:
            features.append(self.content_proj(content_emb))
        else:
            features.append(torch.zeros(
                elig_summary.shape[0], self.content_proj.out_features,
                device=elig_summary.device, dtype=elig_summary.dtype
            ))

        x = torch.cat(features, dim=-1)
        h = self.backbone(x)

        g = torch.sigmoid(self.g_head(h)).squeeze(-1)
        slot_logits = self.slot_head(h)
        tau_raw = self.tau_head(h).squeeze(-1)
        tau = self.tau_floor + (self.tau_ceil - self.tau_floor) * torch.sigmoid(tau_raw)

        return g, slot_logits, tau
