"""
Procedural Memory (v4) — holographic modulation slots.

Per-block (B_blocks instances, batched as BS,B dim). Operates in D space
(block level: C columns concatenated). Read: holographic modulation.
Write: neuromodulated EMA commit.
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
        pm_K: [BS, B, r, D] — key bank (unit-normalized)
        pm_V: [BS, B, r, D] — value bank (modulation patterns)
        pm_a: [BS, B, r] — slot strengths (bounded)
    """

    _state_tensor_names = ["pm_K", "pm_V", "pm_a"]

    def __init__(self, dim: int, r_slots: int, config: ModelConfig):
        super().__init__()
        self.dim = dim
        self.r = r_slots
        self.B = config.B_blocks
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
        self.pm_K = torch.zeros(BS, self.B, self.r, self.dim, device=device, dtype=dtype)
        self.pm_V = torch.zeros(BS, self.B, self.r, self.dim, device=device, dtype=dtype)
        self.pm_a = torch.zeros(BS, self.B, self.r, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.pm_K is not None

    def read(self, q: Tensor) -> Tensor:
        """Holographic read. q: [BS, N, B, D] -> [BS, N, B, D].

        scores = normalize(q) @ pm_K.T
        weighted = scores * pm_a
        modulation = weighted @ pm_V
        y = q * modulation  (holographic)
        """
        # q: [BS, N, B, D], pm_K: [BS, B, r, D]
        q_norm = unit_normalize(q)

        # scores via batched matmul (strided views, no copy)
        q_bn = q_norm.transpose(1, 2)                              # [BS, B, N, D]
        scores = torch.matmul(q_bn, self.pm_K.transpose(-1, -2))   # [BS, B, N, r]
        scores = scores.transpose(1, 2)                             # [BS, N, B, r]

        # Weight by slot strength: pm_a [BS, B, r] -> broadcast [BS, 1, B, r]
        weighted = scores * self.pm_a[:, None, :, :]

        # Modulation via batched matmul (strided views, no copy)
        weighted_bn = weighted.transpose(1, 2)                      # [BS, B, N, r]
        modulation = torch.matmul(weighted_bn, self.pm_V)           # [BS, B, N, D]
        modulation = modulation.transpose(1, 2)                     # [BS, N, B, D]

        # Holographic: element-wise multiply
        return q * modulation

    def read_sliced(self, q: Tensor) -> Tensor:
        """Holographic read with per-column D_col slices.

        q: [BS, N_C, B, C, D_col] -> [BS, N_C, B, C, D_col]

        PM state [BS, B, r, D] is viewed as [BS, B, r, C, D_col] and the
        holographic read operates independently per column at D_col width.
        """
        BS, N_C, B, C, D_col = q.shape

        # View state as [BS, B, r, C, D_col] -> permute to [BS, B, C, r, D_col]
        K = self.pm_K.view(BS, B, self.r, C, D_col).permute(0, 1, 3, 2, 4)
        V = self.pm_V.view(BS, B, self.r, C, D_col).permute(0, 1, 3, 2, 4)

        # Query: [BS, N_C, B, C, D_col] -> [BS, B, C, N_C, D_col]
        q_norm = unit_normalize(q)
        q_r = q_norm.permute(0, 2, 3, 1, 4)

        # Scores: [BS, B, C, N_C, D_col] @ [BS, B, C, D_col, r] -> [BS, B, C, N_C, r]
        scores = torch.matmul(q_r, K.transpose(-1, -2))

        # Weight by slot strength: pm_a [BS, B, r] -> [BS, B, 1, 1, r]
        weighted = scores * self.pm_a[:, :, None, None, :]

        # Modulation: [BS, B, C, N_C, r] @ [BS, B, C, r, D_col] -> [BS, B, C, N_C, D_col]
        modulation = torch.matmul(weighted, V)

        # Back to [BS, N_C, B, C, D_col]
        modulation = modulation.permute(0, 3, 1, 2, 4)

        # Holographic: element-wise multiply
        return q * modulation

    def commit(self, elig_K: Tensor, elig_V: Tensor,
               g: Tensor, slot_logits: Tensor, tau: Tensor):
        """EMA commit of aggregated eligibility.

        elig_K, elig_V: [BS, B, r, D] — aggregated across N positions
        g: [BS, B] — write strength
        slot_logits: [BS, B, r] — slot selection bias
        tau: [BS, B] — softmax temperature
        """
        # Slot weights
        slot_weights = F.softmax(slot_logits / tau.unsqueeze(-1).clamp(min=0.01), dim=-1)  # [BS, B, r]

        # Alpha = g * slot_weights
        alpha = (g.unsqueeze(-1) * slot_weights).unsqueeze(-1)  # [BS, B, r, 1]

        # EMA update (reassignment: pm_K/V/a are saved for backward by pm.read(),
        # so in-place ops would invalidate autograd's saved tensor versions)
        elig_K_norm = unit_normalize(elig_K)
        elig_V_norm = unit_normalize(elig_V)
        self.pm_K = (1 - alpha) * self.pm_K + alpha * elig_K_norm
        self.pm_V = (1 - alpha) * self.pm_V + alpha * elig_V_norm

        # Strength update
        self.pm_a = (self.pm_a + alpha.squeeze(-1)).clamp(0, self.a_max)

        # Budget enforcement (per-stream-per-block: last dim is r)
        self.pm_a = budget_enforce(self.pm_a, self.budget)

    def base_decay(self):
        """pm_a *= decay (called after R loop, before backward).

        Must use reassignment: pm_a is saved for backward by pm.read().
        """
        if self.pm_a is not None:
            self.pm_a = self.pm_a * self.decay

    def reset_content(self, mask: Tensor):
        """Zero K/V/a for masked streams (doc boundary, non-lifelong).

        mask: [BS] bool — expand to [BS, 1, 1, 1] for K/V, [BS, 1, 1] for a.
        """
        if self.pm_K is None:
            return
        expanded = mask[:, None, None, None]  # [BS, 1, 1, 1]
        self.pm_K = self.pm_K * ~expanded
        self.pm_V = self.pm_V * ~expanded
        expanded_a = mask[:, None, None]  # [BS, 1, 1]
        self.pm_a = self.pm_a * ~expanded_a


class PMNeuromodulator(nn.Module):
    """Neuromodulator for PM commit decisions.

    Produces fully differentiable outputs for all commit parameters.
    Returns: g [BS*B], slot_logits [BS*B, r], tau [BS*B], ww [BS*B]
    """

    def __init__(self, D_mem: int, config: ModelConfig):
        super().__init__()
        self.pm_enabled = config.pm_enabled
        self.n_slots = config.r
        self.default_g = config.g_pm_default
        self.tau_floor = config.tau_pm_floor
        self.tau_ceil = config.tau_pm_ceil
        self.default_tau = config.tau_pm
        self.default_ww = config.ww_pm_default
        self.ww_floor = config.ww_pm_floor
        self.ww_ceil = config.ww_pm_ceil

        if self.pm_enabled:
            H = config.neuromod_hidden
            n_scalar = 2  # elig_norm, pm_usage
            n_content = config.content_proj_dim
            n_features = n_scalar + n_content

            self.content_proj = nn.Linear(D_mem, n_content)
            self.backbone = nn.Sequential(
                nn.Linear(n_features, H),
                nn.ReLU(),
            )
            self.g_head = nn.Linear(H, 1)
            self.slot_head = nn.Linear(H, config.r)
            self.tau_head = nn.Linear(H, 1)
            self.ww_head = nn.Linear(H, 1)

            nn.init.zeros_(self.backbone[0].bias)
            nn.init.normal_(self.content_proj.weight, std=0.01)
            nn.init.zeros_(self.content_proj.bias)

            # Bias init: sigmoid(bias) = (default - floor) / (ceil - floor)
            _ww_range = self.ww_ceil - self.ww_floor
            if _ww_range > 0:
                _target_sigmoid = (self.default_ww - self.ww_floor) / _ww_range
                _target_sigmoid = max(1e-4, min(1 - 1e-4, _target_sigmoid))
                import math
                self.ww_head.bias.data.fill_(math.log(_target_sigmoid / (1 - _target_sigmoid)))

    def forward(self, elig_summary: Tensor, pm_usage: Tensor,
                content_emb: Tensor | None = None):
        """
        elig_summary: [BSB] — eligibility magnitude (flattened BS*B)
        pm_usage: [BSB] — sum(pm_a)
        content_emb: [BSB, D_col] — mean eligibility key (optional)

        Returns: g [BSB], slot_logits [BSB, r], tau [BSB], ww [BSB]
        """
        if not self.pm_enabled:
            BS = elig_summary.shape[0]
            device = elig_summary.device
            return (
                torch.full((BS,), self.default_g, device=device),
                torch.zeros(BS, self.n_slots, device=device),  # slot_logits
                torch.full((BS,), self.default_tau, device=device),
                torch.full((BS,), self.default_ww, device=device),
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
        ww_raw = self.ww_head(h).squeeze(-1)
        ww = self.ww_floor + (self.ww_ceil - self.ww_floor) * torch.sigmoid(ww_raw)

        return g, slot_logits, tau, ww
