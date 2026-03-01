"""
Procedural Memory (v5) — bias vector with causal write buffers.

Per-bank bias vector. Read: gain modulation y = H * (1 + pm_bias + cum_pm).
Write: surprise-driven bias shifts with prefix-sum causal buffer.
Commit: segment-end bias aggregate with decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import StateMixin, runtime_state_dtype


class ProceduralMemory(nn.Module, StateMixin):
    """PM bias vector with causal write buffers.

    State:
        pm_bias: [BS, B, D] — bias vector per bank
    """

    _state_tensor_names = ["pm_bias"]

    def __init__(self, B: int, D: int, decay_pm: float = 0.999):
        super().__init__()
        self.B = B
        self.D = D
        self.decay_pm = decay_pm

        # Learned per-bank learning rate: softplus(raw) -> always positive
        self.raw_lr_pm = nn.Parameter(torch.full([B], -2.0))

        # State (lazily allocated)
        self.pm_bias: Tensor | None = None

    @property
    def lr_pm(self) -> Tensor:
        """Per-bank learning rate [B], always positive."""
        return F.softplus(self.raw_lr_pm)

    def initialize(self, BS: int, device: torch.device, dtype: torch.dtype):
        """Pre-allocate state tensors."""
        self.pm_bias = torch.zeros(BS, self.B, self.D, device=device, dtype=dtype)

    def is_initialized(self) -> bool:
        return self.pm_bias is not None

    def compute_deltas(self, surprise: Tensor) -> Tensor:
        """Per-token PM deltas.

        Args:
            surprise: [BS, N, D] — vector surprise (flat, all banks share same surprise)

        Returns:
            delta_pm: [BS, N, B, D] — per-token bias deltas
        """
        # lr_pm: [B] -> [1, 1, B, 1]
        lr = self.lr_pm[None, None, :, None]
        # surprise: [BS, N, D] -> [BS, N, 1, D]
        return lr * surprise.unsqueeze(2)  # [BS, N, B, D]

    def read(self, H_flat: Tensor, cum_pm_b: Tensor, b: int) -> Tensor:
        """PM gain read with causal bias for bank b.

        Args:
            H_flat: [BS, N, D] — column states reshaped to flat D
            cum_pm_b: [BS, N, D] — prefix-summed write deltas for bank b
            b: bank index

        Returns:
            pm_read: [BS, N, D] — gain-modulated output
        """
        # pm_bias[:, b]: [BS, D] -> [BS, 1, D]
        bias = self.pm_bias[:, b].unsqueeze(1)
        return H_flat * (1.0 + bias + cum_pm_b)

    def commit_bank(self, delta_pm_sum: Tensor, b: int):
        """Segment-end commit for bank b: pm_bias += sum of deltas, then decay.

        Args:
            delta_pm_sum: [BS, D] — sum of per-token deltas across N positions
            b: bank index
        """
        # Reassignment (not in-place) to preserve autograd graph
        new_bias = self.pm_bias.clone()
        new_bias[:, b] = (self.pm_bias[:, b] + delta_pm_sum) * self.decay_pm
        self.pm_bias = new_bias

    def reset_states(self, mask: Tensor):
        """Zero bias for masked streams (doc boundary, non-lifelong).

        mask: [BS] bool.
        """
        if self.pm_bias is None:
            return
        expanded = mask[:, None, None]  # [BS, 1, 1]
        self.pm_bias = self.pm_bias * ~expanded
