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

    def read_all(self, H_flat: Tensor, cum_pm: Tensor) -> Tensor:
        """PM gain read with causal bias for all banks at once.

        Args:
            H_flat: [BS, N, D] — column states reshaped to flat D
            cum_pm: [BS, N, B, D] — prefix-summed write deltas for all banks

        Returns:
            pm_read: [BS, N, B, D] — gain-modulated output for all banks
        """
        # pm_bias: [BS, B, D] -> [BS, 1, B, D]
        bias = self.pm_bias.unsqueeze(1)
        # H_flat: [BS, N, D] -> [BS, N, 1, D]
        return H_flat.unsqueeze(2) * (1.0 + bias + cum_pm)

    def commit(self, delta_pm_sum: Tensor):
        """Segment-end commit for all banks: pm_bias += sum of deltas, then decay.

        Args:
            delta_pm_sum: [BS, B, D] — sum of per-token deltas across N positions
        """
        # Arithmetic ops create new tensors for autograd — no clone needed
        self.pm_bias = (self.pm_bias + delta_pm_sum) * self.decay_pm

    def reset_states(self, mask: Tensor):
        """Zero bias for masked streams (doc boundary, non-lifelong).

        mask: [BS] bool.
        """
        if self.pm_bias is None:
            return
        expanded = mask[:, None, None]  # [BS, 1, 1]
        self.pm_bias = self.pm_bias * ~expanded
