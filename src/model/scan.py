"""
Causal Linear Scan Layer (v5).

Element-wise linear recurrence: h_t = a_t * h_{t-1} + b_t
where a = sigmoid(proj_a(x)), b = silu(proj_b(x)).

Uses GroupedLinear for per-column independent processing.
Sequential scan for correctness; torch.compile can fuse the loop.
Parallel scan kernel can replace sequential_scan later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .predictive_coding import GroupedLinear, GroupedLayerNorm


def sequential_scan(a: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """Causal linear recurrence: h_t = a_t * h_{t-1} + b_t.

    Args:
        a: [BS, N, C, E] — decay gates (sigmoid, ∈ [0,1])
        b: [BS, N, C, E] — gated inputs
        h0: [BS, C, E] or None — initial hidden state

    Returns:
        h_all: [BS, N, C, E] — all hidden states
    """
    BS, N, C, E = a.shape
    device, dtype = a.device, a.dtype

    if h0 is None:
        h = torch.zeros(BS, C, E, device=device, dtype=dtype)
    else:
        h = h0

    outputs = []
    for t in range(N):
        h = a[:, t] * h + b[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)  # [BS, N, C, E]


class ScanLayer(nn.Module):
    """Single layer of causal linear recurrence with element-wise gating.

    h_t = a_t * h_{t-1} + b_t
    where a = sigmoid(proj_a(x)), b = silu(proj_b(x))

    Pre-norm residual: out = x + proj_out(scan(norm(x)))
    """

    def __init__(self, C: int, D_col: int, expansion: int):
        super().__init__()
        self.C = C
        self.D_col = D_col
        E = D_col * expansion
        self.E = E

        self.norm = GroupedLayerNorm(C, D_col)
        self.proj_in = GroupedLinear(C, D_col, 2 * E)   # -> (a_raw, b_raw)
        self.proj_out = GroupedLinear(C, E, D_col)

    def forward(self, x: Tensor, h_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: [BS, N, C, D_col]
            h_prev: [BS, C, E] or None — carry from previous segment

        Returns:
            out: [BS, N, C, D_col] — output (with residual)
            h_last: [BS, C, E] — last hidden state for carry
        """
        residual = x
        x_norm = self.norm(x)
        ab = self.proj_in(x_norm)              # [BS, N, C, 2E]
        a_raw, b_raw = ab.chunk(2, dim=-1)
        a = torch.sigmoid(a_raw)               # decay ∈ [0,1]
        b = F.silu(b_raw)                      # gated input

        h = sequential_scan(a, b, h_prev)      # [BS, N, C, E]
        out = self.proj_out(h)                  # [BS, N, C, D_col]
        return out + residual, h[:, -1]         # residual connection, carry last state
