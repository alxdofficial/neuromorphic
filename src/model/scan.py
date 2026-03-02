"""
Causal Linear Scan Layer (v5).

Element-wise linear recurrence: h_t = a_t * h_{t-1} + b_t
where a = sigmoid(proj_a(x)), b = silu(proj_b(x)).

Uses GroupedLinear for per-column independent processing.
fused_scan (FLA HGRN Triton kernel) on CUDA, parallel_scan fallback on CPU.
sequential_scan kept for correctness testing.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .predictive_coding import GroupedLinear, GroupedLayerNorm

try:
    from fla.ops.hgrn.fused_recurrent import fused_recurrent_hgrn
    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


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

    out = torch.empty(BS, N, C, E, device=device, dtype=dtype)
    for t in range(N):
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out  # [BS, N, C, E]


def parallel_scan(a: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """Parallel prefix scan for linear recurrence: h_t = a_t * h_{t-1} + b_t.

    Hillis-Steele algorithm — O(log₂ N) sequential steps, each fully vectorized.
    Composition operator: (a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)

    Args:
        a: [BS, N, C, E] — decay gates (sigmoid, ∈ [0,1])
        b: [BS, N, C, E] — gated inputs
        h0: [BS, C, E] or None — initial hidden state

    Returns:
        h_all: [BS, N, C, E] — all hidden states
    """
    BS, N, C, E = a.shape

    aa = a
    if h0 is not None:
        bb = b.clone()
        bb[:, 0] = aa[:, 0] * h0 + bb[:, 0]
    else:
        bb = b

    # Hillis-Steele inclusive prefix scan
    num_steps = math.ceil(math.log2(N)) if N > 1 else 0
    for d in range(num_steps):
        stride = 1 << d
        # Source: positions [0, N-stride)
        a_prev = aa[:, :-stride]
        b_prev = bb[:, :-stride]
        # Target: positions [stride, N)
        a_cur = aa[:, stride:]
        b_cur = bb[:, stride:]
        # Compose: (a_cur, b_cur) ∘ (a_prev, b_prev)
        new_a = a_cur * a_prev
        new_b = a_cur * b_prev + b_cur
        aa = torch.cat([aa[:, :stride], new_a], dim=1)
        bb = torch.cat([bb[:, :stride], new_b], dim=1)

    return bb


def fused_scan(a_raw: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """Fast linear recurrence: h_t = sigmoid(a_raw_t) * h_{t-1} + b_t.

    Uses FLA's HGRN Triton kernel on CUDA (O(N) fused, ~30x faster than
    sequential). Falls back to parallel_scan on CPU or when FLA unavailable.

    Args:
        a_raw: [BS, N, C, E] — raw gate logits (pre-sigmoid)
        b: [BS, N, C, E] — gated inputs
        h0: [BS, C, E] or None — initial hidden state

    Returns:
        h_all: [BS, N, C, E] — all hidden states
    """
    if _HAS_FLA and a_raw.is_cuda:
        BS, N, C, E = a_raw.shape
        # HGRN expects [B, T, D] with g = logsigmoid(a_raw)
        g = F.logsigmoid(a_raw).permute(0, 2, 1, 3).reshape(BS * C, N, E)
        x = b.permute(0, 2, 1, 3).reshape(BS * C, N, E)
        h0_flat = h0.reshape(BS * C, E) if h0 is not None else None
        out, _ = fused_recurrent_hgrn(x, g, initial_state=h0_flat)
        return out.view(BS, C, N, E).permute(0, 2, 1, 3)
    else:
        return parallel_scan(torch.sigmoid(a_raw), b, h0)


class ScanLayer(nn.Module):
    """Single layer of causal linear recurrence with element-wise gating.

    h_t = a_t * h_{t-1} + b_t
    where a = sigmoid(proj_a(x)), b = silu(proj_b(x))

    Pre-norm residual: out = x + proj_out(scan(norm(x)))
    """

    def __init__(self, C: int, D_col: int, expansion: int, dropout: float = 0.0):
        super().__init__()
        self.C = C
        self.D_col = D_col
        E = D_col * expansion
        self.E = E

        self.norm = GroupedLayerNorm(C, D_col)
        self.proj_in = GroupedLinear(C, D_col, 2 * E)   # -> (a_raw, b_raw)
        self.proj_out = GroupedLinear(C, E, D_col)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        b = F.silu(b_raw)                      # gated input

        h = fused_scan(a_raw, b, h_prev)       # [BS, N, C, E]
        out = self.drop(self.proj_out(h))       # [BS, N, C, D_col]
        return out + residual, h[:, -1]         # residual connection, carry last state
