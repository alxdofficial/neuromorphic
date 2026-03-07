"""
Causal Linear Scan Layer (v5.1 — dense).

Element-wise linear recurrence: h_t = a_t * h_{t-1} + b_t
where a = sigmoid(proj_a(x)), b = silu(proj_b(x)).

Dense nn.Linear projections for GPU efficiency. PCM stays grouped separately.
fused_scan (FLA HGRN Triton kernel) on CUDA, parallel_scan fallback on CPU.
sequential_scan kept for correctness testing.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from fla.ops.hgrn.fused_recurrent import fused_recurrent_hgrn
    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


def sequential_scan(a: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """Causal linear recurrence: h_t = a_t * h_{t-1} + b_t.

    Args:
        a: [BS, N, E] — decay gates (sigmoid, in [0,1])
        b: [BS, N, E] — gated inputs
        h0: [BS, E] or None — initial hidden state

    Returns:
        h_all: [BS, N, E] — all hidden states
    """
    BS, N, E = a.shape
    device, dtype = a.device, a.dtype

    if h0 is None:
        h = torch.zeros(BS, E, device=device, dtype=dtype)
    else:
        h = h0

    out = torch.empty(BS, N, E, device=device, dtype=dtype)
    for t in range(N):
        h = a[:, t] * h + b[:, t]
        out[:, t] = h
    return out  # [BS, N, E]


def parallel_scan(a: Tensor, b: Tensor, h0: Tensor | None = None) -> Tensor:
    """Parallel prefix scan for linear recurrence: h_t = a_t * h_{t-1} + b_t.

    Hillis-Steele algorithm — O(log2 N) sequential steps, each fully vectorized.
    Composition operator: (a2, b2) . (a1, b1) = (a2*a1, a2*b1 + b2)

    Args:
        a: [BS, N, E] — decay gates (sigmoid, in [0,1])
        b: [BS, N, E] — gated inputs
        h0: [BS, E] or None — initial hidden state

    Returns:
        h_all: [BS, N, E] — all hidden states
    """
    BS, N, E = a.shape

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
        # Compose: (a_cur, b_cur) . (a_prev, b_prev)
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
        a_raw: [BS, N, E] — raw gate logits (pre-sigmoid)
        b: [BS, N, E] — gated inputs
        h0: [BS, E] or None — initial hidden state

    Returns:
        h_all: [BS, N, E] — all hidden states
    """
    if _HAS_FLA and a_raw.is_cuda:
        # HGRN expects [B, T, D] with g = logsigmoid(a_raw)
        g = F.logsigmoid(a_raw)
        x = b
        out, _ = fused_recurrent_hgrn(x, g, initial_state=h0)
        return out
    else:
        return parallel_scan(torch.sigmoid(a_raw), b, h0)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Cheaper than LayerNorm (no mean subtraction), fuses better with torch.compile.
    """

    def __init__(self, D: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class ScanLayer(nn.Module):
    """Single layer of causal linear recurrence with element-wise gating.

    h_t = a_t * h_{t-1} + b_t
    where a = sigmoid(proj_a(x)), b = silu(proj_b(x))

    Pre-norm residual: out = x + proj_out(scan(norm(x)))
    Dense projections (nn.Linear) for GPU efficiency.
    """

    def __init__(self, D: int, d_inner: int, dropout: float = 0.0,
                 n_layers: int = 1):
        super().__init__()
        self.D = D
        self.d_inner = d_inner

        self.norm = RMSNorm(D)
        self.proj_in = nn.Linear(D, 2 * d_inner)
        self.proj_out = nn.Linear(d_inner, D)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # GPT-2 style depth scaling: each residual branch contributes 1/√(2*n_layers)
        if n_layers > 1:
            with torch.no_grad():
                self.proj_out.weight.mul_(1.0 / math.sqrt(2 * n_layers))

    def forward(self, x: Tensor, h_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: [BS, N, D]
            h_prev: [BS, d_inner] or None — carry from previous segment

        Returns:
            out: [BS, N, D] — output (with residual)
            h_last: [BS, d_inner] — last hidden state for carry
        """
        residual = x
        ab = self.proj_in(self.norm(x))        # [BS, N, 2*d_inner]
        a_raw, b_raw = ab.chunk(2, dim=-1)
        h = fused_scan(a_raw, F.silu(b_raw), h_prev)  # [BS, N, d_inner]
        return self.drop(self.proj_out(h)) + residual, h[:, -1]
