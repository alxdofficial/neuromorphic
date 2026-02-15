"""Parallel affine scan for the neuromorphic LM recurrence.

Implements h_t = a_t * h_{t-1} + b_t for all t in [0, P).

Sequential Python loop; torch.compile unrolls and fuses it into
efficient CUDA kernels with optimized backward.
"""

import torch
from torch import Tensor


def _sequential_scan(a: Tensor, b: Tensor, h_init: Tensor) -> Tensor:
    """Sequential scan — the workhorse.

    torch.compile unrolls the loop (P is a compile-time constant) and
    fuses all elementwise ops into a small number of CUDA kernels,
    including an efficient fused backward.
    """
    BS, P, D = a.shape
    out_dtype = torch.promote_types(
        torch.promote_types(a.dtype, b.dtype), h_init.dtype
    )
    h_all = torch.empty(BS, P, D, dtype=out_dtype, device=a.device)
    h = h_init
    for t in range(P):
        h = a[:, t] * h + b[:, t]
        h_all[:, t] = h
    return h_all


def parallel_affine_scan(a: Tensor, b: Tensor, h_init: Tensor) -> Tensor:
    """Compute the affine recurrence h_t = a_t * h_{t-1} + b_t for all t.

    Args:
        a: [BS, P, D] — retention gates (element-wise)
        b: [BS, P, D] — update values
        h_init: [BS, D] — initial hidden state

    Returns:
        h_all: [BS, P, D] — hidden states for all P timesteps
    """
    return _sequential_scan(a, b, h_init)
