"""Parallel affine scan for the neuromorphic LM recurrence.

Implements h_t = a_t * h_{t-1} + b_t in a sequential loop over P steps.
The loop is NOT the bottleneck — P=32 element-wise ops on [BS, D_h] tensors
are trivial. The speedup comes from batching the expensive matrix
multiplications (gate projections, FFN, attention) around the scan.

A true CUDA parallel scan kernel can replace this later as an optimization.
"""

import torch
from torch import Tensor


def parallel_affine_scan(a: Tensor, b: Tensor, h_init: Tensor) -> Tensor:
    """Compute the affine recurrence h_t = a_t * h_{t-1} + b_t for all t.

    Args:
        a: [BS, P, D] — retention gates (element-wise)
        b: [BS, P, D] — update values
        h_init: [BS, D] — initial hidden state

    Returns:
        h_all: [BS, P, D] — hidden states for all P timesteps
    """
    BS, P, D = a.shape
    # Promote across all three inputs (a, b, h_init) so higher-precision b
    # is not silently downcast.  Uses torch.promote_types on dtype objects
    # (not tensors) so the call is traceable under torch.compile fullgraph.
    out_dtype = torch.promote_types(
        torch.promote_types(a.dtype, b.dtype), h_init.dtype
    )
    h_all = torch.empty(BS, P, D, dtype=out_dtype, device=a.device)
    h = h_init
    for t in range(P):
        h = a[:, t] * h + b[:, t]
        h_all[:, t] = h
    return h_all
