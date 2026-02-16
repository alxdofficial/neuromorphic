"""Parallel affine scan for the neuromorphic LM recurrence.

Implements h_t = a_t * h_{t-1} + b_t for all t in [0, P).

Uses torch.associative_scan (O(log P) parallel prefix) when available on
CUDA, falls back to the sequential loop (which torch.compile unrolls into
efficient fused kernels) otherwise.
"""

import torch
from torch import Tensor

# Try to import associative_scan (prototype in PyTorch 2.4+)
try:
    from torch._higher_order_ops.associative_scan import associative_scan as _assoc_scan
    _HAS_ASSOC_SCAN = True
except ImportError:
    _HAS_ASSOC_SCAN = False


def _combine_fn(left: tuple[Tensor, Tensor],
                right: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Associative operator for affine recurrence.

    (a_L, b_L) ∘ (a_R, b_R) = (a_L * a_R, a_R * b_L + b_R)
    """
    a_l, b_l = left
    a_r, b_r = right
    return (a_l * a_r, a_r * b_l + b_r)


def _sequential_scan(a: Tensor, b: Tensor, h_init: Tensor) -> Tensor:
    """Sequential scan — fallback and torch.compile workhorse.

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


def _parallel_prefix_scan(a: Tensor, b: Tensor, h_init: Tensor) -> Tensor:
    """True parallel prefix scan using associative_scan.

    O(log P) depth instead of O(P) — significant win for large P.
    Prepends h_init as the identity element.
    """
    BS, P, D = a.shape

    # Prepend initial state: (1, h_init) is the identity element
    a_ext = torch.cat([torch.ones(BS, 1, D, dtype=a.dtype, device=a.device), a], dim=1)
    b_ext = torch.cat([h_init.unsqueeze(1).to(a.dtype), b], dim=1)

    # Determine combine mode: pointwise for CUDA (faster), generic for CPU
    mode = "pointwise" if a.is_cuda else "generic"
    result = _assoc_scan(_combine_fn, (a_ext, b_ext), dim=1, combine_mode=mode)

    # Skip the prepended initial state
    return result[1][:, 1:]


def parallel_affine_scan(a: Tensor, b: Tensor, h_init: Tensor) -> Tensor:
    """Compute the affine recurrence h_t = a_t * h_{t-1} + b_t for all t.

    Uses parallel prefix scan on CUDA when available, otherwise falls
    back to the sequential loop (which torch.compile handles well).

    Args:
        a: [BS, P, D] — retention gates (element-wise)
        b: [BS, P, D] — update values
        h_init: [BS, D] — initial hidden state

    Returns:
        h_all: [BS, P, D] — hidden states for all P timesteps
    """
    if _HAS_ASSOC_SCAN and a.is_cuda and not torch.is_grad_enabled():
        # Use parallel prefix scan for inference on CUDA
        # (autograd support is still prototype, so training uses sequential)
        return _parallel_prefix_scan(a, b, h_init)
    return _sequential_scan(a, b, h_init)
