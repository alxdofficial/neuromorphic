"""Parity tests for `src.graph_walker.triton.lif.LIFDepositFunction`.

The new module replaces ``triton_sparse_update.SparseLIFUpdate`` with a
cudagraph-friendly variant that uses static-shape preprocessing
(sort + diff + cumsum + sentinel-padded ``unique_dests``) instead of
``torch.unique``. The Triton kernel itself is the same forward + backward
math; this test confirms the new wrapper preserves numerics.
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.triton.lif import (
    LIFDepositFunction,
    LIFScratch,
    sparse_lif_update_puretorch,
)
from src.graph_walker.triton_sparse_update import SparseLIFUpdate


def _make_padded_inputs(B: int, N: int, D_s: int, M_real: int,
                         dest_range: int | None = None,
                         seed: int = 0,
                         dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(seed)
    BN = B * N
    if dest_range is None:
        dest_range = BN
    s = torch.randn(BN, D_s, dtype=dtype, device="cuda")
    msgs_real = torch.randn(M_real, D_s, dtype=dtype, device="cuda")
    dests_real = torch.randint(
        0, dest_range, (M_real,), dtype=torch.int64, device="cuda",
    )
    alpha = torch.sigmoid(torch.randn(N, device="cuda"))

    # Pad to M_max with sentinel = BN.
    M_max = max(M_real * 2, 4)  # some headroom
    msgs_pad = torch.zeros(M_max, D_s, dtype=dtype, device="cuda")
    msgs_pad[:M_real] = msgs_real
    dests_pad = torch.full((M_max,), BN, dtype=torch.int64, device="cuda")
    dests_pad[:M_real] = dests_real
    return s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_function_forward_matches_old_path():
    """New static-shape Triton kernel must match the old SparseLIFUpdate."""
    B, N, D_s = 2, 16, 64
    M_real = 8
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max = _make_padded_inputs(
        B, N, D_s, M_real, seed=42,
    )
    BN = B * N

    # Old path — no padding.
    out_old = SparseLIFUpdate.apply(
        s.clone(), msgs_real.clone(), dests_real, alpha.clone(), N,
    )

    # New path — padded inputs + scratch.
    scratch = LIFScratch.allocate(M_max=M_max, U_max=M_max, D_s=D_s,
                                   device=s.device, dtype=s.dtype)
    out_new = LIFDepositFunction.apply(
        s.clone(), msgs_pad.clone(), dests_pad, alpha.clone(), N, scratch,
    )

    torch.cuda.synchronize()
    assert torch.allclose(
        out_old.float(), out_new.float(), atol=1e-2, rtol=1e-2,
    ), f"max|diff|={(out_old - out_new).abs().max():.4e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_function_forward_matches_puretorch_ref():
    """Triton output must match the puretorch reference within bf16 tol."""
    B, N, D_s = 2, 16, 128
    M_real = 16
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max = _make_padded_inputs(
        B, N, D_s, M_real, seed=43,
    )

    out_ref = sparse_lif_update_puretorch(
        s.clone(), msgs_real.clone(), dests_real, alpha.clone(), N,
    )

    scratch = LIFScratch.allocate(M_max=M_max, U_max=M_max, D_s=D_s,
                                   device=s.device, dtype=s.dtype)
    out_new = LIFDepositFunction.apply(
        s.clone(), msgs_pad.clone(), dests_pad, alpha.clone(), N, scratch,
    )

    torch.cuda.synchronize()
    assert torch.allclose(
        out_ref.float(), out_new.float(), atol=1e-2, rtol=1e-2,
    ), f"max|diff|={(out_ref - out_new).abs().max():.4e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_function_backward_grad_msgs_matches_old():
    """grad_msgs from the new Triton path must match the old path."""
    B, N, D_s = 2, 16, 64
    M_real = 8
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max = _make_padded_inputs(
        B, N, D_s, M_real, seed=44,
    )
    grad_up = torch.randn_like(s)

    # Old path
    msgs_old = msgs_real.clone().requires_grad_(True)
    alpha_old = alpha.clone().requires_grad_(True)
    out_old = SparseLIFUpdate.apply(s.clone(), msgs_old, dests_real, alpha_old, N)
    (out_old.float() * grad_up.float()).sum().backward()

    # New path
    msgs_new = msgs_pad.clone().requires_grad_(True)
    alpha_new = alpha.clone().requires_grad_(True)
    scratch = LIFScratch.allocate(M_max=M_max, U_max=M_max, D_s=D_s,
                                   device=s.device, dtype=s.dtype)
    out_new = LIFDepositFunction.apply(
        s.clone(), msgs_new, dests_pad, alpha_new, N, scratch,
    )
    (out_new.float() * grad_up.float()).sum().backward()

    torch.cuda.synchronize()
    # Compare gradients on the real (non-padding) entries.
    assert torch.allclose(
        msgs_old.grad.float(), msgs_new.grad[:M_real].float(),
        atol=1e-2, rtol=1e-2,
    ), f"grad_msgs mismatch: max|diff|={(msgs_old.grad - msgs_new.grad[:M_real]).abs().max():.4e}"
    # Padding entries should have zero grad (they correspond to sentinel dests
    # that the kernel skips).
    assert torch.allclose(
        msgs_new.grad[M_real:],
        torch.zeros_like(msgs_new.grad[M_real:]),
        atol=1e-3,
    ), "padding rows of msg grad should be zero"
    # alpha grad — looser tolerance for atomic-add nondeterminism.
    assert torch.allclose(
        alpha_old.grad, alpha_new.grad, atol=5e-2, rtol=5e-3,
    ), f"grad_alpha mismatch: max|diff|={(alpha_old.grad - alpha_new.grad).abs().max():.4e}"
