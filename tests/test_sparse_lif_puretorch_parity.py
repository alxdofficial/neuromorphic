"""Numerical parity: pure-torch sparse LIF must match the Triton path.

The pure-torch path is the cudagraph-compatible default in
`triton_sparse_update.sparse_lif_update`. These tests assert it produces
identical forward output AND identical gradients (within fp tolerance) to
the Triton + custom autograd.Function path it replaces.
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.triton_sparse_update import (
    SparseLIFUpdate,
    sparse_lif_update_puretorch,
)


def _make_inputs(B: int, N: int, D_s: int, M: int,
                 dest_range: int | None = None,
                 seed: int = 0,
                 dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda"):
    torch.manual_seed(seed)
    if dest_range is None:
        dest_range = B * N
    s = torch.randn(B * N, D_s, dtype=dtype, device=device)
    msgs = torch.randn(M, D_s, dtype=dtype, device=device)
    dests = torch.randint(0, dest_range, (M,), dtype=torch.int64, device=device)
    alpha = torch.sigmoid(torch.randn(N, device=device))
    return s, msgs, dests, alpha


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_forward_matches_triton_no_collisions():
    s, msgs, dests, alpha = _make_inputs(B=2, N=8, D_s=64, M=16, seed=0)
    out_triton = SparseLIFUpdate.apply(s.clone(), msgs.clone(), dests, alpha.clone(), 8)
    out_pt = sparse_lif_update_puretorch(s.clone(), msgs.clone(), dests, alpha.clone(), 8)
    assert torch.allclose(out_pt.float(), out_triton.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_forward_matches_triton_with_collisions():
    """Force many messages per destination — exercises the dup handling."""
    s, msgs, dests, alpha = _make_inputs(
        B=2, N=8, D_s=64, M=32, dest_range=4, seed=1,
    )
    out_triton = SparseLIFUpdate.apply(s.clone(), msgs.clone(), dests, alpha.clone(), 8)
    out_pt = sparse_lif_update_puretorch(s.clone(), msgs.clone(), dests, alpha.clone(), 8)
    assert torch.allclose(out_pt.float(), out_triton.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_forward_matches_triton_realistic_scale():
    """B=16, N=1024, D_s=512 — the production hot path scale."""
    B, N, D_s, H = 16, 1024, 512, 4
    M = 2 * B * H
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, seed=2)
    out_triton = SparseLIFUpdate.apply(s.clone(), msgs.clone(), dests, alpha.clone(), N)
    out_pt = sparse_lif_update_puretorch(s.clone(), msgs.clone(), dests, alpha.clone(), N)
    assert torch.allclose(out_pt.float(), out_triton.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_backward_grad_msgs_matches():
    s, msgs, dests, alpha = _make_inputs(B=2, N=16, D_s=128, M=16, seed=3)
    grad_up = torch.randn_like(s)

    # Triton path
    msgs_T = msgs.clone().requires_grad_(True)
    alpha_T = alpha.clone().requires_grad_(True)
    out_T = SparseLIFUpdate.apply(s.clone(), msgs_T, dests, alpha_T, 16)
    (out_T.float() * grad_up.float()).sum().backward()

    # Pure-torch path
    msgs_P = msgs.clone().requires_grad_(True)
    alpha_P = alpha.clone().requires_grad_(True)
    out_P = sparse_lif_update_puretorch(s.clone(), msgs_P, dests, alpha_P, 16)
    (out_P.float() * grad_up.float()).sum().backward()

    assert torch.allclose(
        msgs_T.grad.float(), msgs_P.grad.float(), atol=1e-2, rtol=1e-2,
    ), f"grad_msgs mismatch: max|diff|={(msgs_T.grad - msgs_P.grad).abs().max():.4e}"
    # grad_alpha tolerance is looser because both paths use fp32 atomic adds
    # but in different orders (Triton: per-program atomic to single counter;
    # pure-torch: index_add via inductor codegen). Order-of-sum nondeterminism
    # in fp32 produces ~1e-2 differences at our scales.
    assert torch.allclose(
        alpha_T.grad, alpha_P.grad, atol=5e-2, rtol=5e-3,
    ), f"grad_alpha mismatch: max|diff|={(alpha_T.grad - alpha_P.grad).abs().max():.4e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_backward_untouched_rows_passthrough():
    """grad_s_in[untouched] must equal grad_s_out[untouched] exactly —
    untouched rows pass through identity."""
    B, N, D_s = 2, 16, 64
    M = 8
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, dest_range=4, seed=4)

    s_in = s.clone().requires_grad_(True)
    out = sparse_lif_update_puretorch(
        s_in, msgs.clone(), dests, alpha.clone(), N,
    )
    grad_up = torch.randn_like(s)
    out.backward(grad_up)

    # Identify touched rows.
    touched = torch.zeros(B * N, dtype=torch.bool, device=s.device)
    touched.index_fill_(0, dests, True)
    untouched = ~touched

    # Untouched rows: grad passes through.
    assert torch.equal(s_in.grad[untouched], grad_up[untouched])
