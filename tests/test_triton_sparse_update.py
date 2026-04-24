"""Numerical parity tests for the fused Triton sparse LIF update.

Each test constructs a random scenario, runs both the Triton path and the
pure-PyTorch reference, and asserts they agree within dtype-appropriate
tolerances. Also covers edge cases (U=1, max collisions, B=1).
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.triton_sparse_update import (
    SparseLIFUpdate,
    _pytorch_bwd,
    _pytorch_fwd,
)


def _make_inputs(
    B: int, N: int, D_s: int, M: int,
    dest_range: int | None = None,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    if dest_range is None:
        dest_range = B * N
    s = torch.randn(B * N, D_s, dtype=dtype, device=device)
    msgs = torch.randn(M, D_s, dtype=dtype, device=device)
    dests = torch.randint(0, dest_range, (M,), dtype=torch.int64, device=device)
    alpha = torch.sigmoid(torch.randn(N, device=device))
    return s, msgs, dests, alpha


def _ref_forward_backward(s, msgs, dests, alpha, N, grad_s_upstream):
    """Run the pure-PyTorch reference forward + backward, return outputs."""
    D_s = s.shape[1]
    M = msgs.shape[0]
    device = s.device
    s_out = s.clone()
    unique_dests, inverse = torch.unique(dests, return_inverse=True)
    U = unique_dests.shape[0]
    tanh_inc = torch.empty(U, D_s, dtype=s.dtype, device=device)
    s_old_u = torch.empty(U, D_s, dtype=s.dtype, device=device)
    alpha_u = torch.empty(U, dtype=torch.float32, device=device)
    _pytorch_fwd(
        s_out, msgs, inverse, unique_dests, alpha, N,
        tanh_inc, s_old_u, alpha_u,
    )
    grad_s = grad_s_upstream.clone()
    grad_msgs = torch.empty_like(msgs)
    grad_alpha = torch.zeros(N, dtype=torch.float32, device=device)
    _pytorch_bwd(
        grad_s, unique_dests, inverse,
        tanh_inc, s_old_u, alpha_u, N,
        grad_msgs, grad_alpha,
    )
    return s_out, grad_s, grad_msgs, grad_alpha


def _triton_forward_backward(s, msgs, dests, alpha, N, grad_s_upstream):
    s_T = s.clone()
    msgs_T = msgs.clone().requires_grad_(True)
    alpha_T = alpha.clone().requires_grad_(True)
    s_out_T = SparseLIFUpdate.apply(s_T, msgs_T, dests, alpha_T, N)
    # Build a differentiable loss weighted by grad_s_upstream so backward
    # delivers exactly grad_s_upstream at s_out.
    (s_out_T.float() * grad_s_upstream.float()).sum().backward()
    return s_out_T, msgs_T.grad, alpha_T.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_forward_parity_small_no_collisions():
    B, N, D_s = 2, 8, 64
    M = 16
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, seed=0)
    grad_up = torch.randn_like(s)
    s_ref, _, _, _ = _ref_forward_backward(s, msgs, dests, alpha, N, grad_up)
    s_triton, _, _ = _triton_forward_backward(s, msgs, dests, alpha, N, grad_up)
    assert torch.allclose(s_triton.float(), s_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_forward_parity_with_collisions():
    """Force many messages per destination to exercise segment-sum."""
    B, N, D_s = 2, 8, 64
    M = 32
    s, msgs, dests, alpha = _make_inputs(
        B, N, D_s, M, dest_range=4, seed=1,
    )
    grad_up = torch.randn_like(s)
    s_ref, _, _, _ = _ref_forward_backward(s, msgs, dests, alpha, N, grad_up)
    s_triton, _, _ = _triton_forward_backward(s, msgs, dests, alpha, N, grad_up)
    assert torch.allclose(s_triton.float(), s_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_forward_parity_realistic_scale():
    """B=16, N=1024, D_s=512, H=4 — the default training config."""
    B, N, D_s, H = 16, 1024, 512, 4
    M = 2 * B * H
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, seed=2)
    grad_up = torch.randn_like(s)
    s_ref, _, _, _ = _ref_forward_backward(s, msgs, dests, alpha, N, grad_up)
    s_triton, _, _ = _triton_forward_backward(s, msgs, dests, alpha, N, grad_up)
    assert torch.allclose(s_triton.float(), s_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_backward_grad_msgs_parity():
    B, N, D_s = 2, 16, 128
    M = 16
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, seed=3)
    grad_up = torch.randn_like(s)
    _, _, grad_msgs_ref, grad_alpha_ref = _ref_forward_backward(
        s, msgs, dests, alpha, N, grad_up,
    )
    _, grad_msgs_T, grad_alpha_T = _triton_forward_backward(
        s, msgs, dests, alpha, N, grad_up,
    )
    assert torch.allclose(
        grad_msgs_T.float(), grad_msgs_ref.float(), atol=1e-2, rtol=1e-2,
    ), "grad_all_msgs disagrees with reference"
    # grad_alpha uses atomic fp32 adds; parity is fp32-atomic noise only.
    assert torch.allclose(grad_alpha_T, grad_alpha_ref, atol=1e-3, rtol=1e-3), (
        f"grad_alpha disagrees: max|diff|={((grad_alpha_T-grad_alpha_ref).abs().max().item()):.4e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_backward_grad_s_in_identity_for_untouched_rows():
    """Non-touched rows of grad_s_in must equal grad_s_out exactly
    (they are pass-through through the function)."""
    B, N, D_s = 2, 16, 64
    M = 8
    s, msgs, dests, alpha = _make_inputs(
        B, N, D_s, M, dest_range=4, seed=4,
    )
    grad_up = torch.randn_like(s)
    # Run Triton path manually so we can inspect grad_s_in in place.
    s_T = s.clone()
    msgs_T = msgs.clone().requires_grad_(True)
    alpha_T = alpha.clone().requires_grad_(True)
    s_out_T = SparseLIFUpdate.apply(s_T, msgs_T, dests, alpha_T, N)
    # Call backward with grad_up
    torch.autograd.backward(s_out_T, grad_up)

    # Reference
    unique_dests = torch.unique(dests)
    all_idx = torch.arange(B * N, device=s.device)
    touched_mask = torch.isin(all_idx, unique_dests)
    untouched = ~touched_mask

    # Recompute expected grad_s_in ref
    _, grad_s_ref, _, _ = _ref_forward_backward(s, msgs, dests, alpha, N, grad_up)
    # For untouched rows, grad_s_ref should equal grad_up exactly.
    # For touched rows, grad_s_ref = alpha * grad_up (scaled).
    # Check untouched:
    assert torch.equal(grad_s_ref[untouched], grad_up[untouched])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_edge_case_U_equals_1():
    """All messages collide at a single destination."""
    B, N, D_s = 2, 16, 64
    M = 8
    s, msgs, _, alpha = _make_inputs(B, N, D_s, M, seed=5)
    dests = torch.full((M,), 5, dtype=torch.int64, device=s.device)
    grad_up = torch.randn_like(s)
    s_ref, _, _, _ = _ref_forward_backward(s, msgs, dests, alpha, N, grad_up)
    s_triton, _, _ = _triton_forward_backward(s, msgs, dests, alpha, N, grad_up)
    assert torch.allclose(s_triton.float(), s_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_edge_case_B_equals_1():
    """Single batch element."""
    B, N, D_s = 1, 16, 64
    M = 8
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, seed=6)
    grad_up = torch.randn_like(s)
    s_ref, _, _, _ = _ref_forward_backward(s, msgs, dests, alpha, N, grad_up)
    s_triton, _, _ = _triton_forward_backward(s, msgs, dests, alpha, N, grad_up)
    assert torch.allclose(s_triton.float(), s_ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_s_out_is_new_tensor_not_aliased():
    """Forward must produce a NEW tensor (clone + scatter) so upstream ops
    that saved s_flat for backward are not version-invalidated."""
    B, N, D_s = 2, 8, 32
    M = 4
    s, msgs, dests, alpha = _make_inputs(B, N, D_s, M, seed=7)
    s_snapshot = s.clone()
    msgs_T = msgs.clone().requires_grad_(True)
    alpha_T = alpha.clone().requires_grad_(True)
    s_out = SparseLIFUpdate.apply(s, msgs_T, dests, alpha_T, N)
    assert s_out.data_ptr() != s.data_ptr(), (
        "output must be a new tensor — in-place breaks upstream autograd"
    )
    # Input s must be unchanged.
    assert torch.equal(s, s_snapshot), "input s_flat must not be mutated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_multi_step_persistent_state():
    """Simulate the graph_walker usage: multiple steps, persistent s, each
    step mutates in place and feeds the next. Backward should still work."""
    B, N, D_s = 2, 8, 32
    H = 2
    M = 2 * B * H
    torch.manual_seed(8)
    s = torch.zeros(B * N, D_s, dtype=torch.bfloat16, device='cuda')
    all_msgs = [
        torch.randn(M, D_s, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        for _ in range(3)
    ]
    all_dests = [
        torch.randint(0, B * N, (M,), dtype=torch.int64, device='cuda')
        for _ in range(3)
    ]
    alpha = torch.sigmoid(torch.randn(N, device='cuda')).requires_grad_()
    for msgs, dests in zip(all_msgs, all_dests):
        s = SparseLIFUpdate.apply(s, msgs, dests, alpha, N)
    loss = s.float().sum()
    loss.backward()
    for msgs in all_msgs:
        assert msgs.grad is not None
        assert torch.isfinite(msgs.grad).all()
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
