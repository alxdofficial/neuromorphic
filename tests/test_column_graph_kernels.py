"""Correctness tests for Triton scatter-gather kernel in column_graph.kernels."""

from __future__ import annotations

import pytest
import torch

from src.column_graph.kernels import (
    TRITON_AVAILABLE,
    weighted_gather,
    weighted_gather_reference,
    WeightedGather,
)
from src.column_graph.topology import build_topology


requires_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="Triton kernel needs CUDA",
)


def _setup(B=2, plane_rows=8, plane_cols=8, L=2, K=8, D_s=64, dtype=torch.float32, seed=0):
    topo = build_topology(
        plane_rows=plane_rows, plane_cols=plane_cols, L=L,
        K=K, p_rewire=0.3, K_intra_fraction=0.5, seed=seed,
    )
    topo = topo.move_to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    N = L * plane_rows * plane_cols

    gen = torch.Generator(device=topo.out_nbrs.device).manual_seed(seed)
    m_out = torch.randn(
        B, N, D_s, device=topo.out_nbrs.device, dtype=dtype, generator=gen,
    )
    w_out_flat = torch.rand(
        B, N * K, device=topo.out_nbrs.device, dtype=dtype, generator=gen,
    )
    return topo, m_out, w_out_flat, K


@requires_cuda
def test_forward_matches_reference_fp32():
    topo, m_out, w_out, K = _setup(dtype=torch.float32)
    ref = weighted_gather_reference(m_out, w_out, topo.edge_src, topo.edge_dst)
    got = weighted_gather(
        m_out, w_out, topo.edge_src, topo.edge_dst, topo.out_nbrs,
        topo.in_src, topo.in_edge_flat, topo.in_mask, topo.K_in_max, K,
    )
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_forward_matches_reference_bf16():
    topo, m_out, w_out, K = _setup(dtype=torch.bfloat16)
    ref = weighted_gather_reference(m_out, w_out, topo.edge_src, topo.edge_dst)
    got = weighted_gather(
        m_out, w_out, topo.edge_src, topo.edge_dst, topo.out_nbrs,
        topo.in_src, topo.in_edge_flat, topo.in_mask, topo.K_in_max, K,
    )
    torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)


@requires_cuda
def test_backward_matches_autograd_fp32():
    topo, m_out, w_out, K = _setup(dtype=torch.float32)

    m_out_a = m_out.clone().detach().requires_grad_(True)
    w_out_a = w_out.clone().detach().requires_grad_(True)
    out_a = WeightedGather.apply(
        m_out_a, w_out_a, topo.edge_src, topo.edge_dst, topo.out_nbrs,
        topo.in_src, topo.in_edge_flat, topo.in_mask, topo.K_in_max, K,
    )
    grad_out = torch.randn_like(out_a)
    out_a.backward(grad_out)

    m_out_b = m_out.clone().detach().requires_grad_(True)
    w_out_b = w_out.clone().detach().requires_grad_(True)
    out_b = weighted_gather_reference(m_out_b, w_out_b, topo.edge_src, topo.edge_dst)
    out_b.backward(grad_out)

    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m_out_a.grad, m_out_b.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w_out_a.grad, w_out_b.grad, atol=1e-4, rtol=1e-4)


@requires_cuda
def test_dev_scale_shapes():
    """Realistic dev-scale: N=4096, K=32, D_s=256, BS=4."""
    topo, m_out, w_out, K = _setup(
        B=4, plane_rows=32, plane_cols=32, L=4, K=32, D_s=256,
        dtype=torch.bfloat16,
    )
    m_out_g = m_out.clone().detach().requires_grad_(True)
    w_out_g = w_out.clone().detach().requires_grad_(True)
    out = WeightedGather.apply(
        m_out_g, w_out_g, topo.edge_src, topo.edge_dst, topo.out_nbrs,
        topo.in_src, topo.in_edge_flat, topo.in_mask, topo.K_in_max, K,
    )
    g = torch.randn_like(out)
    out.backward(g)
    assert m_out_g.grad.shape == m_out.shape
    assert w_out_g.grad.shape == w_out.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(m_out_g.grad).all()
    assert torch.isfinite(w_out_g.grad).all()
