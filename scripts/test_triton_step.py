"""Correctness + speed test for the Triton fused memory step."""

import sys
import time
import torch

sys.path.insert(0, ".")
from src.model.triton_memory_step import (
    fused_memory_step_triton, fused_memory_step_torch,
)


def test_correctness():
    torch.manual_seed(0)
    BS, NC, N, D_n, alpha = 64, 8, 32, 256, 4
    device = torch.device("cuda")
    dt = torch.bfloat16

    h = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.5
    msg = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.5
    W = torch.randn(BS, NC, N, N, device=device, dtype=dt) * 0.1
    decay = torch.sigmoid(torch.randn(BS, NC, N, device=device, dtype=dt))
    inject_proj = torch.randn(BS, NC, alpha, D_n, device=device, dtype=dt) * 0.3
    output_port_idx = torch.arange(alpha, 2 * alpha, device=device).unsqueeze(0).expand(NC, alpha).contiguous()

    out_mask = torch.zeros(NC, N, device=device, dtype=dt)
    for c in range(NC):
        out_mask[c, output_port_idx[c]] = 1.0

    readout_scale = alpha ** -0.5

    with torch.no_grad():
        h_ref, r_ref = fused_memory_step_torch(
            h, msg, W, decay, inject_proj, out_mask, readout_scale)
        h_tr, r_tr = fused_memory_step_triton(
            h, msg, W, decay, inject_proj, out_mask, readout_scale)

    # Compare
    h_err = (h_ref.float() - h_tr.float()).abs()
    r_err = (r_ref.float() - r_tr.float()).abs()
    print(f"h_out:  max abs err = {h_err.max().item():.4e}  mean = {h_err.mean().item():.4e}")
    print(f"readout: max abs err = {r_err.max().item():.4e}  mean = {r_err.mean().item():.4e}")
    # bf16 has ~8 bits of precision — tolerate 1e-2 in individual positions
    assert h_err.max().item() < 5e-2, f"h mismatch too large: {h_err.max().item()}"
    assert r_err.max().item() < 5e-2, f"readout mismatch too large: {r_err.max().item()}"
    print("✓ Correctness passes (within bf16 tolerance)")


def test_speed():
    torch.manual_seed(0)
    BS, NC, N, D_n, alpha = 64, 8, 32, 256, 4
    device = torch.device("cuda")
    dt = torch.bfloat16

    h = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.5
    msg = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.5
    W = torch.randn(BS, NC, N, N, device=device, dtype=dt) * 0.1
    decay = torch.sigmoid(torch.randn(BS, NC, N, device=device, dtype=dt))
    inject_proj = torch.randn(BS, NC, alpha, D_n, device=device, dtype=dt) * 0.3
    out_mask = torch.zeros(NC, N, device=device, dtype=dt)
    out_mask[:, alpha:2*alpha] = 1.0
    readout_scale = alpha ** -0.5

    # Warmup
    for _ in range(5):
        fused_memory_step_torch(h, msg, W, decay, inject_proj, out_mask, readout_scale)
        fused_memory_step_triton(h, msg, W, decay, inject_proj, out_mask, readout_scale)
    torch.cuda.synchronize()

    # Time PyTorch
    N = 200
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        fused_memory_step_torch(h, msg, W, decay, inject_proj, out_mask, readout_scale)
    torch.cuda.synchronize()
    t_torch = (time.time() - t0) / N * 1e6  # μs per call

    # Time Triton
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        fused_memory_step_triton(h, msg, W, decay, inject_proj, out_mask, readout_scale)
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / N * 1e6

    print(f"PyTorch:  {t_torch:.1f} μs / token")
    print(f"Triton:   {t_triton:.1f} μs / token")
    print(f"Speedup:  {t_torch/t_triton:.2f}×")


def test_backward_matches_torch():
    """Gradient from Triton path should match pure-PyTorch gradient."""
    from src.model.triton_memory_step import fused_memory_step

    torch.manual_seed(42)
    BS, NC, N, D_n, alpha = 4, 2, 16, 32, 2
    device = torch.device("cuda")
    dt = torch.bfloat16

    def make_inputs():
        h = (torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.5).detach().requires_grad_(True)
        msg = (torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.5).detach().requires_grad_(True)
        W = (torch.randn(BS, NC, N, N, device=device, dtype=dt) * 0.1).detach().requires_grad_(True)
        decay = torch.sigmoid(torch.randn(BS, NC, N, device=device, dtype=dt)).detach().requires_grad_(True)
        inject_proj = (torch.randn(BS, NC, alpha, D_n, device=device, dtype=dt) * 0.3).detach().requires_grad_(True)
        out_mask = torch.zeros(NC, N, device=device, dtype=dt)
        out_mask[:, alpha:2*alpha] = 1.0
        return h, msg, W, decay, inject_proj, out_mask

    readout_scale = alpha ** -0.5

    # Run both paths with same inputs
    inputs_a = make_inputs()
    inputs_b = tuple(
        (x.detach().clone().requires_grad_(True) if x.requires_grad else x.clone())
        for x in inputs_a
    )

    h1, r1 = fused_memory_step(*inputs_a, readout_scale, use_triton=True)
    h2, r2 = fused_memory_step(*inputs_b, readout_scale, use_triton=False)

    # Back-prop arbitrary upstream gradients
    torch.manual_seed(7)
    gh = torch.randn_like(h1)
    gr = torch.randn_like(r1)

    (h1 * gh).sum().backward(retain_graph=True)
    (r1 * gr).sum().backward()

    (h2 * gh).sum().backward(retain_graph=True)
    (r2 * gr).sum().backward()

    names = ("h", "msg", "W", "decay", "inject_proj")
    for i, name in enumerate(names):
        g_triton = inputs_a[i].grad
        g_torch = inputs_b[i].grad
        err = (g_triton.float() - g_torch.float()).abs().max().item()
        print(f"  grad {name:14s} max|Δ|={err:.3e}")
        assert err < 1e-1, f"{name} grad mismatch: {err}"
    print("✓ Backward matches PyTorch reference (within bf16 tolerance)")


if __name__ == "__main__":
    test_correctness()
    print()
    test_speed()
    print()
    test_backward_matches_torch()
