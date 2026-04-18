"""Regression tests for reviewer-flagged gaps:

(a) Invalid Config.NC ≠ NC_pools gets rejected at validate() time.
(b) Triton dispatcher falls through to PyTorch when shapes are not
    kernel-compatible (non-power-of-2 N or D_n), and also on non-CUDA /
    non-bf16.
(c) One full train step runs without crashing on tier_tiny.

These catch the classes of bugs the reviewer found (config validates but
runtime crashes; Triton gated only on cuda+bf16; test coverage thin).
"""

import pytest
import torch

from src.model.config import Config
from src.model.model import Model
from src.model.triton_memory_step import fused_memory_step, fused_memory_step_torch


def test_invalid_nc_rejected_by_validate():
    """NC != NC_pools must raise at validate()."""
    # D=2048, D_n=256 → NC_pools = 8. Set N_cells = 4 → should fail.
    with pytest.raises(ValueError, match="N_cells"):
        Config.tier_a(N_cells=4)


def test_valid_config_accepted():
    """Default tier_a and tier_tiny validate without error."""
    Config.tier_a()
    Config.tier_tiny()


def test_triton_falls_back_on_cpu():
    """Non-CUDA inputs must route to the PyTorch reference, not the Triton kernel."""
    BS, NC, N, D_n, alpha = 2, 2, 16, 16, 2
    h = torch.randn(BS, NC, N, D_n)
    msg = torch.randn(BS, NC, N, D_n)
    W = torch.randn(BS, NC, N, N)
    decay = torch.sigmoid(torch.randn(BS, NC, N))
    inject_proj = torch.randn(BS, NC, alpha, D_n)
    out_mask = torch.zeros(NC, N)
    out_mask[:, alpha:2*alpha] = 1.0

    h_out, readout = fused_memory_step(
        h, msg, W, decay, inject_proj, out_mask, alpha ** -0.5)
    # Should match the explicit PyTorch reference.
    h_ref, r_ref = fused_memory_step_torch(
        h, msg, W, decay, inject_proj, out_mask, alpha ** -0.5)
    assert torch.allclose(h_out, h_ref)
    assert torch.allclose(readout, r_ref)


def test_triton_falls_back_on_non_power_of_2_nc():
    """Non-power-of-2 N must fall through to PyTorch even on CUDA+bf16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    dt = torch.bfloat16
    BS, NC, N, D_n, alpha = 2, 2, 24, 32, 2   # N=24 NOT a power of 2

    h = torch.randn(BS, NC, N, D_n, device=device, dtype=dt)
    msg = torch.randn(BS, NC, N, D_n, device=device, dtype=dt)
    W = torch.randn(BS, NC, N, N, device=device, dtype=dt) * 0.1
    decay = torch.sigmoid(torch.randn(BS, NC, N, device=device, dtype=dt))
    inject_proj = torch.randn(BS, NC, alpha, D_n, device=device, dtype=dt)
    out_mask = torch.zeros(NC, N, device=device, dtype=dt)
    out_mask[:, alpha:2*alpha] = 1.0

    # This should NOT raise a Triton compile error — the dispatcher should
    # detect non-power-of-2 N and route to PyTorch.
    h_out, readout = fused_memory_step(
        h, msg, W, decay, inject_proj, out_mask, alpha ** -0.5)
    assert h_out.shape == h.shape
    assert readout.shape == (BS, NC, D_n)


def test_one_train_step():
    """Smoke: Model + optimizer run one full forward+backward+step on tier_tiny."""
    cfg = Config.tier_tiny()
    model = Model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    BS, T = 2, cfg.T
    input_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    target_ids = torch.randint(0, cfg.vocab_size, (BS, T))

    model.train()
    result = model.forward_chunk(input_ids, target_ids=target_ids)
    loss = result["loss"]
    assert torch.isfinite(loss)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    # State should still be initialized and finite after one step.
    assert model.memory.is_initialized
    assert torch.isfinite(model.memory.h).all()
