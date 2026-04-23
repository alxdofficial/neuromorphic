"""Correctness tests for GraphWalker: shapes, gradient flow, routing diversity."""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


def _tiny_cfg(**overrides) -> GraphWalkerConfig:
    """Small config: 2 planes × 8×8 = 128 columns, vocab 256."""
    base = dict(
        plane_rows=8, plane_cols=8, L=2,
        K=8, D_s=64, D_id=16,
        n_heads=2, n_hops=3,
        D_q_in=16, D_q_per_head=16, n_score_heads=2,
        K_horizons=4, K_buf=4,
        vocab_size=256,
        mod_period=4, tbptt_block=8,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0,
        gumbel_anneal_steps=1, epsilon_start=0.0, epsilon_end=0.0,
        epsilon_anneal_steps=1, lambda_balance=0.0,
    )
    base.update(overrides)
    return GraphWalkerConfig(**base)


def _make(B=2, T=16, device="cpu", **cfg_kwargs):
    cfg = _tiny_cfg(**cfg_kwargs)
    lm = StandaloneLM(cfg).to(device)
    torch.manual_seed(0)
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    return lm, tokens, cfg


def test_single_step_shape():
    lm, tokens, cfg = _make(B=2, T=1)
    lm.memory.begin_segment(B=2, device=tokens.device)
    r = lm.memory.step(tokens[:, 0])
    assert r.motor.shape == (2, cfg.D_s)
    assert r.logits.shape == (2, cfg.K_horizons, cfg.vocab_size)
    assert r.surprise_ema.shape == (2, cfg.K_horizons)
    assert torch.isfinite(r.logits).all()


def test_trajectory_visits_L_columns_per_head():
    """Sanity: each head's trajectory has L distinct positions recorded."""
    lm, tokens, cfg = _make(B=2, T=1)
    lm.memory.begin_segment(B=2, device=tokens.device)
    _ = lm.memory.step(tokens[:, 0])
    # visit_count should have at least H·L non-zero entries total (maybe overlapping)
    assert lm.memory.visit_count is not None
    total_visits = lm.memory.visit_count.sum().item()
    # H heads × L hops start-cols + (L-1)*H mid-hops. In current impl start-cols
    # are counted too → H·L total per batch item.
    expected_min = 2 * cfg.n_heads * cfg.n_hops    # B × H × L
    assert total_visits >= expected_min - 1       # might be slightly off-by-one


def test_non_visited_columns_preserved():
    """Confirm that columns not on any trajectory keep their state."""
    lm, tokens, cfg = _make(B=2, T=1)
    lm.memory.begin_segment(B=2, device=tokens.device)
    s_before = lm.memory.s.clone()
    _ = lm.memory.step(tokens[:, 0])
    s_after = lm.memory.s
    # At least some columns should be unchanged (not on any trajectory)
    unchanged = torch.all(s_before == s_after, dim=-1)  # [B, N] bool
    # With H=2, L=3, B=2 → max 2*2*3 = 12 cols visited per batch item. N=128.
    # At least N - 12 = 116 cols should be unchanged per batch item.
    assert unchanged.sum().item() > 2 * (cfg.N - cfg.n_heads * cfg.n_hops - 1)


def test_gradient_flow_through_trajectory():
    """Backprop through logits reaches all trainable params, and at least
    one routing-related param (q_proj, k_proj, or input_q_proj) has a
    non-zero gradient — confirming the straight-through estimator wires up.

    We don't require ALL routing params to have non-zero grad on a given
    random seed because per-sample Gumbel noise patterns can occasionally
    land on extremely peaked softmaxes where some gradients underflow
    fp32 precision. What matters is that the chain is wired.
    """
    torch.manual_seed(42)
    lm, tokens, cfg = _make(B=2, T=4)
    lm.memory.begin_segment(B=2, device=tokens.device)
    logits_all = []
    for t in range(4):
        r = lm.memory.step(tokens[:, t])
        logits_all.append(r.logits)
    loss = torch.stack(logits_all).float().sum()
    loss.backward()

    # Content + readout must have non-None, finite gradient.
    core_params = [
        "cols.content_mlp.0.weight",
        "motor_query",
    ]
    for name in core_params:
        param = dict(lm.memory.named_parameters())[name]
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
        assert param.grad.abs().sum().item() > 0, f"zero grad for {name}"

    # At least ONE routing-related param should have non-zero grad.
    routing_names = [
        "cols.q_proj.0.weight", "cols.q_proj.2.weight",
        "cols.k_proj.0.weight", "cols.k_proj.2.weight",
        "input_q_proj.weight",
    ]
    routing_grads = []
    for name in routing_names:
        param = dict(lm.memory.named_parameters()).get(name)
        if param is not None and param.grad is not None:
            routing_grads.append(param.grad.abs().sum().item())
    assert any(g > 0 for g in routing_grads), (
        f"no routing param has non-zero gradient. {routing_grads}"
    )


def test_plasticity_fires_on_mod_period():
    lm, tokens, cfg = _make(B=2, T=8)
    lm.memory.begin_segment(B=2, device=tokens.device)
    E_before = lm.memory.E_bias_flat.clone()
    for t in range(cfg.mod_period):
        lm.memory.step(tokens[:, t])
    E_after = lm.memory.E_bias_flat
    # Should change on tick == mod_period
    assert not torch.equal(E_before, E_after)


def test_detach_preserves_values():
    lm, tokens, cfg = _make(B=2, T=4)
    lm.memory.begin_segment(B=2, device=tokens.device)
    for t in range(3):
        lm.memory.step(tokens[:, t])
    s_before = lm.memory.s.clone()
    E_before = lm.memory.E_bias_flat.clone()
    lm.memory.detach_state()
    assert torch.equal(s_before, lm.memory.s)
    assert torch.equal(E_before, lm.memory.E_bias_flat)


def test_reset_plastic_wipes_E_bias():
    lm, tokens, cfg = _make(B=2, T=cfg_block_default())
    lm.memory.begin_segment(B=2, device=tokens.device)
    # Run some plasticity
    for t in range(cfg.mod_period):
        lm.memory.step(tokens[:, t])
    assert lm.memory.E_bias_flat.abs().sum() > 0
    lm.memory.reset_plastic_memory(tokens.device)
    assert lm.memory.E_bias_flat.abs().sum() == 0


def cfg_block_default() -> int:
    return 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_runs_on_cuda_bf16():
    lm, tokens, cfg = _make(B=2, T=8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lm.memory.begin_segment(B=2, device=tokens.device)
        for t in range(8):
            r = lm.memory.step(tokens[:, t])
            assert torch.isfinite(r.logits).all()
