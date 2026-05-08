"""Correctness tests for GraphWalker: shapes, gradient flow, routing diversity."""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def _tiny_cfg(**overrides) -> GraphWalkerConfig:
    """Small config: single 16×8 = 128-column substrate, vocab 256."""
    base = dict(
        grid_rows=16, grid_cols=8, radius=2,
        K=8, D_model=64, D_s=64, D_id=16,
        n_heads=2,
        D_q_per_head=16, n_score_heads=2,
        K_horizons=4,
        vocab_size=256,
        mod_period=4, tbptt_block=4, segment_T=8,
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
    assert r.motor.shape == (2, cfg.D_model)
    assert r.motor_state.shape == (2, cfg.D_s)
    assert r.logits.shape == (2, cfg.K_horizons, cfg.vocab_size)
    assert r.surprise_ema.shape == (2, cfg.K_horizons)
    assert torch.isfinite(r.logits).all()


def test_persistent_walker_step_visits_endpoint():
    """Each token, every walker hops once and visit_count records the landing."""
    lm, tokens, cfg = _make(B=2, T=1)
    lm.memory.begin_segment(B=2, device=tokens.device)
    _ = lm.memory.step(tokens[:, 0])
    assert lm.memory.visit_count is not None
    total_visits = lm.memory.visit_count.sum().item()
    expected = 2 * cfg.n_heads                   # B × H landings per step
    assert total_visits == expected


def test_non_visited_columns_preserved():
    """Confirm that columns not on any trajectory keep their state."""
    lm, tokens, cfg = _make(B=2, T=1)
    lm.memory.begin_segment(B=2, device=tokens.device)
    s_before = lm.memory.s.clone()
    _ = lm.memory.step(tokens[:, 0])
    s_after = lm.memory.s
    # At least some columns should be unchanged (not on any trajectory)
    unchanged = torch.all(s_before == s_after, dim=-1)  # [B, N] bool
    # Walker writes only at its current column per step.
    assert unchanged.sum().item() > 2 * (cfg.N - cfg.n_heads - 1)


def test_walker_positions_persist_across_tokens():
    lm, tokens, cfg = _make(B=2, T=2)
    lm.memory.begin_segment(B=2, device=tokens.device)
    _ = lm.memory.step(tokens[:, 0])
    pos_after_0 = lm.memory.walker_pos.clone()
    _ = lm.memory.step(tokens[:, 1])
    pos_after_1 = lm.memory.walker_pos.clone()
    assert pos_after_0.shape == (2, cfg.n_heads)
    assert pos_after_1.shape == (2, cfg.n_heads)
    assert not torch.equal(pos_after_0, pos_after_1)


def test_gradient_flow_through_trajectory():
    """Backprop through logits reaches all trainable params, and at least
    one routing-related param (q_proj, k_proj) has a non-zero gradient —
    confirming the straight-through estimator wires up.

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

    # content_mlp uses DeepContentMLP (in_proj + ResidualFFN blocks).
    params = dict(lm.memory.named_parameters())
    if "cols.content_mlp.in_proj.weight" in params:
        content_name = "cols.content_mlp.in_proj.weight"
    else:
        content_name = "cols.content_mlp.0.weight"
    core_params = [content_name, "motor_query"]
    for name in core_params:
        param = params[name]
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
        assert param.grad.abs().sum().item() > 0, f"zero grad for {name}"

    # At least ONE routing-related param should have non-zero grad.
    routing_names = [
        "cols.q_proj.0.weight", "cols.q_proj.2.weight",
        "cols.k_proj.0.weight", "cols.k_proj.2.weight",
    ]
    routing_grads = []
    for name in routing_names:
        param = dict(lm.memory.named_parameters()).get(name)
        if param is not None and param.grad is not None:
            routing_grads.append(param.grad.abs().sum().item())
    assert any(g > 0 for g in routing_grads), (
        f"no routing param has non-zero gradient. {routing_grads}"
    )


def test_load_balance_loss_carries_gradient_to_routing():
    """load_balance_loss must actually back-prop into routing params."""
    torch.manual_seed(0)
    lm, tokens, cfg = _make(B=2, T=4)
    lm.memory.begin_segment(B=2, device=tokens.device)
    lb_losses = []
    for t in range(4):
        r = lm.memory.step(tokens[:, t])
        lb_losses.append(r.load_balance_loss)

    lb_total = torch.stack(lb_losses).sum()
    assert lb_total.requires_grad, "load_balance_loss must carry gradient"
    lb_total.backward()

    routing_names = ["cols.q_proj.2.weight", "cols.k_proj.2.weight"]
    grads = {
        n: dict(lm.memory.named_parameters())[n].grad for n in routing_names
    }
    assert all(g is not None for g in grads.values()), f"missing grads: {grads}"
    assert any(g.abs().sum().item() > 0 for g in grads.values()), (
        f"load-balance loss produced zero grad on all routing params: "
        f"{ {n: g.abs().sum().item() for n, g in grads.items()} }"
    )


def test_epsilon_exploration_has_no_gradient():
    """When ε-exploration overrides argmax with a uniform sample, the STE
    must not let gradient flow through `soft` (which wasn't conditioned on
    the override) — otherwise we'd train the router to prefer random edges.
    """
    from src.graph_walker.routing import gumbel_top1_softmax
    torch.manual_seed(0)
    # ε=1.0 forces every row to be an exploration pick.
    scores = torch.randn(4, 8, requires_grad=True)
    r = gumbel_top1_softmax(scores, tau=1.0, epsilon=1.0, training=True)
    loss = r.ste_weights.sum()
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() == 0, (
        f"ε=1.0 (all explored) should produce zero gradient on scores; "
        f"got {scores.grad.abs().sum().item()}"
    )


def test_epsilon_zero_keeps_gradient():
    """Sanity: without exploration, STE carries gradient through soft."""
    from src.graph_walker.routing import gumbel_top1_softmax
    torch.manual_seed(0)
    scores = torch.randn(4, 8, requires_grad=True)
    r = gumbel_top1_softmax(scores, tau=1.0, epsilon=0.0, training=True)
    r.ste_weights.sum().backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() > 0


def test_plasticity_fires_on_mod_period():
    """Plasticity is triggered from the training flush (phase1_step), not
    from step(). This test drives enough tokens to close one window
    through phase1_step and checks E_bias changed."""
    lm, tokens, cfg = _make(B=2, T=cfg_block_default())
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    E_before = lm.memory.E_bias_flat.clone()
    phase1_step(
        lm, opt, tokens, tbptt_block=cfg.mod_period,
        amp_dtype=None, training_step=0,
    )
    E_after = lm.memory.E_bias_flat
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
    """Drive a training step to populate E_bias, then call
    reset_plastic_memory and confirm it wipes."""
    lm, tokens, cfg = _make(B=2, T=cfg_block_default())
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    phase1_step(
        lm, opt, tokens, tbptt_block=cfg.mod_period,
        amp_dtype=None, training_step=0,
    )
    assert lm.memory.E_bias_flat.abs().sum() > 0
    lm.memory.reset_plastic_memory()
    assert lm.memory.E_bias_flat.abs().sum() == 0


def cfg_block_default() -> int:
    return 8


def test_phase1_step_runs_and_returns_finite_stats():
    lm, tokens, cfg = _make(B=2, T=8)
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    stats = phase1_step(
        lm, opt, tokens, tbptt_block=4, amp_dtype=None, training_step=0,
    )
    assert stats.loss > 0
    assert stats.ce_loss > 0
    assert stats.load_balance_loss >= 0
    assert len(stats.per_horizon_loss) == cfg.K_horizons
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.per_horizon_loss)
    assert torch.isfinite(torch.tensor(stats.grad_norm))
    assert 0.0 <= stats.visit_entropy <= 1.0 + 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_runs_on_cuda_bf16():
    lm, tokens, cfg = _make(B=2, T=8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lm.memory.begin_segment(B=2, device=tokens.device)
        for t in range(8):
            r = lm.memory.step(tokens[:, t])
            assert torch.isfinite(r.logits).all()
