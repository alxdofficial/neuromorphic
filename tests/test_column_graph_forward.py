"""Forward-shape correctness + gradient flow for ColumnGraphMemory."""

from __future__ import annotations

import pytest
import torch

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.standalone import StandaloneLM


def _tiny_cfg(**overrides) -> ColumnGraphConfig:
    """Small config for fast tests: 2 planes × 8×8 = 128 columns, vocab 256."""
    base = dict(
        plane_rows=8,
        plane_cols=8,
        L=2,
        K=8,
        D_s=64,
        D_id=16,
        K_horizons=4,
        K_buf=4,
        vocab_size=256,
        mod_period=4,
        tbptt_block=8,
        n_attn_heads_in=2,
        n_attn_heads_out=2,
    )
    base.update(overrides)
    return ColumnGraphConfig(**base)


def _make_lm_and_tokens(B=2, T=16, device="cpu", **cfg_kwargs):
    cfg = _tiny_cfg(**cfg_kwargs)
    lm = StandaloneLM(cfg).to(device)
    torch.manual_seed(0)
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    return lm, tokens, cfg


def test_single_step_shape():
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=1)
    lm.memory.begin_segment(B=2, device=tokens.device)
    r = lm.memory.step(tokens[:, 0])
    assert r.logits.shape == (2, cfg.K_horizons, cfg.vocab_size)
    assert r.surprise_ema.shape == (2, cfg.K_horizons)
    assert torch.isfinite(r.logits).all()


def test_walk_segment_shape():
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=16)
    out = lm.walk_segment(tokens, tbptt_block=cfg.tbptt_block)
    assert len(out.logits_per_tick) == 16
    for logits in out.logits_per_tick:
        assert logits.shape == (2, cfg.K_horizons, cfg.vocab_size)
    assert out.last_surprise.shape == (2, cfg.K_horizons)


def test_state_persists_across_ticks():
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=4)
    lm.memory.begin_segment(B=2, device=tokens.device)
    s0 = lm.memory.s.clone()
    lm.memory.step(tokens[:, 0])
    s1 = lm.memory.s.clone()
    # State should have changed after one injection + propagation
    assert not torch.equal(s0, s1)


def test_plasticity_fires_on_mod_period():
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=16)
    lm.memory.begin_segment(B=2, device=tokens.device)
    E_before = lm.memory.E_bias_flat.clone()
    # Run mod_period=4 ticks; plasticity fires on tick 4
    for t in range(cfg.mod_period):
        r = lm.memory.step(tokens[:, t])
    # E_bias should have changed at least slightly (unless neuromod is stuck at zero)
    # At init, heads are zero-init so η=softplus(0)=0.69 > 0, β=0. So Hebbian
    # update is coact * 0.69 > 0 in general — should produce change.
    E_after = lm.memory.E_bias_flat.clone()
    assert not torch.equal(E_before, E_after), "E_bias didn't change on plasticity step"


def test_gradient_flows_through_segment():
    """Backprop through walk_segment with simple loss — all trainable params
    should have non-None, finite gradients."""
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=8)
    out = lm.walk_segment(tokens, tbptt_block=cfg.tbptt_block)
    # Simple loss: mean of all logits
    loss = sum(l.float().mean() for l in out.logits_per_tick)
    loss.backward()
    for name, p in lm.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_gradient_flows_to_neuromod():
    """Ensure neuromod parameters receive gradient signal (confirms plasticity
    is NOT detached within a TBPTT block — neuromod output affects E_bias
    which affects subsequent logits)."""
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=16)
    # Use big-enough T to trigger plasticity (mod_period=4 → at least 4 updates)
    out = lm.walk_segment(tokens, tbptt_block=cfg.tbptt_block)
    loss = sum(l.float().mean() for l in out.logits_per_tick)
    loss.backward()
    # At least one neuromod parameter should have a nonzero gradient.
    neuromod_grads = [
        p.grad for name, p in lm.named_parameters()
        if "neuromod" in name and p.grad is not None
    ]
    assert len(neuromod_grads) > 0, "no neuromod params found"
    nonzero = sum(g.abs().sum().item() for g in neuromod_grads)
    assert nonzero > 0, "neuromod gradients are all exactly zero — check path"


def test_detach_state_breaks_autograd():
    """After detach_state(), a backward pass computed before detach shouldn't
    produce NaN or error; and subsequent forward shouldn't retain graph to before."""
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=2 * cfg_block_for_test())
    lm.memory.begin_segment(B=2, device=tokens.device)
    block = cfg.tbptt_block
    for t in range(block):
        r = lm.memory.step(tokens[:, t])
    loss1 = r.logits.float().mean()
    loss1.backward()
    lm.memory.detach_state()
    # Zero grads and run more
    for p in lm.parameters():
        if p.grad is not None:
            p.grad = None
    for t in range(block, 2 * block):
        r = lm.memory.step(tokens[:, t])
    loss2 = r.logits.float().mean()
    loss2.backward()


def cfg_block_for_test() -> int:
    """Small helper — keep tbptt_block default sync'd with _tiny_cfg."""
    return 8
    # If detach_state worked, this should complete without "trying to backward
    # through a graph a second time" errors.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_runs_on_cuda_bf16():
    """Smoke test on CUDA with bf16 autocast — shape and finite-ness only."""
    lm, tokens, cfg = _make_lm_and_tokens(B=2, T=8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm.walk_segment(tokens, tbptt_block=cfg.tbptt_block)
    for logits in out.logits_per_tick:
        assert logits.shape == (2, cfg.K_horizons, cfg.vocab_size)
        assert torch.isfinite(logits).all()
