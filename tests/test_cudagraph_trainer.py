"""Smoke + correctness tests for the manual cudagraph capture path.

Validates:
- Captured trainer builds without errors on a tiny config
- Replay produces finite loss values
- Loss decreases over a few iterations (sanity)
- Multiple iterations don't crash (state evolves correctly across replays)
"""

from __future__ import annotations

import math

import pytest
import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step_cudagraph


def _tiny_cfg() -> GraphWalkerConfig:
    return GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=64, D_s=32, D_id=16,
        n_heads=2, n_score_heads=2, D_q_per_head=16, D_q_in=16,
        K_horizons=2, K_buf=2,
        mod_period=8, tbptt_block=8, segment_T=16,
        ffn_mult_content=2,
        content_mlp_depth=1,
        post_model_depth=1,
        vocab_size=64,
        use_neuromod=False,                 # required by CapturedBlockTrainer
        compile_on_train=False,
        state_dtype="bf16",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_capture_runs_and_returns_finite_loss():
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-3)

    B = 2
    tokens = torch.randint(0, cfg.vocab_size, (B, cfg.segment_T), device="cuda")

    stats = phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    assert math.isfinite(stats.loss), f"loss not finite: {stats.loss}"
    assert math.isfinite(stats.ce_loss)
    assert math.isfinite(stats.grad_norm)
    assert stats.ce_loss > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_capture_loss_decreases_over_iterations():
    """A few SGD steps on the same data should drive loss down — confirms
    backward / opt.step are wired through the captured graph correctly."""
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=5e-3)

    B = 2
    tokens = torch.randint(0, cfg.vocab_size, (B, cfg.segment_T), device="cuda")

    losses = []
    for step in range(15):
        stats = phase1_step_cudagraph(lm, opt, tokens, training_step=step)
        losses.append(stats.ce_loss)

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early, f"loss did not decrease: early={early:.4f} late={late:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_rejects_neuromod_enabled():
    """use_neuromod=True must raise — neuromod's _active_delta_nm rebuilds
    per-window with a fresh address, breaking captured-buffer stability."""
    cfg = _tiny_cfg()
    cfg.use_neuromod = True
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    with pytest.raises(NotImplementedError, match="use_neuromod=False"):
        phase1_step_cudagraph(lm, opt, tokens, training_step=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_state_evolves_across_replays():
    """After multiple replays, lm.memory.s should differ from its zero init —
    confirms state writeback into the captured buffer threads correctly."""
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-3)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    # First call builds + captures + replays.
    phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    s_after_one = lm.memory.s.clone()

    # Subsequent calls just replay.
    phase1_step_cudagraph(lm, opt, tokens, training_step=1)
    s_after_two = lm.memory.s.clone()

    # State should have evolved between the two segments (each begin_segment
    # zeroes s, but block_forward writes to it during the segment).
    assert not torch.allclose(s_after_one, s_after_two, atol=1e-3), (
        "state did not evolve across segments"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_plasticity_fires():
    """E_bias_flat must change after a cudagraph step — confirms the
    captured Hebbian update is actually running on each replay."""
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    e_bias_before = lm.memory.E_bias_flat.clone()
    phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    e_bias_after = lm.memory.E_bias_flat
    assert not torch.equal(e_bias_before, e_bias_after), (
        "E_bias_flat did not change — captured Hebbian update never fired"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_surprise_ema_streamed():
    """surprise_ema must move away from zero after a step — confirms the
    captured per-token EMA loop populated it from CE values."""
    cfg = _tiny_cfg()
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    # First step: builds + warms up + captures + replays. begin_segment
    # inside this call allocates surprise_ema (zeros).
    phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    # After first step + Hebbian fire, surprise_ema reflects the block CE
    # values streamed in by the captured per-token EMA loop.
    assert lm.memory.surprise_ema.abs().max() > 1e-3, (
        "surprise_ema did not move from zero — captured EMA loop never fired"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_cudagraph_param_grads_accumulate_across_blocks():
    """A segment with T = 2 * mod_period should produce non-zero param
    gradients — confirms multi-block replay and grad accumulation work."""
    cfg = _tiny_cfg()
    # T = 2 * mod_period so we replay 2 blocks per step
    cfg.segment_T = 2 * cfg.mod_period
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    # opt.step ran inside phase1_step_cudagraph; param.grad has been zeroed
    # by AdamW for the next step. Re-run to populate grads.
    phase1_step_cudagraph(lm, opt, tokens, training_step=1)
    # Sample a few learnable params and verify they have non-zero grads after
    # the captured backward chain.
    for p in lm.parameters():
        if p.grad is None:
            continue
        if p.grad.abs().max() > 0:
            return
    raise AssertionError("no param had non-zero grad after cudagraph replay")
