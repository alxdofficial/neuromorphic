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
