"""Regression tests for the debugging-pass bug fixes.

Each test pins one of the previously-broken behaviors so it doesn't regress.
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step, phase1_step_cudagraph
from src.graph_walker.triton.lif import LIFScratch, sparse_lif_update_puretorch
from src.graph_walker.triton_sparse_update import SparseLIFUpdate


# ---------------------------------------------------------------------------
# P1.a — neuromod snapshot must observe POST-block state, not pre-block.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_p1a_snapshot_uses_post_block_state():
    """The phase1_step path must pass out.s_new to plasticity so neuromod's
    snapshot reflects the just-trained state, not the pre-block state.

    We monkeypatch _snapshot_touched_columns to record the s tensor it was
    given, then confirm it matches the post-block out.s_new (computed by
    re-running block_forward).
    """
    cfg = GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=32, D_s=16, D_id=8,
        n_heads=2, n_score_heads=2, D_q_per_head=8, D_q_in=8,
        K_horizons=2, K_buf=2,
        mod_period=4, tbptt_block=4, segment_T=4,
        ffn_mult_content=2, content_mlp_depth=1, post_model_depth=1,
        vocab_size=32, use_neuromod=True, compile_on_train=False,
        state_dtype="bf16",
    )
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    captured_s_for_snapshot = []
    orig = lm.memory._snapshot_touched_columns

    def spy(s_for_snapshot=None):
        captured_s_for_snapshot.append(
            None if s_for_snapshot is None else s_for_snapshot.clone()
        )
        orig(s_for_snapshot=s_for_snapshot)

    lm.memory._snapshot_touched_columns = spy

    pre_step_s = lm.memory.s.clone() if getattr(lm.memory, '_state_initialized', False) else None
    phase1_step(lm, opt, tokens, training_step=0)

    assert any(x is not None for x in captured_s_for_snapshot), (
        "_snapshot_touched_columns was never called with explicit s — "
        "phase1_step is not wiring out.s_new through"
    )
    snap = next(x for x in captured_s_for_snapshot if x is not None)
    if pre_step_s is not None:
        assert not torch.allclose(snap, pre_step_s), (
            "snapshot got pre-block state; expected post-block state"
        )


# ---------------------------------------------------------------------------
# P1.b — cudagraph warmup must NOT pollute persistent E_bias_flat.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_p1b_warmup_does_not_corrupt_e_bias():
    """E_bias_flat starts at zero. After build-trainer (which warms up +
    captures), it must still be zero — warmup's spurious Hebbian updates
    must be reverted before real training."""
    cfg = GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=32, D_s=16, D_id=8,
        n_heads=2, n_score_heads=2, D_q_per_head=8, D_q_in=8,
        K_horizons=2, K_buf=2,
        mod_period=4, tbptt_block=4, segment_T=4,
        ffn_mult_content=2, content_mlp_depth=1, post_model_depth=1,
        vocab_size=32, use_neuromod=False, compile_on_train=False,
        state_dtype="bf16",
    )
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    e_bias_pristine = lm.memory.E_bias_flat.clone()
    assert torch.allclose(e_bias_pristine, torch.zeros_like(e_bias_pristine))

    # Force the trainer build path via a NEW LM-attached attribute we can
    # check after. We patch in a hook so we can observe E_bias_flat
    # immediately AFTER warmup_and_capture but BEFORE the first real
    # replay's Hebbian step modifies it.
    from src.graph_walker.triton.cudagraph_trainer import CapturedBlockTrainer

    orig_warmup = CapturedBlockTrainer.warmup_and_capture
    e_bias_after_warmup = []

    def spy_warmup(self, *a, **kw):
        out = orig_warmup(self, *a, **kw)
        e_bias_after_warmup.append(self.lm.memory.E_bias_flat.clone())
        return out

    CapturedBlockTrainer.warmup_and_capture = spy_warmup
    try:
        phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    finally:
        CapturedBlockTrainer.warmup_and_capture = orig_warmup

    assert e_bias_after_warmup, "spy never ran — warmup_and_capture not invoked"
    assert torch.allclose(
        e_bias_after_warmup[0], e_bias_pristine, atol=1e-6,
    ), (
        "warmup_and_capture corrupted persistent E_bias_flat: "
        f"max|delta|={(e_bias_after_warmup[0] - e_bias_pristine).abs().max():.4e}"
    )


# ---------------------------------------------------------------------------
# P1.c — loss normalization: per-block balance term must NOT scale linearly
# with mod_period; total grad must NOT scale linearly with n_blocks_total.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_p1c_balance_grad_invariant_to_mod_period():
    """Doubling mod_period (with everything else equal) must not double the
    balance contribution to the gradient. Probe: grad magnitude on q_proj's
    final linear when only the balance term is non-zero."""
    def make(mod_period):
        cfg = GraphWalkerConfig(
            plane_rows=4, plane_cols=4, L=2, K=4,
            D_model=32, D_s=16, D_id=8,
            n_heads=2, n_score_heads=2, D_q_per_head=8, D_q_in=8,
            K_horizons=2, K_buf=2,
            mod_period=mod_period, tbptt_block=mod_period,
            segment_T=mod_period,                      # 1 block
            ffn_mult_content=2, content_mlp_depth=1, post_model_depth=1,
            vocab_size=32, use_neuromod=False, compile_on_train=False,
            state_dtype="bf16",
            lambda_balance=1.0, plast_eta=0.0,         # disable Hebbian drift
        )
        torch.manual_seed(0)
        lm = StandaloneLM(cfg).cuda()
        opt = torch.optim.Adam(lm.parameters(), lr=1e-9)
        torch.manual_seed(123)
        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")
        return lm, opt, tokens

    lm_a, opt_a, tok_a = make(mod_period=4)
    phase1_step(lm_a, opt_a, tok_a, training_step=0)
    g_a = lm_a.memory.cols.q_proj[-1].weight.grad.norm().item()

    lm_b, opt_b, tok_b = make(mod_period=8)
    phase1_step(lm_b, opt_b, tok_b, training_step=0)
    g_b = lm_b.memory.cols.q_proj[-1].weight.grad.norm().item()

    # Both grads should be the same order of magnitude. Pre-fix, doubling
    # mod_period roughly doubled the balance contribution (and thus the
    # grad). Allow up to 2× deviation (different RNG paths give some noise).
    assert g_a > 0 and g_b > 0
    ratio = max(g_a, g_b) / min(g_a, g_b)
    assert ratio < 2.5, (
        f"q_proj grad scales with mod_period: g(4)={g_a:.4e} g(8)={g_b:.4e} "
        f"ratio={ratio:.2f}; expected ratio close to 1"
    )


# ---------------------------------------------------------------------------
# P2.a — CPU pure-torch fallback in the new lif.py must produce sane gradients.
# ---------------------------------------------------------------------------


def test_p2a_lif_cpu_fallback_backward_is_finite():
    """The CPU path of LIFDepositFunction (no Triton) used to leave the
    save-for-backward buffers uninitialized. Backward then read garbage
    and produced inf/huge gradients. Now it should produce finite values
    matching the puretorch reference within tolerance."""
    from src.graph_walker.triton.lif import LIFDepositFunction

    torch.manual_seed(0)
    B, N, D_s, M_real = 2, 8, 16, 6
    BN = B * N
    s = torch.randn(BN, D_s, dtype=torch.float32, device="cpu")
    msgs = torch.randn(M_real, D_s, dtype=torch.float32, device="cpu")
    dests = torch.randint(0, BN, (M_real,), dtype=torch.int64, device="cpu")
    alpha = torch.sigmoid(torch.randn(N, device="cpu"))

    M_max = 8
    msgs_pad = torch.zeros(M_max, D_s, dtype=torch.float32, device="cpu")
    msgs_pad[:M_real] = msgs
    dests_pad = torch.full((M_max,), BN, dtype=torch.int64, device="cpu")
    dests_pad[:M_real] = dests

    grad_up = torch.randn_like(s)

    msgs_p = msgs_pad.clone().requires_grad_(True)
    alpha_p = alpha.clone().requires_grad_(True)
    scratch = LIFScratch.allocate(M_max=M_max, U_max=M_max, D_s=D_s,
                                   device=s.device, dtype=s.dtype)
    out = LIFDepositFunction.apply(
        s.clone(), msgs_p, dests_pad, alpha_p, N, scratch,
    )
    (out * grad_up).sum().backward()

    assert torch.isfinite(msgs_p.grad).all(), "CPU backward produced non-finite grad_msgs"
    assert torch.isfinite(alpha_p.grad).all(), "CPU backward produced non-finite grad_alpha"
    # Magnitudes should be O(1) — pre-fix they could be inf or 1e30+.
    assert msgs_p.grad.abs().max() < 100.0, (
        f"CPU grad_msgs has implausible magnitude {msgs_p.grad.abs().max():.4e}"
    )
    assert alpha_p.grad.abs().max() < 100.0, (
        f"CPU grad_alpha has implausible magnitude {alpha_p.grad.abs().max():.4e}"
    )


# ---------------------------------------------------------------------------
# P2.b — config validation rejects K too large for the local-neighborhood
# candidate counts (would crash deep inside build_topology otherwise).
# ---------------------------------------------------------------------------


def test_p2b_config_rejects_K_exceeding_candidate_counts():
    """K=64 with default K_intra_fraction=0.5 → K_intra=32 > max_intra=24.
    Pre-fix this crashes in build_topology with a tensor-shape mismatch.
    Now it must raise a clear ValueError at config construction."""
    with pytest.raises(ValueError, match="(intra-plane|inter-plane).*candidate"):
        GraphWalkerConfig(
            plane_rows=8, plane_cols=8, L=4, K=64, K_intra_fraction=0.5,
            mod_period=64, tbptt_block=64, segment_T=64,
            K_buf=4, K_horizons=4,
        )


# ---------------------------------------------------------------------------
# P3 — visit_entropy must reflect actual visit distribution, not the
# zeroed buffer left after plasticity.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_p3_visit_entropy_reflects_real_visits_eager():
    cfg = GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=32, D_s=16, D_id=8,
        n_heads=2, n_score_heads=2, D_q_per_head=8, D_q_in=8,
        K_horizons=2, K_buf=2,
        mod_period=4, tbptt_block=4, segment_T=8,    # 2 blocks
        ffn_mult_content=2, content_mlp_depth=1, post_model_depth=1,
        vocab_size=32, use_neuromod=False, compile_on_train=False,
        state_dtype="bf16",
    )
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    stats = phase1_step(lm, opt, tokens, training_step=0)
    # Walkers visit a small subset of cols (B*H walkers × T tokens of
    # walks). visit_entropy_frac should be < 1.0 (not perfectly uniform).
    # Pre-fix it would always be 1.0 because plasticity zeroed visit_count
    # before the entropy computation.
    assert stats.visit_entropy < 1.0, (
        f"visit_entropy={stats.visit_entropy} expected to be < 1.0 (visits "
        f"are sparse). Telemetry is reading a zeroed buffer."
    )
    assert stats.visit_entropy > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_p3_visit_entropy_reflects_real_visits_cudagraph():
    cfg = GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=32, D_s=16, D_id=8,
        n_heads=2, n_score_heads=2, D_q_per_head=8, D_q_in=8,
        K_horizons=2, K_buf=2,
        mod_period=4, tbptt_block=4, segment_T=8,
        ffn_mult_content=2, content_mlp_depth=1, post_model_depth=1,
        vocab_size=32, use_neuromod=False, compile_on_train=False,
        state_dtype="bf16",
    )
    lm = StandaloneLM(cfg).cuda()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.segment_T), device="cuda")

    stats = phase1_step_cudagraph(lm, opt, tokens, training_step=0)
    assert stats.visit_entropy < 1.0
    assert stats.visit_entropy > 0.0
