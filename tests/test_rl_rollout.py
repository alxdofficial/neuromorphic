"""Unit tests for src/training/rl_rollout.py — RL rollout engine."""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.model.state import save_runtime_state, load_runtime_state
from src.training.rl_rollout import (
    BoundarySnapshot,
    RLRolloutEngine,
    detached_runtime_state,
    select_rl_spans,
)
from tests.conftest import make_tiny_config, forward_n_tokens

BS = 2
VOCAB = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_model(phase="D"):
    cfg = make_tiny_config()
    cfg.set_phase(phase)
    return NeuromorphicLM(cfg), cfg


def _make_snapshot(model, cfg, span_end=4):
    """Create a minimal BoundarySnapshot for testing."""
    T = cfg.T
    input_ids = torch.randint(0, VOCAB, (BS, T))
    target_ids = torch.randint(0, VOCAB, (BS, T))
    runtime_state = detached_runtime_state(model)
    return BoundarySnapshot(
        runtime_state=runtime_state,
        span_start=span_end,
        input_ids=input_ids,
        target_ids=target_ids,
        span_surprise_mean=torch.ones(BS),
    )


# ============================================================================
# select_rl_spans
# ============================================================================

class TestSelectRlSpans:
    def test_basic_selection(self):
        result = select_rl_spans(num_spans=8, rl_events=2)
        assert len(result) == 2
        # Should be evenly spaced within usable range [0, 6]
        assert all(0 <= idx < 7 for idx in result)

    def test_excludes_last_span(self):
        """Finding 2 fix: last span index should never be returned."""
        for num_spans in range(2, 20):
            for rl_events in range(1, num_spans + 2):
                result = select_rl_spans(num_spans, rl_events)
                if result:
                    assert max(result) < num_spans - 1, (
                        f"select_rl_spans({num_spans}, {rl_events}) returned "
                        f"{result} with max >= {num_spans - 1}"
                    )

    def test_zero_events_returns_empty(self):
        assert select_rl_spans(num_spans=10, rl_events=0) == []

    def test_single_span_returns_empty(self):
        # Only 1 span means usable=0, can't snapshot
        assert select_rl_spans(num_spans=1, rl_events=1) == []

    def test_more_events_than_usable(self):
        """Requesting more events than usable spans should cap at usable."""
        result = select_rl_spans(num_spans=4, rl_events=10)
        # usable = 3, so at most 3 events
        assert len(result) <= 3
        assert len(result) == 3

    def test_evenly_spaced(self):
        result = select_rl_spans(num_spans=10, rl_events=3)
        assert len(result) == 3
        # Should be roughly evenly distributed
        diffs = [result[i+1] - result[i] for i in range(len(result) - 1)]
        assert all(d > 0 for d in diffs)  # strictly increasing


# ============================================================================
# detached_runtime_state
# ============================================================================

class TestDetachedRuntimeState:
    def test_returns_dict(self):
        model, cfg = _tiny_model("B")
        forward_n_tokens(model, cfg.P, BS=BS, vocab=VOCAB)
        state = detached_runtime_state(model)
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_tensors_detached(self):
        model, cfg = _tiny_model("B")
        forward_n_tokens(model, cfg.P, BS=BS, vocab=VOCAB)
        state = detached_runtime_state(model)
        for path, sub_state in state.items():
            if isinstance(sub_state, dict):
                for k, v in sub_state.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        assert not v.requires_grad, f"{path}.{k} still requires grad"


# ============================================================================
# BoundarySnapshot
# ============================================================================

class TestBoundarySnapshot:
    def test_creation_with_defaults(self):
        model, cfg = _tiny_model("D")
        snap = _make_snapshot(model, cfg)
        assert snap.span_start == 4
        assert snap.span_surprise_mean.shape == (BS,)
        assert snap.pm_elig_norms == {}
        assert snap.em_novelties == {}

    def test_creation_with_em_candidates(self):
        model, cfg = _tiny_model("D")
        D_em = cfg.D_em
        snap = BoundarySnapshot(
            runtime_state=detached_runtime_state(model),
            span_start=4,
            input_ids=torch.randint(0, VOCAB, (BS, cfg.T)),
            target_ids=torch.randint(0, VOCAB, (BS, cfg.T)),
            span_surprise_mean=torch.ones(BS),
            em_cand_K=[torch.randn(BS, cfg.P, D_em) for _ in range(cfg.B)],
            em_cand_V=[torch.randn(BS, cfg.P, D_em) for _ in range(cfg.B)],
            em_cand_score=[torch.rand(BS, cfg.P) for _ in range(cfg.B)],
            em_cand_valid=[torch.ones(BS, cfg.P, dtype=torch.bool) for _ in range(cfg.B)],
        )
        assert len(snap.em_cand_K) == cfg.B


# ============================================================================
# RLRolloutEngine
# ============================================================================

class TestRLRolloutEngine:
    def test_creation(self):
        model, cfg = _tiny_model("D")
        engine = RLRolloutEngine(
            model=model, config=cfg, device=torch.device("cpu"),
            use_amp=False, amp_dtype=torch.float32,
        )
        assert engine.model is model
        assert engine.config is cfg

    def test_creation_with_optimizer(self):
        model, cfg = _tiny_model("D")
        rl_params = list(model.rl_parameters())
        rl_opt = torch.optim.Adam(rl_params, lr=1e-4) if rl_params else None
        engine = RLRolloutEngine(
            model=model, config=cfg, device=torch.device("cpu"),
            use_amp=False, amp_dtype=torch.float32,
            rl_optimizer=rl_opt,
        )
        assert engine.rl_optimizer is rl_opt

    def test_rl_step_empty_snapshots(self):
        model, cfg = _tiny_model("D")
        engine = RLRolloutEngine(
            model=model, config=cfg, device=torch.device("cpu"),
            use_amp=False, amp_dtype=torch.float32,
        )
        result = engine.rl_step([], final_runtime_state={})
        assert isinstance(result, dict)

    def test_rl_step_with_snapshot(self):
        model, cfg = _tiny_model("D")
        # Forward some tokens to populate state
        forward_n_tokens(model, cfg.T, BS=BS, vocab=VOCAB, with_commits=True)

        rl_params = list(model.rl_parameters())
        rl_opt = torch.optim.Adam(rl_params, lr=1e-4) if rl_params else None

        engine = RLRolloutEngine(
            model=model, config=cfg, device=torch.device("cpu"),
            use_amp=False, amp_dtype=torch.float32,
            rl_optimizer=rl_opt,
        )

        snap = _make_snapshot(model, cfg, span_end=cfg.P)
        final_state = detached_runtime_state(model)
        result = engine.rl_step([snap], final_state)
        assert isinstance(result, dict)
        assert "rl_events" in result

    def test_collect_neuromod_grad_norms(self):
        model, cfg = _tiny_model("D")
        engine = RLRolloutEngine(
            model=model, config=cfg, device=torch.device("cpu"),
            use_amp=False, amp_dtype=torch.float32,
        )
        norms = engine.collect_neuromod_grad_norms()
        assert isinstance(norms, dict)

    def test_set_rl_warmup(self):
        model, cfg = _tiny_model("D")
        engine = RLRolloutEngine(
            model=model, config=cfg, device=torch.device("cpu"),
            use_amp=False, amp_dtype=torch.float32,
        )
        engine.set_rl_warmup(100)
        assert engine._rl_warmup_steps == 100
