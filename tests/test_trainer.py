"""Unit tests for src/training/trainer.py — TBPTTTrainer."""

import pytest
import torch

from src.data.streaming import StreamBatch
from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.training.trainer import TBPTTTrainer
from src.training import span_ops
from tests.conftest import make_tiny_config

BS = 2
VOCAB = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_batch(BS, T, vocab=VOCAB):
    """Create a synthetic StreamBatch."""
    input_ids = torch.randint(0, vocab, (BS, T))
    target_ids = torch.randint(0, vocab, (BS, T))
    prev_token = torch.randint(0, vocab, (BS,))
    return StreamBatch(input_ids=input_ids, target_ids=target_ids, prev_token=prev_token)


def _make_trainer(phase="B", use_rl=False, **overrides):
    """Create a minimal TBPTTTrainer for testing."""
    cfg = make_tiny_config(**overrides)
    cfg.set_phase(phase)
    if use_rl:
        cfg.rl_enabled = True
    model = NeuromorphicLM(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    rl_optimizer = None
    if use_rl:
        rl_params = list(model.rl_parameters())
        if rl_params:
            rl_optimizer = torch.optim.Adam(rl_params, lr=1e-4)

    def dummy_dataloader():
        while True:
            yield _make_batch(BS, cfg.T)

    trainer = TBPTTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloader=dummy_dataloader(),
        config=cfg,
        device=torch.device("cpu"),
        rl_optimizer=rl_optimizer,
    )
    return trainer, cfg, model


# ============================================================================
# train_chunk basic execution
# ============================================================================

class TestTrainChunk:
    def test_returns_metrics_dict(self):
        trainer, cfg, model = _make_trainer("B")
        batch = _make_batch(BS, cfg.T)
        metrics = trainer.train_chunk(batch)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "ppl" in metrics
        assert "grad_norm" in metrics
        assert "valid_tokens" in metrics

    def test_loss_is_finite(self):
        trainer, cfg, model = _make_trainer("B")
        batch = _make_batch(BS, cfg.T)
        metrics = trainer.train_chunk(batch)
        assert metrics["loss"] < 1e6
        assert metrics["ppl"] > 0

    def test_global_step_starts_at_zero(self):
        trainer, cfg, model = _make_trainer("A")
        assert trainer.global_step == 0
        # global_step is incremented in train_steps(), not train_chunk()
        batch = _make_batch(BS, cfg.T)
        trainer.train_chunk(batch)
        # train_chunk itself doesn't increment global_step
        assert trainer.global_step == 0

    def test_multiple_steps(self):
        trainer, cfg, model = _make_trainer("B")
        losses = []
        for _ in range(3):
            batch = _make_batch(BS, cfg.T)
            metrics = trainer.train_chunk(batch)
            losses.append(metrics["loss"])
        assert all(l < 1e6 for l in losses)

    def test_phase_c_with_em(self):
        """Phase C enables EM — train_chunk should still work."""
        trainer, cfg, model = _make_trainer("C")
        batch = _make_batch(BS, cfg.T)
        metrics = trainer.train_chunk(batch)
        assert isinstance(metrics, dict)
        assert metrics["loss"] < 1e6

    def test_phase_d_with_rl(self):
        """Phase D enables RL — train_chunk should run RL rollouts."""
        trainer, cfg, model = _make_trainer("D", use_rl=True)
        batch = _make_batch(BS, cfg.T)
        metrics = trainer.train_chunk(batch)
        assert isinstance(metrics, dict)
        assert "rl_events" in metrics

    def test_valid_fraction_range(self):
        trainer, cfg, model = _make_trainer("B")
        batch = _make_batch(BS, cfg.T)
        metrics = trainer.train_chunk(batch)
        assert 0.0 <= metrics["valid_fraction"] <= 1.0


# ============================================================================
# _forward_span_and_loss
# ============================================================================

class TestForwardSpanAndLoss:
    def test_returns_expected_keys(self):
        trainer, cfg, model = _make_trainer("B")
        model.train()
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        span_targets = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        amp_ctx = torch.autocast("cpu", enabled=False)
        accum = span_ops.SpanAccumulator.create(BS, cfg.B, torch.device("cpu"))

        fwd = trainer._forward_span_and_loss(
            span_ids, span_targets, reset_first, amp_ctx, accum,
            span_start=0, span_end=cfg.P,
        )
        assert "span_loss" in fwd
        assert "span_valid" in fwd
        assert "eot_count" in fwd
        assert "reset_count" in fwd

    def test_loss_is_tensor(self):
        trainer, cfg, model = _make_trainer("B")
        model.train()
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        span_targets = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        amp_ctx = torch.autocast("cpu", enabled=False)
        accum = span_ops.SpanAccumulator.create(BS, cfg.B, torch.device("cpu"))

        fwd = trainer._forward_span_and_loss(
            span_ids, span_targets, reset_first, amp_ctx, accum,
            span_start=0, span_end=cfg.P,
        )
        assert isinstance(fwd["span_loss"], torch.Tensor)


# ============================================================================
# _backward_and_step
# ============================================================================

class TestBackwardAndStep:
    def test_basic_backward(self):
        trainer, cfg, model = _make_trainer("B")
        model.train()
        # Do a forward pass to get a loss
        batch = _make_batch(BS, cfg.T)
        input_ids = batch.input_ids
        target_ids = batch.target_ids

        # Manually do a forward span to get a loss
        span_ids = input_ids[:, :cfg.P]
        reset_first = torch.zeros(BS, dtype=torch.bool)
        logits, _, _ = model.forward_span(span_ids, reset_first)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            target_ids[:, :cfg.P].reshape(-1),
        )

        avg_loss, reg, grad_norm, rl_metrics = trainer._backward_and_step(
            loss, BS * cfg.P, [],
        )
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0
        assert isinstance(rl_metrics, dict)

    def test_nan_loss_raises_with_fail_fast(self):
        """NaN loss with fail_fast=True should raise RuntimeError."""
        trainer, cfg, model = _make_trainer("B")
        model.train()

        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        logits, _, _ = model.forward_span(span_ids, reset_first)
        loss = logits.sum() * float("inf") * 0  # NaN

        with pytest.raises(RuntimeError, match="Non-finite total loss"):
            trainer._backward_and_step(loss, BS * cfg.P, [])


# ============================================================================
# _apply_boundary_updates
# ============================================================================

class TestApplyBoundaryUpdates:
    def test_pm_boundary_applied(self):
        trainer, cfg, model = _make_trainer("B")
        # Forward some tokens first
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        model.forward_span(span_ids, reset_first)

        surprise_mean = torch.ones(BS) * 2.0
        result = span_ops.SpanResult(surprise_mean=surprise_mean, em_stacked={})
        # Should not raise
        trainer._apply_boundary_updates(result, surprise_mean)

    def test_em_boundary_applied_phase_c(self):
        trainer, cfg, model = _make_trainer("C")
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        model.forward_span(span_ids, reset_first)

        D_em = cfg.D_em
        em_stacked = {}
        for b in range(cfg.B):
            em_stacked[b] = (
                torch.randn(BS, 4, D_em),
                torch.randn(BS, 4, D_em),
                torch.rand(BS, 4),
                torch.ones(BS, 4, dtype=torch.bool),
                torch.rand(BS),
            )
        surprise_mean = torch.ones(BS) * 2.0
        result = span_ops.SpanResult(surprise_mean=surprise_mean, em_stacked=em_stacked)
        # Should not raise
        trainer._apply_boundary_updates(result, surprise_mean)
