"""Procedural Memory tests (v5) — bias vector with causal write buffers."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.procedural_memory import ProceduralMemory


BS = 2


class TestProceduralMemory:
    def test_initialize(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        assert not pm.is_initialized()

        pm.initialize(BS, torch.device("cpu"), torch.float32)
        assert pm.is_initialized()
        assert pm.pm_bias.shape == (BS, cfg.B, cfg.D)

    def test_lr_pm_positive(self):
        """softplus(raw_lr_pm) should always be positive."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        lr = pm.lr_pm
        assert (lr > 0).all()

    def test_compute_deltas_shape(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        surprise = torch.randn(BS, 8, cfg.D)
        deltas = pm.compute_deltas(surprise)
        assert deltas.shape == (BS, 8, cfg.B, cfg.D)

    def test_read_all_shape(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        H_flat = torch.randn(BS, 8, cfg.D)
        cum_pm = torch.randn(BS, 8, cfg.B, cfg.D)
        y = pm.read_all(H_flat, cum_pm)
        assert y.shape == (BS, 8, cfg.B, cfg.D)

    def test_read_all_delta_zero_at_init(self):
        """Delta-only read: zero bias + zero cum_pm -> zero delta."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        H_flat = torch.randn(BS, 8, cfg.D)
        cum_pm = torch.zeros(BS, 8, cfg.B, cfg.D)
        y = pm.read_all(H_flat, cum_pm)
        # Zero bias + zero cum_pm -> delta = H * 0 = 0
        assert torch.allclose(y, torch.zeros_like(y))

    def test_read_all_with_bias(self):
        """Non-zero bias produces non-zero delta."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_bias = torch.randn(BS, cfg.B, cfg.D)
        H_flat = torch.randn(BS, 8, cfg.D)
        cum_pm = torch.zeros(BS, 8, cfg.B, cfg.D)
        y = pm.read_all(H_flat, cum_pm)
        # Non-zero bias -> non-zero delta
        assert y.abs().sum() > 0

    def test_commit(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        delta_sum = torch.randn(BS, cfg.B, cfg.D)
        bias_before = pm.pm_bias.clone()
        pm.commit(delta_sum)
        # All banks should have changed
        assert not torch.allclose(pm.pm_bias, bias_before)

    def test_commit_applies_decay(self):
        """After commit, bias = (bias + delta) * decay."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_bias = torch.ones(BS, cfg.B, cfg.D) * 2.0
        delta_sum = torch.ones(BS, cfg.B, cfg.D)
        pm.commit(delta_sum)
        expected = (2.0 + 1.0) * cfg.decay_pm
        assert torch.allclose(pm.pm_bias,
                              torch.full_like(pm.pm_bias, expected),
                              atol=1e-5)

    def test_reset_states(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_bias = torch.randn(BS, cfg.B, cfg.D)

        # Reset first stream only
        mask = torch.tensor([True, False])
        pm.reset_states(mask)

        assert pm.pm_bias[0].abs().sum() == 0
        assert pm.pm_bias[1].abs().sum() > 0

    def test_differentiable(self):
        """PM bias read should be differentiable through lr_pm."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        surprise = torch.randn(BS, 8, cfg.D)
        deltas = pm.compute_deltas(surprise)
        loss = deltas.sum()
        loss.backward()
        assert pm.raw_lr_pm.grad is not None
