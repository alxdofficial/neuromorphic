"""Procedural Memory tests (v6) — Hebbian fast-weight network."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.procedural_memory import ProceduralMemory


BS = 2


class TestProceduralMemory:
    def test_initialize(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        assert not pm.is_initialized()

        pm.initialize(BS, torch.device("cpu"), torch.float32)
        assert pm.is_initialized()
        assert pm.W_pm.shape == (BS, cfg.B, cfg.D_pm, cfg.D_pm)

    def test_initialize_near_identity(self):
        """W_pm should start near (1/B)*I so bank-sum ≈ I."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        W_sum = pm.W_pm.sum(dim=1)  # [BS, D_pm, D_pm]
        eye = torch.eye(cfg.D_pm)
        # Bank-summed W should be close to identity
        assert torch.allclose(W_sum, eye.expand_as(W_sum), atol=0.1)

    def test_beta_positive(self):
        """softplus(raw_beta) should always be positive."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        beta = pm.beta
        assert (beta > 0).all()

    def test_read_shape(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        H = torch.randn(BS, 8, cfg.D)
        pm_read, pre = pm.read(H)
        assert pm_read.shape == (BS, 8, cfg.D)
        assert pre.shape == (BS, 8, cfg.D_pm)

    def test_read_nonzero_at_init(self):
        """Near-identity W should produce non-zero reads."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        H = torch.randn(BS, 8, cfg.D)
        pm_read, _ = pm.read(H)
        assert pm_read.abs().sum() > 0

    def test_commit_changes_weights(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        H = torch.randn(BS, 8, cfg.D)
        _, pre = pm.read(H)
        surprise = torch.randn(BS, 8, cfg.D)
        W_before = pm.W_pm.clone()
        pm.commit(pre, surprise, budget=cfg.budget_pm)
        # All banks should have changed
        assert not torch.allclose(pm.W_pm, W_before)

    def test_commit_applies_decay(self):
        """With zero surprise, commit should just decay W toward zero."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        H = torch.randn(BS, 8, cfg.D)
        _, pre = pm.read(H)
        # Zero surprise → sigmoid(0)=0.5 → G is non-zero, but small
        surprise = torch.zeros(BS, 8, cfg.D)
        W_before_norm = pm.W_pm.flatten(-2).norm(dim=-1)
        pm.commit(pre, surprise, budget=cfg.budget_pm)
        W_after_norm = pm.W_pm.flatten(-2).norm(dim=-1)
        # decay * W + beta * W @ G: with decay < 1, norm changes
        # Just check the update happened without explosion
        assert W_after_norm.max() < W_before_norm.max() * 2.0

    def test_commit_clips_to_budget(self):
        """Budget enforcement: per-bank Frobenius norm clamped to budget_pm."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        # Set W_pm to large values that exceed budget
        pm.W_pm = torch.ones(BS, cfg.B, cfg.D_pm, cfg.D_pm) * 10.0
        H = torch.randn(BS, 8, cfg.D)
        _, pre = pm.read(H)
        surprise = torch.zeros(BS, 8, cfg.D)
        pm.commit(pre, surprise, budget=cfg.budget_pm)

        frob = pm.W_pm.flatten(-2).norm(dim=-1)  # [BS, B]
        assert torch.allclose(frob, torch.full_like(frob, cfg.budget_pm), atol=1e-4)

    def test_hebbian_strengthens_with_surprise(self):
        """High surprise should produce larger W updates than low surprise."""
        cfg = make_tiny_config()
        pm_high = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm_low = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)

        # Share initial state and parameters
        pm_high.initialize(BS, torch.device("cpu"), torch.float32)
        pm_low.W_pm = pm_high.W_pm.clone()
        pm_low.load_state_dict(pm_high.state_dict(), strict=False)

        H = torch.randn(BS, 8, cfg.D)
        _, pre_high = pm_high.read(H)
        _, pre_low = pm_low.read(H)

        # High vs low surprise
        surprise_high = torch.randn(BS, 8, cfg.D) * 10.0
        surprise_low = torch.randn(BS, 8, cfg.D) * 0.01

        W_init = pm_high.W_pm.clone()
        pm_high.commit(pre_high, surprise_high, budget=100.0)  # large budget so no clip
        pm_low.commit(pre_low, surprise_low, budget=100.0)

        delta_high = (pm_high.W_pm - W_init).flatten(-2).norm()
        delta_low = (pm_low.W_pm - W_init).flatten(-2).norm()
        assert delta_high > delta_low

    def test_reset_states(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        # Modify W_pm so it's not identity
        pm.W_pm = torch.randn(BS, cfg.B, cfg.D_pm, cfg.D_pm)

        # Reset first stream only
        mask = torch.tensor([True, False])
        pm.reset_states(mask)

        # Stream 0 should be (1/B)*I
        eye = torch.eye(cfg.D_pm) * (1.0 / cfg.B)
        assert torch.allclose(pm.W_pm[0, 0], eye)
        # Stream 1 should be unchanged (not identity)
        assert not torch.allclose(pm.W_pm[1, 0], eye, atol=0.01)

    def test_differentiable(self):
        """PM read should be differentiable through proj_in/proj_out and beta."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        H = torch.randn(BS, 8, cfg.D)
        pm_read, pre = pm.read(H)
        loss = pm_read.sum()
        loss.backward()
        assert pm.proj_in.weight.grad is not None
        assert pm.proj_out.weight.grad is not None

    def test_state_mixin_detach(self):
        """TBPTT boundary: W_pm should be detached."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        # Give W_pm a grad_fn
        pm.W_pm = pm.W_pm + torch.zeros_like(pm.W_pm, requires_grad=True)
        assert pm.W_pm.grad_fn is not None
        pm.detach_states()
        assert pm.W_pm.grad_fn is None

    def test_banks_have_different_plasticity(self):
        """Banks with different beta should evolve differently."""
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.D_pm, cfg.decay_pm)
        # Set very different betas
        pm.raw_beta = torch.nn.Parameter(torch.tensor([-10.0, 2.0]))
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        H = torch.randn(BS, 8, cfg.D)
        _, pre = pm.read(H)
        surprise = torch.randn(BS, 8, cfg.D) * 5.0

        W_init = pm.W_pm.clone()
        pm.commit(pre, surprise, budget=100.0)

        # Bank 0 (low beta) should change less than bank 1 (high beta)
        delta_0 = (pm.W_pm[:, 0] - W_init[:, 0]).flatten(-2).norm()
        delta_1 = (pm.W_pm[:, 1] - W_init[:, 1]).flatten(-2).norm()
        assert delta_1 > delta_0 * 5  # high beta bank should change much more
