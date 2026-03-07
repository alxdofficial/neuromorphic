"""Episodic Memory tests (v5) — trail-based primitive dictionary."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.episodic_memory import EpisodicMemory, EMNeuromodulator
from src.model.utils import unit_normalize


BS = 2


class TestEpisodicMemory:
    def test_initialize(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        assert not em.is_initialized()

        em.initialize(BS, torch.device("cpu"), torch.float32)
        assert em.is_initialized()
        assert em.em_K.shape == (BS, cfg.B, cfg.M, cfg.D)
        assert em.em_V.shape == (BS, cfg.B, cfg.M, cfg.D)
        assert em.em_S.shape == (BS, cfg.B, cfg.M)

    def test_trail_read_all_empty(self):
        """Trail read from empty memory should not crash."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        seed = torch.randn(BS, 8, cfg.D)
        y = em.trail_read_all(seed)
        assert y.shape == (BS, 8, cfg.D)
        assert torch.isfinite(y).all()

    def test_trail_read_all_with_content(self):
        """Trail read with active primitives should produce non-zero output."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = unit_normalize(torch.randn(BS, cfg.B, cfg.M, cfg.D))
        em.em_V = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, cfg.B, cfg.M)
        seed = torch.randn(BS, 8, cfg.D)
        y = em.trail_read_all(seed)
        assert y.shape == (BS, 8, cfg.D)
        # With content, trail should produce non-zero contribution
        assert y.abs().sum() > 0

    def test_compute_novelty_all_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        w_cand = torch.randn(BS, 8, cfg.D)
        surprise = torch.randn(BS, 8, cfg.D)
        novelty = em.compute_novelty_all(w_cand, surprise)
        assert novelty.shape == (BS, 8, cfg.B)

    def test_compute_novelty_all_nonnegative(self):
        """Novelty should be non-negative."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        w_cand = torch.randn(BS, 8, cfg.D)
        surprise = torch.randn(BS, 8, cfg.D)
        novelty = em.compute_novelty_all(w_cand, surprise)
        assert (novelty >= -0.01).all()

    def test_compute_write_deltas_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        novelty = torch.rand(BS, 8, cfg.B)
        w_cand = torch.randn(BS, 8, cfg.D)
        deltas = em.compute_write_deltas(novelty, w_cand)
        assert deltas.shape == (BS, 8, cfg.B, cfg.D)

    def test_commit_all_updates_state(self):
        """Commit should change em_S (strength increases)."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        w_cand = torch.randn(BS, 8, cfg.D)
        novelty = torch.rand(BS, 8, cfg.B) + 0.1
        g_em = torch.full((BS, cfg.B), 0.5)

        S_before = em.em_S.sum().item()
        em.commit_all(w_cand, novelty, g_em)
        S_after = em.em_S.sum().item()
        assert S_after > S_before

    def test_base_decay(self):
        """base_decay should reduce strengths."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_S = torch.ones(BS, cfg.B, cfg.M)

        S_before = em.em_S.sum().item()
        em.base_decay()
        S_after = em.em_S.sum().item()
        assert S_after < S_before

    def test_usage_all_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        u = em.usage_all()
        assert u.shape == (BS, cfg.B)

    def test_reset_states(self):
        """Reset should re-initialize (not zero) — em_S=0 kills primitives permanently."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_V = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, cfg.B, cfg.M)

        mask = torch.tensor([True, False])
        em.reset_states(mask)

        # Stream 0 re-initialized (not zeroed — zeroing kills EM permanently)
        assert torch.allclose(em.em_S[0], torch.full_like(em.em_S[0], 0.01))
        assert em.em_V[0].abs().sum() == 0  # values reset to zero
        # Keys should be unit-normalized random (not zero)
        norms = em.em_K[0].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        # Stream 1 preserved
        assert em.em_S[1].sum() > 0
        assert em.em_K[1].abs().sum() > 0

    def test_budget_enforcement(self):
        """After many commits, strength should not exceed budget."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps,
                            budget=cfg.budget_em)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        for _ in range(20):
            w_cand = torch.randn(BS, 8, cfg.D)
            novelty = torch.rand(BS, 8, cfg.B) + 0.5
            g_em = torch.ones(BS, cfg.B)
            em.commit_all(w_cand, novelty, g_em)

        # Budget should be enforced
        assert em.em_S.sum(dim=-1).max().item() <= cfg.budget_em + 0.01


class TestEMNeuromodulator:
    def test_output_shape(self):
        B = 2
        neuromod = EMNeuromodulator(hidden=8)
        novelty_mean = torch.rand(BS, B)
        usage = torch.rand(BS, B)
        g = neuromod(novelty_mean, usage)
        assert g.shape == (BS, B)

    def test_g_bounded(self):
        B = 2
        neuromod = EMNeuromodulator(hidden=8)
        g = neuromod(torch.rand(BS, B), torch.rand(BS, B))
        assert (g >= 0.001).all()
        assert (g <= 0.95).all()

    def test_differentiable(self):
        B = 2
        neuromod = EMNeuromodulator(hidden=8)
        novelty = torch.randn(BS, B, requires_grad=True)
        usage = torch.randn(BS, B, requires_grad=True)
        g = neuromod(novelty, usage)
        g.sum().backward()
        assert novelty.grad is not None
        assert usage.grad is not None
