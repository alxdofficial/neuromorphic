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
        assert em.em_age.shape == (BS, cfg.B, cfg.M)

    def test_trail_read_empty(self):
        """Trail read from empty memory should not crash."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        seed = torch.randn(BS, 8, cfg.D)
        y = em.trail_read(seed, b=0)
        assert y.shape == (BS, 8, cfg.D)
        assert torch.isfinite(y).all()

    def test_trail_read_with_content(self):
        """Trail read with active primitives should produce non-zero output."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = unit_normalize(torch.randn(BS, cfg.B, cfg.M, cfg.D))
        em.em_V = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, cfg.B, cfg.M)
        seed = torch.randn(BS, 8, cfg.D)
        y = em.trail_read(seed, b=0)
        assert y.shape == (BS, 8, cfg.D)
        # With content, trail should produce non-zero contribution
        assert y.abs().sum() > 0

    def test_compute_novelty_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        w_cand = torch.randn(BS, 8, cfg.D)
        surprise = torch.randn(BS, 8, cfg.D)
        novelty = em.compute_novelty(w_cand, surprise, b=0)
        assert novelty.shape == (BS, 8)

    def test_compute_novelty_nonnegative(self):
        """Novelty should be non-negative."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        w_cand = torch.randn(BS, 8, cfg.D)
        surprise = torch.randn(BS, 8, cfg.D)
        novelty = em.compute_novelty(w_cand, surprise, b=0)
        assert (novelty >= -0.01).all()

    def test_compute_write_deltas_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        novelty = torch.rand(BS, 8)
        w_cand = torch.randn(BS, 8, cfg.D)
        deltas = em.compute_write_deltas(novelty, w_cand)
        assert deltas.shape == (BS, 8, cfg.D)

    def test_commit_updates_state(self):
        """Commit should change em_S (strength increases)."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        w_cand = torch.randn(BS, 8, cfg.D)
        novelty = torch.rand(BS, 8) + 0.1
        g_em = torch.full((BS,), 0.5)

        S_before = em.em_S.sum().item()
        em.commit(w_cand, novelty, g_em, b=0)
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

    def test_usage_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        u = em.usage(b=0)
        assert u.shape == (BS,)

    def test_reset_states(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_V = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, cfg.B, cfg.M)
        em.em_age = torch.ones(BS, cfg.B, cfg.M) * 100

        mask = torch.tensor([True, False])
        em.reset_states(mask)

        # Stream 0 zeroed
        assert em.em_S[0].sum() == 0
        assert em.em_age[0].sum() == 0
        assert em.em_K[0].abs().sum() == 0
        assert em.em_V[0].abs().sum() == 0
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
            novelty = torch.rand(BS, 8) + 0.5
            g_em = torch.ones(BS)
            em.commit(w_cand, novelty, g_em, b=0)

        # Budget should be enforced
        assert em.em_S.sum(dim=-1).max().item() <= cfg.budget_em + 0.01


class TestEMNeuromodulator:
    def test_output_shape(self):
        neuromod = EMNeuromodulator(hidden=8)
        novelty_mean = torch.rand(BS)
        usage = torch.rand(BS)
        g = neuromod(novelty_mean, usage)
        assert g.shape == (BS,)

    def test_g_bounded(self):
        neuromod = EMNeuromodulator(hidden=8)
        g = neuromod(torch.rand(BS), torch.rand(BS))
        assert (g >= 0.001).all()
        assert (g <= 0.95).all()

    def test_differentiable(self):
        neuromod = EMNeuromodulator(hidden=8)
        novelty = torch.randn(BS, requires_grad=True)
        usage = torch.randn(BS, requires_grad=True)
        g = neuromod(novelty, usage)
        g.sum().backward()
        assert novelty.grad is not None
        assert usage.grad is not None
