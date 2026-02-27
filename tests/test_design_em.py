"""Episodic Memory tests (v4)."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.episodic_memory import EpisodicMemory, EMNeuromodulator
from src.model.utils import unit_normalize


BS = 2


class TestEpisodicMemory:
    def test_initialize(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        assert not em.is_initialized()

        em.initialize(BS, torch.device("cpu"), torch.float32)
        assert em.is_initialized()
        assert em.em_K.shape == (BS, cfg.M, cfg.D_mem)
        assert em.em_V.shape == (BS, cfg.M, cfg.D_mem)
        assert em.em_S.shape == (BS, cfg.M)
        assert em.em_age.shape == (BS, cfg.M)

    def test_read_empty(self):
        """Reading from empty EM should not crash."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        q = torch.randn(BS, 4, 2, cfg.D_mem)
        y = em.read(q)
        assert y.shape == q.shape

    def test_read_with_content(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = unit_normalize(torch.randn(BS, cfg.M, cfg.D_mem))
        em.em_V = torch.randn(BS, cfg.M, cfg.D_mem)
        em.em_S = torch.ones(BS, cfg.M)

        q = torch.randn(BS, 4, 2, cfg.D_mem)
        y = em.read(q)
        assert y.shape == q.shape

    def test_novelty_scoring(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        q_nov = torch.randn(BS, 4, 2, cfg.D_mem)
        surprise = torch.rand(BS, 4, 2)
        w_nov = torch.rand(BS, 4, 2)

        novelty = em.score_novelty(q_nov, surprise, w_nov)
        assert novelty.shape == (BS, 4, 2)
        # Novelty should be non-negative
        assert (novelty >= -0.01).all()

    def test_write_updates_strength(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        C_em = cfg.C_em
        cand_K = unit_normalize(torch.randn(BS, C_em, cfg.D_mem))
        cand_V = torch.randn(BS, C_em, cfg.D_mem)
        cand_scores = torch.rand(BS, C_em) + 0.1

        g_em = torch.full((BS,), 0.5)
        tau = torch.ones(BS)
        decay = torch.full((BS,), 0.999)

        s_before = em.em_S.sum().item()
        em.write(cand_K, cand_V, cand_scores, g_em, tau, decay)
        s_after = em.em_S.sum().item()

        assert s_after > s_before

    def test_budget_enforcement(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        # Write many times to fill up
        C_em = cfg.C_em
        for _ in range(20):
            cand_K = unit_normalize(torch.randn(BS, C_em, cfg.D_mem))
            cand_V = torch.randn(BS, C_em, cfg.D_mem)
            cand_scores = torch.rand(BS, C_em) + 0.5
            g_em = torch.ones(BS)
            tau = torch.ones(BS)
            decay = torch.ones(BS)  # no decay
            em.write(cand_K, cand_V, cand_scores, g_em, tau, decay)

        assert em.em_S.sum(dim=-1).max().item() <= cfg.budget_em + 0.01

    def test_age_tick(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_S = torch.ones(BS, cfg.M)  # activate all slots

        em.age_tick(10)
        assert em.em_age.min().item() == 10.0

    def test_reset_states(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = torch.randn(BS, cfg.M, cfg.D_mem)
        em.em_V = torch.randn(BS, cfg.M, cfg.D_mem)
        em.em_S = torch.ones(BS, cfg.M)
        em.em_age = torch.ones(BS, cfg.M) * 100

        mask = torch.tensor([True, False])
        em.reset_states(mask)

        # S and age should be zeroed for masked stream
        assert em.em_S[0].sum() == 0
        assert em.em_age[0].sum() == 0
        # But not for unmasked
        assert em.em_S[1].sum() > 0
        # K/V are preserved (custom reset_states)
        assert em.em_K[0].abs().sum() > 0


class TestEMNeuromodulator:
    def test_output_shapes(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg)

        novelty_mean = torch.rand(BS)
        em_usage = torch.rand(BS)
        content = torch.randn(BS, cfg.D_mem)

        g_em, tau, decay = neuromod(novelty_mean, em_usage, content)
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert decay.shape == (BS,)

    def test_g_bounded(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg)

        g_em, _, _ = neuromod(torch.rand(BS), torch.rand(BS))
        assert (g_em >= cfg.g_em_floor).all()
        assert (g_em <= cfg.g_em_ceil).all()

    def test_decay_bounded(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg)

        _, _, decay = neuromod(torch.rand(BS), torch.rand(BS))
        assert (decay >= cfg.decay_em_floor).all()
        assert (decay <= cfg.decay_em_ceil).all()
