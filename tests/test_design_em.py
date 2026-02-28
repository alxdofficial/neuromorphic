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
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        assert not em.is_initialized()

        em.initialize(BS, torch.device("cpu"), torch.float32)
        assert em.is_initialized()
        assert em.em_K.shape == (BS, B, cfg.M, cfg.D)
        assert em.em_V.shape == (BS, B, cfg.M, cfg.D)
        assert em.em_S.shape == (BS, B, cfg.M)
        assert em.em_age.shape == (BS, B, cfg.M)

    def test_read_empty(self):
        """Reading from empty EM should not crash."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        q = torch.randn(BS, 4, B, cfg.D)
        y = em.read(q)
        assert y.shape == q.shape

    def test_read_with_content(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = unit_normalize(torch.randn(BS, B, cfg.M, cfg.D))
        em.em_V = torch.randn(BS, B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, B, cfg.M)

        q = torch.randn(BS, 4, B, cfg.D)
        y = em.read(q)
        assert y.shape == q.shape

    def test_novelty_scoring(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        q_nov = torch.randn(BS, 4, B, cfg.D)
        surprise = torch.rand(BS, 4, B)
        w_nov = torch.rand(BS, 4, B)

        novelty = em.score_novelty(q_nov, surprise, w_nov)
        assert novelty.shape == (BS, 4, B)
        # Novelty should be non-negative
        assert (novelty >= -0.01).all()

    def test_write_updates_strength(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        C_em = cfg.C_em
        cand_K = unit_normalize(torch.randn(BS, B, C_em, cfg.D))
        cand_V = torch.randn(BS, B, C_em, cfg.D)
        cand_scores = torch.rand(BS, B, C_em) + 0.1

        g_em = torch.full((BS, B), 0.5)
        tau = torch.ones(BS, B)

        s_before = em.em_S.sum().item()
        em.write(cand_K, cand_V, cand_scores, g_em, tau)
        s_after = em.em_S.sum().item()

        assert s_after > s_before

    def test_budget_enforcement(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        # Write many times to fill up
        C_em = cfg.C_em
        for _ in range(20):
            cand_K = unit_normalize(torch.randn(BS, B, C_em, cfg.D))
            cand_V = torch.randn(BS, B, C_em, cfg.D)
            cand_scores = torch.rand(BS, B, C_em) + 0.5
            g_em = torch.ones(BS, B)
            tau = torch.ones(BS, B)
            em.write(cand_K, cand_V, cand_scores, g_em, tau)

        assert em.em_S.sum(dim=-1).max().item() <= cfg.budget_em + 0.01

    def test_age_tick(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_S = torch.ones(BS, B, cfg.M)  # activate all slots

        em.age_tick(10)
        assert em.em_age.min().item() == 10.0

    def test_reset_states(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = torch.randn(BS, B, cfg.M, cfg.D)
        em.em_V = torch.randn(BS, B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, B, cfg.M)
        em.em_age = torch.ones(BS, B, cfg.M) * 100

        mask = torch.tensor([True, False])
        em.reset_states(mask)

        # S, age, K, V should be zeroed for masked stream (Bug 4)
        assert em.em_S[0].sum() == 0
        assert em.em_age[0].sum() == 0
        assert em.em_K[0].abs().sum() == 0
        assert em.em_V[0].abs().sum() == 0
        # But not for unmasked
        assert em.em_S[1].sum() > 0
        assert em.em_K[1].abs().sum() > 0


class TestEMNeuromodulator:
    def test_output_shapes(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg.D, cfg)

        novelty_mean = torch.rand(BS)
        em_usage = torch.rand(BS)
        content = torch.randn(BS, cfg.D)

        g_em, tau, decay, ww = neuromod(novelty_mean, em_usage, content)
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert decay.shape == (BS,)
        assert ww.shape == (BS,)

    def test_g_bounded(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg.D, cfg)

        g_em, _, _, _ = neuromod(torch.rand(BS), torch.rand(BS))
        assert (g_em >= cfg.g_em_floor).all()
        assert (g_em <= cfg.g_em_ceil).all()

    def test_decay_bounded(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg.D, cfg)

        _, _, decay, _ = neuromod(torch.rand(BS), torch.rand(BS))
        assert (decay >= cfg.decay_em_floor).all()
        assert (decay <= cfg.decay_em_ceil).all()

    def test_ww_bounded(self):
        cfg = make_tiny_config()
        neuromod = EMNeuromodulator(cfg.D, cfg)

        _, _, _, ww = neuromod(torch.rand(BS), torch.rand(BS))
        assert (ww >= cfg.ww_em_floor).all()
        assert (ww <= cfg.ww_em_ceil).all()
