"""Tensor shape contracts (v4) — NEVER change.

If these fail, an interface changed (real bug).
"""

import torch
import pytest
from tests.conftest import make_tiny_config, forward_one_segment

from src.model.model import NeuromorphicLM
from src.model.predictive_coding import GroupedLinear, GroupedLayerNorm, CrossPassPCM
from src.model.column import CorticalColumnGroup
from src.model.procedural_memory import ProceduralMemory
from src.model.episodic_memory import EpisodicMemory


BS = 2
VOCAB = 64


class TestGroupedLinear:
    def test_shape(self):
        C, D_in, D_out = 3, 8, 16
        layer = GroupedLinear(C, D_in, D_out)
        x = torch.randn(BS, 4, C, D_in)  # [BS, N, C, D_in]
        y = layer(x)
        assert y.shape == (BS, 4, C, D_out)

    def test_no_bias(self):
        layer = GroupedLinear(2, 4, 8, bias=False)
        assert layer.bias is None


class TestGroupedLayerNorm:
    def test_shape(self):
        C, D = 3, 8
        norm = GroupedLayerNorm(C, D)
        x = torch.randn(BS, 4, C, D)
        y = norm(x)
        assert y.shape == x.shape


class TestCrossPassPCM:
    def test_encode_shape(self):
        C, D_col, D_pcm = 2, 16, 8
        pcm = CrossPassPCM(C, D_col, D_pcm)
        x = torch.randn(BS, 4, C, D_col)
        z = pcm.encode(x)
        assert z.shape == (BS, 4, C, D_pcm)

    def test_predict_shape(self):
        C, D_col, D_pcm = 2, 16, 8
        pcm = CrossPassPCM(C, D_col, D_pcm)
        z = torch.randn(BS, 4, C, D_pcm)
        z_hat = pcm.predict(z)
        assert z_hat.shape == (BS, 4, C, D_pcm)

    def test_surprise_shape(self):
        C, D_col, D_pcm = 2, 16, 8
        pcm = CrossPassPCM(C, D_col, D_pcm)
        z = torch.randn(BS, 4, C, D_pcm)
        z_hat = torch.randn(BS, 4, C, D_pcm)
        surprise, delta = pcm.compute_surprise(z, z_hat)
        assert surprise.shape == (BS, 4, C)
        assert delta.shape == (BS, 4, C, D_pcm)

    def test_surprise_none_prev(self):
        C, D_col, D_pcm = 2, 16, 8
        pcm = CrossPassPCM(C, D_col, D_pcm)
        z = torch.randn(BS, 4, C, D_pcm)
        surprise, delta = pcm.compute_surprise(z, None)
        assert surprise.shape == (BS, 4, C)
        assert (surprise == 0).all()


class TestProceduralMemory:
    def test_read_shape(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        # Set some content
        pm.pm_K = torch.randn(BS, cfg.r, cfg.D_mem)
        pm.pm_V = torch.randn(BS, cfg.r, cfg.D_mem)
        pm.pm_a = torch.ones(BS, cfg.r)

        q = torch.randn(BS, 4, 2, cfg.D_mem)  # [BS, N, C, D_mem]
        y = pm.read(q)
        assert y.shape == q.shape

    def test_read_3d_shape(self):
        """PM read also works with 3D input [BSB, NC, D_mem]."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        BSB = BS * B
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BSB, torch.device("cpu"), torch.float32)
        pm.pm_K = torch.randn(BSB, cfg.r, cfg.D_mem)
        pm.pm_V = torch.randn(BSB, cfg.r, cfg.D_mem)
        pm.pm_a = torch.ones(BSB, cfg.r)

        NC = 4 * 2  # N*C
        q = torch.randn(BSB, NC, cfg.D_mem)
        y = pm.read(q)
        assert y.shape == (BSB, NC, cfg.D_mem)

    def test_commit(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        elig_K = torch.randn(BS, cfg.r, cfg.D_mem)
        elig_V = torch.randn(BS, cfg.r, cfg.D_mem)
        g = torch.full((BS,), 0.5)
        slot_logits = torch.randn(BS, cfg.r)
        tau = torch.ones(BS)

        pm.commit(elig_K, elig_V, g, slot_logits, tau)
        assert pm.pm_a is not None
        assert pm.pm_a.sum() > 0


class TestEpisodicMemory:
    def test_read_shape(self):
        """EM read with batched input [BSB, NC, D_mem]."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        BSB = BS * B
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BSB, torch.device("cpu"), torch.float32)
        # Set some content
        em.em_K = torch.randn(BSB, cfg.M, cfg.D_mem)
        em.em_S = torch.ones(BSB, cfg.M)

        NC = 4 * cfg.C
        q = torch.randn(BSB, NC, cfg.D_mem)  # [BSB, NC, D_mem]
        y = em.read(q)
        assert y.shape == q.shape

    def test_novelty_score_shape(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        BSB = BS * B
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BSB, torch.device("cpu"), torch.float32)

        NC = 4 * cfg.C
        q = torch.randn(BSB, NC, cfg.D_mem)
        surprise = torch.rand(BSB, NC)
        w_nov = torch.rand(BSB, NC)

        novelty = em.score_novelty(q, surprise, w_nov)
        assert novelty.shape == (BSB, NC)

    def test_select_top_candidates(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        BSB = BS * B
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        em.initialize(BSB, torch.device("cpu"), torch.float32)

        NC = 4 * cfg.C
        q = torch.randn(BSB, NC, cfg.D_mem)
        v = torch.randn(BSB, NC, cfg.D_mem)
        novelty = torch.rand(BSB, NC)

        cand_K, cand_V, cand_scores = em.select_top_candidates(q, v, novelty, cfg.C_em)
        assert cand_K.shape == (BSB, cfg.C_em, cfg.D_mem)
        assert cand_V.shape == (BSB, cfg.C_em, cfg.D_mem)
        assert cand_scores.shape == (BSB, cfg.C_em)


class TestColumnGroup:
    def test_forward_shape(self):
        cfg = make_tiny_config(pcm_enabled=True)
        G = cfg.B_blocks * cfg.C
        col = CorticalColumnGroup(cfg)
        BSB = BS * cfg.B_blocks
        pm = ProceduralMemory(cfg.D_mem, cfg.r, cfg)
        em = EpisodicMemory(cfg.D_mem, cfg.M, cfg)
        pm.initialize(BSB, torch.device("cpu"), torch.float32)
        em.initialize(BSB, torch.device("cpu"), torch.float32)

        x = torch.randn(BS, cfg.N, G, cfg.D_col)
        x_out, z, z_hat, surprise, elig_info, nov_info = col.forward(x, pm, em, None)

        assert x_out.shape == (BS, cfg.N, G, cfg.D_col)
        assert z.shape == (BS, cfg.N, G, cfg.D_pcm)
        assert z_hat.shape == (BS, cfg.N, G, cfg.D_pcm)
        assert surprise.shape == (BS, cfg.N, G)

        k_cand, v_cand, gate = elig_info
        assert k_cand.shape == (BS, cfg.N, G, cfg.D_mem)
        assert gate.shape == (BS, cfg.N, G)

        q_nov, v_nov, w_nov, surp = nov_info
        assert q_nov.shape == (BS, cfg.N, G, cfg.D_mem)
        assert w_nov.shape == (BS, cfg.N, G)


class TestFullModel:
    def test_forward_segment_shape(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        logits, aux_loss = forward_one_segment(model, BS=BS, vocab=VOCAB)

        assert logits.shape == (BS, cfg.N, VOCAB)
        assert aux_loss.shape == ()

    def test_forward_segment_with_pcm(self):
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)
        logits, aux_loss = forward_one_segment(model, BS=BS, vocab=VOCAB)

        assert logits.shape == (BS, cfg.N, VOCAB)

    def test_multi_segment(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        N = cfg.N
        for seg in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            assert logits.shape == (BS, N, VOCAB)

    def test_with_reset(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        N = cfg.N
        input_ids = torch.randint(0, VOCAB, (BS, N))
        reset_mask = torch.ones(BS, dtype=torch.bool)  # reset all
        logits, _ = model.forward_segment(input_ids, reset_mask)
        assert logits.shape == (BS, N, VOCAB)

    def test_param_count(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        count = model.param_count()
        assert count > 0
        assert isinstance(count, int)
