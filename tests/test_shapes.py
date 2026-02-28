"""Tensor shape contracts (v4) — NEVER change.

If these fail, an interface changed (real bug).
"""

import torch
import pytest
from tests.conftest import make_tiny_config, forward_one_segment

from src.model.model import NeuromorphicLM
from src.model.predictive_coding import GroupedLinear, GroupedLayerNorm, CrossPassPCM
from src.model.column import CorticalColumnGroup, LateralMixer
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
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        # Set some content
        pm.pm_K = torch.randn(BS, B, cfg.r, cfg.D)
        pm.pm_V = torch.randn(BS, B, cfg.r, cfg.D)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        q = torch.randn(BS, 4, B, cfg.D)  # [BS, N, B, D]
        y = pm.read(q)
        assert y.shape == q.shape

    def test_commit(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        elig_K = torch.randn(BS, B, cfg.r, cfg.D)
        elig_V = torch.randn(BS, B, cfg.r, cfg.D)
        g = torch.full((BS, B), 0.5)
        slot_logits = torch.randn(BS, B, cfg.r)
        tau = torch.ones(BS, B)

        pm.commit(elig_K, elig_V, g, slot_logits, tau)
        assert pm.pm_a is not None
        assert pm.pm_a.sum() > 0


class TestEpisodicMemory:
    def test_read_shape(self):
        """EM read with 4D input [BS, N, B, D]."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        # Set some content
        em.em_K = torch.randn(BS, B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, B, cfg.M)

        N = 4
        q = torch.randn(BS, N, B, cfg.D)
        y = em.read(q)
        assert y.shape == q.shape

    def test_novelty_score_shape(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        N = 4
        q = torch.randn(BS, N, B, cfg.D)
        surprise = torch.rand(BS, N, B)
        w_nov = torch.rand(BS, N, B)

        novelty = em.score_novelty(q, surprise, w_nov)
        assert novelty.shape == (BS, N, B)

    def test_select_top_candidates(self):
        cfg = make_tiny_config()
        B = cfg.B_blocks
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        N = 4
        q = torch.randn(BS, N, B, cfg.D)
        v = torch.randn(BS, N, B, cfg.D)
        novelty = torch.rand(BS, N, B)

        cand_K, cand_V, cand_scores = em.select_top_candidates(q, v, novelty, cfg.C_em)
        assert cand_K.shape == (BS, B, cfg.C_em, cfg.D)
        assert cand_V.shape == (BS, B, cfg.C_em, cfg.D)
        assert cand_scores.shape == (BS, B, cfg.C_em)


class TestLateralMixer:
    def test_shape(self):
        D_col = 16
        mixer = LateralMixer(D_col)
        B, C = 2, 3
        x = torch.randn(BS, 4, B, C, D_col)
        y = mixer(x)
        assert y.shape == x.shape

    def test_residual_identity_at_init(self):
        """W_out is zero-init, so output should equal input at init."""
        D_col = 16
        mixer = LateralMixer(D_col)
        B, C = 2, 3
        x = torch.randn(BS, 4, B, C, D_col)
        y = mixer(x)
        assert torch.allclose(y, x, atol=1e-5)


class TestColumnGroup:
    def test_forward_shape(self):
        cfg = make_tiny_config(pcm_enabled=True)
        G = cfg.B_blocks * cfg.C
        B = cfg.B_blocks
        col = CorticalColumnGroup(cfg)
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        x = torch.randn(BS, cfg.N, G, cfg.D_col)
        x_out, z, z_hat, surprise, elig_info, nov_info = col.forward(x, pm, em, None)

        assert x_out.shape == (BS, cfg.N, G, cfg.D_col)
        assert z.shape == (BS, cfg.N, G, cfg.D_pcm)
        assert z_hat.shape == (BS, cfg.N, G, cfg.D_pcm)
        assert surprise.shape == (BS, cfg.N, G)

        k_cand, v_cand, gate = elig_info
        assert k_cand.shape == (BS, cfg.N, B, cfg.C, cfg.D_col)
        assert gate.shape == (BS, cfg.N, B, cfg.C)

        q_nov, v_nov, w_nov, surp = nov_info
        assert q_nov.shape == (BS, cfg.N, B, cfg.C, cfg.D_col)
        assert w_nov.shape == (BS, cfg.N, B, cfg.C)


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

    def test_d_embed_decoupled_forward(self):
        """D_embed != D should produce correct output shape."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)
        logits, aux_loss = forward_one_segment(model, BS=BS, vocab=VOCAB)
        assert logits.shape == (BS, cfg.N, VOCAB)

    def test_d_embed_equal_d_no_proj(self):
        """When D_embed == D, proj_up/proj_down should be None."""
        cfg = make_tiny_config(D=64, D_embed=64)
        model = NeuromorphicLM(cfg)
        assert model.proj_up is None
        assert model.proj_down is None

    def test_d_embed_decoupled_has_proj(self):
        """When D_embed != D, proj_up/proj_down should exist."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)
        assert model.proj_up is not None
        assert model.proj_down is not None
