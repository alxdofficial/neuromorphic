"""Tensor shape contracts (v5) — NEVER change.

If these fail, an interface changed (real bug).
"""

import torch
import torch.nn.functional as F
import pytest
from tests.conftest import make_tiny_config, forward_one_segment

from src.model.model import NeuromorphicLM
from src.model.scan import ScanLayer, sequential_scan
from src.model.predictive_coding import GroupedLinear, GroupedLayerNorm, WithinScanPCM
from src.model.procedural_memory import ProceduralMemory
from src.model.episodic_memory import EpisodicMemory, EMNeuromodulator


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


class TestSequentialScan:
    def test_shape(self):
        a = torch.rand(BS, 8, 3, 16)
        b = torch.randn(BS, 8, 3, 16)
        h = sequential_scan(a, b)
        assert h.shape == (BS, 8, 3, 16)

    def test_with_h0(self):
        a = torch.rand(BS, 8, 3, 16)
        b = torch.randn(BS, 8, 3, 16)
        h0 = torch.randn(BS, 3, 16)
        h = sequential_scan(a, b, h0)
        assert h.shape == (BS, 8, 3, 16)

    def test_manual_loop(self):
        """sequential_scan should match manual loop."""
        torch.manual_seed(42)
        a = torch.rand(1, 4, 2, 8)
        b = torch.randn(1, 4, 2, 8)
        h = sequential_scan(a, b)

        # Manual
        h_manual = torch.zeros(1, 2, 8)
        for t in range(4):
            h_manual = a[0, t] * h_manual + b[0, t]
            assert torch.allclose(h[0, t], h_manual, atol=1e-5)


class TestScanLayer:
    def test_shape(self):
        C, D_col, expansion = 3, 8, 2
        layer = ScanLayer(C, D_col, expansion)
        x = torch.randn(BS, 8, C, D_col)
        out, h_last = layer(x)
        assert out.shape == x.shape
        assert h_last.shape == (BS, C, D_col * expansion)

    def test_with_h_prev(self):
        C, D_col, expansion = 3, 8, 2
        E = D_col * expansion
        layer = ScanLayer(C, D_col, expansion)
        x = torch.randn(BS, 8, C, D_col)
        h_prev = torch.randn(BS, C, E)
        out, h_last = layer(x, h_prev)
        assert out.shape == x.shape
        assert h_last.shape == (BS, C, E)

    def test_residual_connection(self):
        """Output should differ from input (not zero) but include residual."""
        C, D_col, expansion = 2, 4, 2
        layer = ScanLayer(C, D_col, expansion)
        x = torch.randn(BS, 4, C, D_col)
        out, _ = layer(x)
        # Should not be identical to input (scan adds something)
        # but also not wildly different (residual connection)
        diff = (out - x).abs().mean()
        assert diff > 0  # not identity


class TestWithinScanPCM:
    def test_compute_surprise_shape(self):
        C, D_col = 2, 16
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, 8, C, D_col)
        x_col = torch.randn(BS, 8, C, D_col)
        surprise, z_hat, z = pcm.compute_surprise(H, x_col)
        assert surprise.shape == (BS, 8, C, D_col)
        assert z_hat.shape == (BS, 8, C, D_col)
        assert z.shape == (BS, 8, C, D_col)

    def test_surprise_zero_at_position_zero(self):
        C, D_col = 2, 16
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, 8, C, D_col)
        x_col = torch.randn(BS, 8, C, D_col)
        surprise, _, _ = pcm.compute_surprise(H, x_col)
        assert (surprise[:, 0] == 0).all()

    def test_apply_gain_shape(self):
        C, D_col = 2, 16
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, 8, C, D_col)
        surprise = torch.randn(BS, 8, C, D_col)
        H_mod = pcm.apply_gain(H, surprise)
        assert H_mod.shape == H.shape

    def test_gain_bounded_at_init(self):
        """W_gain zero-init means gain = 1.0 at init."""
        C, D_col = 2, 16
        pcm = WithinScanPCM(C, D_col)
        H = torch.randn(BS, 8, C, D_col)
        surprise = torch.randn(BS, 8, C, D_col)
        H_mod = pcm.apply_gain(H, surprise)
        # At init, W_gain=0 so gain = 1 + 0.1*tanh(0) = 1.0
        assert torch.allclose(H_mod, H, atol=1e-5)

    def test_prediction_loss(self):
        C, D_col = 2, 16
        pcm = WithinScanPCM(C, D_col)
        z_hat = torch.randn(BS, 8, C, D_col)
        z = torch.randn(BS, 8, C, D_col)
        loss = pcm.prediction_loss(z_hat, z)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestProceduralMemory:
    def test_initialize(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        assert not pm.is_initialized()
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        assert pm.is_initialized()
        assert pm.pm_bias.shape == (BS, cfg.B, cfg.D)

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

    def test_commit(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg.B, cfg.D, cfg.decay_pm)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        delta_sum = torch.randn(BS, cfg.B, cfg.D)
        bias_before = pm.pm_bias.clone()
        pm.commit(delta_sum)
        # Bias should have changed
        assert not torch.allclose(pm.pm_bias, bias_before)


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

    def test_trail_read_all_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        # Need some active primitives for trail to work
        em.em_K = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_V = torch.randn(BS, cfg.B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, cfg.B, cfg.M)
        seed = torch.randn(BS, 8, cfg.D)
        y = em.trail_read_all(seed)
        assert y.shape == (BS, 8, cfg.B, cfg.D)

    def test_trail_read_all_empty_memory(self):
        """Trail read from empty memory should not crash."""
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        seed = torch.randn(BS, 8, cfg.D)
        y = em.trail_read_all(seed)
        assert y.shape == (BS, 8, cfg.B, cfg.D)
        assert torch.isfinite(y).all()

    def test_compute_novelty_all_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        w_cand = torch.randn(BS, 8, cfg.D)
        surprise = torch.randn(BS, 8, cfg.D)
        novelty = em.compute_novelty_all(w_cand, surprise)
        assert novelty.shape == (BS, 8, cfg.B)

    def test_compute_write_deltas_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        novelty = torch.rand(BS, 8, cfg.B)
        w_cand = torch.randn(BS, 8, cfg.D)
        deltas = em.compute_write_deltas(novelty, w_cand)
        assert deltas.shape == (BS, 8, cfg.B, cfg.D)

    def test_usage_all_shape(self):
        cfg = make_tiny_config()
        em = EpisodicMemory(cfg.B, cfg.M, cfg.D, cfg.n_trail_steps)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        u = em.usage_all()
        assert u.shape == (BS, cfg.B)


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
        # PCM should contribute to aux_loss
        assert aux_loss.item() >= 0

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

    def test_stage_layer_counts(self):
        """Model should have L_scan layers in each stage."""
        cfg = make_tiny_config(L_scan=3)
        model = NeuromorphicLM(cfg)
        assert len(model.stage1) == 3
        assert len(model.stage3) == 3
