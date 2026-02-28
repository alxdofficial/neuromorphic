"""Tensor shape contracts (v4) — NEVER change.

If these fail, an interface changed (real bug).
"""

import torch
import torch.nn.functional as F
import pytest
from tests.conftest import make_tiny_config, forward_one_segment

from src.model.model import NeuromorphicLM
from src.model.predictive_coding import GroupedLinear, GroupedLayerNorm, CrossPassPCM
from src.model.column import CorticalColumnGroup, LateralMixer, CrossBlockMixer, FFNStack, PositionAttention
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
        C = 3
        mixer = LateralMixer(C)
        B = 2
        x = torch.randn(BS, 4, B, C, D_col)
        y = mixer(x)
        assert y.shape == x.shape

    def test_residual_identity_at_init(self):
        """mix is zero-init, so output should equal input at init."""
        D_col = 16
        C = 3
        mixer = LateralMixer(C)
        B = 2
        x = torch.randn(BS, 4, B, C, D_col)
        y = mixer(x)
        assert torch.allclose(y, x, atol=1e-5)


class TestCrossBlockMixer:
    def test_shape(self):
        D_col = 16
        B = 3
        C = 2
        mixer = CrossBlockMixer(B)
        x = torch.randn(BS, 4, B, C, D_col)
        y = mixer(x)
        assert y.shape == x.shape

    def test_residual_identity_at_init(self):
        """cross_mix is zero-init, so output should equal input at init."""
        D_col = 16
        B = 3
        C = 2
        mixer = CrossBlockMixer(B)
        x = torch.randn(BS, 4, B, C, D_col)
        y = mixer(x)
        assert torch.allclose(y, x, atol=1e-5)


class TestPositionAttention:
    def test_shape(self):
        G, D_col, D_attn = 4, 16, 8
        N_C = 8
        pa = PositionAttention(G, D_col, D_attn)
        x = torch.randn(BS, N_C, G, D_col)
        y = pa(x)
        assert y.shape == (BS, N_C, G, D_col)

    def test_residual_identity_at_init(self):
        """out_proj is zero-init, so output should equal input at init."""
        G, D_col, D_attn = 4, 16, 8
        N_C = 8
        pa = PositionAttention(G, D_col, D_attn)
        x = torch.randn(BS, N_C, G, D_col)
        y = pa(x)
        assert torch.allclose(y, x, atol=1e-5)

    def test_disabled_when_dim_zero(self):
        """position_attn_dim=0 should result in pos_attn=None on column group."""
        cfg = make_tiny_config(position_attn_dim=0)
        col = CorticalColumnGroup(cfg)
        assert col.pos_attn is None

    def test_config_auto_derives(self):
        """position_attn_dim=-1 should auto-derive to D_col // 4."""
        cfg = make_tiny_config()  # D=64, C=2 -> D_col=32
        assert cfg.position_attn_dim == 8  # 32 // 4

    def test_column_with_pos_attn(self):
        """Full column forward shape preserved with PositionAttention."""
        cfg = make_tiny_config()
        assert cfg.position_attn_dim > 0
        G = cfg.B_blocks * cfg.C
        col = CorticalColumnGroup(cfg)
        assert col.pos_attn is not None
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        x = torch.randn(BS, cfg.N_C, G, cfg.D_col)
        x_out, z, z_hat, surprise, elig_info, nov_info = col.forward(x, pm, em, None)
        assert x_out.shape == (BS, cfg.N_C, G, cfg.D_col)


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


class TestSingleLoopModel:
    def test_single_loop_forward_shape(self):
        """Full model with ffn_depth=2 (2L=4 total FFN layers) produces correct output shape."""
        cfg = make_tiny_config(ffn_depth=2)
        model = NeuromorphicLM(cfg)
        logits, aux_loss = forward_one_segment(model, BS=BS, vocab=VOCAB)
        assert logits.shape == (BS, cfg.N, VOCAB)

    def test_single_loop_with_pcm(self):
        """Single-loop model with PCM produces correct output."""
        cfg = make_tiny_config(pcm_enabled=True, ffn_depth=2)
        model = NeuromorphicLM(cfg)
        logits, aux_loss = forward_one_segment(model, BS=BS, vocab=VOCAB)
        assert logits.shape == (BS, cfg.N, VOCAB)

    def test_single_loop_multi_segment(self):
        """Single-loop model works across multiple segments."""
        cfg = make_tiny_config(ffn_depth=2)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        N = cfg.N
        for seg in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            assert logits.shape == (BS, N, VOCAB)

    def test_single_loop_gradients(self):
        """All FFN layers in ffn_pre and ffn_post get gradients."""
        cfg = make_tiny_config(ffn_depth=3)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        reset_mask = torch.zeros(BS, dtype=torch.bool)
        logits, aux_loss = model.forward_segment(input_ids, reset_mask)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1)) + aux_loss
        loss.backward()

        for stack_name in ("ffn_pre", "ffn_post"):
            stack = getattr(model.columns, stack_name)
            for i in range(3):
                up_w = stack.ups[i].weight
                down_w = stack.downs[i].weight
                assert up_w.grad is not None, f"No gradient for {stack_name}.ups[{i}]"
                assert up_w.grad.abs().sum() > 0, f"Zero gradient for {stack_name}.ups[{i}]"
                assert down_w.grad is not None, f"No gradient for {stack_name}.downs[{i}]"
                assert down_w.grad.abs().sum() > 0, f"Zero gradient for {stack_name}.downs[{i}]"


class TestFFNStack:
    def test_ffnstack_shape(self):
        """FFNStack produces correct output shape."""
        G, D_col, D_hidden, L = 4, 16, 32, 3
        stack = FFNStack(G, D_col, D_hidden, L)
        x = torch.randn(BS, 8, G, D_col)
        y = stack(x)
        assert y.shape == x.shape

    def test_ffnstack_with_gain(self):
        """FFNStack applies gain on first layer only."""
        G, D_col, D_hidden, L = 4, 16, 32, 2
        stack = FFNStack(G, D_col, D_hidden, L)
        x = torch.randn(BS, 8, G, D_col)
        gain = torch.ones(BS, 8, G, D_col) * 1.5
        y = stack(x, gain=gain)
        assert y.shape == x.shape

    def test_column_has_ffn_pre_post(self):
        """CorticalColumnGroup should have ffn_pre and ffn_post FFNStack instances."""
        cfg = make_tiny_config(ffn_depth=3)
        col = CorticalColumnGroup(cfg)
        assert isinstance(col.ffn_pre, FFNStack)
        assert isinstance(col.ffn_post, FFNStack)
        assert col.ffn_pre.L == 3
        assert col.ffn_post.L == 3
        assert len(col.ffn_pre.ups) == 3
        assert len(col.ffn_post.ups) == 3

    def test_forward_shape_depth4(self):
        """Column with depth=4 (2*4=8 total FFN layers) produces correct output shape."""
        cfg = make_tiny_config(pcm_enabled=True, ffn_depth=4)
        G = cfg.B_blocks * cfg.C
        col = CorticalColumnGroup(cfg)
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        em.initialize(BS, torch.device("cpu"), torch.float32)

        x = torch.randn(BS, cfg.N, G, cfg.D_col)
        x_out, z, z_hat, surprise, elig_info, nov_info = col.forward(x, pm, em, None)
        assert x_out.shape == (BS, cfg.N, G, cfg.D_col)


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


class TestInterleavedPartitioning:
    def test_to_cols_shape(self):
        """_to_cols produces [BS, N_C, G, D_col] from [BS, N, D]."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        x = torch.randn(BS, cfg.N, cfg.D)
        x_cols = model._to_cols(x)
        assert x_cols.shape == (BS, cfg.N_C, cfg.B_blocks * cfg.C, cfg.D_col)

    def test_from_cols_shape(self):
        """_from_cols produces [BS, N, D] from [BS, N_C, G, D_col] + skip."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        G = cfg.B_blocks * cfg.C
        x_cols = torch.randn(BS, cfg.N_C, G, cfg.D_col)
        x_input = torch.randn(BS, cfg.N, cfg.D)
        x_out = model._from_cols(x_cols, x_input)
        assert x_out.shape == (BS, cfg.N, cfg.D)

    def test_roundtrip_approx_at_init(self):
        """At init, fan_in is small so _from_cols(_to_cols(x), x) ≈ x."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        x = torch.randn(BS, cfg.N, cfg.D)
        x_cols = model._to_cols(x)
        x_rt = model._from_cols(x_cols, x)
        # fan_in has small-scale init (std=0.02), so residual is small
        residual = (x_rt - x).abs().mean()
        assert residual < 1.0  # small but not exactly zero

    def test_n_c_config(self):
        """N_C is correctly derived as N // C."""
        cfg = make_tiny_config(N=16, C=2)
        assert cfg.N_C == 8
        cfg = make_tiny_config(N=16, C=4)
        assert cfg.N_C == 4

    def test_n_mod_c_validation(self):
        """Config should reject N not divisible by C."""
        with pytest.raises(ValueError, match="divisible by C"):
            make_tiny_config(N=15, C=2)

    def test_fan_in_small_init(self):
        """fan_in weight should be small-scale, bias zero."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        assert model.fan_in.weight.abs().mean() < 0.1  # small-scale
        assert model.fan_in.weight.abs().sum() > 0  # not zero (for gradient flow)
        assert (model.fan_in.bias == 0).all()

    def test_fan_in_dimensions(self):
        """fan_in should be Linear(D_col, D)."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        assert model.fan_in.in_features == cfg.D_col
        assert model.fan_in.out_features == cfg.D


class TestReadSliced:
    def test_pm_read_sliced_shape(self):
        """PM read_sliced with 5D input [BS, N_C, B, C, D_col]."""
        cfg = make_tiny_config()
        B, C, D_col = cfg.B_blocks, cfg.C, cfg.D_col
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_K = torch.randn(BS, B, cfg.r, cfg.D)
        pm.pm_V = torch.randn(BS, B, cfg.r, cfg.D)
        pm.pm_a = torch.ones(BS, B, cfg.r)

        N_C = 4
        q = torch.randn(BS, N_C, B, C, D_col)
        y = pm.read_sliced(q)
        assert y.shape == (BS, N_C, B, C, D_col)

    def test_em_read_sliced_shape(self):
        """EM read_sliced with 5D input [BS, N_C, B, C, D_col]."""
        cfg = make_tiny_config()
        B, C, D_col = cfg.B_blocks, cfg.C, cfg.D_col
        em = EpisodicMemory(cfg.D, cfg.M, cfg)
        em.initialize(BS, torch.device("cpu"), torch.float32)
        em.em_K = torch.randn(BS, B, cfg.M, cfg.D)
        em.em_V = torch.randn(BS, B, cfg.M, cfg.D)
        em.em_S = torch.ones(BS, B, cfg.M)

        N_C = 4
        q = torch.randn(BS, N_C, B, C, D_col)
        y = em.read_sliced(q)
        assert y.shape == (BS, N_C, B, C, D_col)
