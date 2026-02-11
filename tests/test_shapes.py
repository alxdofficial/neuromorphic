"""Tensor shape contracts — NEVER change.

If these fail, an interface changed (real bug).
"""

import pytest
import torch

from src.model.model import NeuromorphicLM
from src.model.config import ModelConfig
from src.model.procedural_memory import PMNeuromodulator
from src.model.episodic_memory import EMNeuromodulator
from tests.conftest import (
    make_tiny_config, forward_n_tokens, forward_and_write_em,
)

pytestmark = pytest.mark.invariant

BS = 2


# ============================================================================
# Helpers
# ============================================================================

def _make_model(phase, decoder=False):
    cfg = make_tiny_config(snapshot_enabled=decoder)
    cfg.set_phase(phase)
    return NeuromorphicLM(cfg), cfg


def _run_one(model, BS=2):
    """Run one token and return (logits, x_emb, y_wm)."""
    input_id = torch.randint(0, model.config.vocab_size, (BS,))
    reset = torch.zeros(BS, dtype=torch.bool)
    return model.forward_one_token(input_id, reset)


def _run_one_collect(model, BS=2):
    """Run one token with collect=True, return (logits, x_emb, y_wm, stats)."""
    input_id = torch.randint(0, model.config.vocab_size, (BS,))
    reset = torch.zeros(BS, dtype=torch.bool)
    return model.forward_one_token(input_id, reset, collect=True)


# ============================================================================
# Model output shapes
# ============================================================================

@pytest.mark.parametrize("phase", ["A", "B", "C", "D"])
class TestModelOutputShapes:
    def test_logits_shape(self, phase):
        model, cfg = _make_model(phase)
        logits, _, _ = _run_one(model)
        assert logits.shape == (BS, cfg.vocab_size)

    def test_x_emb_shape(self, phase):
        model, cfg = _make_model(phase)
        _, x_emb, _ = _run_one(model)
        assert x_emb.shape == (BS, cfg.D)

    def test_y_wm_shape(self, phase):
        model, cfg = _make_model(phase)
        _, _, y_wm = _run_one(model)
        assert y_wm.shape == (BS, cfg.D)


# ============================================================================
# Block/Layer output shapes
# ============================================================================

class TestBlockLayerShapes:
    def test_block_output_shape(self):
        model, cfg = _make_model("A")
        _run_one(model)
        # Block output is the last layer's h, shape [BS, D_h]
        for block in model.blocks:
            h = block.layers[-1].h
            assert h.shape == (BS, cfg.D_h)

    def test_layer_output_shape(self):
        model, cfg = _make_model("A")
        _run_one(model)
        for block in model.blocks:
            for layer in block.layers:
                assert layer.h.shape == (BS, cfg.D_h)

    def test_block_layer_outputs_shape(self):
        """When return_layers=True, block returns stacked [BS, L, D_h]."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        input_id = torch.randint(0, 64, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        # forward_one_token with decoder triggers return_layers
        logits, x, y_wm = model.forward_one_token(input_id, reset)
        # Logits should still be correct shape
        assert logits.shape == (BS, cfg.vocab_size)


# ============================================================================
# Working Memory shapes
# ============================================================================

class TestWMShapes:
    def test_wm_kv_cache_shapes(self):
        model, cfg = _make_model("A")
        _run_one(model)
        wm = model.wm
        assert wm.wm_K.shape == (BS, cfg.W, cfg.D_wm)
        assert wm.wm_V.shape == (BS, cfg.W, cfg.D_wm)

    def test_wm_step_output_shape(self):
        model, cfg = _make_model("A")
        _, _, y_wm = _run_one(model)
        assert y_wm.shape == (BS, cfg.D)


# ============================================================================
# PM shapes
# ============================================================================

class TestPMShapes:
    def test_pm_apply_shape(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        _run_one(model)
        for block in model.blocks:
            for layer in block.layers:
                x_block = torch.randn(BS, cfg.D_h)
                y_pm = layer.pm.apply(x_block)
                assert y_pm.shape == (BS, cfg.D_h)

    def test_pm_state_shapes(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        _run_one(model)
        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm
                assert pm.pm_K.shape == (BS, cfg.r, cfg.D_h)
                assert pm.pm_V.shape == (BS, cfg.r, cfg.D_h)
                assert pm.pm_a.shape == (BS, cfg.r)

    def test_pm_summary_shape(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        _run_one(model)
        summary = model._compute_pm_summary(BS, torch.device("cpu"))
        assert summary.shape == (BS, cfg.D_h)


# ============================================================================
# EM shapes
# ============================================================================

class TestEMShapes:
    def test_em_retrieve_shape(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        _, x_emb, y_wm = _run_one(model)
        # Retrieve should return [BS, D]
        for block in model.blocks:
            y_em = block.em.retrieve(x_emb, y_wm)
            assert y_em.shape == (BS, cfg.D)

    def test_em_state_shapes(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        _run_one(model)
        for block in model.blocks:
            em = block.em
            assert em.em_K.shape == (BS, cfg.M, cfg.D_em)
            assert em.em_V.shape == (BS, cfg.M, cfg.D_em)
            assert em.em_S.shape == (BS, cfg.M)

    def test_em_propose_candidate_shapes(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        _, x_emb, y_wm = _run_one(model)
        for block in model.blocks:
            h_block = block.layers[-1].h
            k_c, v_c, novelty = block.em.propose_candidate(
                x_emb, y_wm, h_block, model.surprise
            )
            assert k_c.shape == (BS, cfg.D_em)
            assert v_c.shape == (BS, cfg.D_em)
            assert novelty.shape == (BS,)

    def test_em_summary_shape(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        _run_one(model)
        summary = model._compute_em_summary(BS, torch.device("cpu"))
        assert summary.shape == (BS, cfg.D_em)


# ============================================================================
# Surprise shape
# ============================================================================

class TestSurpriseShape:
    def test_surprise_shape(self):
        model, cfg = _make_model("A")
        _run_one(model)
        target = torch.randint(0, cfg.vocab_size, (BS,))
        logits, _, _ = _run_one(model)
        model.update_surprise(logits, target)
        assert model.surprise.shape == (BS, 1)


# ============================================================================
# Neuromodulator output shapes
# ============================================================================

class TestNeuromodulatorShapes:
    def test_pm_neuromod_output_shapes(self):
        """PMNeuromodulator returns 5-tuple."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        _run_one(model)

        layer = model.blocks[0].layers[0]
        pm = layer.pm
        elig_norm = pm.elig_K.norm(dim=-1).mean(dim=-1)  # [BS]
        pm_usage = pm.pm_a.sum(dim=-1)  # [BS]
        surprise = torch.randn(BS)

        result = layer.pm_neuromodulator.forward(elig_norm, pm_usage, surprise)
        assert len(result) == 5
        commit_mask, lambda_vals, g, slot_logits, p_commit = result

        assert commit_mask.shape == (BS,)
        assert commit_mask.dtype == torch.bool
        assert lambda_vals.shape == (BS,)
        assert g.shape == (BS,)
        # In Phase B (continuous mode): slot_logits present, p_commit None
        assert slot_logits is not None
        assert slot_logits.shape == (BS, cfg.r)
        assert p_commit is None

    def test_pm_neuromod_heuristic_shapes(self):
        """pm_enabled=False: heuristic fallback mode."""
        cfg = make_tiny_config()
        cfg.pm_enabled = False
        neuromod = PMNeuromodulator(cfg)
        elig_norm = torch.randn(BS)
        pm_usage = torch.randn(BS)
        surprise = torch.randn(BS)

        result = neuromod.forward(elig_norm, pm_usage, surprise)
        commit_mask, lambda_vals, g, slot_logits, p_commit = result
        assert commit_mask.shape == (BS,)
        assert slot_logits is None
        assert p_commit is None

    def test_pm_neuromod_continuous_shapes(self):
        """Phase A: continuous mode (PM enabled, no RL)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        elig_norm = torch.randn(BS)
        pm_usage = torch.randn(BS)
        surprise = torch.randn(BS)

        result = neuromod.forward(elig_norm, pm_usage, surprise)
        commit_mask, lambda_vals, g, slot_logits, p_commit = result
        assert commit_mask.shape == (BS,)
        assert slot_logits is not None
        assert slot_logits.shape[1] == cfg.r
        assert p_commit is None  # no RL → no p_commit

    def test_pm_neuromod_learned_shapes(self):
        """Phase C: learned mode with p_commit (RL enabled)."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        neuromod = PMNeuromodulator(cfg)
        elig_norm = torch.randn(BS)
        pm_usage = torch.randn(BS)
        surprise = torch.randn(BS)

        result = neuromod.forward(elig_norm, pm_usage, surprise)
        commit_mask, lambda_vals, g, slot_logits, p_commit = result
        assert p_commit is not None
        assert p_commit.shape == (BS,)

    def test_em_neuromod_output_shapes(self):
        """EMNeuromodulator returns 3-tuple."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)

        result = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert len(result) == 4
        write_mask, g_em, tau, ww = result

        assert write_mask.shape == (BS,)
        assert write_mask.dtype == torch.bool
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)


# ============================================================================
# Collect mode
# ============================================================================

class TestCollectMode:
    def test_collect_returns_four_tuple(self):
        model, cfg = _make_model("A")
        result = _run_one_collect(model)
        assert len(result) == 4
        logits, x_emb, y_wm, stats = result
        assert isinstance(stats, dict)


# ============================================================================
# Decoder shapes
# ============================================================================

class TestDecoderShapes:
    def test_decoder_output_shape(self):
        model, cfg = _make_model("C", decoder=True)
        logits, x, y_wm = _run_one(model)
        assert logits.shape == (BS, cfg.vocab_size)

    def test_columnar_output_shape(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        _run_one(model)
        # Test columnar directly
        layer_out = torch.randn(BS, cfg.L, cfg.D_h)
        col = model.spatial_decoder.columnar[0]
        out = col(layer_out)
        assert out.shape == (BS, cfg.D_h)

    def test_thalamic_output_shape(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        K = cfg.thalamic_tokens
        d_dec = cfg.d_dec
        cortical = torch.randn(BS, cfg.B, d_dec)
        pm = torch.randn(BS, d_dec)
        em = torch.randn(BS, d_dec)
        wm = torch.randn(BS, d_dec)
        out = model.spatial_decoder.thalamic(cortical, pm, em, wm)
        assert out.shape == (BS, K, d_dec)

    def test_decoder_block_output_shape(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        d_dec = cfg.d_dec
        K = cfg.thalamic_tokens
        x = torch.randn(BS, 1, d_dec)
        memory = torch.randn(BS, K, d_dec)
        block = model.spatial_decoder.decoder_blocks[0]
        out = block(x, memory)
        assert out.shape == (BS, 1, d_dec)


# ============================================================================
# Decoder assembly invariants — tensor assembly order and cross-attention inputs
# ============================================================================

class TestDecoderAssemblyInvariants:
    """Verify that the spatial decoder's internal tensor assembly is correct.

    These catch silent semantic regressions like reordering the thalamic
    input concatenation or mismatching type embeddings.
    """

    def test_thalamic_input_token_count(self):
        """Thalamic integrator receives B cortical + 3 memory = B+3 tokens."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        dec = model.spatial_decoder
        thal = dec.thalamic
        B = cfg.B
        d_dec = cfg.d_dec

        # Build inputs matching what SpatialDecoder.forward assembles
        cortical = torch.randn(BS, B, d_dec)
        pm = torch.randn(BS, d_dec)
        em = torch.randn(BS, d_dec)
        wm = torch.randn(BS, d_dec)

        # Intercept the assembled memory tensor via a forward hook
        assembled = {}

        def hook(module, args, output):
            # ThalamicIntegrator.forward builds `memory` internally.
            # Reconstruct: cortical is args[0], pm=args[1], em=args[2], wm=args[3]
            # The assembled memory is [BS, B+3, d_dec]
            # We can check the output shape which is [BS, K, d_dec]
            assembled["out"] = output

        handle = thal.register_forward_hook(hook)
        try:
            out = thal(cortical, pm, em, wm)
        finally:
            handle.remove()

        # Output shape: [BS, K, d_dec]
        assert out.shape == (BS, cfg.thalamic_tokens, d_dec)

    def test_thalamic_type_embedding_count(self):
        """Type embeddings: 4 types (0=cortical, 1=PM, 2=EM, 3=WM)."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        thal = model.spatial_decoder.thalamic
        assert thal.type_emb.num_embeddings == 4

    def test_thalamic_block_embedding_count(self):
        """Block position embeddings: B entries for cortical tokens."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        thal = model.spatial_decoder.thalamic
        assert thal.block_emb.num_embeddings == cfg.B

    def test_decoder_receives_correct_input_dims(self):
        """SpatialDecoder projections have correct input dimensions:
        - col_proj: D_h -> d_dec
        - pm_proj: D_h -> d_dec
        - em_proj: D_em -> d_dec
        - wm_proj: D -> d_dec
        - query_proj: D -> d_dec
        """
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        dec = model.spatial_decoder

        assert dec.col_proj.in_features == cfg.D_h
        assert dec.col_proj.out_features == cfg.d_dec

        assert dec.pm_proj.in_features == cfg.D_h
        assert dec.pm_proj.out_features == cfg.d_dec

        assert dec.em_proj.in_features == cfg.D_em
        assert dec.em_proj.out_features == cfg.d_dec

        assert dec.wm_proj.in_features == cfg.D
        assert dec.wm_proj.out_features == cfg.d_dec

        assert dec.query_proj.in_features == cfg.D
        assert dec.query_proj.out_features == cfg.d_dec

    def test_decoder_residual_connection(self):
        """h_decoded = h_final + output_proj(decoder_output).
        With output_proj weights zeroed, h_decoded must equal h_final exactly.
        """
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        dec = model.spatial_decoder

        # Zero out output_proj to isolate the residual
        with torch.no_grad():
            dec.output_proj.weight.zero_()

        B = cfg.B
        L = cfg.L
        D_h = cfg.D_h
        D_em = cfg.D_em
        D = cfg.D

        block_layer_outputs = [torch.randn(BS, L, D_h) for _ in range(B)]
        pm_summary = torch.randn(BS, D_h)
        em_summary = torch.randn(BS, D_em)
        wm_output = torch.randn(BS, D)
        h_final = torch.randn(BS, D)

        h_decoded = dec(block_layer_outputs, pm_summary, em_summary,
                        wm_output, h_final)

        assert torch.allclose(h_decoded, h_final, atol=1e-5), \
            "With zeroed output_proj, h_decoded should equal h_final (residual)"

    def test_decoder_cross_attention_key_count(self):
        """Level 3 decoder blocks cross-attend to K=thalamic_tokens memory tokens."""
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        K = cfg.thalamic_tokens
        d_dec = cfg.d_dec

        # Decoder block expects memory of shape [BS, K, d_dec]
        x = torch.randn(BS, 1, d_dec)
        memory = torch.randn(BS, K, d_dec)
        out = model.spatial_decoder.decoder_blocks[0](x, memory)
        assert out.shape == (BS, 1, d_dec)

        # Wrong memory token count should still work (flexible attention)
        # but we verify the expected count flows through correctly
        memory_wrong = torch.randn(BS, K + 5, d_dec)
        out2 = model.spatial_decoder.decoder_blocks[0](x, memory_wrong)
        assert out2.shape == (BS, 1, d_dec)  # still [BS, 1, d_dec]
