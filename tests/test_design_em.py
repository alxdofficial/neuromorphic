"""Design-specific EM tests — update design_constants.py when design changes."""

import pytest
import torch
import torch.nn as nn

from src.model.model import NeuromorphicLM
from src.model.episodic_memory import EpisodicMemory, EMNeuromodulator
from src.model.config import ModelConfig
from tests.conftest import make_tiny_config, forward_and_write_em
from tests.design_constants import (
    DEFAULTS, EM_READOUT_FFN_EXPANSION, EM_NEUROMOD_INPUT_DIM,
    EM_NEUROMOD_NOVELTY_THRESHOLD, EM_NEUROMOD_DEFAULT_G,
)

pytestmark = pytest.mark.design

BS = 2


# ============================================================================
# EM capacity and retrieval
# ============================================================================

class TestEMCapacity:
    def test_default_capacity(self):
        cfg = ModelConfig()
        assert cfg.M == DEFAULTS["M"]

    def test_default_k_ret(self):
        cfg = ModelConfig()
        assert cfg.k_ret == DEFAULTS["k_ret"]

    def test_default_C_em(self):
        cfg = ModelConfig()
        assert cfg.C_em == DEFAULTS["C_em"]

    def test_default_k_write(self):
        cfg = ModelConfig()
        assert cfg.k_write == DEFAULTS["k_write"]


# ============================================================================
# EM readout FFN
# ============================================================================

class TestEMReadoutFFN:
    def test_readout_ffn_exists(self):
        cfg = make_tiny_config(em_readout_ffn=True)
        em = EpisodicMemory(cfg)
        assert em.readout_ffn is not None
        assert em.readout_norm is not None

    def test_readout_ffn_disabled(self):
        cfg = make_tiny_config(em_readout_ffn=False)
        em = EpisodicMemory(cfg)
        assert em.readout_ffn is None

    def test_readout_ffn_structure(self):
        """Linear -> GELU -> Linear, expansion=4x."""
        cfg = make_tiny_config(em_readout_ffn=True)
        em = EpisodicMemory(cfg)
        ffn = em.readout_ffn
        assert isinstance(ffn[0], nn.Linear)
        assert isinstance(ffn[1], nn.GELU)
        assert isinstance(ffn[2], nn.Linear)
        assert ffn[0].out_features == cfg.D_em * EM_READOUT_FFN_EXPANSION

    def test_default_em_readout_ffn_enabled(self):
        cfg = ModelConfig()
        assert cfg.em_readout_ffn == DEFAULTS["em_readout_ffn"]


# ============================================================================
# EM g_em clamping
# ============================================================================

class TestEMGemClamping:
    def test_g_em_floor_default(self):
        cfg = ModelConfig()
        assert cfg.g_em_floor == DEFAULTS["g_em_floor"]

    def test_g_em_ceil_default(self):
        cfg = ModelConfig()
        assert cfg.g_em_ceil == DEFAULTS["g_em_ceil"]

    def test_continuous_g_em_in_range(self):
        """In continuous mode, g_em should be in [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        _, g_em, _, _ = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert (g_em >= cfg.g_em_floor - 1e-6).all()
        assert (g_em <= cfg.g_em_ceil + 1e-6).all()

    def test_learned_g_em_in_range(self):
        """In learned mode, g_em should be in [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        _, g_em, _, _ = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert (g_em >= cfg.g_em_floor - 1e-6).all()
        assert (g_em <= cfg.g_em_ceil + 1e-6).all()


# ============================================================================
# EM Neuromodulator per phase
# ============================================================================

class TestEMNeuromodulator:
    def test_heuristic_mode_no_params(self):
        """Phase A-B: no learnable params."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = EMNeuromodulator(cfg)
        param_count = sum(1 for _ in neuromod.parameters())
        assert param_count == 0

    def test_continuous_mode_has_backbone(self):
        """Phase C: backbone + g_head."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        assert hasattr(neuromod, "backbone")
        assert hasattr(neuromod, "g_head")

    def test_backbone_input_dim(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        assert neuromod.backbone[0].in_features == EM_NEUROMOD_INPUT_DIM

    def test_heuristic_novelty_threshold(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = EMNeuromodulator(cfg)
        assert neuromod.novelty_threshold == EM_NEUROMOD_NOVELTY_THRESHOLD

    def test_heuristic_default_g(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = EMNeuromodulator(cfg)
        assert neuromod.default_g == EM_NEUROMOD_DEFAULT_G

    def test_learned_write_mask_all_true(self):
        """Phase D: write_mask is always True."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        write_mask, _, _, _ = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert write_mask.all()
