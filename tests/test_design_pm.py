"""Design-specific PM tests — update design_constants.py when design changes."""

import pytest
import torch
import torch.nn as nn

from src.model.model import NeuromorphicLM
from src.model.procedural_memory import ProceduralMemory, PMNeuromodulator
from src.model.config import ModelConfig
from tests.conftest import make_tiny_config, forward_n_tokens
from tests.design_constants import (
    DEFAULTS, PM_READOUT_FFN_EXPANSION, PM_NEUROMOD_INPUT_DIM,
    PM_NEUROMOD_THRESHOLD, PM_NEUROMOD_DEFAULT_G,
)

pytestmark = pytest.mark.design

BS = 2


# ============================================================================
# PM slot counts
# ============================================================================

class TestPMSlots:
    def test_default_slot_count(self):
        cfg = ModelConfig()
        assert cfg.r == DEFAULTS["r"]

    def test_tiny_slot_count(self):
        cfg = make_tiny_config()
        pm = ProceduralMemory(cfg)
        assert pm.r == cfg.r

    def test_commit_top_k_default(self):
        cfg = ModelConfig()
        assert cfg.commit_top_k == DEFAULTS["commit_top_k"]


# ============================================================================
# PM readout FFN
# ============================================================================

class TestPMReadoutFFN:
    def test_readout_ffn_exists(self):
        cfg = make_tiny_config(pm_readout_ffn=True)
        pm = ProceduralMemory(cfg)
        assert pm.readout_ffn is not None
        assert pm.readout_norm is not None

    def test_readout_ffn_disabled(self):
        cfg = make_tiny_config(pm_readout_ffn=False)
        pm = ProceduralMemory(cfg)
        assert pm.readout_ffn is None

    def test_readout_ffn_structure(self):
        """Linear -> GELU -> Linear, expansion=4x."""
        cfg = make_tiny_config(pm_readout_ffn=True)
        pm = ProceduralMemory(cfg)
        ffn = pm.readout_ffn
        assert isinstance(ffn[0], nn.Linear)
        assert isinstance(ffn[1], nn.GELU)
        assert isinstance(ffn[2], nn.Linear)
        # Expansion factor
        assert ffn[0].out_features == cfg.D_h * PM_READOUT_FFN_EXPANSION

    def test_default_pm_readout_ffn_enabled(self):
        cfg = ModelConfig()
        assert cfg.pm_readout_ffn == DEFAULTS["pm_readout_ffn"]


# ============================================================================
# Eligibility decay rate
# ============================================================================

class TestEligibilityConfig:
    def test_default_rho(self):
        cfg = ModelConfig()
        assert cfg.rho == DEFAULTS["rho"]

    def test_default_decay_pm(self):
        cfg = ModelConfig()
        assert cfg.decay_pm == DEFAULTS["decay_pm"]


# ============================================================================
# PM Neuromodulator architecture per phase
# ============================================================================

class TestPMNeuromodulator:
    def test_heuristic_mode_no_params(self):
        """Phase A: no learnable params."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        param_count = sum(1 for _ in neuromod.parameters())
        assert param_count == 0

    def test_continuous_mode_has_backbone(self):
        """Phase B: backbone + continuous heads, no gate_head."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = PMNeuromodulator(cfg)
        assert hasattr(neuromod, "backbone")
        assert hasattr(neuromod, "lambda_head")
        assert hasattr(neuromod, "g_head")
        assert hasattr(neuromod, "slot_head")
        assert not hasattr(neuromod, "gate_head")

    def test_learned_mode_has_gate_head(self):
        """Phase D: full MLP with gate_head."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        neuromod = PMNeuromodulator(cfg)
        assert hasattr(neuromod, "gate_head")

    def test_backbone_input_dim(self):
        """Backbone takes 3 features."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = PMNeuromodulator(cfg)
        assert neuromod.backbone[0].in_features == PM_NEUROMOD_INPUT_DIM

    def test_heuristic_commit_threshold(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        assert neuromod.threshold == PM_NEUROMOD_THRESHOLD

    def test_heuristic_default_g(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        assert neuromod.default_g == PM_NEUROMOD_DEFAULT_G
