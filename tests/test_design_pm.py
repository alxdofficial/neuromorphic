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
    PM_NEUROMOD_DEFAULT_G,
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
        """pm_enabled=False: no learnable params (heuristic fallback)."""
        cfg = make_tiny_config()
        cfg.pm_enabled = False
        neuromod = PMNeuromodulator(cfg)
        param_count = sum(1 for _ in neuromod.parameters())
        assert param_count == 0

    def test_learned_mode_has_all_heads(self):
        """Phase A: backbone + all heads (gate, lambda, g, slot, tau)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        assert hasattr(neuromod, "backbone")
        assert hasattr(neuromod, "gate_head")
        assert hasattr(neuromod, "lambda_head")
        assert hasattr(neuromod, "g_head")
        assert hasattr(neuromod, "slot_head")
        assert hasattr(neuromod, "tau_head")
        assert hasattr(neuromod, "content_proj")

    def test_backbone_input_dim_default(self):
        """Backbone takes 3 scalar features + content_proj_dim (default config)."""
        cfg = ModelConfig()
        neuromod = PMNeuromodulator(cfg)
        assert neuromod.backbone[0].in_features == PM_NEUROMOD_INPUT_DIM

    def test_backbone_input_dim_tiny(self):
        """Backbone input dim = 3 scalars + content_proj_dim (tiny config)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        expected = 3 + cfg.content_proj_dim
        assert neuromod.backbone[0].in_features == expected

    def test_heuristic_default_g(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        assert neuromod.default_g == PM_NEUROMOD_DEFAULT_G

    def test_forward_returns_5_tuple(self):
        """Learned forward returns (p_commit, lambda_vals, g, slot_logits, tau)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        elig_norm = torch.rand(BS)
        pm_usage = torch.rand(BS)
        span_surprise = torch.rand(BS)
        result = neuromod(elig_norm, pm_usage, span_surprise)
        assert len(result) == 5
        p_commit, lambda_vals, g, slot_logits, tau = result
        assert p_commit.shape == (BS,)
        assert lambda_vals.shape == (BS,)
        assert g.shape == (BS,)
        assert slot_logits.shape == (BS, cfg.r)
        assert tau.shape == (BS,)
        # p_commit is continuous [0, 1]
        assert (p_commit >= 0).all() and (p_commit <= 1).all()

    def test_heuristic_forward_returns_5_tuple(self):
        """Heuristic forward returns (p_commit, lambda_vals, g, None, tau)."""
        cfg = make_tiny_config()
        cfg.pm_enabled = False
        neuromod = PMNeuromodulator(cfg)
        elig_norm = torch.rand(BS)
        result = neuromod(elig_norm, torch.rand(BS), torch.rand(BS))
        assert len(result) == 5
        p_commit, lambda_vals, g, slot_logits, tau = result
        assert p_commit.shape == (BS,)
        assert slot_logits is None

    def test_content_emb_accepted(self):
        """Learned mode accepts optional content_emb kwarg."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = PMNeuromodulator(cfg)
        elig_norm = torch.rand(BS)
        pm_usage = torch.rand(BS)
        span_surprise = torch.rand(BS)
        content_emb = torch.randn(BS, cfg.D_h)
        result = neuromod(elig_norm, pm_usage, span_surprise, content_emb=content_emb)
        assert len(result) == 5
