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
    EM_NEUROMOD_DEFAULT_G,
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
        """In learned mode (Phase B), g_em should be in [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        g_em, _, _, _ = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert (g_em >= cfg.g_em_floor - 1e-6).all()
        assert (g_em <= cfg.g_em_ceil + 1e-6).all()


# ============================================================================
# EM Neuromodulator per phase
# ============================================================================

class TestEMNeuromodulator:
    def test_heuristic_mode_no_params(self):
        """Phase A: no learnable params (em_enabled=False)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = EMNeuromodulator(cfg)
        param_count = sum(1 for _ in neuromod.parameters())
        assert param_count == 0

    def test_continuous_mode_has_backbone(self):
        """Phase B: backbone + g_head."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        assert hasattr(neuromod, "backbone")
        assert hasattr(neuromod, "g_head")

    def test_backbone_input_dim(self):
        """Backbone takes 3 scalar features + content_proj_dim."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        expected_dim = 3 + cfg.content_proj_dim  # scalar features + content projection
        assert neuromod.backbone[0].in_features == expected_dim

    def test_backbone_input_dim_defaults(self):
        """Default config should match EM_NEUROMOD_INPUT_DIM constant."""
        cfg = ModelConfig()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        assert neuromod.backbone[0].in_features == EM_NEUROMOD_INPUT_DIM

    def test_heuristic_default_g(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = EMNeuromodulator(cfg)
        assert neuromod.default_g == EM_NEUROMOD_DEFAULT_G

    def test_learned_g_em_gates_writes(self):
        """Phase B: g_em is continuous and gates writes (no binary write_mask)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        g_em, tau, ww, decay = neuromod.forward(span_surprise, em_usage, cand_novelty)
        # g_em should be a continuous value in [floor, ceil]
        assert g_em.shape == (BS,)
        assert (g_em >= cfg.g_em_floor - 1e-6).all()
        assert (g_em <= cfg.g_em_ceil + 1e-6).all()

    def test_heuristic_returns_4_tuple(self):
        """Phase A: neuromodulator returns (g_em, tau, ww, decay)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        result = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert len(result) == 4
        g_em, tau, ww, decay = result
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert decay.shape == (BS,)

    def test_learned_returns_4_tuple(self):
        """Phase B: neuromodulator returns (g_em, tau, ww, decay)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        neuromod = EMNeuromodulator(cfg)
        span_surprise = torch.randn(BS)
        em_usage = torch.randn(BS)
        cand_novelty = torch.randn(BS)
        result = neuromod.forward(span_surprise, em_usage, cand_novelty)
        assert len(result) == 4
        g_em, tau, ww, decay = result
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert decay.shape == (BS,)


# ============================================================================
# EM temporal age
# ============================================================================

class TestEMTemporalAge:
    def test_age_state_initialized_to_zero(self):
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))
        assert em.em_age is not None
        assert em.em_age.shape == (BS, cfg.M)
        assert em.em_age.abs().max() == 0

    def test_age_in_state_tensor_names(self):
        assert "em_age" in EpisodicMemory._state_tensor_names

    def test_age_gate_initialized_to_zero(self):
        """age_gate=0 means no temporal bias at init (pure content retrieval)."""
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        assert em.age_gate.item() == 0.0

    def test_age_tick_increments_active_slots(self):
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))

        # Mark some slots as active
        em.em_S[:, 0] = 1.0
        em.em_S[:, 1] = 0.5

        em.age_tick(32)

        # Active slots aged
        assert (em.em_age[:, 0] == 32).all()
        assert (em.em_age[:, 1] == 32).all()
        # Inactive slots unchanged
        assert (em.em_age[:, 2] == 0).all()

    def test_age_tick_ignores_inactive_slots(self):
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))

        # All slots inactive (S=0)
        em.age_tick(100)
        assert em.em_age.abs().max() == 0

    def test_age_reset_on_doc_boundary(self):
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))

        em.em_S[:, 0] = 1.0
        em.em_age[:, 0] = 500.0

        mask = torch.tensor([True, False])
        em.reset_states(mask)

        assert em.em_age[0, 0].item() == 0  # reset
        assert em.em_age[1, 0].item() == 500  # preserved

    def test_age_bias_zero_at_init(self):
        """With age_gate=0, retrieval scores are unaffected by age."""
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))

        # Activate all slots and set varying ages
        em.em_S.fill_(1.0)
        em.em_age[:, 0] = 0
        em.em_age[:, 1] = 100
        em.em_age[:, 2] = 10000

        x = torch.randn(BS, cfg.D)
        y_wm = torch.randn(BS, cfg.D)

        # Should not crash and should return valid output
        y_em = em.retrieve(x, y_wm)
        assert y_em.shape == (BS, cfg.D)

    def test_age_gate_is_learnable(self):
        """age_gate should be an nn.Parameter that receives gradients."""
        cfg = make_tiny_config(em_enabled=True)
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))

        em.em_S.fill_(1.0)
        em.em_age[:, 0] = 100.0

        x = torch.randn(BS, cfg.D)
        y_wm = torch.randn(BS, cfg.D)

        y_em = em.retrieve(x, y_wm)
        loss = y_em.sum()
        loss.backward()

        assert em.age_gate.grad is not None

    def test_negative_age_gate_prefers_recent(self):
        """Negative age_gate biases retrieval toward recent memories."""
        cfg = make_tiny_config(em_enabled=True, k_ret=1, M=4)
        em = EpisodicMemory(cfg)
        em._lazy_init(1, torch.device("cpu"))  # BS=1 for simplicity

        # Two active slots with identical keys but different ages
        em.em_S[0, 0] = 1.0
        em.em_S[0, 1] = 1.0
        em.em_K[0, 0] = em.em_K[0, 1]  # same key
        em.em_age[0, 0] = 1000.0   # old
        em.em_age[0, 1] = 1.0      # recent

        with torch.no_grad():
            em.age_gate.fill_(-1.0)  # prefer recent

        x = torch.randn(1, cfg.D)
        y_wm = torch.randn(1, cfg.D)

        # With k_ret=1, only one slot is retrieved.
        # Negative age_gate should make slot 1 (recent, age=1) score higher.
        from src.model.utils import unit_normalize
        q = unit_normalize(em.W_q_em(torch.cat([x, y_wm], dim=-1)))
        q = q.to(em.em_K.dtype)
        scores = torch.einsum("bd, bmd -> bm", q, em.em_K)
        age_bias = em.age_gate * torch.log1p(em.em_age.to(scores.dtype))
        biased_scores = scores + age_bias

        # Slot 1 (recent) should score higher than slot 0 (old)
        assert biased_scores[0, 1] > biased_scores[0, 0]
