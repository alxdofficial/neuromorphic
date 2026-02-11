"""Per-phase neuromodulator behavior — update design_constants.py when design changes."""

import pytest
import torch

from src.model.model import NeuromorphicLM
from src.model.procedural_memory import PMNeuromodulator
from src.model.episodic_memory import EMNeuromodulator
from tests.conftest import make_tiny_config

pytestmark = pytest.mark.design

BS = 2


# ============================================================================
# Neuromodulator mode per phase
# ============================================================================

class TestNeuromodulatorModes:
    def test_pm_continuous_in_phase_a(self):
        """Phase A: PM enabled, neuromodulator uses continuous heads (no RL)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = PMNeuromodulator(cfg)
        assert cfg.pm_enabled
        assert not cfg.rl_enabled
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        _, _, _, slot_logits, p_commit = result
        assert slot_logits is not None
        assert p_commit is None

    def test_pm_continuous_in_phase_b(self):
        """Phase B: PM neuromodulator still uses continuous heads (EM added, no RL)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = PMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        _, _, _, slot_logits, p_commit = result
        assert slot_logits is not None
        assert p_commit is None

    def test_pm_learned_in_phase_c(self):
        """Phase C: PM neuromodulator uses learned gate (RL enabled)."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = PMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        _, _, _, slot_logits, p_commit = result
        assert slot_logits is not None
        assert p_commit is not None

    def test_em_heuristic_in_phase_a(self):
        """Phase A: EM neuromodulator uses heuristic."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        write_mask, g_em, tau, ww = result
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert g_em.shape == (BS,)

    def test_em_continuous_in_phase_b(self):
        """Phase B: EM neuromodulator uses continuous g_em."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        _, g_em, tau, ww = result
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert (g_em >= cfg.g_em_floor - 1e-6).all()

    def test_em_learned_in_phase_c(self):
        """Phase C: EM neuromodulator uses learned gate (RL enabled)."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        write_mask, g_em, tau, ww = result
        assert write_mask.all()
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)


# ============================================================================
# RL parameter isolation
# ============================================================================

class TestRLParameterIsolation:
    def test_rl_parameters_only_neuromodulator(self):
        """model.rl_parameters() yields only params with 'neuromodulator' in name."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)

        rl_param_names = set()
        for name, param in model.named_parameters():
            if "neuromodulator" in name:
                rl_param_names.add(id(param))

        yielded_ids = set()
        for param in model.rl_parameters():
            yielded_ids.add(id(param))

        assert yielded_ids == rl_param_names, \
            "rl_parameters() should yield exactly the neuromodulator params"
