"""Per-phase neuromodulator behavior — update design_constants.py when design changes."""

import pytest
import torch

from src.model.procedural_memory import PMNeuromodulator
from src.model.episodic_memory import EMNeuromodulator
from tests.conftest import make_tiny_config

pytestmark = pytest.mark.design

BS = 2


# ============================================================================
# Neuromodulator mode per phase
# ============================================================================

class TestNeuromodulatorModes:
    def test_pm_heuristic_in_phase_a(self):
        """Phase A: PM enabled, neuromodulator in heuristic mode (no learnable params)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = PMNeuromodulator(cfg)
        assert cfg.pm_enabled
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        p_commit, lambda_vals, g, slot_logits, tau = result
        assert p_commit.shape == (BS,)
        assert tau.shape == (BS,)

    def test_pm_continuous_in_phase_b(self):
        """Phase B: PM neuromodulator uses learned backbone + heads."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = PMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        p_commit, lambda_vals, g, slot_logits, tau = result
        assert p_commit.shape == (BS,)
        assert lambda_vals.shape == (BS,)
        assert g.shape == (BS,)
        assert tau.shape == (BS,)

    def test_pm_phase_d_raises(self):
        """Phase D was renamed to C — set_phase('D') must raise ValueError."""
        cfg = make_tiny_config()
        with pytest.raises(ValueError, match="Unknown phase"):
            cfg.set_phase("D")

    def test_em_heuristic_in_phase_a(self):
        """Phase A: EM disabled, neuromodulator uses heuristic defaults."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        g_em, tau, ww, decay = result
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert decay.shape == (BS,)

    def test_em_continuous_in_phase_b(self):
        """Phase B: EM neuromodulator uses learned continuous g_em."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        g_em, tau, ww, decay = result
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert decay.shape == (BS,)
        assert (g_em >= cfg.g_em_floor - 1e-6).all()

    def test_em_continuous_in_phase_c(self):
        """Phase C: EM neuromodulator uses learned continuous g_em (lifelong)."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        g_em, tau, ww, decay = result
        assert g_em.shape == (BS,)
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert decay.shape == (BS,)

    def test_pm_content_emb_accepted(self):
        """PM neuromodulator accepts optional content_emb kwarg."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = PMNeuromodulator(cfg)
        content_emb = torch.randn(BS, cfg.D_h)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS),
                            content_emb=content_emb)
        assert len(result) == 5

    def test_em_content_emb_accepted(self):
        """EM neuromodulator accepts optional content_emb kwarg."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        content_emb = torch.randn(BS, cfg.D_em)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS),
                            content_emb=content_emb)
        assert len(result) == 4
