"""Tests for learnable EM write hyperparameters (tau, weakness_weight)."""

import pytest
import torch
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.episodic_memory import EpisodicMemory, EMNeuromodulator
from tests.conftest import make_tiny_config

pytestmark = pytest.mark.design

BS = 2


# ============================================================================
# Config: new floor/ceil fields
# ============================================================================

class TestConfigTauWwFields:
    def test_tau_em_floor_default(self):
        cfg = ModelConfig()
        assert cfg.tau_em_floor == 0.05

    def test_tau_em_ceil_default(self):
        cfg = ModelConfig()
        assert cfg.tau_em_ceil == 5.0

    def test_ww_em_floor_default(self):
        cfg = ModelConfig()
        assert cfg.ww_em_floor == 0.0

    def test_ww_em_ceil_default(self):
        cfg = ModelConfig()
        assert cfg.ww_em_ceil == 2.0


# ============================================================================
# EMNeuromodulator: tau/ww output
# ============================================================================

class TestEMNeuromodulatorTauWw:
    def test_heuristic_returns_defaults(self):
        """Phase A: tau/ww/decay are fixed defaults from config."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert len(result) == 4
        g_em, tau, ww, decay = result
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert torch.allclose(tau, torch.full((BS,), cfg.tau_em))
        assert torch.allclose(ww, torch.full((BS,), cfg.weakness_weight_em))

    def test_continuous_tau_in_range(self):
        """Phase B: learned tau is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        g_em, tau, ww, decay = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (tau >= cfg.tau_em_floor - 1e-6).all()
        assert (tau <= cfg.tau_em_ceil + 1e-6).all()

    def test_continuous_ww_in_range(self):
        """Phase B: learned ww is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        g_em, tau, ww, decay = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (ww >= cfg.ww_em_floor - 1e-6).all()
        assert (ww <= cfg.ww_em_ceil + 1e-6).all()

    def test_learned_tau_in_range_phase_c(self):
        """Phase C: learned tau is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = EMNeuromodulator(cfg)
        g_em, tau, ww, decay = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (tau >= cfg.tau_em_floor - 1e-6).all()
        assert (tau <= cfg.tau_em_ceil + 1e-6).all()

    def test_learned_ww_in_range_phase_c(self):
        """Phase C: learned ww is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = EMNeuromodulator(cfg)
        g_em, tau, ww, decay = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (ww >= cfg.ww_em_floor - 1e-6).all()
        assert (ww <= cfg.ww_em_ceil + 1e-6).all()

    def test_tau_head_exists_phase_b(self):
        """Phase B+: tau_head and ww_head are created."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        assert hasattr(nm, "tau_head")
        assert hasattr(nm, "ww_head")

    def test_no_heads_in_phase_a(self):
        """Phase A: no tau/ww heads."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = EMNeuromodulator(cfg)
        assert not hasattr(nm, "tau_head")
        assert not hasattr(nm, "ww_head")

    def test_init_tau_matches_heuristic_default(self):
        """At init, learned tau should be close to heuristic default (tau_em)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        # Use zero inputs to get pure-bias output (exact match)
        g_em, tau, _, _ = nm(torch.zeros(BS), torch.zeros(BS), torch.zeros(BS))
        assert torch.allclose(tau, torch.full((BS,), cfg.tau_em), atol=0.1)

    def test_init_ww_matches_heuristic_default(self):
        """At init, learned ww should be close to heuristic default (weakness_weight_em)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        # Use zero inputs to get pure-bias output (exact match)
        g_em, _, ww, _ = nm(torch.zeros(BS), torch.zeros(BS), torch.zeros(BS))
        assert torch.allclose(ww, torch.full((BS,), cfg.weakness_weight_em), atol=0.1)


# ============================================================================
# write_at_boundary: per-stream tau/ww
# ============================================================================

class TestWriteAtBoundaryTauWw:
    def _make_em_with_state(self, cfg):
        em = EpisodicMemory(cfg)
        em._lazy_init(BS, torch.device("cpu"))
        # Write some initial data to give non-zero strengths
        em.em_S[:] = 1.0
        return em

    def test_accepts_tensor_tau_ww(self):
        """write_at_boundary works with [BS] tau and ww tensors."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        em = self._make_em_with_state(cfg)

        P = cfg.P
        cand_K = torch.randn(BS, P, cfg.D_em)
        cand_V = torch.randn(BS, P, cfg.D_em)
        cand_score = torch.rand(BS, P)
        g_em = torch.full((BS,), 0.3)
        tau = torch.tensor([0.5, 2.0])
        ww = torch.tensor([0.1, 1.5])

        # Should not raise
        decay = torch.full((BS,), cfg.decay_em)
        em.write_at_boundary(
            cand_K, cand_V, cand_score, g_em,
            tau=tau, weakness_weight=ww, decay=decay,
        )

    def test_explicit_tau_ww_from_config_defaults(self):
        """Passing config defaults for tau/ww should work (callers always provide them)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        em = self._make_em_with_state(cfg)

        P = cfg.P
        cand_K = torch.randn(BS, P, cfg.D_em)
        cand_V = torch.randn(BS, P, cfg.D_em)
        cand_score = torch.rand(BS, P)
        g_em = torch.full((BS,), 0.3)
        tau = torch.full((BS,), cfg.tau_em)
        ww = torch.full((BS,), cfg.weakness_weight_em)
        decay = torch.full((BS,), cfg.decay_em)

        # Should not raise
        em.write_at_boundary(
            cand_K, cand_V, cand_score, g_em, tau=tau, weakness_weight=ww,
            decay=decay,
        )

    def test_different_tau_gives_different_writes(self):
        """Different per-stream tau values produce different EM states."""
        cfg = make_tiny_config()
        cfg.set_phase("B")

        P = cfg.P
        cand_K = torch.randn(BS, P, cfg.D_em)
        cand_V = torch.randn(BS, P, cfg.D_em)
        cand_score = torch.rand(BS, P)
        g_em = torch.full((BS,), 0.5)

        ww = torch.full((BS,), cfg.weakness_weight_em)
        decay = torch.full((BS,), cfg.decay_em)

        # Run with tau=0.1 (sharp)
        em1 = self._make_em_with_state(cfg)
        em1.write_at_boundary(
            cand_K.clone(), cand_V.clone(), cand_score.clone(),
            g_em,
            tau=torch.full((BS,), 0.1),
            weakness_weight=ww, decay=decay,
        )

        # Run with tau=5.0 (flat)
        em2 = self._make_em_with_state(cfg)
        em2.write_at_boundary(
            cand_K.clone(), cand_V.clone(), cand_score.clone(),
            g_em,
            tau=torch.full((BS,), 5.0),
            weakness_weight=ww, decay=decay,
        )

        # States should differ
        assert not torch.allclose(em1.em_V, em2.em_V, atol=1e-6)


# ============================================================================
# Gradient flow: tau_head and ww_head
# ============================================================================

class TestTauWwGradientFlow:
    def test_tau_head_has_grad_after_loss(self):
        """tau_head params get gradients from the main loss."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        g_em, tau, ww, decay = nm(surprise, usage, novelty)

        # Target differs from init value (init = default_tau = 1.0)
        target_tau = torch.full_like(tau, 3.0)
        loss = F.mse_loss(tau, target_tau)
        loss.backward()

        # bias always gets grad; weight gets grad because backbone output is nonzero
        assert nm.tau_head.bias.grad is not None
        assert nm.tau_head.bias.grad.abs().sum() > 0

    def test_ww_head_has_grad_after_loss(self):
        """ww_head params get gradients from the main loss."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        g_em, tau, ww, decay = nm(surprise, usage, novelty)

        # Target differs from init value (init = default_ww = 0.5)
        target_ww = torch.full_like(ww, 1.5)
        loss = F.mse_loss(ww, target_ww)
        loss.backward()

        assert nm.ww_head.bias.grad is not None
        assert nm.ww_head.bias.grad.abs().sum() > 0

    def test_backbone_gets_grad_from_all_heads(self):
        """Backbone gets gradients from g_em, tau, ww, and decay heads."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        g_em, tau, ww, decay = nm(surprise, usage, novelty)

        loss = g_em.sum() + tau.sum() + ww.sum() + decay.sum()
        loss.backward()

        backbone_grad = nm.backbone[0].weight.grad
        assert backbone_grad is not None
        assert backbone_grad.abs().sum() > 0


# ============================================================================
# Config: decay_em floor/ceil fields
# ============================================================================

class TestConfigDecayFields:
    def test_decay_em_floor_default(self):
        cfg = ModelConfig()
        assert cfg.decay_em_floor == 0.99

    def test_decay_em_ceil_default(self):
        cfg = ModelConfig()
        assert cfg.decay_em_ceil == 0.9999


# ============================================================================
# EMNeuromodulator: decay output
# ============================================================================

class TestEMNeuromodulatorDecay:
    def test_continuous_decay_in_range(self):
        """Phase B: learned decay is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        g_em, tau, ww, decay = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (decay >= cfg.decay_em_floor - 1e-6).all()
        assert (decay <= cfg.decay_em_ceil + 1e-6).all()

    def test_decay_head_exists_phase_b(self):
        """Phase B+: decay_head is created."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        assert hasattr(nm, "decay_head")

    def test_no_decay_head_in_phase_a(self):
        """Phase A: no decay_head."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = EMNeuromodulator(cfg)
        assert not hasattr(nm, "decay_head")

    def test_init_decay_matches_heuristic_default(self):
        """At init, learned decay should be close to heuristic default (decay_em)."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)
        # Use zero inputs to get pure-bias output (exact match)
        _, _, _, decay = nm(torch.zeros(BS), torch.zeros(BS), torch.zeros(BS))
        assert torch.allclose(decay, torch.full((BS,), cfg.decay_em), atol=0.01)

    def test_decay_head_has_grad_after_loss(self):
        """decay_head params get gradients from the main loss."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        g_em, tau, ww, decay = nm(surprise, usage, novelty)

        # Target differs from init value (init = default_decay = 0.999)
        target_decay = torch.full_like(decay, 0.995)
        loss = F.mse_loss(decay, target_decay)
        loss.backward()

        assert nm.decay_head.bias.grad is not None
        assert nm.decay_head.bias.grad.abs().sum() > 0
