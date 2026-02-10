"""Tests for learnable EM write hyperparameters (tau, weakness_weight) and
parallel RL rollouts via forward_span."""

import pytest
import torch
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.episodic_memory import EpisodicMemory, EMNeuromodulator
from src.model.utils import soft_topk
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
        """Phase A: tau/ww are fixed defaults from config."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        nm = EMNeuromodulator(cfg)
        result = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert len(result) == 4
        _, _, tau, ww = result
        assert tau.shape == (BS,)
        assert ww.shape == (BS,)
        assert torch.allclose(tau, torch.full((BS,), cfg.tau_em))
        assert torch.allclose(ww, torch.full((BS,), cfg.weakness_weight_em))

    def test_continuous_tau_in_range(self):
        """Phase C: learned tau is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = EMNeuromodulator(cfg)
        _, _, tau, ww = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (tau >= cfg.tau_em_floor - 1e-6).all()
        assert (tau <= cfg.tau_em_ceil + 1e-6).all()

    def test_continuous_ww_in_range(self):
        """Phase C: learned ww is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        nm = EMNeuromodulator(cfg)
        _, _, tau, ww = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (ww >= cfg.ww_em_floor - 1e-6).all()
        assert (ww <= cfg.ww_em_ceil + 1e-6).all()

    def test_learned_tau_in_range(self):
        """Phase D: learned tau is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)
        _, _, tau, ww = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (tau >= cfg.tau_em_floor - 1e-6).all()
        assert (tau <= cfg.tau_em_ceil + 1e-6).all()

    def test_learned_ww_in_range(self):
        """Phase D: learned ww is within [floor, ceil]."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)
        _, _, tau, ww = nm.forward(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert (ww >= cfg.ww_em_floor - 1e-6).all()
        assert (ww <= cfg.ww_em_ceil + 1e-6).all()

    def test_tau_head_exists_phase_c(self):
        """Phase C+: tau_head and ww_head are created."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
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
        """At init, learned tau should equal the heuristic default (tau_em)."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)
        _, _, tau, _ = nm(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert torch.allclose(tau, torch.full((BS,), cfg.tau_em), atol=1e-5)

    def test_init_ww_matches_heuristic_default(self):
        """At init, learned ww should equal the heuristic default (weakness_weight_em)."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)
        _, _, _, ww = nm(torch.randn(BS), torch.randn(BS), torch.randn(BS))
        assert torch.allclose(ww, torch.full((BS,), cfg.weakness_weight_em), atol=1e-5)


# ============================================================================
# soft_topk: batched tau
# ============================================================================

class TestSoftTopkBatchedTau:
    def test_scalar_tau_unchanged(self):
        """Scalar tau still works as before."""
        scores = torch.randn(BS, 10)
        w = soft_topk(scores, k=3, tau=1.0)
        assert w.shape == (BS, 10)
        assert torch.allclose(w.sum(dim=-1), torch.ones(BS), atol=1e-5)

    def test_batched_tau_shape(self):
        """Per-stream [BS] tau produces correct output shape."""
        scores = torch.randn(BS, 10)
        tau = torch.tensor([0.5, 2.0])
        w = soft_topk(scores, k=3, tau=tau)
        assert w.shape == (BS, 10)
        assert torch.allclose(w.sum(dim=-1), torch.ones(BS), atol=1e-5)

    def test_batched_tau_affects_sharpness(self):
        """Lower tau -> sharper distribution (higher max weight)."""
        scores = torch.randn(4, 10)
        tau_sharp = torch.full((4,), 0.1)
        tau_flat = torch.full((4,), 5.0)
        w_sharp = soft_topk(scores, k=3, tau=tau_sharp)
        w_flat = soft_topk(scores, k=3, tau=tau_flat)
        # Sharper distribution should have higher max weight
        assert (w_sharp.max(dim=-1).values >= w_flat.max(dim=-1).values - 1e-5).all()

    def test_batched_tau_k_geq_N(self):
        """When k >= N, batched tau should still work."""
        scores = torch.randn(BS, 4)
        tau = torch.tensor([0.5, 2.0])
        w = soft_topk(scores, k=10, tau=tau)
        assert w.shape == (BS, 4)
        assert torch.allclose(w.sum(dim=-1), torch.ones(BS), atol=1e-5)


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
        cfg.set_phase("C")
        em = self._make_em_with_state(cfg)

        P = cfg.P
        cand_K = torch.randn(BS, P, cfg.D_em)
        cand_V = torch.randn(BS, P, cfg.D_em)
        cand_score = torch.rand(BS, P)
        write_mask = torch.ones(BS, dtype=torch.bool)
        g_em = torch.full((BS,), 0.3)
        tau = torch.tensor([0.5, 2.0])
        ww = torch.tensor([0.1, 1.5])

        # Should not raise
        em.write_at_boundary(
            cand_K, cand_V, cand_score, write_mask, g_em,
            tau=tau, weakness_weight=ww,
        )

    def test_none_tau_ww_uses_defaults(self):
        """When tau/ww are None, defaults from config are used."""
        cfg = make_tiny_config()
        cfg.set_phase("C")
        em = self._make_em_with_state(cfg)

        P = cfg.P
        cand_K = torch.randn(BS, P, cfg.D_em)
        cand_V = torch.randn(BS, P, cfg.D_em)
        cand_score = torch.rand(BS, P)
        write_mask = torch.ones(BS, dtype=torch.bool)
        g_em = torch.full((BS,), 0.3)

        # Should not raise (uses self.tau, self.weakness_weight)
        em.write_at_boundary(
            cand_K, cand_V, cand_score, write_mask, g_em,
        )

    def test_different_tau_gives_different_writes(self):
        """Different per-stream tau values produce different EM states."""
        cfg = make_tiny_config()
        cfg.set_phase("C")

        P = cfg.P
        cand_K = torch.randn(BS, P, cfg.D_em)
        cand_V = torch.randn(BS, P, cfg.D_em)
        cand_score = torch.rand(BS, P)
        write_mask = torch.ones(BS, dtype=torch.bool)
        g_em = torch.full((BS,), 0.5)

        # Run with tau=0.1 (sharp)
        em1 = self._make_em_with_state(cfg)
        em1.write_at_boundary(
            cand_K.clone(), cand_V.clone(), cand_score.clone(),
            write_mask, g_em,
            tau=torch.full((BS,), 0.1),
        )

        # Run with tau=5.0 (flat)
        em2 = self._make_em_with_state(cfg)
        em2.write_at_boundary(
            cand_K.clone(), cand_V.clone(), cand_score.clone(),
            write_mask, g_em,
            tau=torch.full((BS,), 5.0),
        )

        # States should differ
        assert not torch.allclose(em1.em_V, em2.em_V, atol=1e-6)


# ============================================================================
# Gradient flow: tau_head and ww_head
# ============================================================================

class TestTauWwGradientFlow:
    def test_tau_head_has_grad_after_rl_loss(self):
        """tau_head params get gradients from the RL loss."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        _, g_em, tau, ww = nm(surprise, usage, novelty)

        # Target differs from init value (init = default_tau = 1.0)
        target_tau = torch.full_like(tau, 3.0)
        loss = F.mse_loss(tau, target_tau)
        loss.backward()

        # bias always gets grad; weight gets grad because backbone output is nonzero
        assert nm.tau_head.bias.grad is not None
        assert nm.tau_head.bias.grad.abs().sum() > 0

    def test_ww_head_has_grad_after_rl_loss(self):
        """ww_head params get gradients from the RL loss."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        _, g_em, tau, ww = nm(surprise, usage, novelty)

        # Target differs from init value (init = default_ww = 0.5)
        target_ww = torch.full_like(ww, 1.5)
        loss = F.mse_loss(ww, target_ww)
        loss.backward()

        assert nm.ww_head.bias.grad is not None
        assert nm.ww_head.bias.grad.abs().sum() > 0

    def test_backbone_gets_grad_from_all_heads(self):
        """Backbone gets gradients from g_em, tau, and ww heads."""
        cfg = make_tiny_config()
        cfg.set_phase("D")
        nm = EMNeuromodulator(cfg)

        surprise = torch.randn(BS)
        usage = torch.randn(BS)
        novelty = torch.randn(BS)

        _, g_em, tau, ww = nm(surprise, usage, novelty)

        loss = g_em.sum() + tau.sum() + ww.sum()
        loss.backward()

        backbone_grad = nm.backbone[0].weight.grad
        assert backbone_grad is not None
        assert backbone_grad.abs().sum() > 0
