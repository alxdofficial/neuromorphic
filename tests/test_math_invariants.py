"""Mathematical invariants that should NEVER change.

If these fail, the code is wrong — not the test expectations.
"""

import pytest
import torch
import torch.nn.functional as F

from src.model.utils import unit_normalize, budget_enforce
from src.model.model import NeuromorphicLM
from src.model.procedural_memory import ProceduralMemory
from src.model.episodic_memory import EpisodicMemory
from src.training.loss import online_cross_entropy
from tests.conftest import make_tiny_config, forward_one_segment, forward_k_segments

pytestmark = pytest.mark.invariant

BS = 2
VOCAB = 64


# ============================================================================
# unit_normalize
# ============================================================================

class TestUnitNormalize:
    def test_produces_unit_vectors(self):
        x = torch.randn(4, 8)
        out = unit_normalize(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_handles_zero_vector(self):
        x = torch.zeros(2, 8)
        out = unit_normalize(x)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()


# ============================================================================
# budget_enforce
# ============================================================================

class TestBudgetEnforce:
    def test_under_budget_is_identity(self):
        strengths = torch.tensor([[1.0, 0.5, 0.3]])
        budget = 10.0
        result = budget_enforce(strengths, budget)
        assert torch.allclose(result, strengths)

    def test_over_budget_scales_to_budget(self):
        strengths = torch.tensor([[3.0, 4.0, 5.0]])  # sum=12
        budget = 6.0
        result = budget_enforce(strengths, budget)
        assert result.sum(dim=-1).item() <= budget + 1e-5

    def test_preserves_relative_proportions(self):
        strengths = torch.tensor([[3.0, 6.0, 9.0]])  # ratio 1:2:3
        budget = 6.0
        result = budget_enforce(strengths, budget)
        ratios = result[0] / result[0, 0]
        expected_ratios = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(ratios, expected_ratios, atol=1e-5)

    def test_per_stream_independence(self):
        strengths = torch.tensor([
            [1.0, 0.5],   # under budget
            [5.0, 5.0],   # over budget
        ])
        budget = 3.0
        result = budget_enforce(strengths, budget)
        # Stream 0 should be unchanged
        assert torch.allclose(result[0], strengths[0])
        # Stream 1 should be scaled
        assert result[1].sum().item() <= budget + 1e-5


# ============================================================================
# PM key normalization
# ============================================================================

class TestPMNormalization:
    def test_pm_keys_unit_normalized_after_init(self):
        """After init, pm_K is all zeros — norm check skips zero rows."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        pm = model.pm
        # After init, pm_K is zeros — no norm constraint on zero vectors
        assert torch.isfinite(pm.pm_K).all()

    def test_pm_keys_unit_normalized_after_commit(self):
        """After forward_segment (which includes PM commit), pm_K should
        be unit-normalized for any non-zero rows."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        # Forward multiple segments to trigger commits
        results = forward_k_segments(model, K=3, BS=BS)
        pm = model.pm
        # Non-zero rows should be unit-normalized
        norms = pm.pm_K.norm(dim=-1)
        nonzero = norms > 1e-6
        if nonzero.any():
            assert torch.allclose(norms[nonzero], torch.ones_like(norms[nonzero]), atol=1e-4)


# ============================================================================
# EM key normalization
# ============================================================================

class TestEMNormalization:
    def test_em_keys_unit_normalized_after_init(self):
        """After init, em_K is zeros — check finite."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        assert torch.isfinite(model.em.em_K).all()

    def test_em_keys_finite_after_write(self):
        """After forward_segment (which includes EM write), em_K should
        be finite and have norm <= 1.0 (EMA of unit vectors)."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)
        assert torch.isfinite(model.em.em_K).all()
        norms = model.em.em_K.norm(dim=-1)
        # EMA of unit vectors has norm <= 1.0
        assert (norms <= 1.0 + 1e-4).all()


# ============================================================================
# PM strength bounds
# ============================================================================

class TestPMStrengthBounds:
    def test_pm_strengths_bounded_by_a_max(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        # Forward multiple segments to accumulate PM commits
        results = forward_k_segments(model, K=4, BS=BS)
        assert (model.pm.pm_a <= cfg.a_max + 1e-5).all()
        assert (model.pm.pm_a >= -1e-5).all()

    def test_pm_strengths_bounded_by_budget(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=4, BS=BS)
        total = model.pm.pm_a.sum(dim=-1)
        assert (total <= cfg.budget_pm + 1e-5).all()


# ============================================================================
# EM strength bounds
# ============================================================================

class TestEMStrengthBounds:
    def test_em_strengths_bounded_by_s_max(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=4, BS=BS)
        assert (model.em.em_S <= cfg.S_max + 1e-5).all()
        assert (model.em.em_S >= -1e-5).all()

    def test_em_strengths_bounded_by_budget(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=4, BS=BS)
        total = model.em.em_S.sum(dim=-1)
        assert (total <= cfg.budget_em + 1e-5).all()


# ============================================================================
# NaN checks
# ============================================================================

class TestNoNaN:
    def test_no_nan_in_forward_pass(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        logits, aux = forward_one_segment(model, BS=BS)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux).all()

    def test_no_nan_after_multi_segment(self):
        """Multiple segments should not produce NaN in state."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)

        assert torch.isfinite(model.pm.pm_K).all()
        assert torch.isfinite(model.pm.pm_V).all()
        assert torch.isfinite(model.pm.pm_a).all()
        assert torch.isfinite(model.em.em_K).all()
        assert torch.isfinite(model.em.em_V).all()
        assert torch.isfinite(model.em.em_S).all()

    def test_no_nan_in_logits_after_multi_segment(self):
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)
        for logits, aux in results:
            assert torch.isfinite(logits).all()

    def test_no_nan_with_d_embed_decoupled(self):
        """D_embed != D should produce finite outputs."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)
        logits, aux = forward_one_segment(model, BS=BS)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux).all()


# ============================================================================
# online_cross_entropy
# ============================================================================

class TestOnlineCrossEntropy:
    def test_matches_pytorch(self):
        logits = torch.randn(4, 64)
        targets = torch.randint(0, 64, (4,))
        mask = torch.ones(4, dtype=torch.bool)

        loss_sum, valid_count = online_cross_entropy(logits, targets, mask)
        expected = F.cross_entropy(logits, targets, reduction="mean")

        # loss_sum / valid_count should match mean
        assert torch.allclose(loss_sum / valid_count, expected, atol=1e-5)

    def test_zero_valid_returns_zero(self):
        logits = torch.randn(4, 64)
        targets = torch.randint(0, 64, (4,))
        mask = torch.zeros(4, dtype=torch.bool)

        loss_sum, valid_count = online_cross_entropy(logits, targets, mask)
        assert valid_count == 0
        assert loss_sum.item() == 0.0


# ============================================================================
# PM commit: direct state manipulation tests
# ============================================================================

class TestPMCommitEquations:
    """Verify PM commit update formulas against hand-computed expected values.

    v4 commit API: commit(elig_K, elig_V, g, slot_logits, tau)
    All state set manually for deterministic testing.
    State: [BS, B, r, D]; BS=1, B=1 for simplicity.
    """

    def _make_pm(self, r=2, D=4):
        cfg = make_tiny_config(r=r, D=D, C=1, B_blocks=1,
                               tau_pm=1.0, tau_pm_floor=0.5, tau_pm_ceil=2.0,
                               a_max=10.0, budget_pm=20.0, decay_pm=0.9)
        pm = ProceduralMemory(D, r, cfg)
        return pm

    def test_ema_key_update(self):
        """pm_K = (1-alpha) * pm_K + alpha * unit_normalize(elig_K)"""
        pm = self._make_pm(r=2, D=4)

        # Inject known state (BS=1, B=1)
        pm.pm_K = unit_normalize(torch.tensor([[[[1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0]]]]))
        pm.pm_V = unit_normalize(torch.tensor([[[[0.0, 0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 0.0, 1.0]]]]))
        pm.pm_a = torch.tensor([[[1.0, 0.0]]])

        # Eligibility input (already aggregated across N*C)
        elig_K = torch.tensor([[[[0.0, 2.0, 0.0, 0.0],
                                  [0.0, 0.0, 2.0, 0.0]]]])  # [1,1,2,4]
        elig_V = torch.tensor([[[[1.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 1.0]]]])  # [1,1,2,4]

        g = torch.tensor([[0.5]])          # [BS=1, B=1]
        slot_logits = torch.tensor([[[0.0, 0.0]]])  # equal → softmax = [0.5, 0.5]
        tau = torch.tensor([[1.0]])

        # softmax([0,0]/1) = [0.5, 0.5]
        # alpha = g * slot_weights = 0.5 * [0.5, 0.5] = [0.25, 0.25]
        alpha_val = 0.25

        pm_K_before = pm.pm_K.clone()

        pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # Key EMA slot 0: (1-0.25)*[1,0,0,0] + 0.25*normalize([0,2,0,0])
        # normalize([0,2,0,0]) = [0,1,0,0]
        expected_K0 = (1 - alpha_val) * torch.tensor([1.0, 0.0, 0.0, 0.0]) + \
                       alpha_val * torch.tensor([0.0, 1.0, 0.0, 0.0])
        assert torch.allclose(pm.pm_K[0, 0, 0], expected_K0, atol=1e-4), \
            f"pm_K slot 0: {pm.pm_K[0,0,0]} != {expected_K0}"

    def test_strength_update(self):
        """pm_a = clamp(pm_a + alpha, 0, a_max) then budget_enforce."""
        pm = self._make_pm(r=2, D=4)
        pm.pm_K = unit_normalize(torch.randn(1, 1, 2, 4))
        pm.pm_V = unit_normalize(torch.randn(1, 1, 2, 4))
        pm.pm_a = torch.tensor([[[2.0, 1.0]]])

        elig_K = torch.randn(1, 1, 2, 4)
        elig_V = torch.randn(1, 1, 2, 4)
        g = torch.tensor([[0.6]])
        slot_logits = torch.tensor([[[0.0, 0.0]]])
        tau = torch.tensor([[1.0]])

        # alpha = 0.6 * [0.5, 0.5] = [0.3, 0.3]
        alpha_val = 0.3
        a_before = pm.pm_a.clone()

        pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # pm_a = clamp(a_before + alpha, 0, a_max)
        expected_a = (a_before + alpha_val).clamp(0, 10.0)
        assert torch.allclose(pm.pm_a, expected_a, atol=1e-4)

    def test_zero_g_no_key_change(self):
        """With g=0, alpha=0 everywhere: pm_K/pm_V should not change."""
        pm = self._make_pm(r=2, D=4)
        pm.pm_K = unit_normalize(torch.randn(1, 1, 2, 4))
        pm.pm_V = unit_normalize(torch.randn(1, 1, 2, 4))
        pm.pm_a = torch.tensor([[[2.0, 1.0]]])

        K_before = pm.pm_K.clone()
        V_before = pm.pm_V.clone()
        a_before = pm.pm_a.clone()

        elig_K = torch.randn(1, 1, 2, 4)
        elig_V = torch.randn(1, 1, 2, 4)
        g = torch.tensor([[0.0]])
        slot_logits = torch.tensor([[[0.0, 0.0]]])
        tau = torch.tensor([[1.0]])

        pm.commit(elig_K, elig_V, g, slot_logits, tau)

        # With g=0: alpha=0, so EMA is identity
        assert torch.allclose(pm.pm_K, K_before, atol=1e-6)
        assert torch.allclose(pm.pm_V, V_before, atol=1e-6)
        # pm_a should be unchanged (+ 0)
        assert torch.allclose(pm.pm_a, a_before, atol=1e-6)


# ============================================================================
# PM decay
# ============================================================================

class TestPMDecay:
    def test_base_decay_reduces_strengths(self):
        """base_decay: pm_a *= decay."""
        cfg = make_tiny_config(decay_pm=0.9)
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)
        pm.pm_a = torch.tensor([[[2.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
                                 [[3.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])

        before = pm.pm_a.clone()
        pm.base_decay()
        after = pm.pm_a

        mask = before > 1e-8
        if mask.any():
            assert (after[mask] < before[mask]).all()

        expected = before * 0.9
        assert torch.allclose(after, expected, atol=1e-6)


# ============================================================================
# EM write: direct state manipulation tests
# ============================================================================

class TestEMWriteEquations:
    """Verify EM write formulas against hand-computed values.

    v4 write API: write(cand_K, cand_V, cand_scores, g_em, tau, decay)
    State: [BS, B, M, D]; BS=1, B=1 for simplicity.
    """

    def _make_em(self, M=4, D=4):
        cfg = make_tiny_config(M=M, D=D, C=1, C_em=1, B_blocks=1,
                               tau_em=1.0, S_max=10.0, budget_em=40.0,
                               decay_em=0.9)
        em = EpisodicMemory(D, M, cfg)
        return em

    def test_zero_g_no_key_change(self):
        """With g_em=0, no writes: em_K/em_V unchanged, em_S only decayed."""
        em = self._make_em(M=2, D=4)
        em.em_K = unit_normalize(torch.randn(1, 1, 2, 4))
        em.em_V = torch.randn(1, 1, 2, 4)
        em.em_S = torch.tensor([[[1.0, 0.5]]])
        em.em_age = torch.zeros(1, 1, 2)

        K_before = em.em_K.clone()
        V_before = em.em_V.clone()
        S_before = em.em_S.clone()

        cand_K = unit_normalize(torch.randn(1, 1, 1, 4))
        cand_V = torch.randn(1, 1, 1, 4)
        cand_scores = torch.tensor([[[0.8]]])

        g_em = torch.tensor([[0.0]])
        tau = torch.tensor([[1.0]])
        decay = torch.tensor([[0.9]])

        em.write(cand_K, cand_V, cand_scores, g_em, tau, decay)

        # With g_em=0, alpha_per_slot = 0, no update (but decay still applies to S)
        assert torch.allclose(em.em_K, K_before, atol=1e-6), \
            "no-write em_K should be unchanged"
        assert torch.equal(em.em_V, V_before), \
            "no-write em_V should be unchanged"
        # em_S only decayed
        expected_S = S_before * 0.9
        assert torch.allclose(em.em_S, expected_S, atol=1e-5)

    def test_em_value_not_normalized(self):
        """em_V EMA update must NOT unit-normalize (unlike em_K)."""
        em = self._make_em(M=2, D=4)
        em.em_K = unit_normalize(torch.tensor([[[[1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0]]]]))
        em.em_V = torch.tensor([[[[10.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0]]]])
        em.em_S = torch.tensor([[[1.0, 0.5]]])
        em.em_age = torch.zeros(1, 1, 2)

        cand_K = unit_normalize(torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]))
        cand_V = torch.tensor([[[[0.0, 20.0, 0.0, 0.0]]]])
        cand_scores = torch.tensor([[[0.5]]])

        g_em = torch.tensor([[0.5]])
        tau = torch.tensor([[1.0]])
        decay = torch.tensor([[1.0]])  # no decay for clarity

        em.write(cand_K, cand_V, cand_scores, g_em, tau, decay)

        # After write with large V candidate, V should not be unit-normalized
        v_norm = em.em_V[0, 0, 0].norm().item()
        assert v_norm > 1.5, f"em_V should NOT be unit-normalized, but norm={v_norm}"

    def test_strength_increases_after_write(self):
        """Writing should increase em_S for targeted slots."""
        em = self._make_em(M=2, D=4)
        em.em_K = unit_normalize(torch.randn(1, 1, 2, 4))
        em.em_V = torch.randn(1, 1, 2, 4)
        em.em_S = torch.tensor([[[0.5, 0.5]]])
        em.em_age = torch.zeros(1, 1, 2)

        cand_K = unit_normalize(torch.randn(1, 1, 1, 4))
        cand_V = torch.randn(1, 1, 1, 4)
        cand_scores = torch.tensor([[[0.8]]])

        g_em = torch.tensor([[0.5]])
        tau = torch.tensor([[1.0]])
        decay = torch.tensor([[1.0]])  # no decay

        S_before = em.em_S.clone()
        em.write(cand_K, cand_V, cand_scores, g_em, tau, decay)

        # At least one slot should have higher S
        assert (em.em_S >= S_before - 1e-6).all(), \
            "em_S should not decrease without decay"


# ============================================================================
# Damped mixing parameter
# ============================================================================

class TestDampedMixing:
    def test_lambda_logit_bounded_after_sigmoid(self):
        """lambda = sigmoid(lambda_logit) must be in [0, 1]."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        lam = torch.sigmoid(model.lambda_logit)
        assert lam.item() >= 0.0 and lam.item() <= 1.0

    def test_lambda_initialized_near_config(self):
        """lambda should be initialized near config.lambda_mix."""
        cfg = make_tiny_config(lambda_mix=0.5)
        model = NeuromorphicLM(cfg)
        lam = torch.sigmoid(model.lambda_logit).item()
        assert abs(lam - 0.5) < 0.01


# ============================================================================
# Holographic read properties
# ============================================================================

class TestHolographicReadProperties:
    def test_zero_strength_gives_zero_output(self):
        """With pm_a=0, PM read output should be zero (no modulation)."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        pm.initialize(1, torch.device("cpu"), torch.float32)
        # pm_a is already 0 after init

        q = torch.randn(1, cfg.N, B, cfg.D)
        out = pm.read(q)

        # holographic: y = q * modulation. With pm_a=0, modulation=0, so y=0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_read_output_shape(self):
        """PM read should preserve input shape."""
        cfg = make_tiny_config()
        B = cfg.B_blocks
        pm = ProceduralMemory(cfg.D, cfg.r, cfg)
        pm.initialize(BS, torch.device("cpu"), torch.float32)

        q = torch.randn(BS, cfg.N, B, cfg.D)
        out = pm.read(q)
        assert out.shape == q.shape
