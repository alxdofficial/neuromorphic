"""Mathematical invariants that should NEVER change.

If these fail, the code is wrong — not the test expectations.
"""

import pytest
import torch
import torch.nn.functional as F

from src.model.utils import unit_normalize, budget_enforce
from src.model.model import NeuromorphicLM
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
# PM bias finite after operations
# ============================================================================

class TestPMBiasInvariants:
    def test_pm_bias_finite_after_init(self):
        """After init, pm_bias is zeros."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        assert torch.isfinite(model.pm.pm_bias).all()
        assert model.pm.pm_bias.abs().sum() == 0

    def test_pm_bias_finite_after_commits(self):
        """After forward_segment (which includes PM commit), pm_bias should be finite."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)
        assert torch.isfinite(model.pm.pm_bias).all()


# ============================================================================
# EM key normalization
# ============================================================================

class TestEMNormalization:
    def test_em_keys_finite_after_init(self):
        """After init, em_K is zeros — check finite."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        assert torch.isfinite(model.em.em_K).all()

    def test_em_keys_finite_after_write(self):
        """After forward_segment (which includes EM write), em_K should be finite."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)
        results = forward_k_segments(model, K=3, BS=BS)
        assert torch.isfinite(model.em.em_K).all()


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

        assert torch.isfinite(model.pm.pm_bias).all()
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
# PM commit: bias update verification
# ============================================================================

class TestPMCommitEquations:
    """Verify PM commit update formula: bias = (bias + delta) * decay."""

    def test_commit_equation(self):
        """pm_bias = (pm_bias + delta_sum) * decay_pm."""
        from src.model.procedural_memory import ProceduralMemory
        pm = ProceduralMemory(B=1, D=4, decay_pm=0.9)
        pm.initialize(1, torch.device("cpu"), torch.float32)

        pm.pm_bias = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        delta_sum = torch.tensor([[[0.5, -0.5, 0.0, 1.0]]])  # [BS=1, B=1, D=4]

        pm.commit(delta_sum)

        expected = (torch.tensor([1.0, 2.0, 3.0, 4.0]) +
                    torch.tensor([0.5, -0.5, 0.0, 1.0])) * 0.9
        assert torch.allclose(pm.pm_bias[0, 0], expected, atol=1e-5)

    def test_zero_delta_only_decays(self):
        """With zero delta, commit only decays bias."""
        from src.model.procedural_memory import ProceduralMemory
        pm = ProceduralMemory(B=1, D=4, decay_pm=0.9)
        pm.initialize(1, torch.device("cpu"), torch.float32)

        pm.pm_bias = torch.tensor([[[2.0, 2.0, 2.0, 2.0]]])
        delta_sum = torch.zeros(1, 1, 4)  # [BS=1, B=1, D=4]

        pm.commit(delta_sum)
        expected = torch.full((4,), 2.0 * 0.9)
        assert torch.allclose(pm.pm_bias[0, 0], expected, atol=1e-5)
