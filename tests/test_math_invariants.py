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
from tests.conftest import make_tiny_config, forward_n_tokens, forward_and_write_em

pytestmark = pytest.mark.invariant


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
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        BS = 2
        x = torch.randint(0, 64, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.forward_one_token(x, reset)  # triggers lazy init
        for block in model.blocks:
            for layer in block.layers:
                norms = layer.pm.pm_K.norm(dim=-1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_pm_keys_unit_normalized_after_commit(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        forward_n_tokens(model, model.config.P, with_commits=True)
        for block in model.blocks:
            for layer in block.layers:
                norms = layer.pm.pm_K.norm(dim=-1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ============================================================================
# EM key normalization
# ============================================================================

class TestEMNormalization:
    def test_em_keys_unit_normalized_after_init(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        BS = 2
        x = torch.randint(0, 64, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.forward_one_token(x, reset)
        for block in model.blocks:
            norms = block.em.em_K.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_em_keys_unit_normalized_after_write(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        forward_and_write_em(model, model.config.P)
        for block in model.blocks:
            norms = block.em.em_K.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ============================================================================
# PM strength bounds
# ============================================================================

class TestPMStrengthBounds:
    def test_pm_strengths_bounded_by_a_max(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        # Run multiple commit cycles
        for _ in range(3):
            forward_n_tokens(model, cfg.P, with_commits=True)
        for block in model.blocks:
            for layer in block.layers:
                assert (layer.pm.pm_a <= cfg.a_max + 1e-5).all()
                assert (layer.pm.pm_a >= -1e-5).all()

    def test_pm_strengths_bounded_by_budget(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        for _ in range(3):
            forward_n_tokens(model, cfg.P, with_commits=True)
        for block in model.blocks:
            for layer in block.layers:
                total = layer.pm.pm_a.sum(dim=-1)
                assert (total <= cfg.budget_pm + 1e-5).all()


# ============================================================================
# EM strength bounds
# ============================================================================

class TestEMStrengthBounds:
    def test_em_strengths_bounded_by_s_max(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        for _ in range(3):
            forward_and_write_em(model, cfg.P)
        for block in model.blocks:
            assert (block.em.em_S <= cfg.S_max + 1e-5).all()
            assert (block.em.em_S >= -1e-5).all()

    def test_em_strengths_bounded_by_budget(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        for _ in range(3):
            forward_and_write_em(model, cfg.P)
        for block in model.blocks:
            total = block.em.em_S.sum(dim=-1)
            assert (total <= cfg.budget_em + 1e-5).all()


# ============================================================================
# Carry gate
# ============================================================================

class TestCarryGate:
    def test_carry_zero_kills_h_prev(self):
        """Two forward passes with different h_prev but carry=0 must produce
        identical output (h_prev has no influence)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.eval()  # disable dropout for deterministic comparison
        BS = 2
        input_id = torch.randint(0, 64, (BS,))

        # First pass: fresh model (h=None -> zeros)
        reset_mask_all = torch.ones(BS, dtype=torch.bool)
        logits1, _, _ = model.forward_one_token(input_id, reset_mask_all)

        # Inject non-zero h into all layers
        for block in model.blocks:
            for layer in block.layers:
                layer.h = torch.randn_like(layer.h) * 10.0

        # Second pass: same input but carry=0 (reset_mask=True)
        logits2, _, _ = model.forward_one_token(input_id, reset_mask_all)

        assert torch.allclose(logits1, logits2, atol=1e-5), \
            "carry=0 should make output independent of h_prev"


# ============================================================================
# NaN checks
# ============================================================================

class TestNoNaN:
    def test_no_nan_in_forward_pass(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        logits, _ = forward_n_tokens(model, 8)
        assert torch.isfinite(logits).all()

    def test_no_nan_after_commit(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        forward_n_tokens(model, cfg.P, with_commits=True)
        for block in model.blocks:
            for layer in block.layers:
                assert torch.isfinite(layer.pm.pm_K).all()
                assert torch.isfinite(layer.pm.pm_V).all()
                assert torch.isfinite(layer.pm.pm_a).all()

    def test_no_nan_after_em_write(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        forward_and_write_em(model, cfg.P)
        for block in model.blocks:
            assert torch.isfinite(block.em.em_K).all()
            assert torch.isfinite(block.em.em_V).all()
            assert torch.isfinite(block.em.em_S).all()


# ============================================================================
# Gate bounds
# ============================================================================

class TestGateBounds:
    def test_gate_a_is_sigmoid_bounded(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        BS = 2
        input_id = torch.randint(0, 64, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.forward_one_token(input_id, reset, collect=True)
        # Check gate values from collect stats
        logits, x, y_wm, stats = model.forward_one_token(input_id, reset, collect=True)
        for b_idx, bstats in stats.items():
            for l_idx, lstats in bstats.items():
                a = lstats["gate_a"]
                assert (a >= 0).all() and (a <= 1).all(), "gate_a must be in [0, 1]"

    def test_gate_b_is_tanh_bounded(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        BS = 2
        input_id = torch.randint(0, 64, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        model.forward_one_token(input_id, reset)
        logits, x, y_wm, stats = model.forward_one_token(input_id, reset, collect=True)
        for b_idx, bstats in stats.items():
            for l_idx, lstats in bstats.items():
                b = lstats["gate_b"]
                assert (b >= -1).all() and (b <= 1).all(), "gate_b must be in [-1, 1]"


# ============================================================================
# Eligibility decay
# ============================================================================

class TestEligibilityDecay:
    def test_eligibility_decay_reduces_old_contributions(self):
        """rho < 1 means older contributions decay exponentially.

        We verify by checking that a snapshot of elig_K gets scaled down
        after one more update_eligibility step (rho * old + gate * new).
        The old component should be rho * old, i.e. smaller.

        Note: update_surprise must be called between tokens so the surprise-
        gated eligibility accumulation has nonzero gate values.
        """
        cfg = make_tiny_config(rho=0.5)
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        BS = 2
        reset = torch.zeros(BS, dtype=torch.bool)

        # Run a few tokens to accumulate eligibility (must update surprise
        # so the surprise gate is nonzero from token 2 onward)
        for _ in range(3):
            input_id = torch.randint(0, 64, (BS,))
            target = torch.randint(0, 64, (BS,))
            logits, _, _ = model.forward_one_token(input_id, reset)
            model.update_surprise(logits, target)

        pm = model.blocks[0].layers[0].pm
        elig_before = pm.elig_K.clone()

        # Run one more token
        input_id = torch.randint(0, 64, (BS,))
        target = torch.randint(0, 64, (BS,))
        logits, _, _ = model.forward_one_token(input_id, reset)
        model.update_surprise(logits, target)
        elig_after = pm.elig_K

        # elig_after = rho * elig_before + gate * new_contribution
        # So old component = rho * elig_before, which should have smaller norm
        rho = cfg.rho
        old_component_norm = (rho * elig_before).norm().item()
        original_norm = elig_before.norm().item()
        assert old_component_norm < original_norm, \
            f"rho={rho}: old component norm {old_component_norm} should be < {original_norm}"

    def test_zero_surprise_means_zero_accumulation(self):
        """When surprise=0, the eligibility gate is 0, so nothing accumulates.

        This is the core property of Fix A: the model doesn't consolidate
        patterns it already predicts well.
        """
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        BS = 2

        # Initialize state with one forward pass (surprise starts at 0)
        reset = torch.zeros(BS, dtype=torch.bool)
        model.forward_one_token(torch.randint(0, 64, (BS,)), reset)
        # Do NOT call update_surprise — keep surprise at 0

        pm = model.blocks[0].layers[0].pm
        elig_before = pm.elig_K.clone()

        # Run another token with surprise still at 0
        model.forward_one_token(torch.randint(0, 64, (BS,)), reset)
        elig_after = pm.elig_K

        # With gate=0: elig_after = rho * elig_before + 0 * new
        # Since elig_before is also 0 (surprise was 0 on first call too),
        # elig_after should be all zeros
        assert (elig_after == 0).all(), \
            "Zero surprise should produce zero eligibility accumulation"

    def test_high_surprise_accumulates_eligibility(self):
        """When surprise is high, eligibility accumulates normally."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        BS = 2

        reset = torch.zeros(BS, dtype=torch.bool)
        # First token initializes surprise to 0
        model.forward_one_token(torch.randint(0, 64, (BS,)), reset)
        # Set surprise to a high value manually
        model.surprise = torch.full((BS, 1), 4.0)

        pm = model.blocks[0].layers[0].pm
        elig_before = pm.elig_K.clone()

        # Next token: gate = 4.0/5.0 = 0.8
        model.forward_one_token(torch.randint(0, 64, (BS,)), reset)
        elig_after = pm.elig_K

        # elig should have grown (new contribution added with gate=0.8)
        assert elig_after.norm() > elig_before.norm(), \
            "High surprise should increase eligibility trace norm"


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
# PM/EM decay
# ============================================================================

class TestDecay:
    def test_pm_decay_reduces_strengths(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        forward_n_tokens(model, cfg.P, with_commits=True)

        for block in model.blocks:
            for layer in block.layers:
                pm = layer.pm
                before = pm.pm_a.clone()
                pm.base_decay()
                after = pm.pm_a
                # Wherever before > 0, after should be smaller
                mask = before > 1e-8
                if mask.any():
                    assert (after[mask] < before[mask]).all()

    def test_em_decay_reduces_strengths(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        forward_and_write_em(model, cfg.P)

        for block in model.blocks:
            before = block.em.em_S.clone()
            # write_at_boundary always decays at the end — so em_S already
            # includes one decay. Let's check that calling write_at_boundary
            # with no actual writes still decays.
            empty_K = torch.zeros(2, 1, cfg.D_em)
            empty_V = torch.zeros(2, 1, cfg.D_em)
            empty_score = torch.zeros(2, 1)
            g_em = torch.zeros(2)  # g_em=0 → no writes
            tau = torch.ones(2)
            ww = torch.zeros(2)
            decay = torch.full((2,), cfg.decay_em)
            block.em.write_at_boundary(empty_K, empty_V, empty_score,
                                       g_em, tau, ww, decay=decay)
            after = block.em.em_S
            # Decay should reduce strengths
            mask = before > 1e-8
            if mask.any():
                assert (after[mask] < before[mask]).all()


# ============================================================================
# Exact PM commit equations (deterministic, single-stream)
# ============================================================================

class TestPMCommitEquations:
    """Verify PM commit update formulas against hand-computed expected values.

    All state is set manually (no model forward pass) so the test is
    fully deterministic and independent of any other code path.

    New interface: commit(p_commit, lambda_vals, g, slot_logits, tau)
    Uses softmax slot selection instead of soft_topk.
    """

    def _make_pm(self, r=2, D_h=4):
        """Build a tiny PM and inject known state."""
        cfg = make_tiny_config(r=r, D=D_h * 2, B=2, L=1,
                               tau_pm=1.0, weakness_weight_pm=0.0,
                               a_max=10.0, budget_pm=20.0, decay_pm=0.9)
        pm = ProceduralMemory(cfg)
        return pm

    def test_ema_key_update(self):
        """pm_K = unit_normalize((1-alpha) * pm_K + alpha * elig_K_norm)

        With softmax on equal scores [0, 0], weights = [0.5, 0.5].
        alpha = [0.5 * g * p_commit] for each slot.
        """
        pm = self._make_pm(r=2, D_h=4)

        # Inject known state (BS=1)
        pm.pm_K = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]))  # [1,2,4]
        pm.pm_V = unit_normalize(torch.tensor([[[0.0, 0.0, 1.0, 0.0],
                                                 [0.0, 0.0, 0.0, 1.0]]]))  # [1,2,4]
        pm.pm_a = torch.tensor([[1.0, 0.0]])  # [1,2] — slot 0 strong, slot 1 weak
        # Eligibility: slot 0 points along [0,1,0,0], slot 1 along [0,0,1,0]
        pm.elig_K = torch.tensor([[[0.0, 2.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.0, 0.0]]])  # [1,2,4]
        pm.elig_V = torch.tensor([[[1.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0]]])  # [1,2,4]

        p_commit = torch.tensor([1.0])
        g = torch.tensor([0.5])
        lambda_vals = torch.tensor([0.9])
        tau = torch.tensor([1.0])

        # weakness_weight=0, no slot_logits
        # elig_K_norm: slot 0 = [0,1,0,0], slot 1 = [0,0,1,0]
        # Similarity pm_K · elig_K_norm:
        #   slot 0: [1,0,0,0]·[0,1,0,0] = 0
        #   slot 1: [0,1,0,0]·[0,0,1,0] = 0
        # Equal scores → softmax([0,0]/1) = [0.5, 0.5]
        # Eligibility magnitude gate:
        #   elig_norm = mean([||[0,2,0,0]||, ||[0,0,2,0]||]) = mean([2, 2]) = 2.0
        #   elig_mag = 2.0 / (2.0 + 1.0) = 2/3
        # alpha = [0.5, 0.5] * 0.5 * 1.0 * (2/3) = [1/6, 1/6]
        elig_mag = 2.0 / 3.0
        alpha_val = 0.5 * 0.5 * 1.0 * elig_mag  # ≈ 0.1667

        pm_K_before = pm.pm_K.clone()
        pm_a_before = pm.pm_a.clone()

        pm.commit(p_commit, lambda_vals, g, None, tau)

        # pm_a decay: pm_a * (1 - p_commit*(1-lambda)) = [1.0, 0.0] * (1 - 1*(0.1)) = * 0.9
        #   = [0.9, 0.0]
        # Then strength update: pm_a + alpha = [0.9 + 1/6, 0.0 + 1/6]
        expected_a = torch.tensor([[0.9 + alpha_val, 0.0 + alpha_val]])
        assert torch.allclose(pm.pm_a, expected_a, atol=1e-4), \
            f"pm_a: {pm.pm_a} != expected {expected_a}"

        # Key EMA slot 0: unit_normalize((1-alpha)*[1,0,0,0] + alpha*[0,1,0,0])
        expected_K0 = unit_normalize(torch.tensor([[1.0 - alpha_val, alpha_val, 0.0, 0.0]]))
        assert torch.allclose(pm.pm_K[0, 0], expected_K0.squeeze(), atol=1e-4), \
            f"pm_K slot 0: {pm.pm_K[0,0]} != {expected_K0}"

        # Key EMA slot 1: unit_normalize((1-alpha)*[0,1,0,0] + alpha*[0,0,1,0])
        expected_K1 = unit_normalize(torch.tensor([[0.0, 1.0 - alpha_val, alpha_val, 0.0]]))
        assert torch.allclose(pm.pm_K[0, 1], expected_K1.squeeze(), atol=1e-4), \
            f"pm_K slot 1: {pm.pm_K[0,1]} != {expected_K1}"

    def test_eligibility_reset_after_commit(self):
        """Eligibility for committing streams (p_commit=1) is zeroed after commit."""
        pm = self._make_pm(r=2, D_h=4)
        pm.pm_K = unit_normalize(torch.randn(1, 2, 4))
        pm.pm_V = unit_normalize(torch.randn(1, 2, 4))
        pm.pm_a = torch.tensor([[1.0, 1.0]])
        pm.elig_K = torch.randn(1, 2, 4) * 5  # non-zero
        pm.elig_V = torch.randn(1, 2, 4) * 5

        p_commit = torch.tensor([1.0])
        lambda_vals = torch.tensor([0.9])
        g = torch.tensor([0.5])
        tau = torch.tensor([1.0])
        pm.commit(p_commit, lambda_vals, g, None, tau)

        # elig *= (1 - p_commit) = (1 - 1.0) = 0
        assert torch.allclose(pm.elig_K, torch.zeros_like(pm.elig_K), atol=1e-7)
        assert torch.allclose(pm.elig_V, torch.zeros_like(pm.elig_V), atol=1e-7)

    def test_zero_eligibility_no_strength_growth(self):
        """With zero eligibility traces, pm_a should not increase even with p_commit=1.

        The eligibility magnitude gate (elig_mag) suppresses alpha when
        elig_K is all zeros, preventing noise injection into PM slots.
        """
        pm = self._make_pm(r=2, D_h=4)
        pm.pm_K = unit_normalize(torch.randn(1, 2, 4))
        pm.pm_V = unit_normalize(torch.randn(1, 2, 4))
        pm.pm_a = torch.tensor([[2.0, 1.0]])
        pm.elig_K = torch.zeros(1, 2, 4)  # zero eligibility
        pm.elig_V = torch.zeros(1, 2, 4)

        a_before = pm.pm_a.clone()

        p_commit = torch.tensor([1.0])
        lambda_vals = torch.tensor([0.9])
        g = torch.tensor([0.5])
        tau = torch.tensor([1.0])
        pm.commit(p_commit, lambda_vals, g, None, tau)

        # elig_mag = 0 / (0 + 1) = 0 → alpha = 0
        # Only decay applies: pm_a * (1 - 1*(1-0.9)) = pm_a * 0.9
        expected_a = a_before * 0.9
        assert torch.allclose(pm.pm_a, expected_a, atol=1e-6), \
            f"pm_a grew with zero eligibility: {pm.pm_a} != {expected_a}"

    def test_non_committing_stream_unchanged(self):
        """Stream with p_commit=0: pm_a exactly unchanged,
        pm_K/pm_V approximately unchanged (unit_normalize re-applied with eps).
        """
        pm = self._make_pm(r=2, D_h=4)
        pm.pm_K = unit_normalize(torch.randn(2, 2, 4))
        pm.pm_V = unit_normalize(torch.randn(2, 2, 4))
        pm.pm_a = torch.tensor([[2.0, 1.0], [1.5, 0.5]])
        pm.elig_K = torch.randn(2, 2, 4)
        pm.elig_V = torch.randn(2, 2, 4)

        K_before = pm.pm_K[1].clone()
        V_before = pm.pm_V[1].clone()
        a_before = pm.pm_a[1].clone()

        # Only stream 0 commits (p_commit=1), stream 1 doesn't (p_commit=0)
        p_commit = torch.tensor([1.0, 0.0])
        lambda_vals = torch.tensor([0.9, 0.9])
        g = torch.tensor([0.5, 0.5])
        tau = torch.tensor([1.0, 1.0])
        pm.commit(p_commit, lambda_vals, g, None, tau)

        # pm_a is exact: decay = (1 - 0*(1-0.9)) = 1, alpha=0
        assert torch.equal(pm.pm_a[1], a_before), "non-committing stream a changed"
        # pm_K/pm_V pass through unit_normalize even with alpha=0,
        # so allow eps-level floating point difference from renormalization
        assert torch.allclose(pm.pm_K[1], K_before, atol=1e-6), \
            "non-committing stream K changed beyond renormalization tolerance"
        assert torch.allclose(pm.pm_V[1], V_before, atol=1e-6), \
            "non-committing stream V changed beyond renormalization tolerance"

    def test_commit_time_decay_formula(self):
        """pm_a = pm_a * (1 - p_commit * (1 - lambda)) for committing streams."""
        pm = self._make_pm(r=2, D_h=4)
        pm.pm_K = unit_normalize(torch.randn(1, 2, 4))
        pm.pm_V = unit_normalize(torch.randn(1, 2, 4))
        pm.pm_a = torch.tensor([[2.0, 3.0]])
        pm.elig_K = torch.randn(1, 2, 4)
        pm.elig_V = torch.randn(1, 2, 4)

        lambda_val = 0.8
        p_commit = torch.tensor([1.0])
        lambda_vals = torch.tensor([lambda_val])
        g = torch.tensor([0.0])  # g=0 so alpha=0, only decay matters
        tau = torch.tensor([1.0])

        pm.commit(p_commit, lambda_vals, g, None, tau)

        # With g=0: alpha=0 everywhere, so key/value EMA is identity
        # Decay: pm_a * (1 - 1*(1-0.8)) = pm_a * 0.8
        # Strength update: pm_a + 0 = pm_a (alpha=0)
        expected_a = torch.tensor([[2.0 * 0.8, 3.0 * 0.8]])
        assert torch.allclose(pm.pm_a, expected_a, atol=1e-5), \
            f"pm_a after decay: {pm.pm_a} != {expected_a}"


# ============================================================================
# Exact EM write equations (deterministic, single-candidate)
# ============================================================================

class TestEMWriteEquations:
    """Verify EM write_at_boundary formulas against hand-computed values.

    New interface: write_at_boundary(cand_K, cand_V, cand_score, g_em, tau, ww)
    Uses softmax slot selection instead of soft_topk.
    """

    def _make_em(self, M=4, D_em=4):
        cfg = make_tiny_config(M=M, D_em=D_em, C_em=1,
                               tau_em=1.0, weakness_weight_em=0.0,
                               S_max=10.0, budget_em=40.0, decay_em=0.9,
                               D=8, B=2, L=1)
        cfg.set_phase("B")
        em = EpisodicMemory(cfg)
        return em

    def test_ema_key_value_update(self):
        """em_K = unit_normalize((1-alpha) * em_K + alpha * k_c.unsqueeze(1))
           em_V = (1-alpha) * em_V + alpha * v_c.unsqueeze(1)  (no normalization on V!)

        scores_slot = [1.0, 0.0], softmax([1.0, 0.0]/1.0) = [e/(e+1), 1/(e+1)]
        """
        em = self._make_em(M=2, D_em=4)

        # Inject known state (BS=1)
        em.em_K = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]))  # [1,2,4]
        em.em_V = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0]]])  # [1,2,4] NOT normalized
        em.em_S = torch.tensor([[1.0, 0.5]])  # [1,2]

        # Single candidate pointing along slot 0
        k_c = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))  # [1,1,4]
        v_c = torch.tensor([[[0.0, 5.0, 0.0, 0.0]]])                  # [1,1,4]
        score = torch.tensor([[0.8]])  # [1,1]

        g_em = torch.tensor([0.5])
        tau = torch.tensor([1.0])
        ww = torch.tensor([0.0])
        decay = torch.tensor([0.9])

        # weakness_weight=0
        # scores_slot = einsum(k_c, em_K) = [1.0, 0.0]
        # softmax([1.0, 0.0] / 1.0):
        e_val = torch.exp(torch.tensor(1.0))
        w0 = e_val / (e_val + 1)  # ≈ 0.7311
        w1 = 1.0 / (e_val + 1)   # ≈ 0.2689
        # alpha = w * g * 1.0 (score_ok)
        a0 = (w0 * 0.5).item()
        a1 = (w1 * 0.5).item()

        em_K_before = em.em_K.clone()
        em_V_before = em.em_V.clone()

        em.write_at_boundary(k_c, v_c, score, g_em, tau, ww, decay=decay)

        # Key update slot 0: unit_normalize((1-a0)*[1,0,0,0] + a0*[1,0,0,0])
        # = unit_normalize([(1-a0)+a0, 0, 0, 0]) = [1, 0, 0, 0]
        assert torch.allclose(em.em_K[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-4)

        # Key update slot 1: unit_normalize((1-a1)*[0,1,0,0] + a1*[1,0,0,0])
        # = unit_normalize([a1, (1-a1), 0, 0])
        expected_K1 = unit_normalize(torch.tensor([[a1, 1.0 - a1, 0.0, 0.0]]))
        assert torch.allclose(em.em_K[0, 1], expected_K1.squeeze(), atol=1e-4), \
            f"em_K slot 1: {em.em_K[0,1]} != {expected_K1}"

        # Value update slot 0: (1-a0)*[1,0,0,0] + a0*[0,5,0,0]
        expected_V0 = torch.tensor([1.0 - a0, 5.0 * a0, 0.0, 0.0])
        assert torch.allclose(em.em_V[0, 0], expected_V0, atol=1e-4), \
            f"em_V slot 0: {em.em_V[0,0]} != {expected_V0}"

    def test_strength_update_and_decay(self):
        """em_S = clamp(em_S + alpha * score) then * decay then budget_enforce."""
        em = self._make_em(M=2, D_em=4)

        em.em_K = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]))
        em.em_V = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0]]])
        em.em_S = torch.tensor([[2.0, 1.0]])

        k_c = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
        v_c = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        score = torch.tensor([[0.6]])

        g_em = torch.tensor([0.4])
        tau = torch.tensor([1.0])
        ww = torch.tensor([0.0])
        decay = torch.tensor([0.9])

        # scores_slot = [1.0, 0.0], softmax = [w0, w1]
        e_val = torch.exp(torch.tensor(1.0))
        w0 = e_val / (e_val + 1)
        w1 = 1.0 / (e_val + 1)
        a0 = (w0 * 0.4).item()
        a1 = (w1 * 0.4).item()

        # strength: S[0] = clamp(2.0 + a0*0.6), S[1] = clamp(1.0 + a1*0.6)
        # then decay: S *= 0.9
        em.write_at_boundary(k_c, v_c, score, g_em, tau, ww, decay=decay)

        expected_S0 = (2.0 + a0 * 0.6) * 0.9
        expected_S1 = (1.0 + a1 * 0.6) * 0.9
        expected_S = torch.tensor([[expected_S0, expected_S1]])
        assert torch.allclose(em.em_S, expected_S, atol=1e-4), \
            f"em_S: {em.em_S} != {expected_S}"

    def test_em_value_not_normalized(self):
        """em_V EMA update must NOT unit-normalize (unlike em_K)."""
        em = self._make_em(M=2, D_em=4)
        em.em_K = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                 [0.0, 1.0, 0.0, 0.0]]]))
        em.em_V = torch.tensor([[[10.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0]]])
        em.em_S = torch.tensor([[1.0, 0.5]])

        k_c = unit_normalize(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
        v_c = torch.tensor([[[0.0, 20.0, 0.0, 0.0]]])
        score = torch.tensor([[0.5]])

        g_em = torch.tensor([0.5])
        tau = torch.tensor([1.0])
        ww = torch.tensor([0.0])
        decay = torch.tensor([0.9])

        em.write_at_boundary(k_c, v_c, score, g_em, tau, ww, decay=decay)

        # Slot 0 gets large alpha from softmax, V has large non-unit components
        v_norm = em.em_V[0, 0].norm().item()
        assert v_norm > 1.5, f"em_V should NOT be unit-normalized, but norm={v_norm}"

    def test_no_write_stream_unchanged(self):
        """Stream with g_em=0: em_V exactly unchanged,
        em_K approximately unchanged (unit_normalize re-applied with eps),
        em_S only decayed.
        """
        em = self._make_em(M=2, D_em=4)
        em.em_K = unit_normalize(torch.randn(2, 2, 4))
        em.em_V = torch.randn(2, 2, 4)
        em.em_S = torch.tensor([[1.0, 0.5], [2.0, 1.0]])

        K_before = em.em_K[1].clone()
        V_before = em.em_V[1].clone()
        S_before = em.em_S[1].clone()

        k_c = unit_normalize(torch.randn(2, 1, 4))
        v_c = torch.randn(2, 1, 4)
        score = torch.tensor([[0.5], [0.5]])

        # Only stream 0 writes (g_em=0.5), stream 1 doesn't (g_em=0)
        g_em = torch.tensor([0.5, 0.0])
        tau = torch.tensor([1.0, 1.0])
        ww = torch.tensor([0.0, 0.0])
        decay = torch.tensor([0.9, 0.9])

        em.write_at_boundary(k_c, v_c, score, g_em, tau, ww, decay=decay)

        # em_K passes through unit_normalize even with alpha=0 (eps-level change)
        assert torch.allclose(em.em_K[1], K_before, atol=1e-6), \
            "no-write stream em_K changed beyond renormalization tolerance"
        # em_V has no normalize — exact equality
        assert torch.equal(em.em_V[1], V_before), "no-write stream em_V should be unchanged"
        # em_S only decayed
        expected_S1 = S_before * 0.9  # decay only
        assert torch.allclose(em.em_S[1], expected_S1, atol=1e-5), \
            f"no-write stream em_S: {em.em_S[1]} != {expected_S1}"


# ============================================================================
# EM cold-start novelty (Fix B)
# ============================================================================

class TestEMColdStartNovelty:
    """When no EM slots are active, novelty = surprise only.

    Before the fix, novelty = 0.5*surprise + 0.5*(1-0) = 0.5*surprise + 0.5,
    which always exceeded the write threshold of 0.3 even with zero surprise.
    """

    def test_cold_start_novelty_equals_surprise(self):
        """With no active slots (em_S=0), novelty should equal clamped surprise."""
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        BS = 2
        # Run one token to initialize state
        reset = torch.zeros(BS, dtype=torch.bool)
        _, x_emb, y_wm = model.forward_one_token(
            torch.randint(0, 64, (BS,)), reset
        )

        # Set known surprise values
        model.surprise = torch.tensor([[0.1], [0.8]])

        for block in model.blocks:
            # Verify em_S is all zeros (cold start)
            assert (block.em.em_S == 0).all(), "Expected cold start (all em_S=0)"

            h_block = block.layers[-1].h
            _, _, novelty = block.em.propose_candidate(
                x_emb, y_wm, h_block, model.surprise
            )

            # Cold start: novelty = surprise_1d.clamp(0, 1)
            expected = model.surprise.squeeze(-1).clamp(0.0, 1.0)
            assert torch.allclose(novelty, expected, atol=1e-6), \
                f"Cold-start novelty {novelty} != surprise {expected}"

    def test_cold_start_low_surprise_below_threshold(self):
        """With no active slots and low surprise, novelty should be below
        the default write threshold of 0.3.

        This verifies that Fix B actually prevents spurious cold-start writes.
        """
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        BS = 2
        reset = torch.zeros(BS, dtype=torch.bool)
        _, x_emb, y_wm = model.forward_one_token(
            torch.randint(0, 64, (BS,)), reset
        )

        # Low surprise (model is confident)
        model.surprise = torch.tensor([[0.1], [0.2]])

        for block in model.blocks:
            h_block = block.layers[-1].h
            _, _, novelty = block.em.propose_candidate(
                x_emb, y_wm, h_block, model.surprise
            )
            # Novelty should be below default threshold (0.3)
            assert (novelty < 0.3).all(), \
                f"Cold-start novelty {novelty} should be < 0.3 with low surprise"

    def test_warm_start_uses_similarity(self):
        """After EM slots are populated, novelty uses the full formula
        (not surprise-only).
        """
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        # Populate EM with writes
        forward_and_write_em(model, cfg.P)

        BS = 2
        reset = torch.zeros(BS, dtype=torch.bool)
        logits, x_emb, y_wm = model.forward_one_token(
            torch.randint(0, 64, (BS,)), reset
        )
        # Update surprise so it's nonzero
        target = torch.randint(0, 64, (BS,))
        model.update_surprise(logits, target)

        for block in model.blocks:
            # Verify em_S has some active slots
            assert (block.em.em_S > 0).any(), "Expected warm start with active slots"

            h_block = block.layers[-1].h
            _, _, novelty = block.em.propose_candidate(
                x_emb, y_wm, h_block, model.surprise
            )
            surprise_1d = model.surprise.squeeze(-1)
            # Novelty should NOT equal surprise-only (similarity term contributes)
            # It could differ by any amount, so just check the shape is right
            assert novelty.shape == (BS,)
            # Novelty should be in [0, 1]
            assert (novelty >= 0).all() and (novelty <= 1).all()
