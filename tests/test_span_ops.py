"""Unit tests for src/training/span_ops.py — shared span-boundary operations."""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.training import span_ops
from tests.conftest import make_tiny_config

BS = 2
VOCAB = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_model(phase="B"):
    cfg = make_tiny_config()
    cfg.set_phase(phase)
    return NeuromorphicLM(cfg), cfg


# ============================================================================
# compute_loss_mask
# ============================================================================

class TestComputeLossMask:
    def test_eot_masked_when_reset_on_doc_boundary(self):
        eot_id = 1
        span_ids = torch.tensor([[1, 2, 3, 1], [4, 1, 5, 6]])  # EOT at various positions
        is_eot, loss_mask = span_ops.compute_loss_mask(span_ids, eot_id, reset_on_doc_boundary=True)
        # is_eot should flag positions where span_ids == eot_id
        assert is_eot[0, 0].item() is True
        assert is_eot[0, 3].item() is True
        assert is_eot[1, 1].item() is True
        # loss_mask should be inverse of is_eot
        assert loss_mask[0, 0].item() is False
        assert loss_mask[0, 1].item() is True

    def test_all_valid_when_no_reset(self):
        eot_id = 1
        span_ids = torch.tensor([[1, 2, 3, 1]])
        is_eot, loss_mask = span_ops.compute_loss_mask(span_ids, eot_id, reset_on_doc_boundary=False)
        assert is_eot[0, 0].item() is True  # still detects EOT
        assert loss_mask.all()  # but all positions are valid

    def test_shapes(self):
        span_ids = torch.randint(0, 64, (BS, 8))
        is_eot, loss_mask = span_ops.compute_loss_mask(span_ids, eot_id=0, reset_on_doc_boundary=True)
        assert is_eot.shape == (BS, 8)
        assert loss_mask.shape == (BS, 8)


# ============================================================================
# compute_reset_mask
# ============================================================================

class TestComputeResetMask:
    def test_zeroed_when_reset_disabled(self):
        model, cfg = _tiny_model("B")
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.ones(BS, dtype=torch.bool)
        mask = span_ops.compute_reset_mask(model, span_ids, reset_first, reset_on_doc_boundary=False)
        assert mask.shape == (BS, cfg.P)
        assert not mask.any()

    def test_nonzero_when_reset_enabled(self):
        model, cfg = _tiny_model("B")
        eot = cfg.eot_id
        # Put EOT at position 0 so the model computes resets
        span_ids = torch.full((BS, cfg.P), eot, dtype=torch.long)
        reset_first = torch.ones(BS, dtype=torch.bool)
        mask = span_ops.compute_reset_mask(model, span_ids, reset_first, reset_on_doc_boundary=True)
        assert mask.shape == (BS, cfg.P)
        assert mask[:, 0].all()  # first position should reset (reset_first is True)


# ============================================================================
# accumulate_span_surprise
# ============================================================================

class TestAccumulateSpanSurprise:
    def test_basic_accumulation(self):
        span_P = 4
        surprise = torch.ones(BS, span_P, 1)
        loss_mask = torch.ones(BS, span_P, dtype=torch.bool)
        reset_mask = torch.zeros(BS, span_P, dtype=torch.bool)
        accum = torch.zeros(BS)
        valid = torch.zeros(BS)
        last_reset = torch.zeros(BS, dtype=torch.long)

        result = span_ops.accumulate_span_surprise(
            surprise, loss_mask, reset_mask, True, accum, valid, last_reset,
        )
        # 4 tokens of surprise=1.0, mean should be 1.0
        assert torch.allclose(result, torch.ones(BS))
        assert valid[0].item() == 4.0

    def test_mid_span_reset_zeros_accumulators(self):
        span_P = 4
        surprise = torch.ones(BS, span_P, 1) * 2.0
        loss_mask = torch.ones(BS, span_P, dtype=torch.bool)
        reset_mask = torch.zeros(BS, span_P, dtype=torch.bool)
        reset_mask[0, 2] = True  # reset stream 0 at position 2
        accum = torch.zeros(BS)
        valid = torch.zeros(BS)
        last_reset = torch.zeros(BS, dtype=torch.long)

        result = span_ops.accumulate_span_surprise(
            surprise, loss_mask, reset_mask, True, accum, valid, last_reset,
        )
        # Stream 0: reset at t=2, then tokens at t=2,3 → accum=4, valid=2, mean=2
        # Stream 1: all 4 tokens → accum=8, valid=4, mean=2
        assert torch.allclose(result, torch.tensor([2.0, 2.0]))
        assert last_reset[0].item() == 2

    def test_masked_tokens_excluded(self):
        span_P = 4
        surprise = torch.ones(BS, span_P, 1) * 3.0
        loss_mask = torch.ones(BS, span_P, dtype=torch.bool)
        loss_mask[:, 0] = False  # mask first token
        reset_mask = torch.zeros(BS, span_P, dtype=torch.bool)
        accum = torch.zeros(BS)
        valid = torch.zeros(BS)
        last_reset = torch.zeros(BS, dtype=torch.long)

        result = span_ops.accumulate_span_surprise(
            surprise, loss_mask, reset_mask, True, accum, valid, last_reset,
        )
        assert valid[0].item() == 3.0  # only 3 of 4 tokens counted

    def test_no_reset_when_disabled(self):
        span_P = 4
        surprise = torch.ones(BS, span_P, 1)
        loss_mask = torch.ones(BS, span_P, dtype=torch.bool)
        reset_mask = torch.zeros(BS, span_P, dtype=torch.bool)
        reset_mask[0, 1] = True  # would reset, but disabled
        accum = torch.zeros(BS)
        valid = torch.zeros(BS)
        last_reset = torch.zeros(BS, dtype=torch.long)

        span_ops.accumulate_span_surprise(
            surprise, loss_mask, reset_mask, False, accum, valid, last_reset,
        )
        # No reset should have happened
        assert last_reset[0].item() == 0
        assert valid[0].item() == 4.0


# ============================================================================
# SpanAccumulator
# ============================================================================

class TestSpanAccumulator:
    def test_create_shapes(self):
        accum = span_ops.SpanAccumulator.create(BS, num_blocks=2, device=torch.device("cpu"))
        assert accum.surprise_accum.shape == (BS,)
        assert accum.valid_tokens.shape == (BS,)
        assert accum.last_reset.shape == (BS,)
        assert len(accum.em_cand_K) == 2
        assert len(accum.em_cand_V) == 2

    def test_reset_span_zeros(self):
        accum = span_ops.SpanAccumulator.create(BS, num_blocks=2, device=torch.device("cpu"))
        accum.surprise_accum.fill_(5.0)
        accum.valid_tokens.fill_(10.0)
        accum.em_cand_K[0].append(torch.randn(2, 4))
        accum.reset_span()
        assert accum.surprise_accum.sum().item() == 0.0
        assert accum.valid_tokens.sum().item() == 0.0
        assert len(accum.em_cand_K[0]) == 0

    def test_finalize_no_em(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")  # EM disabled
        accum = span_ops.SpanAccumulator.create(BS, cfg.B, torch.device("cpu"))
        accum.surprise_accum.fill_(6.0)
        accum.valid_tokens.fill_(3.0)
        result = accum.finalize(torch.device("cpu"), cfg)
        assert isinstance(result, span_ops.SpanResult)
        assert torch.allclose(result.surprise_mean, torch.tensor([2.0, 2.0]))
        assert result.em_stacked == {}

    def test_finalize_with_em(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")  # EM enabled
        accum = span_ops.SpanAccumulator.create(BS, cfg.B, torch.device("cpu"))
        accum.surprise_accum.fill_(4.0)
        accum.valid_tokens.fill_(2.0)
        # Add some fake candidates
        D_em = cfg.D_em
        for b in range(cfg.B):
            accum.em_cand_K[b].append(torch.randn(BS, 3, D_em))
            accum.em_cand_V[b].append(torch.randn(BS, 3, D_em))
            accum.em_cand_score[b].append(torch.rand(BS, 3))
            accum.em_cand_valid[b].append(torch.ones(BS, 3, dtype=torch.bool))
        result = accum.finalize(torch.device("cpu"), cfg)
        assert len(result.em_stacked) == cfg.B
        for b in result.em_stacked:
            sK, sV, sScore, sValid, novelty = result.em_stacked[b]
            assert sK.shape == (BS, 3, D_em)


# ============================================================================
# stack_em_candidates
# ============================================================================

class TestStackEmCandidates:
    def test_basic_stacking(self):
        D = 8
        cand_K = [[torch.randn(BS, 4, D)]]
        cand_V = [[torch.randn(BS, 4, D)]]
        cand_score = [[torch.rand(BS, 4)]]
        cand_valid = [[torch.ones(BS, 4, dtype=torch.bool)]]
        last_reset = torch.zeros(BS, dtype=torch.long)

        result = span_ops.stack_em_candidates(
            cand_K, cand_V, cand_score, cand_valid, last_reset, torch.device("cpu"),
        )
        assert 0 in result
        sK, sV, sScore, sValid, novelty = result[0]
        assert sK.shape == (BS, 4, D)
        assert novelty.shape == (BS,)

    def test_last_reset_masks_early_tokens(self):
        D = 8
        cand_K = [[torch.randn(BS, 4, D)]]
        cand_V = [[torch.randn(BS, 4, D)]]
        cand_score = [[torch.ones(BS, 4)]]  # all scores = 1.0
        cand_valid = [[torch.ones(BS, 4, dtype=torch.bool)]]
        # Stream 0: last reset at position 2 → only positions 2,3 valid
        last_reset = torch.tensor([2, 0])

        result = span_ops.stack_em_candidates(
            cand_K, cand_V, cand_score, cand_valid, last_reset, torch.device("cpu"),
        )
        _, _, _, sValid, novelty = result[0]
        # Stream 0: positions 0,1 invalid (before reset), 2,3 valid
        assert sValid[0, 0].item() is False
        assert sValid[0, 1].item() is False
        assert sValid[0, 2].item() is True
        assert sValid[0, 3].item() is True
        # Stream 1: all valid (last_reset=0)
        assert sValid[1].all()

    def test_empty_candidates(self):
        cand_K = [[]]
        cand_V = [[]]
        cand_score = [[]]
        cand_valid = [[]]
        last_reset = torch.zeros(BS, dtype=torch.long)

        result = span_ops.stack_em_candidates(
            cand_K, cand_V, cand_score, cand_valid, last_reset, torch.device("cpu"),
        )
        assert result == {}


# ============================================================================
# apply_mid_span_resets
# ============================================================================

class TestApplyMidSpanResets:
    def test_clears_pm_content_on_mid_span_reset(self):
        """PM committed content (pm_a) should be zeroed for streams with mid-span resets."""
        model, cfg = _tiny_model("B")
        # Populate PM state by running tokens + committing
        reset = torch.zeros(BS, dtype=torch.bool)
        for _ in range(cfg.P):
            x = torch.randint(0, VOCAB, (BS,))
            model.forward_one_token(x, reset)
            model.update_surprise(model.forward_one_token(x, reset)[0], x)
        model.commit_at_boundary(force_mode="force_on")

        # Verify PM has nonzero content
        pm0 = model.blocks[0].layers[0].pm
        assert pm0.pm_a is not None
        assert pm0.pm_a.abs().sum() > 0, "PM should have nonzero content after commit"

        # Simulate mid-span reset at position 2 for stream 0 only
        reset_mask_all = torch.zeros(BS, cfg.P, dtype=torch.bool)
        reset_mask_all[0, 2] = True  # mid-span reset for stream 0

        span_ops.apply_mid_span_resets(model, reset_mask_all, cfg)

        # Stream 0: PM content should be zeroed
        assert pm0.pm_a[0].abs().sum() == 0, "Stream 0 PM content should be cleared"
        # Stream 1: PM content should be preserved
        assert pm0.pm_a[1].abs().sum() > 0, "Stream 1 PM content should be preserved"

    def test_clears_em_strengths_on_mid_span_reset(self):
        """EM strengths (em_S) should be zeroed for streams with mid-span resets."""
        model, cfg = _tiny_model("B")
        # Initialize EM state
        reset = torch.zeros(BS, dtype=torch.bool)
        x = torch.randint(0, VOCAB, (BS,))
        model.forward_one_token(x, reset)  # lazy-init EM

        # Manually set EM strengths
        for block in model.blocks:
            if block.em.em_S is not None:
                block.em.em_S = torch.ones_like(block.em.em_S) * 0.5

        # Mid-span reset at position 1 for stream 0
        reset_mask_all = torch.zeros(BS, cfg.P, dtype=torch.bool)
        reset_mask_all[0, 1] = True

        span_ops.apply_mid_span_resets(model, reset_mask_all, cfg)

        for block in model.blocks:
            assert block.em.em_S[0].abs().sum() == 0, "Stream 0 EM strengths should be cleared"
            assert block.em.em_S[1].abs().sum() > 0, "Stream 1 EM strengths should be preserved"

    def test_preserves_eligibility(self):
        """Eligibility traces should NOT be cleared by mid-span reset."""
        model, cfg = _tiny_model("B")
        # Forward a span to populate eligibility
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        model.forward_span(span_ids, reset_first)

        # Run PM eligibility batch to populate elig_K/V
        reset_mask = torch.zeros(BS, cfg.P, dtype=torch.bool)
        surprise = torch.ones(BS, cfg.P, 1)
        span_ops.apply_pm_eligibility_batch(
            model, surprise, reset_mask, cfg,
        )

        # Snapshot eligibility before reset
        pm0 = model.blocks[0].layers[0].pm
        elig_before = pm0.elig_K.clone()

        # Mid-span reset for stream 0
        reset_mask_all = torch.zeros(BS, cfg.P, dtype=torch.bool)
        reset_mask_all[0, 2] = True
        span_ops.apply_mid_span_resets(model, reset_mask_all, cfg)

        # Eligibility should be untouched
        assert torch.equal(pm0.elig_K, elig_before), "Eligibility should not be cleared"

    def test_noop_for_first_position_reset(self):
        """Reset at position 0 only should not trigger mid-span reset."""
        model, cfg = _tiny_model("B")
        reset = torch.zeros(BS, dtype=torch.bool)
        x = torch.randint(0, VOCAB, (BS,))
        model.forward_one_token(x, reset)

        # Set EM strengths
        for block in model.blocks:
            if block.em.em_S is not None:
                block.em.em_S = torch.ones_like(block.em.em_S) * 0.5

        # Reset only at position 0 — this is handled by forward_span, not mid-span
        reset_mask_all = torch.zeros(BS, cfg.P, dtype=torch.bool)
        reset_mask_all[:, 0] = True  # first position only

        span_ops.apply_mid_span_resets(model, reset_mask_all, cfg)

        # EM strengths should be unchanged (no mid-span reset)
        for block in model.blocks:
            assert block.em.em_S.abs().sum() > 0, "No mid-span reset should have occurred"

    def test_noop_in_lifelong_mode(self):
        """Lifelong mode should preserve PM/EM content across doc boundaries."""
        model, cfg = _tiny_model("B")
        cfg.lifelong_mode = True
        reset = torch.zeros(BS, dtype=torch.bool)
        x = torch.randint(0, VOCAB, (BS,))
        model.forward_one_token(x, reset)

        for block in model.blocks:
            if block.em.em_S is not None:
                block.em.em_S = torch.ones_like(block.em.em_S) * 0.5

        reset_mask_all = torch.zeros(BS, cfg.P, dtype=torch.bool)
        reset_mask_all[0, 2] = True

        span_ops.apply_mid_span_resets(model, reset_mask_all, cfg)

        # Lifelong mode: EM strengths should be preserved
        for block in model.blocks:
            assert block.em.em_S[0].abs().sum() > 0, "Lifelong mode should preserve EM"

    def test_noop_when_no_mid_span_resets(self):
        """No reset at positions 1..P-1 means no PM/EM clearing."""
        model, cfg = _tiny_model("B")
        reset = torch.zeros(BS, dtype=torch.bool)
        x = torch.randint(0, VOCAB, (BS,))
        model.forward_one_token(x, reset)

        for block in model.blocks:
            if block.em.em_S is not None:
                block.em.em_S = torch.ones_like(block.em.em_S) * 0.5

        # No resets at all
        reset_mask_all = torch.zeros(BS, cfg.P, dtype=torch.bool)
        span_ops.apply_mid_span_resets(model, reset_mask_all, cfg)

        for block in model.blocks:
            assert block.em.em_S.abs().sum() > 0


# ============================================================================
# PM.reset_content
# ============================================================================

class TestPMResetContent:
    def test_zeros_content_preserves_eligibility(self):
        """reset_content should zero pm_K/pm_V/pm_a but NOT elig_K/elig_V."""
        model, cfg = _tiny_model("B")
        # Populate PM content via commit
        reset = torch.zeros(BS, dtype=torch.bool)
        for _ in range(cfg.P):
            x = torch.randint(0, VOCAB, (BS,))
            model.forward_one_token(x, reset)
            model.update_surprise(model.forward_one_token(x, reset)[0], x)
        model.commit_at_boundary(force_mode="force_on")

        # Run more tokens to re-populate eligibility after commit's soft-reset
        for _ in range(cfg.P):
            x = torch.randint(0, VOCAB, (BS,))
            model.forward_one_token(x, reset)
            model.update_surprise(model.forward_one_token(x, reset)[0], x)

        pm = model.blocks[0].layers[0].pm
        # Verify everything is populated
        assert pm.pm_a.abs().sum() > 0
        assert pm.elig_K.abs().sum() > 0

        elig_before = pm.elig_K.clone()

        # Reset content for stream 0 only
        mask = torch.tensor([True, False])
        pm.reset_content(mask)

        assert pm.pm_a[0].abs().sum() == 0
        assert pm.pm_K[0].abs().sum() == 0
        assert pm.pm_V[0].abs().sum() == 0
        assert pm.pm_a[1].abs().sum() > 0  # stream 1 preserved
        assert torch.equal(pm.elig_K, elig_before)  # eligibility untouched


# ============================================================================
# apply_pm_boundary
# ============================================================================

class TestApplyPmBoundary:
    def test_returns_commit_info(self):
        model, cfg = _tiny_model("B")
        # Run some tokens to populate state
        reset = torch.zeros(BS, dtype=torch.bool)
        for _ in range(cfg.P):
            x = torch.randint(0, VOCAB, (BS,))
            model.forward_one_token(x, reset)
            model.update_surprise(model.forward_one_token(x, reset)[0], x)
        surprise = torch.ones(BS) * 2.0
        commit_info = span_ops.apply_pm_boundary(model, surprise)
        assert isinstance(commit_info, dict)


# ============================================================================
# apply_em_boundary
# ============================================================================

class TestApplyEmBoundary:
    def test_returns_write_info(self):
        model, cfg = _tiny_model("B")
        D_em = cfg.D_em
        em_stacked = {}
        for b in range(cfg.B):
            em_stacked[b] = (
                torch.randn(BS, 4, D_em),  # sK
                torch.randn(BS, 4, D_em),  # sV
                torch.rand(BS, 4),          # sScore
                torch.ones(BS, 4, dtype=torch.bool),  # sValid
                torch.rand(BS),             # novelty_mean
            )
        surprise = torch.ones(BS) * 2.0
        write_info = span_ops.apply_em_boundary(model, em_stacked, surprise, cfg)
        assert isinstance(write_info, list)
        assert len(write_info) == cfg.B
        for b_idx, novelty_mean, g_em_mean in write_info:
            assert isinstance(b_idx, int)


# ============================================================================
# propose_em_candidates
# ============================================================================

class TestProposeEmCandidates:
    def test_appends_to_lists(self):
        model, cfg = _tiny_model("B")
        # Forward a span to populate _last_h_all
        span_ids = torch.randint(0, VOCAB, (BS, cfg.P))
        reset_first = torch.zeros(BS, dtype=torch.bool)
        logits, x_emb, y_wm = model.forward_span(span_ids, reset_first)
        surprise = torch.rand(BS, cfg.P, 1)
        loss_mask = torch.ones(BS, cfg.P, dtype=torch.bool)

        cand_K = [[] for _ in range(cfg.B)]
        cand_V = [[] for _ in range(cfg.B)]
        cand_score = [[] for _ in range(cfg.B)]
        cand_valid = [[] for _ in range(cfg.B)]

        span_ops.propose_em_candidates(
            model, x_emb, y_wm, surprise, loss_mask,
            cand_K, cand_V, cand_score, cand_valid,
        )

        for b in range(cfg.B):
            assert len(cand_K[b]) == 1
            assert cand_K[b][0].shape[0] == BS
            assert cand_K[b][0].shape[1] == cfg.P
