"""Tests for FITB (Fill-In-The-Blank) training architecture."""

import pytest
import torch
from tests.conftest import make_tiny_config, forward_one_segment, forward_one_segment_fitb
from src.model.model import NeuromorphicLM
from src.training.masking import (
    generate_random_mask,
    generate_span_mask,
    generate_fitb_mask,
    apply_special_token_protection,
)
from src.training.loss import fitb_cross_entropy


BS = 2
VOCAB = 64


def _make_fitb_model(**overrides):
    cfg = make_tiny_config(**overrides)
    return NeuromorphicLM(cfg)


class TestFITBForwardShape:
    """Per-pass logits shape: R × [BS, N, vocab]."""

    def test_per_pass_logits_count(self):
        model = _make_fitb_model()
        per_pass_logits, aux_loss, _ = forward_one_segment_fitb(model)
        assert len(per_pass_logits) == model.config.R

    def test_per_pass_logits_shape(self):
        model = _make_fitb_model()
        N = model.config.N
        per_pass_logits, _, _ = forward_one_segment_fitb(model, BS=BS)
        for logits_r in per_pass_logits:
            assert logits_r.shape == (BS, N, VOCAB)

    def test_aux_loss_scalar(self):
        model = _make_fitb_model()
        _, aux_loss, _ = forward_one_segment_fitb(model)
        assert aux_loss.dim() == 0

    def test_different_R_values(self):
        for R in [1, 2, 4]:
            model = _make_fitb_model(R=R)
            per_pass_logits, _, _ = forward_one_segment_fitb(model)
            assert len(per_pass_logits) == R


class TestFITBBackwardCompat:
    """Without fitb_mask, returns single logits tensor (NTP path)."""

    def test_ntp_returns_tensor(self):
        model = _make_fitb_model()
        logits, aux_loss = forward_one_segment(model)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (BS, model.config.N, VOCAB)

    def test_ntp_and_fitb_same_model(self):
        """Same model can do both NTP and FITB."""
        model = _make_fitb_model()
        model.initialize_states(BS, torch.device("cpu"))

        # NTP
        ids = torch.randint(0, VOCAB, (BS, model.config.N))
        reset = torch.zeros(BS, dtype=torch.bool)
        logits, _ = model.forward_segment(ids, reset)
        assert isinstance(logits, torch.Tensor)

        # FITB
        fitb_mask = torch.rand(BS, model.config.N) < 0.3
        ids_masked = ids.clone()
        ids_masked[fitb_mask] = model.config.fitb_id
        per_pass, _ = model.forward_segment(ids_masked, reset, fitb_mask=fitb_mask)
        assert isinstance(per_pass, list)


class TestFITBGradients:
    """Each pass gets gradients, no cross-pass gradient leak."""

    def test_backward_runs(self):
        model = _make_fitb_model()
        per_pass_logits, aux_loss, fitb_mask = forward_one_segment_fitb(model)
        targets = torch.randint(0, VOCAB, (BS, model.config.N))
        loss, valid = fitb_cross_entropy(per_pass_logits, targets, fitb_mask)
        loss.backward()
        # Check that gradients exist
        assert model.embedding.weight.grad is not None
        assert model.embedding.weight.grad.abs().sum() > 0

    def test_per_pass_independent_gradients(self):
        """Each pass's logits produce independent gradients (no cross-pass flow)."""
        model = _make_fitb_model()
        N = model.config.N
        input_ids = torch.randint(0, VOCAB, (BS, N))
        fitb_mask = torch.rand(BS, N) < 0.3
        ids_masked = input_ids.clone()
        ids_masked[fitb_mask] = model.config.fitb_id
        targets = input_ids.clone()

        model.initialize_states(BS, torch.device("cpu"))
        per_pass_logits, _ = model.forward_segment(
            ids_masked, torch.zeros(BS, dtype=torch.bool), fitb_mask=fitb_mask
        )

        # Backprop from last pass only
        flat_mask = fitb_mask.reshape(-1)
        flat_tgt = targets.reshape(-1).clone()
        flat_tgt[~flat_mask] = -100
        last_logits = per_pass_logits[-1]
        loss = torch.nn.functional.cross_entropy(
            last_logits.reshape(-1, VOCAB), flat_tgt,
            ignore_index=-100, reduction="sum",
        )
        loss.backward()
        assert model.fan_in.weight.grad is not None

    def test_no_grad_on_reembed_argmax(self):
        """Re-embedding via argmax should not carry gradients."""
        model = _make_fitb_model(R=3)
        N = model.config.N
        input_ids = torch.randint(0, VOCAB, (BS, N))
        fitb_mask = torch.ones(BS, N, dtype=torch.bool)  # mask everything
        ids_masked = torch.full((BS, N), model.config.fitb_id)

        model.initialize_states(BS, torch.device("cpu"))
        per_pass_logits, _ = model.forward_segment(
            ids_masked, torch.zeros(BS, dtype=torch.bool), fitb_mask=fitb_mask
        )
        # If argmax leaked gradients, this would error or produce wrong grads
        loss = per_pass_logits[-1].sum()
        loss.backward()


class TestFITBMasking:
    """Mask generation utilities."""

    def test_random_mask_rate(self):
        mask = generate_random_mask(1000, 128, 0.3, torch.device("cpu"))
        rate = mask.float().mean().item()
        assert 0.2 < rate < 0.4  # stochastic, but should be close

    def test_span_mask_produces_spans(self):
        mask = generate_span_mask(1, 256, 0.3, 5, torch.device("cpu"))
        # Should have contiguous runs > 1
        row = mask[0]
        diffs = row[1:].int() - row[:-1].int()
        # Number of span starts should be less than total masked
        span_starts = (diffs == 1).sum().item()
        total_masked = row.sum().item()
        if total_masked > 5:
            assert span_starts < total_masked  # spans cluster tokens

    def test_fitb_mask_respects_rate(self):
        mask = generate_fitb_mask(100, 128, 0.3, 0.5, 3, torch.device("cpu"))
        rate = mask.float().mean().item()
        assert 0.1 < rate < 0.6  # wide tolerance due to stochasticity

    def test_zero_mask_rate(self):
        mask = generate_fitb_mask(2, 16, 0.0, 0.5, 3, torch.device("cpu"))
        assert mask.sum() == 0

    def test_special_token_protection(self):
        BS, N = 2, 16
        eot_id, null_id = 2, 62
        seg_ids = torch.randint(0, VOCAB, (BS, N))
        seg_ids[0, 3] = eot_id
        seg_ids[1, 7] = null_id
        fitb_mask = torch.ones(BS, N, dtype=torch.bool)

        protected = apply_special_token_protection(fitb_mask, seg_ids, eot_id, null_id)
        assert not protected[0, 3].item()  # EOT not masked
        assert not protected[1, 7].item()  # NULL not masked
        # Other positions still masked
        assert protected[0, 0].item()


class TestFITBLoss:
    """Loss only at masked positions, per-pass independence."""

    def test_loss_only_at_masked_positions(self):
        """Loss should be zero when no positions are masked."""
        logits = [torch.randn(BS, 16, VOCAB, requires_grad=True)]
        targets = torch.randint(0, VOCAB, (BS, 16))
        no_mask = torch.zeros(BS, 16, dtype=torch.bool)
        loss, valid = fitb_cross_entropy(logits, targets, no_mask)
        assert valid.item() == 0
        # Loss should still be zero (no masked positions)
        assert loss.item() == 0.0

    def test_loss_scales_with_passes(self):
        """More passes = more loss terms."""
        targets = torch.randint(0, VOCAB, (BS, 16))
        fitb_mask = torch.ones(BS, 16, dtype=torch.bool)
        logits_1 = [torch.randn(BS, 16, VOCAB)]
        logits_2 = [torch.randn(BS, 16, VOCAB)] * 2

        loss_1, valid_1 = fitb_cross_entropy(logits_1, targets, fitb_mask)
        loss_2, valid_2 = fitb_cross_entropy(logits_2, targets, fitb_mask)

        # 2 passes should have 2x the valid count
        assert valid_2.item() == 2 * valid_1.item()

    def test_loss_finite(self):
        model = _make_fitb_model()
        per_pass_logits, _, fitb_mask = forward_one_segment_fitb(model)
        targets = torch.randint(0, VOCAB, (BS, model.config.N))
        loss, valid = fitb_cross_entropy(per_pass_logits, targets, fitb_mask)
        assert torch.isfinite(loss)
        assert valid.item() > 0


class TestFITBIterativeRefinement:
    """Predictions should change across passes."""

    def test_predictions_change_across_passes(self):
        model = _make_fitb_model(R=3)
        per_pass_logits, _, fitb_mask = forward_one_segment_fitb(
            model, mask_rate=0.5
        )
        # At least one pass should produce different argmax predictions
        preds = [l.argmax(dim=-1) for l in per_pass_logits]
        any_different = False
        for i in range(1, len(preds)):
            if not torch.equal(preds[i], preds[0]):
                any_different = True
                break
        # With random init, predictions almost certainly differ across passes
        assert any_different

    def test_logits_differ_across_passes(self):
        model = _make_fitb_model(R=2)
        per_pass_logits, _, _ = forward_one_segment_fitb(model, mask_rate=0.5)
        # Logits should not be identical
        assert not torch.allclose(per_pass_logits[0], per_pass_logits[1])


class TestInlineFITBLoss:
    """Inline FITB loss (target_ids path) matches legacy per_pass_logits path."""

    def test_inline_loss_backward_runs(self):
        model = _make_fitb_model()
        N = model.config.N
        input_ids = torch.randint(0, VOCAB, (BS, N))
        fitb_mask = torch.rand(BS, N) < 0.3
        ids_masked = input_ids.clone()
        ids_masked[fitb_mask] = model.config.fitb_id

        model.initialize_states(BS, torch.device("cpu"))
        ce_loss, aux, valid = model.forward_segment(
            ids_masked, torch.zeros(BS, dtype=torch.bool),
            fitb_mask=fitb_mask, target_ids=input_ids,
        )
        assert ce_loss.dim() == 0
        assert valid.item() > 0
        loss = ce_loss / valid.float().clamp(min=1) + aux
        loss.backward()
        assert model.embedding.weight.grad is not None

    def test_inline_loss_matches_legacy(self):
        """Inline path loss should match legacy per_pass_logits + fitb_cross_entropy."""
        torch.manual_seed(42)
        model = _make_fitb_model(R=2)
        N = model.config.N
        input_ids = torch.randint(0, VOCAB, (BS, N))
        fitb_mask = torch.rand(BS, N) < 0.5
        ids_masked = input_ids.clone()
        ids_masked[fitb_mask] = model.config.fitb_id

        # Legacy path
        model.initialize_states(BS, torch.device("cpu"))
        per_pass_logits, aux1 = model.forward_segment(
            ids_masked, torch.zeros(BS, dtype=torch.bool),
            fitb_mask=fitb_mask,
        )
        loss_legacy, valid_legacy = fitb_cross_entropy(
            per_pass_logits, input_ids, fitb_mask
        )

        # Reset states for fresh inline path
        model.initialize_states(BS, torch.device("cpu"))
        loss_inline, aux2, valid_inline = model.forward_segment(
            ids_masked, torch.zeros(BS, dtype=torch.bool),
            fitb_mask=fitb_mask, target_ids=input_ids,
        )

        assert valid_inline.item() == valid_legacy.item()
        assert torch.allclose(loss_inline, loss_legacy, atol=1e-4)


class TestResizeTokenEmbeddings:
    """resize_token_embeddings preserves old weights and expands vocab."""

    def test_expand_vocab(self):
        model = _make_fitb_model(vocab_size=64, tie_embeddings=False)
        old_weight = model.embedding.weight.data[:64].clone()
        model.resize_token_embeddings(66)
        assert model.embedding.num_embeddings == 66
        assert model.config.vocab_size == 66
        assert torch.equal(model.embedding.weight.data[:64], old_weight)

    def test_tied_embeddings_after_resize(self):
        model = _make_fitb_model(vocab_size=64, tie_embeddings=True)
        model.resize_token_embeddings(66)
        assert model.lm_head.weight is model.embedding.weight

    def test_same_size_noop(self):
        model = _make_fitb_model(vocab_size=64)
        old_emb = model.embedding
        model.resize_token_embeddings(64)
        assert model.embedding is old_emb  # same object, no reallocation
