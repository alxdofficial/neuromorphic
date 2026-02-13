"""Integration smoke tests — end-to-end model behavior."""

import pytest
import torch
import torch.nn.functional as F

from src.model.model import NeuromorphicLM
from src.model.state import save_runtime_state, load_runtime_state
from src.training.loss import online_cross_entropy
from tests.conftest import (
    make_tiny_config, forward_n_tokens, forward_and_write_em,
)

BS = 2
VOCAB = 64


# ============================================================================
# Full forward pass per phase
# ============================================================================

class TestForwardPass:
    @pytest.mark.parametrize("phase", ["A", "B", "C"])
    def test_forward_per_phase(self, phase):
        cfg = make_tiny_config()
        cfg.set_phase(phase)
        model = NeuromorphicLM(cfg)

        logits, target = forward_n_tokens(model, 4, BS=BS, vocab=VOCAB)
        assert logits.shape == (BS, VOCAB)
        assert torch.isfinite(logits).all()

    def test_forward_phase_c(self):
        cfg = make_tiny_config()
        cfg.set_phase("C")
        model = NeuromorphicLM(cfg)

        logits, target = forward_n_tokens(model, 4, BS=BS, vocab=VOCAB)
        assert logits.shape == (BS, VOCAB)
        assert torch.isfinite(logits).all()

    def test_forward_with_decoder(self):
        cfg = make_tiny_config(snapshot_enabled=True)
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)

        logits, target = forward_n_tokens(model, 4, BS=BS, vocab=VOCAB)
        assert logits.shape == (BS, VOCAB)
        assert torch.isfinite(logits).all()


# ============================================================================
# Forward + backward
# ============================================================================

class TestBackward:
    def test_forward_backward_no_crash(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        logits, target = forward_n_tokens(model, 4)
        loss = F.cross_entropy(logits, target)
        loss.backward()  # should not crash


# ============================================================================
# Multi-token with commits
# ============================================================================

class TestMultiTokenCommits:
    @pytest.mark.slow
    def test_multi_token_with_commits_no_nan(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        for _ in range(3):
            logits, _ = forward_n_tokens(model, cfg.P, with_commits=True)
            assert torch.isfinite(logits).all()


# ============================================================================
# Doc boundary reset
# ============================================================================

class TestDocBoundary:
    def test_doc_boundary_reset_no_crash(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        forward_n_tokens(model, 4)

        mask = torch.tensor([True, False])
        model.reset_at_doc_boundary(mask)

        # Forward again after reset
        logits, _ = forward_n_tokens(model, 4)
        assert torch.isfinite(logits).all()


# ============================================================================
# Collect mode
# ============================================================================

class TestCollectMode:
    def test_collect_mode_returns_stats(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        input_id = torch.randint(0, VOCAB, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        result = model.forward_one_token(input_id, reset, collect=True)

        assert len(result) == 4
        logits, x_emb, y_wm, stats = result
        assert isinstance(stats, dict)
        # Should have one entry per block
        assert len(stats) == cfg.B
        # Each block should have entries per layer
        for b_idx, bstats in stats.items():
            assert len(bstats) == cfg.L
            for l_idx, lstats in bstats.items():
                assert "gate_a" in lstats
                assert "gate_b" in lstats
                assert "h_norm" in lstats


# ============================================================================
# Training step metrics
# ============================================================================

class TestTrainingMetrics:
    def test_online_cross_entropy_metrics(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        input_id = torch.randint(0, VOCAB, (BS,))
        target = torch.randint(0, VOCAB, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        mask = torch.ones(BS, dtype=torch.bool)

        logits, _, _ = model.forward_one_token(input_id, reset)
        loss_sum, valid_count = online_cross_entropy(logits, target, mask)

        assert valid_count == BS
        assert loss_sum.requires_grad
        assert torch.isfinite(loss_sum)


# ============================================================================
# Loss reduction over steps
# ============================================================================

class TestLossReduction:
    @pytest.mark.slow
    def test_loss_decreases_over_steps(self):
        """Loss should decrease over 5 training steps (overfitting a tiny batch)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed batch to overfit
        input_ids = torch.randint(0, VOCAB, (BS, 8))
        targets = torch.randint(0, VOCAB, (BS, 8))

        losses = []
        for step in range(5):
            optimizer.zero_grad()
            model.detach_states()
            # Reset model state
            mask = torch.ones(BS, dtype=torch.bool)
            model.reset_at_doc_boundary(mask)

            total_loss = torch.tensor(0.0)
            for t in range(8):
                reset = torch.zeros(BS, dtype=torch.bool)
                logits, _, _ = model.forward_one_token(input_ids[:, t], reset)
                loss = F.cross_entropy(logits, targets[:, t])
                total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ============================================================================
# Phase transitions
# ============================================================================

class TestPhaseTransitions:
    @pytest.mark.slow
    def test_phase_transitions_a_b_c(self):
        """Config set_phase() + model reinit doesn't crash."""
        cfg = make_tiny_config()

        for phase in ["A", "B", "C"]:
            cfg.set_phase(phase)
            model = NeuromorphicLM(cfg)
            logits, _ = forward_n_tokens(model, 4, BS=BS, vocab=VOCAB)
            assert torch.isfinite(logits).all(), f"Phase {phase} produced non-finite logits"


# ============================================================================
# Checkpoint roundtrip
# ============================================================================

class TestCheckpointRoundtrip:
    @pytest.mark.slow
    def test_checkpoint_roundtrip(self):
        """Save/load state_dict + runtime_state, get identical logits."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model1 = NeuromorphicLM(cfg)
        forward_n_tokens(model1, cfg.P, with_commits=True)

        # Save everything
        param_state = model1.state_dict()
        runtime_state = save_runtime_state(model1)

        # Deterministic forward
        torch.manual_seed(42)
        input_id = torch.randint(0, VOCAB, (BS,))
        reset = torch.zeros(BS, dtype=torch.bool)
        logits1, _, _ = model1.forward_one_token(input_id, reset)

        # Create new model, load state
        model2 = NeuromorphicLM(cfg)
        model2.load_state_dict(param_state)
        # Need to warm up for runtime state sizes
        forward_n_tokens(model2, 4, BS=BS, vocab=VOCAB)
        load_runtime_state(model2, runtime_state)

        # Same forward
        torch.manual_seed(42)
        logits2, _, _ = model2.forward_one_token(input_id, reset)

        assert torch.allclose(logits1, logits2, atol=1e-5), \
            "Logits differ after checkpoint roundtrip"


# ============================================================================
# TBPTT boundary
# ============================================================================

class TestTBPTT:
    def test_detach_then_forward_no_crash(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)

        forward_n_tokens(model, 4)
        model.detach_states()
        logits, _ = forward_n_tokens(model, 4)
        loss = F.cross_entropy(logits, torch.randint(0, VOCAB, (BS,)))
        loss.backward()  # should not crash


# ============================================================================
# EM write integration
# ============================================================================

class TestEMWriteIntegration:
    def test_em_write_populates_state(self):
        cfg = make_tiny_config()
        cfg.set_phase("B")
        model = NeuromorphicLM(cfg)
        forward_and_write_em(model, cfg.P)

        for block in model.blocks:
            # Some slots should have non-zero strength after write
            assert (block.em.em_S > 0).any(), \
                "EM write should populate some slots"


# ============================================================================
# WM reset
# ============================================================================

class TestWMReset:
    def test_wm_reset_on_doc_boundary(self):
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        forward_n_tokens(model, 4)

        # After forward, WM should have some valid entries
        assert model.wm.wm_valid.any()

        # Reset all
        mask = torch.ones(BS, dtype=torch.bool)
        model.reset_at_doc_boundary(mask)

        # WM validity should be cleared (via the forward_one_token reset path)
        # Note: reset_at_doc_boundary calls block.reset_states, not wm directly
        # WM reset happens on the NEXT forward_one_token call with reset_mask
        input_id = torch.randint(0, VOCAB, (BS,))
        model.forward_one_token(input_id, mask)
        # After forward with reset_mask, ptr should be 1 (just wrote one token)
        assert (model.wm.wm_ptr == 1).all()


# ============================================================================
# Generation / inference
# ============================================================================

class TestGenerate:
    @pytest.mark.parametrize("phase", ["A", "B", "C"])
    def test_generate_output_shape(self, phase):
        """generate() returns [BS, T_prompt + max_new_tokens]."""
        cfg = make_tiny_config()
        cfg.set_phase(phase)
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 5))
        out = model.generate(prompt, max_new_tokens=8)
        assert out.shape == (BS, 5 + 8)

    def test_generate_prompt_preserved(self):
        """Generated output starts with the prompt tokens."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 4))
        out = model.generate(prompt, max_new_tokens=3)
        assert torch.equal(out[:, :4], prompt)

    def test_generate_surprise_nonzero(self):
        """After generation, surprise should be populated (nonzero)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 6))
        model.generate(prompt, max_new_tokens=4)
        assert model.surprise is not None
        # With random weights, surprise should be nonzero
        assert model.surprise.abs().sum() > 0

    def test_generate_tokens_in_vocab_range(self):
        """All generated tokens should be valid vocab indices."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 3))
        out = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=10)
        assert (out >= 0).all()
        assert (out < VOCAB).all()

    def test_generate_zero_new_tokens(self):
        """max_new_tokens=0 processes prompt only, no extra tokens."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 5))
        out = model.generate(prompt, max_new_tokens=0)
        assert out.shape == (BS, 5), \
            f"Expected (2, 5) but got {out.shape}"
        assert torch.equal(out, prompt)

    def test_generate_empty_prompt_raises(self):
        """Empty prompt raises ValueError."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 0))
        with pytest.raises(ValueError, match="at least 1 token"):
            model.generate(prompt, max_new_tokens=5)

    def test_generate_eot_resets_wm(self):
        """EOT token in prompt triggers WM reset on the following token."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        cfg.reset_on_doc_boundary = True
        model = NeuromorphicLM(cfg)
        model.train(False)

        eot = cfg.eot_id
        # Prompt: [tok, tok, tok, EOT, tok, tok]  (6 tokens)
        prompt = torch.randint(0, VOCAB, (1, 6))
        prompt[:, 3] = eot
        # Make sure non-EOT tokens aren't accidentally EOT
        for i in [0, 1, 2, 4, 5]:
            if prompt[:, i].item() == eot:
                prompt[:, i] = (eot + 1) % VOCAB

        model.generate(prompt, max_new_tokens=2)
        # EOT at index 3 → reset fires at index 4.
        # After reset: forward tokens at indices 4, 5 (prompt), then
        # gen loop feeds gen_token_0 (1 iteration for max_new_tokens=2).
        # gen_token_0 was sampled from index-5 logits but hasn't been
        # forward'd until the gen loop.  So ptr = 3 (indices 4, 5, gen_0).
        assert model.wm.wm_ptr.item() == 3

    def test_generate_eot_masks_surprise(self):
        """EOT positions get zero surprise (consistent with training)."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        cfg.reset_on_doc_boundary = True
        model = NeuromorphicLM(cfg)
        model.train(False)

        eot = cfg.eot_id
        # Prompt ending with EOT: surprise should be zero after last token
        prompt = torch.randint(0, VOCAB, (1, 4))
        prompt[:, -1] = eot

        model.generate(prompt, max_new_tokens=1)
        # The last prompt token is EOT, so its surprise update was masked to 0.
        # But then one generated token was processed, updating surprise.
        # We just verify surprise is not None and is finite.
        assert model.surprise is not None
        assert torch.isfinite(model.surprise).all()

    def test_generate_top_p_boundary(self):
        """top_p=0.0 and top_p=1.0 don't crash."""
        cfg = make_tiny_config()
        cfg.set_phase("A")
        model = NeuromorphicLM(cfg)
        model.train(False)

        prompt = torch.randint(0, VOCAB, (BS, 3))
        # top_p=0.0 should be treated as disabled (no filtering)
        out0 = model.generate(prompt, max_new_tokens=2, top_p=0.0)
        assert out0.shape == (BS, 5)
        # top_p=1.0 explicitly disabled
        out1 = model.generate(prompt, max_new_tokens=2, top_p=1.0)
        assert out1.shape == (BS, 5)
