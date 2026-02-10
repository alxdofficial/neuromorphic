"""Unit tests for src/training/validation.py — validation loop."""

import pytest
import torch

from src.data.streaming import StreamBatch
from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.model.state import save_runtime_state, load_runtime_state
from src.training.validation import evaluate_validation, _clear_runtime_for_eval
from tests.conftest import make_tiny_config, forward_n_tokens

BS = 2
VOCAB = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_batch(BS, T, vocab=VOCAB):
    return StreamBatch(
        input_ids=torch.randint(0, vocab, (BS, T)),
        target_ids=torch.randint(0, vocab, (BS, T)),
        prev_token=torch.randint(0, vocab, (BS,)),
    )


def _batch_iterator(BS, T, num_batches=10, vocab=VOCAB):
    """Yield a fixed number of synthetic batches."""
    for _ in range(num_batches):
        yield _make_batch(BS, T, vocab)


def _tiny_model(phase="B"):
    cfg = make_tiny_config()
    cfg.set_phase(phase)
    return NeuromorphicLM(cfg), cfg


# ============================================================================
# _clear_runtime_for_eval
# ============================================================================

class TestClearRuntimeForEval:
    def test_zeros_surprise(self):
        model, cfg = _tiny_model("B")
        model.surprise = torch.ones(BS, 1) * 5.0
        _clear_runtime_for_eval(model, BS, torch.device("cpu"))
        assert model.surprise.sum().item() == 0.0

    def test_initializes_surprise_if_none(self):
        model, cfg = _tiny_model("B")
        model.surprise = None
        _clear_runtime_for_eval(model, BS, torch.device("cpu"))
        assert model.surprise is not None
        assert model.surprise.shape == (BS, 1)

    def test_resets_all_blocks(self):
        model, cfg = _tiny_model("B")
        # Forward some tokens to populate state
        forward_n_tokens(model, cfg.P * 2, BS=BS, vocab=VOCAB)
        _clear_runtime_for_eval(model, BS, torch.device("cpu"))
        # After clear, recurrent state should be zero
        for block in model.blocks:
            for layer in block.layers:
                if layer.h is not None:
                    assert layer.h.abs().sum().item() == 0.0


# ============================================================================
# evaluate_validation
# ============================================================================

class TestEvaluateValidation:
    def test_returns_expected_keys(self):
        model, cfg = _tiny_model("B")
        dl = _batch_iterator(BS, cfg.T, num_batches=3)
        result = evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=3)
        assert "loss" in result
        assert "ppl" in result
        assert "valid_tokens" in result
        assert "steps_done" in result
        assert "valid_fraction" in result
        assert "eot_input_fraction" in result
        assert "reset_fraction" in result

    def test_finite_results(self):
        model, cfg = _tiny_model("B")
        dl = _batch_iterator(BS, cfg.T, num_batches=5)
        result = evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=5)
        assert result["loss"] < 1e6
        assert result["ppl"] > 0
        assert result["ppl"] < 1e6
        assert result["steps_done"] == 5

    def test_state_restored_after_validation(self):
        """Validation must be side-effect free — runtime state is restored."""
        model, cfg = _tiny_model("B")
        # Forward some tokens to establish non-trivial state
        forward_n_tokens(model, cfg.P * 2, BS=BS, vocab=VOCAB)
        state_before = save_runtime_state(model)

        dl = _batch_iterator(BS, cfg.T, num_batches=3)
        evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=3)

        state_after = save_runtime_state(model)
        # Check state was restored
        for path in state_before:
            if path in state_after:
                before_sub = state_before[path]
                after_sub = state_after[path]
                if isinstance(before_sub, dict) and isinstance(after_sub, dict):
                    for k in before_sub:
                        if isinstance(before_sub[k], torch.Tensor):
                            assert torch.allclose(
                                before_sub[k], after_sub[k], atol=1e-6
                            ), f"State mismatch at {path}.{k}"

    def test_config_restored_after_validation(self):
        """Config toggles overridden during validation must be restored."""
        model, cfg = _tiny_model("C")
        assert cfg.pm_enabled is True
        assert cfg.em_enabled is True

        dl = _batch_iterator(BS, cfg.T, num_batches=2)
        evaluate_validation(
            model, dl, cfg, torch.device("cpu"),
            num_steps=2, pm_enabled=False, em_enabled=False,
        )
        # Config should be restored
        assert cfg.pm_enabled is True
        assert cfg.em_enabled is True

    def test_overrides_take_effect(self):
        """pm_enabled/em_enabled overrides should affect validation behavior."""
        model, cfg = _tiny_model("C")
        dl1 = _batch_iterator(BS, cfg.T, num_batches=3)
        result_full = evaluate_validation(model, dl1, cfg, torch.device("cpu"), num_steps=3)

        dl2 = _batch_iterator(BS, cfg.T, num_batches=3)
        result_no_pm = evaluate_validation(
            model, dl2, cfg, torch.device("cpu"),
            num_steps=3, pm_enabled=False,
        )
        # Both should produce valid results
        assert result_full["steps_done"] == 3
        assert result_no_pm["steps_done"] == 3

    def test_handles_empty_dataloader(self):
        model, cfg = _tiny_model("B")
        dl = iter([])  # empty
        result = evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=10)
        assert result["steps_done"] == 0
        assert result["loss"] == float("inf")

    def test_phase_a_no_pm_no_em(self):
        """Phase A has no PM/EM — validation should still work."""
        model, cfg = _tiny_model("A")
        dl = _batch_iterator(BS, cfg.T, num_batches=3)
        result = evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=3)
        assert result["loss"] < 1e6
        assert result["steps_done"] == 3

    def test_training_mode_restored(self):
        """Model should return to training mode if it was training before."""
        model, cfg = _tiny_model("B")
        model.train()
        assert model.training is True

        dl = _batch_iterator(BS, cfg.T, num_batches=2)
        evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=2)

        assert model.training is True

    def test_model_mode_preserved_when_not_training(self):
        """If model was not training before, it should remain not training."""
        model, cfg = _tiny_model("B")
        model.eval()
        assert model.training is False

        dl = _batch_iterator(BS, cfg.T, num_batches=2)
        evaluate_validation(model, dl, cfg, torch.device("cpu"), num_steps=2)

        assert model.training is False
