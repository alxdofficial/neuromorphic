"""Unit tests for validation loop (v5)."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.model import NeuromorphicLM
from src.training.validation import evaluate_validation
from src.data.streaming import StreamBatch


BS = 2
VOCAB = 64


def _make_val_batches(config, n_batches=2):
    """Create fake validation batches."""
    T = config.T
    batches = []
    for _ in range(n_batches):
        batches.append(StreamBatch(
            input_ids=torch.randint(0, VOCAB, (BS, T)),
            target_ids=torch.randint(0, VOCAB, (BS, T)),
            prev_token=torch.randint(0, VOCAB, (BS,)),
        ))
    return iter(batches)


class TestValidation:
    def test_basic_validation(self):
        cfg = make_tiny_config(vocab_size=VOCAB)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        dataloader = _make_val_batches(cfg)
        result = evaluate_validation(
            model=model,
            dataloader=dataloader,
            config=cfg,
            device=torch.device("cpu"),
            num_steps=2,
        )

        assert "loss" in result
        assert "ppl" in result
        assert result["valid_tokens"] > 0
        assert result["loss"] > 0

    def test_validation_restores_state(self):
        """Validation should not corrupt model state."""
        cfg = make_tiny_config(vocab_size=VOCAB)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Forward a segment to set some state
        input_ids = torch.randint(0, VOCAB, (BS, cfg.N))
        model.forward_segment(input_ids)
        model.detach_states()

        # Snapshot PM state before validation
        W_pm_before = model.pm.W_pm.clone()

        dataloader = _make_val_batches(cfg)
        evaluate_validation(
            model=model,
            dataloader=dataloader,
            config=cfg,
            device=torch.device("cpu"),
            num_steps=1,
        )

        # PM state should be restored
        assert torch.equal(model.pm.W_pm, W_pm_before)

    def test_validation_ablation(self):
        """Should work with pm/em disabled."""
        cfg = make_tiny_config(vocab_size=VOCAB)
        model = NeuromorphicLM(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        dataloader = _make_val_batches(cfg)
        result = evaluate_validation(
            model=model,
            dataloader=dataloader,
            config=cfg,
            device=torch.device("cpu"),
            num_steps=1,
            pm_enabled=False,
            em_enabled=False,
        )

        assert result["valid_tokens"] > 0
        # Config should be restored
        assert cfg.pm_enabled
        assert cfg.em_enabled
