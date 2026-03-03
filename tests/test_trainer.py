"""Unit tests for TBPTTTrainer (v5)."""

import pytest
import torch
from tests.conftest import make_tiny_config

from src.model.model import NeuromorphicLM
from src.training.trainer import TBPTTTrainer
from src.data.streaming import StreamBatch


BS = 2
VOCAB = 64


def _make_batch(config, eot_id=2):
    """Create a fake StreamBatch for testing."""
    T = config.T
    input_ids = torch.randint(0, VOCAB, (BS, T))
    target_ids = torch.randint(0, VOCAB, (BS, T))
    prev_token = torch.randint(0, VOCAB, (BS,))
    return StreamBatch(input_ids=input_ids, target_ids=target_ids, prev_token=prev_token)


def _make_trainer(config=None, **kwargs):
    """Create a trainer with default config."""
    if config is None:
        config = make_tiny_config(vocab_size=VOCAB, **kwargs)
    model = NeuromorphicLM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = iter([])  # empty, we'll call train_chunk directly

    return TBPTTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloader=dataloader,
        config=config,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
        log_interval=50,
        fail_fast=True,
    )


class TestTrainChunk:
    def test_basic_forward_backward(self):
        trainer = _make_trainer()
        batch = _make_batch(trainer.config)
        metrics = trainer.train_chunk(batch)

        assert "loss" in metrics
        assert "ppl" in metrics
        assert "valid_tokens" in metrics
        assert metrics["loss"] > 0
        assert metrics["valid_tokens"] > 0

    def test_gradient_update(self):
        trainer = _make_trainer()
        batch = _make_batch(trainer.config)

        # Save params before
        p_before = trainer.model.embedding.weight.data.clone()

        trainer.train_chunk(batch)

        # Params should have changed
        p_after = trainer.model.embedding.weight.data
        assert not torch.allclose(p_before, p_after)

    def test_states_detached_after_chunk(self):
        trainer = _make_trainer()
        batch = _make_batch(trainer.config)
        trainer.train_chunk(batch)

        if trainer.model.pm.W_pm is not None:
            assert trainer.model.pm.W_pm.grad_fn is None

    def test_multi_segment_chunk(self):
        """Chunk should process K_segments * N tokens."""
        cfg = make_tiny_config(vocab_size=VOCAB, K_segments=3, N=8)
        trainer = _make_trainer(config=cfg)
        batch = _make_batch(cfg)
        metrics = trainer.train_chunk(batch)

        assert metrics["valid_tokens"] > 0

    def test_doc_boundary_handling(self):
        """EOT in prev_token should trigger reset."""
        cfg = make_tiny_config(vocab_size=VOCAB, eot_id=2)
        trainer = _make_trainer(config=cfg)
        batch = _make_batch(cfg, eot_id=2)
        # Force prev_token to eot
        batch = StreamBatch(
            input_ids=batch.input_ids,
            target_ids=batch.target_ids,
            prev_token=torch.full((BS,), 2),  # eot
        )
        metrics = trainer.train_chunk(batch)
        assert metrics["loss"] > 0

    def test_pcm_enabled(self):
        cfg = make_tiny_config(vocab_size=VOCAB, pcm_enabled=True)
        trainer = _make_trainer(config=cfg)
        batch = _make_batch(cfg)
        metrics = trainer.train_chunk(batch)
        assert metrics["loss"] > 0

    def test_override_prev_token(self):
        trainer = _make_trainer()
        trainer.override_prev_token = torch.randint(0, VOCAB, (BS,))
        batch = _make_batch(trainer.config)
        metrics = trainer.train_chunk(batch)
        assert metrics["loss"] > 0
        assert trainer.override_prev_token is None  # consumed
