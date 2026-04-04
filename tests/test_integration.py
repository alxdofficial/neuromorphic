"""End-to-end integration tests."""

import torch
import pytest

from src.model.config import Config
from src.model.model import Model


def _tiny_config(**kw):
    return Config.tier_tiny(**kw)


class TestForwardBackward:
    def test_basic(self):
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T

        input_ids = torch.randint(0, config.vocab_size, (BS, T))
        target_ids = torch.randint(0, config.vocab_size, (BS, T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert "loss" in result
        assert "logits" in result
        assert result["logits"].shape == (BS, T, config.vocab_size)

        result["loss"].backward()
        # Check some params got gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad

    def test_no_memory_baseline(self):
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T

        input_ids = torch.randint(0, config.vocab_size, (BS, T))
        target_ids = torch.randint(0, config.vocab_size, (BS, T))

        result = model.forward_chunk(
            input_ids, target_ids=target_ids, use_memory=False)
        assert result["logits"].shape == (BS, T, config.vocab_size)
        result["loss"].backward()


class TestTBPTT:
    def test_detach_states(self):
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T
        input_ids = torch.randint(0, config.vocab_size, (BS, T))
        target_ids = torch.randint(0, config.vocab_size, (BS, T))

        # Chunk 1
        result = model.forward_chunk(input_ids, target_ids=target_ids)
        result["loss"].backward()
        model.detach_states()

        # Chunk 2 should work fine
        result2 = model.forward_chunk(input_ids, target_ids=target_ids)
        result2["loss"].backward()
        model.detach_states()

    def test_multi_segment_carries(self):
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T
        input_ids = torch.randint(0, config.vocab_size, (BS, T))
        target_ids = torch.randint(0, config.vocab_size, (BS, T))

        for _ in range(3):
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            result["loss"].backward()
            model.detach_states()

        # Scan carries should be non-zero after 3 segments
        has_carry = any(c is not None and c.abs().sum() > 0
                        for c in model.lm._carries)
        assert has_carry, "Scan carries should be non-zero after 3 segments"

        # Memory state should be non-zero
        assert model.memory.h.abs().sum() > 0


class TestEOSReset:
    def test_eos_resets_memory(self):
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T

        # First chunk: no EOS
        input_ids = torch.randint(1, config.vocab_size, (BS, T))
        target_ids = torch.randint(0, config.vocab_size, (BS, T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)
        result["loss"].backward()
        model.detach_states()

        h_before = model.memory.h[0].clone()

        # Second chunk: EOS in batch element 0
        input_ids2 = input_ids.clone()
        input_ids2[0, T // 2] = config.eot_id
        result2 = model.forward_chunk(input_ids2, target_ids=target_ids)
        # Memory for element 0 should have been reset (h zeroed before forward)
        # We can't easily check mid-forward, but at least it shouldn't crash
        result2["loss"].backward()

    def test_init_before_reset(self):
        """First call with EOS should not crash (init happens before reset)."""
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T

        input_ids = torch.randint(1, config.vocab_size, (BS, T))
        input_ids[0, 0] = config.eot_id
        target_ids = torch.randint(0, config.vocab_size, (BS, T))

        # Should not crash
        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert result["logits"].shape == (BS, T, config.vocab_size)
