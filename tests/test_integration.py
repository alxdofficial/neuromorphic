"""End-to-end integration tests (dense-W design)."""

import torch
import pytest
import torch.nn.functional as F

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
        assert result["logits"].shape == (BS, T, config.vocab_size)
        result["loss"].backward()

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

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        result["loss"].backward()
        model.detach_states()

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

        has_carry = any(c is not None and c.abs().sum() > 0
                        for c in model.lm._carries)
        assert has_carry
        assert model.memory.h.abs().sum() > 0


class TestPersistence:
    def test_memory_persists_across_chunks(self):
        """W and h should be nonzero after multiple chunks."""
        config = _tiny_config()
        model = Model(config)
        BS, T = 2, config.T
        input_ids = torch.randint(0, config.vocab_size, (BS, T))

        for _ in range(3):
            model.forward_chunk(input_ids)
            model.detach_states()

        assert model.memory.W.abs().sum() > 0, "W should be nonzero"
        assert model.memory.h.abs().sum() > 0, "h should be nonzero"


class TestLearningSignal:
    def test_detached_memory_gradients_depend_on_history(self):
        config = _tiny_config()
        torch.manual_seed(0)
        base_model = Model(config).float()
        base_model.train(False)

        BS, T = 1, config.T
        last_token = torch.tensor([[7]])
        last_target = torch.tensor([[11]])
        hist_a = torch.randint(3, config.vocab_size, (BS, T - 1))
        hist_b = torch.randint(3, config.vocab_size, (BS, T - 1))
        seq_a = torch.cat([hist_a, last_token], dim=1)
        seq_b = torch.cat([hist_b, last_token], dim=1)
        tgt_a = torch.randint(0, config.vocab_size, (BS, T))
        tgt_b = torch.randint(0, config.vocab_size, (BS, T))
        tgt_a[:, -1] = last_target
        tgt_b[:, -1] = last_target

        def grad_for(seq, tgt):
            model = Model(config).float()
            model.load_state_dict(base_model.state_dict())
            model.train(False)
            out = model.forward_chunk(seq, target_ids=tgt)
            loss = F.cross_entropy(out["logits"][:, -1], tgt[:, -1])
            loss.backward()
            return model.memory.msg_w1.grad.clone()

        grad_a = grad_for(seq_a, tgt_a)
        grad_b = grad_for(seq_b, tgt_b)
        diff = (grad_a - grad_b).abs().max().item()
        assert diff > 1e-6, (
            "Detached recurrent state no longer produces history-dependent gradients.")
