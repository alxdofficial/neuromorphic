"""End-to-end integration tests."""

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

    def test_rewire_scheduler_counts_processed_tokens(self):
        config = _tiny_config(structural_plasticity=True, plasticity_interval=20)
        model = Model(config)
        BS, T = 2, config.T
        input_ids = torch.randint(0, config.vocab_size, (BS, T))

        calls = []

        def fake_rewire():
            calls.append(model._tokens_since_rewire)

        model.memory.rewire_connections = fake_rewire

        model.forward_chunk(input_ids)
        model.detach_states()
        assert calls == []
        assert model._tokens_since_rewire == BS * T

        model.forward_chunk(input_ids)
        model.detach_states()
        assert len(calls) == 1
        assert calls[0] == 2 * BS * T
        assert model._tokens_since_rewire == (2 * BS * T - config.plasticity_interval)

    def test_no_memory_chunks_do_not_advance_rewire_scheduler(self):
        config = _tiny_config(structural_plasticity=True, plasticity_interval=4)
        model = Model(config)
        BS, T = 2, config.T
        input_ids = torch.randint(0, config.vocab_size, (BS, T))

        calls = []
        model.memory.rewire_connections = lambda: calls.append(True)

        model.forward_chunk(input_ids, use_memory=False)
        model.detach_states()

        assert calls == []
        assert model._tokens_since_rewire == 0


class TestPersistence:
    def test_future_eos_does_not_change_past_logits(self):
        config = _tiny_config()
        torch.manual_seed(0)
        model_a = Model(config).float()
        torch.manual_seed(0)
        model_b = Model(config).float()
        model_b.load_state_dict(model_a.state_dict())
        model_a.train(False)
        model_b.train(False)

        BS, T = 2, config.T
        history = torch.randint(3, config.vocab_size, (BS, T))
        with torch.no_grad():
            model_a.forward_chunk(history, use_memory=False)
            model_b.forward_chunk(history, use_memory=False)

        tokens_base = torch.randint(3, config.vocab_size, (BS, T))
        tokens_eos = tokens_base.clone()
        mid = T // 2
        tokens_eos[:, mid] = config.eot_id

        with torch.no_grad():
            logits_base = model_a.forward_chunk(tokens_base, use_memory=False)["logits"]
            logits_eos = model_b.forward_chunk(tokens_eos, use_memory=False)["logits"]

        diff = (logits_base[:, :mid] - logits_eos[:, :mid]).abs().max().item()
        assert diff < 1e-5, (
            f"Future EOS changed earlier logits (max diff={diff:.6f}); "
            "model is still eagerly resetting state.")

    def test_previous_eos_does_not_reset_next_chunk(self):
        config = _tiny_config()
        torch.manual_seed(0)
        model_hist = Model(config).float()
        torch.manual_seed(0)
        model_fresh = Model(config).float()
        model_fresh.load_state_dict(model_hist.state_dict())
        model_hist.train(False)
        model_fresh.train(False)

        BS, T = 2, config.T
        first_chunk = torch.randint(3, config.vocab_size, (BS, T))
        first_chunk[:, -1] = config.eot_id
        second_chunk = torch.randint(3, config.vocab_size, (BS, T))

        with torch.no_grad():
            model_hist.forward_chunk(first_chunk, use_memory=False)
            model_hist.detach_states()
            logits_hist = model_hist.forward_chunk(second_chunk, use_memory=False)["logits"]
            logits_fresh = model_fresh.forward_chunk(second_chunk, use_memory=False)["logits"]

        diff = (logits_hist - logits_fresh).abs().max().item()
        assert diff > 1e-4, (
            "Previous chunk ending in EOS unexpectedly reset the next chunk state.")


class TestLearningSignal:
    def test_detached_memory_gradients_still_depend_on_history(self):
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
