"""Integration tests for the full v8 model."""

import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.v8.config import V8Config
from src.v8.model import V8Model


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


BS = 2
VOCAB = 64


class TestV8ModelForward:
    def test_forward_chunk_shapes(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert result["aux_loss"].shape == ()
        assert result["surprise"].shape == (BS, cfg.T, cfg.C, cfg.D_cc)

    def test_forward_chunk_finite(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert torch.isfinite(result["logits"]).all()

    def test_forward_without_targets(self):
        """Should work without target_ids (no PPO rewards computed)."""
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, collect_ppo=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert result["ppo_buffer"] is None

    def test_forward_with_reset(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(
            input_ids, target_ids=target_ids, reset_mask=reset_mask
        )
        assert torch.isfinite(result["logits"]).all()

    def test_ppo_buffer_populated(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        buf = result["ppo_buffer"]
        assert buf is not None
        expected_actions = cfg.T // cfg.action_every
        assert buf.step == expected_actions
        assert buf.advantages is not None  # GAE should have been computed


class TestV8Gradients:
    def test_lm_params_get_gradient(self):
        cfg = make_tiny(pcm_enabled=True)
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        logits = result["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), target_ids.reshape(-1)
        ) + result["aux_loss"]
        loss.backward()

        # Check all LM params have gradients (no projections to exclude now)
        no_grad = []
        for name, param in model.lm.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)
        assert len(no_grad) == 0, f"LM params with no gradient: {no_grad}"

    def test_scan_layer_gradients(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1)
        )
        loss.backward()

        for i, layer in enumerate(model.lm.layers):
            assert layer.proj_in.weight.grad is not None, \
                f"No gradient for layers[{i}].proj_in"
            assert layer.proj_in.weight.grad.abs().sum() > 0, \
                f"Zero gradient for layers[{i}].proj_in"

    def test_mem_gate_gradient(self):
        """Memory gate should get gradient through post-memory scan."""
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1)
        )
        loss.backward()

        assert model.lm.mem_gate.grad is not None

    def test_memory_not_in_autograd(self):
        """Memory graph tensors should not require grad."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        assert not model.memory.primitives.requires_grad
        assert not model.memory.thresholds.requires_grad


class TestV8ParamCount:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        count = model.param_count()
        assert count > 0
        assert isinstance(count, int)
