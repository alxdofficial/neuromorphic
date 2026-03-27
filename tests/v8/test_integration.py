"""Integration tests for v9.1 model (LM backprop + differentiable memory)."""

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


class TestForward:
    def test_shapes(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert result["aux_loss"].shape == ()

    def test_finite(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)
        assert torch.isfinite(result["logits"]).all()

    def test_no_memory(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, use_memory=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)

    def test_with_reset(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        result = model.forward_chunk(input_ids, reset_mask=reset_mask, has_reset=True)
        assert torch.isfinite(result["logits"]).all()

    def test_multiple_chunks(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        for _ in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids)
            assert torch.isfinite(result["logits"]).all()
            model.detach_states()

    def test_memory_persists(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.forward_chunk(input_ids)
        h = model.memory.h.clone()
        assert h.abs().sum() > 0

        input_ids2 = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.forward_chunk(input_ids2, reset_mask=reset_mask, has_reset=True)
        assert model.memory.h[0].abs().sum() > 0


class TestGradients:
    def test_lm_params_get_gradient(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1)
        ) + result["aux_loss"]
        loss.backward()

        no_grad = [n for n, p in model.lm.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert len(no_grad) == 0, f"LM params missing grad: {no_grad}"

    def test_mem_gate_gradient(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()
        assert model.lm.mem_gate.grad is not None

    def test_memory_params_get_gradient(self):
        """Memory graph params SHOULD get gradients (differentiable training)."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        assert model.memory.W_msg.grad is not None, "W_msg should have grad"
        assert model.memory.W1.grad is not None, "W1 should have grad"
        assert model.memory.readout_w.grad is not None, \
            "readout_w should have grad"

    def test_w_conn_gets_gradient(self):
        """Connection weights should get gradients through routing."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()
        assert model.memory.w_conn.grad is not None, "w_conn should have grad"


class TestParamCount:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        assert model.param_count() > 0
        assert model.memory_param_count() > 0
        assert model.lm_param_count() > 0
