"""Integration tests for v10-gnn model."""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v10.config import V10Config
from src.v10.model import V10Model

BS = 2
VOCAB = 64


def make_tiny(**overrides):
    cfg = V10Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


class TestV10ModelForward:
    def test_forward_chunk_shapes(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        model.initialize_states(BS)
        ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(ids)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert torch.isfinite(result["logits"]).all()

    def test_multiple_chunks(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        model.initialize_states(BS)
        for _ in range(3):
            ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(ids)
            model.detach_states()
            assert torch.isfinite(result["logits"]).all()


class TestGradientFlow:
    def test_loss_backward(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        model.initialize_states(BS)

        ids = torch.randint(0, VOCAB, (BS, cfg.T))
        tgt = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(ids, target_ids=tgt)

        loss = torch.nn.functional.cross_entropy(
            result["logits"].reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()

    def test_memory_params_get_grad(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        model.initialize_states(BS)

        ids = torch.randint(0, VOCAB, (BS, cfg.T))
        tgt = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(ids, target_ids=tgt)

        loss = torch.nn.functional.cross_entropy(
            result["logits"].reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()

        mg = model.memory
        # Shared MLPs should get gradients
        for name, p in mg.neuron_step.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"neuron_step.{name} has no grad"
                assert p.grad.norm() > 0, f"neuron_step.{name} grad is zero"

        # Identity should get gradient
        assert mg.identity.grad is not None
        assert mg.identity.grad.norm() > 0

    def test_decoder_gets_grad(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        model.initialize_states(BS)

        ids = torch.randint(0, VOCAB, (BS, cfg.T))
        tgt = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(ids, target_ids=tgt)

        loss = torch.nn.functional.cross_entropy(
            result["logits"].reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()

        for name, p in model.decoder.named_parameters():
            if p.requires_grad and 'lm_head' not in name:
                assert p.grad is not None, f"decoder.{name} has no grad"

    def test_lower_scan_gets_grad(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        model.initialize_states(BS)

        ids = torch.randint(0, VOCAB, (BS, cfg.T))
        tgt = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(ids, target_ids=tgt)

        loss = torch.nn.functional.cross_entropy(
            result["logits"].reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()

        for layer in model.lm.layers:
            has_grad = any(p.grad is not None and p.grad.norm() > 0
                          for p in layer.parameters())
            assert has_grad, "Lower scan layer should have grad"


class TestParamCounts:
    def test_param_count(self):
        cfg = make_tiny()
        model = V10Model(cfg).float()
        total = model.param_count()
        assert total > 0
        print(f"Tiny model: {total:,} params")
        print(f"  LM: {model.lm_param_count():,}")
        print(f"  Memory: {model.memory_param_count():,}")
        print(f"  Decoder: {model.decoder_param_count():,}")
