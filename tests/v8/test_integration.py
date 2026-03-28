"""Integration tests for v10 model."""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config
from src.v8.model import V8Model

BS = 2
VOCAB = 64


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


class TestV8ModelForward:
    def test_forward_chunk_shapes(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS)

        result = model.forward_chunk(input_ids)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert torch.isfinite(result["logits"]).all()

    def test_forward_no_memory(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS)

        result = model.forward_chunk(input_ids, use_memory=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)

    def test_forward_with_reset(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.initialize_states(BS)

        result = model.forward_chunk(
            input_ids, reset_mask=reset_mask, has_reset=True)
        assert torch.isfinite(result["logits"]).all()

    def test_multiple_chunks(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS)

        for _ in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids)
            model.detach_states()
            assert torch.isfinite(result["logits"]).all()


class TestGradientFlow:
    def test_loss_backward_runs(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

    def test_modulator_gets_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        mg = model.memory
        assert mg.mod_w1.grad is not None, "mod_w1 should have grad"
        assert mg.mod_w1.grad.norm() > 0, "mod_w1 grad should be nonzero"
        assert mg.mod_w2.grad is not None, "mod_w2 should have grad"

    def test_lm_params_get_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        assert model.lm.mem_gate.grad is not None
        for i, layer in enumerate(model.lm.layers):
            has_grad = any(p.grad is not None and p.grad.norm() > 0
                          for p in layer.parameters())
            assert has_grad, f"Layer {i} should have grad"

    def test_split_mlp_gets_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        if model.lm.split_mlp is not None:
            for p in model.lm.split_mlp.parameters():
                if p.requires_grad:
                    assert p.grad is not None, "split_mlp param should have grad"

    def test_mem_gate_moves(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        gate_before = model.lm.mem_gate.data.clone()

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gate_after = model.lm.mem_gate.data
        assert not torch.equal(gate_before, gate_after), \
            "mem_gate should change after optimizer step"


class TestParamCounts:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        total = model.param_count()
        lm = model.lm_param_count()
        mem = model.memory_param_count()
        assert total > 0
        assert lm > 0
        assert mem > 0
        # Memory params = modulator only
        assert mem == sum(p.numel() for p in model.memory.parameters())
