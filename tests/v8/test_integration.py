"""Integration tests for v9-backprop model."""

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
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS)

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert torch.isfinite(result["logits"]).all()

    def test_forward_no_memory(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS)

        result = model.forward_chunk(input_ids, target_ids=target_ids,
                                     use_memory=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)

    def test_forward_with_reset(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.initialize_states(BS)

        result = model.forward_chunk(
            input_ids, target_ids=target_ids, reset_mask=reset_mask,
            has_reset=True)
        assert torch.isfinite(result["logits"]).all()

    def test_multiple_chunks(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS)

        for _ in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            model.detach_states()
            assert torch.isfinite(result["logits"]).all()


class TestGradientFlow:
    def test_loss_backward_runs(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

    def test_memory_params_get_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        mg = model.memory
        # Modulator
        assert mg.mod_w1.grad is not None, "mod_w1 should have grad"
        assert mg.mod_w1.grad.norm() > 0, "mod_w1 grad should be nonzero"
        assert mg.mod_w2.grad is not None, "mod_w2 should have grad"

        # State MLP
        assert mg.state_w1.grad is not None, "state_w1 should have grad"
        assert mg.state_w1.grad.norm() > 0, "state_w1 grad should be nonzero"

        # Message MLP
        assert mg.msg_w1.grad is not None, "msg_w1 should have grad"
        assert mg.msg_w1.grad.norm() > 0, "msg_w1 grad should be nonzero"

        # Neuron ID
        assert mg.neuron_id.grad is not None, "neuron_id should have grad"
        assert mg.neuron_id.grad.norm() > 0, "neuron_id grad should be nonzero"

        # Dendritic weights
        if mg.use_dendritic_tree:
            assert mg.dendrite_branch_w.grad is not None
            assert mg.dendrite_group_w.grad is not None

    def test_lm_params_get_grad(self):
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        assert any(p.grad is not None for p in model.lm.mem_mlp.parameters())
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
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        if model.lm.split_mlp is not None:
            for p in model.lm.split_mlp.parameters():
                if p.requires_grad:
                    assert p.grad is not None, "split_mlp param should have grad"

    def test_mem_mlp_gets_grad(self):
        """mem_mlp should get gradients from memory path."""
        cfg = make_tiny()
        model = V8Model(cfg).float()
        model.initialize_states(BS)

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        import torch.nn.functional as F
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        for name, p in model.lm.mem_mlp.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"mem_mlp.{name} should have grad"
                assert p.grad.norm() > 0, f"mem_mlp.{name} grad should be nonzero"


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
        # Memory params should include modulator + state_mlp + msg_mlp + neuron_id + dendrites
        assert mem > cfg.N_neurons * cfg.neuromod_hidden
