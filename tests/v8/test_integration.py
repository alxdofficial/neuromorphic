"""Integration tests for the full v8/v9 model."""

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
        assert result["surprise"].shape == (BS, cfg.T, cfg.D)

    def test_forward_chunk_finite(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        assert torch.isfinite(result["logits"]).all()

    def test_forward_no_memory(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, use_memory=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)

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

    def test_memory_persists_across_doc_boundaries(self):
        """Memory graph should NOT reset at document boundaries."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.forward_chunk(input_ids)
        h_before = model.memory.h.clone()
        assert h_before.abs().sum() > 0

        input_ids2 = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.forward_chunk(input_ids2, reset_mask=reset_mask, has_reset=True)

        h_after = model.memory.h
        assert h_after[0].abs().sum() > 0
        assert h_after[1].abs().sum() > 0

    def test_multiple_chunks(self):
        """Model should handle multiple sequential chunks."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        for _ in range(3):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids)
            assert torch.isfinite(result["logits"]).all()
            model.detach_states()


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

        no_grad = []
        for name, param in model.lm.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)
        assert len(no_grad) == 0, f"LM params with no gradient: {no_grad}"

    def test_memory_params_get_gradient(self):
        """Memory graph params should get gradients (differentiable forward)."""
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1)
        ) + result["aux_loss"]
        loss.backward()

        # Primitives and key should have gradients
        assert model.memory.primitives.grad is not None, \
            "primitives should get gradient"
        assert model.memory.primitives.grad.abs().sum() > 0, \
            "primitives gradient should be nonzero"
        assert model.memory.key.grad is not None, \
            "key should get gradient"
        assert model.memory.decay_logit.grad is not None, \
            "decay_logit should get gradient"

    def test_modulator_params_get_gradient(self):
        """Per-neuron modulator params should get gradients."""
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1)
        ) + result["aux_loss"]
        loss.backward()

        assert model.memory.fc1_w.grad is not None, \
            "modulator fc1_w should get gradient"
        assert model.memory.mod_lr_logit.grad is not None, \
            "mod_lr_logit should get gradient"

    def test_dendritic_params_get_gradient(self):
        """Dendritic FC params should get gradients."""
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1)
        ) + result["aux_loss"]
        loss.backward()

        if model.memory.use_dendritic_tree:
            assert model.memory.dendrite_branch_w.grad is not None, \
                "dendrite_branch_w should get gradient"

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


class TestV8ParamCount:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        count = model.param_count()
        assert count > 0
        assert isinstance(count, int)

    def test_memory_has_params(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        mem_count = model.memory_param_count()
        assert mem_count > 0, "Memory graph should have learnable params"
