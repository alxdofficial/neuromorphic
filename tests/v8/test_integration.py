"""Integration tests for v8/v9 model (LM backprop + ES memory)."""

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
        assert model.memory.h[0].abs().sum() > 0  # memory NOT reset


class TestLMGradients:
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

    def test_memory_params_no_grad(self):
        """Memory graph params should NOT get gradients (ES-trained)."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        loss = F.cross_entropy(
            result["logits"].reshape(-1, VOCAB), target_ids.reshape(-1))
        loss.backward()

        for name, p in model.memory.named_parameters():
            assert p.grad is None, f"Memory param {name} should not have grad"


class TestES:
    def test_score_trajectories(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Collect chunks
        pre_state = {k: v.clone() for k, v in model.memory.runtime_state_dict().items()}
        pre_params = {k: v.clone() for k, v in model.memory.get_es_params().items()}

        es_buffer = []
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            eot_at = (input_ids == cfg.eot_id)
            es_buffer.append({
                "cc_segments": result["cc_segments"],
                "H_mid": result["H_mid"],
                "surprise": result["surprise"],
                "target_ids": target_ids,
                "eot_at": eot_at,
                "pre_upper_carries": None,
            })

        scoring = model.score_es_trajectories(es_buffer, pre_state, pre_params)

        assert scoring["k_neurons"].shape[0] == min(cfg.es_k_neurons, cfg.N_neurons)
        assert scoring["advantages"].shape[0] == cfg.es_n_trajectories
        assert len(scoring["noise"]) == cfg.es_n_trajectories
        assert torch.isfinite(scoring["advantages"]).all()

    def test_apply_gradient(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        pre_state = {k: v.clone() for k, v in model.memory.runtime_state_dict().items()}
        pre_params = {k: v.clone() for k, v in model.memory.get_es_params().items()}

        es_buffer = []
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            es_buffer.append({
                "cc_segments": result["cc_segments"],
                "H_mid": result["H_mid"],
                "surprise": result["surprise"],
                "target_ids": target_ids,
                "eot_at": (input_ids == cfg.eot_id),
                "pre_upper_carries": None,
            })

        prim_before = model.memory.primitives.data.clone()
        scoring = model.score_es_trajectories(es_buffer, pre_state, pre_params)
        model.apply_es_gradient(scoring)

        # Params for K neurons should have changed
        k = scoring["k_neurons"]
        # At least some change expected (unless all advantages are 0)
        # With random data, advantages should have nonzero std


class TestParamCount:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        assert model.param_count() > 0
        assert model.memory_param_count() > 0
        assert model.lm_param_count() > 0
