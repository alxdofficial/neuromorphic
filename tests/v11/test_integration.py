"""Integration tests for v11 model (LM + cell memory)."""

import torch
import pytest
from src.v11.config import V11Config
from src.v11.model import V11Model

BS = 2
VOCAB = 64


def make_tiny(**kw):
    cfg = V11Config.tier_tiny(**kw)
    cfg.vocab_size = VOCAB
    cfg.validate()
    return cfg


class TestV11ModelForward:
    def test_forward_chunk_shapes(self):
        cfg = make_tiny()
        model = V11Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        input_ids = torch.randint(0, VOCAB, (BS, T))
        target_ids = torch.randint(0, VOCAB, (BS, T))
        result = model.forward_chunk(input_ids, target_ids=target_ids)

        assert result["logits"].shape == (BS, T, VOCAB)
        assert "aux_loss" in result
        assert "loss" in result

    def test_forward_no_memory(self):
        cfg = make_tiny()
        model = V11Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        input_ids = torch.randint(0, VOCAB, (BS, T))
        result = model.forward_chunk(input_ids, use_memory=False)
        assert result["logits"].shape == (BS, T, VOCAB)

    def test_gradient_flow_to_all_memory_params(self):
        cfg = make_tiny()
        model = V11Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        input_ids = torch.randint(0, VOCAB, (BS, T))
        result = model.forward_chunk(input_ids)
        result["logits"].sum().backward()

        mg = model.memory
        for name, p in mg.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.norm() > 0, f"{name} has zero grad"

    def test_param_counts(self):
        cfg = make_tiny()
        model = V11Model(cfg).float()
        lm = model.lm_param_count()
        mem = model.memory_param_count()
        total = model.param_count()
        assert total == lm + mem
        # Memory should be much smaller than LM
        assert mem < lm

    def test_multiple_segments(self):
        cfg = make_tiny()
        model = V11Model(cfg).float()
        model.initialize_states(BS)

        T = cfg.T
        for _ in range(3):
            ids = torch.randint(0, VOCAB, (BS, T))
            result = model.forward_chunk(ids)
            result["logits"].sum().backward()
            model.detach_states()
            model.zero_grad()

    def test_optimizer_param_groups_cover_all_params(self):
        cfg = make_tiny()
        model = V11Model(cfg).float()
        groups = model.optimizer_param_groups(lr=1e-3, weight_decay=0.1)

        grouped = set()
        for group in groups:
            for p in group["params"]:
                pid = id(p)
                assert pid not in grouped
                grouped.add(pid)

        expected = {
            id(p) for p in model.parameters() if p.requires_grad
        }
        assert grouped == expected

    def test_optimizer_param_groups_apply_mem_lr_scale(self):
        cfg = make_tiny(mem_lr_scale=0.25)
        model = V11Model(cfg).float()
        lr = 1e-3
        groups = model.optimizer_param_groups(lr=lr, weight_decay=0.1)

        lm_ids = {id(p) for p in model.lm.parameters() if p.requires_grad}
        mem_ids = {id(p) for p in model.memory.parameters() if p.requires_grad}

        for group in groups:
            params = {id(p) for p in group["params"]}
            if params & lm_ids:
                assert group["lr"] == pytest.approx(lr)
            if params & mem_ids:
                assert group["lr"] == pytest.approx(lr * cfg.mem_lr_scale)
