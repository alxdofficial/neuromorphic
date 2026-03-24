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

    def test_forward_no_memory(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, use_memory=False)
        assert result["logits"].shape == (BS, cfg.T, VOCAB)
        assert result["rl_data"] is None

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

    def test_rl_data_populated(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.initialize_states(BS, torch.device("cpu"))

        result = model.forward_chunk(input_ids, target_ids=target_ids)
        rl = result["rl_data"]
        assert rl is not None
        n_segments = cfg.T // cfg.action_every
        assert len(rl["actions"]) == n_segments
        assert len(rl["obs"]) == n_segments
        # seg_rewards/seg_losses are now computed by the trainer, not forward_chunk
        assert rl["n_segments"] == n_segments
        assert rl["action_every"] == cfg.action_every
        assert rl["eot_at"].shape == (BS, cfg.T)

    def test_compute_rl_advantages(self):
        """compute_rl_advantages should produce valid advantages from collected chunks."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        n_segments = cfg.T // cfg.action_every

        # Collect 2 chunks, simulating trainer's CE → reward derivation
        collected = []
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            # Derive rewards from CE (same as trainer does)
            logits = result["logits"]
            with torch.no_grad():
                ce = F.cross_entropy(
                    logits.detach().reshape(-1, VOCAB),
                    target_ids.reshape(-1), reduction='none',
                ).reshape(BS, cfg.T)
                mask = (~rl["eot_at"]).float()
                seg_ce = ce.view(BS, n_segments, cfg.action_every)
                seg_mask = mask.view(BS, n_segments, cfg.action_every)
                seg_losses = (seg_ce * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
                rl["seg_rewards"] = -seg_losses
                rl["seg_losses"] = seg_losses
                rl["loss"] = ce.mean().item()
            collected.append(rl)

        combined = model.compute_rl_advantages(collected)
        total_seg = n_segments * 2
        assert combined["advantages"].shape == (BS, total_seg)
        assert combined["returns"].shape == (BS, total_seg)
        assert len(combined["obs"]) == total_seg
        assert len(combined["actions"]) == total_seg
        assert combined["global_obs"].shape[0] == total_seg
        # Advantages should be finite
        assert torch.isfinite(combined["advantages"]).all()

    def test_replay_produces_gradients(self):
        """Replay should produce gradients for neuromod (policy + value)."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Collect and compute advantages (simulate trainer's CE → reward step)
        collected = []
        n_segments = cfg.T // cfg.action_every
        for _ in range(cfg.rl_collect_chunks):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            with torch.no_grad():
                ce = F.cross_entropy(
                    result["logits"].detach().reshape(-1, VOCAB),
                    target_ids.reshape(-1), reduction='none',
                ).reshape(BS, cfg.T)
                mask = (~rl["eot_at"]).float()
                seg_ce = ce.view(BS, n_segments, cfg.action_every)
                seg_mask = mask.view(BS, n_segments, cfg.action_every)
                seg_losses = (seg_ce * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
                rl["seg_rewards"] = -seg_losses
                rl["seg_losses"] = seg_losses
                rl["loss"] = ce.mean().item()
            collected.append(rl)

        combined = model.compute_rl_advantages(collected)
        model.replay_for_neuromod_grads(combined, amp_enabled=False)

        # Action heads should have gradients (policy gradient)
        # Note: backbone gets zero grad on first step due to zero-init heads,
        # self-corrects after first optimizer step.
        has_policy_grad = False
        for name, p in model.neuromod.named_parameters():
            if "head" in name or "logstd" in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_policy_grad = True
                    break
        assert has_policy_grad, "Policy heads should have gradients after replay"

        # Value function should have gradients
        has_value_grad = False
        for p in model.neuromod.value_net.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_value_grad = True
                break
        assert has_value_grad, "Value function should have gradients after replay"


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

    def test_memory_not_in_autograd(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        assert not model.memory.primitives.requires_grad
        assert not model.memory.key.requires_grad


class TestV8ParamCount:
    def test_param_count(self):
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        count = model.param_count()
        assert count > 0
        assert isinstance(count, int)
