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

    def test_counterfactual_advantages(self):
        """Counterfactual advantages should be per-neuron and finite."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        n_segments = cfg.T // cfg.action_every
        N = cfg.N_neurons

        # Collect chunks with neuromod enabled (produces logits_B)
        collected = []
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            assert rl is not None, "rl_data should be populated with neuromod"
            assert rl["logits_B"] is not None, "Counterfactual logits should exist"

            # Derive rewards (same as trainer)
            with torch.no_grad():
                logits_A = result["logits"]
                logits_B = rl["logits_B"]
                mask = (~rl["eot_at"]).float()

                ce_A = F.cross_entropy(logits_A.detach().reshape(-1, VOCAB),
                    target_ids.reshape(-1), reduction='none').reshape(BS, cfg.T)
                ce_B = F.cross_entropy(logits_B.reshape(-1, VOCAB),
                    target_ids.reshape(-1), reduction='none').reshape(BS, cfg.T)

                seg_mask = mask.view(BS, n_segments, cfg.action_every)
                seg_ce_A = ce_A.view(BS, n_segments, cfg.action_every)
                seg_ce_B = ce_B.view(BS, n_segments, cfg.action_every)
                seg_losses_A = (seg_ce_A * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
                seg_losses_B = (seg_ce_B * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)

                rl["seg_rewards"] = -seg_losses_A
                rl["seg_rewards_cf"] = -seg_losses_B
                rl["loss"] = ce_A.mean().item()
            collected.append(rl)

        combined = model.compute_counterfactual_advantages(collected)
        total_seg = n_segments * 2
        # Per-neuron advantages
        assert combined["advantages"].shape == (BS, total_seg, N)
        assert torch.isfinite(combined["advantages"]).all()
        assert len(combined["obs"]) == total_seg
        assert len(combined["actions"]) == total_seg

    def test_replay_produces_gradients(self):
        """Replay should produce gradients for neuromod policy."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))
        n_segments = cfg.T // cfg.action_every
        N = cfg.N_neurons

        collected = []
        for _ in range(cfg.rl_collect_chunks):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            with torch.no_grad():
                logits_A = result["logits"]
                logits_B = rl["logits_B"]
                mask = (~rl["eot_at"]).float()
                ce_A = F.cross_entropy(logits_A.detach().reshape(-1, VOCAB),
                    target_ids.reshape(-1), reduction='none').reshape(BS, cfg.T)
                ce_B = F.cross_entropy(logits_B.reshape(-1, VOCAB),
                    target_ids.reshape(-1), reduction='none').reshape(BS, cfg.T)
                seg_mask = mask.view(BS, n_segments, cfg.action_every)
                seg_ce_A = ce_A.view(BS, n_segments, cfg.action_every)
                seg_ce_B = ce_B.view(BS, n_segments, cfg.action_every)
                seg_losses_A = (seg_ce_A * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
                seg_losses_B = (seg_ce_B * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
                rl["seg_rewards"] = -seg_losses_A
                rl["seg_rewards_cf"] = -seg_losses_B
                rl["loss"] = ce_A.mean().item()
            collected.append(rl)

        combined = model.compute_counterfactual_advantages(collected)
        model.replay_for_neuromod_grads(combined, amp_enabled=False)

        has_policy_grad = False
        for name, p in model.neuromod.named_parameters():
            if "head" in name or "logstd" in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_policy_grad = True
                    break
        assert has_policy_grad, "Policy heads should have gradients after replay"


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
