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
        assert rl["n_segments"] == n_segments
        assert rl["action_every"] == cfg.action_every
        assert rl["eot_at"].shape == (BS, cfg.T)
        assert rl["cc_segments"].shape == (BS, n_segments, cfg.action_every,
                                           cfg.C, cfg.D_cc)
        assert rl["H_mid"].shape == (BS, cfg.T, cfg.D)

    def test_grpo_scoring(self):
        """GRPO trajectory scoring across collected chunks."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Save MG state before collection
        pre_mg_state = {k: v.clone() for k, v in model.memory.state_dict().items()}

        # Collect 2 chunks into buffer
        rl_buffer = []
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            rl["target_ids"] = target_ids
            rl_buffer.append(rl)

        scoring = model.score_trajectories(rl_buffer, pre_mg_state)
        expected_k = min(cfg.rl_counterfactual_k, cfg.N_neurons)
        assert scoring["k_neurons"].shape[0] == expected_k
        assert scoring["trajectory_advantages"].shape[0] == cfg.rl_counterfactual_n
        assert len(scoring["trajectory_k_actions"]) == cfg.rl_counterfactual_n
        assert len(scoring["trajectory_k_obs"]) == cfg.rl_counterfactual_n
        assert torch.isfinite(scoring["trajectory_advantages"]).all()

    def test_replay_produces_gradients(self):
        """Replay should produce gradients for neuromod policy."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        pre_mg_state = {k: v.clone() for k, v in model.memory.state_dict().items()}

        rl_buffer = []
        for _ in range(cfg.rl_collect_chunks):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            rl["target_ids"] = target_ids
            rl_buffer.append(rl)

        scoring = model.score_trajectories(rl_buffer, pre_mg_state)
        replay_data = model.prepare_grpo_replay(scoring)
        model.replay_for_neuromod_grads(replay_data, amp_enabled=False)

        has_policy_grad = False
        for name, p in model.neuromod.named_parameters():
            if "head" in name or "logstd" in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_policy_grad = True
                    break
        assert has_policy_grad, "Policy heads should have gradients after replay"

    def test_best_trajectory_state_persists(self):
        """After scoring, memory graph should be in best trajectory's state."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        pre_mg_state = {k: v.clone() for k, v in model.memory.state_dict().items()}

        rl_buffer = []
        for _ in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            target_ids = torch.randint(0, VOCAB, (BS, cfg.T))
            result = model.forward_chunk(input_ids, target_ids=target_ids)
            rl = result["rl_data"]
            rl["target_ids"] = target_ids
            rl_buffer.append(rl)

        # State before scoring
        h_before = model.memory.h.clone()

        scoring = model.score_trajectories(rl_buffer, pre_mg_state)

        # State after scoring should differ from both pre-scoring and pre-collection
        # (it's the best trajectory's final state)
        h_after = model.memory.h
        assert not torch.equal(h_after, pre_mg_state["h"]), \
            "MG state should not be restored to pre-collection (should be best trajectory)"

    def test_memory_persists_across_doc_boundaries(self):
        """Memory graph should NOT reset at document boundaries."""
        cfg = make_tiny()
        model = V8Model(cfg)
        model.initialize_states(BS, torch.device("cpu"))

        # Run a chunk to build up memory state
        input_ids = torch.randint(0, VOCAB, (BS, cfg.T))
        model.forward_chunk(input_ids, use_neuromod=False)
        h_before = model.memory.h.clone()
        assert h_before.abs().sum() > 0, "Should have nonzero state"

        # Simulate doc boundary: reset_mask=True for batch 0
        input_ids2 = torch.randint(0, VOCAB, (BS, cfg.T))
        reset_mask = torch.tensor([True, False])
        model.forward_chunk(input_ids2, reset_mask=reset_mask,
                            has_reset=True, use_neuromod=False)

        # Memory graph should NOT have been zeroed for batch 0
        # (it persists across doc boundaries)
        h_after = model.memory.h
        # Both batch elements should have nonzero state
        assert h_after[0].abs().sum() > 0, \
            "Memory should persist across doc boundaries (batch 0)"
        assert h_after[1].abs().sum() > 0, \
            "Memory should persist across doc boundaries (batch 1)"


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
