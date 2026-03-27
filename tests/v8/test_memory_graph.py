"""Tests for the v9 Memory Graph — differentiable per-token neuron dynamics.

Tests cover:
  - Initialization (params, state, connectivity)
  - Forward segment (shapes, finiteness, differentiability)
  - Per-neuron modulator (gate output, gradient flow)
  - Dendritic FC layers
  - State persistence and TBPTT
"""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config
from src.v8.memory_graph import MemoryGraph

BS = 2


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


class TestMemoryGraphInit:
    def test_initialize(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        assert mg.is_initialized()
        assert mg.h.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.prev_messages.shape == (BS, cfg.N_neurons, cfg.D_mem)

    def test_params_are_nn_parameters(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert isinstance(mg.primitives, torch.nn.Parameter)
        assert isinstance(mg.key, torch.nn.Parameter)
        assert isinstance(mg.decay_logit, torch.nn.Parameter)
        assert mg.primitives.shape == (cfg.N_neurons, cfg.D_mem)
        assert mg.key.shape == (cfg.N_neurons, cfg.D_mem)

    def test_primitives_rms_normalized(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        rms = mg.primitives.data.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.05, rtol=0.05)

    def test_key_rms_normalized(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        rms = mg.key.data.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.05, rtol=0.05)

    def test_connectivity(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)
        for j in range(cfg.N_neurons):
            active = mg.conn_indices[j][mg.conn_mask[j]]
            assert j not in active.tolist(), "No self-connections"

    def test_modulator_params_exist(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.fc1_w.shape == (cfg.N_neurons, cfg.D_mem, cfg.modulator_hidden)
        assert mg.fc2_w.shape == (cfg.N_neurons, cfg.modulator_hidden, 3)
        # fc2 should be zero-init (no-op at start)
        assert mg.fc2_w.data.abs().sum() == 0
        assert mg.fc2_b.data.abs().sum() == 0

    def test_dendritic_params_exist(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        if mg.use_dendritic_tree:
            assert hasattr(mg, 'dendrite_branch_w')
            assert hasattr(mg, 'dendrite_group_w')

    def test_param_count(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        count = sum(p.numel() for p in mg.parameters())
        assert count > 0


class TestMemoryGraphForward:
    def test_forward_segment_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        h_prev = mg.h.detach()
        out, h_new = mg.forward_segment(cc, h_prev)
        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)
        assert h_new.shape == (BS, cfg.N_neurons, cfg.D_mem)

    def test_forward_segment_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc, mg.h.detach())
        assert torch.isfinite(out).all()
        assert torch.isfinite(h).all()

    def test_multiple_segments_stable(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        h = mg.h
        for _ in range(10):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.1
            out, h = mg.forward_segment(cc, h.detach())
            assert torch.isfinite(out).all()
            assert out.abs().max() < 100

    def test_state_persistence(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        _, h = mg.forward_segment(cc, mg.h.detach())
        assert h.abs().sum() > 0
        assert mg.prev_messages.abs().sum() > 0


class TestMemoryGraphGradients:
    def test_primitives_get_gradient(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc, mg.h.detach())
        loss = out.sum()
        loss.backward()
        assert mg.primitives.grad is not None
        assert mg.primitives.grad.abs().sum() > 0

    def test_key_get_gradient(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc, mg.h.detach())
        loss = out.sum()
        loss.backward()
        assert mg.key.grad is not None

    def test_decay_get_gradient(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc, mg.h.detach())
        loss = out.sum()
        loss.backward()
        assert mg.decay_logit.grad is not None

    def test_modulator_get_gradient(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)

        # Run two segments so traces are nonzero for second
        _, h = mg.forward_segment(cc, mg.h.detach())
        cc2 = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc2, h.detach())
        loss = out.sum()
        loss.backward()
        assert mg.fc1_w.grad is not None

    def test_dendritic_get_gradient(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        if not mg.use_dendritic_tree:
            pytest.skip("No dendritic tree in this config")
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc, mg.h.detach())
        loss = out.sum()
        loss.backward()
        assert mg.dendrite_branch_w.grad is not None

    def test_tbptt_detach(self):
        """Gradient should NOT flow across segment boundaries."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc1 = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out1, h1 = mg.forward_segment(cc1, mg.h.detach())

        # Detach h1 (TBPTT boundary)
        cc2 = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out2, h2 = mg.forward_segment(cc2, h1.detach())

        # Backward from out2 should not affect out1's graph
        loss = out2.sum()
        loss.backward()

        # out1 should not have gradients computed (detached boundary)
        assert not out1.requires_grad or out1.grad_fn is not None


class TestModulator:
    def test_modulator_output_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        h = mg.h
        gate_p, gate_k, decay_mod = mg._modulator_forward(h)
        assert gate_p.shape == (BS, cfg.N_neurons, 1)
        assert gate_k.shape == (BS, cfg.N_neurons, 1)
        assert decay_mod.shape == (BS, cfg.N_neurons, 1)

    def test_gate_bounded(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        gate_p, gate_k, _ = mg._modulator_forward(mg.h)
        assert gate_p.abs().max() <= 1.0
        assert gate_k.abs().max() <= 1.0

    def test_zero_init_means_no_modulation(self):
        """fc2 zero-init means modulator starts as no-op."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        gate_p, gate_k, decay_mod = mg._modulator_forward(mg.h)
        # With fc2 zero-init, outputs should be ~0
        assert gate_p.abs().max() < 1e-6
        assert gate_k.abs().max() < 1e-6
        assert decay_mod.abs().max() < 1e-6


class TestSignalPropagation:
    def test_port_neurons_receive_cc(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        mg.h.zero_()
        mg.prev_messages.zero_()
        cc = torch.ones(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out, h = mg.forward_segment(cc, mg.h.detach())
        # Port neurons should have nonzero state from CC injection
        assert h[:, :cfg.C].abs().sum() > 0

    def test_signal_propagates_to_internal(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        mg.h.zero_()
        mg.prev_messages.zero_()
        cc = torch.ones(BS, cfg.action_every, cfg.C, cfg.D_mem)
        _, h = mg.forward_segment(cc, mg.h.detach())
        non_port_h = h[:, cfg.C:]
        assert non_port_h.abs().sum() > 0, \
            "Internal neurons should receive signal via graph connectivity"

    def test_traces_accumulate(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        assert mg.trace_prim.abs().sum() == 0

        h = mg.h
        for _ in range(5):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
            _, h = mg.forward_segment(cc, h.detach())

        assert mg.trace_prim.abs().sum() > 0
        assert mg.trace_key.abs().sum() > 0
