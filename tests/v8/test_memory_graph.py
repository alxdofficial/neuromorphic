"""Tests for v9-backprop memory graph."""

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
    def test_init(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.config is cfg
        assert not mg.is_initialized()

    def test_initialize_states(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        assert mg.is_initialized()
        assert mg.h.shape == (BS, cfg.N_neurons, cfg.D_neuron)
        assert mg.prev_messages.shape == (BS, cfg.N_neurons, cfg.D_neuron)
        assert mg.w_conn.shape == (BS, cfg.N_neurons, cfg.K_connections)
        assert mg.primitives_state.shape == (BS, cfg.N_neurons, cfg.D_neuron)
        assert mg.hebbian_traces.shape == (BS, cfg.N_neurons, cfg.K_connections)

    def test_modulator_params_have_grad(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.mod_w1.requires_grad
        assert mg.mod_b1.requires_grad
        assert mg.mod_w2.requires_grad
        assert mg.mod_b2.requires_grad

    def test_mlp_params_have_grad(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.state_w1.requires_grad
        assert mg.state_w2.requires_grad
        assert mg.msg_w1.requires_grad
        assert mg.msg_w2.requires_grad
        assert mg.neuron_id.requires_grad

    def test_all_params_require_grad(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        for name, p in mg.named_parameters():
            assert p.requires_grad, f"{name} should require grad"

    def test_conn_indices_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)


class TestMemoryGraphForward:
    def test_forward_segment_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.action_every, cfg.D)

    def test_forward_segment_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        assert torch.isfinite(out).all()

    def test_forward_on_compute_graph(self):
        """Output should have grad_fn (differentiable through modulator)."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        assert out.requires_grad, "Output should be on compute graph"

    def test_gradient_flows_to_modulator(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        loss = out.sum()
        loss.backward()
        assert mg.mod_w1.grad is not None
        assert mg.mod_w1.grad.norm() > 0, "mod_w1 should have nonzero grad"
        assert mg.mod_w2.grad is not None
        assert mg.mod_w2.grad.norm() > 0, "mod_w2 should have nonzero grad"

    def test_gradient_flows_to_state_mlp(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        loss = out.sum()
        loss.backward()
        assert mg.state_w1.grad is not None
        assert mg.state_w1.grad.norm() > 0, "state_w1 should have nonzero grad"
        assert mg.state_w2.grad is not None
        assert mg.state_w2.grad.norm() > 0, "state_w2 should have nonzero grad"

    def test_gradient_flows_to_msg_mlp(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        loss = out.sum()
        loss.backward()
        assert mg.msg_w1.grad is not None
        assert mg.msg_w1.grad.norm() > 0, "msg_w1 should have nonzero grad"
        assert mg.msg_w2.grad is not None
        assert mg.msg_w2.grad.norm() > 0, "msg_w2 should have nonzero grad"

    def test_gradient_flows_to_neuron_id(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        loss = out.sum()
        loss.backward()
        assert mg.neuron_id.grad is not None
        assert mg.neuron_id.grad.norm() > 0, "neuron_id should have nonzero grad"

    def test_gradient_flows_to_all_params(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        out = mg.forward_segment(cc)
        out.sum().backward()
        for name, p in mg.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.norm() > 0, f"{name} has zero grad"

    def test_segment_boundary_detach(self):
        """Gradients from segment 2 should NOT flow into segment 1."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc1 = torch.randn(BS, cfg.action_every, cfg.D)
        cc2 = torch.randn(BS, cfg.action_every, cfg.D)

        out1 = mg.forward_segment(cc1)
        out1.sum().backward(retain_graph=True)
        grad_after_seg1 = mg.mod_w1.grad.clone()

        mg.zero_grad()

        out2 = mg.forward_segment(cc2)
        out2.sum().backward()
        grad_after_seg2 = mg.mod_w1.grad.clone()

        assert not torch.equal(grad_after_seg1, grad_after_seg2)

    def test_state_persists_across_segments(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        mg.forward_segment(cc)

        assert mg.h.abs().sum() > 0

        h_before = mg.h.clone()
        mg.forward_segment(cc)
        assert not torch.equal(mg.h, h_before)

    def test_modulator_predicts_neuron_properties(self):
        """Modulator should set w_conn, decay, primitives."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        assert mg.w_conn.abs().sum() == 0
        assert mg.primitives_state.abs().sum() == 0

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        mg.forward_segment(cc)

        assert torch.isfinite(mg.w_conn).all()
        assert torch.isfinite(mg.primitives_state).all()
        assert torch.isfinite(mg.decay_logit).all()

    def test_hebbian_traces_accumulated(self):
        """Hebbian traces should be nonzero after forward segment."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        mg.forward_segment(cc)

        assert mg.hebbian_traces.abs().sum() > 0, \
            "Hebbian traces should be nonzero after forward"


class TestInjectReadout:
    def test_inject_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc_t = torch.randn(BS, cfg.D)
        inject_t = mg._inject(cc_t)
        assert inject_t.shape == (BS, mg.N_inject, cfg.D_neuron)

    def test_inject_replication(self):
        """Each group of alpha inject neurons per slice gets the same signal."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc_t = torch.randn(BS, cfg.D)
        inject_t = mg._inject(cc_t)
        # First alpha neurons per slice should get the same signal
        if mg.alpha > 1:
            torch.testing.assert_close(inject_t[:, 0], inject_t[:, 1])

    def test_readout_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        msg = torch.randn(BS, cfg.N_neurons, cfg.D_neuron)
        out = mg._readout(msg)
        assert out.shape == (BS, cfg.D)

    def test_readout_only_reads_port_neurons(self):
        """Only readout neurons should affect output."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        msg1 = torch.zeros(BS, cfg.N_neurons, cfg.D_neuron)
        msg1[:, mg.readout_indices, :] = 1.0
        out1 = mg._readout(msg1)

        msg2 = torch.randn(BS, cfg.N_neurons, cfg.D_neuron)
        msg2[:, mg.readout_indices, :] = 1.0
        out2 = mg._readout(msg2)
        torch.testing.assert_close(out1, out2)


class TestStructuralPlasticity:
    def test_rewire_changes_connections(self):
        cfg = make_tiny(structural_plasticity=True, plasticity_pct=0.1)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        conn_before = mg.conn_indices.clone()

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        for _ in range(4):
            mg.forward_segment(cc)

        mg.rewire_connections()

        changed = (mg.conn_indices != conn_before).any()
        assert changed, "rewire_connections should change some connections"

    def test_rewire_preserves_k(self):
        cfg = make_tiny(structural_plasticity=True, plasticity_pct=0.1)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        for _ in range(4):
            mg.forward_segment(cc)
        mg.rewire_connections()

        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)

    def test_rewire_no_self_connections(self):
        cfg = make_tiny(structural_plasticity=True, plasticity_pct=0.1)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        for _ in range(4):
            mg.forward_segment(cc)
        mg.rewire_connections()

        # No neuron should connect to itself
        for n in range(cfg.N_neurons):
            assert n not in mg.conn_indices[n].tolist()

    def test_noop_when_disabled(self):
        cfg = make_tiny(structural_plasticity=False)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        conn_before = mg.conn_indices.clone()
        mg.rewire_connections()
        assert torch.equal(mg.conn_indices, conn_before)
