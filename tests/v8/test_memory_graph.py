"""Tests for v10 scalar neuron memory graph."""

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
        assert mg.V.shape == (BS, cfg.N_neurons)
        assert mg.activation.shape == (BS, cfg.N_neurons)
        assert mg.w_conn.shape == (BS, cfg.N_neurons, cfg.K_connections)
        assert mg.decay.shape == (BS, cfg.N_neurons)
        assert mg.threshold.shape == (BS, cfg.N_neurons)
        assert mg.hebbian.shape == (BS, cfg.N_neurons, cfg.K_connections)

    def test_modulator_params_have_grad(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.mod_w1.requires_grad
        assert mg.mod_b1.requires_grad
        assert mg.mod_w2.requires_grad
        assert mg.mod_b2.requires_grad

    def test_conn_indices_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)

    def test_conn_no_self_connections(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        for n in range(cfg.N_neurons):
            assert n not in mg.conn_indices[n].tolist()

    def test_neuron_id_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.neuron_id.shape == (cfg.N_neurons, 2)

    def test_groups_contiguous(self):
        """Neurons 0..group_size-1 should be group 0, etc."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        GS = cfg.group_size
        # Check that intra-group connections exist
        for n in range(min(GS, cfg.N_neurons)):
            group_start = (n // GS) * GS
            group_end = group_start + GS
            conns = mg.conn_indices[n].tolist()
            intra = [c for c in conns if group_start <= c < group_end]
            assert len(intra) >= cfg.min_intra_connections, \
                f"Neuron {n} has {len(intra)} intra-group connections, expected >= {cfg.min_intra_connections}"


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
        assert mg.V.abs().sum() > 0

        V_before = mg.V.clone()
        mg.forward_segment(cc)
        assert not torch.equal(mg.V, V_before)

    def test_modulator_sets_properties(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        mg.forward_segment(cc)

        assert torch.isfinite(mg.w_conn).all()
        assert torch.isfinite(mg.decay).all()
        assert torch.isfinite(mg.threshold).all()

    def test_hebbian_accumulated(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        mg.forward_segment(cc)

        assert mg.hebbian.abs().sum() > 0


class TestInjectReadout:
    def test_inject_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        cc = torch.randn(BS, cfg.action_every, cfg.D)
        injected = mg.inject(cc)
        assert injected.shape == (BS, cfg.action_every, cfg.N_neurons)

    def test_inject_replication(self):
        """Replicas of the same LM dim should get the same value."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        cc = torch.randn(BS, 1, cfg.D)
        injected = mg.inject(cc)
        R = cfg.replicas_per_dim
        # Neuron 0 and neuron 1 share the same LM dim
        if R > 1:
            torch.testing.assert_close(injected[:, 0, 0], injected[:, 0, 1])

    def test_readout_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        act = torch.randn(BS, cfg.action_every, cfg.N_neurons)
        out = mg.readout(act)
        assert out.shape == (BS, cfg.action_every, cfg.D)

    def test_readout_averages_replicas(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        R = cfg.replicas_per_dim
        act = torch.zeros(BS, 1, cfg.N_neurons)
        # Set first R neurons (dim 0's replicas) to 1.0
        act[:, :, :R] = 1.0
        out = mg.readout(act)
        torch.testing.assert_close(out[:, 0, 0], torch.tensor(1.0).expand(BS))

    def test_inject_readout_roundtrip(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        cc = torch.randn(BS, 1, cfg.D)
        injected = mg.inject(cc)
        recovered = mg.readout(injected)
        torch.testing.assert_close(recovered, cc)


class TestStructuralPlasticity:
    def test_rewire_changes_connections(self):
        cfg = make_tiny(structural_plasticity=True, plasticity_n_swap=2)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        conn_before = mg.conn_indices.clone()

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        for _ in range(4):
            mg.forward_segment(cc)

        mg.rewire_connections()

        changed = (mg.conn_indices != conn_before).any()
        assert changed, "rewire should change some connections"

    def test_rewire_preserves_k(self):
        cfg = make_tiny(structural_plasticity=True, plasticity_n_swap=2)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.D)
        for _ in range(4):
            mg.forward_segment(cc)
        mg.rewire_connections()

        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)

    def test_noop_when_disabled(self):
        cfg = make_tiny(structural_plasticity=False)
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        conn_before = mg.conn_indices.clone()
        mg.rewire_connections()
        assert torch.equal(mg.conn_indices, conn_before)
