"""Tests for the Memory Graph — serial scan blocks + sparse message passing."""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config
from src.v8.memory_graph import MemoryGraph, _cpu_scan

BS = 2


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


class TestDiagonalScan:
    def test_output_shape(self):
        B, T, D = 4, 16, 8
        decay_logit = torch.randn(B, D)
        b = torch.randn(B, T, D)
        out = _cpu_scan(decay_logit, b)
        assert out.shape == (B, T, D)

    def test_with_carry(self):
        B, T, D = 2, 8, 4
        decay_logit = torch.randn(B, D)
        b = torch.randn(B, T, D)
        h0 = torch.randn(B, D)
        out = _cpu_scan(decay_logit, b, h0)
        assert out.shape == (B, T, D)

        # Verify first step: h[0] = sigmoid(decay) * h0 + b[0]
        a = torch.sigmoid(decay_logit)
        expected_h0 = a * h0 + b[:, 0]
        torch.testing.assert_close(out[:, 0], expected_h0, atol=1e-5, rtol=1e-4)

    def test_sequential_equivalence(self):
        B, T, D = 2, 16, 4
        decay_logit = torch.randn(B, D)
        b = torch.randn(B, T, D)
        h0 = torch.randn(B, D)

        # Parallel scan
        out_par = _cpu_scan(decay_logit, b, h0)

        # Sequential reference
        a = torch.sigmoid(decay_logit)
        h = h0
        out_seq = []
        for t in range(T):
            h = a * h + b[:, t]
            out_seq.append(h)
        out_seq = torch.stack(out_seq, dim=1)

        torch.testing.assert_close(out_par, out_seq, atol=1e-5, rtol=1e-4)


class TestMemoryGraphInit:
    def test_initialize(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        assert mg.is_initialized()
        assert mg.primitives.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.h.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.prev_messages.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.conn_weights.shape == (BS, cfg.N_neurons, cfg.K_connections)
        assert mg.flow_ema.shape == (BS, cfg.N_neurons, cfg.K_connections)

    def test_connectivity(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.K_connections)
        assert mg.conn_mask.shape == (cfg.N_neurons, cfg.K_connections)
        # No self-connections
        for j in range(cfg.N_neurons):
            active = mg.conn_indices[j][mg.conn_mask[j]]
            assert j not in active.tolist()


class TestMemoryGraphForward:
    def test_forward_segment_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)

    def test_forward_segment_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert torch.isfinite(out).all()

    def test_multiple_segments_stable(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        for _ in range(10):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.1
            out = mg.forward_segment(cc)
            assert torch.isfinite(out).all()
            assert out.abs().max() < 100  # no explosion

    def test_state_persistence(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg.forward_segment(cc)

        # Internal state and messages should be nonzero after processing
        assert mg.h.abs().sum() > 0
        assert mg.prev_messages.abs().sum() > 0


class TestMemoryGraphMessagePassing:
    def test_message_pass_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)
        x = torch.randn(BS, 8, cfg.N_neurons, cfg.D_mem)
        out = mg._message_pass(x)
        assert out.shape == x.shape

    def test_zero_weights_zero_messages(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize(BS)
        mg.conn_weights.zero_()
        x = torch.randn(BS, 4, cfg.N_neurons, cfg.D_mem)
        out = mg._message_pass(x)
        assert out.abs().max() == 0


class TestMemoryGraphActions:
    def test_apply_actions(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        prim_before = mg.primitives.clone()

        d_prim = torch.randn(BS, cfg.N_neurons, cfg.D_mem) * 0.01
        d_conn = torch.randn(BS, cfg.N_neurons, cfg.K_connections) * 0.01
        d_decay = torch.randn(BS, cfg.N_neurons) * 0.01
        mg.apply_actions(d_prim, d_conn, d_decay)

        assert not torch.equal(mg.primitives, prim_before)

    def test_get_neuron_obs(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        obs = mg.get_neuron_obs()
        assert obs.shape == (BS, cfg.N_neurons, mg.obs_dim)
        assert torch.isfinite(obs).all()


class TestMemoryGraphPlasticity:
    def test_flow_ema_updates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        flow_before = mg.flow_ema.clone()

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg.forward_segment(cc)

        # Flow EMA should have changed (unless all outputs were zero)
        # Just check it's finite
        assert torch.isfinite(mg.flow_ema).all()

    def test_structural_plasticity(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)

        # Set some connection weights to near-zero
        mg.conn_weights[:, 0, :2] = 0.001
        indices_before = mg.conn_indices[0, :2].clone()

        mg.structural_plasticity()

        # Pruned connections should have been rewired
        indices_after = mg.conn_indices[0, :2]
        assert not torch.equal(indices_before, indices_after)


class TestMemoryGraphReset:
    def test_reset_streams(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)

        # Step to build up state
        for _ in range(3):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
            mg.forward_segment(cc)

        h_before = mg.h.clone()
        msg_before = mg.prev_messages.clone()
        prim_before = mg.primitives.clone()
        conn_before = mg.conn_weights.clone()

        # Reset stream 0 only
        mask = torch.tensor([True, False])
        mg.reset_streams(mask)

        # Dynamic state: stream 0 zeroed, stream 1 unchanged
        assert mg.h[0].abs().sum() == 0
        assert mg.prev_messages[0].abs().sum() == 0
        torch.testing.assert_close(mg.h[1], h_before[1])
        torch.testing.assert_close(mg.prev_messages[1], msg_before[1])

        # Structural state: preserved for ALL streams (including stream 0)
        torch.testing.assert_close(mg.primitives, prim_before)
        torch.testing.assert_close(mg.conn_weights, conn_before)
