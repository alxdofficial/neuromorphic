"""Tests for the Neural Memory Graph."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.v8.config import V8Config
from src.v8.memory_graph import MemoryGraph


def make_tiny():
    cfg = V8Config.tier_tiny()
    cfg.validate()
    return cfg


BS = 2


class TestMemoryGraphInit:
    def test_initialize(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        assert mg.primitives.shape == (BS, cfg.N_neurons, cfg.D_mem)
        assert mg.thresholds.shape == (BS, cfg.N_neurons, cfg.max_connections)
        assert mg.activations.shape == (BS, cfg.N_neurons, cfg.D_mem)

    def test_connectivity(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        # Check conn_indices shape
        assert mg.conn_indices.shape == (cfg.N_neurons, cfg.max_connections)
        # All indices should be valid
        assert (mg.conn_indices < cfg.N_neurons).all()
        # CC port indices
        assert mg.cc_port_idx.shape == (cfg.C,)


class TestMemoryGraphStep:
    def test_step_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        cc_signals = torch.randn(BS, cfg.C, cfg.D_mem)
        mem_signals = mg.step(cc_signals)
        assert mem_signals.shape == (BS, cfg.C, cfg.D_mem)

    def test_step_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        cc_signals = torch.randn(BS, cfg.C, cfg.D_mem)
        mem_signals = mg.step(cc_signals)
        assert torch.isfinite(mem_signals).all()

    def test_multiple_steps(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        for _ in range(10):
            cc_signals = torch.randn(BS, cfg.C, cfg.D_mem)
            mem_signals = mg.step(cc_signals)
        assert torch.isfinite(mem_signals).all()
        assert torch.isfinite(mg.primitives).all()

    def test_step_changes_state(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        prev_out_before = mg.prev_output.clone()
        cc_signals = torch.randn(BS, cfg.C, cfg.D_mem)
        mg.step(cc_signals)
        # prev_output should change after a step
        assert not torch.allclose(mg.prev_output, prev_out_before)


class TestMemoryGraphActions:
    def test_apply_actions(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        prim_before = mg.primitives.clone()
        delta_p = torch.randn(BS, cfg.N_neurons, cfg.D_mem) * 0.01
        delta_t = torch.randn(BS, cfg.N_neurons, cfg.max_connections) * 0.01
        delta_temp = torch.randn(BS, cfg.N_neurons) * 0.01
        delta_decay = torch.randn(BS, cfg.N_neurons) * 0.01
        mg.apply_actions(delta_p, delta_t, delta_temp, delta_decay)
        assert not torch.allclose(mg.primitives, prim_before)

    def test_get_neuron_obs(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        obs = mg.get_neuron_obs()
        assert obs.shape == (BS, cfg.N_neurons, mg.obs_dim)
        assert torch.isfinite(obs).all()

    def test_get_neuron_obs_with_surprise(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        surprise = torch.randn(BS, cfg.C, cfg.D_cc)
        obs = mg.get_neuron_obs(cc_surprise=surprise)
        assert obs.shape == (BS, cfg.N_neurons, mg.obs_dim)


class TestMemoryGraphReset:
    def test_reset_streams(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize(BS)
        # Step a few times to build state
        for _ in range(5):
            mg.step(torch.randn(BS, cfg.C, cfg.D_mem))
        prim_before = mg.primitives.clone()
        # Reset stream 0
        mask = torch.tensor([True, False])
        mg.reset_streams(mask)
        # Stream 0 should change, stream 1 should not
        assert not torch.allclose(mg.primitives[0], prim_before[0])
        assert torch.allclose(mg.primitives[1], prim_before[1])
