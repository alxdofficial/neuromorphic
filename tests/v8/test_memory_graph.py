"""Tests for Memory Graph — per-token dynamics + ES-trained params.

Tests: init, forward, modulator, dendritic FC, ES utilities.
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

    def test_params_no_grad(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        for name, p in mg.named_parameters():
            assert not p.requires_grad, f"{name} should have requires_grad=False"

    def test_primitives_rms_normalized(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        rms = mg.primitives.data.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.05, rtol=0.05)

    def test_connectivity(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        for j in range(cfg.N_neurons):
            active = mg.conn_indices[j][mg.conn_mask[j]]
            assert j not in active.tolist()

    def test_modulator_zero_init(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.fc2_w.data.abs().sum() == 0
        assert mg.fc2_b.data.abs().sum() == 0


class TestMemoryGraphForward:
    def test_forward_segment_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)

    def test_forward_segment_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert torch.isfinite(out).all()

    def test_multiple_segments_stable(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        for _ in range(10):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem) * 0.1
            out = mg.forward_segment(cc)
            assert torch.isfinite(out).all()
            assert out.abs().max() < 100

    def test_state_updates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        h_before = mg.h.clone()
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg.forward_segment(cc)
        assert not torch.equal(mg.h, h_before)

    def test_traces_accumulate(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        assert mg.trace_prim.abs().sum() == 0
        for _ in range(5):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
            mg.forward_segment(cc)
        assert mg.trace_prim.abs().sum() > 0
        assert mg.trace_key.abs().sum() > 0

    def test_signal_propagates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        mg.h.zero_()
        mg.prev_messages.zero_()
        cc = torch.ones(BS, cfg.action_every, cfg.C, cfg.D_mem)
        mg.forward_segment(cc)
        assert mg.h[:, cfg.C:].abs().sum() > 0


class TestModulator:
    def test_output_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        gp, gk, dm = mg._modulator_forward(mg.h)
        assert gp.shape == (BS, cfg.N_neurons, 1)
        assert dm.shape == (BS, cfg.N_neurons, 1)

    def test_gate_bounded(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        gp, gk, _ = mg._modulator_forward(mg.h)
        assert gp.abs().max() <= 1.0
        assert gk.abs().max() <= 1.0

    def test_zero_init_no_modulation(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        gp, gk, dm = mg._modulator_forward(mg.h)
        assert gp.abs().max() < 1e-6
        assert dm.abs().max() < 1e-6


class TestES:
    def test_get_es_params(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        params = mg.get_es_params()
        assert 'primitives' in params
        assert 'fc1_w' in params
        assert params['primitives'].shape == (cfg.N_neurons, cfg.D_mem)

    def test_get_neuron_es_params(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        k = torch.tensor([0, 1, 2])
        params = mg.get_neuron_es_params(k)
        assert params['primitives'].shape == (3, cfg.D_mem)
        assert params['fc1_w'].shape == (3, cfg.D_mem * 5, cfg.modulator_hidden)

    def test_apply_perturbation(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        k = torch.tensor([0, 1])
        prim_before = mg.primitives.data[0].clone()

        noise = mg.get_neuron_es_params(k)
        for name in noise:
            noise[name] = torch.randn_like(noise[name])

        mg.apply_es_perturbation(k, noise, sigma=0.1)
        assert not torch.equal(mg.primitives.data[0], prim_before)
        # Neuron 2 should be unchanged
        # (can't test easily since noise only applied to k=[0,1])

    def test_apply_update(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        k = torch.tensor([0])
        prim_before = mg.primitives.data[0].clone()

        update = {'primitives': torch.ones(1, cfg.D_mem) * 0.01}
        mg.apply_es_update(k, update, lr=1.0)

        diff = (mg.primitives.data[0] - prim_before).abs().sum()
        assert diff > 0
