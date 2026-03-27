"""Tests for Memory Graph — per-neuron MLPs + ES training."""

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


class TestInit:
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
            assert not p.requires_grad, f"{name} should not require grad"

    def test_write_read_neurons_disjoint(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        w = set(mg.write_neurons.tolist())
        r = set(mg.read_neurons.tolist())
        assert w.isdisjoint(r), "Write and read neurons must be different"

    def test_has_mlp_params(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert hasattr(mg, 'int_w1')  # integrate MLP
        assert hasattr(mg, 'msg_w1')  # message MLP
        assert hasattr(mg, 'mod_w1')  # modulator MLP
        assert hasattr(mg, 'key')     # routing key

    def test_modulator_zero_init(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.mod_w2.data.abs().sum() == 0


class TestForward:
    def test_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)

    def test_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem)
        out = mg.forward_segment(cc)
        assert torch.isfinite(out).all()

    def test_stable(self):
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

    def test_signal_propagates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        mg.h.zero_()
        mg.prev_messages.zero_()
        # Run multiple segments with strong CC input
        for _ in range(4):
            cc = torch.ones(BS, cfg.action_every, cfg.C, cfg.D_mem) * 2.0
            mg.forward_segment(cc)
        # Read neurons should have nonzero output (signal traversed graph)
        out = mg.prev_messages[:, mg.read_neurons]
        assert out.abs().mean() > 1e-4, \
            f"Read neurons should receive signal (got mean={out.abs().mean():.6f})"


class TestMLPs:
    def test_integrate_mlp(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        h = mg.h
        received = torch.randn_like(h)
        h_new = mg._integrate(h, received)
        assert h_new.shape == h.shape
        assert torch.isfinite(h_new).all()

    def test_message_mlp(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        msg = mg._message(mg.h)
        assert msg.shape == mg.h.shape
        assert msg.abs().max() <= 1.0  # tanh bounded

    def test_modulator(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        gp, gk, dm = mg._modulator_forward(mg.h)
        assert gp.shape == (BS, cfg.N_neurons, 1)
        assert gp.abs().max() <= 1.0


class TestES:
    def test_get_es_params(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        params = mg.get_es_params()
        assert 'key' in params
        assert 'int_w1' in params
        assert 'msg_w1' in params
        assert 'mod_w1' in params

    def test_get_neuron_params(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        k = torch.tensor([0, 1, 2])
        params = mg.get_neuron_es_params(k)
        assert params['key'].shape == (3, cfg.D_mem)
        assert params['int_w1'].shape[0] == 3

    def test_perturbation(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        k = torch.tensor([0, 1])
        key_before = mg.key.data[0].clone()
        noise = mg.get_neuron_es_params(k)
        for name in noise:
            noise[name] = torch.randn_like(noise[name])
        mg.apply_es_perturbation(k, noise, sigma=0.1)
        assert not torch.equal(mg.key.data[0], key_before)
