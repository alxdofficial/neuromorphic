"""Tests for Memory Graph — lightweight hippocampal neurons."""

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

    def test_params_have_grad(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        for name, p in mg.named_parameters():
            assert p.requires_grad, f"{name} should require grad"

    def test_broadcast_inject_readout(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        C_mem = cfg.D // cfg.D_mem
        assert mg.inject_w.shape == (cfg.N_neurons, C_mem)
        assert mg.readout_w.shape == (C_mem, cfg.N_neurons)

    def test_has_per_neuron_mlp(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert hasattr(mg, 'W1')      # hidden layer
        assert hasattr(mg, 'W_msg')   # message head
        assert hasattr(mg, 'W_mod')   # modulator head
        assert hasattr(mg, 'w_conn')  # connection weights

    def test_modulator_zero_init(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert mg.W_mod.data.abs().sum() == 0
        assert mg.b_mod.data.abs().sum() == 0

    def test_no_projections_needed(self):
        """D_mem matches channel width — no projection layers."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        assert not hasattr(mg, 'proj_to_mem')
        assert not hasattr(mg, 'proj_from_mem')
        assert mg.C_mem == cfg.D // cfg.D_mem

    def test_param_count_per_neuron(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        N = cfg.N_neurons
        D, H, K = cfg.D_mem, cfg.neuron_hidden, cfg.K_connections
        # W1 input is 3D (h, trace_h, trace_received) — MLP runs once per segment
        expected_per_neuron = K + 1 + 3*D*H + H + H*D + D + H*3 + 3
        neuron_params = (mg.w_conn.numel() + mg.decay_logit.numel() +
                        mg.W1.numel() + mg.b1.numel() +
                        mg.W_msg.numel() + mg.b_msg.numel() +
                        mg.W_mod.numel() + mg.b_mod.numel())
        assert neuron_params == N * expected_per_neuron


class TestForward:
    def _cc(self, cfg, scale=1.0):
        """Create cc_signals in memory channel format [BS, T_seg, C_mem, D_mem]."""
        C_mem = cfg.D // cfg.D_mem
        return torch.randn(BS, cfg.action_every, C_mem, cfg.D_mem) * scale

    def test_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        cc = self._cc(cfg)
        out = mg.forward_segment(cc)
        C_mem = cfg.D // cfg.D_mem
        assert out.shape == (BS, cfg.action_every, C_mem, cfg.D_mem)

    def test_finite(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        out = mg.forward_segment(self._cc(cfg))
        assert torch.isfinite(out).all()

    def test_stable(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        for _ in range(10):
            out = mg.forward_segment(self._cc(cfg, scale=0.1))
            assert torch.isfinite(out).all()
            assert out.abs().max() < 100

    def test_state_updates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        h_before = mg.h.clone()
        mg.forward_segment(self._cc(cfg))
        assert not torch.equal(mg.h, h_before)

    def test_differentiable(self):
        """Memory graph output should carry gradients to parameters."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        out = mg.forward_segment(self._cc(cfg))
        loss = out.sum()
        loss.backward()
        assert mg.W_msg.grad is not None, "W_msg should have gradient"
        assert mg.W1.grad is not None, "W1 should have gradient"
        assert mg.w_conn.grad is not None, "w_conn should have gradient"
        assert mg.inject_w.grad is not None, "inject_w should have gradient"
        assert mg.readout_w.grad is not None, "readout_w should have gradient"

    def test_signal_propagates(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        mg.h.zero_()
        mg.prev_messages.zero_()
        for _ in range(4):
            C_mem = cfg.D // cfg.D_mem
            cc = torch.ones(BS, cfg.action_every, C_mem, cfg.D_mem) * 2.0
            mg.forward_segment(cc)
        # With broadcast inject, all neurons get signal directly
        assert mg.prev_messages.abs().mean() > 1e-4, \
            f"Neurons should receive signal (got mean={mg.prev_messages.abs().mean():.6f})"


class TestHebbian:
    def _cc(self, cfg, scale=1.0):
        C_mem = cfg.D // cfg.D_mem
        return torch.randn(BS, cfg.action_every, C_mem, cfg.D_mem) * scale

    def test_hebbian_updates_w_conn(self):
        """Hebbian updates w_conn when modulator is non-zero."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        # W_mod output: [decay_mod, plasticity_gate, plasticity_lr]
        mg.W_mod.data[:, :, 1] = 1.0  # gate head
        mg.b_mod.data[:, 1] = 1.0     # bias for gate
        mg.W_mod.data[:, :, 2] = 1.0  # lr head
        mg.b_mod.data[:, 2] = 1.0     # bias for lr
        w_before = mg.w_conn.data.clone()
        mg.forward_segment(self._cc(cfg, scale=2.0))
        assert not torch.equal(mg.w_conn.data, w_before), \
            "w_conn should change from Hebbian plasticity"

    def test_hebbian_no_update_with_zero_mod(self):
        """With zero-init modulator, Hebbian gate is ~0 so w_conn doesn't change."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        mg.initialize_states(BS)
        w_before = mg.w_conn.data.clone()
        mg.forward_segment(self._cc(cfg))
        assert torch.allclose(mg.w_conn.data, w_before, atol=1e-6), \
            "w_conn should NOT change when modulator is zero-init"


class TestModulator:
    def test_modulator_output_shape(self):
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cpu"))
        N, D, H = cfg.N_neurons, cfg.D_mem, cfg.neuron_hidden
        assert mg.W_mod.shape == (N, H, 3)
        assert mg.b_mod.shape == (N, 3)
