"""Memory graph unit tests (dense-W design)."""

import torch
import pytest

from src.model.config import Config
from src.model.memory import MemoryGraph


def _tiny_config(**kw):
    return Config.tier_tiny(**kw)


def _make_graph(config=None, BS=2):
    if config is None:
        config = _tiny_config()
    mg = MemoryGraph(config)
    device = torch.device("cpu")
    mg.initialize_states(BS, device)
    return mg, config


class TestShapes:
    def test_forward_segment_shape(self):
        mg, config = _make_graph()
        BS, T, D = 2, config.T, config.D
        H_aug = torch.randn(BS, T, D, dtype=torch.bfloat16)
        mem_out = mg.forward_segment(H_aug)
        assert mem_out.shape == (BS, T, D)

    def test_readout_shape(self):
        mg, config = _make_graph()
        BS = 2
        msg = torch.randn(
            BS, config.N_cells, config.neurons_per_cell, config.D_n,
            dtype=torch.bfloat16)
        readout = mg._readout(msg)
        assert readout.shape == (BS, config.D)

    def test_receive_shape(self):
        mg, config = _make_graph()
        BS = 2
        msg = torch.randn(
            BS, config.N_cells, config.neurons_per_cell, config.D_n,
            dtype=torch.bfloat16)
        W = torch.randn(
            BS, config.N_cells, config.neurons_per_cell, config.neurons_per_cell,
            dtype=torch.bfloat16)
        received = mg._receive(msg, W)
        assert received.shape == msg.shape

    def test_W_initialized_sparse(self):
        mg, config = _make_graph()
        W = mg.W
        # Each row should have approximately K nonzeros
        nonzeros_per_row = (W[0, 0].abs() > 0.01).float().sum(dim=-1)
        assert nonzeros_per_row.mean().item() == pytest.approx(config.K, abs=1)


class TestModulator:
    def test_near_zero_deltas_at_init(self):
        mg, config = _make_graph()
        dt = torch.bfloat16
        mod_w1 = mg.mod_w1.to(dt)
        mod_b1 = mg.mod_b1.to(dt)
        mod_w2 = mg.mod_w2.to(dt)
        mod_b2 = mg.mod_b2.to(dt)

        W_before = mg.W.clone()
        decay_before = mg.decay_logit.clone()

        new_W, new_dec, new_ctx, new_border = mg._modulate_cells(
            mg.h, mg.msg, mg.W, mg.decay_logit, mg.cell_context,
            mg.border_gate_logit, mod_w1, mod_b1, mod_w2, mod_b2)

        dw = (new_W - W_before).float().abs().max().item()
        dd = (new_dec - decay_before).float().abs().max().item()
        assert dw < 1.0, f"dW too large: {dw}"
        assert dd < 1.0, f"ddecay too large: {dd}"

    def test_low_rank_delta_W_shape(self):
        """Modulator should produce valid W update."""
        mg, config = _make_graph()
        dt = torch.bfloat16
        mod_w1 = mg.mod_w1.to(dt)
        mod_b1 = mg.mod_b1.to(dt)
        mod_w2 = mg.mod_w2.to(dt)
        mod_b2 = mg.mod_b2.to(dt)

        new_W, _, _, _ = mg._modulate_cells(
            mg.h, mg.msg, mg.W, mg.decay_logit, mg.cell_context,
            mg.border_gate_logit, mod_w1, mod_b1, mod_w2, mod_b2)
        assert new_W.shape == mg.W.shape


class TestInjectReadout:
    def test_roundtrip(self):
        mg, config = _make_graph()
        BS = 2
        H_aug_t = torch.randn(BS, config.D, dtype=torch.bfloat16)
        received = torch.zeros(
            BS, config.N_cells, config.neurons_per_cell, config.D_n,
            dtype=torch.bfloat16)
        gi = mg.cell_to_group
        inject_w = mg.inject_w[gi].to(torch.bfloat16)
        inject_b = mg.inject_b[gi].to(torch.bfloat16)
        result = mg._inject(received, H_aug_t, inject_w, inject_b)

        assert result[:, :, :config.alpha].abs().sum() > 0
        assert result[:, :, config.alpha:].abs().sum() == 0


class TestStateDecay:
    def test_high_decay_preserves_state(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        h_orig = torch.randn(
            BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        received = torch.zeros_like(h_orig)
        identity = mg._identity(BS, dt, torch.device("cpu"))
        decay = torch.sigmoid(torch.full(
            (BS, config.N_cells, config.neurons_per_cell), 10.0, dtype=dt
        )).unsqueeze(-1)

        gi = mg.cell_to_group
        args = (
            mg.state_w1.to(dt), mg.state_b1.to(dt),
            mg.state_gs1[gi].to(dt), mg.state_gb1[gi].to(dt),
            mg.state_w2.to(dt), mg.state_b2.to(dt),
            mg.state_gs2[gi].to(dt), mg.state_gb2[gi].to(dt),
        )
        h_new = mg._state_update(received, h_orig, decay, identity, *args)
        diff = (h_new - h_orig).float().abs().max().item()
        assert diff < 0.05, f"State changed too much with high decay: {diff}"


class TestWDecay:
    def test_soft_sparsity(self):
        """W entries should decay toward zero when modulator is disabled."""
        config = _tiny_config(modulation_interval=9999, w_decay_rate=0.05)  # large decay, disable mod
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))
        W_before = mg.W.clone()

        BS, T, D = 2, config.T, config.D
        H_aug = torch.randn(BS, T, D, dtype=torch.bfloat16)

        mg.train(False)
        with torch.no_grad():
            mg.forward_segment(H_aug)

        # W should have decayed since modulator didn't add anything
        ratio = mg.W.float().abs().mean() / W_before.float().abs().mean()
        expected = (1.0 - config.w_decay_rate) ** config.T
        assert ratio < 1.0, f"W did not decay (ratio={ratio:.4f})"
        assert ratio < expected + 0.01, f"W decayed less than expected ({ratio:.4f} vs {expected:.4f})"


class TestBorderExchange:
    def test_border_exchange_shape(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16
        msg = torch.randn(
            BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        border_gate = torch.sigmoid(torch.zeros(
            BS, config.N_cells, config.border_per_cell, dtype=dt)).unsqueeze(-1)
        result = mg._border_exchange(msg, border_gate)
        assert result.shape == (BS, config.N_cells, config.border_per_cell, config.D_n)


class TestGradientFlow:
    def test_grad_reaches_modulator(self):
        config = _tiny_config()
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))

        H_aug = torch.randn(2, config.T, config.D, dtype=torch.bfloat16)
        mem_out = mg.forward_segment(H_aug)
        loss = mem_out.sum()
        loss.backward()

        assert mg.mod_w1.grad is not None, "mod_w1 should have gradient"
        assert mg.mod_w1.grad.abs().sum() > 0, "mod_w1 gradient should be nonzero"

    def test_grad_reaches_shared_mlps(self):
        config = _tiny_config()
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))

        H_aug = torch.randn(2, config.T, config.D, dtype=torch.bfloat16)
        mem_out = mg.forward_segment(H_aug)
        loss = mem_out.sum()
        loss.backward()

        assert mg.state_w1.grad is not None
        assert mg.msg_w1.grad is not None
        assert mg.state_w1.grad.abs().sum() > 0
        assert mg.msg_w1.grad.abs().sum() > 0

    def test_grad_flows_through_W(self):
        """Gradient should flow through W to the modulator."""
        config = _tiny_config()
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))

        H_aug = torch.randn(2, config.T, config.D, dtype=torch.bfloat16)
        mem_out = mg.forward_segment(H_aug)
        mem_out.sum().backward()

        # The modulator produces delta_W which modifies W.
        # Gradient should flow: loss → readout → msg → h → received = W @ msg → W → delta_W → mod_w2
        assert mg.mod_w2.grad is not None
        assert mg.mod_w2.grad.abs().sum() > 0
