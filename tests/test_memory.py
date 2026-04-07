"""Memory graph unit tests (dense-W design)."""

import torch
import torch.nn as nn
import pytest

from src.model.config import Config
from src.model.memory import MemoryGraph


def _tiny_config(**kw):
    return Config.tier_tiny(**kw)


class _StubLM(nn.Module):
    """Minimal LM stub for memory-graph tests (runs in bf16)."""
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.D_embed, config.vocab_size, bias=False).to(torch.bfloat16)
        self.proj_down = (
            nn.Linear(config.D, config.D_embed).to(torch.bfloat16)
            if config.D != config.D_embed else None)
        self.ln_final = nn.LayerNorm(config.D_embed).to(torch.bfloat16)

    def mem_head_target_logit(self, readout, target):
        x = readout.to(torch.bfloat16)
        if self.proj_down is not None:
            x = self.proj_down(x)
        x = self.ln_final(x)
        target_emb = self.lm_head.weight[target]
        return (x * target_emb).sum(dim=-1)

    def mem_head_logits(self, readouts):
        x = readouts.to(torch.bfloat16)
        if self.proj_down is not None:
            x = self.proj_down(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def _make_graph(config=None, BS=2):
    if config is None:
        config = _tiny_config()
    mg = MemoryGraph(config)
    device = torch.device("cpu")
    mg.initialize_states(BS, device)
    return mg, config


def _run_segment(mg, config, BS=2):
    """Helper: run a full forward_segment with a stub LM."""
    lm = _StubLM(config)
    H_mid = torch.randn(BS, config.T, config.D, dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (BS, config.T))
    return mg.forward_segment(H_mid, input_ids, lm)


class TestShapes:
    def test_forward_segment_shape(self):
        mg, config = _make_graph()
        BS = 2
        mem_out, mem_loss = _run_segment(mg, config, BS)
        assert mem_out.shape == (BS, config.T, config.D)
        assert mem_loss.dim() == 0

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
    def _mod_args(self, mg, BS):
        dt = torch.bfloat16
        return (
            mg.mod_w1.to(dt), mg.mod_b1.to(dt),
            mg.mod_w2.to(dt), mg.mod_b2.to(dt),
        )

    def _surprise_inputs(self, mg, BS):
        dt = torch.bfloat16
        return (
            torch.zeros(BS, mg.N_cells, 1, dtype=dt),   # readout_drift
            torch.zeros(BS, dtype=dt),                   # s_mem_live
            torch.zeros(BS, dtype=dt),                   # s_mem_ema_fast
            torch.zeros(BS, dtype=dt),                   # s_progress
        )

    def test_near_zero_deltas_at_init(self):
        mg, config = _make_graph()
        W_before = mg.W.clone()
        decay_before = mg.decay_logit.clone()
        surp = self._surprise_inputs(mg, 2)
        mod_args = self._mod_args(mg, 2)
        new_W, new_dec, new_ctx, new_border = mg._modulate_cells(
            mg.h, mg.msg, mg.W, mg.decay_logit, mg.cell_context,
            mg.border_gate_logit,
            *surp,
            *mod_args)
        dw = (new_W - W_before).float().abs().max().item()
        dd = (new_dec - decay_before).float().abs().max().item()
        assert dw < 1.0, f"dW too large: {dw}"
        assert dd < 1.0, f"ddecay too large: {dd}"

    def test_delta_W_shape(self):
        mg, config = _make_graph()
        surp = self._surprise_inputs(mg, 2)
        mod_args = self._mod_args(mg, 2)
        new_W, _, _, _ = mg._modulate_cells(
            mg.h, mg.msg, mg.W, mg.decay_logit, mg.cell_context,
            mg.border_gate_logit, *surp, *mod_args)
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

        w1 = mg.state_w1.to(dt)
        w1_recv = w1[:, :config.D_n].contiguous()
        w1_h = w1[:, config.D_n:].contiguous()
        args = (w1_recv, w1_h, mg.state_b1.to(dt), mg.state_w2.to(dt), mg.state_b2.to(dt))
        one_minus_decay = 1.0 - decay
        h_new = mg._state_update(received, h_orig, decay, one_minus_decay, identity, *args)
        diff = (h_new - h_orig).float().abs().max().item()
        assert diff < 0.05, f"State changed too much with high decay: {diff}"


class TestWDecay:
    def test_soft_sparsity(self):
        """W entries should decay toward zero when modulator is disabled."""
        config = _tiny_config(modulation_interval=9999, w_decay_rate=0.05)
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))
        W_before = mg.W.clone()

        mg.train(False)
        with torch.no_grad():
            _run_segment(mg, config, BS=2)

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
        mem_out, _ = _run_segment(mg, config, BS=2)
        mem_out.sum().backward()
        assert mg.mod_w1.grad is not None
        assert mg.mod_w1.grad.abs().sum() > 0

    def test_grad_reaches_shared_mlps(self):
        config = _tiny_config()
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))
        mem_out, _ = _run_segment(mg, config, BS=2)
        mem_out.sum().backward()
        assert mg.state_w1.grad is not None
        assert mg.msg_w1.grad is not None
        assert mg.state_w1.grad.abs().sum() > 0
        assert mg.msg_w1.grad.abs().sum() > 0

    def test_grad_flows_through_W(self):
        config = _tiny_config()
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))
        mem_out, _ = _run_segment(mg, config, BS=2)
        mem_out.sum().backward()
        assert mg.mod_w2.grad is not None
        assert mg.mod_w2.grad.abs().sum() > 0
