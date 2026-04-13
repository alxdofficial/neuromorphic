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
        W_gamma = torch.sigmoid(mg.W_decay_logit).to(dt)
        decay_gamma = torch.sigmoid(mg.decay_gamma_logit).to(dt)
        return (
            mg.mod_w1.to(dt), mg.mod_b1.to(dt),
            mg.mod_w2.to(dt), mg.mod_b2.to(dt),
            W_gamma, decay_gamma,
        )

    def _surprise_inputs(self, mg, BS):
        dt = torch.bfloat16
        return (
            torch.zeros(BS, mg.N_cells, 1, dtype=dt),   # readout_drift
            torch.zeros(BS, dtype=dt),                   # s_mem_live
            torch.zeros(BS, dtype=dt),                   # s_mem_ema_fast
        )

    def test_W_stays_unit_rms_after_update(self):
        """W is maintained at ~unit per-row RMS by the convex-EMA update
        regardless of modulator output scale. Same for decay staying in [0,1]."""
        mg, config = _make_graph()
        # Sanity: W init is already unit per-row RMS
        init_rms = mg.W.float().pow(2).mean(dim=-1).sqrt()
        assert (init_rms > 0.9).all() and (init_rms < 1.1).all(), \
            f"W init not unit RMS: min={init_rms.min()} max={init_rms.max()}"
        # Run one modulator update
        surp = self._surprise_inputs(mg, 2)
        mod_args = self._mod_args(mg, 2)
        new_W, new_decay = mg._modulate_cells(
            mg.h, mg.msg, mg.W, mg.decay, mg.hebbian,
            *surp,
            *mod_args)
        # W stays at ~unit per-row RMS
        new_rms = new_W.float().pow(2).mean(dim=-1).sqrt()
        assert (new_rms <= 1.01).all(), \
            f"W exceeds unit per-row RMS: max={new_rms.max()}"
        assert (new_rms > 0.3).all(), \
            f"W per-row RMS collapsed: min={new_rms.min()}"
        # decay stays in [0,1] by construction (convex comb of [0,1] values)
        assert (new_decay >= 0).all() and (new_decay <= 1).all(), \
            f"decay out of [0,1]: min={new_decay.min()} max={new_decay.max()}"

    def test_delta_W_shape(self):
        mg, config = _make_graph()
        surp = self._surprise_inputs(mg, 2)
        mod_args = self._mod_args(mg, 2)
        new_W, _ = mg._modulate_cells(
            mg.h, mg.msg, mg.W, mg.decay, mg.hebbian,
            *surp, *mod_args)
        assert new_W.shape == mg.W.shape


class TestInjectReadout:
    def test_roundtrip(self):
        mg, config = _make_graph()
        BS = 2
        H_aug_t = torch.randn(BS, config.D, dtype=torch.bfloat16)
        received = torch.zeros(
            BS, config.N_cells, config.neurons_per_cell, config.D_n,
            dtype=torch.bfloat16)
        inject_w = mg.inject_w.to(torch.bfloat16)
        inject_b = mg.inject_b.to(torch.bfloat16)
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
        h_new = mg._state_update(received, h_orig, decay, one_minus_decay, *args)
        diff = (h_new - h_orig).float().abs().max().item()
        assert diff < 0.05, f"State changed too much with high decay: {diff}"


class TestWBoundedness:
    def test_W_stays_bounded_over_many_updates(self):
        """Repeated modulator updates must not let W drift — the convex-EMA
        update keeps per-row RMS bounded by 1 for all time, which is the
        structural fix for the bf16-overflow bug that the accumulator had."""
        config = _tiny_config()
        mg = MemoryGraph(config)
        mg.initialize_states(2, torch.device("cpu"))
        lm = _StubLM(config)
        # Drive many segments through the memory to stress the update loop.
        for _ in range(20):
            H_mid = torch.randn(2, config.T, config.D, dtype=torch.bfloat16) * 3.0
            input_ids = torch.randint(0, config.vocab_size, (2, config.T))
            mg.forward_segment(H_mid, input_ids, lm)
            row_rms = mg.W.float().pow(2).mean(dim=-1).sqrt()
            assert (row_rms <= 1.05).all(), \
                f"W per-row RMS exceeded 1: max={row_rms.max().item()}"
            assert mg.W.float().abs().max().item() < 100.0, \
                "W grew unreasonably large"


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
