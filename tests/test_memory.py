"""Memory graph unit tests."""

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
        msg = torch.randn(BS, config.N, config.D_n, dtype=torch.bfloat16)
        readout = mg._readout(msg)
        assert readout.shape == (BS, config.D)


class TestModulator:
    def test_near_zero_deltas_at_init(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16
        identity = mg.identity
        hebbian = mg.hebbian_traces
        w_conn = mg.w_conn
        decay_logit = mg.decay_logit

        mod_w1 = mg.mod_w1.to(dt)
        mod_b1 = mg.mod_b1.to(dt)
        mod_w2 = mg.mod_w2.to(dt)
        mod_b2 = mg.mod_b2.to(dt)

        new_w, new_dec, new_id = mg._modulate(
            identity, hebbian, w_conn, decay_logit,
            mod_w1, mod_b1, mod_w2, mod_b2)

        # Deltas should be near zero due to small init on mod_w2
        dw = (new_w - w_conn).float().abs().max().item()
        dd = (new_dec - decay_logit).float().abs().max().item()
        di = (new_id - identity).float().abs().max().item()
        assert dw < 0.5, f"dw too large: {dw}"
        assert dd < 0.5, f"dd too large: {dd}"
        assert di < 0.5, f"di too large: {di}"


class TestInjectReadout:
    def test_roundtrip(self):
        mg, config = _make_graph()
        BS = 2

        # Inject a known signal
        H_aug_t = torch.randn(BS, config.D, dtype=torch.bfloat16)
        received = torch.zeros(BS, config.N, config.D_n, dtype=torch.bfloat16)
        result = mg._inject(received, H_aug_t)

        # Input port neurons should be nonzero
        assert result[:, :config.N_port].abs().sum() > 0
        # Non-port neurons should still be zero
        assert result[:, 2 * config.N_port:].abs().sum() == 0


class TestStateDecay:
    def test_high_decay_preserves_state(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        h_orig = torch.randn(BS, config.N, config.D_n, dtype=dt)
        received = torch.zeros(BS, config.N, config.D_n, dtype=dt)
        identity = mg.identity
        # Very high decay_logit → decay ≈ 1 → h_new ≈ h_old
        decay_logit = torch.full((BS, config.N), 10.0, dtype=dt)

        w1 = mg.state_w1.to(dt)
        b1 = mg.state_b1.to(dt)
        w2 = mg.state_w2.to(dt)
        b2 = mg.state_b2.to(dt)

        h_new = mg._state_update(received, h_orig, identity, decay_logit,
                                 w1, b1, w2, b2)
        diff = (h_new - h_orig).float().abs().max().item()
        assert diff < 0.05, f"State changed too much with high decay: {diff}"


class TestHebbian:
    def test_per_token_update(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        hebbian = mg.hebbian_traces.clone()
        h0 = hebbian.clone()

        gathered = torch.randn(BS, config.N, config.K, config.D_n, dtype=dt)
        msg = torch.randn(BS, config.N, config.D_n, dtype=dt)
        mg._update_hebbian(gathered, msg, hebbian)

        assert not torch.equal(hebbian, h0), "Hebbian traces should change after update"

    def test_modulator_sees_fresh_traces(self):
        """Run 2 steps, verify modulator at step 2 uses step-1 updated traces."""
        mg, config = _make_graph()
        BS, T, D = 2, 2, config.D

        H_aug = torch.randn(BS, T, D, dtype=torch.bfloat16)
        hebb_before = mg.hebbian_traces.clone()
        mg.forward_segment(H_aug)
        hebb_after = mg.hebbian_traces

        # After 2 token steps, hebbian should have changed
        assert not torch.equal(hebb_before, hebb_after)


class TestStructuralPlasticity:
    def test_connections_change(self):
        config = _tiny_config(structural_plasticity=True, plasticity_pct=0.1)
        mg, _ = _make_graph(config)
        conn_before = mg.conn_idx.clone()

        # Need some nonzero hebbian for pruning to have signal
        mg.hebbian_traces = torch.randn_like(mg.hebbian_traces)
        mg.msg = torch.randn_like(mg.msg)
        mg.rewire_connections()

        changed = (conn_before != mg.conn_idx).any().item()
        assert changed, "Connections should change after rewire"

    def test_no_duplicates(self):
        config = _tiny_config(structural_plasticity=True, plasticity_pct=0.1)
        mg, _ = _make_graph(config)
        mg.hebbian_traces = torch.randn_like(mg.hebbian_traces)
        mg.msg = torch.randn_like(mg.msg)

        for _ in range(10):
            mg.rewire_connections()
            for n in range(config.N):
                vals = mg.conn_idx[n].tolist()
                assert len(vals) == len(set(vals)), (
                    f"Neuron {n} has duplicate connections: {vals}")

    def test_sorted_after_rewire(self):
        config = _tiny_config(structural_plasticity=True, plasticity_pct=0.1)
        mg, _ = _make_graph(config)
        mg.hebbian_traces = torch.randn_like(mg.hebbian_traces)
        mg.msg = torch.randn_like(mg.msg)

        mg.rewire_connections()
        for n in range(config.N):
            row = mg.conn_idx[n]
            sorted_row, _ = row.sort()
            assert torch.equal(row, sorted_row), (
                f"Neuron {n} connections not sorted after rewire")


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
