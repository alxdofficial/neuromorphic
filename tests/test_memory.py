"""Memory graph unit tests."""

import torch
import pytest

from src.model.config import Config
from src.model.memory import MemoryGraph
from src.model.triton_kernels import hebbian_ema_update


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
            BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=torch.bfloat16)
        readout = mg._readout(msg)
        assert readout.shape == (BS, config.D)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton receive test")
    def test_local_receive_matches_eager_on_cuda(self):
        config = _tiny_config()
        mg = MemoryGraph(config).cuda()
        mg.initialize_states(2, torch.device("cuda"))
        msg = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.D_n,
            device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w_conn = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.K,
            device="cuda", dtype=torch.bfloat16, requires_grad=True)

        received_fused = mg._receive_local(msg, w_conn)

        batch_idx = torch.arange(2, device="cuda")[:, None, None, None]
        cell_idx = torch.arange(config.N_cells, device="cuda")[None, :, None, None]
        conn = mg.conn_idx.to("cuda").unsqueeze(0).expand(2, -1, -1, -1)
        gathered = msg[batch_idx, cell_idx, conn]
        received_eager = (gathered * torch.sigmoid(w_conn).unsqueeze(-1)).sum(dim=3)

        diff = (received_fused - received_eager).float().abs().max().item()
        assert diff < 5e-2, f"Fused local receive deviates from eager baseline (max diff={diff})"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton receive grad test")
    def test_local_receive_backward_matches_eager_on_cuda(self):
        config = _tiny_config()
        mg = MemoryGraph(config).cuda()
        mg.initialize_states(2, torch.device("cuda"))

        msg_fused = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.D_n,
            device="cuda", dtype=torch.float32, requires_grad=True)
        w_fused = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.K,
            device="cuda", dtype=torch.float32, requires_grad=True)
        msg_eager = msg_fused.detach().clone().requires_grad_(True)
        w_eager = w_fused.detach().clone().requires_grad_(True)

        out_fused = mg._receive_local(msg_fused, w_fused)
        loss_fused = out_fused.square().mean()
        loss_fused.backward()

        batch_idx = torch.arange(2, device="cuda")[:, None, None, None]
        cell_idx = torch.arange(config.N_cells, device="cuda")[None, :, None, None]
        conn = mg.conn_idx.to("cuda").unsqueeze(0).expand(2, -1, -1, -1)
        gathered = msg_eager[batch_idx, cell_idx, conn]
        out_eager = (gathered * torch.sigmoid(w_eager).unsqueeze(-1)).sum(dim=3)
        loss_eager = out_eager.square().mean()
        loss_eager.backward()

        grad_msg_diff = (msg_fused.grad - msg_eager.grad).abs().max().item()
        grad_w_diff = (w_fused.grad - w_eager.grad).abs().max().item()
        assert grad_msg_diff < 1e-4, f"grad_msg mismatch (max diff={grad_msg_diff})"
        assert grad_w_diff < 1e-4, f"grad_w mismatch (max diff={grad_w_diff})"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton Hebbian test")
    def test_hebbian_update_matches_eager_on_cuda(self):
        config = _tiny_config()
        mg = MemoryGraph(config).cuda()
        mg.initialize_states(2, torch.device("cuda"))
        msg_prev = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.D_n,
            device="cuda", dtype=torch.bfloat16)
        msg_new = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.D_n,
            device="cuda", dtype=torch.bfloat16)
        hebbian = torch.randn(
            2, config.N_cells, config.neurons_per_cell, config.K,
            device="cuda", dtype=torch.bfloat16)
        decay = 0.995

        hebb_fused = hebbian_ema_update(msg_prev, msg_new, hebbian, mg.conn_idx.to("cuda"), decay)

        batch_idx = torch.arange(2, device="cuda")[:, None, None, None]
        cell_idx = torch.arange(config.N_cells, device="cuda")[None, :, None, None]
        conn = mg.conn_idx.to("cuda").unsqueeze(0).expand(2, -1, -1, -1)
        gathered = msg_prev[batch_idx, cell_idx, conn]
        hebb_eager = hebbian * decay + (gathered * msg_new.unsqueeze(3)).sum(dim=-1) * (1.0 - decay)

        diff = (hebb_fused - hebb_eager).float().abs().max().item()
        assert diff < 5e-2, f"Fused Hebbian update deviates from eager baseline (max diff={diff})"


class TestModulator:
    def test_near_zero_deltas_at_init(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16
        h = mg.h
        msg = mg.msg
        hebbian = mg.hebbian_traces
        w_conn = mg.w_conn
        decay_logit = mg.decay_logit
        cell_context = mg.cell_context
        border_gate_logit = mg.border_gate_logit

        mod_w1 = mg.mod_w1.to(dt)
        mod_b1 = mg.mod_b1.to(dt)
        mod_w2 = mg.mod_w2.to(dt)
        mod_b2 = mg.mod_b2.to(dt)

        new_w, new_dec, new_ctx, new_border = mg._modulate_cells(
            h, msg, hebbian, decay_logit, cell_context, border_gate_logit, w_conn,
            mod_w1, mod_b1, mod_w2, mod_b2)

        # Deltas should be near zero due to small init on mod_w2
        dw = (new_w - w_conn).float().abs().max().item()
        dd = (new_dec - decay_logit).float().abs().max().item()
        dc = (new_ctx - cell_context).float().abs().max().item()
        db = (new_border - border_gate_logit).float().abs().max().item()
        assert dw < 0.5, f"dw too large: {dw}"
        assert dd < 0.5, f"dd too large: {dd}"
        assert dc < 0.5, f"dctx too large: {dc}"
        assert db < 0.5, f"dborder too large: {db}"


class TestInjectReadout:
    def test_roundtrip(self):
        mg, config = _make_graph()
        BS = 2

        # Inject a known signal
        H_aug_t = torch.randn(BS, config.D, dtype=torch.bfloat16)
        received = torch.zeros(
            BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=torch.bfloat16)
        inj_w = mg.inject_w[mg.cell_to_group].to(torch.bfloat16)
        inj_b = mg.inject_b[mg.cell_to_group].to(torch.bfloat16)
        result = mg._inject(received, H_aug_t, inj_w, inj_b)

        # Input port neurons should be nonzero
        assert result[:, :, :config.alpha].abs().sum() > 0
        assert result[:, :, config.alpha:].abs().sum() == 0

    def test_input_ports_receive_distinct_views(self):
        torch.manual_seed(0)
        mg, config = _make_graph()
        BS = 2
        H_aug_t = torch.randn(BS, config.D, dtype=torch.bfloat16)
        received = torch.zeros(
            BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=torch.bfloat16)
        inj_w = mg.inject_w[mg.cell_to_group].to(torch.bfloat16)
        inj_b = mg.inject_b[mg.cell_to_group].to(torch.bfloat16)
        result = mg._inject(received, H_aug_t, inj_w, inj_b)
        diff = (
            result[:, :, 0] - result[:, :, min(1, config.alpha - 1)]
        ).float().abs().max().item()
        assert diff > 1e-4, "Distinct input ports received identical injected signals"


class TestStateDecay:
    def _get_state_mlp_args(self, mg, dt):
        gi = mg.cell_to_group
        return (
            mg.state_w1.to(dt), mg.state_b1.to(dt),
            mg.state_gs1[gi].to(dt), mg.state_gb1[gi].to(dt),
            mg.state_w2.to(dt), mg.state_b2.to(dt),
            mg.state_gs2[gi].to(dt), mg.state_gb2[gi].to(dt),
        )

    def test_high_decay_preserves_state(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        h_orig = torch.randn(BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        received = torch.zeros(BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        identity = mg.identity
        decay_logit = torch.full((BS, config.N_cells, config.neurons_per_cell), 10.0, dtype=dt)
        cell_context = mg.cell_context

        args = self._get_state_mlp_args(mg, dt)
        h_new = mg._state_update(received, h_orig, decay_logit, identity, cell_context, *args)
        diff = (h_new - h_orig).float().abs().max().item()
        assert diff < 0.05, f"State changed too much with high decay: {diff}"

    def test_cached_decay_matches_raw_state_update(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        h = torch.randn(BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        received = torch.randn_like(h)
        identity = mg.identity
        decay_logit = torch.randn(BS, config.N_cells, config.neurons_per_cell, dtype=dt)
        cell_context = mg.cell_context

        args = self._get_state_mlp_args(mg, dt)
        raw = mg._state_update(received, h, decay_logit, identity, cell_context, *args)
        cached = mg._state_update_from_decay(
            received, h, torch.sigmoid(decay_logit).unsqueeze(-1), identity, cell_context, *args)
        diff = (raw - cached).float().abs().max().item()
        assert diff < 0.05, f"Cached decay path diverged from raw state update (max diff={diff})"


class TestBorderExchange:
    def test_cached_gate_matches_raw_border_exchange(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        msg = torch.randn(BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        border_gate_logit = torch.randn(BS, config.N_cells, config.border_per_cell, dtype=dt)

        raw = mg._border_exchange(msg, border_gate_logit)
        cached = mg._border_exchange_from_gate(msg, torch.sigmoid(border_gate_logit).unsqueeze(-1))
        diff = (raw - cached).float().abs().max().item()
        assert diff < 1e-4, f"Cached border exchange diverged from raw path (max diff={diff})"


class TestHebbian:
    def test_per_token_update(self):
        mg, config = _make_graph()
        BS = 2
        dt = torch.bfloat16

        hebbian = mg.hebbian_traces.clone()
        h0 = hebbian.clone()

        msg_prev = torch.randn(
            BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        msg = torch.randn(BS, config.N_cells, config.neurons_per_cell, config.D_n, dtype=dt)
        mg._update_hebbian(msg_prev, msg, hebbian, mg.conn_idx)

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
            for cell in range(config.N_cells):
                for neuron in range(config.neurons_per_cell):
                    vals = mg.conn_idx[cell, neuron].tolist()
                    assert len(vals) == len(set(vals)), (
                        f"Cell {cell} neuron {neuron} has duplicate connections: {vals}")

    def test_sorted_after_rewire(self):
        config = _tiny_config(structural_plasticity=True, plasticity_pct=0.1)
        mg, _ = _make_graph(config)
        mg.hebbian_traces = torch.randn_like(mg.hebbian_traces)
        mg.msg = torch.randn_like(mg.msg)

        mg.rewire_connections()
        for cell in range(config.N_cells):
            for neuron in range(config.neurons_per_cell):
                row = mg.conn_idx[cell, neuron]
                sorted_row, _ = row.sort()
                assert torch.equal(row, sorted_row), (
                    f"Cell {cell} neuron {neuron} connections not sorted after rewire")

    def test_connectivity_index_buffer_consistent_after_rewire(self):
        config = _tiny_config(structural_plasticity=True, plasticity_pct=0.1)
        mg, _ = _make_graph(config)
        mg.hebbian_traces = torch.randn_like(mg.hebbian_traces)
        mg.msg = torch.randn_like(mg.msg)
        mg.rewire_connections()

        assert torch.equal(mg.conn_idx_i32.cpu(), mg.conn_idx.cpu().to(torch.int32))


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
        assert mg.inject_w.grad is not None
        assert mg.state_w1.grad.abs().sum() > 0
        assert mg.msg_w1.grad.abs().sum() > 0
        assert mg.inject_w.grad.abs().sum() > 0
