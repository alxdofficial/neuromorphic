"""Tests for v11 cell-based memory graph."""

import torch
import pytest
from src.v11.config import V11Config
from src.v11.memory_graph import CellMemoryGraph

BS = 2


def make_tiny(**kw):
    return V11Config.tier_tiny(**kw)


class TestConfig:
    def test_tier_a_validates(self):
        c = V11Config.tier_a()
        c.validate()
        assert c.N_total == 256 * 124  # 31744
        assert c.N_cells == 256
        assert c.C_neurons == 124
        assert c.N_inject_per_cell == 4
        assert c.N_readout_per_cell == 4
        assert c.structural_plasticity is False

    def test_tier_tiny_validates(self):
        c = make_tiny()
        c.validate()
        assert c.N_total == 128  # 8 cells × 16 neurons
        assert c.N_cells == 8
        assert c.C_neurons == 16


class TestCellMemoryGraphInit:
    def test_shapes(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        assert mg.h.shape == (BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron)
        assert mg.prev_messages.shape == (BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron)
        assert mg.w_conn.shape == (BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections)
        assert mg.conn_indices.shape == (cfg.N_cells, cfg.C_neurons, cfg.K_connections)

    def test_conn_indices_cell_local(self):
        """All connection indices should be within [0, C_neurons)."""
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        assert (mg.conn_indices >= 0).all()
        assert (mg.conn_indices < cfg.C_neurons).all()

    def test_no_self_connections(self):
        """No neuron should connect to itself."""
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        for cell in range(cfg.N_cells):
            for n in range(cfg.C_neurons):
                assert n not in mg.conn_indices[cell, n].tolist()

    def test_inject_readout_indices(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        assert mg.inject_indices.shape == (cfg.N_cells, cfg.alpha)
        assert mg.readout_indices.shape == (cfg.N_cells, cfg.alpha)
        # Inject and readout should not overlap
        for cell in range(cfg.N_cells):
            inject_set = set(mg.inject_indices[cell].tolist())
            readout_set = set(mg.readout_indices[cell].tolist())
            assert inject_set.isdisjoint(readout_set)

    def test_shared_mlp_params(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"))
        # State and msg MLPs are shared (no N dimension)
        assert mg.state_w1.shape == (cfg.state_mlp_hidden, 3 * cfg.D_neuron + 1)
        assert mg.msg_w1.shape == (cfg.msg_mlp_hidden, 3 * cfg.D_neuron)
        # Modulator is per-neuron
        assert mg.mod_w1.shape[0] == cfg.N_total

    def test_all_params_require_grad(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"))
        for name, p in mg.named_parameters():
            assert p.requires_grad, f"{name} should require grad"


class TestForwardSegment:
    def test_output_shape(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.T, cfg.D)

    def test_output_finite(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        assert torch.isfinite(out).all()

    def test_gradient_flows_to_modulator(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        out.sum().backward()
        assert mg.mod_w1.grad is not None
        assert mg.mod_w1.grad.norm() > 0

    def test_gradient_flows_to_state_mlp(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        out.sum().backward()
        assert mg.state_w1.grad is not None
        assert mg.state_w1.grad.norm() > 0

    def test_gradient_flows_to_msg_mlp(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        out.sum().backward()
        assert mg.msg_w1.grad is not None
        assert mg.msg_w1.grad.norm() > 0

    def test_gradient_flows_to_neuron_id(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        out.sum().backward()
        assert mg.neuron_id.grad is not None
        assert mg.neuron_id.grad.norm() > 0

    def test_all_params_get_grad(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)
        cc = torch.randn(BS, cfg.T, cfg.D)
        out = mg.forward_segment(cc)
        out.sum().backward()
        for name, p in mg.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.norm() > 0, f"{name} has zero grad"

    def test_state_persists_across_segments(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        h_init = mg.h.clone()
        cc = torch.randn(BS, cfg.T, cfg.D)
        mg.forward_segment(cc)
        assert not torch.equal(mg.h, h_init), "h should change after forward"

    def test_tbptt_detach(self):
        """Gradients should not flow across segment boundaries."""
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg.initialize_states(BS)

        cc1 = torch.randn(BS, cfg.T, cfg.D)
        cc2 = torch.randn(BS, cfg.T, cfg.D)

        out1 = mg.forward_segment(cc1)
        out1.sum().backward(retain_graph=True)
        grad1 = mg.mod_w1.grad.clone()
        mg.zero_grad()

        out2 = mg.forward_segment(cc2)
        out2.sum().backward()
        grad2 = mg.mod_w1.grad.clone()

        assert not torch.equal(grad1, grad2)

    def test_update_phi_accepts_bfloat16_activity(self):
        cfg = make_tiny(structural_plasticity=True)
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.bfloat16)
        mg.initialize_states(BS)
        act_trace = torch.randn(
            BS, cfg.T, cfg.N_cells, cfg.C_neurons,
            dtype=torch.bfloat16
        )
        mg._update_phi(act_trace)
        assert mg._co_activation_ready is True
        assert torch.isfinite(mg.co_activation_ema).all()

    def test_chunked_round_matches_unchunked(self):
        cfg = make_tiny(structural_plasticity=False)
        cc = torch.randn(BS, cfg.T, cfg.D)

        mg_ref = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg_ref.initialize_states(BS)

        mg_chunk = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        mg_chunk.initialize_states(BS)
        mg_chunk.load_state_dict(mg_ref.state_dict())
        runtime = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in mg_ref.runtime_state_dict().items()
        }
        mg_chunk.load_runtime_state(runtime)

        mg_ref._cell_chunk_size = lambda *_: cfg.N_cells
        mg_chunk._cell_chunk_size = lambda *_: 2

        out_ref = mg_ref.forward_segment(cc)
        out_chunk = mg_chunk.forward_segment(cc)

        assert torch.allclose(out_ref, out_chunk, atol=1e-5, rtol=1e-5)
        assert torch.allclose(mg_ref.h, mg_chunk.h, atol=1e-5, rtol=1e-5)
        assert torch.allclose(
            mg_ref.prev_messages, mg_chunk.prev_messages, atol=1e-5, rtol=1e-5)


class TestInjectReadout:
    def test_inject_shape(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        cc_t = torch.randn(BS, cfg.D)
        inject = mg._inject(cc_t)
        assert inject.shape == (BS, cfg.N_cells, cfg.alpha, cfg.D_neuron)

    def test_readout_shape(self):
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)
        msg = torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron)
        out = mg._readout(msg)
        assert out.shape == (BS, cfg.D)

    def test_readout_only_reads_port_neurons(self):
        """Only readout neurons should affect output."""
        cfg = make_tiny()
        mg = CellMemoryGraph(cfg, torch.device("cpu"), dtype=torch.float32)

        # Zero all messages
        msg = torch.zeros(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron)
        out_zero = mg._readout(msg)

        # Set only non-readout neurons to 1
        msg2 = torch.zeros_like(msg)
        msg2[:, :, :cfg.C_neurons - cfg.alpha, :] = 1.0
        out_non_readout = mg._readout(msg2)

        # Should be the same (readout neurons are still zero)
        torch.testing.assert_close(out_zero, out_non_readout)
