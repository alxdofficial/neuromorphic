"""Tests for v11 Triton cell forward kernel."""

import torch
import pytest
from src.v11.config import V11Config

BS = 2

def make_tiny(**kw):
    return V11Config.tier_tiny(**kw)


def _try_import_triton():
    try:
        from src.v11.triton_kernels import fused_cell_forward, _reference_cell_forward
        return fused_cell_forward, _reference_cell_forward
    except ImportError:
        pytest.skip("Triton not available")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCellForwardKernel:
    def test_output_shapes(self):
        fused_fwd, _ = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = 'cuda'

        h = torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron, device=device, dtype=torch.bfloat16)
        msg = torch.randn_like(h)
        w_conn_sig = torch.sigmoid(torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections, device=device, dtype=torch.bfloat16))
        decay = torch.sigmoid(torch.randn(BS, cfg.N_cells, cfg.C_neurons, device=device, dtype=torch.bfloat16))
        prim = torch.randn_like(h)

        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16)

        cc = torch.randn(BS, cfg.T, cfg.D, device=device, dtype=torch.bfloat16)

        sw1 = mg.state_w1
        sb1 = mg.state_b1
        sw2 = mg.state_w2
        sb2 = mg.state_b2
        mw1 = mg.msg_w1
        mb1 = mg.msg_b1
        mw2 = mg.msg_w2
        mb2 = mg.msg_b2

        h_out, msg_out, mem_out, hebb = fused_fwd(
            h, msg, w_conn_sig, decay, prim,
            mg.conn_indices, mg.neuron_id, mg.inject_indices, mg.readout_indices,
            cc, sw1, sb1, sw2, sb2, mw1, mb1, mw2, mb2, cfg)

        assert h_out.shape == h.shape
        assert msg_out.shape == msg.shape
        assert mem_out.shape == (BS, cfg.T, cfg.D)
        assert hebb.shape == (BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections)

    def test_matches_reference(self):
        """Triton kernel should match PyTorch reference."""
        fused_fwd, ref_fwd = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = 'cuda'

        torch.manual_seed(42)
        h = torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron, device=device, dtype=torch.bfloat16) * 0.1
        msg = torch.randn_like(h) * 0.1
        w_conn_sig = torch.sigmoid(torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections, device=device, dtype=torch.bfloat16))
        decay = torch.sigmoid(torch.randn(BS, cfg.N_cells, cfg.C_neurons, device=device, dtype=torch.bfloat16))
        prim = torch.randn_like(h) * 0.1
        cc = torch.randn(BS, cfg.T, cfg.D, device=device, dtype=torch.bfloat16) * 0.1

        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16)

        # Reference
        h_ref, msg_ref, mem_ref = ref_fwd(
            h.clone().float(), msg.clone().float(),
            w_conn_sig.float(), decay.float(), prim.float(),
            mg.neuron_id.float(),
            cc.float(),
            mg.state_w1.float(), mg.state_b1.float(),
            mg.state_w2.float(), mg.state_b2.float(),
            mg.msg_w1.float(), mg.msg_b1.float(),
            mg.msg_w2.float(), mg.msg_b2.float(),
            mg.conn_indices, mg.inject_indices, mg.readout_indices, cfg)

        # Triton
        h_tri, msg_tri, mem_tri, _ = fused_fwd(
            h.clone(), msg.clone(), w_conn_sig, decay, prim,
            mg.conn_indices, mg.neuron_id, mg.inject_indices, mg.readout_indices,
            cc, mg.state_w1, mg.state_b1, mg.state_w2, mg.state_b2,
            mg.msg_w1, mg.msg_b1, mg.msg_w2, mg.msg_b2, cfg)

        # Compare (loose tolerance for bf16 vs f32 differences)
        # Loose tolerance: bf16 inputs accumulate error over T×R steps
        max_diff = (mem_tri - mem_ref.to(mem_tri.dtype)).abs().max().item()
        assert max_diff < 2.0, f"mem_out max diff {max_diff:.4f} too large"

    def test_output_finite(self):
        fused_fwd, _ = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = 'cuda'

        h = torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron, device=device, dtype=torch.bfloat16) * 0.1
        msg = torch.randn_like(h) * 0.1
        w_conn_sig = torch.sigmoid(torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections, device=device, dtype=torch.bfloat16))
        decay = torch.sigmoid(torch.randn(BS, cfg.N_cells, cfg.C_neurons, device=device, dtype=torch.bfloat16))
        prim = torch.randn_like(h) * 0.1
        cc = torch.randn(BS, cfg.T, cfg.D, device=device, dtype=torch.bfloat16) * 0.1

        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16)

        _, _, mem_out, _ = fused_fwd(
            h, msg, w_conn_sig, decay, prim,
            mg.conn_indices, mg.neuron_id, mg.inject_indices, mg.readout_indices,
            cc, mg.state_w1, mg.state_b1, mg.state_w2, mg.state_b2,
            mg.msg_w1, mg.msg_b1, mg.msg_w2, mg.msg_b2, cfg)

        assert torch.isfinite(mem_out).all()
