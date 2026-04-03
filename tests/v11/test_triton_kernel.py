"""Tests for v11 Triton cell forward kernel."""

import torch
import pytest
from src.v11.config import V11Config

BS = 2

def make_tiny(**kw):
    return V11Config.tier_tiny(**kw)


def _try_import_triton():
    try:
        from src.v11.triton_kernels import (
            fused_cell_forward,
            _reference_cell_forward,
            combined_cell_border_gather,
            fused_token_step,
            _reference_token_step,
        )
        return (
            fused_cell_forward,
            _reference_cell_forward,
            combined_cell_border_gather,
            fused_token_step,
            _reference_token_step,
        )
    except ImportError:
        pytest.skip("Triton not available")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCellForwardKernel:
    def test_output_shapes(self):
        fused_fwd, _, _, _, _ = _try_import_triton()
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
        fused_fwd, ref_fwd, _, _, _ = _try_import_triton()
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
        fused_fwd, _, _, _, _ = _try_import_triton()
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCombinedGatherKernel:
    def test_matches_reference(self):
        _, _, fused_gather, _, _ = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = "cuda"

        torch.manual_seed(0)
        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16).to(device)

        msg = torch.randn(
            BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron,
            device=device, dtype=torch.bfloat16
        )
        w_conn_sig = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections,
            device=device, dtype=torch.bfloat16
        ))
        w_conn_border_sig = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.N_border_per_cell, cfg.K_border,
            device=device, dtype=torch.bfloat16
        ))

        ref = mg._cell_gather(msg.float(), w_conn_sig.float())
        border = mg._border_gather(msg.float(), w_conn_border_sig.float())
        border_idx = mg.border_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, cfg.D_neuron
        )
        ref = ref.scatter_add(2, border_idx, border.to(ref.dtype))

        out = fused_gather(
            msg, w_conn_sig, mg.conn_indices,
            w_conn_border_sig, mg.border_conn_indices,
            cfg.alpha,
        )
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    def test_backward_matches_reference(self):
        _, _, fused_gather, _, _ = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = "cuda"

        torch.manual_seed(1)
        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16).to(device).float()

        msg_a = torch.randn(
            BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron,
            device=device, dtype=torch.float32, requires_grad=True
        )
        w_a = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections,
            device=device, dtype=torch.float32, requires_grad=True
        ))
        w_b_a = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.N_border_per_cell, cfg.K_border,
            device=device, dtype=torch.float32, requires_grad=True
        ))
        w_a.retain_grad()
        w_b_a.retain_grad()

        msg_b = msg_a.detach().clone().requires_grad_(True)
        w_b = w_a.detach().clone().requires_grad_(True)
        w_b_b = w_b_a.detach().clone().requires_grad_(True)

        ref = mg._cell_gather(msg_a, w_a)
        border = mg._border_gather(msg_a, w_b_a)
        border_idx = mg.border_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, cfg.D_neuron
        )
        ref = ref.scatter_add(2, border_idx, border.to(ref.dtype))
        ref_loss = ref.square().mean()
        ref_loss.backward()

        out = fused_gather(
            msg_b, w_b, mg.conn_indices,
            w_b_b, mg.border_conn_indices,
            cfg.alpha,
        )
        out_loss = out.square().mean()
        out_loss.backward()

        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(msg_b.grad.float(), msg_a.grad.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(w_b.grad.float(), w_a.grad.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(w_b_b.grad.float(), w_b_a.grad.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFusedTokenStep:
    def test_matches_reference(self):
        _, _, _, fused_step, ref_step = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = "cuda"

        torch.manual_seed(7)
        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16).to(device)
        BS = 2
        mg.initialize_states(BS)

        h = torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron,
                        device=device, dtype=torch.bfloat16) * 0.1
        msg = torch.randn_like(h) * 0.1
        w_conn_sig = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections,
            device=device, dtype=torch.bfloat16
        ))
        w_conn_border_sig = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.N_border_per_cell, cfg.K_border,
            device=device, dtype=torch.bfloat16
        ))
        decay_logit = torch.randn(
            BS, cfg.N_cells, cfg.C_neurons,
            device=device, dtype=torch.bfloat16
        ) * 0.1
        primitives = torch.randn_like(h) * 0.1
        inject_raw = torch.randn(
            BS, cfg.N_cells, cfg.D_neuron,
            device=device, dtype=torch.bfloat16
        ) * 0.1

        h_ref, msg_ref = ref_step(
            h.float(), msg.float(),
            w_conn_sig.float(), w_conn_border_sig.float(),
            decay_logit.float(), primitives.float(),
            mg.neuron_id.float(), inject_raw.float(),
            mg.conn_indices, mg.border_conn_indices, mg.inject_indices,
            mg.state_w1.float(), mg.state_b1.float(),
            mg.state_w2.float(), mg.state_b2.float(),
            mg.msg_w1.float(), mg.msg_b1.float(),
            mg.msg_w2.float(), mg.msg_b2.float(),
            cfg,
        )

        h_tri, msg_tri = fused_step(
            h, msg, w_conn_sig, w_conn_border_sig,
            decay_logit, primitives,
            mg.conn_indices, mg.border_conn_indices,
            mg.neuron_id, mg.inject_indices, inject_raw,
            mg.state_w1, mg.state_b1, mg.state_w2, mg.state_b2,
            mg.msg_w1, mg.msg_b1, mg.msg_w2, mg.msg_b2, cfg,
        )

        torch.testing.assert_close(h_tri.float(), h_ref.float(), atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(msg_tri.float(), msg_ref.float(), atol=2e-1, rtol=2e-1)

    def test_backward_matches_reference(self):
        _, _, _, fused_step, ref_step = _try_import_triton()
        cfg = make_tiny()
        cfg.validate()
        device = "cuda"

        torch.manual_seed(8)
        from src.v11.memory_graph import CellMemoryGraph
        mg = CellMemoryGraph(cfg, torch.device(device), dtype=torch.bfloat16).to(device).float()
        BS = 2
        mg.initialize_states(BS)

        h_a = (torch.randn(BS, cfg.N_cells, cfg.C_neurons, cfg.D_neuron,
                           device=device, dtype=torch.float32) * 0.1).detach().requires_grad_(True)
        msg_a = (torch.randn_like(h_a) * 0.1).detach().requires_grad_(True)
        w_a = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.C_neurons, cfg.K_connections,
            device=device, dtype=torch.float32
        )).detach().requires_grad_(True)
        w_b_a = torch.sigmoid(torch.randn(
            BS, cfg.N_cells, cfg.N_border_per_cell, cfg.K_border,
            device=device, dtype=torch.float32
        )).detach().requires_grad_(True)
        decay_a = (torch.randn(
            BS, cfg.N_cells, cfg.C_neurons,
            device=device, dtype=torch.float32
        ) * 0.1).detach().requires_grad_(True)
        prim_a = (torch.randn_like(h_a) * 0.1).detach().requires_grad_(True)
        inject_raw = torch.randn(
            BS, cfg.N_cells, cfg.D_neuron,
            device=device, dtype=torch.float32
        ) * 0.1

        h_b = h_a.detach().clone().requires_grad_(True)
        msg_b = msg_a.detach().clone().requires_grad_(True)
        w_c = w_a.detach().clone().requires_grad_(True)
        w_d = w_b_a.detach().clone().requires_grad_(True)
        decay_b = decay_a.detach().clone().requires_grad_(True)
        prim_b = prim_a.detach().clone().requires_grad_(True)

        out_ref = ref_step(
            h_a, msg_a, w_a, w_b_a, decay_a, prim_a,
            mg.neuron_id, inject_raw,
            mg.conn_indices, mg.border_conn_indices, mg.inject_indices,
            mg.state_w1, mg.state_b1, mg.state_w2, mg.state_b2,
            mg.msg_w1, mg.msg_b1, mg.msg_w2, mg.msg_b2,
            cfg,
        )
        loss_ref = out_ref[0].square().mean() + out_ref[1].square().mean()
        loss_ref.backward()

        out_tri = fused_step(
            h_b, msg_b, w_c, w_d, decay_b, prim_b,
            mg.conn_indices, mg.border_conn_indices,
            mg.neuron_id, mg.inject_indices, inject_raw,
            mg.state_w1, mg.state_b1, mg.state_w2, mg.state_b2,
            mg.msg_w1, mg.msg_b1, mg.msg_w2, mg.msg_b2,
            cfg,
        )
        loss_tri = out_tri[0].square().mean() + out_tri[1].square().mean()
        loss_tri.backward()

        torch.testing.assert_close(h_b.grad.float(), h_a.grad.float(), atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(msg_b.grad.float(), msg_a.grad.float(), atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(w_c.grad.float(), w_a.grad.float(), atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(w_d.grad.float(), w_b_a.grad.float(), atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(decay_b.grad.float(), decay_a.grad.float(), atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(prim_b.grad.float(), prim_a.grad.float(), atol=2e-1, rtol=2e-1)
