"""Tests for Triton kernel equivalence with Python reference implementation.

Compares _forward_segment_triton vs _forward_segment_python on CUDA
to verify the Triton kernel produces numerically equivalent results.
"""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.v8.config import V8Config
from src.v8.memory_graph import MemoryGraph

BS = 2

# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel tests"
)


def make_tiny(**overrides):
    cfg = V8Config.tier_tiny(**overrides)
    cfg.validate()
    return cfg


def make_paired_graphs(cfg, dtype=torch.float32):
    """Create two MemoryGraphs with identical state — one for Python, one for Triton."""
    device = torch.device("cuda")

    mg_py = MemoryGraph(cfg, device, dtype=dtype)
    mg_py.initialize(BS)
    # Force Python path
    mg_py._triton_ready = False

    mg_tr = MemoryGraph(cfg, device, dtype=dtype)
    mg_tr.initialize(BS)

    # Copy identical state
    mg_tr.h = mg_py.h.clone()
    mg_tr.prev_messages = mg_py.prev_messages.clone()
    mg_tr.primitives = mg_py.primitives.clone()
    mg_tr.decay_logit = mg_py.decay_logit.clone()
    mg_tr.conn_weights = mg_py.conn_weights.clone()
    mg_tr.conn_indices = mg_py.conn_indices.clone()
    mg_tr.conn_mask = mg_py.conn_mask.clone()
    mg_tr.mean_input = mg_py.mean_input.clone()
    mg_tr.mean_output = mg_py.mean_output.clone()
    mg_tr.activation_ema = mg_py.activation_ema.clone()
    mg_tr.activation_std_ema = mg_py.activation_std_ema.clone()
    mg_tr.firing_rate = mg_py.firing_rate.clone()
    mg_tr._adjacency_dirty = True

    # Re-init Triton buffers with the copied indices
    if mg_tr._triton_ready:
        mg_tr._conn_idx_i32 = mg_tr.conn_indices.to(torch.int32).contiguous()

    return mg_py, mg_tr


class TestTritonEquivalence:
    """Verify Triton kernel produces same output as Python reference."""

    def test_single_segment_f32(self):
        """Float32 equivalence.

        Tolerance is 1e-2 because sparse gather (Triton) and dense bmm (Python)
        accumulate in different orders, producing different floating point rounding.
        Both are correct — just different reduction trees.
        """
        cfg = make_tiny()
        mg_py, mg_tr = make_paired_graphs(cfg, dtype=torch.float32)

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem, device="cuda")

        out_py = mg_py._forward_segment_python(cc.clone())
        out_tr = mg_tr._forward_segment_triton(cc.clone())

        torch.testing.assert_close(out_tr, out_py, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(mg_tr.h, mg_py.h, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(mg_tr.prev_messages, mg_py.prev_messages,
                                   atol=1e-2, rtol=1e-2)

    def test_single_segment_bf16(self):
        """BFloat16 equivalence — looser tolerance."""
        cfg = make_tiny()
        mg_py, mg_tr = make_paired_graphs(cfg, dtype=torch.bfloat16)

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem,
                         device="cuda", dtype=torch.bfloat16)

        out_py = mg_py._forward_segment_python(cc.clone())
        out_tr = mg_tr._forward_segment_triton(cc.clone())

        torch.testing.assert_close(out_tr, out_py, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(mg_tr.h, mg_py.h, atol=5e-2, rtol=5e-2)

    def test_multi_segment_state_carry(self):
        """State carries correctly across multiple segments."""
        cfg = make_tiny()
        mg_py, mg_tr = make_paired_graphs(cfg, dtype=torch.float32)

        for _ in range(3):
            cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem, device="cuda") * 0.3
            out_py = mg_py._forward_segment_python(cc.clone())
            out_tr = mg_tr._forward_segment_triton(cc.clone())

            torch.testing.assert_close(out_tr, out_py, atol=2e-2, rtol=2e-2)

        torch.testing.assert_close(mg_tr.h, mg_py.h, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(mg_tr.prev_messages, mg_py.prev_messages,
                                   atol=2e-2, rtol=2e-2)

    def test_eot_handling(self):
        """EOT mask produces same state reset in both paths."""
        cfg = make_tiny()
        mg_py, mg_tr = make_paired_graphs(cfg, dtype=torch.float32)

        # Build up state
        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem, device="cuda") * 0.5
        mg_py._forward_segment_python(cc.clone())
        mg_tr._forward_segment_triton(cc.clone())

        # Now with EOT at position 2 for batch 0
        cc2 = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem, device="cuda") * 0.5
        eot_mask = torch.zeros(BS, cfg.action_every, dtype=torch.bool, device="cuda")
        eot_mask[0, 2] = True

        out_py = mg_py._forward_segment_python(cc2.clone(), eot_mask)
        out_tr = mg_tr._forward_segment_triton(cc2.clone(), eot_mask)

        torch.testing.assert_close(out_tr, out_py, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(mg_tr.h, mg_py.h, atol=1e-2, rtol=1e-2)

    def test_zero_input_equivalence(self):
        """With zero CC signal, only graph dynamics matter."""
        cfg = make_tiny()
        mg_py, mg_tr = make_paired_graphs(cfg, dtype=torch.float32)

        # Set nonzero initial state
        mg_py.h = torch.randn_like(mg_py.h) * 0.5
        mg_py.prev_messages = torch.randn_like(mg_py.prev_messages) * 0.3
        mg_tr.h = mg_py.h.clone()
        mg_tr.prev_messages = mg_py.prev_messages.clone()

        cc = torch.zeros(BS, cfg.action_every, cfg.C, cfg.D_mem, device="cuda")

        out_py = mg_py._forward_segment_python(cc.clone())
        out_tr = mg_tr._forward_segment_triton(cc.clone())

        torch.testing.assert_close(out_tr, out_py, atol=1e-2, rtol=1e-2)

    def test_dispatch_uses_triton_on_cuda(self):
        """forward_segment dispatches to Triton when on CUDA."""
        cfg = make_tiny()
        mg = MemoryGraph(cfg, torch.device("cuda"), dtype=torch.float32)
        mg.initialize(BS)

        assert mg._triton_ready, "Triton should be ready on CUDA"

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem, device="cuda")
        out = mg.forward_segment(cc)
        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)
        assert torch.isfinite(out).all()


class TestTritonTierA:
    """Test with Tier A dimensions (N=1024, K=96, D=128) to catch scaling issues."""

    def test_tier_a_dimensions(self):
        """Run one segment at full Tier A scale."""
        cfg = V8Config.tier_a(vocab_size=32000, T=256, action_every=256)
        cfg.validate()
        device = torch.device("cuda")

        mg = MemoryGraph(cfg, device, dtype=torch.bfloat16)
        mg.initialize(BS)

        cc = torch.randn(BS, cfg.action_every, cfg.C, cfg.D_mem,
                         device=device, dtype=torch.bfloat16) * 0.1
        out = mg.forward_segment(cc)

        assert out.shape == (BS, cfg.action_every, cfg.C, cfg.D_mem)
        assert torch.isfinite(out).all()
        assert mg.h.abs().sum() > 0
