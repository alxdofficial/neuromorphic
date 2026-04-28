"""Parity tests for `src.graph_walker.triton.lif.LIFDepositFunction`.

The new module replaces ``triton_sparse_update.SparseLIFUpdate`` with a
cudagraph-friendly variant that uses static-shape preprocessing
(sort + diff + cumsum + sentinel-padded ``unique_dests``) instead of
``torch.unique``. The Triton kernel itself is the same forward + backward
math; this test confirms the new wrapper preserves numerics.
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.triton.lif import (
    LIFDepositFunction,
    sparse_lif_update_puretorch,
)
from src.graph_walker.triton_sparse_update import SparseLIFUpdate


def _make_padded_inputs(B: int, N: int, D_s: int, M_real: int,
                         dest_range: int | None = None,
                         seed: int = 0,
                         dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(seed)
    BN = B * N
    if dest_range is None:
        dest_range = BN
    s = torch.randn(BN, D_s, dtype=dtype, device="cuda")
    msgs_real = torch.randn(M_real, D_s, dtype=dtype, device="cuda")
    dests_real = torch.randint(
        0, dest_range, (M_real,), dtype=torch.int64, device="cuda",
    )
    alpha = torch.sigmoid(torch.randn(N, device="cuda"))

    # Pad to M_max with sentinel = BN.
    M_max = max(M_real * 2, 4)  # some headroom
    msgs_pad = torch.zeros(M_max, D_s, dtype=dtype, device="cuda")
    msgs_pad[:M_real] = msgs_real
    dests_pad = torch.full((M_max,), BN, dtype=torch.int64, device="cuda")
    dests_pad[:M_real] = dests_real
    return s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_function_forward_matches_old_path():
    """New static-shape Triton kernel must match the old SparseLIFUpdate."""
    B, N, D_s = 2, 16, 64
    M_real = 8
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max = _make_padded_inputs(
        B, N, D_s, M_real, seed=42,
    )
    BN = B * N

    # Old path — no padding.
    out_old = SparseLIFUpdate.apply(
        s.clone(), msgs_real.clone(), dests_real, alpha.clone(), N,
    )

    # New path — padded inputs.
    out_new = LIFDepositFunction.apply(
        s.clone(), msgs_pad.clone(), dests_pad, alpha.clone(), N,
    )

    torch.cuda.synchronize()
    assert torch.allclose(
        out_old.float(), out_new.float(), atol=1e-2, rtol=1e-2,
    ), f"max|diff|={(out_old - out_new).abs().max():.4e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_function_forward_matches_puretorch_ref():
    """Triton output must match the puretorch reference within bf16 tol."""
    B, N, D_s = 2, 16, 128
    M_real = 16
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max = _make_padded_inputs(
        B, N, D_s, M_real, seed=43,
    )

    out_ref = sparse_lif_update_puretorch(
        s.clone(), msgs_real.clone(), dests_real, alpha.clone(), N,
    )

    out_new = LIFDepositFunction.apply(
        s.clone(), msgs_pad.clone(), dests_pad, alpha.clone(), N,
    )

    torch.cuda.synchronize()
    assert torch.allclose(
        out_ref.float(), out_new.float(), atol=1e-2, rtol=1e-2,
    ), f"max|diff|={(out_ref - out_new).abs().max():.4e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_function_backward_grad_msgs_matches_old():
    """grad_msgs from the new Triton path must match the old path."""
    B, N, D_s = 2, 16, 64
    M_real = 8
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max = _make_padded_inputs(
        B, N, D_s, M_real, seed=44,
    )
    grad_up = torch.randn_like(s)

    # Old path
    msgs_old = msgs_real.clone().requires_grad_(True)
    alpha_old = alpha.clone().requires_grad_(True)
    out_old = SparseLIFUpdate.apply(s.clone(), msgs_old, dests_real, alpha_old, N)
    (out_old.float() * grad_up.float()).sum().backward()

    # New path
    msgs_new = msgs_pad.clone().requires_grad_(True)
    alpha_new = alpha.clone().requires_grad_(True)
    out_new = LIFDepositFunction.apply(
        s.clone(), msgs_new, dests_pad, alpha_new, N,
    )
    (out_new.float() * grad_up.float()).sum().backward()

    torch.cuda.synchronize()
    # Compare gradients on the real (non-padding) entries.
    assert torch.allclose(
        msgs_old.grad.float(), msgs_new.grad[:M_real].float(),
        atol=1e-2, rtol=1e-2,
    ), f"grad_msgs mismatch: max|diff|={(msgs_old.grad - msgs_new.grad[:M_real]).abs().max():.4e}"
    # Padding entries should have zero grad (they correspond to sentinel dests
    # that the kernel skips).
    assert torch.allclose(
        msgs_new.grad[M_real:],
        torch.zeros_like(msgs_new.grad[M_real:]),
        atol=1e-3,
    ), "padding rows of msg grad should be zero"
    # alpha grad — looser tolerance for atomic-add nondeterminism.
    assert torch.allclose(
        alpha_old.grad, alpha_new.grad, atol=5e-2, rtol=5e-3,
    ), f"grad_alpha mismatch: max|diff|={(alpha_old.grad - alpha_new.grad).abs().max():.4e}"


# =====================================================================
# Production-scale + end-to-end parity tests
# =====================================================================


def _make_prod_scale_inputs(B: int = 4, N: int = 4096, D_s: int = 768, H: int = 4,
                              window_start: bool = False, seed: int = 0,
                              dtype: torch.dtype = torch.bfloat16):
    """Build inputs at production-scale shapes, mirroring what
    `_step_core_pure` would feed to ``sparse_lif_update``:

    - On interior steps: M_real = B·H walker writes
    - On window-start steps: M_real = 2·B·H (anchor injections + walker writes)
    - M_max = 2·B·H always (static shape for cudagraph compat)
    - dests built as ``batch_idx * N + cur_col`` exactly like graph_walker.py
    """
    torch.manual_seed(seed)
    BH = B * H
    BN = B * N
    M_real = 2 * BH if window_start else BH
    M_max = 2 * BH

    s = torch.randn(BN, D_s, dtype=dtype, device="cuda")

    # Build per-walker col positions (mirrors walker_pos[b, h] in graph_walker)
    walker_cols = torch.randint(0, N, (BH,), dtype=torch.int64, device="cuda")
    batch_idx = torch.arange(B, device="cuda").repeat_interleave(H)
    walker_dests = batch_idx * N + walker_cols                          # [BH]

    msgs_real = torch.randn(M_real, D_s, dtype=dtype, device="cuda")

    if window_start:
        # First BH = anchor cols (inject), second BH = walker cols (m_out write).
        # In graph_walker these collide: anchor cols == cur_bh on window-start
        # steps because walker just teleported to anchor. Mirror that here.
        dests_real = torch.cat([walker_dests, walker_dests], dim=0)     # [2BH]
    else:
        dests_real = walker_dests                                        # [BH]

    alpha = torch.sigmoid(torch.randn(N, device="cuda"))

    # Pad to M_max for the Triton path.
    msgs_pad = torch.zeros(M_max, D_s, dtype=dtype, device="cuda")
    msgs_pad[:M_real] = msgs_real
    dests_pad = torch.full((M_max,), BN, dtype=torch.int64, device="cuda")
    dests_pad[:M_real] = dests_real

    return s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max, M_real


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
@pytest.mark.parametrize("window_start", [False, True])
def test_lif_deposit_production_scale_grad_s_in_matches_puretorch(window_start: bool):
    """Production-scale parity test: B=4 H=4 N=4096 D_s=768. Verifies
    ``grad_s_in`` (substrate input gradient) — NOT covered by existing tests,
    but it's the gradient that flows back to the previous step's ``s_new`` in
    multi-step block_forward usage. If this is wrong, gradient through the
    block is silently broken."""
    s, msgs_real, dests_real, msgs_pad, dests_pad, alpha, M_max, M_real = (
        _make_prod_scale_inputs(B=4, N=4096, D_s=768, H=4,
                                  window_start=window_start, seed=11)
    )
    BN, D_s = s.shape
    N = 4096
    grad_up = torch.randn_like(s)

    # ---- Puretorch reference (autograd-tracked) ----
    s_pt = s.clone().requires_grad_(True)
    msgs_pt = msgs_real.clone().requires_grad_(True)
    alpha_pt = alpha.clone().requires_grad_(True)
    out_pt = sparse_lif_update_puretorch(s_pt, msgs_pt, dests_real, alpha_pt, N)
    (out_pt.float() * grad_up.float()).sum().backward()

    # ---- Triton path ----
    s_tr = s.clone().requires_grad_(True)
    msgs_tr = msgs_pad.clone().requires_grad_(True)
    alpha_tr = alpha.clone().requires_grad_(True)
    out_tr = LIFDepositFunction.apply(
        s_tr, msgs_tr, dests_pad, alpha_tr, N,
    )
    (out_tr.float() * grad_up.float()).sum().backward()

    torch.cuda.synchronize()

    # Forward output parity
    assert torch.allclose(out_pt.float(), out_tr.float(), atol=1e-2, rtol=1e-2), (
        f"forward mismatch: max|diff|={(out_pt - out_tr).abs().max():.4e}"
    )
    # grad_s_in parity (the new check)
    assert s_pt.grad is not None and s_tr.grad is not None
    assert torch.allclose(
        s_pt.grad.float(), s_tr.grad.float(), atol=1e-2, rtol=1e-2,
    ), (
        f"grad_s_in mismatch (window_start={window_start}): "
        f"max|diff|={(s_pt.grad - s_tr.grad).abs().max():.4e}"
    )
    # grad_msgs parity (real entries only)
    assert torch.allclose(
        msgs_pt.grad.float(), msgs_tr.grad[:M_real].float(),
        atol=1e-2, rtol=1e-2,
    ), (
        f"grad_msgs mismatch (window_start={window_start}): "
        f"max|diff|={(msgs_pt.grad - msgs_tr.grad[:M_real]).abs().max():.4e}"
    )
    # grad_alpha parity (looser — atomic-add nondeterminism in Triton bwd)
    assert torch.allclose(
        alpha_pt.grad, alpha_tr.grad, atol=5e-2, rtol=5e-2,
    ), (
        f"grad_alpha mismatch (window_start={window_start}): "
        f"max|diff|={(alpha_pt.grad - alpha_tr.grad).abs().max():.4e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_lif_deposit_multistep_chain_matches_puretorch():
    """Chain T=64 sequential calls mirroring how block_forward uses the
    kernel: each step's ``s_new`` becomes the next step's ``s_in``. Verifies
    forward outputs match across the chain AND that the gradient flowing
    backward through the chain matches between Triton and puretorch.

    This is the test that catches multi-step accumulated drift, autograd
    chain breaks, and version-counter conflicts on saved-for-backward
    references — none of which the single-step tests would surface.
    """
    B, N, D_s, H = 2, 256, 128, 4         # smaller than full prod for test speed
    T_block = 64
    BH = B * H
    BN = B * N
    M_max = 2 * BH

    torch.manual_seed(99)
    s_init = torch.randn(BN, D_s, dtype=torch.bfloat16, device="cuda")
    alpha = torch.sigmoid(torch.randn(N, device="cuda"))

    # Build a deterministic per-step msgs/dests sequence so both backends see
    # identical inputs at every step.
    msgs_per_step = []
    dests_per_step_real = []
    dests_per_step_pad = []
    msgs_per_step_pad = []
    for t in range(T_block):
        torch.manual_seed(1000 + t)
        BH = B * H
        walker_cols = torch.randint(0, N, (BH,), dtype=torch.int64, device="cuda")
        batch_idx = torch.arange(B, device="cuda").repeat_interleave(H)
        walker_dests = batch_idx * N + walker_cols
        msgs_real = torch.randn(BH, D_s, dtype=torch.bfloat16, device="cuda")

        msgs_pad = torch.zeros(M_max, D_s, dtype=torch.bfloat16, device="cuda")
        msgs_pad[:BH] = msgs_real
        dests_pad = torch.full((M_max,), BN, dtype=torch.int64, device="cuda")
        dests_pad[:BH] = walker_dests

        msgs_per_step.append(msgs_real)
        dests_per_step_real.append(walker_dests)
        msgs_per_step_pad.append(msgs_pad)
        dests_per_step_pad.append(dests_pad)

    grad_up_final = torch.randn_like(s_init)

    # ---- Puretorch: chain ----
    s_pt = s_init.clone().requires_grad_(True)
    alpha_pt = alpha.clone().requires_grad_(True)
    msgs_pt = [m.clone().requires_grad_(True) for m in msgs_per_step]

    s = s_pt
    for t in range(T_block):
        s = sparse_lif_update_puretorch(
            s, msgs_pt[t], dests_per_step_real[t], alpha_pt, N,
        )
    (s.float() * grad_up_final.float()).sum().backward()
    s_final_pt = s.detach().clone()

    # ---- Triton: chain ----
    s_tr = s_init.clone().requires_grad_(True)
    alpha_tr = alpha.clone().requires_grad_(True)
    msgs_tr = [m.clone().requires_grad_(True) for m in msgs_per_step_pad]

    s = s_tr
    for t in range(T_block):
        s = LIFDepositFunction.apply(
            s, msgs_tr[t], dests_per_step_pad[t], alpha_tr, N,
        )
    (s.float() * grad_up_final.float()).sum().backward()
    s_final_tr = s.detach().clone()

    torch.cuda.synchronize()

    # Forward output parity at end of chain — drift is the killer here
    assert torch.allclose(
        s_final_pt.float(), s_final_tr.float(), atol=5e-2, rtol=5e-2,
    ), (
        f"chain final forward mismatch: "
        f"max|diff|={(s_final_pt - s_final_tr).abs().max():.4e}"
    )
    # Gradient through the chain back to s_init
    assert torch.allclose(
        s_pt.grad.float(), s_tr.grad.float(), atol=5e-2, rtol=5e-2,
    ), (
        f"chain grad_s_in mismatch: "
        f"max|diff|={(s_pt.grad - s_tr.grad).abs().max():.4e}"
    )
    # Per-step msgs gradients (sample first/middle/last)
    for t in [0, T_block // 2, T_block - 1]:
        BH = B * H
        assert torch.allclose(
            msgs_pt[t].grad.float(), msgs_tr[t].grad[:BH].float(),
            atol=5e-2, rtol=5e-2,
        ), (
            f"chain msgs grad mismatch at t={t}: "
            f"max|diff|={(msgs_pt[t].grad - msgs_tr[t].grad[:BH]).abs().max():.4e}"
        )
    # alpha grad (single accumulator across the chain)
    assert torch.allclose(
        alpha_pt.grad, alpha_tr.grad, atol=1e-1, rtol=1e-1,
    ), (
        f"chain alpha grad mismatch: "
        f"max|diff|={(alpha_pt.grad - alpha_tr.grad).abs().max():.4e}"
    )
