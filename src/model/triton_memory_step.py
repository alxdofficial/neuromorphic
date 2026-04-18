"""Triton-fused per-token memory step.

Fuses the per-token hot path into a single kernel:
    received = W @ msg + inject_dense        (bmm + add)
    h_new    = tanh(decay*h + (1-decay)*received)
    readout  = sum_n(h_new[n] * out_port_mask[n]) * READOUT_SCALE

Grid: (BS, NC) — one program per (batch, cell). Each program holds a cell's
full [N, D_n] state in SRAM. Dispatch overhead drops from ~5 kernels/token
to 1 kernel/token.

Forward kernel only. Backward uses the PyTorch reference (via
`fused_memory_step_torch`) and is invoked through a torch.autograd.Function
that calls Triton only when grad is disabled.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


# ======================================================================
# Triton kernel — forward only
# ======================================================================


@triton.jit
def _memory_step_fwd(
    h_in_ptr, msg_ptr, W_ptr, decay_ptr,
    inject_proj_ptr, out_mask_ptr,
    h_out_ptr, readout_ptr,
    # shape constants
    N: tl.constexpr,
    D_n: tl.constexpr,
    ALPHA: tl.constexpr,
    READOUT_SCALE: tl.constexpr,
    # strides
    h_stride_b, h_stride_c,
    W_stride_b, W_stride_c,
    decay_stride_b, decay_stride_c,
    inject_stride_b, inject_stride_c,
    readout_stride_b, readout_stride_c,
):
    """Fused per-token memory step.

    Inject is passed as [BS, NC, ALPHA, D_n] directly — the kernel adds it
    to the first ALPHA rows of `received` in-kernel. This drops the Python
    scatter_add / dense-zero-materialize prologue from the hot loop.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = tl.arange(0, N)
    d_offs = tl.arange(0, D_n)

    # h, msg, h_out: [BS, NC, N, D_n]
    state_base = pid_b * h_stride_b + pid_c * h_stride_c
    state_offs = state_base + n_offs[:, None] * D_n + d_offs[None, :]

    h = tl.load(h_in_ptr + state_offs).to(tl.float32)
    msg = tl.load(msg_ptr + state_offs).to(tl.float32)

    # W [N, N]
    W_base = pid_b * W_stride_b + pid_c * W_stride_c
    W_offs = W_base + n_offs[:, None] * N + n_offs[None, :]
    W = tl.load(W_ptr + W_offs).to(tl.float32)

    # decay [N]
    decay_base = pid_b * decay_stride_b + pid_c * decay_stride_c
    decay = tl.load(decay_ptr + decay_base + n_offs).to(tl.float32)

    # output port mask [N]
    out_mask = tl.load(out_mask_ptr + pid_c * N + n_offs).to(tl.float32)

    # Load inject_proj [ALPHA, D_n] and pad to [N, D_n] (zeros beyond alpha).
    # Clamp the row index so all loads are in-bounds, then zero rows ≥ ALPHA
    # via tl.where. Relies on input ports being the first ALPHA local indices
    # (which is the port_layout convention).
    clamped_n = tl.minimum(n_offs, ALPHA - 1)
    inject_base = pid_b * inject_stride_b + pid_c * inject_stride_c
    inject_offs = inject_base + clamped_n[:, None] * D_n + d_offs[None, :]
    raw_inject = tl.load(inject_proj_ptr + inject_offs).to(tl.float32)
    inject = tl.where(n_offs[:, None] < ALPHA, raw_inject, 0.0)

    # received = W @ msg + inject
    received = tl.dot(W, msg, allow_tf32=True) + inject

    # LIF update
    decay_exp = decay[:, None]
    pre_tanh = decay_exp * h + (1.0 - decay_exp) * received
    exp2x = tl.exp(2.0 * pre_tanh)
    h_new = (exp2x - 1.0) / (exp2x + 1.0)

    tl.store(h_out_ptr + state_offs, h_new.to(tl.bfloat16))

    # Readout: weighted sum over N
    readout = tl.sum(h_new * out_mask[:, None], axis=0) * READOUT_SCALE
    readout_base = pid_b * readout_stride_b + pid_c * readout_stride_c
    tl.store(readout_ptr + readout_base + d_offs, readout.to(tl.bfloat16))


# ======================================================================
# Python helpers
# ======================================================================


def fused_memory_step_triton(
    h: Tensor,                # [BS, NC, N, D_n], bf16
    msg: Tensor,              # [BS, NC, N, D_n], bf16
    W: Tensor,                # [BS, NC, N, N], bf16
    decay: Tensor,            # [BS, NC, N], bf16
    inject_proj: Tensor,      # [BS, NC, ALPHA, D_n], bf16  (NOT pre-scattered)
    out_mask: Tensor,         # [NC, N], bf16 or fp32
    readout_scale: float,
) -> tuple[Tensor, Tensor]:
    """Triton forward pass. Returns (h_out, readout [BS, NC, D_n]).

    Inject is passed as [BS, NC, ALPHA, D_n] — kernel adds to first ALPHA
    rows in-place. Assumes port_layout puts input ports at local indices
    [0, ALPHA) per cell.
    """
    assert h.is_cuda, "Triton kernel requires CUDA"
    assert h.dtype == torch.bfloat16, "Triton kernel expects bf16 state"
    BS, NC, N, D_n = h.shape
    assert msg.shape == h.shape
    assert W.shape == (BS, NC, N, N)
    assert decay.shape == (BS, NC, N)
    assert inject_proj.dim() == 4 and inject_proj.shape[:2] == (BS, NC)
    ALPHA = inject_proj.shape[2]
    assert inject_proj.shape[3] == D_n
    assert out_mask.shape == (NC, N)

    h = h.contiguous(); msg = msg.contiguous(); W = W.contiguous()
    decay = decay.contiguous(); inject_proj = inject_proj.contiguous()
    out_mask = out_mask.contiguous()

    h_out = torch.empty_like(h)
    readout = torch.empty((BS, NC, D_n), dtype=h.dtype, device=h.device)

    grid = (BS, NC)
    _memory_step_fwd[grid](
        h, msg, W, decay, inject_proj, out_mask,
        h_out, readout,
        N=N, D_n=D_n, ALPHA=ALPHA, READOUT_SCALE=readout_scale,
        h_stride_b=h.stride(0), h_stride_c=h.stride(1),
        W_stride_b=W.stride(0), W_stride_c=W.stride(1),
        decay_stride_b=decay.stride(0), decay_stride_c=decay.stride(1),
        inject_stride_b=inject_proj.stride(0), inject_stride_c=inject_proj.stride(1),
        readout_stride_b=readout.stride(0), readout_stride_c=readout.stride(1),
    )
    return h_out, readout


def fused_memory_step_torch(
    h: Tensor,
    msg: Tensor,
    W: Tensor,
    decay: Tensor,
    inject_proj: Tensor,      # [BS, NC, ALPHA, D_n]
    out_mask: Tensor,
    readout_scale: float,
) -> tuple[Tensor, Tensor]:
    """Pure PyTorch reference — same math as the Triton kernel.

    Used for (a) training grad-enabled path (autograd works naturally),
    (b) CPU fallback, (c) correctness testing the Triton kernel.

    inject_proj [BS, NC, ALPHA, D_n] is added to the first ALPHA rows of
    `received` (the input-port neurons). Rest of N rows get no inject.
    """
    BS, NC, N, D_n = h.shape
    ALPHA = inject_proj.shape[2]
    received = torch.matmul(W, msg)
    # Add inject to first ALPHA rows. Equivalent to scatter_add at fixed
    # first indices but without materializing a dense zero buffer.
    received = received.clone()
    received[:, :, :ALPHA, :] = received[:, :, :ALPHA, :] + inject_proj
    decay_exp = decay.unsqueeze(-1)
    h_out = torch.tanh(decay_exp * h + (1.0 - decay_exp) * received)
    # Cast out_mask to h_out.dtype so `h_out * mask` stays in the state
    # dtype instead of upcasting to the mask's default fp32. Buffers
    # (registered via `register_buffer`) don't follow `.to(dtype)` when
    # model weights do, so out_port_mask can easily sit at fp32 while the
    # rest of memory runs in bf16 — that mismatch silently upcasts the
    # returned readout and trips dtype-strict F.linear calls downstream.
    mask = out_mask.to(h_out.dtype)
    readout = (h_out * mask.unsqueeze(0).unsqueeze(-1)).sum(dim=2) * readout_scale
    return h_out, readout


# ======================================================================
# Autograd wrapper: Triton forward + analytical PyTorch backward
# ======================================================================


class _FusedMemoryStep(torch.autograd.Function):
    """Triton forward + analytical PyTorch backward.

    Forward math:
        received  = W @ msg + inject_dense
        pre_tanh  = decay * h + (1 - decay) * received
        h_out     = tanh(pre_tanh)
        readout   = sum_n(h_out[..., n, :] * out_mask[..., n, :]) * scale

    Backward (chain rule):
        effective dL/dh_out = upstream_dh_out + dL_dreadout · mask · scale
        dL/dpre_tanh = effective_dh_out · (1 - h_out²)
        dL/dh        = dL/dpre_tanh · decay
        dL/dreceived = dL/dpre_tanh · (1 - decay)
        dL/ddecay    = (dL/dpre_tanh · (h - received)).sum(D_n)
        dL/dinject_dense = dL/dreceived
        dL/dW        = dL/dreceived @ msgᵀ
        dL/dmsg      = Wᵀ @ dL/dreceived
    """

    @staticmethod
    def forward(ctx, h, msg, W, decay, inject_proj, out_mask, readout_scale):
        with torch.no_grad():
            h_out, readout = fused_memory_step_triton(
                h, msg, W, decay, inject_proj, out_mask, readout_scale)
        ctx.save_for_backward(h, msg, W, decay, inject_proj, h_out, out_mask)
        ctx.readout_scale = readout_scale
        return h_out, readout

    @staticmethod
    def backward(ctx, dL_dh_out, dL_dreadout):
        h, msg, W, decay, inject_proj, h_out, out_mask = ctx.saved_tensors
        scale = ctx.readout_scale
        ALPHA = inject_proj.shape[2]

        # Upstream dh_out gets the readout contribution added.
        dh_out_from_readout = (
            dL_dreadout.unsqueeze(2)
            * out_mask.unsqueeze(0).unsqueeze(-1)
            * scale
        )
        dh_out_total = dL_dh_out + dh_out_from_readout

        # Through tanh
        dpre = dh_out_total * (1.0 - h_out * h_out)

        decay_exp = decay.unsqueeze(-1)
        one_minus = 1.0 - decay_exp

        dL_dh = dpre * decay_exp
        dL_dreceived = dpre * one_minus

        # Reconstruct received for ddecay (needed because inject is only at first ALPHA rows).
        received = torch.matmul(W, msg).clone()
        received[:, :, :ALPHA, :] = received[:, :, :ALPHA, :] + inject_proj
        dL_ddecay = (dpre * (h - received)).sum(dim=-1)

        # inject only flows from the first ALPHA rows of dL_dreceived.
        dL_dinject_proj = dL_dreceived[:, :, :ALPHA, :]
        dL_dW = torch.matmul(dL_dreceived, msg.transpose(-1, -2))
        dL_dmsg = torch.matmul(W.transpose(-1, -2), dL_dreceived)

        return (dL_dh, dL_dmsg, dL_dW, dL_ddecay, dL_dinject_proj, None, None)


def fused_memory_step(
    h: Tensor,
    msg: Tensor,
    W: Tensor,
    decay: Tensor,
    inject_proj: Tensor,       # [BS, NC, ALPHA, D_n] — NOT pre-scattered
    out_mask: Tensor,
    readout_scale: float,
    *,
    use_triton: bool = True,
) -> tuple[Tensor, Tensor]:
    """Dispatch fused step: Triton (with autograd) when CUDA + bf16; else PyTorch."""
    # Triton requirements: CUDA, bf16 state, N and D_n are powers of 2
    # (tl.arange requires it). N also needs to be ≥ 16 for tl.dot. Any
    # failure falls through to the PyTorch reference, which has the same
    # math.
    N = h.shape[2]
    D_n = h.shape[3]
    _triton_shape_ok = (
        _is_pow2(N) and N >= 16
        and _is_pow2(D_n) and D_n >= 16
    )
    if (use_triton
            and h.is_cuda
            and h.dtype == torch.bfloat16
            and _triton_shape_ok):
        return _FusedMemoryStep.apply(
            h, msg, W, decay, inject_proj, out_mask, readout_scale)
    return fused_memory_step_torch(
        h, msg, W, decay, inject_proj, out_mask, readout_scale)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
