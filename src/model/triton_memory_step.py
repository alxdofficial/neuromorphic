"""Triton-fused per-token memory step.

Fuses the per-token hot path into a single kernel:
    received = W @ msg + inject_dense        (bmm + add)
    h_new    = tanh(decay*h + (1-decay)*received)
    readout  = sum_n(h_new[n] * out_port_mask[n]) * READOUT_SCALE

Grid: (BS, NC) — one program per (batch, cell). Each program holds a cell's
full [Nc, D_n] state in SRAM. Dispatch overhead drops from ~5 kernels/token
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
    Nc: tl.constexpr,
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

    n_offs = tl.arange(0, Nc)
    d_offs = tl.arange(0, D_n)

    # h, msg, h_out: [BS, NC, Nc, D_n]
    state_base = pid_b * h_stride_b + pid_c * h_stride_c
    state_offs = state_base + n_offs[:, None] * D_n + d_offs[None, :]

    h = tl.load(h_in_ptr + state_offs).to(tl.float32)
    msg = tl.load(msg_ptr + state_offs).to(tl.float32)

    # W [Nc, Nc]
    W_base = pid_b * W_stride_b + pid_c * W_stride_c
    W_offs = W_base + n_offs[:, None] * Nc + n_offs[None, :]
    W = tl.load(W_ptr + W_offs).to(tl.float32)

    # decay [Nc]
    decay_base = pid_b * decay_stride_b + pid_c * decay_stride_c
    decay = tl.load(decay_ptr + decay_base + n_offs).to(tl.float32)

    # output port mask [Nc]
    out_mask = tl.load(out_mask_ptr + pid_c * Nc + n_offs).to(tl.float32)

    # Load inject_proj [ALPHA, D_n] and pad to [Nc, D_n] (zeros beyond alpha).
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

    # Readout: weighted sum over Nc
    readout = tl.sum(h_new * out_mask[:, None], axis=0) * READOUT_SCALE
    readout_base = pid_b * readout_stride_b + pid_c * readout_stride_c
    tl.store(readout_ptr + readout_base + d_offs, readout.to(tl.bfloat16))


@triton.jit
def _memory_step_bwd(
    # saved from forward
    h_ptr, msg_ptr, W_ptr, decay_ptr, inject_proj_ptr, h_out_ptr, out_mask_ptr,
    # upstream grads
    dL_dh_out_ptr, dL_dreadout_ptr,
    # output grads
    dL_dh_ptr, dL_dmsg_ptr, dL_dW_ptr, dL_ddecay_ptr, dL_dinject_ptr,
    # shape
    Nc: tl.constexpr, D_n: tl.constexpr, ALPHA: tl.constexpr,
    READOUT_SCALE: tl.constexpr,
    # strides (state [BS, NC, Nc, D_n])
    h_stride_b, h_stride_c,
    # strides W [BS, NC, Nc, Nc]
    W_stride_b, W_stride_c,
    # strides decay [BS, NC, Nc]
    decay_stride_b, decay_stride_c,
    # strides inject_proj [BS, NC, ALPHA, D_n]
    inject_stride_b, inject_stride_c,
    # strides readout [BS, NC, D_n]
    readout_stride_b, readout_stride_c,
):
    """Fused backward for the per-token memory step.

    Computes (dL/dh, dL/dmsg, dL/dW, dL/ddecay, dL/dinject_proj) in one
    kernel per (batch, cell), replacing ~5 PyTorch ops + 2 matmuls.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    n_offs = tl.arange(0, Nc)
    d_offs = tl.arange(0, D_n)

    # ---- Load saved tensors ----
    state_base = pid_b * h_stride_b + pid_c * h_stride_c
    state_offs = state_base + n_offs[:, None] * D_n + d_offs[None, :]
    h = tl.load(h_ptr + state_offs).to(tl.float32)
    msg = tl.load(msg_ptr + state_offs).to(tl.float32)
    h_out = tl.load(h_out_ptr + state_offs).to(tl.float32)
    dL_dh_out = tl.load(dL_dh_out_ptr + state_offs).to(tl.float32)

    W_base = pid_b * W_stride_b + pid_c * W_stride_c
    W_offs = W_base + n_offs[:, None] * Nc + n_offs[None, :]
    W = tl.load(W_ptr + W_offs).to(tl.float32)

    decay_base = pid_b * decay_stride_b + pid_c * decay_stride_c
    decay = tl.load(decay_ptr + decay_base + n_offs).to(tl.float32)

    out_mask = tl.load(out_mask_ptr + pid_c * Nc + n_offs).to(tl.float32)

    readout_base = pid_b * readout_stride_b + pid_c * readout_stride_c
    dL_dreadout = tl.load(dL_dreadout_ptr + readout_base + d_offs).to(tl.float32)  # [D_n]

    # ---- Readout contribution to dh_out ----
    # readout = sum_n (h_out[n, d] * out_mask[n]) * scale
    #  → dh_out_total[n, d] += dL_dreadout[d] * out_mask[n] * scale
    dh_out_from_readout = dL_dreadout[None, :] * out_mask[:, None] * READOUT_SCALE
    dh_out_total = dL_dh_out + dh_out_from_readout

    # ---- Through tanh (h_out = tanh(pre_tanh)) ----
    dpre = dh_out_total * (1.0 - h_out * h_out)

    # ---- Split through the LIF blend: pre = decay*h + (1-decay)*received ----
    decay_exp = decay[:, None]
    dL_dh = dpre * decay_exp
    tl.store(dL_dh_ptr + state_offs, dL_dh.to(tl.bfloat16))

    dL_dreceived = dpre * (1.0 - decay_exp)

    # ---- Recompute received (for ddecay): W @ msg + inject ----
    received = tl.dot(W, msg, allow_tf32=True)
    clamped_n = tl.minimum(n_offs, ALPHA - 1)
    inject_base = pid_b * inject_stride_b + pid_c * inject_stride_c
    inject_offs = inject_base + clamped_n[:, None] * D_n + d_offs[None, :]
    raw_inject = tl.load(inject_proj_ptr + inject_offs).to(tl.float32)
    inject = tl.where(n_offs[:, None] < ALPHA, raw_inject, 0.0)
    received = received + inject

    # ---- dL/ddecay = sum_d(dpre * (h - received)) ----
    dL_ddecay = tl.sum(dpre * (h - received), axis=1)  # [Nc]
    tl.store(dL_ddecay_ptr + decay_base + n_offs, dL_ddecay.to(tl.bfloat16))

    # ---- dL/dinject: first ALPHA rows of dL_dreceived; use clamped offsets +
    #       masked store so rows n >= ALPHA don't write out-of-range ----
    dL_dinject_offs = inject_base + clamped_n[:, None] * D_n + d_offs[None, :]
    tl.store(
        dL_dinject_ptr + dL_dinject_offs,
        dL_dreceived.to(tl.bfloat16),
        mask=n_offs[:, None] < ALPHA,
    )

    # ---- dL/dW = dL_dreceived @ msg.T ----
    msg_T = tl.trans(msg)                                 # [D_n, Nc]
    dL_dW = tl.dot(dL_dreceived, msg_T, allow_tf32=True)  # [Nc, Nc]
    tl.store(dL_dW_ptr + W_offs, dL_dW.to(tl.bfloat16))

    # ---- dL/dmsg = W.T @ dL_dreceived ----
    W_T = tl.trans(W)                                         # [Nc, Nc]
    dL_dmsg = tl.dot(W_T, dL_dreceived, allow_tf32=True)     # [Nc, D_n]
    tl.store(dL_dmsg_ptr + state_offs, dL_dmsg.to(tl.bfloat16))


# ======================================================================
# Python helpers
# ======================================================================


def fused_memory_step_triton(
    h: Tensor,                # [BS, NC, Nc, D_n], bf16
    msg: Tensor,              # [BS, NC, Nc, D_n], bf16
    W: Tensor,                # [BS, NC, Nc, Nc], bf16
    decay: Tensor,            # [BS, NC, Nc], bf16
    inject_proj: Tensor,      # [BS, NC, ALPHA, D_n], bf16  (NOT pre-scattered)
    out_mask: Tensor,         # [NC, Nc], bf16 or fp32
    readout_scale: float,
) -> tuple[Tensor, Tensor]:
    """Triton forward pass. Returns (h_out, readout [BS, NC, D_n]).

    Inject is passed as [BS, NC, ALPHA, D_n] — kernel adds to first ALPHA
    rows in-place. Assumes port_layout puts input ports at local indices
    [0, ALPHA) per cell.
    """
    assert h.is_cuda, "Triton kernel requires CUDA"
    assert h.dtype == torch.bfloat16, "Triton kernel expects bf16 state"
    BS, NC, Nc, D_n = h.shape
    assert msg.shape == h.shape
    assert W.shape == (BS, NC, Nc, Nc)
    assert decay.shape == (BS, NC, Nc)
    assert inject_proj.dim() == 4 and inject_proj.shape[:2] == (BS, NC)
    ALPHA = inject_proj.shape[2]
    assert inject_proj.shape[3] == D_n
    assert out_mask.shape == (NC, Nc)

    h = h.contiguous(); msg = msg.contiguous(); W = W.contiguous()
    decay = decay.contiguous(); inject_proj = inject_proj.contiguous()
    out_mask = out_mask.contiguous()

    h_out = torch.empty_like(h)
    readout = torch.empty((BS, NC, D_n), dtype=h.dtype, device=h.device)

    grid = (BS, NC)
    _memory_step_fwd[grid](
        h, msg, W, decay, inject_proj, out_mask,
        h_out, readout,
        Nc=Nc, D_n=D_n, ALPHA=ALPHA, READOUT_SCALE=readout_scale,
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
    `received` (the input-port neurons). Rest of Nc rows get no inject.
    """
    BS, NC, Nc, D_n = h.shape
    ALPHA = inject_proj.shape[2]
    received = torch.matmul(W, msg)
    # Add inject to first ALPHA rows. Equivalent to scatter_add at fixed
    # first indices but without materializing a dense zero buffer.
    received = received.clone()
    received[:, :, :ALPHA, :] = received[:, :, :ALPHA, :] + inject_proj
    decay_exp = decay.unsqueeze(-1)
    h_out = torch.tanh(decay_exp * h + (1.0 - decay_exp) * received)
    readout = (h_out * out_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=2) * readout_scale
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

        # NOTE: a Triton-fused backward was prototyped (see `_memory_step_bwd`
        # and `_triton_backward` below, kept for reference) but it's ~8.7×
        # SLOWER than this analytical PyTorch path at our shape regime
        # (Nc=32, D_n=256, BS=64, NC=8). Reason: the fused kernel uses grid
        # (BS, NC) = 512 programs each doing three 32×32 / 32×256 dots; those
        # tiles are too small for tensor cores to amortize launch+load cost.
        # PyTorch's analytical path, by contrast, does 2 well-batched bmms at
        # [BS*NC, 32, ...] which saturate tensor cores. Forward benefited from
        # Triton fusion because the per-token ops were small+many; backward
        # is dominated by two matmuls that PyTorch already fuses effectively.
        return _torch_backward(
            h, msg, W, decay, inject_proj, h_out, out_mask,
            dL_dh_out, dL_dreadout, scale,
        )


def _torch_backward(
    h, msg, W, decay, inject_proj, h_out, out_mask,
    dL_dh_out, dL_dreadout, scale,
):
    """PyTorch reference backward (used by fallback paths + tests)."""
    ALPHA = inject_proj.shape[2]

    dh_out_from_readout = (
        dL_dreadout.unsqueeze(2)
        * out_mask.unsqueeze(0).unsqueeze(-1)
        * scale
    )
    dh_out_total = dL_dh_out + dh_out_from_readout
    dpre = dh_out_total * (1.0 - h_out * h_out)

    decay_exp = decay.unsqueeze(-1)
    dL_dh = dpre * decay_exp
    dL_dreceived = dpre * (1.0 - decay_exp)

    received = torch.matmul(W, msg).clone()
    received[:, :, :ALPHA, :] = received[:, :, :ALPHA, :] + inject_proj
    dL_ddecay = (dpre * (h - received)).sum(dim=-1)

    dL_dinject_proj = dL_dreceived[:, :, :ALPHA, :]
    dL_dW = torch.matmul(dL_dreceived, msg.transpose(-1, -2))
    dL_dmsg = torch.matmul(W.transpose(-1, -2), dL_dreceived)
    return (dL_dh, dL_dmsg, dL_dW, dL_ddecay, dL_dinject_proj, None, None)


def _triton_backward(
    h, msg, W, decay, inject_proj, h_out, out_mask,
    dL_dh_out, dL_dreadout, scale,
):
    """Triton-fused backward. Grid (BS, NC), one program per (batch, cell)."""
    BS, NC, Nc, D_n = h.shape
    ALPHA = inject_proj.shape[2]

    # Ensure contiguity for stride assumptions.
    h = h.contiguous(); msg = msg.contiguous(); W = W.contiguous()
    decay = decay.contiguous(); inject_proj = inject_proj.contiguous()
    h_out = h_out.contiguous(); out_mask = out_mask.contiguous()
    dL_dh_out = dL_dh_out.contiguous(); dL_dreadout = dL_dreadout.contiguous()

    dL_dh = torch.empty_like(h)
    dL_dmsg = torch.empty_like(msg)
    dL_dW = torch.empty_like(W)
    dL_ddecay = torch.empty_like(decay)
    dL_dinject = torch.empty_like(inject_proj)

    grid = (BS, NC)
    _memory_step_bwd[grid](
        h, msg, W, decay, inject_proj, h_out, out_mask,
        dL_dh_out, dL_dreadout,
        dL_dh, dL_dmsg, dL_dW, dL_ddecay, dL_dinject,
        Nc=Nc, D_n=D_n, ALPHA=ALPHA, READOUT_SCALE=scale,
        h_stride_b=h.stride(0), h_stride_c=h.stride(1),
        W_stride_b=W.stride(0), W_stride_c=W.stride(1),
        decay_stride_b=decay.stride(0), decay_stride_c=decay.stride(1),
        inject_stride_b=inject_proj.stride(0), inject_stride_c=inject_proj.stride(1),
        readout_stride_b=dL_dreadout.stride(0), readout_stride_c=dL_dreadout.stride(1),
    )
    return (dL_dh, dL_dmsg, dL_dW, dL_ddecay, dL_dinject, None, None)


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
    # Triton requirements: CUDA, bf16 state, Nc and D_n are powers of 2
    # (tl.arange requires it). Nc also needs to be ≥ 16 for tl.dot. Any
    # failure falls through to the PyTorch reference, which has the same
    # math.
    Nc = h.shape[2]
    D_n = h.shape[3]
    _triton_shape_ok = (
        _is_pow2(Nc) and Nc >= 16
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
