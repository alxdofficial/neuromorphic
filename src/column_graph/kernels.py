"""Triton kernels for ColumnGraph hot-path ops.

Current scope: atomic-free scatter-gather for the edge-weighted message
sum. Replaces:

    msg_edge = w_out_flat.unsqueeze(-1) * m_out.unsqueeze(2)   # materializes [B, N, K, D_s]
    msg_flat = msg_edge.reshape(B, N * K, D_s)
    incoming = torch.zeros(...).index_add(1, edge_dst, msg_flat)  # atomic scatter

with a Triton kernel that, per destination column, reads its K_in_max
in-edges, fetches w_out and source m_out, accumulates the weighted sum
in registers, writes once. No atomic adds, no [B, N, K, D_s]
materialization.

Profile on the compiled block-path showed `aten::index_add` at 30% of
CUDA time at BS=8 T=32. This kernel targets that.

Backward math (implemented in the autograd.Function, also Triton):
  grad_m_out[b, src] = Σ_{k_out: edge (src→dst) exists} w[b, e] * grad_incoming[b, dst]
  grad_w_out[b, e]   = <m_out[b, edge_src[e]], grad_incoming[b, edge_dst[e]]>

The backward for grad_m_out has the *same structure* as the forward but
uses the OUT-adjacency (each source has K fixed out-edges) — also
gatherable without atomics.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


# =====================================================================
# Forward kernel: weighted in-gather
# =====================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _weighted_gather_fwd_kernel(
        m_out_ptr,                # [B, N, D_s]   — same dtype as m_out (bf16/fp32)
        w_out_ptr,                # [B, N*K]      — same dtype
        in_src_ptr,               # [N, K_in_max] int64
        in_edge_ptr,              # [N, K_in_max] int64
        in_mask_ptr,              # [N, K_in_max] fp32
        incoming_ptr,             # [B, N, D_s]   — output
        # shape
        N: tl.constexpr,
        N_edges: tl.constexpr,
        D_s: tl.constexpr,
        K_in_max: tl.constexpr,
        # strides (elements)
        m_stride_b, m_stride_n,
        w_stride_b,
        in_stride_n,              # stride for in_src/in_edge/in_mask at dim 0
        inc_stride_b, inc_stride_n,
    ):
        """One program per (batch, destination column).

        Accumulates in fp32, casts output back to storage dtype.
        """
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)

        d_offs = tl.arange(0, D_s)
        acc = tl.zeros([D_s], dtype=tl.float32)

        in_base = pid_c * in_stride_n

        for k in tl.static_range(K_in_max):
            src = tl.load(in_src_ptr + in_base + k)           # int64
            edge_flat = tl.load(in_edge_ptr + in_base + k)    # int64
            mask_val = tl.load(in_mask_ptr + in_base + k)     # fp32

            w = tl.load(w_out_ptr + pid_b * w_stride_b + edge_flat).to(tl.float32)
            msg_ptr = m_out_ptr + pid_b * m_stride_b + src * m_stride_n + d_offs
            msg = tl.load(msg_ptr).to(tl.float32)

            acc += mask_val * w * msg

        out_ptr = incoming_ptr + pid_b * inc_stride_b + pid_c * inc_stride_n + d_offs
        tl.store(out_ptr, acc)  # caller-supplied dtype via ptr bit-width


    # =================================================================
    # Backward kernels
    # =================================================================

    @triton.jit
    def _weighted_gather_bwd_m_kernel(
        grad_incoming_ptr,        # [B, N, D_s]  (grad wrt incoming from upstream)
        w_out_ptr,                # [B, N*K]
        out_nbrs_ptr,             # [N, K]  int64 — for each src column, its K out-neighbours
        grad_m_out_ptr,           # [B, N, D_s]  (accumulated — zeroed by caller)
        # shape
        N: tl.constexpr,
        D_s: tl.constexpr,
        K: tl.constexpr,
        # strides
        gin_stride_b, gin_stride_n,
        w_stride_b,
        out_stride_n,
        gm_stride_b, gm_stride_n,
    ):
        """grad_m_out[b, src] = Σ_k w[b, src*K+k] * grad_incoming[b, out_nbrs[src, k]]

        One program per (batch, source column). Iterates over K out-edges
        of this source. K is small (32), static_range-unrollable.
        """
        pid_b = tl.program_id(0)
        pid_src = tl.program_id(1)

        d_offs = tl.arange(0, D_s)
        acc = tl.zeros([D_s], dtype=tl.float32)

        nbr_base = pid_src * out_stride_n

        for k in tl.static_range(K):
            dst = tl.load(out_nbrs_ptr + nbr_base + k)
            edge_flat = pid_src * K + k
            w = tl.load(w_out_ptr + pid_b * w_stride_b + edge_flat).to(tl.float32)
            grad_ptr = grad_incoming_ptr + pid_b * gin_stride_b + dst * gin_stride_n + d_offs
            g = tl.load(grad_ptr).to(tl.float32)
            acc += w * g

        out_ptr = grad_m_out_ptr + pid_b * gm_stride_b + pid_src * gm_stride_n + d_offs
        tl.store(out_ptr, acc)


    @triton.jit
    def _weighted_gather_bwd_w_kernel(
        m_out_ptr,                # [B, N, D_s]
        grad_incoming_ptr,        # [B, N, D_s]
        edge_src_ptr,             # [N*K] int64
        edge_dst_ptr,             # [N*K] int64
        grad_w_ptr,               # [B, N*K]  — output
        # shape
        N_edges: tl.constexpr,
        D_s: tl.constexpr,
        # strides
        m_stride_b, m_stride_n,
        gin_stride_b, gin_stride_n,
        gw_stride_b,
    ):
        """grad_w_out[b, e] = <m_out[b, edge_src[e], :], grad_incoming[b, edge_dst[e], :]>

        Grid: (N_edges, B). N_edges goes along X because it can be huge
        (>65535); CUDA's Y limit is 65535.
        """
        pid_e = tl.program_id(0)
        pid_b = tl.program_id(1)

        src = tl.load(edge_src_ptr + pid_e)
        dst = tl.load(edge_dst_ptr + pid_e)

        d_offs = tl.arange(0, D_s)
        m = tl.load(m_out_ptr + pid_b * m_stride_b + src * m_stride_n + d_offs).to(tl.float32)
        g = tl.load(grad_incoming_ptr + pid_b * gin_stride_b + dst * gin_stride_n + d_offs).to(tl.float32)

        dot = tl.sum(m * g, axis=0)
        tl.store(grad_w_ptr + pid_b * gw_stride_b + pid_e, dot)


# =====================================================================
# Reference implementations (CPU fallback + correctness oracle)
# =====================================================================


def weighted_gather_reference(
    m_out: torch.Tensor,          # [B, N, D_s]
    w_out_flat: torch.Tensor,     # [B, N*K]
    edge_src: torch.Tensor,       # [N*K]
    edge_dst: torch.Tensor,       # [N*K]
) -> torch.Tensor:
    """PyTorch reference — matches the math in _propagate_pure's scatter branch."""
    B, N, D_s = m_out.shape
    msg_flat = w_out_flat.unsqueeze(-1) * m_out[:, edge_src]  # [B, N*K, D_s]
    incoming = torch.zeros_like(m_out)
    incoming.index_add_(1, edge_dst, msg_flat)
    return incoming


# =====================================================================
# autograd.Function: Triton forward + Triton backward
# =====================================================================


class WeightedGather(torch.autograd.Function):
    """Forward:
        incoming[b, c, :] = Σ_{e: edge_dst[e]=c} w[b, e] * m_out[b, edge_src[e], :]

    Backward:
        grad_m_out[b, s, :] = Σ_{e: edge_src[e]=s} w[b, e] * grad_incoming[b, edge_dst[e], :]
        grad_w_out[b, e]   = <m_out[b, edge_src[e], :], grad_incoming[b, edge_dst[e], :]>
    """

    @staticmethod
    def forward(
        ctx: Any,
        m_out: torch.Tensor,              # [B, N, D_s]
        w_out_flat: torch.Tensor,         # [B, N*K]
        edge_src: torch.Tensor,           # [N*K] int64
        edge_dst: torch.Tensor,           # [N*K] int64
        out_nbrs: torch.Tensor,           # [N, K] int64
        in_src: torch.Tensor,             # [N, K_in_max] int64
        in_edge_flat: torch.Tensor,       # [N, K_in_max] int64
        in_mask: torch.Tensor,            # [N, K_in_max] fp32
        K_in_max: int,
        K: int,
    ) -> torch.Tensor:
        assert m_out.is_cuda and TRITON_AVAILABLE

        B, N, D_s = m_out.shape
        N_edges = N * K

        m_out_c = m_out.contiguous()
        w_out_c = w_out_flat.contiguous()
        incoming = torch.empty_like(m_out_c)

        grid = (B, N)
        _weighted_gather_fwd_kernel[grid](
            m_out_c, w_out_c, in_src, in_edge_flat, in_mask,
            incoming,
            N=N, N_edges=N_edges, D_s=D_s, K_in_max=K_in_max,
            m_stride_b=m_out_c.stride(0), m_stride_n=m_out_c.stride(1),
            w_stride_b=w_out_c.stride(0),
            in_stride_n=in_src.stride(0),
            inc_stride_b=incoming.stride(0), inc_stride_n=incoming.stride(1),
        )

        ctx.save_for_backward(
            m_out_c, w_out_c, edge_src, edge_dst, out_nbrs,
        )
        ctx.K = K
        ctx.K_in_max = K_in_max
        return incoming

    @staticmethod
    def backward(
        ctx: Any, grad_incoming: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None, None, None, None, None, None]:
        m_out, w_out_flat, edge_src, edge_dst, out_nbrs = ctx.saved_tensors
        K = ctx.K

        B, N, D_s = m_out.shape
        N_edges = N * K

        grad_incoming_c = grad_incoming.contiguous()
        grad_m_out = torch.empty_like(m_out)
        grad_w_out = torch.empty_like(w_out_flat)

        grid_m = (B, N)
        _weighted_gather_bwd_m_kernel[grid_m](
            grad_incoming_c, w_out_flat, out_nbrs, grad_m_out,
            N=N, D_s=D_s, K=K,
            gin_stride_b=grad_incoming_c.stride(0), gin_stride_n=grad_incoming_c.stride(1),
            w_stride_b=w_out_flat.stride(0),
            out_stride_n=out_nbrs.stride(0),
            gm_stride_b=grad_m_out.stride(0), gm_stride_n=grad_m_out.stride(1),
        )

        grid_w = (N_edges, B)
        _weighted_gather_bwd_w_kernel[grid_w](
            m_out, grad_incoming_c, edge_src, edge_dst, grad_w_out,
            N_edges=N_edges, D_s=D_s,
            m_stride_b=m_out.stride(0), m_stride_n=m_out.stride(1),
            gin_stride_b=grad_incoming_c.stride(0), gin_stride_n=grad_incoming_c.stride(1),
            gw_stride_b=grad_w_out.stride(0),
        )

        return grad_m_out, grad_w_out, None, None, None, None, None, None, None, None


def weighted_gather(
    m_out: torch.Tensor,
    w_out_flat: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    out_nbrs: torch.Tensor,
    in_src: torch.Tensor,
    in_edge_flat: torch.Tensor,
    in_mask: torch.Tensor,
    K_in_max: int,
    K: int,
) -> torch.Tensor:
    """Public entry: Triton-fused edge-weighted in-gather.

    Falls back to PyTorch reference on CPU.
    """
    if not (TRITON_AVAILABLE and m_out.is_cuda):
        return weighted_gather_reference(m_out, w_out_flat, edge_src, edge_dst)
    return WeightedGather.apply(
        m_out, w_out_flat, edge_src, edge_dst, out_nbrs,
        in_src, in_edge_flat, in_mask, K_in_max, K,
    )
