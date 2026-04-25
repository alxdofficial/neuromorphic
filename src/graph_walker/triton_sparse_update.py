"""Fused Triton kernel for sparse LIF state update with custom backward.

Replaces the torch.unique + index_add + gather + LIF + index_copy sequence
in graph_walker._step_core_impl with:

- one Triton forward kernel that touches only O(U) rows of the [B*N, D_s] state
- one Triton backward kernel that does the same for the gradient

The `SparseLIFUpdate.apply(s_flat, ...)` entry point:

- preprocesses destinations in PyTorch (unique, stable argsort, segment_offs)
- forward: runs the Triton kernel in place on s_flat, marks s_flat dirty,
  returns s_flat itself (aliased, not a new allocation)
- backward: runs the Triton kernel in place on grad_s_out, producing
  grad_s_in without allocating another [B*N, D_s] tensor

Falls back to a pure-PyTorch implementation on CPU / when Triton is missing.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:                                     # pragma: no cover
    _HAS_TRITON = False


# =====================================================================
# Triton kernels
# =====================================================================


if _HAS_TRITON:

    @triton.jit
    def _sparse_lif_fwd_kernel(
        s_ptr,                 # [B*N, D_s]  -- modified in place
        all_msgs_ptr,          # [M, D_s]
        sort_idx_ptr,          # [M] int64   -- argsort(inverse)
        segment_offs_ptr,      # [U+1] int64 -- boundaries in sort_idx
        unique_dests_ptr,      # [U] int64
        alpha_ptr,             # [N] fp32
        tanh_inc_ptr,          # [U, D_s]    -- save-for-bwd
        s_old_u_ptr,           # [U, D_s]    -- save-for-bwd
        alpha_u_ptr,           # [U] fp32    -- save-for-bwd
        N,
        D_s: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (unique_dest, D tile)."""
        u = tl.program_id(0)
        d_block = tl.program_id(1)

        offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D_s

        dest = tl.load(unique_dests_ptr + u).to(tl.int64)
        n_local = dest % N
        alpha = tl.load(alpha_ptr + n_local).to(tl.float32)

        seg_start = tl.load(segment_offs_ptr + u).to(tl.int64)
        seg_end = tl.load(segment_offs_ptr + u + 1).to(tl.int64)

        # Segment-sum all messages for this destination.
        # Runtime loop — segment sizes typically 1–3 (inject + walker hop).
        incoming = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(seg_start, seg_end):
            m = tl.load(sort_idx_ptr + i).to(tl.int64)
            msg = tl.load(
                all_msgs_ptr + m * D_s + offs, mask=mask, other=0.0,
            ).to(tl.float32)
            incoming += msg

        # Gather current state and LIF-blend.
        s_row_base = dest * D_s
        s_old = tl.load(
            s_ptr + s_row_base + offs, mask=mask, other=0.0,
        ).to(tl.float32)
        # tanh(x) = 2*sigmoid(2x) - 1 — numerically stable, uses Triton sigmoid.
        tanh_in = 2.0 * tl.sigmoid(2.0 * incoming) - 1.0
        s_new = alpha * s_old + (1.0 - alpha) * tanh_in

        store_dtype = s_ptr.dtype.element_ty
        # In-place writeback + save-for-backward (same storage dtype as s).
        tl.store(s_ptr + s_row_base + offs, s_new.to(store_dtype), mask=mask)
        tl.store(
            tanh_inc_ptr + u * D_s + offs, tanh_in.to(store_dtype), mask=mask,
        )
        tl.store(
            s_old_u_ptr + u * D_s + offs, s_old.to(store_dtype), mask=mask,
        )
        if d_block == 0:
            tl.store(alpha_u_ptr + u, alpha)

    @triton.jit
    def _sparse_lif_bwd_kernel(
        grad_s_ptr,            # [B*N, D_s] -- in-place: grad_s_out -> grad_s_in
        unique_dests_ptr,      # [U] int64
        sort_idx_ptr,          # [M] int64
        segment_offs_ptr,      # [U+1] int64
        tanh_inc_ptr,          # [U, D_s]
        s_old_u_ptr,           # [U, D_s]
        alpha_u_ptr,           # [U] fp32
        grad_all_msgs_ptr,     # [M, D_s]  -- output
        grad_alpha_ptr,        # [N] fp32  -- output (atomic accumulation)
        N,
        D_s: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (unique_dest, D tile)."""
        u = tl.program_id(0)
        d_block = tl.program_id(1)

        offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D_s

        dest = tl.load(unique_dests_ptr + u).to(tl.int64)
        n_local = dest % N
        alpha = tl.load(alpha_u_ptr + u).to(tl.float32)

        grad_row_base = dest * D_s
        grad_out = tl.load(
            grad_s_ptr + grad_row_base + offs, mask=mask, other=0.0,
        ).to(tl.float32)
        tanh_in = tl.load(
            tanh_inc_ptr + u * D_s + offs, mask=mask, other=0.0,
        ).to(tl.float32)
        s_old = tl.load(
            s_old_u_ptr + u * D_s + offs, mask=mask, other=0.0,
        ).to(tl.float32)

        # Chain through tanh + affine: grad_incoming = grad_out * (1-alpha) * (1 - tanh^2)
        grad_inc = grad_out * (1.0 - alpha) * (1.0 - tanh_in * tanh_in)

        # alpha gradient: reduce over this d_block's D values, atomic-add into [n_local].
        # OOB lanes were loaded as 0, so products are 0 and the sum is safe.
        alpha_contrib = tl.sum(grad_out * (s_old - tanh_in))
        tl.atomic_add(grad_alpha_ptr + n_local, alpha_contrib)

        # grad_s_in[u] = alpha * grad_s_out[u]. In-place on grad_s_ptr.
        # (Non-touched rows remain identity — no write needed, they already
        # hold the upstream gradient for their pass-through path.)
        store_dtype = grad_s_ptr.dtype.element_ty
        tl.store(
            grad_s_ptr + grad_row_base + offs,
            (grad_out * alpha).to(store_dtype),
            mask=mask,
        )

        # Scatter grad_incoming to every message in this segment.
        msg_store_dtype = grad_all_msgs_ptr.dtype.element_ty
        seg_start = tl.load(segment_offs_ptr + u).to(tl.int64)
        seg_end = tl.load(segment_offs_ptr + u + 1).to(tl.int64)
        for i in range(seg_start, seg_end):
            m = tl.load(sort_idx_ptr + i).to(tl.int64)
            tl.store(
                grad_all_msgs_ptr + m * D_s + offs,
                grad_inc.to(msg_store_dtype),
                mask=mask,
            )


# =====================================================================
# autograd.Function
# =====================================================================


class SparseLIFUpdate(torch.autograd.Function):
    """Sparse LIF state update with fused forward + backward.

    Semantics (equivalent to the PyTorch reference below):

        unique_dests, inverse = torch.unique(all_dests, return_inverse=True)
        incoming[u]  = Σ all_msgs[m]   over m s.t. inverse[m] == u
        s_old[u]     = s_in[unique_dests[u]]
        alpha_u[u]   = alpha[unique_dests[u] % N]
        s_new[u]     = alpha_u[u] * s_old[u] + (1 - alpha_u[u]) * tanh(incoming[u])
        s_out        = s_in.clone(); s_out[unique_dests] = s_new

    Forward is NOT in-place on s_in: upstream ops (e.g. torch.gather(self.s,
    ...)) save self.s for their own backward and a version-bumped view would
    trip autograd's version check. Instead we clone and scatter. The backward,
    however, IS in-place on grad_s_out: that tensor is autograd's handoff to
    us and nothing else reads it. The O(U) touched-row update in backward is
    the main win.

    The clone is the dominant VRAM cost at long TBPTT blocks. Eliminating it
    cleanly requires the "active-row overlay" memory model (detached base +
    sparse differentiable touched rows) — see issue tracker.
    """

    @staticmethod
    def forward(ctx, s_flat, all_msgs, all_dests, alpha, N):
        assert s_flat.is_contiguous(), "s_flat must be contiguous"
        assert all_msgs.is_contiguous(), "all_msgs must be contiguous"
        assert all_dests.is_contiguous(), "all_dests must be contiguous"
        assert alpha.is_contiguous(), "alpha must be contiguous"
        assert s_flat.dtype == all_msgs.dtype, (
            f"dtype mismatch: s_flat={s_flat.dtype}, all_msgs={all_msgs.dtype}"
        )
        assert all_dests.dtype == torch.int64, (
            f"all_dests must be int64, got {all_dests.dtype}"
        )
        assert s_flat.shape[1] == all_msgs.shape[1], (
            "s_flat and all_msgs must share D_s"
        )

        M, D_s = all_msgs.shape
        device = s_flat.device

        # Clone: pass-through base for the new state tensor. The Triton kernel
        # only overwrites touched rows.
        s_out = s_flat.clone()

        # Empty-destination fast path (shouldn't happen but handle cleanly).
        if M == 0:
            empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
            empty_offs = torch.zeros(1, dtype=torch.int64, device=device)
            empty_u = torch.empty(0, D_s, dtype=s_flat.dtype, device=device)
            empty_alpha = torch.empty(0, dtype=torch.float32, device=device)
            ctx.save_for_backward(
                empty_i64, empty_i64, empty_i64, empty_offs,
                empty_u, empty_u, empty_alpha,
            )
            ctx.N = N
            ctx.M = 0
            ctx.D_s = D_s
            ctx.alpha_dtype = alpha.dtype
            return s_out

        # --- Preprocessing (PyTorch) ---
        unique_dests, inverse = torch.unique(all_dests, return_inverse=True)
        U = unique_dests.shape[0]

        # Stable argsort puts identical inverse values in contiguous runs.
        sort_idx = inverse.argsort(stable=True)
        sorted_inverse = inverse[sort_idx]
        # segment_offs[u] = first index in sort_idx belonging to u
        # segment_offs[U] = M (end sentinel)
        segment_offs = torch.searchsorted(
            sorted_inverse, torch.arange(U + 1, device=device),
        )

        # --- Save-for-backward buffers ---
        tanh_inc = torch.empty(U, D_s, dtype=s_flat.dtype, device=device)
        s_old_u = torch.empty(U, D_s, dtype=s_flat.dtype, device=device)
        alpha_u = torch.empty(U, dtype=torch.float32, device=device)

        if _HAS_TRITON and s_flat.is_cuda:
            BLOCK_D = 64
            grid = (U, triton.cdiv(D_s, BLOCK_D))
            # Kernel reads s_flat (unchanged) for s_old_u, writes s_out at
            # touched rows only. s_out already holds a clone of s_flat so
            # non-touched rows are correct by construction.
            _sparse_lif_fwd_kernel[grid](
                s_out, all_msgs, sort_idx, segment_offs,
                unique_dests, alpha,
                tanh_inc, s_old_u, alpha_u,
                N,
                D_s=D_s, BLOCK_D=BLOCK_D,
            )
        else:
            _pytorch_fwd(
                s_out, all_msgs, inverse, unique_dests, alpha, N,
                tanh_inc, s_old_u, alpha_u,
            )

        ctx.save_for_backward(
            unique_dests, inverse, sort_idx, segment_offs,
            tanh_inc, s_old_u, alpha_u,
        )
        ctx.N = N
        ctx.M = M
        ctx.D_s = D_s
        ctx.alpha_dtype = alpha.dtype
        return s_out

    @staticmethod
    def backward(ctx, grad_s_out):
        (
            unique_dests, inverse, sort_idx, segment_offs,
            tanh_inc, s_old_u, alpha_u,
        ) = ctx.saved_tensors
        N = ctx.N
        M = ctx.M
        D_s = ctx.D_s
        alpha_dtype = ctx.alpha_dtype
        device = grad_s_out.device
        U = unique_dests.shape[0]

        # Atomic-add path requires fp32 accumulator.
        grad_alpha_fp32 = torch.zeros(N, dtype=torch.float32, device=device)

        if U == 0 or M == 0:
            grad_all_msgs = torch.zeros(
                M, D_s, dtype=grad_s_out.dtype, device=device,
            )
            return (
                grad_s_out, grad_all_msgs, None,
                grad_alpha_fp32.to(alpha_dtype), None,
            )

        grad_all_msgs = torch.empty(
            M, D_s, dtype=grad_s_out.dtype, device=device,
        )

        # grad_s_out must be contiguous for in-place row addressing.
        if not grad_s_out.is_contiguous():
            grad_s_out = grad_s_out.contiguous()

        if _HAS_TRITON and grad_s_out.is_cuda:
            BLOCK_D = 64
            grid = (U, triton.cdiv(D_s, BLOCK_D))
            _sparse_lif_bwd_kernel[grid](
                grad_s_out, unique_dests, sort_idx, segment_offs,
                tanh_inc, s_old_u, alpha_u,
                grad_all_msgs, grad_alpha_fp32,
                N,
                D_s=D_s, BLOCK_D=BLOCK_D,
            )
        else:
            _pytorch_bwd(
                grad_s_out, unique_dests, inverse,
                tanh_inc, s_old_u, alpha_u, N,
                grad_all_msgs, grad_alpha_fp32,
            )

        return (
            grad_s_out, grad_all_msgs, None,
            grad_alpha_fp32.to(alpha_dtype), None,
        )


# =====================================================================
# PyTorch reference implementations (fallback / correctness oracle)
# =====================================================================


def _pytorch_fwd(
    s_flat, all_msgs, inverse, unique_dests, alpha, N,
    tanh_inc_out, s_old_u_out, alpha_u_out,
):
    """In-place forward using existing PyTorch ops."""
    D_s = s_flat.shape[1]
    U = unique_dests.shape[0]
    device = s_flat.device

    incoming_fp = torch.zeros(U, D_s, dtype=torch.float32, device=device)
    incoming_fp.index_add_(0, inverse, all_msgs.float())

    s_old_fp = s_flat[unique_dests].float()
    alpha_vals_fp = alpha[unique_dests % N].float()
    tanh_fp = torch.tanh(incoming_fp)
    s_new = (
        alpha_vals_fp.unsqueeze(-1) * s_old_fp
        + (1.0 - alpha_vals_fp.unsqueeze(-1)) * tanh_fp
    )
    s_flat.index_copy_(0, unique_dests, s_new.to(s_flat.dtype))

    # Save-for-backward fills (cast to storage dtype to match Triton path).
    tanh_inc_out.copy_(tanh_fp.to(tanh_inc_out.dtype))
    s_old_u_out.copy_(s_old_fp.to(s_old_u_out.dtype))
    alpha_u_out.copy_(alpha_vals_fp)


def _pytorch_bwd(
    grad_s_out, unique_dests, inverse,
    tanh_inc, s_old_u, alpha_u, N,
    grad_all_msgs_out, grad_alpha_out,
):
    """Vectorized PyTorch reference backward."""
    grad_u_fp = grad_s_out[unique_dests].float()
    alpha_u_col = alpha_u.unsqueeze(-1)
    tanh_fp = tanh_inc.float()
    s_old_fp = s_old_u.float()
    one_minus = 1.0 - alpha_u_col

    # grad_alpha: scatter-add (grad_u · (s_old - tanh_in)) summed over D_s
    alpha_contrib = (grad_u_fp * (s_old_fp - tanh_fp)).sum(dim=-1)
    grad_alpha_out.index_add_(0, unique_dests % N, alpha_contrib)

    # grad_incoming
    grad_inc = grad_u_fp * one_minus * (1.0 - tanh_fp * tanh_fp)

    # grad_all_msgs[m] = grad_incoming[inverse[m]]
    grad_all_msgs_out.copy_(grad_inc[inverse].to(grad_all_msgs_out.dtype))

    # grad_s_in[u] = alpha * grad_s_out[u] — in-place on grad_s_out.
    grad_s_out.index_copy_(
        0, unique_dests, (grad_u_fp * alpha_u_col).to(grad_s_out.dtype),
    )


# =====================================================================
# Pure-torch cudagraph-compatible variant
# =====================================================================


def sparse_lif_update_puretorch(
    s_flat: torch.Tensor,
    all_msgs: torch.Tensor,
    all_dests: torch.Tensor,
    alpha: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Sparse LIF update built from standard PyTorch ops only.

    Equivalent semantics to `SparseLIFUpdate.apply` but no custom
    autograd.Function and no `torch.unique` — everything is static-shape
    so torch.compile + reduce-overhead can capture the whole step into
    a CUDA graph.

    Trade-off: this path computes the LIF blend densely over all `[B*N, D_s]`
    rows (then masks untouched rows back to s_old), instead of the Triton
    path's O(U) sparse update. Fine when launch overhead dominates (most
    of our hot path); revisit if memory bandwidth becomes the bottleneck.
    """
    BN, D_s = s_flat.shape
    M = all_msgs.shape[0]
    device = s_flat.device

    # Aggregate messages per destination (handles duplicates via index_add).
    # Compute everything in fp32 internally so gradient precision matches the
    # Triton path; cast output back to s_flat.dtype at the end.
    incoming = torch.zeros(BN, D_s, dtype=torch.float32, device=device)
    incoming = incoming.index_add(0, all_dests, all_msgs.float())

    # Touched-row mask via index_add of ones.
    counts = torch.zeros(BN, dtype=torch.float32, device=device)
    ones_m = torch.ones(M, dtype=torch.float32, device=device)
    counts = counts.index_add(0, all_dests, ones_m)
    touched_mask = (counts > 0).float().unsqueeze(-1)                  # [BN, 1]

    # Per-row alpha lookup, kept in fp32. Inductor can constant-fold the
    # `arange % N` expression when BN, N are static.
    n_index = torch.arange(BN, device=device) % N
    alpha_row = alpha[n_index].float().unsqueeze(-1)                   # [BN, 1]

    # LIF blend over all rows in fp32; mask gates touched vs untouched.
    s_flat_fp = s_flat.float()
    tanh_inc = torch.tanh(incoming)
    s_blend = alpha_row * s_flat_fp + (1.0 - alpha_row) * tanh_inc
    s_new = touched_mask * s_blend + (1.0 - touched_mask) * s_flat_fp

    return s_new.to(s_flat.dtype)


# =====================================================================
# Public convenience wrapper
# =====================================================================


def sparse_lif_update(
    s_flat: torch.Tensor,
    all_msgs: torch.Tensor,
    all_dests: torch.Tensor,
    alpha: torch.Tensor,
    N: int,
    *,
    backend: str = "puretorch",
) -> torch.Tensor:
    """Public entry point. backend in {'puretorch', 'triton'}.

    'puretorch' (default): cudagraph-compatible, dense LIF blend with mask.
        Inductor can fuse the body into Triton kernels under torch.compile,
        so we don't lose much speed at the bench scale and we unlock
        reduce-overhead's CUDA graph capture.

    'triton': original sparse Triton kernel + custom autograd.Function.
        Lower memory at production scale but cudagraph-incompatible
        (allocations escape the pool).
    """
    if backend == "puretorch":
        return sparse_lif_update_puretorch(
            s_flat, all_msgs, all_dests, alpha, N,
        )
    if backend == "triton":
        return SparseLIFUpdate.apply(s_flat, all_msgs, all_dests, alpha, N)
    raise ValueError(f"unknown backend: {backend!r}")
