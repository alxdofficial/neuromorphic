"""Sparse LIF deposit kernel — cudagraph-friendly Triton + pure-torch fallback.

Replaces ``triton_sparse_update.py`` from the pre-rewrite tree. Two changes
that matter for the rewrite:

1. **No ``torch.unique``.** The previous Triton path called
   ``torch.unique(all_dests, return_inverse=True)`` which has dynamic
   output shape — incompatible with CUDA-graph capture. We replace it
   with a static-shape pipeline:

       sorted_dests, sort_idx = all_dests.sort(stable=True)   # [M_max]
       diff = sorted_dests differ from previous?               # [M_max] int64
       unique_idx_in_sorted = diff.cumsum(0) - 1               # [M_max]
       U_real = diff.sum()                                     # tensor scalar
       unique_dests pre-filled with sentinel BN, then          # [U_max]
           scatter_(unique_idx_in_sorted, sorted_dests)
       segment_offs = searchsorted(unique_idx_in_sorted,
                                   arange(U_max + 1))          # [U_max + 1]

   The kernel runs over a static grid ``(U_max, D_tile)`` and exits
   early on the sentinel rows (``dest >= BN``).

2. **Pre-allocated save-for-backward buffers.** The forward writes into
   caller-supplied buffers so capture sees stable addresses across
   replays. The backward kernel reads from those same buffers.

Pure-torch fallback (``sparse_lif_update_puretorch``) is the default for
non-CUDA paths and stays bit-exact with the Triton output (within bf16
tolerance) — the existing parity tests catch regressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:                                     # pragma: no cover
    HAS_TRITON = False


# =====================================================================
# Triton kernels — static-shape, cudagraph-friendly
# =====================================================================


if HAS_TRITON:

    @triton.jit
    def _lif_deposit_fwd_kernel(
        s_in_ptr,             # [BN, D_s] (read)
        s_out_ptr,            # [BN, D_s] (write — caller pre-copied s_in)
        msgs_ptr,             # [M_max, D_s]
        sort_idx_ptr,         # [M_max] int64 — argsort(all_dests)
        segment_offs_ptr,     # [U_max + 1] int64 — boundaries in sort_idx
        unique_dests_ptr,     # [U_max] int64 — sentinel BN for padding
        alpha_ptr,            # [N] fp32
        # Save-for-backward buffers (sentinel slots remain uninitialized;
        # caller relies on the same sentinel guard to skip them in bwd):
        tanh_inc_save_ptr,    # [U_max, D_s]
        s_old_save_ptr,       # [U_max, D_s]
        alpha_save_ptr,       # [U_max] fp32
        BN, N,
        D_s: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (unique_dest, D tile)."""
        u = tl.program_id(0)
        d_block = tl.program_id(1)

        dest = tl.load(unique_dests_ptr + u).to(tl.int64)
        # Sentinel skip — padding rows (dest == BN) do nothing.
        if dest >= BN:
            return

        offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D_s

        n_local = dest % N
        alpha = tl.load(alpha_ptr + n_local).to(tl.float32)

        seg_start = tl.load(segment_offs_ptr + u).to(tl.int64)
        seg_end = tl.load(segment_offs_ptr + u + 1).to(tl.int64)

        # Segment-sum messages for this destination.
        incoming = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(seg_start, seg_end):
            m = tl.load(sort_idx_ptr + i).to(tl.int64)
            msg = tl.load(
                msgs_ptr + m * D_s + offs, mask=mask, other=0.0,
            ).to(tl.float32)
            incoming += msg

        s_row_base = dest * D_s
        s_old = tl.load(
            s_in_ptr + s_row_base + offs, mask=mask, other=0.0,
        ).to(tl.float32)
        # tanh(x) = 2σ(2x) - 1 — numerically stable.
        tanh_in = 2.0 * tl.sigmoid(2.0 * incoming) - 1.0
        s_new = alpha * s_old + (1.0 - alpha) * tanh_in

        store_dtype = s_out_ptr.dtype.element_ty
        tl.store(
            s_out_ptr + s_row_base + offs, s_new.to(store_dtype), mask=mask,
        )
        tl.store(
            tanh_inc_save_ptr + u * D_s + offs,
            tanh_in.to(store_dtype),
            mask=mask,
        )
        tl.store(
            s_old_save_ptr + u * D_s + offs,
            s_old.to(store_dtype),
            mask=mask,
        )
        if d_block == 0:
            tl.store(alpha_save_ptr + u, alpha)

    @triton.jit
    def _lif_deposit_bwd_kernel(
        grad_s_ptr,            # [BN, D_s] -- in-place: grad_out -> grad_in
        unique_dests_ptr,      # [U_max] int64
        sort_idx_ptr,          # [M_max] int64
        segment_offs_ptr,      # [U_max + 1] int64
        tanh_inc_ptr,          # [U_max, D_s]
        s_old_u_ptr,           # [U_max, D_s]
        alpha_u_ptr,           # [U_max] fp32
        grad_msgs_ptr,         # [M_max, D_s] (write — sentinel slots zeroed by caller)
        grad_alpha_ptr,        # [N] fp32 (atomic accum)
        BN, N,
        D_s: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per (unique_dest, D tile)."""
        u = tl.program_id(0)
        d_block = tl.program_id(1)

        dest = tl.load(unique_dests_ptr + u).to(tl.int64)
        if dest >= BN:
            return  # sentinel — no real work

        offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D_s

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

        # ∂L/∂incoming = grad_out · (1-α) · (1 - tanh²)
        grad_inc = grad_out * (1.0 - alpha) * (1.0 - tanh_in * tanh_in)

        # ∂L/∂α reduce over D, atomic-add into [n_local].
        alpha_contrib = tl.sum(grad_out * (s_old - tanh_in))
        tl.atomic_add(grad_alpha_ptr + n_local, alpha_contrib)

        # ∂L/∂s_in[dest] = α · ∂L/∂s_out[dest]; non-touched rows pass through identity.
        store_dtype = grad_s_ptr.dtype.element_ty
        tl.store(
            grad_s_ptr + grad_row_base + offs,
            (grad_out * alpha).to(store_dtype),
            mask=mask,
        )

        # Scatter grad_inc to every message in this segment.
        msg_dtype = grad_msgs_ptr.dtype.element_ty
        seg_start = tl.load(segment_offs_ptr + u).to(tl.int64)
        seg_end = tl.load(segment_offs_ptr + u + 1).to(tl.int64)
        for i in range(seg_start, seg_end):
            m = tl.load(sort_idx_ptr + i).to(tl.int64)
            tl.store(
                grad_msgs_ptr + m * D_s + offs,
                grad_inc.to(msg_dtype),
                mask=mask,
            )


# =====================================================================
# Static-shape preprocessing
# =====================================================================


@dataclass
class LIFScratch:
    """Pre-allocated scratch buffers for one LIF deposit step.

    Shape arguments are constants for the entire training run — once
    allocated, addresses stay stable across cudagraph replays.
    """
    sorted_dests: torch.Tensor       # [M_max] int64
    sort_idx: torch.Tensor           # [M_max] int64
    diff: torch.Tensor               # [M_max] int64
    unique_idx_in_sorted: torch.Tensor  # [M_max] int64
    unique_dests: torch.Tensor       # [U_max] int64
    segment_offs: torch.Tensor       # [U_max + 1] int64
    arange_u: torch.Tensor           # [U_max + 1] int64 (constant)
    tanh_inc_save: torch.Tensor      # [U_max, D_s]
    s_old_save: torch.Tensor         # [U_max, D_s]
    alpha_save: torch.Tensor         # [U_max] fp32

    @classmethod
    def allocate(
        cls, M_max: int, U_max: int, D_s: int,
        device: torch.device, dtype: torch.dtype,
    ) -> "LIFScratch":
        i64 = dict(dtype=torch.int64, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        return cls(
            sorted_dests=torch.empty(M_max, **i64),
            sort_idx=torch.empty(M_max, **i64),
            diff=torch.empty(M_max, **i64),
            unique_idx_in_sorted=torch.empty(M_max, **i64),
            unique_dests=torch.full((U_max,), 0, **i64),
            segment_offs=torch.empty(U_max + 1, **i64),
            arange_u=torch.arange(U_max + 1, device=device, dtype=torch.int64),
            tanh_inc_save=torch.empty(U_max, D_s, dtype=dtype, device=device),
            s_old_save=torch.empty(U_max, D_s, dtype=dtype, device=device),
            alpha_save=torch.empty(U_max, **f32),
        )


def _lif_preprocess(
    all_dests: torch.Tensor,          # [M_max] int64 — pad with sentinel BN
    BN: int,
    scratch: LIFScratch,
) -> None:
    """Compute sort_idx, unique_dests (sentinel-padded), segment_offs in scratch.

    Pure-torch, static shapes. Inductor / autograd treat this as a
    constant w.r.t. value-flow but cudagraph-safe because every output
    shape is fixed at scratch-allocation time.
    """
    sorted_dests, sort_idx = all_dests.sort(stable=True)
    scratch.sorted_dests.copy_(sorted_dests)
    scratch.sort_idx.copy_(sort_idx)

    M_max = all_dests.shape[0]
    # diff[m] = 1 if sorted_dests[m] != sorted_dests[m-1] else 0; diff[0] = 1.
    diff = torch.empty_like(scratch.diff)
    diff[0] = 1
    diff[1:] = (sorted_dests[1:] != sorted_dests[:-1]).to(torch.int64)
    scratch.diff.copy_(diff)
    scratch.unique_idx_in_sorted.copy_(diff.cumsum(0) - 1)

    # Reset unique_dests to sentinel; scatter overwrites real entries.
    scratch.unique_dests.fill_(BN)
    scratch.unique_dests.scatter_(0, scratch.unique_idx_in_sorted, sorted_dests)

    # segment_offs[u] = first sort_idx belonging to group u.
    # For padding u >= U_real, returns M_max (empty range in kernel loop).
    scratch.segment_offs.copy_(
        torch.searchsorted(scratch.unique_idx_in_sorted, scratch.arange_u),
    )


# =====================================================================
# autograd.Function
# =====================================================================


class LIFDepositFunction(torch.autograd.Function):
    """Sparse LIF deposit. Allocates per-call scratch (no caller-supplied
    reusable scratch — that pattern conflicted with multi-step usage where
    autograd's saved-for-backward refs version-clash with the next step's
    in-place preprocess writes).

    Forward: returns ``s_out``, a fresh tensor of shape [BN, D_s] equal to
    ``alpha · s_in + (1-alpha) · tanh(Σ msgs)`` at touched rows and equal
    to ``s_in`` at untouched rows.

    The Triton path requires ``all_dests`` padded to a static ``M_max`` with
    sentinel = ``B*N`` for the unused slots. Per-step scratch buffers
    (sort_idx / unique_dests / segment_offs + the per-U save-for-backward
    arrays) are allocated fresh inside this function. Under cudagraph
    capture they land in the captured pool and reuse the same address
    across replays — same memory characteristics as the prior
    static-buffer design, without the multi-step version-mismatch trap.

    Backward returns gradients w.r.t. ``s_in``, ``all_msgs``, ``alpha``.
    """

    @staticmethod
    def forward(
        ctx,
        s_in: torch.Tensor,           # [BN, D_s]
        all_msgs: torch.Tensor,       # [M_max, D_s]
        all_dests: torch.Tensor,      # [M_max] int64 (sentinel = BN for pad)
        alpha: torch.Tensor,          # [N] fp32
        N: int,
    ) -> torch.Tensor:
        BN, D_s = s_in.shape
        M_max = all_msgs.shape[0]
        U_max = M_max  # in the worst case all dests are unique

        # Pre-copy: kernel only writes touched rows.
        s_out = s_in.clone()

        # Allocate per-call scratch INSIDE forward. These buffers are passed
        # to save_for_backward and live until backward consumes them — they
        # MUST NOT be reused across calls, else autograd's version check on
        # saved tensors trips when the next step's preprocess mutates them.
        # Under cudagraph, the captured allocator pool reuses the same
        # addresses across replays so this is still O(1) memory amortized.
        device = s_in.device
        sort_idx = torch.empty(M_max, dtype=torch.int64, device=device)
        diff = torch.empty(M_max, dtype=torch.int64, device=device)
        unique_idx_in_sorted = torch.empty(M_max, dtype=torch.int64, device=device)
        unique_dests = torch.full((U_max,), BN, dtype=torch.int64, device=device)
        segment_offs = torch.empty(U_max + 1, dtype=torch.int64, device=device)
        arange_u = torch.arange(U_max + 1, device=device, dtype=torch.int64)
        tanh_inc_save = torch.empty(U_max, D_s, dtype=s_in.dtype, device=device)
        s_old_save = torch.empty(U_max, D_s, dtype=s_in.dtype, device=device)
        alpha_save = torch.empty(U_max, dtype=torch.float32, device=device)

        if M_max == 0:
            ctx.save_for_backward(
                unique_dests, sort_idx, segment_offs,
                tanh_inc_save, s_old_save, alpha_save,
            )
            ctx.N = N
            ctx.BN = BN
            ctx.D_s = D_s
            ctx.M_max = M_max
            ctx.U_max = U_max
            ctx.alpha_dtype = alpha.dtype
            return s_out

        # Static-shape preprocess (inlined; no shared scratch).
        sorted_dests, sort_idx_local = all_dests.sort(stable=True)
        sort_idx.copy_(sort_idx_local)
        diff[0] = 1
        diff[1:] = (sorted_dests[1:] != sorted_dests[:-1]).to(torch.int64)
        unique_idx_in_sorted.copy_(diff.cumsum(0) - 1)
        unique_dests.scatter_(0, unique_idx_in_sorted, sorted_dests)
        segment_offs.copy_(
            torch.searchsorted(unique_idx_in_sorted, arange_u),
        )

        if HAS_TRITON and s_in.is_cuda:
            BLOCK_D = 64
            grid = (U_max, triton.cdiv(D_s, BLOCK_D))
            _lif_deposit_fwd_kernel[grid](
                s_in, s_out, all_msgs,
                sort_idx, segment_offs,
                unique_dests, alpha,
                tanh_inc_save, s_old_save, alpha_save,
                BN, N,
                D_s=D_s, BLOCK_D=BLOCK_D,
            )
        else:
            # CPU/no-Triton fallback. Build a temporary LIFScratch-like view
            # so the existing _pure_torch_fwd helper works unchanged.
            from types import SimpleNamespace
            _scratch = SimpleNamespace(
                sorted_dests=sorted_dests, sort_idx=sort_idx, diff=diff,
                unique_idx_in_sorted=unique_idx_in_sorted,
                unique_dests=unique_dests, segment_offs=segment_offs,
                arange_u=arange_u,
                tanh_inc_save=tanh_inc_save, s_old_save=s_old_save,
                alpha_save=alpha_save,
            )
            _pure_torch_fwd(
                s_in, s_out, all_msgs, all_dests, alpha, N, BN, _scratch,
            )

        ctx.save_for_backward(
            unique_dests, sort_idx, segment_offs,
            tanh_inc_save, s_old_save, alpha_save,
        )
        ctx.N = N
        ctx.BN = BN
        ctx.D_s = D_s
        ctx.M_max = M_max
        ctx.U_max = U_max
        ctx.alpha_dtype = alpha.dtype
        return s_out

    @staticmethod
    def backward(ctx, grad_s_out):
        (
            unique_dests, sort_idx, segment_offs,
            tanh_inc, s_old_u, alpha_u,
        ) = ctx.saved_tensors
        N = ctx.N
        BN = ctx.BN
        D_s = ctx.D_s
        M_max = ctx.M_max
        U_max = ctx.U_max
        alpha_dtype = ctx.alpha_dtype
        device = grad_s_out.device

        grad_alpha_fp32 = torch.zeros(N, dtype=torch.float32, device=device)

        if M_max == 0:
            grad_msgs = torch.zeros(0, D_s, dtype=grad_s_out.dtype, device=device)
            return (
                grad_s_out, grad_msgs, None,
                grad_alpha_fp32.to(alpha_dtype), None,
            )

        # Sentinel rows of grad_msgs need zero, otherwise inductor / autograd
        # might propagate uninitialized memory back through the chain.
        grad_msgs = torch.zeros(M_max, D_s, dtype=grad_s_out.dtype, device=device)

        if not grad_s_out.is_contiguous():
            grad_s_out = grad_s_out.contiguous()

        if HAS_TRITON and grad_s_out.is_cuda:
            BLOCK_D = 64
            grid = (U_max, triton.cdiv(D_s, BLOCK_D))
            _lif_deposit_bwd_kernel[grid](
                grad_s_out, unique_dests, sort_idx, segment_offs,
                tanh_inc, s_old_u, alpha_u,
                grad_msgs, grad_alpha_fp32,
                BN, N,
                D_s=D_s, BLOCK_D=BLOCK_D,
            )
        else:
            _pure_torch_bwd(
                grad_s_out, unique_dests, sort_idx, segment_offs,
                tanh_inc, s_old_u, alpha_u, BN, N,
                grad_msgs, grad_alpha_fp32,
            )

        return (
            grad_s_out, grad_msgs, None,
            grad_alpha_fp32.to(alpha_dtype), None,
        )


# =====================================================================
# Pure-torch reference (fallback + correctness oracle)
# =====================================================================


def _pure_torch_fwd(
    s_in, s_out, all_msgs, all_dests, alpha, N, BN, scratch,
):
    """Reference forward — used by CPU path and test parity.

    Mirrors the Triton kernel exactly, including populating the
    save-for-backward buffers (``tanh_inc_save``, ``s_old_save``,
    ``alpha_save``) per ``unique_dests`` slot. Without these,
    ``_pure_torch_bwd`` reads uninitialized memory and produces garbage
    gradients — earlier passes accidentally left this as ``pass``.

    Sentinel slots (``unique_dests[u] >= BN``) are skipped; the kernel /
    backward both early-exit on the same sentinel guard so leaving those
    save buffers untouched is fine.
    """
    D_s = s_in.shape[1]
    device = s_in.device
    U_max = scratch.unique_dests.shape[0]

    # Forward: dense LIF blend over all rows, masked by touched.
    incoming_fp = torch.zeros(BN, D_s, dtype=torch.float32, device=device)
    incoming_fp.index_add_(0, all_dests.clamp(max=BN - 1), all_msgs.float())

    counts = torch.zeros(BN, dtype=torch.float32, device=device)
    ones = torch.ones(all_dests.shape[0], dtype=torch.float32, device=device)
    counts.index_add_(0, all_dests.clamp(max=BN - 1), ones)
    touched = (counts > 0)

    n_index = torch.arange(BN, device=device) % N
    alpha_row = alpha[n_index].float().unsqueeze(-1)

    s_in_fp = s_in.float()
    tanh_inc_full = torch.tanh(incoming_fp)
    s_blend = alpha_row * s_in_fp + (1.0 - alpha_row) * tanh_inc_full

    s_out.copy_(
        torch.where(touched.unsqueeze(-1), s_blend, s_in_fp).to(s_in.dtype),
    )

    # Save-for-backward: gather per-unique-dest from the dense buffers.
    # ``unique_dests`` was filled by ``_lif_preprocess`` and is
    # sentinel-padded with ``BN`` for slots beyond the real U. We clamp
    # those reads to a valid index (BN-1) so .index_select is safe; the
    # backward kernel will skip sentinel slots anyway via its dest>=BN
    # guard.
    safe_dests = scratch.unique_dests.clamp(max=BN - 1)
    scratch.tanh_inc_save.copy_(
        tanh_inc_full.index_select(0, safe_dests).to(scratch.tanh_inc_save.dtype),
    )
    scratch.s_old_save.copy_(
        s_in_fp.index_select(0, safe_dests).to(scratch.s_old_save.dtype),
    )
    scratch.alpha_save.copy_(
        alpha[(safe_dests % N).long()].float(),
    )


def _pure_torch_bwd(
    grad_s_out, unique_dests, sort_idx, segment_offs,
    tanh_inc, s_old_u, alpha_u, BN, N,
    grad_msgs, grad_alpha,
):
    """Reference backward — used by CPU path."""
    U_max = unique_dests.shape[0]
    D_s = grad_msgs.shape[1]
    valid = unique_dests < BN

    for u in range(U_max):
        if not valid[u]:
            continue
        dest = int(unique_dests[u].item())
        n_local = dest % N
        alpha = float(alpha_u[u].item())
        grad_out = grad_s_out[dest].float()
        tin = tanh_inc[u].float()
        sold = s_old_u[u].float()

        grad_inc = grad_out * (1.0 - alpha) * (1.0 - tin * tin)
        grad_alpha[n_local] += float(((grad_out * (sold - tin)).sum()).item())
        grad_s_out[dest] = (grad_out * alpha).to(grad_s_out.dtype)

        seg_start = int(segment_offs[u].item())
        seg_end = int(segment_offs[u + 1].item())
        for i in range(seg_start, seg_end):
            m = int(sort_idx[i].item())
            grad_msgs[m] = grad_inc.to(grad_msgs.dtype)


# =====================================================================
# Pure-torch sparse-LIF (cudagraph-safe path; no custom autograd)
# =====================================================================


def sparse_lif_update_puretorch(
    s_flat: torch.Tensor,
    all_msgs: torch.Tensor,
    all_dests: torch.Tensor,
    alpha: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Dense LIF blend with mask. Cudagraph-friendly, no autograd.Function.

    Trade-off: O(BN) memory traffic vs the Triton path's O(U). At our
    bench scale (B=4, N=1024, U≈16) the dense path is fine because launch
    overhead dominates; revisit only if memory bandwidth bottlenecks.

    .. warning::
        Cannot handle sentinel-padded ``all_dests`` (the BN-valued sentinel
        used by ``_step_core_pure`` for interior steps). ``index_add`` here
        does NO bounds masking, so a sentinel index ``== BN`` raises
        "index ... is out of bounds". The Triton path handles sentinels by
        skipping them in the kernel. Use this backend only via
        ``lif_deposit(..., backend="puretorch")`` on a code path that
        feeds REAL destinations (no sentinels) — i.e. the new-window
        path that pads to ``2*BH`` with valid (non-sentinel) destinations.
    """
    BN, D_s = s_flat.shape
    M = all_msgs.shape[0]
    device = s_flat.device

    incoming = torch.zeros(BN, D_s, dtype=torch.float32, device=device)
    incoming = incoming.index_add(0, all_dests, all_msgs.float())

    counts = torch.zeros(BN, dtype=torch.float32, device=device)
    ones_m = torch.ones(M, dtype=torch.float32, device=device)
    counts = counts.index_add(0, all_dests, ones_m)
    touched_mask = (counts > 0).float().unsqueeze(-1)

    n_index = torch.arange(BN, device=device) % N
    alpha_row = alpha[n_index].float().unsqueeze(-1)

    s_flat_fp = s_flat.float()
    tanh_inc = torch.tanh(incoming)
    s_blend = alpha_row * s_flat_fp + (1.0 - alpha_row) * tanh_inc
    s_new = touched_mask * s_blend + (1.0 - touched_mask) * s_flat_fp

    return s_new.to(s_flat.dtype)


# =====================================================================
# Public dispatcher (matches the old triton_sparse_update API)
# =====================================================================


def lif_deposit(
    s_flat: torch.Tensor,
    all_msgs: torch.Tensor,
    all_dests: torch.Tensor,
    alpha: torch.Tensor,
    N: int,
    *,
    backend: str = "triton",
) -> torch.Tensor:
    """Public entry. backend in {'triton', 'puretorch'}.

    'triton' (default): sparse Triton kernel via :class:`LIFDepositFunction`.
        Per-step activations are O(B·H·D_s) — only the touched rows are saved
        for backward, not the full ``[B·N, D_s]`` substrate.
        ``all_dests`` MUST be padded to a static ``M_max`` with sentinel
        ``B*N`` for unused slots so the captured grid shape stays constant.
        Allocates per-call scratch internally (cudagraph captures these in
        the pool so they reuse addresses across replays).

    'puretorch': dense fp32 fallback. Saves O(B·N·D_s) per step in
        activations — at production scale this is ~150 MB/step, T_block of
        them blows past the GPU memory ceiling. Kept for CPU/no-Triton
        fallback and as the gradient-numerics oracle.
    """
    if backend == "triton":
        return LIFDepositFunction.apply(
            s_flat, all_msgs, all_dests, alpha, N,
        )
    if backend == "puretorch":
        return sparse_lif_update_puretorch(
            s_flat, all_msgs, all_dests, alpha, N,
        )
    raise ValueError(f"unknown backend: {backend!r}")


# Backward-compat alias for the old triton_sparse_update.sparse_lif_update API.
sparse_lif_update = lif_deposit
