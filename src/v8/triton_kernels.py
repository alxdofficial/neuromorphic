"""Triton kernels for v9-backprop memory graph neuron dynamics.

Main kernel: fused_dendritic_gather — fuses gather + weight + dendritic tree
into a single kernel, eliminating the [BS, N, K, D] intermediate tensor.

Forward: prev_msg, conn_indices, w_conn_sig → received [BS, N, D]
Backward: grad_received → grad_w_conn_sig, grad_branch_w, grad_group_w, grad_prev_msg

The per-neuron MLPs (state MLP, message MLP) are left as PyTorch bmm calls
since they're already efficient batched matmuls.

Python fallback in memory_graph.py remains the source of truth for correctness.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def fused_dendritic_gather_fwd_kernel(
    prev_msg_ptr,       # [BS, N, D] f32
    conn_idx_ptr,       # [N, K] int64
    w_conn_sig_ptr,     # [BS, N, K] f32
    branch_w_ptr,       # [N, NB, BRANCH_SIZE, D] f32
    group_w_ptr,        # [N, NG, BPG, D] f32
    received_ptr,       # [BS, N, D] f32 — output
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    BRANCH_SIZE: tl.constexpr,
    BRANCHES_PER_GROUP: tl.constexpr,
    N_GROUPS: tl.constexpr,
    USE_DENDRITE: tl.constexpr,
):
    """Fused gather + weight + dendritic tree → received [D].

    Eliminates [BS, N, K, D] intermediate. One program per (batch, neuron).
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, D)

    NB = N_GROUPS * BRANCHES_PER_GROUP  # total branches
    n_tree = NB * BRANCH_SIZE

    if USE_DENDRITE:
        # Dendritic tree: branch → tanh → group → tanh → mean
        conn_base = 0
        br_global = 0
        soma = tl.zeros([D], dtype=tl.float32)

        for g in range(N_GROUPS):
            group_acc = tl.zeros([D], dtype=tl.float32)
            for br in range(BRANCHES_PER_GROUP):
                branch_acc = tl.zeros([D], dtype=tl.float32)
                for k in range(BRANCH_SIZE):
                    idx = conn_base + k
                    src = tl.load(conn_idx_ptr + n * K + idx)
                    w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                    val = tl.load(prev_msg_ptr + (b * N + src) * D + d)
                    bw = tl.load(
                        branch_w_ptr +
                        ((n * NB + br_global) * BRANCH_SIZE + k) * D + d)
                    branch_acc += w * val * bw
                conn_base += BRANCH_SIZE
                tanh_branch = libdevice.tanh(branch_acc)
                gw = tl.load(
                    group_w_ptr +
                    ((n * N_GROUPS + g) * BRANCHES_PER_GROUP + br) * D + d)
                group_acc += tanh_branch * gw
                br_global += 1
            soma += libdevice.tanh(group_acc)

        received = soma * (1.0 / N_GROUPS)

        # Handle leftover connections (if n_tree < K)
        if n_tree < K:
            leftover = tl.zeros([D], dtype=tl.float32)
            for k in range(K - n_tree):
                idx = n_tree + k
                src = tl.load(conn_idx_ptr + n * K + idx)
                w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                val = tl.load(prev_msg_ptr + (b * N + src) * D + d)
                leftover += w * val
            tree_frac = n_tree / K
            received = tree_frac * received + (1.0 - tree_frac) * leftover
    else:
        # Simple weighted sum (no dendritic tree)
        received = tl.zeros([D], dtype=tl.float32)
        for k in range(K):
            src = tl.load(conn_idx_ptr + n * K + k)
            w = tl.load(w_conn_sig_ptr + (b * N + n) * K + k)
            val = tl.load(prev_msg_ptr + (b * N + src) * D + d)
            received += w * val

    tl.store(received_ptr + (b * N + n) * D + d, received)


@triton.jit
def fused_dendritic_gather_bwd_kernel(
    # Forward inputs (saved for backward)
    prev_msg_ptr,       # [BS, N, D] f32
    conn_idx_ptr,       # [N, K] int64
    w_conn_sig_ptr,     # [BS, N, K] f32
    branch_w_ptr,       # [N, NB, BRANCH_SIZE, D] f32
    group_w_ptr,        # [N, NG, BPG, D] f32
    # Grad output
    grad_received_ptr,  # [BS, N, D] f32
    # Grad inputs (outputs of this kernel)
    grad_w_conn_sig_ptr,  # [BS, N, K] f32
    grad_branch_w_ptr,    # [N, NB, BRANCH_SIZE, D] f32 — atomically accumulated
    grad_group_w_ptr,     # [N, NG, BPG, D] f32 — atomically accumulated
    grad_prev_msg_ptr,    # [BS, N, D] f32 — atomically accumulated
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    BRANCH_SIZE: tl.constexpr,
    BRANCHES_PER_GROUP: tl.constexpr,
    N_GROUPS: tl.constexpr,
    USE_DENDRITE: tl.constexpr,
):
    """Backward of fused dendritic gather.

    Computes gradients w.r.t. w_conn_sig, branch_w, group_w, prev_msg.
    Branch/group/prev_msg grads use atomic adds since multiple programs
    may write to the same location.
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, D)

    grad_recv = tl.load(grad_received_ptr + (b * N + n) * D + d)

    NB = N_GROUPS * BRANCHES_PER_GROUP
    n_tree = NB * BRANCH_SIZE

    if USE_DENDRITE:
        # Forward recomputation needed for backward through tanh
        # Scale factor for soma mean
        scale = 1.0 / N_GROUPS

        # When n_tree < K, forward is: tree_frac * tree_out + (1-tree_frac) * leftover
        # So grad into tree portion must be scaled by tree_frac
        # Always compute in f32 for consistency
        grad_recv_f32 = grad_recv.to(tl.float32)
        if n_tree < K:
            tree_frac = n_tree / K
            grad_tree = grad_recv_f32 * tree_frac
        else:
            grad_tree = grad_recv_f32

        conn_base = 0
        br_global = 0

        # We need to recompute forward intermediates for the backward pass
        # Store branch outputs and group inputs
        for g in range(N_GROUPS):
            # Recompute forward for this group
            # We need group_acc to compute tanh backward
            group_acc = tl.zeros([D], dtype=tl.float32)
            branch_tanh_vals = tl.zeros([D], dtype=tl.float32)

            # Save per-branch tanh outputs and branch_acc for backward
            # Since we can't store variable-length arrays, process branch by branch
            conn_base_g = conn_base

            # First pass: compute group_acc (forward recomputation)
            br_g = br_global
            for br in range(BRANCHES_PER_GROUP):
                branch_acc = tl.zeros([D], dtype=tl.float32)
                cb = conn_base_g
                for k in range(BRANCH_SIZE):
                    idx = cb + k
                    src = tl.load(conn_idx_ptr + n * K + idx)
                    w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                    val = tl.load(prev_msg_ptr + (b * N + src) * D + d)
                    bw = tl.load(
                        branch_w_ptr +
                        ((n * NB + br_g) * BRANCH_SIZE + k) * D + d)
                    branch_acc += w * val * bw
                conn_base_g += BRANCH_SIZE
                tanh_branch = libdevice.tanh(branch_acc)
                gw = tl.load(
                    group_w_ptr +
                    ((n * N_GROUPS + g) * BRANCHES_PER_GROUP + br) * D + d)
                group_acc += tanh_branch * gw
                br_g += 1

            # Backward through tanh(group_acc) → soma contribution
            tanh_group = libdevice.tanh(group_acc)
            # d_soma/d_tanh_group = scale (from mean)
            # d_tanh_group/d_group_acc = 1 - tanh_group^2
            grad_group_acc = grad_tree * scale * (1.0 - tanh_group * tanh_group)

            # Second pass: backward through branches
            conn_base_g2 = conn_base
            br_g2 = br_global
            for br in range(BRANCHES_PER_GROUP):
                # Recompute branch_acc and tanh_branch
                branch_acc = tl.zeros([D], dtype=tl.float32)
                cb = conn_base_g2
                for k in range(BRANCH_SIZE):
                    idx = cb + k
                    src = tl.load(conn_idx_ptr + n * K + idx)
                    w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                    val = tl.load(prev_msg_ptr + (b * N + src) * D + d)
                    bw = tl.load(
                        branch_w_ptr +
                        ((n * NB + br_g2) * BRANCH_SIZE + k) * D + d)
                    branch_acc += w * val * bw

                tanh_branch = libdevice.tanh(branch_acc)
                gw = tl.load(
                    group_w_ptr +
                    ((n * N_GROUPS + g) * BRANCHES_PER_GROUP + br) * D + d)

                # grad_group_w: d_loss/d_gw = grad_group_acc * tanh_branch
                grad_gw = grad_group_acc * tanh_branch
                gw_offset = ((n * N_GROUPS + g) * BRANCHES_PER_GROUP + br) * D + d
                tl.atomic_add(grad_group_w_ptr + gw_offset, grad_gw)

                # grad through tanh_branch
                # d_group_acc/d_tanh_branch = gw
                # d_tanh_branch/d_branch_acc = 1 - tanh_branch^2
                grad_branch_acc = (
                    grad_group_acc * gw * (1.0 - tanh_branch * tanh_branch))

                # Third pass: backward through branch synapses
                cb = conn_base_g2
                for k in range(BRANCH_SIZE):
                    idx = cb + k
                    src = tl.load(conn_idx_ptr + n * K + idx)
                    w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                    val = tl.load(prev_msg_ptr + (b * N + src) * D + d)
                    bw = tl.load(
                        branch_w_ptr +
                        ((n * NB + br_g2) * BRANCH_SIZE + k) * D + d)

                    # grad_branch_w: d_loss/d_bw = grad_branch_acc * w * val
                    grad_bw = grad_branch_acc * w * val
                    bw_offset = (
                        (n * NB + br_g2) * BRANCH_SIZE + k) * D + d
                    tl.atomic_add(grad_branch_w_ptr + bw_offset, grad_bw)

                    # grad_w_conn_sig: scalar grad per connection
                    # d_loss/d_w = sum_d(grad_branch_acc * val * bw)
                    grad_w_k = tl.sum(grad_branch_acc * val * bw)
                    tl.atomic_add(
                        grad_w_conn_sig_ptr + (b * N + n) * K + idx, grad_w_k)

                    # grad_prev_msg: d_loss/d_val = grad_branch_acc * w * bw
                    grad_val = grad_branch_acc * w * bw
                    tl.atomic_add(
                        grad_prev_msg_ptr + (b * N + src) * D + d, grad_val)

                conn_base_g2 += BRANCH_SIZE
                br_g2 += 1

            conn_base = conn_base_g
            br_global = br_g

        # Handle leftover connections
        if n_tree < K:
            tree_frac = n_tree / K
            leftover_frac = 1.0 - tree_frac
            # grad flows to leftover: grad_recv * leftover_frac
            grad_leftover = grad_recv * leftover_frac
            for k in range(K - n_tree):
                idx = n_tree + k
                src = tl.load(conn_idx_ptr + n * K + idx)
                w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                val = tl.load(prev_msg_ptr + (b * N + src) * D + d)

                grad_w_k = tl.sum(grad_leftover * val)
                tl.atomic_add(
                    grad_w_conn_sig_ptr + (b * N + n) * K + idx, grad_w_k)
                grad_val = grad_leftover * w
                tl.atomic_add(
                    grad_prev_msg_ptr + (b * N + src) * D + d, grad_val)
    else:
        # Simple weighted sum backward
        for k in range(K):
            src = tl.load(conn_idx_ptr + n * K + k)
            w = tl.load(w_conn_sig_ptr + (b * N + n) * K + k)
            val = tl.load(prev_msg_ptr + (b * N + src) * D + d)

            grad_w_k = tl.sum(grad_recv * val)
            tl.atomic_add(
                grad_w_conn_sig_ptr + (b * N + n) * K + k, grad_w_k)
            grad_val = grad_recv * w
            tl.atomic_add(
                grad_prev_msg_ptr + (b * N + src) * D + d, grad_val)


class FusedDendriticGather(torch.autograd.Function):
    """Autograd wrapper for fused dendritic gather Triton kernels."""

    @staticmethod
    def forward(ctx, prev_msg, conn_indices, w_conn_sig,
                branch_w, group_w,
                BS, N, D, K,
                branch_size, branches_per_group, n_groups,
                use_dendrite):
        """
        Args:
            prev_msg: [BS, N, D] f32
            conn_indices: [N, K] int64
            w_conn_sig: [BS, N, K] f32
            branch_w: [N, NB, branch_size, D] f32 or None
            group_w: [N, NG, BPG, D] f32 or None

        Returns:
            received: [BS, N, D] f32
        """
        received = torch.empty(BS, N, D, device=prev_msg.device,
                               dtype=prev_msg.dtype)

        # Handle None branch/group weights
        if branch_w is None:
            branch_w = torch.empty(0, device=prev_msg.device)
        if group_w is None:
            group_w = torch.empty(0, device=prev_msg.device)

        grid = (BS, N)
        fused_dendritic_gather_fwd_kernel[grid](
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w, received,
            BS=BS, N=N, D=D, K=K,
            BRANCH_SIZE=branch_size,
            BRANCHES_PER_GROUP=branches_per_group,
            N_GROUPS=n_groups,
            USE_DENDRITE=use_dendrite,
        )

        ctx.save_for_backward(prev_msg, conn_indices, w_conn_sig,
                              branch_w, group_w)
        ctx.BS = BS
        ctx.N = N
        ctx.D = D
        ctx.K = K
        ctx.branch_size = branch_size
        ctx.branches_per_group = branches_per_group
        ctx.n_groups = n_groups
        ctx.use_dendrite = use_dendrite

        return received

    @staticmethod
    def backward(ctx, grad_received):
        (prev_msg, conn_indices, w_conn_sig,
         branch_w, group_w) = ctx.saved_tensors

        BS, N, D, K = ctx.BS, ctx.N, ctx.D, ctx.K
        out_dtype = prev_msg.dtype

        # Grad buffers in f32 for atomic adds, cast to input dtype at end
        grad_w_conn_sig = torch.zeros(BS, N, K, device=grad_received.device,
                                      dtype=torch.float32)
        grad_prev_msg = torch.zeros(BS, N, D, device=grad_received.device,
                                    dtype=torch.float32)

        if ctx.use_dendrite:
            grad_branch_w = torch.zeros(branch_w.shape, device=branch_w.device,
                                        dtype=torch.float32)
            grad_group_w = torch.zeros(group_w.shape, device=group_w.device,
                                       dtype=torch.float32)
        else:
            grad_branch_w = torch.empty(0, device=grad_received.device)
            grad_group_w = torch.empty(0, device=grad_received.device)

        grad_received = grad_received.contiguous()

        grid = (BS, N)
        fused_dendritic_gather_bwd_kernel[grid](
            prev_msg, conn_indices, w_conn_sig,
            branch_w, group_w,
            grad_received,
            grad_w_conn_sig, grad_branch_w, grad_group_w, grad_prev_msg,
            BS=BS, N=N, D=D, K=K,
            BRANCH_SIZE=ctx.branch_size,
            BRANCHES_PER_GROUP=ctx.branches_per_group,
            N_GROUPS=ctx.n_groups,
            USE_DENDRITE=ctx.use_dendrite,
        )

        # Cast grads to match each input's dtype — prev_msg/w_conn_sig may be
        # bf16 runtime tensors, but branch_w/group_w are f32 Parameters.
        # Returning f32 grads for f32 params avoids needless bf16 quantization.
        grad_branch_out = grad_branch_w.to(branch_w.dtype) if ctx.use_dendrite else None
        grad_group_out = grad_group_w.to(group_w.dtype) if ctx.use_dendrite else None

        return (grad_prev_msg.to(prev_msg.dtype), None, grad_w_conn_sig.to(w_conn_sig.dtype),
                grad_branch_out, grad_group_out,
                None, None, None, None, None, None, None, None)


def fused_dendritic_gather(prev_msg, conn_indices, w_conn_sig,
                           branch_w, group_w,
                           branch_size, branches_per_group, n_groups,
                           use_dendrite):
    """Python-callable wrapper for fused dendritic gather.

    Args:
        prev_msg: [BS, N, D] — previous messages (f32)
        conn_indices: [N, K] — sparse connectivity (int64)
        w_conn_sig: [BS, N, K] — sigmoid(w_conn) (f32)
        branch_w: [N, NB, branch_size, D] or None
        group_w: [N, NG, BPG, D] or None
        branch_size, branches_per_group, n_groups: dendritic tree structure
        use_dendrite: whether to use dendritic tree

    Returns:
        received: [BS, N, D] — gathered + weighted + tree-reduced signals
    """
    BS, N, D = prev_msg.shape
    K = conn_indices.shape[1]

    return FusedDendriticGather.apply(
        prev_msg.contiguous(),
        conn_indices.contiguous(),
        w_conn_sig.contiguous(),
        branch_w.contiguous() if branch_w is not None else None,
        group_w.contiguous() if group_w is not None else None,
        BS, N, D, K,
        branch_size, branches_per_group, n_groups,
        use_dendrite,
    )


# ============================================================================
# Fused neuron step kernel — ALL ops in one kernel per step
# ============================================================================

@triton.jit
def fused_neuron_step_kernel(
    # State (read from prev, write to next)
    h_in_ptr,           # [BS, N, D] f32 — h before this step
    prev_msg_in_ptr,    # [BS, N, D] f32 — messages from previous step
    h_out_ptr,          # [BS, N, D] f32 — h after this step
    msg_out_ptr,        # [BS, N, D] f32 — messages after this step

    # Connectivity
    conn_idx_ptr,       # [N, K] int64

    # Per-segment constants (from modulator, on compute graph)
    w_conn_sig_ptr,     # [BS, N, K] f32 — sigmoid(w_conn)
    decay_ptr,          # [BS, N] f32 — sigmoid(decay_logit)
    primitives_ptr,     # [BS, N, D] f32
    neuron_id_ptr,      # [N, D] f32

    # Per-step inject signal
    inject_ptr,         # [BS, N, D] f32 — inject signal for this step

    # Dendritic tree weights
    branch_w_ptr,       # [N, NB, BS_SIZE, D] f32
    group_w_ptr,        # [N, NG, BPG, D] f32

    # Per-neuron MLP weights — state MLP
    state_w1_ptr,       # [N, STATE_IN, H_STATE] f32
    state_b1_ptr,       # [N, H_STATE] f32
    state_w2_ptr,       # [N, H_STATE, D] f32
    state_b2_ptr,       # [N, D] f32

    # Per-neuron MLP weights — message MLP
    msg_w1_ptr,         # [N, 2*D, H_MSG] f32
    msg_b1_ptr,         # [N, H_MSG] f32
    msg_w2_ptr,         # [N, H_MSG, D] f32
    msg_b2_ptr,         # [N, D] f32

    # Hebbian accumulator (atomically added to)
    hebbian_ptr,        # [BS, N, K] f32

    # Dimensions
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    STATE_IN: tl.constexpr,  # 2*D + 1
    H_STATE: tl.constexpr,
    H_MSG: tl.constexpr,

    # Dendritic tree structure
    BRANCH_SIZE: tl.constexpr,
    BRANCHES_PER_GROUP: tl.constexpr,
    N_GROUPS: tl.constexpr,
    USE_DENDRITE: tl.constexpr,
):
    """Fully fused single neuron step: gather→dendritic→inject→state_MLP→msg_MLP→id→hebbian.

    One program per (batch, neuron). Eliminates all intermediate tensors and
    reduces kernel launches from ~16 to 1 per step.
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, D)

    bn_offset = (b * N + n) * D

    # Load per-neuron constants (reused across the whole step)
    h_val = tl.load(h_in_ptr + bn_offset + d)
    prim = tl.load(primitives_ptr + bn_offset + d)
    nid = tl.load(neuron_id_ptr + n * D + d)
    decay_val = tl.load(decay_ptr + b * N + n)
    inject_val = tl.load(inject_ptr + bn_offset + d)

    # ---- 1-3. Gather + weight + dendritic tree ----
    NB = N_GROUPS * BRANCHES_PER_GROUP
    n_tree = NB * BRANCH_SIZE

    if USE_DENDRITE:
        conn_base = 0
        br_global = 0
        soma = tl.zeros([D], dtype=tl.float32)

        for g in range(N_GROUPS):
            group_acc = tl.zeros([D], dtype=tl.float32)
            for br in range(BRANCHES_PER_GROUP):
                branch_acc = tl.zeros([D], dtype=tl.float32)
                for k in range(BRANCH_SIZE):
                    idx = conn_base + k
                    src = tl.load(conn_idx_ptr + n * K + idx)
                    w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                    val = tl.load(prev_msg_in_ptr + (b * N + src) * D + d)
                    bw = tl.load(
                        branch_w_ptr +
                        ((n * NB + br_global) * BRANCH_SIZE + k) * D + d)
                    branch_acc += w * val * bw
                conn_base += BRANCH_SIZE
                tanh_branch = libdevice.tanh(branch_acc)
                gw = tl.load(
                    group_w_ptr +
                    ((n * N_GROUPS + g) * BRANCHES_PER_GROUP + br) * D + d)
                group_acc += tanh_branch * gw
                br_global += 1
            soma += libdevice.tanh(group_acc)

        received = soma * (1.0 / N_GROUPS)

        if n_tree < K:
            leftover = tl.zeros([D], dtype=tl.float32)
            for k in range(K - n_tree):
                idx = n_tree + k
                src = tl.load(conn_idx_ptr + n * K + idx)
                w = tl.load(w_conn_sig_ptr + (b * N + n) * K + idx)
                val = tl.load(prev_msg_in_ptr + (b * N + src) * D + d)
                leftover += w * val
            tree_frac = n_tree / K
            received = tree_frac * received + (1.0 - tree_frac) * leftover
    else:
        received = tl.zeros([D], dtype=tl.float32)
        for k in range(K):
            src = tl.load(conn_idx_ptr + n * K + k)
            w = tl.load(w_conn_sig_ptr + (b * N + n) * K + k)
            val = tl.load(prev_msg_in_ptr + (b * N + src) * D + d)
            received += w * val

    # ---- 4. Inject ----
    input_vec = received + inject_val

    # ---- 5. State MLP: cat(input_vec[D], h[D], decay[1]) → [H_STATE] → [D] ----
    # Fused layer 1 + layer 2: iterate over hidden units, accumulate output
    h_new_acc = tl.load(state_b2_ptr + n * D + d)  # start with bias2

    for hid in range(H_STATE):
        # Layer 1: dot product of [input_vec, h, decay] with W1[n, hid, :]
        # W1 layout: [N, H_STATE, STATE_IN] (transposed — row is contiguous)
        w1_row = (n * H_STATE + hid) * STATE_IN
        # input_vec part: first D values (contiguous)
        w1_iv = tl.load(state_w1_ptr + w1_row + d)
        # h part: next D values (contiguous)
        w1_h = tl.load(state_w1_ptr + w1_row + D + d)
        # decay part: last value
        w1_dc = tl.load(state_w1_ptr + w1_row + 2 * D)

        dot = tl.sum(input_vec * w1_iv) + tl.sum(h_val * w1_h) + decay_val * w1_dc
        bias1 = tl.load(state_b1_ptr + n * H_STATE + hid)
        hidden_val = libdevice.tanh(dot + bias1)

        # Layer 2: W2[n, hid, :] is contiguous (layout [N, H, D])
        w2_row = tl.load(state_w2_ptr + (n * H_STATE + hid) * D + d)
        h_new_acc += hidden_val * w2_row

    h_new = libdevice.tanh(h_new_acc)

    # ---- 6. Message MLP: cat(h_new[D], prim[D]) → [H_MSG] → [D] ----
    msg_acc = tl.load(msg_b2_ptr + n * D + d)  # start with bias2

    for hid in range(H_MSG):
        # Layer 1: W1 layout [N, H_MSG, 2*D] (transposed — row contiguous)
        w1_row = (n * H_MSG + hid) * 2 * D
        w1_hn = tl.load(msg_w1_ptr + w1_row + d)
        w1_pr = tl.load(msg_w1_ptr + w1_row + D + d)

        dot = tl.sum(h_new * w1_hn) + tl.sum(prim * w1_pr)
        bias1 = tl.load(msg_b1_ptr + n * H_MSG + hid)
        hidden_val = libdevice.tanh(dot + bias1)

        # Layer 2: W2[n, hid, :] contiguous (layout [N, H, D])
        w2_row = tl.load(msg_w2_ptr + (n * H_MSG + hid) * D + d)
        msg_acc += hidden_val * w2_row

    msg = libdevice.tanh(msg_acc)

    # ---- 7. Neuron ID ----
    msg = msg + nid

    # ---- 8. Store outputs ----
    tl.store(h_out_ptr + bn_offset + d, h_new)
    tl.store(msg_out_ptr + bn_offset + d, msg)

    # ---- 9. Hebbian: |msg| * sigmoid(w_conn) ----
    msg_norm = tl.sqrt(tl.sum(msg * msg))
    for k in range(K):
        w = tl.load(w_conn_sig_ptr + (b * N + n) * K + k)
        old_heb = tl.load(hebbian_ptr + (b * N + n) * K + k)
        tl.store(hebbian_ptr + (b * N + n) * K + k, old_heb + msg_norm * w)


def fused_neuron_step(h, prev_msg, inject_signal, conn_indices,
                      w_conn_sig, decay, primitives, neuron_id,
                      branch_w, group_w,
                      state_w1, state_b1, state_w2, state_b2,
                      msg_w1, msg_b1, msg_w2, msg_b2,
                      hebbian_accum,
                      branch_size, branches_per_group, n_groups,
                      use_dendrite):
    """Run one fused neuron step via Triton.

    All inputs must be contiguous f32 CUDA tensors.

    Args:
        h: [BS, N, D] — hidden state before step
        prev_msg: [BS, N, D] — messages from previous step
        inject_signal: [BS, N, D] — CC inject for this step
        conn_indices: [N, K] — sparse connectivity
        w_conn_sig: [BS, N, K] — sigmoid(w_conn)
        decay: [BS, N] — sigmoid(decay_logit)
        primitives: [BS, N, D]
        neuron_id: [N, D]
        branch_w, group_w: dendritic weights or None
        state_w1/b1/w2/b2: state MLP weights
        msg_w1/b1/w2/b2: message MLP weights
        hebbian_accum: [BS, N, K] — accumulated (modified in-place)
        branch_size, branches_per_group, n_groups: dendritic config
        use_dendrite: bool

    Returns:
        h_new: [BS, N, D]
        msg_new: [BS, N, D]
    """
    BS, N, D = h.shape
    K = conn_indices.shape[1]
    STATE_IN = 2 * D + 1

    h_out = torch.empty_like(h)
    msg_out = torch.empty_like(h)

    if branch_w is None:
        branch_w = torch.empty(0, device=h.device)
    if group_w is None:
        group_w = torch.empty(0, device=h.device)

    grid = (BS, N)
    fused_neuron_step_kernel[grid](
        h, prev_msg, h_out, msg_out,
        conn_indices,
        w_conn_sig, decay, primitives, neuron_id,
        inject_signal,
        branch_w, group_w,
        state_w1, state_b1, state_w2, state_b2,
        msg_w1, msg_b1, msg_w2, msg_b2,
        hebbian_accum,
        BS=BS, N=N, D=D, K=K,
        STATE_IN=STATE_IN,
        H_STATE=state_b1.shape[1],
        H_MSG=msg_b1.shape[1],
        BRANCH_SIZE=branch_size,
        BRANCHES_PER_GROUP=branches_per_group,
        N_GROUPS=n_groups,
        USE_DENDRITE=use_dendrite,
    )

    return h_out, msg_out
