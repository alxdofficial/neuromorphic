"""Triton kernels for memory graph per-token neuron dynamics.

Two kernels:
  1. memory_graph_routing_kernel — computes sigmoid routing weights inline
     from key × neighbor_messages (replaces Python-side _compute_routing_weights)
  2. memory_graph_step_kernel — single token step with dendritic tree gather

Together these eliminate the 200MB temporary [BS, N, K, D] tensor from the
Python-side routing computation. The step kernel is launched T_seg times
per segment (128 times at Tier A).

Dendritic tree: branch → tanh → group → tanh → soma at each token.
Python fallback in memory_graph.py remains the source of truth for correctness.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def memory_graph_routing_kernel(
    prev_msg_ptr,       # [BS, N, D] bf16
    key_ptr,            # [BS, N, D] bf16
    conn_idx_ptr,       # [N, K] int32
    conn_w_ptr,         # [BS, N, K] bf16 — output
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
):
    """Compute sigmoid routing weights from key × neighbor messages.

    Each connection independently gated — no normalization across connections.
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, D)

    key_val = tl.load(key_ptr + (b * N + n) * D + d).to(tl.float32)

    for k_idx in range(K):
        src = tl.load(conn_idx_ptr + n * K + k_idx)
        neighbor_msg = tl.load(prev_msg_ptr + (b * N + src) * D + d).to(tl.float32)
        dot = tl.sum(key_val * neighbor_msg)
        # Sigmoid: each connection independently gated [0, 1]
        w = 1.0 / (1.0 + tl.exp(-dot))
        tl.store(conn_w_ptr + (b * N + n) * K + k_idx, w)


@triton.jit
def memory_graph_step_kernel(
    # Persistent state (read+write in-place)
    h_ptr,              # [BS, N, D] bf16
    prev_msg_ptr,       # [BS, N, D] bf16

    # Sparse connectivity (read-only)
    conn_idx_ptr,       # [N, K] int32
    conn_w_ptr,         # [BS, N, K] bf16

    # Per-neuron effective params (modulated, computed before loop)
    decay_ptr,          # [BS, N] float32
    primitives_ptr,     # [BS, N, D] bf16

    # Per-neuron dendritic FC weights (optional, None-safe via USE_DENDRITE_FC)
    branch_w_ptr,       # [N, NB, BS_SIZE, D] bf16  (NB = n_branches)
    group_w_ptr,        # [N, NG, BPG, D] bf16

    # Per-token inputs/outputs
    cc_signals_ptr,     # [BS, T_seg, C, D] bf16
    output_ptr,         # [BS, T_seg, C, D] bf16

    # Scalar accumulators (f32, always active)
    recv_accum_ptr,     # [BS, N, D] float32
    msg_accum_ptr,      # [BS, N, D] float32
    msg_mag_accum_ptr,  # [BS, N] float32 — accumulated message norm

    # Optional: per-token activation trace (only for co-activation segments)
    act_trace_ptr,      # [BS, T_seg, N] float32 OR null (0)

    # Current token step
    t_step,

    # Dimensions
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    K: tl.constexpr, C: tl.constexpr, T_seg: tl.constexpr,

    # Dendritic tree structure
    BRANCH_SIZE: tl.constexpr,
    BRANCHES_PER_GROUP: tl.constexpr,
    N_GROUPS: tl.constexpr,

    # Whether to write act_trace
    WRITE_ACT_TRACE: tl.constexpr,
    # Whether to use per-neuron FC weights in dendritic tree
    USE_DENDRITE_FC: tl.constexpr,
):
    """One step of neuron dynamics with dendritic tree gather."""
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, D)

    decay_val = tl.load(decay_ptr + b * N + n).to(tl.float32)
    eff_decay = decay_val
    eff_omd = 1.0 - eff_decay

    prim = tl.load(primitives_ptr + (b * N + n) * D + d).to(tl.float32)
    h_val = tl.load(h_ptr + (b * N + n) * D + d).to(tl.float32)

    # --- Dendritic tree gather (with optional per-neuron FC weights) ---
    NB = N_GROUPS * BRANCHES_PER_GROUP  # total branches
    soma = tl.zeros([D], dtype=tl.float32)
    conn_base = 0
    br_global = 0
    for g in range(N_GROUPS):
        group_acc = tl.zeros([D], dtype=tl.float32)
        for br in range(BRANCHES_PER_GROUP):
            branch_acc = tl.zeros([D], dtype=tl.float32)
            for k in range(BRANCH_SIZE):
                idx = conn_base + k
                src = tl.load(conn_idx_ptr + n * K + idx)
                w = tl.load(conn_w_ptr + (b * N + n) * K + idx).to(tl.float32)
                val = tl.load(prev_msg_ptr + (b * N + src) * D + d).to(tl.float32)
                if USE_DENDRITE_FC:
                    # branch_w: [N, NB, BRANCH_SIZE, D]
                    bw = tl.load(branch_w_ptr + ((n * NB + br_global) * BRANCH_SIZE + k) * D + d).to(tl.float32)
                    branch_acc += w * val * bw
                else:
                    branch_acc += w * val
            tanh_branch = libdevice.tanh(branch_acc)
            if USE_DENDRITE_FC:
                # group_w: [N, NG, BPG, D]
                gw = tl.load(group_w_ptr + ((n * N_GROUPS + g) * BRANCHES_PER_GROUP + br) * D + d).to(tl.float32)
                group_acc += tanh_branch * gw
            else:
                group_acc += tanh_branch
            conn_base += BRANCH_SIZE
            br_global += 1
        soma += libdevice.tanh(group_acc)
    received = soma * (1.0 / N_GROUPS)

    # --- CC signal injection ---
    if n < C:
        cc_offset = b * T_seg * C * D + t_step * C * D + n * D + d
        cc = tl.load(cc_signals_ptr + cc_offset).to(tl.float32)
        received += cc

    # --- Temporal integration ---
    h_new = eff_decay * h_val + eff_omd * received

    # --- Compute outgoing message ---
    msg = libdevice.tanh(h_new * prim)

    # --- Store results ---
    tl.store(h_ptr + (b * N + n) * D + d, h_new.to(tl.bfloat16))
    tl.store(prev_msg_ptr + (b * N + n) * D + d, msg.to(tl.bfloat16))

    # --- Message norm (always accumulated, optionally written to trace) ---
    msg_norm = tl.sqrt(tl.sum(msg * msg))

    # Accumulate into msg_mag_accum (scalar per neuron)
    mag_offset = b * N + n
    old_mag = tl.load(msg_mag_accum_ptr + mag_offset)
    tl.store(msg_mag_accum_ptr + mag_offset, old_mag + msg_norm)

    # Optionally write per-token trace (only for co-activation segments)
    if WRITE_ACT_TRACE:
        tl.store(act_trace_ptr + b * T_seg * N + t_step * N + n, msg_norm)

    # --- Accumulate received and msg for mean_input / mean_output ---
    accum_offset = (b * N + n) * D + d
    old_recv = tl.load(recv_accum_ptr + accum_offset)
    tl.store(recv_accum_ptr + accum_offset, old_recv + received)
    old_msg = tl.load(msg_accum_ptr + accum_offset)
    tl.store(msg_accum_ptr + accum_offset, old_msg + msg)

    # --- Write read-neuron output (neurons C..2C-1 → output slot 0..C-1) ---
    if n >= C and n < 2 * C:
        read_idx = n - C  # map neuron C+i to output slot i
        out_offset = b * T_seg * C * D + t_step * C * D + read_idx * D + d
        tl.store(output_ptr + out_offset, msg.to(tl.bfloat16))
