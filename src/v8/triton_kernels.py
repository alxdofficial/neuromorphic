"""Triton forward + backward kernels for v9.1 neuron recurrence.

Each program handles one (batch, neuron) and loops over T_seg steps.
Block sizes are padded to powers of 2 with masking for irregular dims.

The inject broadcast and readout are computed outside the kernel.
The kernel handles: gather → weighted receive → add inject → MLP → integrate.

Backward kernel reverses the loop, recomputes intermediates from saved h_all/msg_all,
and accumulates parameter gradients via atomic adds.
"""

import triton
import triton.language as tl


def _next_pow2(n):
    """Next power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# ====================================================================
# Helper: extract scalar from vector (used for small matmuls)
# ====================================================================
# Pattern: val = tl.sum(vec * (range == idx).to(tl.float32))
# This is O(BLOCK) per extraction but BLOCK is small (16-64).


# ====================================================================
# Forward kernel
# ====================================================================

@triton.jit
def neuron_fwd_kernel(
    h_ptr, prev_msg_ptr,
    inject_bc_ptr,
    w_conn_ptr, decay_logit_ptr,
    W1_ptr, b1_ptr, W_msg_ptr, b_msg_ptr, W_mod_ptr, b_mod_ptr,
    trace_h_ptr, trace_recv_ptr,
    conn_idx_ptr,
    h_all_ptr, msg_all_ptr,
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    K: tl.constexpr, H: tl.constexpr, T_SEG: tl.constexpr,
    MLP_IN: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    h_range = tl.arange(0, BLOCK_H)
    h_mask = h_range < H

    h_base = (pid_b * N + pid_n) * D

    h_vec = tl.load(h_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    trace_h_vec = tl.load(trace_h_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    trace_recv_vec = tl.load(trace_recv_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    decay_logit_val = tl.load(decay_logit_ptr + pid_n).to(tl.float32)

    for t in range(T_SEG):
        # --- Gather + weighted sum ---
        received = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(K):
            nb_idx = tl.load(conn_idx_ptr + pid_n * K + k)
            nb_off = pid_b * N * D + nb_idx * D
            nb_msg = tl.load(prev_msg_ptr + nb_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
            rk_raw = tl.load(w_conn_ptr + pid_n * K + k).to(tl.float32)
            rk = 1.0 / (1.0 + tl.exp(-rk_raw))
            received += rk * nb_msg

        # --- Add inject ---
        inject_off = pid_b * T_SEG * N * D + t * N * D + pid_n * D
        inject_val = tl.load(inject_bc_ptr + inject_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
        received += inject_val

        # --- MLP: hidden = tanh(W1 @ [h, recv, trace_h, trace_recv] + b1) ---
        hidden = tl.load(b1_ptr + pid_n * H + h_range, mask=h_mask, other=0.0).to(tl.float32)
        parts = [h_vec, received, trace_h_vec, trace_recv_vec]
        for part_idx in tl.static_range(4):
            for d in tl.static_range(BLOCK_D):
                if d < D:
                    scalar_mask = (d_range == d).to(tl.float32)
                    val = tl.sum(parts[part_idx] * scalar_mask)
                    w_off = pid_n * MLP_IN * H + (part_idx * D + d) * H
                    w_row = tl.load(W1_ptr + w_off + h_range, mask=h_mask, other=0.0).to(tl.float32)
                    hidden += val * w_row

        hidden = tl.extra.cuda.libdevice.tanh(hidden)

        # --- Message: msg = tanh(W_msg @ hidden + b_msg) ---
        msg = tl.load(b_msg_ptr + pid_n * D + d_range, mask=d_mask, other=0.0).to(tl.float32)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                w_off = pid_n * H * D + j * D
                w_col = tl.load(W_msg_ptr + w_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
                msg += hj * w_col
        msg = tl.extra.cuda.libdevice.tanh(msg)

        # --- Modulator (only decay_mod, index 0) ---
        mod_0 = tl.load(b_mod_ptr + pid_n * 3 + 0).to(tl.float32)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                w_mod_val = tl.load(W_mod_ptr + pid_n * H * 3 + j * 3 + 0).to(tl.float32)
                mod_0 += hj * w_mod_val

        # --- Integrate ---
        eff_decay = 1.0 / (1.0 + tl.exp(-(decay_logit_val + mod_0)))
        h_vec = eff_decay * h_vec + (1.0 - eff_decay) * received

        # --- Save ---
        save_off = pid_b * T_SEG * N * D + t * N * D + pid_n * D
        tl.store(h_all_ptr + save_off + d_range, h_vec.to(tl.bfloat16), mask=d_mask)
        tl.store(msg_all_ptr + save_off + d_range, msg.to(tl.bfloat16), mask=d_mask)
        tl.store(prev_msg_ptr + h_base + d_range, msg.to(tl.bfloat16), mask=d_mask)

    tl.store(h_ptr + h_base + d_range, h_vec.to(tl.bfloat16), mask=d_mask)


# ====================================================================
# Backward kernel
# ====================================================================

@triton.jit
def neuron_bwd_kernel(
    # Saved from forward
    h_all_ptr, msg_all_ptr,
    inject_bc_ptr,
    # Params (read only)
    w_conn_ptr, decay_logit_ptr,
    W1_ptr, b1_ptr, W_msg_ptr, b_msg_ptr, W_mod_ptr, b_mod_ptr,
    trace_h_ptr, trace_recv_ptr,
    conn_idx_ptr,
    # Upstream gradient
    d_msg_all_ptr,      # [BS, T, N, D]
    # Gradient outputs (atomic add)
    d_w_conn_ptr,       # [N, K]
    d_decay_logit_ptr,  # [N]
    d_W1_ptr,           # [N, 4D, H]
    d_b1_ptr,           # [N, H]
    d_W_msg_ptr,        # [N, H, D]
    d_b_msg_ptr,        # [N, D]
    d_W_mod_ptr,        # [N, H, 3]
    d_b_mod_ptr,        # [N, 3]
    d_inject_bc_ptr,    # [BS, T, N, D]
    # Dims
    BS: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    K: tl.constexpr, H: tl.constexpr, T_SEG: tl.constexpr,
    MLP_IN: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    h_range = tl.arange(0, BLOCK_H)
    h_mask = h_range < H

    h_base = (pid_b * N + pid_n) * D

    # Load traces (constant across T)
    trace_h_vec = tl.load(trace_h_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    trace_recv_vec = tl.load(trace_recv_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    decay_logit_val = tl.load(decay_logit_ptr + pid_n).to(tl.float32)

    # Gradient carried backward through h
    d_h = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_decay_logit_acc = 0.0

    for t_rev in range(T_SEG):
        t = T_SEG - 1 - t_rev
        save_off = pid_b * T_SEG * N * D + t * N * D + pid_n * D

        # Load saved h[t], msg[t]
        h_t = tl.load(h_all_ptr + save_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
        msg_t = tl.load(msg_all_ptr + save_off + d_range, mask=d_mask, other=0.0).to(tl.float32)

        # Load h[t-1] (h before integration at step t)
        if t > 0:
            prev_off = pid_b * T_SEG * N * D + (t - 1) * N * D + pid_n * D
            h_prev = tl.load(h_all_ptr + prev_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
        else:
            h_prev = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Load upstream d_msg[t]
        d_msg_t = tl.load(d_msg_all_ptr + save_off + d_range, mask=d_mask, other=0.0).to(tl.float32)

        # ================================================================
        # Recompute forward intermediates from saved states
        # ================================================================

        # Recompute received: gather neighbor msgs from t-1
        received = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(K):
            nb_idx = tl.load(conn_idx_ptr + pid_n * K + k)
            if t > 0:
                nb_off = pid_b * T_SEG * N * D + (t - 1) * N * D + nb_idx * D
                nb_msg = tl.load(msg_all_ptr + nb_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
            else:
                nb_msg = tl.zeros([BLOCK_D], dtype=tl.float32)
            rk_raw = tl.load(w_conn_ptr + pid_n * K + k).to(tl.float32)
            rk = 1.0 / (1.0 + tl.exp(-rk_raw))
            received += rk * nb_msg

        inject_val = tl.load(inject_bc_ptr + save_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
        received += inject_val

        # Recompute hidden = tanh(W1 @ [h_prev, received, trace_h, trace_recv] + b1)
        hidden_pre = tl.load(b1_ptr + pid_n * H + h_range, mask=h_mask, other=0.0).to(tl.float32)
        mlp_parts = [h_prev, received, trace_h_vec, trace_recv_vec]
        for part_idx in tl.static_range(4):
            for d in tl.static_range(BLOCK_D):
                if d < D:
                    s_mask = (d_range == d).to(tl.float32)
                    val = tl.sum(mlp_parts[part_idx] * s_mask)
                    w_off = pid_n * MLP_IN * H + (part_idx * D + d) * H
                    w_row = tl.load(W1_ptr + w_off + h_range, mask=h_mask, other=0.0).to(tl.float32)
                    hidden_pre += val * w_row
        hidden = tl.extra.cuda.libdevice.tanh(hidden_pre)

        # Recompute mod_0 for eff_decay
        mod_0 = tl.load(b_mod_ptr + pid_n * 3 + 0).to(tl.float32)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                mod_0 += hj * tl.load(W_mod_ptr + pid_n * H * 3 + j * 3 + 0).to(tl.float32)
        eff_decay = 1.0 / (1.0 + tl.exp(-(decay_logit_val + mod_0)))

        # ================================================================
        # Backward: msg = tanh(W_msg @ hidden + b_msg)
        # ================================================================
        d_pre_msg = d_msg_t * (1.0 - msg_t * msg_t)  # [BLOCK_D]

        # d_b_msg += d_pre_msg (atomic across batch)
        tl.atomic_add(d_b_msg_ptr + pid_n * D + d_range, d_pre_msg, mask=d_mask)

        # d_W_msg[n, j, :] += hidden[j] * d_pre_msg[:], d_hidden[j] += W_msg[j,:] . d_pre_msg
        d_hidden = tl.zeros([BLOCK_H], dtype=tl.float32)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                w_off = pid_n * H * D + j * D
                w_col = tl.load(W_msg_ptr + w_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
                # d_W_msg
                tl.atomic_add(d_W_msg_ptr + w_off + d_range, hj * d_pre_msg, mask=d_mask)
                # d_hidden
                d_hj = tl.sum(w_col * d_pre_msg)
                d_hidden += d_hj * hj_mask

        # ================================================================
        # Backward: h_t = eff_decay * h_prev + (1-eff_decay) * received
        # ================================================================
        d_h_prev_int = d_h * eff_decay
        d_received = d_h * (1.0 - eff_decay)
        d_eff_decay = tl.sum(d_h * (h_prev - received))
        d_sigmoid = d_eff_decay * eff_decay * (1.0 - eff_decay)
        d_decay_logit_acc += d_sigmoid
        d_mod_0 = d_sigmoid

        # ================================================================
        # Backward: mod_0 = W_mod[:,0] @ hidden + b_mod[0]
        # ================================================================
        tl.atomic_add(d_b_mod_ptr + pid_n * 3 + 0, d_mod_0)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                w_val = tl.load(W_mod_ptr + pid_n * H * 3 + j * 3 + 0).to(tl.float32)
                # d_W_mod[n, j, 0]
                tl.atomic_add(d_W_mod_ptr + pid_n * H * 3 + j * 3 + 0, hj * d_mod_0)
                # d_hidden from mod
                d_hidden += d_mod_0 * w_val * hj_mask

        # ================================================================
        # Backward: hidden = tanh(W1 @ mlp_in + b1)
        # ================================================================
        d_pre_hidden = d_hidden * (1.0 - hidden * hidden)  # [BLOCK_H]

        tl.atomic_add(d_b1_ptr + pid_n * H + h_range, d_pre_hidden, mask=h_mask)

        # d_W1 and d_mlp_in
        d_h_from_mlp = tl.zeros([BLOCK_D], dtype=tl.float32)
        d_recv_from_mlp = tl.zeros([BLOCK_D], dtype=tl.float32)

        for part_idx in tl.static_range(4):
            for d in tl.static_range(BLOCK_D):
                if d < D:
                    s_mask = (d_range == d).to(tl.float32)
                    val = tl.sum(mlp_parts[part_idx] * s_mask)
                    w_off = pid_n * MLP_IN * H + (part_idx * D + d) * H
                    w_row = tl.load(W1_ptr + w_off + h_range, mask=h_mask, other=0.0).to(tl.float32)
                    # d_W1[n, part*D+d, :] += val * d_pre_hidden
                    tl.atomic_add(d_W1_ptr + w_off + h_range, val * d_pre_hidden, mask=h_mask)
                    # d_mlp_in[part*D+d] += W1[n, part*D+d, :] . d_pre_hidden
                    d_val = tl.sum(w_row * d_pre_hidden)
                    if part_idx == 0:
                        d_h_from_mlp += d_val * s_mask
                    elif part_idx == 1:
                        d_recv_from_mlp += d_val * s_mask

        # ================================================================
        # Combine d_h for next (earlier) step
        # ================================================================
        d_h = d_h_prev_int + d_h_from_mlp

        # d_received total = from integrate + from MLP
        d_recv_total = d_received + d_recv_from_mlp

        # d_inject_bc
        tl.store(d_inject_bc_ptr + save_off + d_range, d_recv_total.to(tl.bfloat16), mask=d_mask)

        # d_w_conn: for each k, d_rk * sigmoid'(rk) where d_rk = d_recv . nb_msg
        for k in range(K):
            nb_idx = tl.load(conn_idx_ptr + pid_n * K + k)
            if t > 0:
                nb_off = pid_b * T_SEG * N * D + (t - 1) * N * D + nb_idx * D
                nb_msg = tl.load(msg_all_ptr + nb_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
            else:
                nb_msg = tl.zeros([BLOCK_D], dtype=tl.float32)
            rk_raw = tl.load(w_conn_ptr + pid_n * K + k).to(tl.float32)
            rk = 1.0 / (1.0 + tl.exp(-rk_raw))
            d_weighted_k = tl.sum(d_recv_total * nb_msg)
            d_rk = d_weighted_k * rk * (1.0 - rk)
            tl.atomic_add(d_w_conn_ptr + pid_n * K + k, d_rk)

    # Store accumulated decay_logit gradient
    tl.atomic_add(d_decay_logit_ptr + pid_n, d_decay_logit_acc)
