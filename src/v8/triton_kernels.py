"""Triton forward + backward kernels for v9.1 neuron recurrence.

Each program handles one (batch, neuron) and loops over T_seg steps.
Block sizes are padded to powers of 2 with masking for irregular dims.

The inject broadcast and readout are computed outside the kernel.
The kernel handles: gather → weighted receive → add inject → MLP → integrate.
"""

import triton
import triton.language as tl


def _next_pow2(n):
    """Next power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


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
    # Padded block sizes (power of 2)
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    h_range = tl.arange(0, BLOCK_H)
    h_mask = h_range < H

    h_base = (pid_b * N + pid_n) * D

    # Load h [D]
    h_vec = tl.load(h_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    trace_h_vec = tl.load(trace_h_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    trace_recv_vec = tl.load(trace_recv_ptr + h_base + d_range, mask=d_mask, other=0.0).to(tl.float32)

    decay_logit_val = tl.load(decay_logit_ptr + pid_n)

    for t in range(T_SEG):
        # --- Gather + weighted sum ---
        received = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(K):
            nb_idx = tl.load(conn_idx_ptr + pid_n * K + k)
            nb_off = pid_b * N * D + nb_idx * D
            nb_msg = tl.load(prev_msg_ptr + nb_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
            rk_raw = tl.load(w_conn_ptr + pid_n * K + k)
            rk = 1.0 / (1.0 + tl.exp(-rk_raw))
            received += rk * nb_msg

        # --- Add inject broadcast ---
        inject_off = pid_b * T_SEG * N * D + t * N * D + pid_n * D
        inject_val = tl.load(inject_bc_ptr + inject_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
        received += inject_val

        # --- Per-neuron MLP: hidden = tanh(W1 @ [h, recv, trace_h, trace_recv] + b1) ---
        hidden = tl.load(b1_ptr + pid_n * H + h_range, mask=h_mask, other=0.0).to(tl.float32)

        # Four groups of D input dims: h, received, trace_h, trace_recv
        # For each input element, accumulate: hidden += val * W1[n, i, :]
        parts = [h_vec, received, trace_h_vec, trace_recv_vec]
        for part_idx in tl.static_range(4):
            for d in tl.static_range(BLOCK_D):
                if d < D:
                    # Extract scalar from vector via masked sum
                    scalar_mask = (d_range == d).to(tl.float32)
                    val = tl.sum(parts[part_idx] * scalar_mask)
                    w_off = pid_n * MLP_IN * H + (part_idx * D + d) * H
                    w_row = tl.load(W1_ptr + w_off + h_range, mask=h_mask, other=0.0).to(tl.float32)
                    hidden += val * w_row

        hidden = tl.extra.cuda.libdevice.tanh(hidden)

        # --- Message head: msg = tanh(W_msg @ hidden + b_msg) ---
        msg = tl.load(b_msg_ptr + pid_n * D + d_range, mask=d_mask, other=0.0).to(tl.float32)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                w_off = pid_n * H * D + j * D
                w_col = tl.load(W_msg_ptr + w_off + d_range, mask=d_mask, other=0.0).to(tl.float32)
                msg += hj * w_col
        msg = tl.extra.cuda.libdevice.tanh(msg)

        # --- Modulator: mod = W_mod @ hidden + b_mod ---
        mod_0 = tl.load(b_mod_ptr + pid_n * 3 + 0).to(tl.float32)
        for j in tl.static_range(BLOCK_H):
            if j < H:
                hj_mask = (h_range == j).to(tl.float32)
                hj = tl.sum(hidden * hj_mask)
                mod_0 += hj * tl.load(W_mod_ptr + pid_n * H * 3 + j * 3 + 0).to(tl.float32)

        # --- Integrate ---
        eff_decay = 1.0 / (1.0 + tl.exp(-(decay_logit_val + mod_0)))
        h_vec = eff_decay * h_vec + (1.0 - eff_decay) * received

        # --- Save ---
        save_off = pid_b * T_SEG * N * D + t * N * D + pid_n * D
        tl.store(h_all_ptr + save_off + d_range, h_vec.to(tl.bfloat16), mask=d_mask)
        tl.store(msg_all_ptr + save_off + d_range, msg.to(tl.bfloat16), mask=d_mask)
        tl.store(prev_msg_ptr + h_base + d_range, msg.to(tl.bfloat16), mask=d_mask)

    # Final h
    tl.store(h_ptr + h_base + d_range, h_vec.to(tl.bfloat16), mask=d_mask)
