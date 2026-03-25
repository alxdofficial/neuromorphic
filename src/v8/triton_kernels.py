"""Triton kernels for memory graph per-token neuron dynamics.

Fuses the per-token loop body into a single kernel launch per token step:
  - Sparse gather from K presynaptic neighbors using precomputed scalar weights
  - Temporal integration: h = decay * h + (1-decay) * received
  - Message: tanh(h * primitives)
  - Fused message norm → act_trace
  - Accumulate received and msg into f32 running sums for mean_input/mean_output

Routing weights are computed once per segment from key × neighbor messages
(softmax over K neighbors). The kernel uses these fixed scalar weights
for all tokens in the segment — same structure as the original fixed-weight
kernel, full speed.

Python fallback in memory_graph.py remains the source of truth for correctness.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def memory_graph_step_kernel(
    # Persistent state (read+write in-place)
    h_ptr,              # [BS, N, D] bf16
    prev_msg_ptr,       # [BS, N, D] bf16

    # Sparse connectivity (read-only)
    conn_idx_ptr,       # [N, K] int32 — presynaptic neuron indices
    conn_w_ptr,         # [BS, N, K] bf16 — precomputed softmax routing weights

    # Per-neuron constants (read-only within segment)
    decay_ptr,          # [BS, N] float32 — pre-computed sigmoid(decay_logit)
    primitives_ptr,     # [BS, N, D] bf16

    # Per-token inputs/outputs
    cc_signals_ptr,     # [BS, T_seg, C, D] bf16 — full segment CC signals
    eot_flags_ptr,      # [BS, T_seg] float32 — 1.0 at EOT, 0.0 otherwise
    output_ptr,         # [BS, T_seg, C, D] bf16 — port neuron messages

    # Activation trace (fused norm output)
    act_trace_ptr,      # [BS, T_seg, N] float32 — message norm per neuron per token

    # Running accumulators for mean_input / mean_output (f32, read-modify-write)
    recv_accum_ptr,     # [BS, N, D] float32 — sum of received signals across segment
    msg_accum_ptr,      # [BS, N, D] float32 — sum of outgoing messages across segment

    # Current token step
    t_step,             # int — which token in the segment

    # Dimensions
    BS: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    T_seg: tl.constexpr,
):
    """One step of neuron dynamics with precomputed routing weights.

    Each program handles one (batch, neuron) pair, processing all D dims.
    Routing weights computed once per segment from key-based softmax.
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    d = tl.arange(0, D)

    # --- Load per-neuron constants ---
    decay_val = tl.load(decay_ptr + b * N + n).to(tl.float32)
    eot_val = tl.load(eot_flags_ptr + b * T_seg + t_step).to(tl.float32)
    eff_decay = decay_val * (1.0 - eot_val)
    eff_omd = 1.0 - eff_decay

    prim = tl.load(primitives_ptr + (b * N + n) * D + d).to(tl.float32)
    h_val = tl.load(h_ptr + (b * N + n) * D + d).to(tl.float32)

    # --- Sparse gather: received = sum_k w[k] * prev_msg[src[k], :] ---
    received = tl.zeros([D], dtype=tl.float32)

    for k in range(K):
        src = tl.load(conn_idx_ptr + n * K + k)
        w = tl.load(conn_w_ptr + (b * N + n) * K + k).to(tl.float32)
        val = tl.load(prev_msg_ptr + (b * N + src) * D + d).to(tl.float32)
        received += w * val

    # --- CC signal injection (port neurons only) ---
    if n < C:
        cc_offset = b * T_seg * C * D + t_step * C * D + n * D + d
        cc = tl.load(cc_signals_ptr + cc_offset).to(tl.float32)
        received += cc

    # --- Temporal integration ---
    h_new = eff_decay * h_val + eff_omd * received

    # --- Compute outgoing message ---
    msg = libdevice.tanh(h_new * prim)

    # --- Store results (in-place) ---
    tl.store(h_ptr + (b * N + n) * D + d, h_new.to(tl.bfloat16))
    tl.store(prev_msg_ptr + (b * N + n) * D + d, msg.to(tl.bfloat16))

    # --- Fused message norm → act_trace ---
    msg_norm = tl.sqrt(tl.sum(msg * msg))
    tl.store(act_trace_ptr + b * T_seg * N + t_step * N + n, msg_norm)

    # --- Accumulate received and msg for mean_input / mean_output ---
    accum_offset = (b * N + n) * D + d
    old_recv = tl.load(recv_accum_ptr + accum_offset)
    tl.store(recv_accum_ptr + accum_offset, old_recv + received)
    old_msg = tl.load(msg_accum_ptr + accum_offset)
    tl.store(msg_accum_ptr + accum_offset, old_msg + msg)

    # --- Write port neuron output ---
    if n < C:
        out_offset = b * T_seg * C * D + t_step * C * D + n * D + d
        tl.store(output_ptr + out_offset, msg.to(tl.bfloat16))
