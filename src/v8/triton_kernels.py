"""Triton kernels for memory graph per-token neuron dynamics.

Fuses the per-token loop body (sparse gather + integration + tanh + output)
into a single kernel launch per token step. Replaces the dense [N,N] bmm
with sparse gather from conn_indices (96 loads instead of 1024 multiplies).

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
    conn_w_ptr,         # [BS, N, K] bf16 — pre-normalized connection weights

    # Per-neuron constants (read-only within segment)
    decay_ptr,          # [BS, N] float32 — pre-computed sigmoid(decay_logit)
    primitives_ptr,     # [BS, N, D] bf16

    # Per-token inputs/outputs
    cc_signals_ptr,     # [BS, T_seg, C, D] bf16 — full segment CC signals
    eot_flags_ptr,      # [BS, T_seg] float32 — 1.0 at EOT, 0.0 otherwise
    output_ptr,         # [BS, T_seg, C, D] bf16 — port neuron messages

    # Activation trace (fused norm output)
    act_trace_ptr,      # [BS, T_seg, N] float32 — message norm per neuron per token

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
    """One step of neuron dynamics for all neurons in one batch element.

    Each program handles one (batch, neuron) pair, processing all D dims.
    Includes RMSNorm on h (bounds state), no tanh (RMSNorm sufficient).
    Fuses message L2 norm into act_trace (no extra kernel).
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

    # --- RMSNorm on h: bounds state, eliminates port/internal gap ---
    h_rms = tl.sqrt(tl.sum(h_new * h_new) / D + 1e-8)
    h_new = h_new / h_rms

    # --- Compute outgoing message (no tanh — RMSNorm already bounds h) ---
    msg = h_new * prim

    # --- Store results (in-place) ---
    tl.store(h_ptr + (b * N + n) * D + d, h_new.to(tl.bfloat16))
    tl.store(prev_msg_ptr + (b * N + n) * D + d, msg.to(tl.bfloat16))

    # --- Fused message norm → act_trace ---
    msg_norm = tl.sqrt(tl.sum(msg * msg))
    tl.store(act_trace_ptr + b * T_seg * N + t_step * N + n, msg_norm)

    # --- Write port neuron output ---
    if n < C:
        out_offset = b * T_seg * C * D + t_step * C * D + n * D + d
        tl.store(output_ptr + out_offset, msg.to(tl.bfloat16))


@triton.jit
def prepare_sparse_weights_kernel(
    conn_weights_ptr,   # [BS, N, K] bf16 — L1-normalized weights
    conn_mask_ptr,      # [N, K] bool — active connections
    conn_w_norm_ptr,    # [BS, N, K] bf16 — output: masked weights
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Apply connection mask to L1-normalized weights.

    Weights are already L1-normalized (sum |w| = 1 per neuron).
    This just masks inactive connections (rare — only when K > N-1).
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, BLOCK_K)
    mask_k = k < K

    w = tl.load(conn_weights_ptr + (b * N + n) * K + k, mask=mask_k, other=0.0)
    m = tl.load(conn_mask_ptr + n * K + k, mask=mask_k, other=0).to(tl.float32)

    w_masked = w.to(tl.float32) * m

    tl.store(conn_w_norm_ptr + (b * N + n) * K + k, w_masked.to(tl.bfloat16), mask=mask_k)
