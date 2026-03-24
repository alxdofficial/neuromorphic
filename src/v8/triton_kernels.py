"""Triton kernels for memory graph per-token neuron dynamics.

Fuses the per-token loop body into a single kernel launch per token step:
  - Key-based softmax routing: cosine sim + softmax over K neighbors
  - Weighted sum of neighbor messages (no materialized [BS,N,K,D] tensor)
  - Temporal integration: h = decay * h + (1-decay) * received
  - Message: tanh(h * primitives)
  - Fused message norm → act_trace

Two passes over K neighbors per program (all in registers):
  Pass 1: compute cosine similarities → softmax weights
  Pass 2: weighted sum of neighbor messages

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

    # Per-neuron constants (read-only within segment)
    decay_ptr,          # [BS, N] float32 — pre-computed sigmoid(decay_logit)
    primitives_ptr,     # [BS, N, D] bf16
    key_ptr,            # [BS, N, D] bf16 — L2-normalized key vectors

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
    """One step of neuron dynamics with key-based softmax routing.

    Each program handles one (batch, neuron) pair, processing all D dims.
    Two passes over K neighbors (all in registers, no materialization):
      Pass 1: cosine similarity → softmax weights
      Pass 2: weighted sum of neighbor messages
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

    # Load this neuron's key (already L2-normalized)
    key = tl.load(key_ptr + (b * N + n) * D + d).to(tl.float32)

    # === Pass 1: Compute cosine similarities, find max (online) ===
    max_sim = tl.full([], value=-float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros([], dtype=tl.float32)

    for k in range(K):
        src = tl.load(conn_idx_ptr + n * K + k)
        msg_k = tl.load(prev_msg_ptr + (b * N + src) * D + d).to(tl.float32)
        dot = tl.sum(key * msg_k)
        msg_norm = tl.sqrt(tl.sum(msg_k * msg_k) + 1e-8)
        sim_k = dot / msg_norm
        # Online softmax: update max and sum_exp in one pass
        new_max = tl.maximum(max_sim, sim_k)
        sum_exp = sum_exp * libdevice.exp(max_sim - new_max) + libdevice.exp(sim_k - new_max)
        max_sim = new_max

    # === Pass 2: weighted sum with softmax weights ===
    received = tl.zeros([D], dtype=tl.float32)

    for k in range(K):
        src = tl.load(conn_idx_ptr + n * K + k)
        msg_k = tl.load(prev_msg_ptr + (b * N + src) * D + d).to(tl.float32)
        dot = tl.sum(key * msg_k)
        msg_norm = tl.sqrt(tl.sum(msg_k * msg_k) + 1e-8)
        sim_k = dot / msg_norm
        w_k = libdevice.exp(sim_k - max_sim) / sum_exp
        received += w_k * msg_k

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

    # --- Write port neuron output ---
    if n < C:
        out_offset = b * T_seg * C * D + t_step * C * D + n * D + d
        tl.store(output_ptr + out_offset, msg.to(tl.bfloat16))
