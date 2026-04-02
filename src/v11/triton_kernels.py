"""Triton kernels for v11 cell-based memory graph.

Main kernel: cell_forward_kernel — one program per (batch, cell).
Loads cell state into registers/L1, processes T×R steps locally,
writes back final state and readout. No global memory access in inner loop.

Forward only. Backward via PyTorch reference recomputation.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def cell_forward_kernel(
    # State tensors [BS, NC, C, D] or [BS, NC, C, K] or [BS, NC, C]
    h_ptr,              # [BS, NC, C, D] bf16 — in/out
    msg_ptr,            # [BS, NC, C, D] bf16 — in/out
    w_conn_sig_ptr,     # [BS, NC, C, K] bf16 — pre-sigmoided weights
    decay_ptr,          # [BS, NC, C] bf16 — pre-sigmoided decay
    prim_ptr,           # [BS, NC, C, D] bf16
    # Fixed buffers
    conn_idx_ptr,       # [NC, C, K] int64 — cell-local indices
    nid_ptr,            # [NC, C, D] f32 — neuron identity
    inject_idx_ptr,     # [NC, alpha] int64 — inject neuron indices
    readout_idx_ptr,    # [NC, alpha] int64 — readout neuron indices
    # LM signals
    cc_ptr,             # [BS, T, D_LM] bf16 — input signals
    mem_out_ptr,        # [BS, T, D_LM] f32 — output readout
    # Shared MLP weights (f32, small — fit in registers)
    sw1_ptr, sb1_ptr, sw2_ptr, sb2_ptr,   # state MLP
    mw1_ptr, mb1_ptr, mw2_ptr, mb2_ptr,   # msg MLP
    # Hebbian output
    hebb_out_ptr,       # [BS, NC, C, K] f32 — accumulated
    # Dimensions
    BS: tl.constexpr, NC: tl.constexpr,
    C: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    T: tl.constexpr, D_LM: tl.constexpr,
    R: tl.constexpr, ALPHA: tl.constexpr,
    H_S: tl.constexpr, H_M: tl.constexpr,
    STATE_IN: tl.constexpr,  # 3*D + 1
    MSG_IN: tl.constexpr,    # 3*D
):
    """Process one (batch, cell) for all T tokens × R rounds.

    Each program handles C=256 neurons sequentially. All neuron data for
    the cell is accessed from contiguous global memory that fits in L1
    cache (~10 KB per cell at D=8, C=256).
    """
    b = tl.program_id(0)
    cell = tl.program_id(1)
    d = tl.arange(0, D)

    # Base offsets for this (batch, cell)
    cell_base = (b * NC + cell) * C  # offset into [BS, NC, C, ...] flattened
    cell_nid_base = cell * C          # offset into [NC, C, ...]
    cell_conn_base = cell * C         # offset into [NC, C, K]
    readout_scale = 1.0 / ALPHA

    # Load inject/readout indices for this cell
    # inject_idx: ALPHA indices into [0..C)
    # We'll check per-neuron if it's an inject neuron

    # ---- TOKEN LOOP ----
    for t in range(T):
        # Load inject signal for this token: cc[b, t, cell*D : cell*D + D]
        cc_offset = (b * T + t) * D_LM + cell * D
        inject = tl.load(cc_ptr + cc_offset + d).to(tl.float32)

        # ---- ROUND LOOP ----
        for r in range(R):
            # Process each neuron in the cell
            for n in range(C):
                n_offset = (cell_base + n) * D
                n_k_offset = (cell_base + n) * K

                # ---- GATHER: read K neighbors' messages ----
                received = tl.zeros([D], dtype=tl.float32)
                for k in range(K):
                    src = tl.load(conn_idx_ptr + (cell_conn_base + n) * K + k)
                    w = tl.load(w_conn_sig_ptr + n_k_offset + k).to(tl.float32)
                    src_msg = tl.load(
                        msg_ptr + (cell_base + src) * D + d).to(tl.float32)
                    received += w * src_msg

                # ---- INJECT: add LM signal to inject neurons only ----
                # Check if neuron n is an inject neuron
                is_inject = tl.full([], 0, tl.int32)
                for a in range(ALPHA):
                    idx = tl.load(inject_idx_ptr + cell * ALPHA + a)
                    is_inject = tl.where(idx == n, 1, is_inject)
                input_vec = received + inject * is_inject.to(tl.float32)

                # ---- LOAD per-neuron constants ----
                prim = tl.load(prim_ptr + n_offset + d).to(tl.float32)
                nid = tl.load(nid_ptr + (cell_nid_base + n) * D + d).to(tl.float32)
                decay_val = tl.load(decay_ptr + cell_base + n).to(tl.float32)
                h_val = tl.load(h_ptr + n_offset + d).to(tl.float32)

                # ---- STATE MLP: input_vec, prim, nid, decay → update ----
                # Layer 1: H_S hidden units
                update_acc = tl.load(sb2_ptr + d).to(tl.float32)  # bias2
                for hid in range(H_S):
                    row = hid * STATE_IN
                    w_in = tl.load(sw1_ptr + row + d).to(tl.float32)
                    w_prim = tl.load(sw1_ptr + row + D + d).to(tl.float32)
                    w_id = tl.load(sw1_ptr + row + 2 * D + d).to(tl.float32)
                    w_decay = tl.load(sw1_ptr + row + 3 * D).to(tl.float32)
                    b1 = tl.load(sb1_ptr + hid).to(tl.float32)

                    dot = (tl.sum(input_vec * w_in) +
                           tl.sum(prim * w_prim) +
                           tl.sum(nid * w_id) +
                           decay_val * w_decay + b1)
                    hidden_val = libdevice.tanh(dot)

                    w2_row = tl.load(sw2_ptr + hid * D + d).to(tl.float32)
                    update_acc += hidden_val * w2_row

                update = libdevice.tanh(update_acc)

                # ---- TEMPORAL: h = decay * h + (1-decay) * update ----
                h_new = decay_val * h_val + (1.0 - decay_val) * update
                tl.store(h_ptr + n_offset + d, h_new.to(tl.bfloat16))

                # ---- MSG MLP: h, prim, nid → msg ----
                msg_acc = tl.load(mb2_ptr + d).to(tl.float32)  # bias2
                for hid in range(H_M):
                    row = hid * MSG_IN
                    w_h = tl.load(mw1_ptr + row + d).to(tl.float32)
                    w_prim = tl.load(mw1_ptr + row + D + d).to(tl.float32)
                    w_id = tl.load(mw1_ptr + row + 2 * D + d).to(tl.float32)
                    b1 = tl.load(mb1_ptr + hid).to(tl.float32)

                    dot = (tl.sum(h_new * w_h) +
                           tl.sum(prim * w_prim) +
                           tl.sum(nid * w_id) + b1)
                    hidden_val = libdevice.tanh(dot)

                    w2_row = tl.load(mw2_ptr + hid * D + d).to(tl.float32)
                    msg_acc += hidden_val * w2_row

                msg_new = libdevice.tanh(msg_acc) + nid
                tl.store(msg_ptr + n_offset + d, msg_new.to(tl.bfloat16))

        # ---- READOUT: average readout neurons ----
        readout = tl.zeros([D], dtype=tl.float32)
        for a in range(ALPHA):
            rn = tl.load(readout_idx_ptr + cell * ALPHA + a)
            rn_msg = tl.load(msg_ptr + (cell_base + rn) * D + d).to(tl.float32)
            readout += rn_msg
        readout = readout * readout_scale

        # Store readout to mem_out[b, t, cell*D : cell*D + D]
        out_offset = (b * T + t) * D_LM + cell * D
        tl.store(mem_out_ptr + out_offset + d, readout)

    # ---- WRITE BACK hebbian traces (accumulated |msg| × w) ----
    for n in range(C):
        n_offset = (cell_base + n) * D
        n_k_offset = (cell_base + n) * K
        final_msg = tl.load(msg_ptr + n_offset + d).to(tl.float32)
        msg_norm = tl.sqrt(tl.sum(final_msg * final_msg))
        for k in range(K):
            w = tl.load(w_conn_sig_ptr + n_k_offset + k).to(tl.float32)
            tl.store(hebb_out_ptr + n_k_offset + k, msg_norm * w)


class _FusedCellForwardFn(torch.autograd.Function):
    """Autograd wrapper: Triton forward, PyTorch recompute backward."""

    @staticmethod
    def forward(ctx, h, msg, w_conn_sig, decay, primitives,
                conn_indices, neuron_id, inject_indices, readout_indices,
                cc_signals, state_w1, state_b1, state_w2, state_b2,
                msg_w1, msg_b1, msg_w2, msg_b2, config):
        BS = h.shape[0]
        NC = config.N_cells
        C = config.C_neurons
        D = config.D_neuron
        K = config.K_connections
        T = cc_signals.shape[1]
        D_LM = cc_signals.shape[2]
        R = config.R_rounds
        ALPHA = config.alpha
        H_S = config.state_mlp_hidden
        H_M = config.msg_mlp_hidden

        dt = h.dtype
        mem_out = torch.zeros(BS, T, D_LM, device=h.device, dtype=torch.float32)
        hebb_out = torch.zeros(BS, NC, C, K, device=h.device, dtype=torch.float32)

        # Make h and msg contiguous copies (kernel writes in-place)
        h_work = h.contiguous().clone()
        msg_work = msg.contiguous().clone()

        grid = (BS, NC)
        cell_forward_kernel[grid](
            h_work, msg_work,
            w_conn_sig.contiguous(),
            decay.contiguous(),
            primitives.contiguous(),
            conn_indices.contiguous(),
            neuron_id.contiguous().float(),
            inject_indices.contiguous(),
            readout_indices.contiguous(),
            cc_signals.contiguous(),
            mem_out,
            state_w1.contiguous().float(),
            state_b1.contiguous().float(),
            state_w2.contiguous().float(),
            state_b2.contiguous().float(),
            msg_w1.contiguous().float(),
            msg_b1.contiguous().float(),
            msg_w2.contiguous().float(),
            msg_b2.contiguous().float(),
            hebb_out,
            BS=BS, NC=NC, C=C, D=D, K=K,
            T=T, D_LM=D_LM, R=R, ALPHA=ALPHA,
            H_S=H_S, H_M=H_M,
            STATE_IN=3*D+1, MSG_IN=3*D,
        )

        ctx.save_for_backward(
            h, msg, w_conn_sig, decay, primitives,
            neuron_id, cc_signals,
            state_w1, state_b1, state_w2, state_b2,
            msg_w1, msg_b1, msg_w2, msg_b2,
        )
        ctx.config = config
        ctx.conn_indices = conn_indices
        ctx.inject_indices = inject_indices
        ctx.readout_indices = readout_indices

        return h_work, msg_work, mem_out, hebb_out

    @staticmethod
    def backward(ctx, grad_h, grad_msg, grad_mem_out, grad_hebb):
        """Recompute forward with PyTorch autograd, then backprop."""
        saved = ctx.saved_tensors
        (h0, msg0, w_conn_sig, decay, primitives,
         neuron_id, cc_signals,
         state_w1, state_b1, state_w2, state_b2,
         msg_w1, msg_b1, msg_w2, msg_b2) = saved

        config = ctx.config
        conn_indices = ctx.conn_indices
        inject_indices = ctx.inject_indices
        readout_indices = ctx.readout_indices

        # Recompute with autograd using the reference implementation
        inputs = [
            h0.detach().requires_grad_(True),
            msg0.detach().requires_grad_(True),
            w_conn_sig.detach().requires_grad_(True),
            decay.detach().requires_grad_(True),
            primitives.detach().requires_grad_(True),
            neuron_id.detach().requires_grad_(True),
            cc_signals.detach(),  # detached from LM
            state_w1.detach().requires_grad_(True),
            state_b1.detach().requires_grad_(True),
            state_w2.detach().requires_grad_(True),
            state_b2.detach().requires_grad_(True),
            msg_w1.detach().requires_grad_(True),
            msg_b1.detach().requires_grad_(True),
            msg_w2.detach().requires_grad_(True),
            msg_b2.detach().requires_grad_(True),
        ]

        with torch.enable_grad():
            h_ref, msg_ref, mem_out_ref = _reference_cell_forward(
                *inputs, conn_indices, inject_indices, readout_indices, config)

        grads = torch.autograd.grad(
            outputs=(h_ref, msg_ref, mem_out_ref),
            inputs=[inp for inp in inputs if inp.requires_grad],
            grad_outputs=(
                grad_h.to(h_ref.dtype),
                grad_msg.to(msg_ref.dtype),
                grad_mem_out.to(mem_out_ref.dtype),
            ),
            allow_unused=True,
        )

        # Map back to input order
        grad_iter = iter(grads)
        result = []
        for inp in inputs:
            if inp.requires_grad:
                result.append(next(grad_iter))
            else:
                result.append(None)

        # Return grads for: h, msg, w_conn_sig, decay, primitives,
        #   conn_indices(None), neuron_id, inject_indices(None),
        #   readout_indices(None), cc_signals(None),
        #   state_w1..msg_b2, config(None)
        return (
            result[0], result[1], result[2], result[3], result[4],
            None,  # conn_indices
            result[5],  # neuron_id
            None, None,  # inject/readout indices
            None,  # cc_signals
            result[6], result[7], result[8], result[9],   # state MLP
            result[10], result[11], result[12], result[13],  # msg MLP
            None,  # config
        )


def _reference_cell_forward(
    h, msg, w_conn_sig, decay, primitives, neuron_id, cc_signals,
    state_w1, state_b1, state_w2, state_b2,
    msg_w1, msg_b1, msg_w2, msg_b2,
    conn_indices, inject_indices, readout_indices, config,
):
    """PyTorch reference for one full cell forward pass (all tokens, all rounds)."""
    import torch.nn.functional as F
    BS = h.shape[0]
    NC = config.N_cells
    C = config.C_neurons
    D = config.D_neuron
    K = config.K_connections
    T = cc_signals.shape[1]
    R = config.R_rounds
    alpha = config.alpha

    # Cast everything to a common dtype (f32 for backward precision)
    dt = torch.float32
    h = h.to(dt)
    msg = msg.to(dt)
    w_conn_sig = w_conn_sig.to(dt)
    decay = decay.to(dt)
    primitives = primitives.to(dt)
    neuron_id = neuron_id.to(dt)
    cc_signals = cc_signals.to(dt)
    state_w1 = state_w1.to(dt)
    state_b1 = state_b1.to(dt)
    state_w2 = state_w2.to(dt)
    state_b2 = state_b2.to(dt)
    msg_w1 = msg_w1.to(dt)
    msg_b1 = msg_b1.to(dt)
    msg_w2 = msg_w2.to(dt)
    msg_b2 = msg_b2.to(dt)

    d_val = decay.unsqueeze(-1)  # [BS, NC, C, 1]
    omd = 1 - d_val

    # Split shared weights
    w_in_s = state_w1[:, :D]
    w_prim_s = state_w1[:, D:2*D]
    w_id_s = state_w1[:, 2*D:3*D]
    w_decay_s = state_w1[:, 3*D:]

    w_h_m = msg_w1[:, :D]
    w_prim_m = msg_w1[:, D:2*D]
    w_id_m = msg_w1[:, 2*D:]

    nid = neuron_id

    def _one_token_step(h_in, msg_in, inject_raw):
        """Process one token: R rounds of message passing."""
        h_t, msg_t = h_in, msg_in
        for r in range(R):
            batch_idx = torch.arange(BS, device=h_t.device)[:, None, None, None]
            cell_idx = torch.arange(NC, device=h_t.device)[None, :, None, None]
            conn_exp = conn_indices.unsqueeze(0).expand(BS, -1, -1, -1)
            neighbor_msgs = msg_t[batch_idx, cell_idx, conn_exp]
            received = (w_conn_sig.unsqueeze(-1) * neighbor_msgs).sum(dim=3)

            inject_addend = torch.zeros_like(received)
            inject_signal = inject_raw.unsqueeze(2).expand(-1, -1, alpha, -1)
            idx = inject_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)
            inject_addend.scatter_(2, idx, inject_signal)
            input_vec = received + inject_addend

            hidden = (
                F.linear(input_vec, w_in_s) +
                F.linear(primitives, w_prim_s) +
                F.linear(nid, w_id_s).unsqueeze(0) +
                F.linear(decay.unsqueeze(-1), w_decay_s) +
                state_b1
            )
            update = torch.tanh(F.linear(torch.tanh(hidden), state_w2, state_b2))
            h_t = d_val * h_t + omd * update

            hidden_m = (
                F.linear(h_t, w_h_m) +
                F.linear(primitives, w_prim_m) +
                F.linear(nid, w_id_m).unsqueeze(0) +
                msg_b1
            )
            msg_t = torch.tanh(F.linear(torch.tanh(hidden_m), msg_w2, msg_b2)) + nid

        idx_r = readout_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)
        readout_msgs = torch.gather(msg_t, 2, idx_r)
        averaged = readout_msgs.mean(dim=2)
        return h_t, msg_t, averaged.reshape(BS, NC * D)

    readouts = []
    for t in range(T):
        inject_raw = cc_signals[:, t].view(BS, NC, D)
        h, msg, readout_t = torch.utils.checkpoint.checkpoint(
            _one_token_step, h, msg, inject_raw, use_reentrant=False)
        readouts.append(readout_t)

    mem_out = torch.stack(readouts, dim=1)  # [BS, T, D_lm]
    return h, msg, mem_out


def fused_cell_forward(h, msg, w_conn_sig, decay, primitives,
                       conn_indices, neuron_id, inject_indices, readout_indices,
                       cc_signals, state_w1, state_b1, state_w2, state_b2,
                       msg_w1, msg_b1, msg_w2, msg_b2, config):
    """Run the fused cell forward pass.

    Returns: (h_final, msg_final, mem_out, hebb_traces)
    """
    return _FusedCellForwardFn.apply(
        h, msg, w_conn_sig, decay, primitives,
        conn_indices, neuron_id, inject_indices, readout_indices,
        cc_signals, state_w1, state_b1, state_w2, state_b2,
        msg_w1, msg_b1, msg_w2, msg_b2, config)
