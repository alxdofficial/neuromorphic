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

                    w2_row = tl.load(sw2_ptr + d * H_S + hid).to(tl.float32)
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

                    w2_row = tl.load(mw2_ptr + d * H_M + hid).to(tl.float32)
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


@triton.jit
def combined_cell_border_gather_fwd_kernel(
    msg_ptr,
    w_conn_sig_ptr,
    conn_idx_ptr,
    w_conn_border_sig_ptr,
    border_conn_idx_ptr,
    out_ptr,
    NC: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    K_B: tl.constexpr,
    ALPHA: tl.constexpr,
):
    """Compute one cell's received tensor, including border contributions.

    One program handles one (batch, cell) pair and emits [C, D].
    This avoids materializing [BS, NC, C, K, D] neighbor tensors.
    """
    pid = tl.program_id(0)
    b = pid // NC
    cell = pid % NC
    d = tl.arange(0, D)

    cell_base = (b * NC + cell) * C
    conn_base = cell * C
    border_base = (b * NC + cell) * B

    for n in range(C):
        acc = tl.zeros([D], dtype=tl.float32)
        edge_base = (cell_base + n) * K

        for k in range(K):
            src = tl.load(conn_idx_ptr + (conn_base + n) * K + k).to(tl.int32)
            w = tl.load(w_conn_sig_ptr + edge_base + k).to(tl.float32)
            src_offset = (cell_base + src) * D + d
            src_msg = tl.load(msg_ptr + src_offset).to(tl.float32)
            acc += w * src_msg

        if B > 0 and K_B > 0 and n >= ALPHA and n < (ALPHA + B):
            border_slot = n - ALPHA
            border_edge_base = (border_base + border_slot) * K_B
            for kb in range(K_B):
                src_border = tl.load(
                    border_conn_idx_ptr + (cell * B + border_slot) * K_B + kb
                ).to(tl.int32)
                remote_cell = src_border // B
                remote_slot = src_border % B
                remote_n = ALPHA + remote_slot
                w_b = tl.load(w_conn_border_sig_ptr + border_edge_base + kb).to(tl.float32)
                remote_offset = ((b * NC + remote_cell) * C + remote_n) * D + d
                src_msg = tl.load(msg_ptr + remote_offset).to(tl.float32)
                acc += w_b * src_msg

        tl.store(out_ptr + (cell_base + n) * D + d, acc)


class _CombinedCellBorderGatherFn(torch.autograd.Function):
    """Fused local+border gather for the live v11 path."""

    @staticmethod
    def forward(ctx, msg, w_conn_sig, conn_indices,
                w_conn_border_sig, border_conn_indices, alpha):
        if not msg.is_cuda:
            raise RuntimeError("combined gather Triton path requires CUDA tensors")

        msg_c = msg.contiguous()
        w_conn_sig_c = w_conn_sig.contiguous()
        conn_indices_c = conn_indices.contiguous()
        w_conn_border_sig_c = w_conn_border_sig.contiguous()
        border_conn_indices_c = border_conn_indices.contiguous()

        BS, NC, C, D = msg_c.shape
        K = w_conn_sig_c.shape[-1]
        B = w_conn_border_sig_c.shape[2]
        K_B = w_conn_border_sig_c.shape[3]

        out = torch.empty_like(msg_c)
        grid = (BS * NC,)
        combined_cell_border_gather_fwd_kernel[grid](
            msg_c,
            w_conn_sig_c,
            conn_indices_c,
            w_conn_border_sig_c,
            border_conn_indices_c,
            out,
            NC=NC,
            C=C,
            D=D,
            K=K,
            B=B,
            K_B=K_B,
            ALPHA=alpha,
        )

        ctx.save_for_backward(
            msg_c,
            w_conn_sig_c,
            conn_indices_c,
            w_conn_border_sig_c,
            border_conn_indices_c,
        )
        ctx.alpha = alpha
        return out

    @staticmethod
    def backward(ctx, grad_out):
        msg, w_conn_sig, conn_indices, w_conn_border_sig, border_conn_indices = ctx.saved_tensors
        alpha = ctx.alpha

        BS, NC, C, D = msg.shape
        K = w_conn_sig.shape[-1]
        B = w_conn_border_sig.shape[2]
        K_B = w_conn_border_sig.shape[3]
        dt = grad_out.dtype

        grad_out_f = grad_out.float()
        msg_f = msg.float()
        w_conn_sig_f = w_conn_sig.float()
        w_conn_border_sig_f = w_conn_border_sig.float()

        grad_msg = torch.zeros_like(msg_f)
        grad_w_conn_sig = torch.empty_like(w_conn_sig_f)

        batch_idx = torch.arange(BS, device=msg.device)[:, None, None]
        cell_idx = torch.arange(NC, device=msg.device)[None, :, None]
        conn_expanded = conn_indices.unsqueeze(0).expand(BS, -1, -1, -1)

        # Local cell gather backward, streamed over K to avoid a huge [.., K, D] tensor.
        for k in range(K):
            src = conn_expanded[..., k]
            src_msg = msg_f[batch_idx, cell_idx, src]
            grad_w_conn_sig[..., k] = (grad_out_f * src_msg).sum(dim=-1)
            src_exp = src.unsqueeze(-1).expand(-1, -1, -1, D)
            grad_msg.scatter_add_(
                2, src_exp, grad_out_f * w_conn_sig_f[..., k].unsqueeze(-1)
            )

        grad_w_conn_border_sig = torch.empty_like(w_conn_border_sig_f)
        if B > 0 and K_B > 0:
            border_local_idx = torch.arange(alpha, alpha + B, device=msg.device)
            border_local_idx = border_local_idx.unsqueeze(0).unsqueeze(0).expand(BS, NC, -1)
            border_grad = grad_out_f[:, :, alpha:alpha + B, :]
            border_flat = msg_f[:, :, alpha:alpha + B, :].reshape(BS, NC * B, D)
            grad_border_flat = torch.zeros(BS, NC * B, D, device=msg.device, dtype=msg_f.dtype)
            conn_flat = border_conn_indices.reshape(NC * B, K_B)
            batch_border_idx = torch.arange(BS, device=msg.device)[:, None]

            for kb in range(K_B):
                src = conn_flat[:, kb]
                src_msg = border_flat[batch_border_idx, src.unsqueeze(0).expand(BS, -1)]
                src_msg = src_msg.reshape(BS, NC, B, D)
                grad_w_conn_border_sig[..., kb] = (border_grad * src_msg).sum(dim=-1)

                contrib = border_grad * w_conn_border_sig_f[..., kb].unsqueeze(-1)
                grad_border_flat.scatter_add_(
                    1,
                    src.unsqueeze(0).unsqueeze(-1).expand(BS, -1, D),
                    contrib.reshape(BS, NC * B, D),
                )

            grad_border = grad_border_flat.reshape(BS, NC, B, D)
            grad_msg[:, :, alpha:alpha + B, :] += grad_border
        else:
            grad_w_conn_border_sig.zero_()

        return (
            grad_msg.to(msg.dtype),
            grad_w_conn_sig.to(w_conn_sig.dtype),
            None,
            grad_w_conn_border_sig.to(w_conn_border_sig.dtype),
            None,
            None,
        )


def combined_cell_border_gather(msg, w_conn_sig, conn_indices,
                                w_conn_border_sig, border_conn_indices,
                                alpha):
    """CUDA Triton fused gather for the live cell memory graph."""
    return _CombinedCellBorderGatherFn.apply(
        msg, w_conn_sig, conn_indices,
        w_conn_border_sig, border_conn_indices, alpha,
    )


@triton.jit
def cell_round_kernel(
    h_ptr,
    msg_ptr,
    w_conn_sig_ptr,
    w_conn_border_sig_ptr,
    decay_logit_ptr,
    prim_ptr,
    conn_idx_ptr,
    border_conn_idx_ptr,
    nid_ptr,
    inject_idx_ptr,
    inject_sig_ptr,
    sw1_ptr, sb1_ptr, sw2_ptr, sb2_ptr,
    mw1_ptr, mb1_ptr, mw2_ptr, mb2_ptr,
    h_out_ptr,
    msg_out_ptr,
    NC: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    K_B: tl.constexpr,
    ALPHA: tl.constexpr,
    H_S: tl.constexpr,
    H_M: tl.constexpr,
    STATE_IN: tl.constexpr,
    MSG_IN: tl.constexpr,
):
    """One synchronous message-passing round for one (batch, cell)."""
    pid = tl.program_id(0)
    b = pid // NC
    cell = pid % NC
    d = tl.arange(0, D)

    cell_base = (b * NC + cell) * C
    cell_nid_base = cell * C
    cell_conn_base = cell * C
    border_base = (b * NC + cell) * B
    inject_offset = (b * NC + cell) * D + d
    inject = tl.load(inject_sig_ptr + inject_offset).to(tl.float32)

    for n in range(C):
        n_offset = (cell_base + n) * D
        n_k_offset = (cell_base + n) * K

        received = tl.zeros([D], dtype=tl.float32)
        for k in range(K):
            src = tl.load(conn_idx_ptr + (cell_conn_base + n) * K + k).to(tl.int32)
            w = tl.load(w_conn_sig_ptr + n_k_offset + k).to(tl.float32)
            src_msg = tl.load(msg_ptr + (cell_base + src) * D + d).to(tl.float32)
            received += w * src_msg

        if B > 0 and K_B > 0 and n >= ALPHA and n < (ALPHA + B):
            border_slot = n - ALPHA
            border_edge_base = (border_base + border_slot) * K_B
            for kb in range(K_B):
                src_border = tl.load(
                    border_conn_idx_ptr + (cell * B + border_slot) * K_B + kb
                ).to(tl.int32)
                remote_cell = src_border // B
                remote_slot = src_border % B
                remote_n = ALPHA + remote_slot
                w_b = tl.load(w_conn_border_sig_ptr + border_edge_base + kb).to(tl.float32)
                src_msg = tl.load(
                    msg_ptr + ((b * NC + remote_cell) * C + remote_n) * D + d
                ).to(tl.float32)
                received += w_b * src_msg

        is_inject = tl.full([], 0, tl.int32)
        for a in range(ALPHA):
            idx = tl.load(inject_idx_ptr + cell * ALPHA + a).to(tl.int32)
            is_inject = tl.where(idx == n, 1, is_inject)
        input_vec = received + inject * is_inject.to(tl.float32)

        prim = tl.load(prim_ptr + n_offset + d).to(tl.float32)
        nid = tl.load(nid_ptr + (cell_nid_base + n) * D + d).to(tl.float32)
        decay_logit_val = tl.load(decay_logit_ptr + cell_base + n).to(tl.float32)
        decay_val = tl.sigmoid(decay_logit_val)
        h_prev = tl.load(h_ptr + n_offset + d).to(tl.float32)

        update_acc = tl.load(sb2_ptr + d).to(tl.float32)
        for hid in range(H_S):
            row = hid * STATE_IN
            w_in = tl.load(sw1_ptr + row + d).to(tl.float32)
            w_prim = tl.load(sw1_ptr + row + D + d).to(tl.float32)
            w_id = tl.load(sw1_ptr + row + 2 * D + d).to(tl.float32)
            w_decay = tl.load(sw1_ptr + row + 3 * D).to(tl.float32)
            b1 = tl.load(sb1_ptr + hid).to(tl.float32)

            dot = (
                tl.sum(input_vec * w_in) +
                tl.sum(prim * w_prim) +
                tl.sum(nid * w_id) +
                decay_logit_val * w_decay + b1
            )
            hidden_val = libdevice.tanh(dot)
            w2_row = tl.load(sw2_ptr + d * H_S + hid).to(tl.float32)
            update_acc += hidden_val * w2_row

        update = libdevice.tanh(update_acc)
        h_new = decay_val * h_prev + (1.0 - decay_val) * update

        msg_acc = tl.load(mb2_ptr + d).to(tl.float32)
        for hid in range(H_M):
            row = hid * MSG_IN
            w_h = tl.load(mw1_ptr + row + d).to(tl.float32)
            w_prim = tl.load(mw1_ptr + row + D + d).to(tl.float32)
            w_id = tl.load(mw1_ptr + row + 2 * D + d).to(tl.float32)
            b1 = tl.load(mb1_ptr + hid).to(tl.float32)

            dot = (
                tl.sum(h_new * w_h) +
                tl.sum(prim * w_prim) +
                tl.sum(nid * w_id) + b1
            )
            hidden_val = libdevice.tanh(dot)
            w2_row = tl.load(mw2_ptr + d * H_M + hid).to(tl.float32)
            msg_acc += hidden_val * w2_row

        msg_new = libdevice.tanh(msg_acc) + nid
        tl.store(h_out_ptr + n_offset + d, h_new.to(tl.bfloat16))
        tl.store(msg_out_ptr + n_offset + d, msg_new.to(tl.bfloat16))


def _reference_token_step(
    h, msg, w_conn_sig, w_conn_border_sig, decay_logit, primitives,
    neuron_id, inject_signal, conn_indices, border_conn_indices, inject_indices,
    state_w1, state_b1, state_w2, state_b2,
    msg_w1, msg_b1, msg_w2, msg_b2, config,
):
    import torch.nn.functional as F

    BS = h.shape[0]
    NC = config.N_cells
    C = config.C_neurons
    D = config.D_neuron
    alpha = config.alpha

    dt = torch.float32
    h = h.to(dt)
    msg = msg.to(dt)
    w_conn_sig = w_conn_sig.to(dt)
    w_conn_border_sig = w_conn_border_sig.to(dt)
    decay_logit = decay_logit.to(dt)
    primitives = primitives.to(dt)
    neuron_id = neuron_id.to(dt)
    inject_signal = inject_signal.to(dt)
    state_w1 = state_w1.to(dt)
    state_b1 = state_b1.to(dt)
    state_w2 = state_w2.to(dt)
    state_b2 = state_b2.to(dt)
    msg_w1 = msg_w1.to(dt)
    msg_b1 = msg_b1.to(dt)
    msg_w2 = msg_w2.to(dt)
    msg_b2 = msg_b2.to(dt)

    d_val = torch.sigmoid(decay_logit).unsqueeze(-1)
    omd = 1.0 - d_val

    w_in_s = state_w1[:, :D]
    w_prim_s = state_w1[:, D:2 * D]
    w_id_s = state_w1[:, 2 * D:3 * D]
    w_decay_s = state_w1[:, 3 * D:]
    w_h_m = msg_w1[:, :D]
    w_prim_m = msg_w1[:, D:2 * D]
    w_id_m = msg_w1[:, 2 * D:]

    batch_idx = torch.arange(BS, device=h.device)[:, None, None, None]
    cell_idx = torch.arange(NC, device=h.device)[None, :, None, None]
    border_idx = torch.arange(alpha, alpha + config.N_border_per_cell, device=h.device)
    border_idx = border_idx.unsqueeze(0).unsqueeze(0).expand(BS, NC, -1)
    inject_idx = inject_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)

    for _ in range(config.R_rounds):
        conn_exp = conn_indices.unsqueeze(0).expand(BS, -1, -1, -1)
        neighbor_msgs = msg[batch_idx, cell_idx, conn_exp]
        received = (w_conn_sig.unsqueeze(-1) * neighbor_msgs).sum(dim=3)

        if config.N_border_per_cell > 0 and config.K_border > 0:
            border_flat = msg[:, :, alpha:alpha + config.N_border_per_cell, :].reshape(
                BS, NC * config.N_border_per_cell, D
            )
            conn_flat = border_conn_indices.reshape(NC * config.N_border_per_cell, config.K_border)
            border_neighbor = border_flat[
                torch.arange(BS, device=h.device)[:, None, None],
                conn_flat.unsqueeze(0).expand(BS, -1, -1),
            ]
            border_neighbor = border_neighbor.reshape(
                BS, NC, config.N_border_per_cell, config.K_border, D
            )
            border_received = (w_conn_border_sig.unsqueeze(-1) * border_neighbor).sum(dim=3)
            received[:, :, alpha:alpha + config.N_border_per_cell, :] += border_received

        inject_addend = torch.zeros_like(received)
        inject_addend.scatter_(2, inject_idx, inject_signal.unsqueeze(2).expand(-1, -1, alpha, -1))
        input_vec = received + inject_addend

        hidden = (
            F.linear(input_vec, w_in_s) +
            F.linear(primitives, w_prim_s) +
            F.linear(neuron_id, w_id_s).unsqueeze(0) +
            F.linear(decay_logit.unsqueeze(-1), w_decay_s) +
            state_b1
        )
        update = torch.tanh(F.linear(torch.tanh(hidden), state_w2, state_b2))
        h = d_val * h + omd * update

        hidden_m = (
            F.linear(h, w_h_m) +
            F.linear(primitives, w_prim_m) +
            F.linear(neuron_id, w_id_m).unsqueeze(0) +
            msg_b1
        )
        msg = torch.tanh(F.linear(torch.tanh(hidden_m), msg_w2, msg_b2)) + neuron_id

    return h, msg


class _FusedTokenStepFn(torch.autograd.Function):
    """Fused one-token update with sequential rounds inside the op."""

    @staticmethod
    def forward(ctx, h, msg, w_conn_sig, w_conn_border_sig,
                decay_logit, primitives,
                conn_indices, border_conn_indices, neuron_id, inject_indices,
                inject_signal,
                state_w1, state_b1, state_w2, state_b2,
                msg_w1, msg_b1, msg_w2, msg_b2, config):
        BS, NC, C, D = h.shape
        K = w_conn_sig.shape[-1]
        B = w_conn_border_sig.shape[2]
        K_B = w_conn_border_sig.shape[3]

        h_cur = h.contiguous()
        msg_cur = msg.contiguous()
        h_next = torch.empty_like(h_cur)
        msg_next = torch.empty_like(msg_cur)
        inject_signal_c = inject_signal.contiguous()

        grid = (BS * NC,)
        for _ in range(config.R_rounds):
            cell_round_kernel[grid](
                h_cur, msg_cur,
                w_conn_sig.contiguous(),
                w_conn_border_sig.contiguous(),
                decay_logit.contiguous(),
                primitives.contiguous(),
                conn_indices.contiguous(),
                border_conn_indices.contiguous(),
                neuron_id.contiguous().float(),
                inject_indices.contiguous(),
                inject_signal_c,
                state_w1.contiguous().float(),
                state_b1.contiguous().float(),
                state_w2.contiguous().float(),
                state_b2.contiguous().float(),
                msg_w1.contiguous().float(),
                msg_b1.contiguous().float(),
                msg_w2.contiguous().float(),
                msg_b2.contiguous().float(),
                h_next,
                msg_next,
                NC=NC,
                C=C,
                D=D,
                K=K,
                B=B,
                K_B=K_B,
                ALPHA=config.alpha,
                H_S=config.state_mlp_hidden,
                H_M=config.msg_mlp_hidden,
                STATE_IN=3 * D + 1,
                MSG_IN=3 * D,
            )
            h_cur, h_next = h_next, h_cur
            msg_cur, msg_next = msg_next, msg_cur

        ctx.save_for_backward(
            h, msg, w_conn_sig, w_conn_border_sig, decay_logit, primitives,
            neuron_id, inject_signal,
            state_w1, state_b1, state_w2, state_b2,
            msg_w1, msg_b1, msg_w2, msg_b2,
        )
        ctx.config = config
        ctx.conn_indices = conn_indices
        ctx.border_conn_indices = border_conn_indices
        ctx.inject_indices = inject_indices
        return h_cur, msg_cur

    @staticmethod
    def backward(ctx, grad_h, grad_msg):
        (h0, msg0, w_conn_sig, w_conn_border_sig, decay_logit, primitives,
         neuron_id, inject_signal,
         state_w1, state_b1, state_w2, state_b2,
         msg_w1, msg_b1, msg_w2, msg_b2) = ctx.saved_tensors

        config = ctx.config
        conn_indices = ctx.conn_indices
        border_conn_indices = ctx.border_conn_indices
        inject_indices = ctx.inject_indices

        inputs = [
            h0.detach().requires_grad_(True),
            msg0.detach().requires_grad_(True),
            w_conn_sig.detach().requires_grad_(True),
            w_conn_border_sig.detach().requires_grad_(True),
            decay_logit.detach().requires_grad_(True),
            primitives.detach().requires_grad_(True),
            neuron_id.detach().requires_grad_(True),
            inject_signal.detach(),
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
            h_ref, msg_ref = _reference_token_step(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                inputs[6], inputs[7],
                conn_indices, border_conn_indices, inject_indices,
                inputs[8], inputs[9], inputs[10], inputs[11],
                inputs[12], inputs[13], inputs[14], inputs[15],
                config,
            )

        grads = torch.autograd.grad(
            outputs=(h_ref, msg_ref),
            inputs=[inp for inp in inputs if inp.requires_grad],
            grad_outputs=(grad_h.to(h_ref.dtype), grad_msg.to(msg_ref.dtype)),
            allow_unused=True,
        )

        grad_iter = iter(grads)
        result = []
        for inp in inputs:
            if inp.requires_grad:
                result.append(next(grad_iter))
            else:
                result.append(None)

        grad_h0 = result[0]
        grad_msg0 = result[1]
        grad_w_conn_sig = result[2]
        grad_w_conn_border_sig = result[3]
        grad_decay_logit = result[4]
        grad_primitives = result[5]
        grad_neuron_id = result[6]
        grad_state_w1 = result[8]
        grad_state_b1 = result[9]
        grad_state_w2 = result[10]
        grad_state_b2 = result[11]
        grad_msg_w1 = result[12]
        grad_msg_b1 = result[13]
        grad_msg_w2 = result[14]
        grad_msg_b2 = result[15]

        return (
            grad_h0, grad_msg0, grad_w_conn_sig, grad_w_conn_border_sig,
            grad_decay_logit, grad_primitives,
            None, None, grad_neuron_id, None,
            None,
            grad_state_w1, grad_state_b1, grad_state_w2, grad_state_b2,
            grad_msg_w1, grad_msg_b1, grad_msg_w2, grad_msg_b2,
            None,
        )


def fused_token_step(
    h, msg, w_conn_sig, w_conn_border_sig, decay_logit, primitives,
    conn_indices, border_conn_indices, neuron_id, inject_indices,
    inject_signal,
    state_w1, state_b1, state_w2, state_b2,
    msg_w1, msg_b1, msg_w2, msg_b2, config,
):
    return _FusedTokenStepFn.apply(
        h, msg, w_conn_sig, w_conn_border_sig, decay_logit, primitives,
        conn_indices, border_conn_indices, neuron_id, inject_indices,
        inject_signal,
        state_w1, state_b1, state_w2, state_b2,
        msg_w1, msg_b1, msg_w2, msg_b2, config,
    )
