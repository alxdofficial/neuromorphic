"""Triton kernels for the cell-grid memory graph hot path."""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def local_receive_fwd_kernel(
        msg_ptr,
        w_ptr,
        conn_ptr,
        out_ptr,
        NC: tl.constexpr,
        CN: tl.constexpr,
        DN: tl.constexpr,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid % CN
        tmp = pid // CN
        cell = tmp % NC
        b = tmp // NC

        d = tl.arange(0, BLOCK_D)
        dmask = d < DN

        cell_msg_base = ((b * NC + cell) * CN) * DN
        w_base = (((b * NC + cell) * CN + n) * K)
        conn_base = ((cell * CN + n) * K)
        out_base = (((b * NC + cell) * CN + n) * DN)

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(K):
            src = tl.load(conn_ptr + conn_base + k)
            w = tl.load(w_ptr + w_base + k).to(tl.float32)
            src_base = cell_msg_base + src * DN + d
            src_msg = tl.load(msg_ptr + src_base, mask=dmask, other=0).to(tl.float32)
            acc += w * src_msg

        tl.store(out_ptr + out_base + d, acc, mask=dmask)


    @triton.jit
    def local_receive_bwd_atomic_kernel(
        grad_out_ptr,
        msg_ptr,
        w_ptr,
        conn_ptr,
        grad_msg_ptr,
        grad_w_ptr,
        NC: tl.constexpr,
        CN: tl.constexpr,
        DN: tl.constexpr,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid % CN
        tmp = pid // CN
        cell = tmp % NC
        b = tmp // NC

        d = tl.arange(0, BLOCK_D)
        dmask = d < DN

        cell_msg_base = ((b * NC + cell) * CN) * DN
        w_base = (((b * NC + cell) * CN + n) * K)
        conn_base = ((cell * CN + n) * K)
        out_base = (((b * NC + cell) * CN + n) * DN)

        grad = tl.load(grad_out_ptr + out_base + d, mask=dmask, other=0).to(tl.float32)

        for k in range(K):
            src = tl.load(conn_ptr + conn_base + k)
            w = tl.load(w_ptr + w_base + k).to(tl.float32)
            src_base = cell_msg_base + src * DN + d
            src_msg = tl.load(msg_ptr + src_base, mask=dmask, other=0).to(tl.float32)
            grad_w = tl.sum(grad * src_msg, axis=0)
            tl.store(grad_w_ptr + w_base + k, grad_w)
            tl.atomic_add(grad_msg_ptr + src_base, grad * w, mask=dmask)


    @triton.jit
    def local_receive_bwd_csr_kernel(
        grad_out_ptr,
        msg_ptr,
        w_ptr,
        offsets_ptr,
        dst_ptr,
        slot_ptr,
        grad_msg_ptr,
        grad_w_ptr,
        NC: tl.constexpr,
        CN: tl.constexpr,
        DN: tl.constexpr,
        K: tl.constexpr,
        E: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        src = pid % CN
        tmp = pid // CN
        cell = tmp % NC
        b = tmp // NC

        d = tl.arange(0, BLOCK_D)
        dmask = d < DN

        cell_msg_base = ((b * NC + cell) * CN) * DN
        src_base = cell_msg_base + src * DN
        src_msg = tl.load(msg_ptr + src_base + d, mask=dmask, other=0).to(tl.float32)

        offsets_base = cell * (CN + 1)
        start = tl.load(offsets_ptr + offsets_base + src)
        end = tl.load(offsets_ptr + offsets_base + src + 1)
        edge_base = cell * E

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        edge = start
        while edge < end:
            edge_idx = edge_base + edge
            dst = tl.load(dst_ptr + edge_idx)
            slot = tl.load(slot_ptr + edge_idx)

            out_base = (((b * NC + cell) * CN + dst) * DN)
            grad = tl.load(grad_out_ptr + out_base + d, mask=dmask, other=0).to(tl.float32)

            w_base = (((b * NC + cell) * CN + dst) * K + slot)
            w = tl.load(w_ptr + w_base).to(tl.float32)
            acc += grad * w

            grad_w = tl.sum(grad * src_msg, axis=0)
            tl.store(grad_w_ptr + w_base, grad_w)
            edge += 1

        tl.store(grad_msg_ptr + src_base + d, acc, mask=dmask)


    @triton.jit
    def border_exchange_fwd_kernel(
        border_ptr,
        gate_ptr,
        neighbor_ptr,
        src_port_ptr,
        out_ptr,
        NC: tl.constexpr,
        BP: tl.constexpr,
        DN: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        port = pid % BP
        tmp = pid // BP
        cell = tmp % NC
        b = tmp // NC

        d = tl.arange(0, BLOCK_D)
        dmask = d < DN

        neighbor = tl.load(neighbor_ptr + cell * BP + port)
        out_base = (((b * NC + cell) * BP + port) * DN)

        if neighbor >= 0:
            src_port = tl.load(src_port_ptr + port)
            src_base = (((b * NC + neighbor) * BP + src_port) * DN)
            src = tl.load(border_ptr + src_base + d, mask=dmask, other=0).to(tl.float32)
            gate = tl.load(gate_ptr + ((b * NC + cell) * BP + port)).to(tl.float32)
            tl.store(out_ptr + out_base + d, src * gate, mask=dmask)
        else:
            tl.store(out_ptr + out_base + d, tl.zeros([BLOCK_D], dtype=tl.float32), mask=dmask)


    @triton.jit
    def border_exchange_bwd_kernel(
        grad_out_ptr,
        border_ptr,
        gate_ptr,
        neighbor_ptr,
        src_port_ptr,
        grad_border_ptr,
        grad_gate_ptr,
        NC: tl.constexpr,
        BP: tl.constexpr,
        DN: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        port = pid % BP
        tmp = pid // BP
        cell = tmp % NC
        b = tmp // NC

        d = tl.arange(0, BLOCK_D)
        dmask = d < DN

        neighbor = tl.load(neighbor_ptr + cell * BP + port)
        grad_gate_base = ((b * NC + cell) * BP + port)
        out_base = (((b * NC + cell) * BP + port) * DN)
        grad = tl.load(grad_out_ptr + out_base + d, mask=dmask, other=0).to(tl.float32)

        if neighbor >= 0:
            src_port = tl.load(src_port_ptr + port)
            src_base = (((b * NC + neighbor) * BP + src_port) * DN)
            src = tl.load(border_ptr + src_base + d, mask=dmask, other=0).to(tl.float32)
            gate = tl.load(gate_ptr + grad_gate_base).to(tl.float32)
            tl.store(grad_border_ptr + src_base + d, grad * gate, mask=dmask)
            tl.store(grad_gate_ptr + grad_gate_base, tl.sum(grad * src, axis=0))
        else:
            tl.store(grad_gate_ptr + grad_gate_base, 0.0)


    @triton.jit
    def hebbian_ema_kernel(
        msg_prev_ptr,
        msg_new_ptr,
        conn_ptr,
        hebb_ptr,
        out_ptr,
        NC: tl.constexpr,
        CN: tl.constexpr,
        DN: tl.constexpr,
        K: tl.constexpr,
        DECAY: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid % CN
        tmp = pid // CN
        cell = tmp % NC
        b = tmp // NC

        d = tl.arange(0, BLOCK_D)
        dmask = d < DN
        k_idx = tl.arange(0, K)

        cell_msg_base = ((b * NC + cell) * CN) * DN
        dst_base = (((b * NC + cell) * CN + n) * DN)
        conn_base = ((cell * CN + n) * K)
        hebb_base = (((b * NC + cell) * CN + n) * K)

        dst = tl.load(msg_new_ptr + dst_base + d, mask=dmask, other=0).to(tl.float32)
        src = tl.load(conn_ptr + conn_base + k_idx)
        src_offsets = cell_msg_base + src[:, None] * DN + d[None, :]
        src_msg = tl.load(msg_prev_ptr + src_offsets, mask=dmask[None, :], other=0).to(tl.float32)
        corr = tl.sum(src_msg * dst[None, :], axis=1)

        hebb_prev = tl.load(hebb_ptr + hebb_base + k_idx).to(tl.float32)
        updated = hebb_prev * DECAY + corr * (1.0 - DECAY)
        tl.store(out_ptr + hebb_base + k_idx, updated)

class _LocalReceiveAtomicFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg: torch.Tensor, w_conn: torch.Tensor, conn_idx: torch.Tensor):
        if not (_HAS_TRITON and msg.is_cuda and w_conn.is_cuda and conn_idx.is_cuda):
            raise RuntimeError("Triton local receive forward called without CUDA Triton support")

        bs, nc, cn, dn = msg.shape
        k = w_conn.shape[-1]
        block_d = 8 if dn <= 8 else 16 if dn <= 16 else 32

        out = torch.empty_like(msg)
        grid = (bs * nc * cn,)
        local_receive_fwd_kernel[grid](
            msg.contiguous(),
            w_conn.contiguous(),
            conn_idx.contiguous(),
            out,
            NC=nc,
            CN=cn,
            DN=dn,
            K=k,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(msg, w_conn, conn_idx)
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        msg, w_conn, conn_idx = ctx.saved_tensors
        bs, nc, cn, dn = msg.shape
        k = w_conn.shape[-1]

        grad_msg_acc = torch.zeros_like(msg, dtype=torch.float32)
        grad_w = torch.empty_like(w_conn, dtype=torch.float32)

        grid = (bs * nc * cn,)
        local_receive_bwd_atomic_kernel[grid](
            grad_out.contiguous(),
            msg.contiguous(),
            w_conn.contiguous(),
            conn_idx.contiguous(),
            grad_msg_acc,
            grad_w,
            NC=nc,
            CN=cn,
            DN=dn,
            K=k,
            BLOCK_D=ctx.block_d,
        )
        return grad_msg_acc.to(msg.dtype), grad_w.to(w_conn.dtype), None


class _LocalReceiveCsrFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        msg: torch.Tensor,
        w_conn: torch.Tensor,
        conn_idx: torch.Tensor,
        src_edge_offsets: torch.Tensor,
        src_edge_dst: torch.Tensor,
        src_edge_slot: torch.Tensor,
    ):
        if not (_HAS_TRITON and msg.is_cuda and w_conn.is_cuda and conn_idx.is_cuda):
            raise RuntimeError("Triton local receive forward called without CUDA Triton support")

        bs, nc, cn, dn = msg.shape
        k = w_conn.shape[-1]
        block_d = 8 if dn <= 8 else 16 if dn <= 16 else 32

        out = torch.empty_like(msg)
        grid = (bs * nc * cn,)
        local_receive_fwd_kernel[grid](
            msg.contiguous(),
            w_conn.contiguous(),
            conn_idx.contiguous(),
            out,
            NC=nc,
            CN=cn,
            DN=dn,
            K=k,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(msg, w_conn, src_edge_offsets, src_edge_dst, src_edge_slot)
        ctx.block_d = block_d
        ctx.edge_count = cn * k
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        msg, w_conn, src_edge_offsets, src_edge_dst, src_edge_slot = ctx.saved_tensors
        bs, nc, cn, dn = msg.shape
        k = w_conn.shape[-1]

        grad_msg_acc = torch.empty_like(msg, dtype=torch.float32)
        grad_w = torch.empty_like(w_conn, dtype=torch.float32)

        grid = (bs * nc * cn,)
        local_receive_bwd_csr_kernel[grid](
            grad_out.contiguous(),
            msg.contiguous(),
            w_conn.contiguous(),
            src_edge_offsets.contiguous(),
            src_edge_dst.contiguous(),
            src_edge_slot.contiguous(),
            grad_msg_acc,
            grad_w,
            NC=nc,
            CN=cn,
            DN=dn,
            K=k,
            E=ctx.edge_count,
            BLOCK_D=ctx.block_d,
        )
        return grad_msg_acc.to(msg.dtype), grad_w.to(w_conn.dtype), None, None, None, None


class _BorderExchangeActivatedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        border_msg: torch.Tensor,
        gate: torch.Tensor,
        neighbor_cell: torch.Tensor,
        src_port: torch.Tensor,
    ):
        if not (_HAS_TRITON and border_msg.is_cuda and gate.is_cuda):
            raise RuntimeError("Triton border exchange forward called without CUDA Triton support")

        border_msg = border_msg.contiguous()
        gate = gate.contiguous()
        bs, nc, bp, dn = border_msg.shape
        block_d = 8 if dn <= 8 else 16 if dn <= 16 else 32

        out = torch.empty_like(border_msg)
        grid = (bs * nc * bp,)
        border_exchange_fwd_kernel[grid](
            border_msg,
            gate.reshape(bs, nc, bp).contiguous(),
            neighbor_cell.contiguous(),
            src_port.contiguous(),
            out,
            NC=nc,
            BP=bp,
            DN=dn,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(border_msg, gate, neighbor_cell, src_port)
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        border_msg, gate, neighbor_cell, src_port = ctx.saved_tensors
        bs, nc, bp, dn = border_msg.shape

        grad_border = torch.zeros_like(border_msg, dtype=torch.float32)
        grad_gate = torch.empty(bs, nc, bp, device=gate.device, dtype=torch.float32)

        grid = (bs * nc * bp,)
        border_exchange_bwd_kernel[grid](
            grad_out.contiguous(),
            border_msg.contiguous(),
            gate.reshape(bs, nc, bp).contiguous(),
            neighbor_cell.contiguous(),
            src_port.contiguous(),
            grad_border,
            grad_gate,
            NC=nc,
            BP=bp,
            DN=dn,
            BLOCK_D=ctx.block_d,
        )
        return grad_border.to(border_msg.dtype), grad_gate.unsqueeze(-1).to(gate.dtype), None, None


def local_receive_activated(
    msg: torch.Tensor,
    w_conn: torch.Tensor,
    conn_idx: torch.Tensor,
    src_edge_offsets: torch.Tensor | None = None,
    src_edge_dst: torch.Tensor | None = None,
    src_edge_slot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted local neighbor receive.

    Falls back to eager PyTorch on CPU or when Triton is unavailable.
    """
    if (
        _HAS_TRITON
        and msg.is_cuda
        and w_conn.is_cuda
        and conn_idx.is_cuda
        and src_edge_offsets is not None
        and src_edge_dst is not None
        and src_edge_slot is not None
    ):
        return _LocalReceiveCsrFn.apply(
            msg, w_conn, conn_idx, src_edge_offsets, src_edge_dst, src_edge_slot)
    if _HAS_TRITON and msg.is_cuda and w_conn.is_cuda and conn_idx.is_cuda:
        return _LocalReceiveAtomicFn.apply(msg, w_conn, conn_idx)

    bs = msg.shape[0]
    device = msg.device
    batch_idx = torch.arange(bs, device=device)[:, None, None, None]
    cell_idx = torch.arange(msg.shape[1], device=device)[None, :, None, None]
    conn = conn_idx.to(device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1, -1, -1)
    gathered = msg[batch_idx, cell_idx, conn]
    return (gathered * w_conn.unsqueeze(-1)).sum(dim=3)


def local_receive(
    msg: torch.Tensor,
    w_conn: torch.Tensor,
    conn_idx: torch.Tensor,
    src_edge_offsets: torch.Tensor | None = None,
    src_edge_dst: torch.Tensor | None = None,
    src_edge_slot: torch.Tensor | None = None,
) -> torch.Tensor:
    return local_receive_activated(
        msg,
        torch.sigmoid(w_conn),
        conn_idx,
        src_edge_offsets,
        src_edge_dst,
        src_edge_slot,
    )


def border_exchange_activated(
    border_msg: torch.Tensor,
    gate: torch.Tensor,
    neighbor_cell: torch.Tensor,
    src_port: torch.Tensor,
) -> torch.Tensor:
    if _HAS_TRITON and border_msg.is_cuda and gate.is_cuda and neighbor_cell.is_cuda and src_port.is_cuda:
        return _BorderExchangeActivatedFn.apply(border_msg, gate, neighbor_cell, src_port)

    bs, nc, bp, dn = border_msg.shape
    out = torch.zeros_like(border_msg)
    neighbors = neighbor_cell.to(device=border_msg.device, dtype=torch.long)
    ports = src_port.to(device=border_msg.device, dtype=torch.long)
    for port in range(bp):
        valid = neighbors[:, port] >= 0
        if valid.any():
            src_cells = neighbors[valid, port]
            out[:, valid, port] = border_msg[:, src_cells, ports[port]]
    return out * gate


def hebbian_ema_update(
    msg_prev: torch.Tensor,
    msg_new: torch.Tensor,
    hebbian: torch.Tensor,
    conn_idx: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """Detached Hebbian EMA update over local connectivity.

    Falls back to eager PyTorch on CPU or when Triton is unavailable.
    """
    msg_prev = msg_prev.detach()
    msg_new = msg_new.detach()

    if _HAS_TRITON and msg_prev.is_cuda and msg_new.is_cuda and hebbian.is_cuda and conn_idx.is_cuda:
        _, nc, cn, dn = msg_prev.shape
        k = hebbian.shape[-1]
        block_d = 8 if dn <= 8 else 16 if dn <= 16 else 32
        out = torch.empty_like(hebbian)
        grid = (msg_prev.shape[0] * nc * cn,)
        hebbian_ema_kernel[grid](
            msg_prev.contiguous(),
            msg_new.contiguous(),
            conn_idx.contiguous(),
            hebbian.contiguous(),
            out,
            NC=nc,
            CN=cn,
            DN=dn,
            K=k,
            DECAY=decay,
            BLOCK_D=block_d,
        )
        return out

    bs = msg_prev.shape[0]
    device = msg_prev.device
    batch_idx = torch.arange(bs, device=device)[:, None, None, None]
    cell_idx = torch.arange(msg_prev.shape[1], device=device)[None, :, None, None]
    conn = conn_idx.to(device=device, dtype=torch.long).unsqueeze(0).expand(bs, -1, -1, -1)
    gathered = msg_prev[batch_idx, cell_idx, conn]
    correlation = (gathered * msg_new.unsqueeze(3)).sum(dim=-1)
    return hebbian * decay + correlation * (1.0 - decay)
