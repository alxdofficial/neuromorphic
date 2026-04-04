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
            w_raw = tl.load(w_ptr + w_base + k).to(tl.float32)
            w = 1.0 / (1.0 + tl.exp(-w_raw))
            src_base = cell_msg_base + src * DN + d
            src_msg = tl.load(msg_ptr + src_base, mask=dmask, other=0).to(tl.float32)
            acc += w * src_msg

        tl.store(out_ptr + out_base + d, acc, mask=dmask)


    @triton.jit
    def local_receive_bwd_kernel(
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
            w_raw = tl.load(w_ptr + w_base + k).to(tl.float32)
            sig = 1.0 / (1.0 + tl.exp(-w_raw))
            src_base = cell_msg_base + src * DN + d
            src_msg = tl.load(msg_ptr + src_base, mask=dmask, other=0).to(tl.float32)
            grad_w = tl.sum(grad * src_msg, axis=0) * sig * (1.0 - sig)
            tl.store(grad_w_ptr + w_base + k, grad_w)
            tl.atomic_add(grad_msg_ptr + src_base, grad * sig, mask=dmask)


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


class _LocalReceiveFn(torch.autograd.Function):
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
        local_receive_bwd_kernel[grid](
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


def local_receive(msg: torch.Tensor, w_conn: torch.Tensor, conn_idx: torch.Tensor) -> torch.Tensor:
    """Weighted local neighbor receive.

    Falls back to eager PyTorch on CPU or when Triton is unavailable.
    """
    if _HAS_TRITON and msg.is_cuda and w_conn.is_cuda and conn_idx.is_cuda:
        return _LocalReceiveFn.apply(msg, w_conn, conn_idx)

    bs = msg.shape[0]
    device = msg.device
    batch_idx = torch.arange(bs, device=device)[:, None, None, None]
    cell_idx = torch.arange(msg.shape[1], device=device)[None, :, None, None]
    conn = conn_idx.to(device).unsqueeze(0).expand(bs, -1, -1, -1)
    gathered = msg[batch_idx, cell_idx, conn]
    return (gathered * torch.sigmoid(w_conn).unsqueeze(-1)).sum(dim=3)


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
    conn = conn_idx.to(device).unsqueeze(0).expand(bs, -1, -1, -1)
    gathered = msg_prev[batch_idx, cell_idx, conn]
    correlation = (gathered * msg_new.unsqueeze(3)).sum(dim=-1)
    return hebbian * decay + correlation * (1.0 - decay)
