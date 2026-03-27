"""Memory Graph — lightweight hippocampal neurons, trained by backprop.

v9.1: 8192 neurons at D=16, per-neuron MLP every token step.
Custom autograd.Function with Triton forward+backward kernels.

Per token step (fused in Triton kernel):
  1. Gather K=32 neighbor messages
  2. Scalar-weighted sum (w_conn)
  3. Add precomputed inject broadcast
  4. Per-neuron MLP: cat(h, received, trace_h, trace_recv) → hidden → msg + mod
  5. Integrate: h = eff_decay * h + (1-eff_decay) * received
  6. Save h, msg for backward

Inject broadcast and readout are precomputed outside kernel as einsums.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config

try:
    import triton
    from .triton_kernels import neuron_fwd_kernel, neuron_bwd_kernel
    _HAS_TRITON = True
except (ImportError, AttributeError):
    _HAS_TRITON = False


class MemoryGraph(nn.Module):
    """Hippocampal memory graph with per-neuron MLPs + Triton recurrence."""

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K = config.K_connections
        D = config.D_mem
        H = config.neuron_hidden
        C_mem = config.D // config.D_mem
        self.C_mem = C_mem

        # Fixed sparse topology
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')
        K_actual = min(K, N - 1)
        conn_indices = torch.zeros(N, K, dtype=torch.long, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        sorted_idx, _ = conn_indices.sort(dim=-1)
        self.register_buffer('conn_indices', sorted_idx)

        # Broadcast inject/readout
        self.inject_w = nn.Parameter(torch.zeros(N, C_mem))
        nn.init.uniform_(self.inject_w, -1.0, 1.0)
        self.readout_w = nn.Parameter(torch.zeros(C_mem, N))
        nn.init.uniform_(self.readout_w, -1.0, 1.0)

        # Per-neuron recurrence params
        self.w_conn = nn.Parameter(torch.zeros(N, K))
        nn.init.uniform_(self.w_conn, -0.5, 0.5)
        self.decay_logit = nn.Parameter(torch.zeros(N))

        # Per-neuron MLP: cat(h, received, trace_h, trace_recv) [4D] → hidden [H]
        self.W1 = nn.Parameter(torch.empty(N, 4 * D, H))
        self.b1 = nn.Parameter(torch.zeros(N, H))
        nn.init.kaiming_uniform_(self.W1, nonlinearity='tanh')

        # Message head [H → D]
        self.W_msg = nn.Parameter(torch.empty(N, H, D))
        self.b_msg = nn.Parameter(torch.zeros(N, D))
        nn.init.kaiming_uniform_(self.W_msg, nonlinearity='tanh')
        self.W_msg.data.mul_(0.1)

        # Modulator head [H → 3] (decay_mod, plasticity_gate, plasticity_lr)
        self.W_mod = nn.Parameter(torch.zeros(N, H, 3))
        self.b_mod = nn.Parameter(torch.zeros(N, 3))

        # Hebbian learning rate
        self.hebbian_lr_logit = nn.Parameter(torch.tensor(-4.0))

        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        device = self.w_conn.device
        N, D = self.config.N_neurons, self.config.D_mem

        self.h = torch.randn(BS, N, D, device=device, dtype=self.dtype) * 0.01
        self.prev_messages = torch.zeros(BS, N, D, device=device, dtype=self.dtype)
        self.trace_h = torch.zeros(BS, N, D, device=device, dtype=self.dtype)
        self.trace_received = torch.zeros(BS, N, D, device=device, dtype=self.dtype)
        self.msg_magnitude = torch.zeros(BS, N, device=device, dtype=self.dtype)
        self._initialized = True

    # ================================================================
    # Forward segment
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment. Inject → recurrence → readout.

        Args:
            cc_signals: [BS, T_seg, C_mem, D_mem]
        Returns:
            output: [BS, T_seg, C_mem, D_mem]
        """
        BS, T_seg, C_mem, D = cc_signals.shape
        cc_mem = cc_signals.to(self.dtype)

        # TBPTT: detach at segment boundary
        h = self.h.detach()
        prev_msg = self.prev_messages.detach()

        # 1. Precompute inject broadcast: [BS, T, N, D]
        inject_weights = torch.sigmoid(self.inject_w)  # [N, C_mem]
        inject_bc = torch.einsum(
            'nc,btcd->btnd', inject_weights.float(),
            cc_mem.float()).to(self.dtype)

        # 2. Recurrence via custom autograd.Function
        h_final, msg_final, msg_all = NeuronRecurrence.apply(
            h, prev_msg, inject_bc,
            self.w_conn, self.decay_logit,
            self.W1, self.b1, self.W_msg, self.b_msg,
            self.W_mod, self.b_mod,
            self.trace_h, self.trace_received,
            self.conn_indices, T_seg)

        # 3. Readout: [BS, T, C_mem, D]
        readout_weights = torch.sigmoid(self.readout_w)  # [C_mem, N]
        output = torch.einsum(
            'cn,btnd->btcd', readout_weights.float(),
            msg_all.float()).to(cc_signals.dtype)

        # Side effects (detached)
        with torch.no_grad():
            # Hebbian plasticity (gated by modulator from last step)
            eta = torch.sigmoid(self.hebbian_lr_logit)
            neighbor_msgs = msg_final.detach()[:, self.conn_indices]
            pre = neighbor_msgs.mean(dim=-1)
            post = msg_final.detach().norm(dim=-1, keepdim=True)
            hebb = pre * post

            # Run modulator to get gate (cheap — one forward on final state)
            mlp_in = torch.cat([
                h_final.detach().float(), msg_final.detach().float(),
                self.trace_h.float(), self.trace_received.float(),
            ], dim=-1)
            _hidden = torch.tanh(
                torch.einsum('bnd,ndh->bnh', mlp_in, self.W1.data.float())
                + self.b1.data.float())
            _mod = (torch.einsum('bnh,nho->bno', _hidden, self.W_mod.data.float())
                    + self.b_mod.data.float())
            p_gate = torch.tanh(_mod[..., 1])    # plasticity gate
            p_lr = torch.sigmoid(_mod[..., 2])    # plasticity lr
            gate = (p_gate * p_lr).unsqueeze(-1)  # [BS, N, 1]

            self.w_conn.data += (eta * gate * hebb).mean(dim=0).float()

            # Update traces
            td = 0.95
            self.trace_h = (td * self.trace_h + (1 - td) * h_final.detach()).to(self.dtype)
            self.trace_received = (td * self.trace_received +
                                   (1 - td) * msg_final.detach()).to(self.dtype)

        self.h = h_final.detach()
        self.prev_messages = msg_final.detach()

        with torch.no_grad():
            alpha = 0.05
            seg_mag = msg_final.detach().norm(dim=-1)
            self.msg_magnitude = (1 - alpha) * self.msg_magnitude + alpha * seg_mag

        return output

    # ================================================================
    # Utilities
    # ================================================================

    def detach_states(self):
        self.h = self.h.detach()
        self.prev_messages = self.prev_messages.detach()

    def runtime_state_dict(self) -> dict:
        return {
            'h': self.h, 'prev_messages': self.prev_messages,
            'trace_h': self.trace_h, 'trace_received': self.trace_received,
            'msg_magnitude': self.msg_magnitude,
        }

    def load_runtime_state(self, state: dict):
        buffer_names = {name for name, _ in self.named_buffers()}
        param_names = {name for name, _ in self.named_parameters()}
        for key, val in state.items():
            if key not in buffer_names and key not in param_names and hasattr(self, key):
                setattr(self, key, val)
        self._initialized = True


# ====================================================================
# Custom autograd.Function
# ====================================================================

class NeuronRecurrence(torch.autograd.Function):
    """Per-token recurrence with per-neuron MLP. Triton or Python."""

    @staticmethod
    def forward(ctx, h, prev_msg, inject_bc,
                w_conn, decay_logit, W1, b1, W_msg, b_msg, W_mod, b_mod,
                trace_h, trace_recv, conn_indices, T_seg):
        BS, N, D = h.shape
        K = w_conn.shape[1]
        H = W1.shape[2]
        device = h.device
        dtype = h.dtype

        h_all = torch.empty(BS, T_seg, N, D, device=device, dtype=dtype)
        msg_all = torch.empty(BS, T_seg, N, D, device=device, dtype=dtype)

        # Triton kernels are slower for per-neuron MLP (scalar extraction overhead).
        # Use Python forward with GPU-optimized einsum instead.
        use_triton = False  # _HAS_TRITON and h.is_cuda

        if use_triton:
            # Make contiguous copies for kernel
            h_work = h.contiguous().clone()
            prev_msg_work = prev_msg.contiguous().clone()

            from .triton_kernels import _next_pow2
            grid = (BS, N)
            neuron_fwd_kernel[grid](
                h_work, prev_msg_work,
                inject_bc.contiguous(),
                w_conn, decay_logit,
                W1, b1, W_msg, b_msg, W_mod, b_mod,
                trace_h, trace_recv,
                conn_indices,
                h_all, msg_all,
                BS=BS, N=N, D=D, K=K, H=H,
                T_SEG=T_seg, MLP_IN=4*D,
                BLOCK_D=_next_pow2(D), BLOCK_K=_next_pow2(K),
                BLOCK_H=_next_pow2(H),
            )
            h_final = h_work
            msg_final = prev_msg_work
        else:
            # Python fallback
            h_final, msg_final, h_all, msg_all = _python_recurrence_fwd(
                h, prev_msg, inject_bc,
                w_conn, decay_logit, W1, b1, W_msg, b_msg, W_mod, b_mod,
                trace_h, trace_recv, conn_indices, T_seg)

        ctx.save_for_backward(h_all, msg_all, inject_bc,
                              w_conn, decay_logit, W1, b1, W_msg, b_msg,
                              W_mod, b_mod, trace_h, trace_recv)
        ctx.conn_indices = conn_indices
        ctx.T_seg = T_seg
        ctx.use_triton = use_triton

        return h_final, msg_final, msg_all

    @staticmethod
    def backward(ctx, d_h_final, d_msg_final, d_msg_all):
        (h_all, msg_all, inject_bc,
         w_conn, decay_logit, W1, b1, W_msg, b_msg,
         W_mod, b_mod, trace_h, trace_recv) = ctx.saved_tensors
        conn_indices = ctx.conn_indices
        T_seg = ctx.T_seg

        BS, N, D = d_msg_all.shape[0], d_msg_all.shape[2], d_msg_all.shape[3]
        K = w_conn.shape[1]
        H = W1.shape[2]

        if ctx.use_triton:
            grads = _triton_recurrence_bwd(
                h_all, msg_all, inject_bc,
                w_conn, decay_logit, W1, b1, W_msg, b_msg, W_mod, b_mod,
                trace_h, trace_recv, conn_indices,
                d_msg_all, d_h_final, T_seg)
        else:
            grads = _python_recurrence_bwd(
                h_all, msg_all, inject_bc,
                w_conn, decay_logit, W1, b1, W_msg, b_msg, W_mod, b_mod,
                trace_h, trace_recv, conn_indices,
                d_msg_all, d_h_final, T_seg)

        # Return grads in same order as forward inputs
        # h, prev_msg, inject_bc, w_conn, decay_logit, W1, b1, W_msg, b_msg,
        # W_mod, b_mod, trace_h, trace_recv, conn_indices, T_seg
        return (None, None, grads['d_inject_bc'],
                grads['d_w_conn'], grads['d_decay_logit'],
                grads['d_W1'], grads['d_b1'],
                grads['d_W_msg'], grads['d_b_msg'],
                grads['d_W_mod'], grads['d_b_mod'],
                None, None, None, None)


# ====================================================================
# Python forward/backward (reference implementation)
# ====================================================================

def _python_recurrence_fwd(h, prev_msg, inject_bc,
                           w_conn, decay_logit, W1, b1, W_msg, b_msg,
                           W_mod, b_mod, trace_h, trace_recv,
                           conn_indices, T_seg):
    """Python reference forward. Returns h_final, msg_final, h_all, msg_all."""
    BS, N, D = h.shape
    K = w_conn.shape[1]
    dtype = h.dtype
    device = h.device

    routing = torch.sigmoid(w_conn)  # [N, K]

    h_all = torch.empty(BS, T_seg, N, D, device=device, dtype=dtype)
    msg_all = torch.empty(BS, T_seg, N, D, device=device, dtype=dtype)

    for t in range(T_seg):
        # Gather + weighted sum
        neighbor_msgs = prev_msg[:, conn_indices]  # [BS, N, K, D]
        weighted = routing.unsqueeze(0).unsqueeze(-1) * neighbor_msgs
        received = weighted.sum(dim=2)

        # Add inject broadcast
        received = received + inject_bc[:, t]

        # Per-neuron MLP
        mlp_in = torch.cat([
            h.float(), received.float(),
            trace_h.float(), trace_recv.float(),
        ], dim=-1)
        hidden = torch.tanh(
            torch.einsum('bnd,ndh->bnh', mlp_in, W1.float()) + b1.float())

        # Message head
        msg = torch.tanh(
            torch.einsum('bnh,nhd->bnd', hidden, W_msg.float()) + b_msg.float())
        msg = msg.to(dtype)

        # Modulator head
        mod_out = torch.einsum('bnh,nho->bno', hidden, W_mod.float()) + b_mod.float()
        decay_mod = mod_out[..., 0]

        # Integrate
        eff_decay = torch.sigmoid(decay_logit.unsqueeze(0) + decay_mod)
        h = (eff_decay.unsqueeze(-1) * h +
             (1 - eff_decay.unsqueeze(-1)) * received)

        prev_msg = msg
        h_all[:, t] = h
        msg_all[:, t] = msg

    return h, prev_msg, h_all, msg_all


def _python_recurrence_bwd(h_all, msg_all, inject_bc,
                           w_conn, decay_logit, W1, b1, W_msg, b_msg,
                           W_mod, b_mod, trace_h, trace_recv,
                           conn_indices, d_msg_all, d_h_final, T_seg):
    """Python reference backward."""
    BS, N, D = h_all.shape[0], h_all.shape[2], h_all.shape[3]
    K = w_conn.shape[1]
    H = W1.shape[2]
    MLP_IN = 4 * D
    device = h_all.device

    routing = torch.sigmoid(w_conn.float())

    d_w_conn = torch.zeros_like(w_conn, dtype=torch.float32)
    d_decay_logit = torch.zeros_like(decay_logit, dtype=torch.float32)
    d_W1 = torch.zeros_like(W1, dtype=torch.float32)
    d_b1 = torch.zeros_like(b1, dtype=torch.float32)
    d_W_msg = torch.zeros_like(W_msg, dtype=torch.float32)
    d_b_msg = torch.zeros_like(b_msg, dtype=torch.float32)
    d_W_mod = torch.zeros_like(W_mod, dtype=torch.float32)
    d_b_mod = torch.zeros_like(b_mod, dtype=torch.float32)
    d_inject_bc = torch.zeros_like(inject_bc, dtype=torch.float32)

    d_h = d_h_final.float() if d_h_final is not None else torch.zeros(
        BS, N, D, device=device, dtype=torch.float32)

    for t in range(T_seg - 1, -1, -1):
        h_t = h_all[:, t].float()
        msg_t = msg_all[:, t].float()
        h_prev = h_all[:, t - 1].float() if t > 0 else torch.zeros_like(h_t)
        prev_msg_t = msg_all[:, t - 1] if t > 0 else torch.zeros_like(msg_all[:, 0])

        # Recompute received
        neighbor_msgs = prev_msg_t.float()[:, conn_indices]
        weighted = routing.unsqueeze(0).unsqueeze(-1) * neighbor_msgs
        received = weighted.sum(dim=2) + inject_bc[:, t].float()

        # Recompute hidden
        mlp_in = torch.cat([h_prev, received,
                            trace_h.float(), trace_recv.float()], dim=-1)
        pre_tanh_hidden = torch.einsum('bnd,ndh->bnh', mlp_in, W1.float()) + b1.float()
        hidden = torch.tanh(pre_tanh_hidden)

        # Recompute mod for eff_decay
        mod_out = torch.einsum('bnh,nho->bno', hidden, W_mod.float()) + b_mod.float()
        eff_decay = torch.sigmoid(decay_logit.float().unsqueeze(0) + mod_out[..., 0])

        # --- Backward through integrate ---
        d_h_prev = d_h * eff_decay.unsqueeze(-1)
        d_received = d_h * (1 - eff_decay.unsqueeze(-1))
        d_eff_decay = (d_h * (h_prev - received)).sum(dim=-1)  # [BS, N]

        # d_decay_logit and d_mod_0
        d_sigmoid = d_eff_decay * eff_decay * (1 - eff_decay)
        d_decay_logit += d_sigmoid.sum(dim=0)
        d_mod_0 = d_sigmoid

        # --- Backward through msg = tanh(W_msg @ hidden + b_msg) ---
        d_msg_t_total = d_msg_all[:, t].float()
        d_pre_tanh_msg = d_msg_t_total * (1 - msg_t * msg_t)
        d_b_msg += d_pre_tanh_msg.sum(dim=0)

        d_hidden_from_msg = torch.einsum('bnd,nhd->bnh', d_pre_tanh_msg, W_msg.float())
        d_W_msg += torch.einsum('bnh,bnd->nhd', hidden, d_pre_tanh_msg).sum(dim=0) / BS

        # --- Backward through mod head (for d_mod_0) ---
        d_hidden_from_mod = torch.zeros_like(hidden)
        d_mod_out = torch.zeros(BS, N, 3, device=device, dtype=torch.float32)
        d_mod_out[..., 0] = d_mod_0
        d_hidden_from_mod = torch.einsum('bno,nho->bnh', d_mod_out, W_mod.float())
        d_W_mod += torch.einsum('bnh,bno->nho', hidden, d_mod_out).sum(dim=0) / BS
        d_b_mod += d_mod_out.sum(dim=0).mean(dim=0) if d_mod_out.dim() > 2 else d_mod_out.sum(dim=0)

        # --- Backward through hidden = tanh(W1 @ mlp_in + b1) ---
        d_hidden = d_hidden_from_msg + d_hidden_from_mod
        d_pre_tanh = d_hidden * (1 - hidden * hidden)
        d_b1 += d_pre_tanh.sum(dim=0)
        d_W1 += torch.einsum('bnd,bnh->ndh', mlp_in, d_pre_tanh).sum(dim=0) / BS
        d_mlp_in = torch.einsum('bnh,ndh->bnd', d_pre_tanh, W1.float())

        # Split d_mlp_in → d_h_prev_from_mlp, d_received_from_mlp
        d_h_prev_from_mlp = d_mlp_in[:, :, :D]
        d_received_from_mlp = d_mlp_in[:, :, D:2*D]

        # --- Combine d_h for next step ---
        d_h = d_h_prev + d_h_prev_from_mlp

        # --- d_received total (from integrate + from MLP) ---
        d_received_total = d_received + d_received_from_mlp
        d_inject_bc[:, t] = d_received_total

        # --- d_w_conn ---
        d_weighted = d_received_total.unsqueeze(2).expand_as(neighbor_msgs)
        d_routing = (d_weighted * neighbor_msgs).sum(dim=-1).mean(dim=0)
        d_w_conn += d_routing * routing * (1 - routing)

    return {
        'd_w_conn': d_w_conn,
        'd_decay_logit': d_decay_logit,
        'd_W1': d_W1, 'd_b1': d_b1,
        'd_W_msg': d_W_msg, 'd_b_msg': d_b_msg,
        'd_W_mod': d_W_mod, 'd_b_mod': d_b_mod,
        'd_inject_bc': d_inject_bc.to(inject_bc.dtype),
    }


def _triton_recurrence_bwd(h_all, msg_all, inject_bc,
                           w_conn, decay_logit, W1, b1, W_msg, b_msg,
                           W_mod, b_mod, trace_h, trace_recv,
                           conn_indices, d_msg_all, d_h_final, T_seg):
    """Triton backward — uses kernel for speed."""
    BS, N, D = h_all.shape[0], h_all.shape[2], h_all.shape[3]
    K = w_conn.shape[1]
    H = W1.shape[2]
    device = h_all.device

    d_w_conn = torch.zeros_like(w_conn, dtype=torch.float32)
    d_decay_logit = torch.zeros_like(decay_logit, dtype=torch.float32)
    d_W1 = torch.zeros_like(W1, dtype=torch.float32)
    d_b1 = torch.zeros_like(b1, dtype=torch.float32)
    d_W_msg = torch.zeros_like(W_msg, dtype=torch.float32)
    d_b_msg = torch.zeros_like(b_msg, dtype=torch.float32)
    d_W_mod = torch.zeros_like(W_mod, dtype=torch.float32)
    d_b_mod = torch.zeros_like(b_mod, dtype=torch.float32)
    d_inject_bc = torch.zeros_like(inject_bc, dtype=torch.float32)

    from .triton_kernels import _next_pow2
    grid = (BS, N)
    neuron_bwd_kernel[grid](
        h_all.contiguous(), msg_all.contiguous(), inject_bc.contiguous(),
        w_conn, decay_logit, W1, b1, W_msg, b_msg, W_mod, b_mod,
        trace_h, trace_recv, conn_indices,
        d_msg_all.contiguous(),
        d_w_conn, d_decay_logit, d_W1, d_b1, d_W_msg, d_b_msg,
        d_W_mod, d_b_mod, d_inject_bc,
        BS=BS, N=N, D=D, K=K, H=H,
        T_SEG=T_seg, MLP_IN=4*D,
        BLOCK_D=_next_pow2(D), BLOCK_K=_next_pow2(K),
        BLOCK_H=_next_pow2(H),
    )

    return {
        'd_w_conn': d_w_conn,
        'd_decay_logit': d_decay_logit,
        'd_W1': d_W1, 'd_b1': d_b1,
        'd_W_msg': d_W_msg, 'd_b_msg': d_b_msg,
        'd_W_mod': d_W_mod, 'd_b_mod': d_b_mod,
        'd_inject_bc': d_inject_bc.to(inject_bc.dtype),
    }
