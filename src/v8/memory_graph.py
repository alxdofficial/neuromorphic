"""Memory Graph — lightweight hippocampal neurons, trained by backprop.

v9.1: Many simple neurons (N=8192, D=16) instead of few complex ones.

Architecture split for Triton compatibility:
  1. Per-segment MLP (PyTorch, differentiable, runs ONCE per segment):
     cat(h, trace_h, trace_recv) → hidden → msg_mod [N,D] + decay_mod [N]
     + plasticity_gate [N] + plasticity_lr [N]
  2. Per-token recurrence (Triton kernel, 128 steps fused):
     Simple element-wise: gather → weighted receive → inject → integrate → message
     msg = tanh(h * msg_mod), h = decay*h + (1-decay)*received
  3. Readout (PyTorch, differentiable):
     output = readout_w @ msg_all

The MLP produces per-neuron modulations that stay constant within a segment.
The recurrence uses these as constants — no matmuls in the inner loop.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config

try:
    from .triton_kernels import neuron_recurrence_fwd, neuron_recurrence_bwd
    _HAS_TRITON_KERNELS = True
except ImportError:
    _HAS_TRITON_KERNELS = False


class MemoryGraph(nn.Module):
    """Hippocampal memory graph with per-neuron MLP + Triton recurrence.

    Per-segment MLP (PyTorch, ~4K params per neuron):
        W1 [3D, H], b1 [H] — hidden layer (input: h, trace_h, trace_recv)
        W_msg [H, D], b_msg [D] — message modulation head
        W_mod [H, 3], b_mod [3] — modulator head (decay_mod, gate, lr)

    Per-token recurrence (Triton, simple element-wise):
        w_conn [K] — scalar synaptic weights
        decay_logit [1] — base decay
        msg_mod [D] — per-neuron message modulation (from MLP)
        eff_decay [1] — effective decay (from MLP)

    State: h [BS, N, D], prev_messages [BS, N, D], traces
    """

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

        # ============================================================
        # Fixed sparse topology
        # ============================================================
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

        # ============================================================
        # Per-neuron recurrence params (used inside Triton kernel)
        # ============================================================
        self.w_conn = nn.Parameter(torch.zeros(N, K))
        nn.init.uniform_(self.w_conn, -0.5, 0.5)

        self.decay_logit = nn.Parameter(torch.zeros(N))

        # ============================================================
        # Per-neuron MLP (runs once per segment, produces modulations)
        # Input: cat(h, trace_h, trace_recv) [3D]
        # ============================================================
        self.W1 = nn.Parameter(torch.empty(N, 3 * D, H))
        self.b1 = nn.Parameter(torch.zeros(N, H))
        nn.init.kaiming_uniform_(self.W1, nonlinearity='tanh')

        # Message modulation head [H → D] (like learned primitives)
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
    # Per-segment MLP (PyTorch, differentiable)
    # ================================================================

    def _segment_mlp(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-neuron MLP: compute modulations for this segment.

        Args:
            h: [BS, N, D] neuron state at segment start

        Returns:
            msg_mod: [BS, N, D] message modulation (like primitives)
            eff_decay: [BS, N] effective decay for this segment
            plasticity_gate: [BS, N] Hebbian gate
            plasticity_lr: [BS, N] Hebbian learning rate modulation
        """
        mlp_in = torch.cat([
            h.float(), self.trace_h.float(), self.trace_received.float(),
        ], dim=-1)  # [BS, N, 3D]

        hidden = torch.tanh(
            torch.einsum('bnd,ndh->bnh', mlp_in,
                         self.W1.float()) + self.b1.float())

        # Message modulation: per-neuron "broadcast identity"
        msg_mod = torch.tanh(
            torch.einsum('bnh,nhd->bnd', hidden,
                         self.W_msg.float()) + self.b_msg.float())

        # Modulator outputs
        mod_out = (torch.einsum('bnh,nho->bno', hidden,
                                self.W_mod.float()) + self.b_mod.float())
        decay_mod = mod_out[..., 0]
        plasticity_gate = torch.tanh(mod_out[..., 1])
        plasticity_lr = torch.sigmoid(mod_out[..., 2])

        # Effective decay = sigmoid(base + modulation)
        eff_decay = torch.sigmoid(self.decay_logit.unsqueeze(0) + decay_mod)

        return msg_mod.to(self.dtype), eff_decay.to(self.dtype), \
               plasticity_gate, plasticity_lr

    # ================================================================
    # Per-token recurrence (Python reference — Triton kernel replaces this)
    # ================================================================

    def _recurrence_python(self, h: Tensor, prev_msg: Tensor,
                           msg_mod: Tensor, eff_decay: Tensor,
                           cc_mem: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Python reference: 128-step recurrence with simple element-wise ops.

        This is what the Triton kernel replaces. Kept for CPU fallback and
        correctness testing.

        Returns:
            h: [BS, N, D] final hidden state
            prev_msg: [BS, N, D] final messages
            msg_all: [BS, T_seg, N, D] messages at every step (for readout)
        """
        BS, T_seg, C_mem, D = cc_mem.shape
        N = self.config.N_neurons
        stride = self.config.memory_update_stride

        inject_weights = torch.sigmoid(self.inject_w)  # [N, C_mem]
        routing = torch.sigmoid(self.w_conn)  # [N, K]

        msg_list = []
        for t in range(T_seg):
            if t % stride == 0:
                # Gather + weighted receive
                neighbor_msgs = prev_msg[:, self.conn_indices]
                weighted = routing.unsqueeze(0).unsqueeze(-1) * neighbor_msgs
                received = weighted.sum(dim=2)

                # Broadcast CC inject
                broadcast = torch.einsum(
                    'nc,bcd->bnd', inject_weights,
                    cc_mem[:, t].float()).to(self.dtype)
                received = received + broadcast

                # Integrate
                h = eff_decay.unsqueeze(-1) * h + \
                    (1 - eff_decay.unsqueeze(-1)) * received

                # Message with per-neuron modulation
                prev_msg = torch.tanh(h * msg_mod)

            msg_list.append(prev_msg)

        msg_all = torch.stack(msg_list, dim=1)  # [BS, T_seg, N, D]
        return h, prev_msg, msg_all

    # ================================================================
    # Forward segment
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment. MLP + recurrence + readout.

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

        # 1. Per-segment MLP (differentiable, runs once)
        msg_mod, eff_decay, p_gate, p_lr = self._segment_mlp(h)

        # 2. Per-token recurrence via custom autograd.Function
        # (saves only h_all + msg_all, recomputes intermediates in backward)
        if self.config.memory_update_stride == 1:
            h, prev_msg, msg_all = NeuronRecurrenceFunction.apply(
                h, prev_msg, msg_mod, eff_decay,
                self.w_conn, self.inject_w, cc_mem,
                self.conn_indices, T_seg)
        else:
            # Stride > 1: use Python reference (no custom backward)
            h, prev_msg, msg_all = self._recurrence_python(
                h, prev_msg, msg_mod, eff_decay, cc_mem)

        # 3. Readout (differentiable)
        readout_weights = torch.sigmoid(self.readout_w)  # [C_mem, N]
        output = torch.einsum(
            'cn,btnd->btcd', readout_weights,
            msg_all.float()).to(cc_signals.dtype)

        # Side effects (detached)
        with torch.no_grad():
            # Hebbian plasticity
            eta = torch.sigmoid(self.hebbian_lr_logit)
            neighbor_msgs = prev_msg.detach()[:, self.conn_indices]
            pre = neighbor_msgs.mean(dim=-1)
            post = prev_msg.detach().norm(dim=-1, keepdim=True)
            hebb = pre * post
            gate = (p_gate.detach() * p_lr.detach()).unsqueeze(-1)
            self.w_conn.data += (eta * gate * hebb).mean(dim=0).float()

            # Update traces
            td = 0.95
            self.trace_h = (td * self.trace_h + (1 - td) * h.detach()).to(self.dtype)
            self.trace_received = (td * self.trace_received +
                                   (1 - td) * prev_msg.detach()).to(self.dtype)

        # Update persistent state
        self.h = h.detach()
        self.prev_messages = prev_msg.detach()

        with torch.no_grad():
            alpha = 0.05
            seg_mag = prev_msg.detach().norm(dim=-1)
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
# Custom autograd.Function for the recurrence
# ====================================================================

class NeuronRecurrenceFunction(torch.autograd.Function):
    """Custom forward+backward for the per-token recurrence.

    Forward: runs Triton kernel (or Python fallback), saves h_all for backward.
    Backward: recomputes intermediates from h_all, accumulates param gradients.

    The recurrence is simple element-wise ops:
      received = sum(sigmoid(w_conn) * prev_msg[conn_idx]) + sigmoid(inject_w) @ cc_t
      h = eff_decay * h + (1-eff_decay) * received
      msg = tanh(h * msg_mod)
    """

    @staticmethod
    def forward(ctx, h, prev_msg, msg_mod, eff_decay,
                w_conn, inject_w, cc_mem, conn_indices, T_seg):
        """Forward recurrence — saves h and msg at each step."""
        BS, N, D = h.shape
        K = w_conn.shape[1]
        C_mem = inject_w.shape[1]
        device = h.device
        dtype = h.dtype

        routing = torch.sigmoid(w_conn)  # [N, K]
        inject_weights = torch.sigmoid(inject_w)  # [N, C_mem]

        h_all = torch.empty(BS, T_seg, N, D, device=device, dtype=dtype)
        msg_all = torch.empty(BS, T_seg, N, D, device=device, dtype=dtype)

        for t in range(T_seg):
            neighbor_msgs = prev_msg[:, conn_indices]
            weighted = routing.unsqueeze(0).unsqueeze(-1) * neighbor_msgs
            received = weighted.sum(dim=2)
            broadcast = torch.einsum(
                'nc,bcd->bnd', inject_weights,
                cc_mem[:, t].float()).to(dtype)
            received = received + broadcast

            h = eff_decay.unsqueeze(-1) * h + \
                (1 - eff_decay.unsqueeze(-1)) * received
            prev_msg = torch.tanh(h * msg_mod)

            h_all[:, t] = h
            msg_all[:, t] = prev_msg

        ctx.save_for_backward(h_all, msg_all, msg_mod, eff_decay,
                              w_conn, inject_w, cc_mem)
        ctx.conn_indices = conn_indices
        ctx.T_seg = T_seg

        return h, prev_msg, msg_all

    @staticmethod
    def backward(ctx, d_h_final, d_msg_final, d_msg_all):
        """Backward: reverse loop, recompute from saved h, accumulate grads."""
        h_all, msg_all, msg_mod, eff_decay, w_conn, inject_w, cc_mem = \
            ctx.saved_tensors
        conn_indices = ctx.conn_indices
        T_seg = ctx.T_seg

        BS, N, D = msg_mod.shape
        K = w_conn.shape[1]
        C_mem = inject_w.shape[1]
        dtype = h_all.dtype

        routing = torch.sigmoid(w_conn)
        inject_weights = torch.sigmoid(inject_w)

        # Gradient accumulators
        d_msg_mod = torch.zeros_like(msg_mod)
        d_eff_decay = torch.zeros(BS, N, device=msg_mod.device, dtype=torch.float32)
        d_w_conn = torch.zeros_like(w_conn, dtype=torch.float32)
        d_inject_w = torch.zeros_like(inject_w, dtype=torch.float32)
        d_cc_mem = torch.zeros_like(cc_mem, dtype=torch.float32)

        # d_h carries gradient backward through time
        d_h = d_h_final.float() if d_h_final is not None else torch.zeros(
            BS, N, D, device=msg_mod.device, dtype=torch.float32)

        for t in range(T_seg - 1, -1, -1):
            msg_t = msg_all[:, t]
            h_t = h_all[:, t]
            h_prev = h_all[:, t - 1] if t > 0 else torch.zeros_like(h_t)

            # Add gradient from readout path
            d_msg_t = d_msg_all[:, t].float()

            # Backward through msg = tanh(h * msg_mod)
            # d_pre_tanh = d_msg * (1 - msg²)
            msg_t_f = msg_t.float()
            d_pre_tanh = (d_msg_t) * (1 - msg_t_f * msg_t_f)
            d_h += d_pre_tanh * msg_mod.float()  # d_h from message
            d_msg_mod += (d_pre_tanh * h_t.float()).to(d_msg_mod.dtype)

            # Backward through h = eff_decay * h_prev + (1-eff_decay) * received
            d_h_prev = d_h * eff_decay.unsqueeze(-1).float()
            d_received = d_h * (1 - eff_decay.unsqueeze(-1).float())
            recv_recomp = _recompute_received(
                msg_all[:, t-1] if t > 0 else torch.zeros_like(msg_t),
                routing, inject_weights, cc_mem[:, t], conn_indices)
            d_eff_decay += (d_h * (h_prev.float() - recv_recomp)).sum(dim=-1)

            # Backward through received = sum(routing * neighbor_msgs) + inject @ cc
            # d_routing contribution
            prev_msg_t = msg_all[:, t - 1] if t > 0 else torch.zeros_like(msg_t)
            neighbor_msgs = prev_msg_t[:, conn_indices]  # [BS, N, K, D]
            d_weighted = d_received.unsqueeze(2).expand_as(neighbor_msgs)
            # d_w_conn: sum over BS and D
            d_routing = (d_weighted * neighbor_msgs.float()).sum(dim=-1).mean(dim=0)
            d_w_conn += d_routing * routing * (1 - routing)

            # d_inject_w
            d_broadcast = d_received  # [BS, N, D]
            d_inject_logit = torch.einsum(
                'bnd,bcd->nc', d_broadcast.float(),
                cc_mem[:, t].float()) / BS
            d_inject_w += d_inject_logit * inject_weights * (1 - inject_weights)

            # d_cc_mem
            d_cc_mem[:, t] = torch.einsum(
                'nc,bnd->bcd', inject_weights.float(), d_broadcast.float())

            # Propagate d_h to previous step
            # Also need scatter of d_received back to source neurons (gather backward)
            d_prev_msg_from_gather = torch.zeros(BS, N, D, device=d_h.device,
                                                  dtype=torch.float32)
            d_neighbor_weighted = d_weighted * routing.float().unsqueeze(0).unsqueeze(-1)
            # Scatter add: for each neuron n, connection k → source neuron conn_indices[n,k]
            for k_idx in range(K):
                src_neurons = conn_indices[:, k_idx]  # [N]
                d_prev_msg_from_gather.index_add_(
                    1, src_neurons,
                    d_neighbor_weighted[:, :, k_idx].float())

            d_h = d_h_prev
            # d_prev_msg_from_gather feeds into d_msg at step t-1 (handled next iter)
            if t > 0:
                d_msg_all_update = d_prev_msg_from_gather
                # This should be added to d_msg_all[:, t-1] for the next backward step
                # But d_msg_all is the upstream gradient from readout, already given
                # We need to add the "through the graph" gradient
                d_msg_all = d_msg_all.clone()
                d_msg_all[:, t-1] += d_prev_msg_from_gather.to(d_msg_all.dtype)

        return (d_h.to(dtype), None, d_msg_mod.to(dtype), d_eff_decay.to(dtype),
                d_w_conn, d_inject_w, d_cc_mem.to(cc_mem.dtype), None, None)


def _recompute_received(prev_msg, routing, inject_weights,
                        cc_t, conn_indices):
    """Helper: recompute received signal for decay gradient. All float32."""
    neighbor_msgs = prev_msg.float()[:, conn_indices]
    weighted = routing.float().unsqueeze(0).unsqueeze(-1) * neighbor_msgs
    received = weighted.sum(dim=2)
    broadcast = torch.einsum(
        'nc,bcd->bnd', inject_weights.float(), cc_t.float())
    return received + broadcast
