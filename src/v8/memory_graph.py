"""Memory Graph — differentiable 2-pass neuron simulation (v9-backprop).

Sparse graph topology and per-neuron segment modulator stay specialized, but
the hot token-time dynamics use a shared scan-friendly core so the expensive
work becomes dense GEMMs plus a fused scan over [BS*N, T, D_neuron].

2-pass simulation per segment (T tokens):
  Pass 1: ONE gather (frozen initial messages) → T steps of MLP dynamics
  Pass 2: ONE gather (from Pass 1 end state) → T steps refined

Per-step dynamics within each pass (frozen inter-neuron messages):
  1. input_vec = frozen_received + inject[t]  (inject varies per token)
  2. Shared state core predicts update_t from input/primitive/id/decay
  3. Fused scan applies h_t = decay * h_{t-1} + (1 - decay) * update_t
  4. Shared message core maps h_t + primitive + id → msg_t
  5. msg = msg + neuron_id

Segment-boundary modulator (runs FIRST, once per segment):
  mod(hebbian_traces, h, decay, primitive) → new w_conn, decay, primitive

Inject: H_mid [BS,T,D] → replicate → [BS,T,N,D_neuron]. No parameters.
Readout: msgs [BS,T,N,D_neuron] → mean over replicas → [BS,T,D].

Structural plasticity: at chunk boundaries, rewire weakest connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor

from ..model.scan import fused_scan
from .config import V8Config

# Try to import Triton kernels
try:
    from .triton_kernels import fused_dendritic_gather as _triton_gather
    from .triton_kernels import fused_shared_memory_pass_forward as _triton_shared_pass
    from .triton_kernels import fused_msg_readout_forward as _triton_msg_readout
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    _triton_shared_pass = None
    _triton_msg_readout = None


def _reference_shared_memory_pass(
    received: Tensor,
    cc_signals: Tensor,
    h0: Tensor,
    decay_logit: Tensor,
    primitives: Tensor,
    neuron_id: Tensor,
    state_w1: Tensor,
    state_b1: Tensor,
    state_w2: Tensor,
    state_b2: Tensor,
    msg_w1: Tensor,
    msg_b1: Tensor,
    msg_w2: Tensor,
    msg_b2: Tensor,
    n_per_slice: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """PyTorch reference for one shared-weight memory pass."""
    BS, N, D = received.shape
    T_seg = cc_signals.shape[1]
    D_lm = cc_signals.shape[2]
    C_mem = D_lm // D

    dt = received.dtype
    received_g = received.reshape(BS, C_mem, n_per_slice, D).to(dt)
    primitives_g = primitives.reshape(BS, C_mem, n_per_slice, D).to(dt)
    inject_grouped = cc_signals.reshape(BS, T_seg, C_mem, D).permute(0, 2, 1, 3)
    inject_grouped = inject_grouped.unsqueeze(2).expand(-1, -1, n_per_slice, -1, -1).to(dt)
    h0_flat = h0.reshape(BS * N, D).to(dt)
    decay_logit = decay_logit.to(dt)
    neuron_id_g = neuron_id.to(dt).reshape(C_mem, n_per_slice, D)

    w_in = state_w1[:, :D].to(dt)
    w_prim = state_w1[:, D:2 * D].to(dt)
    w_id = state_w1[:, 2 * D:3 * D].to(dt)
    w_decay = state_w1[:, 3 * D:].to(dt)
    state_b1 = state_b1.to(dt)
    state_w2 = state_w2.to(dt)
    state_b2 = state_b2.to(dt)

    base_hidden = (
        F.linear(received_g, w_in).unsqueeze(3) +
        F.linear(primitives_g, w_prim).unsqueeze(3) +
        F.linear(neuron_id_g, w_id).unsqueeze(0).unsqueeze(3) +
        F.linear(decay_logit.reshape(BS, C_mem, n_per_slice, 1), w_decay).unsqueeze(3) +
        state_b1.view(1, 1, 1, 1, -1)
    )
    hidden = torch.tanh(base_hidden + F.linear(inject_grouped, w_in))
    update = torch.tanh(F.linear(hidden, state_w2, state_b2))

    decay = torch.sigmoid(decay_logit).reshape(BS * N, 1, 1)
    b_flat = (1 - decay) * update.reshape(BS * N, T_seg, D)
    a_raw = decay_logit.reshape(BS * N, 1, 1).expand(-1, T_seg, D)
    h_seq = fused_scan(a_raw, b_flat, h0_flat).reshape(BS, C_mem, n_per_slice, T_seg, D)

    msg_w_h = msg_w1[:, :D].to(dt)
    msg_w_prim = msg_w1[:, D:2 * D].to(dt)
    msg_w_id = msg_w1[:, 2 * D:].to(dt)
    msg_b1 = msg_b1.to(dt)
    msg_w2 = msg_w2.to(dt)
    msg_b2 = msg_b2.to(dt)
    base_msg_hidden = (
        F.linear(primitives_g, msg_w_prim).unsqueeze(3) +
        F.linear(neuron_id_g, msg_w_id).unsqueeze(0).unsqueeze(3) +
        msg_b1.view(1, 1, 1, 1, -1)
    )
    msg_hidden = torch.tanh(base_msg_hidden + F.linear(h_seq, msg_w_h))
    msg_seq = torch.tanh(F.linear(msg_hidden, msg_w2, msg_b2))
    msg_seq = msg_seq + neuron_id_g.view(1, C_mem, n_per_slice, 1, D)

    h_last = h_seq[:, :, :, -1, :].reshape(BS, N, D)
    msg_last = msg_seq[:, :, :, -1, :].reshape(BS, N, D)
    mem_out = msg_seq.sum(dim=2) * (n_per_slice ** -0.5)
    mem_out = mem_out.permute(0, 2, 1, 3).reshape(BS, T_seg, D_lm).to(torch.float32)
    return h_last, msg_last, mem_out


class _FusedSharedMemoryPassFn(torch.autograd.Function):
    """Autograd wrapper: Triton forward, PyTorch recompute backward."""

    @staticmethod
    def forward(ctx, received, cc_signals, h0, decay_logit, primitives,
                neuron_id, state_w1, state_b1, state_w2, state_b2,
                msg_w1, msg_b1, msg_w2, msg_b2, n_per_slice, write_readout):
        dt = received.dtype
        h_last, msg_last, mem_out, msg_norm_mean, act_trace = _triton_shared_pass(
            received,
            cc_signals,
            h0,
            decay_logit,
            primitives,
            neuron_id.to(dt),
            state_w1.to(dt),
            state_b1.to(dt),
            state_w2.t().contiguous().to(dt),
            state_b2.to(dt),
            msg_w1.to(dt),
            msg_b1.to(dt),
            msg_w2.t().contiguous().to(dt),
            msg_b2.to(dt),
            n_per_slice,
            write_readout,
        )

        ctx.save_for_backward(
            received, cc_signals, h0, decay_logit, primitives, neuron_id,
            state_w1, state_b1, state_w2, state_b2,
            msg_w1, msg_b1, msg_w2, msg_b2,
        )
        ctx.n_per_slice = n_per_slice
        ctx.write_readout = write_readout
        return h_last, msg_last, mem_out, msg_norm_mean, act_trace

    @staticmethod
    def backward(ctx, grad_h_last, grad_msg_last, grad_mem_out,
                 grad_msg_norm_mean, grad_act_trace):
        saved = ctx.saved_tensors
        (received, cc_signals, h0, decay_logit, primitives, neuron_id,
         state_w1, state_b1, state_w2, state_b2,
         msg_w1, msg_b1, msg_w2, msg_b2) = saved

        inputs = [
            received.detach().requires_grad_(ctx.needs_input_grad[0]),
            cc_signals.detach().requires_grad_(ctx.needs_input_grad[1]),
            h0.detach().requires_grad_(ctx.needs_input_grad[2]),
            decay_logit.detach().requires_grad_(ctx.needs_input_grad[3]),
            primitives.detach().requires_grad_(ctx.needs_input_grad[4]),
            neuron_id.detach().requires_grad_(ctx.needs_input_grad[5]),
            state_w1.detach().requires_grad_(ctx.needs_input_grad[6]),
            state_b1.detach().requires_grad_(ctx.needs_input_grad[7]),
            state_w2.detach().requires_grad_(ctx.needs_input_grad[8]),
            state_b2.detach().requires_grad_(ctx.needs_input_grad[9]),
            msg_w1.detach().requires_grad_(ctx.needs_input_grad[10]),
            msg_b1.detach().requires_grad_(ctx.needs_input_grad[11]),
            msg_w2.detach().requires_grad_(ctx.needs_input_grad[12]),
            msg_b2.detach().requires_grad_(ctx.needs_input_grad[13]),
        ]

        with torch.enable_grad():
            ref_h_last, ref_msg_last, ref_mem_out = _reference_shared_memory_pass(
                *inputs, ctx.n_per_slice
            )

        grad_outputs = (
            grad_h_last.to(ref_h_last.dtype),
            grad_msg_last.to(ref_msg_last.dtype),
            grad_mem_out.to(ref_mem_out.dtype),
        )
        grad_input_indices = [
            i for i, tensor in enumerate(inputs) if tensor.requires_grad
        ]
        grad_input_tensors = [inputs[i] for i in grad_input_indices]
        computed_grads = torch.autograd.grad(
            outputs=(ref_h_last, ref_msg_last, ref_mem_out),
            inputs=grad_input_tensors,
            grad_outputs=grad_outputs,
            allow_unused=True,
        )
        grads = [None] * len(inputs)
        for idx, grad in zip(grad_input_indices, computed_grads):
            grads[idx] = grad

        return (*grads, None, None)


class _FusedMsgReadoutFn(torch.autograd.Function):
    """Autograd wrapper: Triton readout forward, analytic backward."""

    @staticmethod
    def forward(ctx, msg_chunk: Tensor, n_per_slice: int):
        ctx.msg_shape = tuple(msg_chunk.shape)
        ctx.msg_dtype = msg_chunk.dtype
        ctx.n_per_slice = n_per_slice
        return _triton_msg_readout(msg_chunk, n_per_slice)

    @staticmethod
    def backward(ctx, grad_mem_out: Tensor):
        BS, C_mem, N_per_slice, T_chunk, D = ctx.msg_shape
        scale = ctx.n_per_slice ** -0.5
        grad = grad_mem_out.reshape(BS, T_chunk, C_mem, D).permute(0, 2, 1, 3)
        grad = grad.unsqueeze(2).expand(-1, -1, N_per_slice, -1, -1)
        return (grad.to(ctx.msg_dtype) * scale).contiguous(), None


class MemoryGraph(nn.Module):
    """Differentiable memory graph with per-neuron modulator + shared scan core.

    nn.Parameters (requires_grad=True, trained by backprop):
        mod_w1, mod_b1, mod_w2, mod_b2 — segment-boundary modulator
        state_w1, state_b1, state_w2, state_b2 — per-step state core
        msg_w1, msg_b1, msg_w2, msg_b2 — per-step message core
        neuron_id — learnable per-neuron identity embedding
        dendrite_branch_w, dendrite_group_w — dendritic tree FC weights

    Runtime state (per-batch, set by modulator, NOT learned directly):
        h, prev_messages, w_conn, primitives_state, decay_logit
        hebbian_traces — per-segment average of |msg| * sigmoid(w_conn)
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K = config.K_connections
        D = config.D_neuron

        # Fixed sparse topology
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')
        K_actual = min(K, N - 1)
        conn_indices = torch.zeros(N, K, dtype=torch.long, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        sorted_idx, _ = conn_indices.sort(dim=-1)
        self.register_buffer('conn_indices', sorted_idx)

        # Dendritic tree structure
        branch_size = config.dendrite_branch_size
        if branch_size > 0 and K >= branch_size:
            self.n_branches = K // branch_size
            self.branch_size = branch_size
            self.branches_per_group = min(4, self.n_branches)
            self.n_groups = max(1, self.n_branches // self.branches_per_group)
            self.branches_per_group = self.n_branches // self.n_groups
            self.use_dendritic_tree = True
        else:
            self.use_dendritic_tree = False

        # ================================================================
        # Learned parameters (backprop-trained)
        # ================================================================

        # --- Segment-boundary modulator MLP ---
        # Input: hebbian[K] + h[D] + decay[1] + primitive[D]
        mod_input_dim = K + 2 * D + 1
        # Output: new w_conn[K] + new decay[1] + new primitive[D]
        mod_output_dim = K + 1 + D
        H_mod = config.neuromod_hidden

        # w1 layouts are [N, H, I] (transposed) for contiguous Triton access
        self.mod_w1 = nn.Parameter(
            torch.randn(N, H_mod, mod_input_dim, device=device) *
            (2.0 / (mod_input_dim + H_mod)) ** 0.5)
        self.mod_b1 = nn.Parameter(torch.zeros(N, H_mod, device=device))
        # All weight matrices use Xavier/Glorot: std = sqrt(2 / (fan_in + fan_out))
        # This ensures signals neither vanish nor explode through the network.
        self.mod_w2 = nn.Parameter(
            torch.randn(N, H_mod, mod_output_dim, device=device) *
            (2.0 / (H_mod + mod_output_dim)) ** 0.5)
        self.mod_b2 = nn.Parameter(torch.zeros(N, mod_output_dim, device=device))

        # --- Per-step state update core (shared weights, scan-friendly) ---
        # Input contributions: input_vec[D] + primitive[D] + neuron_id[D] + decay[1]
        # state_w1 keeps the historical name for tests/diagnostics.
        state_in = 3 * D + 1
        H_state = config.state_mlp_hidden
        self.state_w1 = nn.Parameter(
            torch.randn(H_state, state_in, device=device) *
            (2.0 / (state_in + H_state)) ** 0.5)
        self.state_b1 = nn.Parameter(torch.zeros(H_state, device=device))
        self.state_w2 = nn.Parameter(
            torch.randn(D, H_state, device=device) *
            (2.0 / (H_state + D)) ** 0.5)
        self.state_b2 = nn.Parameter(torch.zeros(D, device=device))

        # --- Per-step message core (shared weights, identity-conditioned) ---
        # Input contributions: h_new[D] + primitive[D] + neuron_id[D]
        H_msg = config.msg_mlp_hidden
        self.msg_w1 = nn.Parameter(
            torch.randn(H_msg, 3 * D, device=device) *
            (2.0 / (3 * D + H_msg)) ** 0.5)
        self.msg_b1 = nn.Parameter(torch.zeros(H_msg, device=device))
        self.msg_w2 = nn.Parameter(
            torch.randn(D, H_msg, device=device) *
            (2.0 / (H_msg + D)) ** 0.5)
        self.msg_b2 = nn.Parameter(torch.zeros(D, device=device))

        # --- Neuron ID embedding ---
        # Scale similar to positional embeddings in transformers
        self.neuron_id = nn.Parameter(
            torch.randn(N, D, device=device) * (1.0 / D ** 0.5))

        # --- Dendritic FC weights ---
        if self.use_dendritic_tree:
            nb, bs = self.n_branches, self.branch_size
            ng, bpg = self.n_groups, self.branches_per_group
            # Mean = uniform average, noise breaks symmetry across branches/dims
            self.dendrite_branch_w = nn.Parameter(
                torch.full((N, nb, bs, D), 1.0 / bs, device=device)
                + torch.randn(N, nb, bs, D, device=device) * (0.1 / bs))
            self.dendrite_group_w = nn.Parameter(
                torch.full((N, ng, bpg, D), 1.0 / bpg, device=device)
                + torch.randn(N, ng, bpg, D, device=device) * (0.1 / bpg))

        # Inject/readout constants
        self.C_mem = config.C_mem
        self.N_per_slice = config.N_per_slice

        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        """Initialize runtime state for batch size BS."""
        device = self.mod_w1.device
        N, D = self.config.N_neurons, self.config.D_neuron
        K = self.config.K_connections
        dt = self.dtype

        self.h = torch.randn(BS, N, D, device=device, dtype=dt) * 0.1
        self.prev_messages = torch.zeros(BS, N, D, device=device, dtype=dt)

        # Neuron properties (set by modulator, not directly learned)
        self.w_conn = torch.zeros(BS, N, K, device=device, dtype=dt)
        self.primitives_state = torch.zeros(BS, N, D, device=device, dtype=dt)
        self.decay_logit = torch.zeros(BS, N, device=device, dtype=dt)

        # Hebbian traces (per-segment average of |msg| * sigmoid(w_conn))
        self.hebbian_traces = torch.zeros(BS, N, K, device=device, dtype=dt)

        # Structural plasticity — phi correlation matrix (EMA-smoothed)
        if self.config.structural_plasticity:
            self.co_activation_ema = torch.zeros(N, N, device=device,
                                                  dtype=torch.float32)
            self._co_activation_ready = False

        # Diagnostics
        self.msg_magnitude = torch.zeros(BS, N, device=device, dtype=dt)

        self._initialized = True

    # ================================================================
    # Per-neuron modulator (segment boundary)
    # ================================================================

    def _run_modulator(self, h: Tensor, hebbian_traces: Tensor,
                       decay_logit: Tensor,
                       primitives: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Segment-boundary modulator: predicts new w_conn, decay, primitives.

        Runs FIRST each segment so we observe the modulator's effects.
        """
        K = self.config.K_connections
        D = self.config.D_neuron

        idt = h.dtype  # bf16 (runtime state dtype)
        mod_input = torch.cat([
            hebbian_traces,             # [BS, N, K]
            h,                          # [BS, N, D]
            decay_logit.unsqueeze(-1),  # [BS, N, 1]
            primitives,                 # [BS, N, D]
        ], dim=-1)  # [BS, N, K+2D+1]

        # Per-neuron MLP: cast weights to input dtype (bf16) for einsum.
        # Gradients flow through the cast back to f32 parameters.
        hidden = torch.einsum(
            'bni,nhi->bnh', mod_input, self.mod_w1.to(idt)
        ) + self.mod_b1.to(idt)  # [BS, N, H]
        hidden = torch.tanh(hidden)

        output = torch.einsum(
            'bnh,nho->bno', hidden, self.mod_w2.to(idt)
        ) + self.mod_b2.to(idt)  # [BS, N, K+1+D]

        new_w_conn = output[..., :K]              # [BS, N, K]
        new_decay_logit = output[..., K]           # [BS, N]
        new_primitives = output[..., K + 1:]       # [BS, N, D]

        return new_w_conn, new_decay_logit, new_primitives

    # ================================================================
    # Per-step MLPs
    # ================================================================

    def _state_mlp(self, input_vec: Tensor, h_prev: Tensor,
                   decay: Tensor, primitives: Tensor | None = None) -> Tensor:
        """Single-step wrapper around the shared scan-friendly state core."""
        D = self.config.D_neuron
        dt = input_vec.dtype
        if primitives is None:
            primitives = torch.zeros_like(input_vec)

        state_w1 = self.state_w1.to(dt)
        state_w2 = self.state_w2.to(dt)
        state_b1 = self.state_b1.to(dt)
        state_b2 = self.state_b2.to(dt)
        neuron_id = self.neuron_id.to(dt)

        w_in = state_w1[:, :D]
        w_prim = state_w1[:, D:2 * D]
        w_id = state_w1[:, 2 * D:3 * D]
        w_decay = state_w1[:, 3 * D:]

        hidden = (
            F.linear(input_vec, w_in) +
            F.linear(primitives, w_prim) +
            F.linear(neuron_id, w_id).unsqueeze(0) +
            F.linear(decay.unsqueeze(-1), w_decay) +
            state_b1
        )
        update = torch.tanh(
            F.linear(torch.tanh(hidden), state_w2, state_b2)
        )
        d = decay.unsqueeze(-1)
        return d * h_prev + (1 - d) * update

    def _msg_mlp(self, h_new: Tensor, primitives: Tensor) -> Tensor:
        """Single-step wrapper around the shared identity-conditioned msg core."""
        D = self.config.D_neuron
        dt = h_new.dtype
        msg_w1 = self.msg_w1.to(dt)
        msg_w2 = self.msg_w2.to(dt)
        msg_b1 = self.msg_b1.to(dt)
        msg_b2 = self.msg_b2.to(dt)
        neuron_id = self.neuron_id.to(dt)

        w_h = msg_w1[:, :D]
        w_prim = msg_w1[:, D:2 * D]
        w_id = msg_w1[:, 2 * D:]
        hidden = (
            F.linear(h_new, w_h) +
            F.linear(primitives, w_prim) +
            F.linear(neuron_id, w_id).unsqueeze(0) +
            msg_b1
        )
        return torch.tanh(
            F.linear(torch.tanh(hidden), msg_w2, msg_b2)
        )

    # ================================================================
    # Dendritic gather
    # ================================================================

    def _dendritic_gather(self, weighted: Tensor) -> Tensor:
        """Dendritic tree with per-neuron FC at branch and group levels.

        Python reference implementation. Used when Triton is not available.
        """
        BS, N, _, D = weighted.shape
        K = self.config.K_connections
        bsz, bpg = self.branch_size, self.branches_per_group
        ng, nb = self.n_groups, self.n_branches

        n_tree = ng * bpg * bsz
        tree_msgs = weighted[:, :, :n_tree].view(BS, N, nb, bsz, D)
        branch_out = torch.tanh(
            (tree_msgs * self.dendrite_branch_w.unsqueeze(0)).sum(dim=3))
        branch_grouped = branch_out.view(BS, N, ng, bpg, D)
        group_out = torch.tanh(
            (branch_grouped * self.dendrite_group_w.unsqueeze(0)).sum(dim=3))
        received = group_out.mean(dim=2)

        if n_tree < K:
            leftover = weighted[:, :, n_tree:].sum(dim=2)
            tree_frac = n_tree / K
            received = tree_frac * received + (1 - tree_frac) * leftover
        return received

    def _fused_gather(self, prev_msg: Tensor,
                      w_conn_sig: Tensor) -> Tensor:
        """Fused gather + weight + dendritic tree via Triton.

        Eliminates the [BS, N, K, D] intermediate tensor.
        Falls back to Python if Triton unavailable.
        """
        if _HAS_TRITON and prev_msg.is_cuda:
            branch_w = self.dendrite_branch_w if self.use_dendritic_tree else None
            group_w = self.dendrite_group_w if self.use_dendritic_tree else None
            bsz = self.branch_size if self.use_dendritic_tree else 1
            bpg = self.branches_per_group if self.use_dendritic_tree else 1
            ng = self.n_groups if self.use_dendritic_tree else 1
            return _triton_gather(
                prev_msg, self.conn_indices, w_conn_sig,
                branch_w, group_w,
                bsz, bpg, ng,
                self.use_dendritic_tree,
            )

        # Python fallback
        neighbor_msgs = prev_msg[:, self.conn_indices]  # [BS, N, K, D]
        weighted = w_conn_sig.unsqueeze(-1) * neighbor_msgs
        if self.use_dendritic_tree:
            return self._dendritic_gather(weighted)
        else:
            return weighted.sum(dim=2)

    # ================================================================
    # Inject / Readout (parameter-free)
    # ================================================================

    def _inject_single(self, H_mid_t: Tensor) -> Tensor:
        """Single-token inject: [BS, D_lm] → [BS, N, D_neuron].

        Avoids materializing the full [BS, T, N, D] tensor (saves ~1.6 GB
        at BS=48). Called per-step inside the token loop.
        """
        BS = H_mid_t.shape[0]
        slices = H_mid_t.view(BS, self.C_mem, self.config.D_neuron)
        return slices.unsqueeze(2).expand(
            -1, -1, self.N_per_slice, -1
        ).reshape(BS, self.config.N_neurons, self.config.D_neuron)

    def _inject_grouped(self, H_mid: Tensor) -> Tensor:
        """All-token inject as a grouped view: [BS, C_mem, Nps, T, D_neuron]."""
        BS, T_seg, _ = H_mid.shape
        grouped = H_mid.view(BS, T_seg, self.C_mem, self.config.D_neuron)
        grouped = grouped.permute(0, 2, 1, 3)  # [BS, C_mem, T, D_neuron]
        return grouped.unsqueeze(2).expand(-1, -1, self.N_per_slice, -1, -1)

    def _readout_single(self, msg: Tensor) -> Tensor:
        """Single-token readout: [BS, N, D_neuron] → [BS, D_lm].

        Avoids materializing the full [BS, T, N, D] msg_all tensor
        (saves ~1.6 GB at BS=48). Called per-step inside the token loop.
        """
        BS = msg.shape[0]
        grouped = msg.view(
            BS, self.C_mem, self.N_per_slice, self.config.D_neuron)
        scaled = grouped.sum(dim=2) * (self.N_per_slice ** -0.5)
        return scaled.reshape(BS, self.config.D)

    def _readout_all(self, msg_seq: Tensor) -> Tensor:
        """Vectorized readout: [BS, C_mem, Nps, T, D_neuron] → [BS, T, D_lm]."""
        BS, _, _, T_seg, _ = msg_seq.shape
        scaled = msg_seq.sum(dim=2) * (self.N_per_slice ** -0.5)
        return scaled.permute(0, 2, 1, 3).reshape(BS, T_seg, self.config.D)

    def _run_state_scan(
        self,
        received: Tensor,
        inject_grouped: Tensor,
        h0: Tensor,
        decay_logit: Tensor,
        primitives: Tensor,
    ) -> Tensor:
        """Run one pass of the shared state core with fused scan.

        Args:
            received: [BS, N, D]
            inject_grouped: [BS, C_mem, Nps, T, D]
            h0: [BS, N, D]
            decay_logit: [BS, N]
            primitives: [BS, N, D]

        Returns:
            h_seq: [BS, C_mem, Nps, T, D]
        """
        BS, _, _, T_seg, D = inject_grouped.shape
        H_state = self.config.state_mlp_hidden
        B_flat = BS * self.config.N_neurons
        dt = received.dtype

        received_g = received.reshape(BS, self.C_mem, self.N_per_slice, D).to(dt)
        primitives_g = primitives.reshape(BS, self.C_mem, self.N_per_slice, D).to(dt)
        inject_grouped = inject_grouped.to(dt)
        h0_flat = h0.to(dt).reshape(B_flat, D)
        decay_logit = decay_logit.to(dt)

        state_w1 = self.state_w1.to(dt)
        state_w2 = self.state_w2.to(dt)
        state_b1 = self.state_b1.to(dt)
        state_b2 = self.state_b2.to(dt)
        neuron_id = self.neuron_id.to(dt).view(self.C_mem, self.N_per_slice, D)

        w_in = state_w1[:, :D]
        w_prim = state_w1[:, D:2 * D]
        w_id = state_w1[:, 2 * D:3 * D]
        w_decay = state_w1[:, 3 * D:]

        base_hidden = (
            F.linear(received_g, w_in).unsqueeze(3) +
            F.linear(primitives_g, w_prim).unsqueeze(3) +
            F.linear(neuron_id, w_id).unsqueeze(0).unsqueeze(3) +
            F.linear(
                decay_logit.reshape(BS, self.C_mem, self.N_per_slice, 1),
                w_decay,
            ).unsqueeze(3) +
            state_b1.view(1, 1, 1, 1, H_state)
        )

        inject_hidden = F.linear(inject_grouped, w_in)
        hidden = torch.tanh(base_hidden + inject_hidden)
        update = torch.tanh(F.linear(hidden, state_w2, state_b2))

        decay = torch.sigmoid(decay_logit).reshape(B_flat, 1, 1)
        b_flat = (1 - decay) * update.reshape(B_flat, T_seg, D)
        a_raw = decay_logit.reshape(B_flat, 1, 1).expand(-1, T_seg, D)
        h_seq = self._scan_in_chunks(a_raw, b_flat, h0_flat)
        return h_seq.view(BS, self.C_mem, self.N_per_slice, T_seg, D)

    def _run_msg_core(self, h_seq: Tensor, primitives: Tensor) -> Tensor:
        """Vectorized message core over a full pass.

        Args:
            h_seq: [BS, C_mem, Nps, T, D]
            primitives: [BS, N, D]

        Returns:
            msg_seq: [BS, C_mem, Nps, T, D]
        """
        BS, _, _, _, D = h_seq.shape
        H_msg = self.config.msg_mlp_hidden
        dt = h_seq.dtype
        primitives_g = primitives.reshape(BS, self.C_mem, self.N_per_slice, D).to(dt)

        msg_w1 = self.msg_w1.to(dt)
        msg_w2 = self.msg_w2.to(dt)
        msg_b1 = self.msg_b1.to(dt)
        msg_b2 = self.msg_b2.to(dt)
        neuron_id = self.neuron_id.to(dt).view(self.C_mem, self.N_per_slice, D)

        w_h = msg_w1[:, :D]
        w_prim = msg_w1[:, D:2 * D]
        w_id = msg_w1[:, 2 * D:]

        base_hidden = (
            F.linear(primitives_g, w_prim).unsqueeze(3) +
            F.linear(neuron_id, w_id).unsqueeze(0).unsqueeze(3) +
            msg_b1.view(1, 1, 1, 1, H_msg)
        )
        hidden = torch.tanh(base_hidden + F.linear(h_seq, w_h))
        msg = torch.tanh(F.linear(hidden, msg_w2, msg_b2))
        return msg + neuron_id.view(1, self.C_mem, self.N_per_slice, 1, D)

    def _run_msg_core_last(self, h_last: Tensor, primitives: Tensor) -> Tensor:
        """Message core on the final scan state only.

        Args:
            h_last: [BS, C_mem, Nps, D]
            primitives: [BS, N, D]

        Returns:
            msg_last: [BS, C_mem, Nps, D]
        """
        BS, _, _, D = h_last.shape
        H_msg = self.config.msg_mlp_hidden
        dt = h_last.dtype
        primitives_g = primitives.reshape(BS, self.C_mem, self.N_per_slice, D).to(dt)

        msg_w1 = self.msg_w1.to(dt)
        msg_w2 = self.msg_w2.to(dt)
        msg_b1 = self.msg_b1.to(dt)
        msg_b2 = self.msg_b2.to(dt)
        neuron_id = self.neuron_id.to(dt).view(self.C_mem, self.N_per_slice, D)

        w_h = msg_w1[:, :D]
        w_prim = msg_w1[:, D:2 * D]
        w_id = msg_w1[:, 2 * D:]

        base_hidden = (
            F.linear(primitives_g, w_prim) +
            F.linear(neuron_id, w_id).unsqueeze(0) +
            msg_b1.view(1, 1, 1, H_msg)
        )
        hidden = torch.tanh(base_hidden + F.linear(h_last, w_h))
        msg = torch.tanh(F.linear(hidden, msg_w2, msg_b2))
        return msg + neuron_id.view(1, self.C_mem, self.N_per_slice, D)

    def _readout_chunk(self, msg_chunk: Tensor, use_triton: bool) -> Tensor:
        """Read out one message chunk to [BS, T_chunk, D_lm]."""
        if use_triton:
            return _FusedMsgReadoutFn.apply(msg_chunk, self.N_per_slice)
        return self._readout_all(msg_chunk)

    def _scan_in_chunks(self, a_raw: Tensor, b: Tensor, h0: Tensor) -> Tensor:
        """Run fused scan in manageable flattened-batch chunks."""
        B_flat = a_raw.shape[0]
        if not a_raw.is_cuda or B_flat <= 16384:
            return fused_scan(a_raw, b, h0)

        outs = []
        for start in range(0, B_flat, 16384):
            end = min(start + 16384, B_flat)
            outs.append(fused_scan(a_raw[start:end], b[start:end], h0[start:end]))
        return torch.cat(outs, dim=0)

    def _run_pass_core(
        self,
        received: Tensor,
        inject_grouped: Tensor,
        h: Tensor,
        decay_logit: Tensor,
        primitives: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one memory pass and return only the recurrent carry."""
        BS = h.shape[0]
        N = self.config.N_neurons
        D = self.config.D_neuron
        h_seq = self._run_state_scan(received, inject_grouped, h, decay_logit, primitives)
        h_last = h_seq[:, :, :, -1, :].reshape(BS, N, D)
        msg_last = self._run_msg_core_last(h_seq[:, :, :, -1, :], primitives)
        prev_msg = msg_last.reshape(BS, N, D)
        return h_last, prev_msg

    def _run_pass_with_readout(
        self,
        received: Tensor,
        inject_grouped: Tensor,
        h: Tensor,
        decay_logit: Tensor,
        primitives: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one memory pass and keep the final-token readout/activity."""
        BS = h.shape[0]
        T_seg = inject_grouped.shape[3]
        N = self.config.N_neurons
        D = self.config.D_neuron
        h_seq = self._run_state_scan(received, inject_grouped, h, decay_logit, primitives)
        msg_seq = self._run_msg_core(h_seq, primitives)
        h_last = h_seq[:, :, :, -1, :].reshape(BS, N, D)
        prev_msg = msg_seq[:, :, :, -1, :].reshape(BS, N, D)
        mem_out = self._readout_all(msg_seq)
        act_trace = msg_seq.norm(dim=-1).permute(0, 3, 1, 2).reshape(BS, T_seg, N)
        msg_norm_mean = act_trace.mean(dim=1)
        return h_last, prev_msg, mem_out, act_trace, msg_norm_mean

    def _run_pass_with_readout_chunked(
        self,
        received: Tensor,
        inject_grouped: Tensor,
        h: Tensor,
        decay_logit: Tensor,
        primitives: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run one pass with chunked message core and fused Triton readout."""
        BS = h.shape[0]
        T_seg = inject_grouped.shape[3]
        N = self.config.N_neurons
        D = self.config.D_neuron
        h_seq = self._run_state_scan(received, inject_grouped, h, decay_logit, primitives)
        h_last = h_seq[:, :, :, -1, :].reshape(BS, N, D)

        use_triton = (
            _HAS_TRITON and
            _triton_msg_readout is not None and
            h_seq.is_cuda
        )
        chunk_len = min(32, T_seg)
        mem_out_chunks = []
        act_chunks = []
        msg_norm_sum = torch.zeros(BS, N, device=h_seq.device, dtype=torch.float32)
        prev_msg = None

        for start in range(0, T_seg, chunk_len):
            end = min(start + chunk_len, T_seg)
            h_chunk = h_seq[:, :, :, start:end, :]
            msg_chunk = self._run_msg_core(h_chunk, primitives)
            mem_out_chunks.append(self._readout_chunk(msg_chunk, use_triton))

            with torch.no_grad():
                act_chunk = msg_chunk.norm(dim=-1).permute(0, 3, 1, 2)
                act_chunk = act_chunk.reshape(BS, end - start, N).float()
                msg_norm_sum += act_chunk.sum(dim=1)
                if self.config.structural_plasticity:
                    act_chunks.append(act_chunk)

            if end == T_seg:
                prev_msg = msg_chunk[:, :, :, -1, :].reshape(BS, N, D)

        mem_out = torch.cat(mem_out_chunks, dim=1)
        if self.config.structural_plasticity:
            act_trace = torch.cat(act_chunks, dim=1)
        else:
            act_trace = torch.empty(BS, 0, N, device=h_seq.device, dtype=torch.float32)
        msg_norm_mean = msg_norm_sum / max(T_seg, 1)
        return h_last, prev_msg, mem_out, act_trace, msg_norm_mean

    # ================================================================
    # Forward segment (differentiable)
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment of tokens with 2-pass sparse gather + scan core.

        Args:
            cc_signals: [BS, T_seg, D_lm] — detached H_mid for this segment

        Returns:
            mem_out: [BS, T_seg, D_lm] — readout signal to inject into LM
        """
        BS = cc_signals.shape[0]
        T_seg = cc_signals.shape[1]
        N = self.config.N_neurons
        K = self.config.K_connections

        # TBPTT: detach state at segment boundary
        h = self.h.detach().to(self.dtype)
        prev_msg = self.prev_messages.detach().to(self.dtype)
        cc_signals = cc_signals.to(h.dtype)

        # Run modulator FIRST (on compute graph — this is what backprop trains)
        w_conn, decay_logit, primitives = self._run_modulator(
            h, self.hebbian_traces.detach(),
            self.decay_logit.detach(),
            self.primitives_state.detach())

        # Precompute sigmoid (on graph through w_conn)
        w_conn_sig = torch.sigmoid(w_conn)      # [BS, N, K]

        inject_grouped = self._inject_grouped(cc_signals)
        act_trace = None
        mem_out = None
        use_triton_readout = (
            self.config.experimental_triton_pass and
            _HAS_TRITON and
            _triton_msg_readout is not None and
            prev_msg.is_cuda
        )
        use_checkpoint = (
            self.training and inject_grouped.is_cuda and BS * N > 16384
        )

        # Both passes stay on the autograd graph — pass 2's gradient flows
        # back through pass 1's prev_msg to the modulator through prev_msg.
        n_passes = 2

        for pass_idx in range(n_passes):
            received = self._fused_gather(prev_msg, w_conn_sig)  # [BS, N, D]

            if pass_idx == n_passes - 1:
                pass_fn = (
                    self._run_pass_with_readout_chunked
                    if use_triton_readout else
                    self._run_pass_with_readout
                )
                if use_checkpoint:
                    h, prev_msg, mem_out, act_trace, msg_norm_mean = checkpoint(
                        pass_fn,
                        received, inject_grouped, h, decay_logit, primitives,
                        use_reentrant=False,
                    )
                else:
                    h, prev_msg, mem_out, act_trace, msg_norm_mean = pass_fn(
                        received, inject_grouped, h, decay_logit, primitives)
                with torch.no_grad():
                    total_hebbian = msg_norm_mean.unsqueeze(-1) * w_conn_sig.detach()
            else:
                if use_checkpoint:
                    h, prev_msg = checkpoint(
                        self._run_pass_core,
                        received, inject_grouped, h, decay_logit, primitives,
                        use_reentrant=False,
                    )
                else:
                    h, prev_msg = self._run_pass_core(
                        received, inject_grouped, h, decay_logit, primitives)

        # Update persistent state (detached, for next segment)
        with torch.no_grad():
            self.h = h.detach().to(self.dtype)
            self.prev_messages = prev_msg.detach().to(self.dtype)
            self.w_conn = w_conn.detach().to(self.dtype)
            self.primitives_state = primitives.detach().to(self.dtype)
            self.decay_logit = decay_logit.detach().to(self.dtype)

            # Hebbian traces: per-segment average
            self.hebbian_traces = total_hebbian.to(self.dtype)

            # Structural plasticity: phi correlation from per-step activity
            if (self.config.structural_plasticity and
                    hasattr(self, 'co_activation_ema') and act_trace is not None and
                    act_trace.numel() > 0):
                self._update_phi(act_trace)

            # Diagnostics
            alpha = 0.05
            seg_mag = self.prev_messages.norm(dim=-1)
            self.msg_magnitude = (
                (1 - alpha) * self.msg_magnitude + alpha * seg_mag
            ).to(self.dtype)

        return mem_out

    # ================================================================
    # Structural plasticity
    # ================================================================

    @torch.no_grad()
    def _update_phi(self, act_trace: Tensor):
        """Compute Pearson phi from per-step activity traces, EMA-smooth.

        Args:
            act_trace: [BS, T_seg, N] — per-step message norms from last pass
        """
        BS, T_seg, N = act_trace.shape

        # Binary firing: 1 if above 75th percentile (per batch element)
        threshold = torch.quantile(
            act_trace, 0.75, dim=1, keepdim=True)  # [BS, 1, N]
        fired = (act_trace > threshold).float()       # [BS, T_seg, N]

        # Pearson correlation (phi coefficient)
        p_i = fired.mean(dim=1, keepdim=True)         # [BS, 1, N]
        fired_centered = fired - p_i                   # [BS, T_seg, N]
        var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)  # [BS, N]
        cov = torch.bmm(
            fired_centered.transpose(1, 2),
            fired_centered,
        ) / T_seg                                      # [BS, N, N]
        std_i = var_i.sqrt().unsqueeze(2)              # [BS, N, 1]
        std_j = var_i.sqrt().unsqueeze(1)              # [BS, 1, N]
        phi = cov / (std_i * std_j).clamp(min=1e-8)   # [BS, N, N]

        # Average over batch, EMA update
        phi_mean = phi.mean(dim=0)  # [N, N]
        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = (
            ca_decay * self.co_activation_ema + (1 - ca_decay) * phi_mean)
        self._co_activation_ready = True

    def rewire_connections(self):
        """Structural plasticity: globally prune/grow connections by phi rank.

        Prunes the bottom plasticity_pct% of existing connections (lowest phi)
        and creates new connections for the top plasticity_pct% of unconnected
        pairs (highest phi). 20% of new connections are random (exploration).
        Number of swaps varies per neuron — well-connected neurons may keep
        all their connections while poorly-connected ones lose several.

        Called at chunk boundaries. Non-differentiable.
        """
        if not self.config.structural_plasticity:
            return
        if not hasattr(self, 'co_activation_ema'):
            return
        if not self._co_activation_ready:
            return

        N = self.config.N_neurons
        K = self.config.K_connections
        explore_frac = self.config.plasticity_exploration_frac
        phi = self.co_activation_ema
        phi.fill_diagonal_(0.0)
        device = phi.device
        conn = self.conn_indices  # [N, K]

        total_conns = N * K
        n_prune = max(1, int(total_conns * self.config.plasticity_pct))

        with torch.no_grad():
            # Gather phi for all existing connections: [N, K]
            conn_phi = phi[
                torch.arange(N, device=device).unsqueeze(1), conn]

            # Flatten and find the globally weakest connections
            flat_phi = conn_phi.reshape(-1)  # [N*K]
            _, prune_flat_idx = flat_phi.topk(n_prune, largest=False)

            # Convert flat indices to (neuron, slot) pairs
            prune_n = prune_flat_idx // K
            prune_k = prune_flat_idx % K

            # Build mask of all current connections: [N, N] bool
            conn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            conn_mask.scatter_(1, conn, True)
            conn_mask.fill_diagonal_(True)  # no self-connections

            # Find globally strongest UNCONNECTED pairs
            phi_candidates = phi.clone()
            phi_candidates[conn_mask] = -float('inf')
            flat_candidates = phi_candidates.reshape(-1)  # [N*N]
            _, grow_flat_idx = flat_candidates.topk(n_prune, largest=True)

            # Convert to (source_neuron, target_neuron) pairs
            grow_target = grow_flat_idx % N

            # 20% exploration: replace some targets with random
            rand_targets = torch.randint(0, N - 1, (n_prune,), device=device)
            rand_targets = rand_targets + (rand_targets >= prune_n).long()
            use_random = torch.rand(n_prune, device=device) < explore_frac
            grow_target = torch.where(use_random, rand_targets, grow_target)

            # grow_target comes from a global ranking over candidate pairs.
            # Once targets are reassigned into different source rows, re-check
            # the no-self-connection invariant for the actual destination row.
            is_self = grow_target == prune_n
            if is_self.any():
                repl = torch.randint(0, N - 1, (is_self.sum(),), device=device)
                repl = repl + (repl >= prune_n[is_self]).long()
                grow_target = grow_target.clone()
                grow_target[is_self] = repl

            # Apply: replace pruned connections with new targets
            conn[prune_n, prune_k] = grow_target

            # Re-sort for efficient gather
            sorted_idx, _ = conn.sort(dim=-1)
            self.conn_indices.copy_(sorted_idx)

        self._last_rewire_swaps = n_prune

    # ================================================================
    # State management
    # ================================================================

    def detach_states(self):
        """Detach all runtime state from compute graph."""
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.prev_messages = self.prev_messages.detach()

    def runtime_state_dict(self) -> dict:
        state = {
            'h': self.h, 'prev_messages': self.prev_messages,
            'w_conn': self.w_conn, 'primitives_state': self.primitives_state,
            'decay_logit': self.decay_logit,
            'hebbian_traces': self.hebbian_traces,
            'msg_magnitude': self.msg_magnitude,
        }
        if (self.config.structural_plasticity and
                hasattr(self, 'co_activation_ema')):
            state['co_activation_ema'] = self.co_activation_ema
            state['_co_activation_ready'] = self._co_activation_ready
        return state

    def load_runtime_state(self, state: dict):
        for key, val in state.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if isinstance(current, torch.Tensor) and isinstance(val, torch.Tensor):
                if current.shape != val.shape:
                    raise ValueError(
                        f"Runtime state shape mismatch for '{key}': "
                        f"expected {current.shape}, got {val.shape}. "
                        f"This usually means the checkpoint was saved with "
                        f"a different batch size or config.")
            setattr(self, key, val)
        self._initialized = True
