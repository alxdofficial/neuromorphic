"""Memory Graph — per-token recurrence with inter-neuron message passing.

v9-ES: No-grad forward (fast, ~64K tok/s) + Evolution Strategies for params.
Per-neuron modulators + dendritic FC layers (from v9), trained by ES not backprop.

At each token timestep within a segment:
  1. Receive: dendritic FC gather from K neighbors (+ CC signal for ports)
  2. Integrate: h = decay * h + (1-decay) * received
  3. Message: prev_messages = tanh(h * effective_primitives)

Per-neuron modulator (runs at segment start, no grad):
  modulator(h, traces, primitives, key) → gate_prim, gate_key, decay_mod
  effective_prim = primitives + mod_lr * gate_prim * normalize(trace_prim)
  effective_key = key + mod_lr * gate_key * normalize(trace_key)
  effective_decay = sigmoid(decay_logit + decay_mod)

All memory graph params updated by ES (not backprop). The modulator is
active at inference for adaptation without any training signal.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config

try:
    import triton
    from .triton_kernels import memory_graph_routing_kernel, memory_graph_step_kernel
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class MemoryGraph(nn.Module):
    """Memory graph with per-neuron modulators + dendritic FC, trained by ES.

    nn.Parameters (requires_grad=False, updated by ES):
        primitives, key, decay_logit — base neuron properties
        dendrite_branch_w, dendrite_group_w — per-neuron dendritic FC
        fc1_w, fc1_b, fc2_w, fc2_b — per-neuron modulator MLP
        mod_lr_logit — learnable modulation step size

    State tensors (per-batch, not learned):
        h, prev_messages — neuron hidden state
        trace_prim, trace_key — eligibility traces
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K_conn = config.K_connections
        D = config.D_mem

        # Fixed topology
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')
        K_actual = min(K_conn, N - 1)
        conn_indices = torch.zeros(N, K_conn, dtype=torch.long, device=device)
        conn_mask = torch.ones(N, K_conn, dtype=torch.bool, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        if K_actual < K_conn:
            conn_mask[:, K_actual:] = False
        sorted_idx, order = conn_indices.sort(dim=-1)
        self.register_buffer('conn_indices', sorted_idx)
        self.register_buffer('conn_mask', conn_mask.gather(1, order))

        # Dendritic tree structure
        branch_size = config.dendrite_branch_size
        if branch_size > 0 and K_conn >= branch_size:
            self.n_branches = K_conn // branch_size
            self.branch_size = branch_size
            self.branches_per_group = min(4, self.n_branches)
            self.n_groups = max(1, self.n_branches // self.branches_per_group)
            self.branches_per_group = self.n_branches // self.n_groups
            self.use_dendritic_tree = True
        else:
            self.use_dendritic_tree = False

        # ================================================================
        # Learned parameters (ES-updated, requires_grad=False)
        # ================================================================
        self.primitives = nn.Parameter(self._rms_init(N, D, device), requires_grad=False)
        self.key = nn.Parameter(self._rms_init(N, D, device), requires_grad=False)
        self.decay_logit = nn.Parameter(torch.zeros(N, device=device), requires_grad=False)

        # Per-neuron dendritic FC
        if self.use_dendritic_tree:
            nb, bs = self.n_branches, self.branch_size
            ng, bpg = self.n_groups, self.branches_per_group
            self.dendrite_branch_w = nn.Parameter(
                torch.full((N, nb, bs, D), 1.0 / bs, device=device), requires_grad=False)
            self.dendrite_group_w = nn.Parameter(
                torch.full((N, ng, bpg, D), 1.0 / bpg, device=device), requires_grad=False)

        # Per-neuron modulator MLP
        hidden = config.modulator_hidden
        mod_input_dim = D * 5  # h + trace_prim + trace_key + primitives + key
        fc1_w = torch.randn(N, mod_input_dim, hidden, device=device) * (2.0 / (mod_input_dim + hidden)) ** 0.5
        self.fc1_w = nn.Parameter(fc1_w, requires_grad=False)
        self.fc1_b = nn.Parameter(torch.zeros(N, hidden, device=device), requires_grad=False)
        self.fc2_w = nn.Parameter(torch.zeros(N, hidden, 3, device=device), requires_grad=False)
        self.fc2_b = nn.Parameter(torch.zeros(N, 3, device=device), requires_grad=False)
        self.mod_lr_logit = nn.Parameter(torch.tensor(-2.0, device=device), requires_grad=False)

        # Broadcast inject/readout weights (replace port neuron I/O)
        C_mem = config.D // config.D_mem
        self.C_mem = C_mem
        self.inject_w = nn.Parameter(torch.zeros(N, C_mem, device=device), requires_grad=False)
        nn.init.uniform_(self.inject_w, -1.0, 1.0)
        self.readout_w = nn.Parameter(torch.zeros(C_mem, N, device=device), requires_grad=False)
        nn.init.uniform_(self.readout_w, -1.0, 1.0)

        # Triton kernel buffers
        if _HAS_TRITON and device.type == 'cuda':
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
            self._triton_ready = True
        else:
            self._triton_ready = False

        self._initialized = False

    def _rms_init(self, N: int, D: int, device: torch.device) -> Tensor:
        raw = torch.randn(N, D, device=device)
        rms = raw.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        return raw / rms

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        device = self.primitives.device
        N, D = self.config.N_neurons, self.config.D_mem
        dtype = self.dtype

        self.h = torch.randn(BS, N, D, device=device, dtype=dtype) * 0.1
        self.prev_messages = torch.tanh(
            self.h * self.primitives.unsqueeze(0).to(dtype))

        self.trace_prim = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.trace_key = torch.zeros(BS, N, D, device=device, dtype=dtype)

        self.mean_input = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.mean_output = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.msg_magnitude = torch.zeros(BS, N, device=device, dtype=dtype)

        # Refresh Triton buffers after potential device change
        if _HAS_TRITON and device.type == 'cuda':
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
            self._triton_ready = True
        else:
            self._triton_ready = False

        self._initialized = True

    # ================================================================
    # Per-neuron modulator
    # ================================================================

    @torch.no_grad()
    def _modulator_forward(self, h: Tensor,
                           _trace_prim: Tensor | None = None,
                           _trace_key: Tensor | None = None,
                           ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-neuron MLP: [h, trace_prim, trace_key, prim, key] → 3 scalars."""
        BS = h.shape[0]
        tp = _trace_prim if _trace_prim is not None else self.trace_prim
        tk = _trace_key if _trace_key is not None else self.trace_key

        mod_input = torch.cat([
            h.float(), tp.float(), tk.float(),
            self.primitives.unsqueeze(0).expand(BS, -1, -1).float(),
            self.key.unsqueeze(0).expand(BS, -1, -1).float(),
        ], dim=-1)

        x = torch.einsum('bnd,ndh->bnh', mod_input, self.fc1_w.float()) + self.fc1_b.float()
        x = torch.tanh(x)
        out = torch.einsum('bnh,nho->bno', x, self.fc2_w.float()) + self.fc2_b.float()

        gate_prim = torch.tanh(out[..., 0:1])
        gate_key = torch.tanh(out[..., 1:2])
        decay_mod = out[..., 2:3]
        return gate_prim, gate_key, decay_mod

    # ================================================================
    # Dendritic gather
    # ================================================================

    def _dendritic_gather_fc(self, weighted: Tensor) -> Tensor:
        """Dendritic tree with per-neuron FC at branch and group levels."""
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

    # ================================================================
    # Forward segment (no grad — fast)
    # ================================================================

    @torch.no_grad()
    def _compute_effective_params(self):
        """Compute modulated effective parameters (once per segment)."""
        gate_prim, gate_key, decay_mod = self._modulator_forward(self.h)
        mod_lr = torch.sigmoid(self.mod_lr_logit)

        trace_prim_dir = self.trace_prim / self.trace_prim.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)
        trace_key_dir = self.trace_key / self.trace_key.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)

        eff_prim = (self.primitives.unsqueeze(0).to(self.dtype)
                    + mod_lr * gate_prim.to(self.dtype) * trace_prim_dir)
        eff_key = (self.key.unsqueeze(0).to(self.dtype)
                   + mod_lr * gate_key.to(self.dtype) * trace_key_dir)
        eff_decay = torch.sigmoid(
            self.decay_logit.unsqueeze(0).to(self.dtype)
            + decay_mod.squeeze(-1).to(self.dtype))
        return eff_prim, eff_key, eff_decay

    @torch.no_grad()
    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment. Dispatches to Triton on CUDA."""
        eff_prim, eff_key, eff_decay = self._compute_effective_params()

        if self._triton_ready and cc_signals.is_cuda:
            return self._forward_segment_triton(cc_signals, eff_prim, eff_key, eff_decay)
        return self._forward_segment_python(cc_signals, eff_prim, eff_key, eff_decay)

    def _forward_segment_python(self, cc_signals, eff_prim, eff_key, eff_decay):
        """Python reference implementation."""
        BS, T_seg, C_mem, D = cc_signals.shape
        N = self.config.N_neurons
        stride = self.config.memory_update_stride
        eff_decay_exp = eff_decay.unsqueeze(-1)

        h = self.h
        prev_msg = self.prev_messages
        output = torch.empty(BS, T_seg, C_mem, D, device=cc_signals.device, dtype=self.dtype)
        received_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=self.dtype)
        msg_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=self.dtype)

        inject_weights = torch.sigmoid(self.inject_w)    # [N, C_mem]
        readout_weights = torch.sigmoid(self.readout_w)  # [C_mem, N]

        for t in range(T_seg):
            if t % stride == 0:
                neighbor_msgs = prev_msg[:, self.conn_indices]
                sim = (eff_key.unsqueeze(2) * neighbor_msgs).sum(dim=-1)
                routing = torch.sigmoid(sim)
                weighted = routing.unsqueeze(-1) * neighbor_msgs

                if self.use_dendritic_tree:
                    received = self._dendritic_gather_fc(weighted)
                else:
                    received = weighted.sum(dim=2)

                # Broadcast inject: cc_signals [BS, C_mem, D] → all N neurons
                broadcast = torch.einsum('nc,bcd->bnd', inject_weights, cc_signals[:, t].float()).to(self.dtype)
                received = received + broadcast
                h = eff_decay_exp * h + (1 - eff_decay_exp) * received
                prev_msg = torch.tanh(h * eff_prim)
                received_accum += received
                msg_accum += prev_msg

            # Weighted readout: prev_msg [BS, N, D] → [BS, C_mem, D]
            output[:, t] = torch.einsum('cn,bnd->bcd', readout_weights, prev_msg.float()).to(cc_signals.dtype)

        self._post_segment_update(h, prev_msg, received_accum, msg_accum, T_seg)
        return output

    def _forward_segment_triton(self, cc_signals, eff_prim, eff_key, eff_decay):
        """Triton-accelerated per-token dynamics with dendritic FC.

        Inject broadcast precomputed as [BS, T_seg, N, D], passed to kernel.
        Per-step messages saved to msg_all buffer inside kernel.
        Readout computed as single batched einsum after all T steps.
        """
        BS, T_seg, C_mem, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.K_connections
        stride = self.config.memory_update_stride

        # Prepare eff_decay as f32 [BS, N] for kernel
        eff_decay_f32 = eff_decay.float().contiguous()
        eff_prim_c = eff_prim.contiguous()
        eff_key_c = eff_key.contiguous()
        cc_signals = cc_signals.contiguous()

        # Precompute broadcast inject for all timesteps: [BS, T_seg, N, D]
        inject_weights = torch.sigmoid(self.inject_w.float())  # [N, C_mem] float32
        inject_bc = torch.einsum('nc,btcd->btnd', inject_weights,
                                 cc_signals.float()).to(self.dtype).contiguous()

        recv_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=torch.float32)
        msg_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=torch.float32)
        msg_mag_accum = torch.zeros(BS, N, device=cc_signals.device, dtype=torch.float32)

        # Buffer for per-step messages (for readout)
        msg_all = torch.empty(BS, T_seg, N, D, device=cc_signals.device, dtype=self.dtype)

        grid = (BS, N)

        # Routing kernel
        routing_w = torch.empty(BS, N, K, device=cc_signals.device, dtype=self.dtype)
        memory_graph_routing_kernel[grid](
            self.prev_messages, eff_key_c,
            self._conn_idx_i32, routing_w,
            BS=BS, N=N, D=D, K=K)

        # Dendritic tree params
        if self.use_dendritic_tree:
            branch_size = self.branch_size
            branches_per_group = self.branches_per_group
            n_groups = self.n_groups
            branch_w = self.dendrite_branch_w.contiguous()
            group_w = self.dendrite_group_w.contiguous()
            use_fc = 1
        else:
            branch_size = K
            branches_per_group = 1
            n_groups = 1
            branch_w = eff_prim_c
            group_w = eff_prim_c
            use_fc = 0

        act_trace = msg_mag_accum

        for t in range(0, T_seg, stride):
            # Kernel: dendritic gather + broadcast inject + integration + messaging
            # inject_bc passed as cc_signals_ptr, msg_all as output_ptr
            # C=1 enables the inject/output code paths in the kernel
            memory_graph_step_kernel[grid](
                self.h, self.prev_messages,
                self._conn_idx_i32, routing_w,
                eff_decay_f32, eff_prim_c,
                branch_w, group_w,
                inject_bc, msg_all,
                recv_accum, msg_accum, msg_mag_accum,
                act_trace,
                t,
                BS=BS, N=N, D=D, K=K, C=1, T_seg=T_seg,
                BRANCH_SIZE=branch_size,
                BRANCHES_PER_GROUP=branches_per_group,
                N_GROUPS=n_groups,
                WRITE_ACT_TRACE=0,
                USE_DENDRITE_FC=use_fc)

        # Fill held output slots for stride > 1
        if stride > 1:
            for s in range(1, stride):
                msg_all[:, s::stride] = msg_all[:, ::stride]

        # Batched readout from msg_all: one einsum for all T
        readout_weights = torch.sigmoid(self.readout_w.float())  # [C_mem, N]
        output = torch.einsum('cn,btnd->btcd', readout_weights,
                              msg_all.float()).to(cc_signals.dtype)

        n_steps = max(T_seg // stride, 1)
        self._post_segment_update(
            self.h, self.prev_messages,
            (recv_accum / n_steps).to(self.dtype),
            (msg_accum / n_steps).to(self.dtype),
            T_seg, skip_mean=True)
        return output

    def _post_segment_update(self, h, prev_msg, received_accum, msg_accum,
                             T_seg, skip_mean=False):
        """Update persistent state after a segment."""
        stride = self.config.memory_update_stride
        self.h = h
        self.prev_messages = prev_msg

        if not skip_mean:
            n_steps = max(T_seg // stride, 1)
            self.mean_input = received_accum / n_steps
            self.mean_output = msg_accum / n_steps
        else:
            self.mean_input = received_accum
            self.mean_output = msg_accum

        td = self.config.trace_decay
        self.trace_prim = (td * self.trace_prim + (1 - td) * h).to(self.dtype)
        self.trace_key = (td * self.trace_key + (1 - td) * self.mean_input).to(self.dtype)

        alpha = 0.05
        seg_mag = prev_msg.norm(dim=-1)
        self.msg_magnitude = (1 - alpha) * self.msg_magnitude + alpha * seg_mag

    # ================================================================
    # ES utilities
    # ================================================================

    def get_es_params(self) -> dict[str, Tensor]:
        """Get all ES-trainable parameter tensors (per-neuron params only)."""
        params = {
            'primitives': self.primitives.data,
            'key': self.key.data,
            'decay_logit': self.decay_logit.data,
            'fc1_w': self.fc1_w.data,
            'fc1_b': self.fc1_b.data,
            'fc2_w': self.fc2_w.data,
            'fc2_b': self.fc2_b.data,
            'inject_w': self.inject_w.data,
            'readout_w': self.readout_w.data,
        }
        if self.use_dendritic_tree:
            params['dendrite_branch_w'] = self.dendrite_branch_w.data
            params['dendrite_group_w'] = self.dendrite_group_w.data
        return params

    # Names of params where the neuron axis is dim 1 (not dim 0)
    _NEURON_DIM1_PARAMS = frozenset({'readout_w'})

    def get_neuron_es_params(self, neuron_ids: Tensor) -> dict[str, Tensor]:
        """Get ES params for a subset of neurons (for sparse ES)."""
        K_idx = neuron_ids  # [K]
        params = {
            'primitives': self.primitives.data[K_idx],
            'key': self.key.data[K_idx],
            'decay_logit': self.decay_logit.data[K_idx],
            'fc1_w': self.fc1_w.data[K_idx],
            'fc1_b': self.fc1_b.data[K_idx],
            'fc2_w': self.fc2_w.data[K_idx],
            'fc2_b': self.fc2_b.data[K_idx],
            'inject_w': self.inject_w.data[K_idx],
            'readout_w': self.readout_w.data[:, K_idx],
        }
        if self.use_dendritic_tree:
            params['dendrite_branch_w'] = self.dendrite_branch_w.data[K_idx]
            params['dendrite_group_w'] = self.dendrite_group_w.data[K_idx]
        return params

    def apply_es_perturbation(self, neuron_ids: Tensor,
                              noise: dict[str, Tensor], sigma: float):
        """Add scaled noise to K neurons' parameters."""
        K_idx = neuron_ids
        for name, eps in noise.items():
            param = getattr(self, name)
            if name in self._NEURON_DIM1_PARAMS:
                param.data[:, K_idx] += sigma * eps
            else:
                param.data[K_idx] += sigma * eps

    def apply_es_update(self, neuron_ids: Tensor,
                        weighted_noise: dict[str, Tensor], lr: float):
        """Apply ES gradient estimate to K neurons' parameters."""
        K_idx = neuron_ids
        for name, update in weighted_noise.items():
            param = getattr(self, name)
            if name in self._NEURON_DIM1_PARAMS:
                param.data[:, K_idx] += lr * update
            else:
                param.data[K_idx] += lr * update

    def detach_states(self):
        pass  # States are never on compute graph

    def runtime_state_dict(self) -> dict:
        return {
            'h': self.h, 'prev_messages': self.prev_messages,
            'trace_prim': self.trace_prim, 'trace_key': self.trace_key,
            'mean_input': self.mean_input, 'mean_output': self.mean_output,
            'msg_magnitude': self.msg_magnitude,
        }

    def load_runtime_state(self, state: dict):
        buffer_names = {name for name, _ in self.named_buffers()}
        param_names = {name for name, _ in self.named_parameters()}
        for key, val in state.items():
            if key not in buffer_names and key not in param_names and hasattr(self, key):
                setattr(self, key, val)
        self._initialized = True
