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
        # Per-neuron learned parameters (ES-updated, requires_grad=False)
        # Each neuron is a small neural network with 4 components:
        #   1. Routing key — what signals to attend to
        #   2. Dendritic FC — how to process incoming signals
        #   3. Integrate MLP — how to update internal state from h + received
        #   4. Message MLP — how to produce outgoing message from h
        #   5. Modulator MLP — how to adapt plasticity from state + traces
        # ================================================================

        H = config.modulator_hidden  # hidden dim shared across neuron MLPs

        # 1. Routing key [N, D]
        self.key = nn.Parameter(self._rms_init(N, D, device), requires_grad=False)

        # 2. Dendritic FC (branch + group learnable weights)
        if self.use_dendritic_tree:
            nb, bs = self.n_branches, self.branch_size
            ng, bpg = self.n_groups, self.branches_per_group
            self.dendrite_branch_w = nn.Parameter(
                torch.full((N, nb, bs, D), 1.0 / bs, device=device), requires_grad=False)
            self.dendrite_group_w = nn.Parameter(
                torch.full((N, ng, bpg, D), 1.0 / bpg, device=device), requires_grad=False)

        # 3. Integrate MLP: [h, received] → new_h
        #    Replaces: h = decay*h + (1-decay)*received
        int_in = D * 2  # concat(h, received)
        int_w1 = torch.randn(N, int_in, H, device=device) * (2.0 / (int_in + H)) ** 0.5
        self.int_w1 = nn.Parameter(int_w1, requires_grad=False)
        self.int_b1 = nn.Parameter(torch.zeros(N, H, device=device), requires_grad=False)
        int_w2 = torch.randn(N, H, D, device=device) * (2.0 / (H + D)) ** 0.5
        # Init to approximate identity: new_h ≈ 0.5*h + 0.5*received at init
        self.int_w2 = nn.Parameter(int_w2 * 0.1, requires_grad=False)
        self.int_b2 = nn.Parameter(torch.zeros(N, D, device=device), requires_grad=False)
        # Residual gate: h_new = gate*MLP([h,received]) + (1-gate)*h
        self.int_gate = nn.Parameter(torch.zeros(N, D, device=device), requires_grad=False)

        # 4. Message MLP: h → message
        #    Replaces: tanh(h * primitives)
        msg_w1 = torch.randn(N, D, H, device=device) * (2.0 / (D + H)) ** 0.5
        self.msg_w1 = nn.Parameter(msg_w1, requires_grad=False)
        self.msg_b1 = nn.Parameter(torch.zeros(N, H, device=device), requires_grad=False)
        msg_w2 = torch.randn(N, H, D, device=device) * (2.0 / (H + D)) ** 0.5
        self.msg_w2 = nn.Parameter(msg_w2 * 0.1, requires_grad=False)
        self.msg_b2 = nn.Parameter(torch.zeros(N, D, device=device), requires_grad=False)

        # 5. Modulator MLP: [h, trace_prim, trace_key] → gate_prim, gate_key, decay_mod
        mod_input_dim = D * 3  # h + trace_prim + trace_key
        mod_w1 = torch.randn(N, mod_input_dim, H, device=device) * (2.0 / (mod_input_dim + H)) ** 0.5
        self.mod_w1 = nn.Parameter(mod_w1, requires_grad=False)
        self.mod_b1 = nn.Parameter(torch.zeros(N, H, device=device), requires_grad=False)
        self.mod_w2 = nn.Parameter(torch.zeros(N, H, 3, device=device), requires_grad=False)
        self.mod_b2 = nn.Parameter(torch.zeros(N, 3, device=device), requires_grad=False)

        # Modulation step size
        self.mod_lr_logit = nn.Parameter(torch.tensor(-2.0, device=device), requires_grad=False)

        # Separate write (CC input) and read (LM output) neurons
        # Neurons 0..C-1 = write (receive CC signals from LM)
        # Neurons C..2C-1 = read (send messages back to LM)
        # Forces signal to traverse graph between input and output.
        C = config.C
        self.register_buffer('write_neurons', torch.arange(C, device=device))
        self.register_buffer('read_neurons', torch.arange(C, 2*C, device=device))

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
        device = self.key.device
        N, D = self.config.N_neurons, self.config.D_mem
        dtype = self.dtype

        self.h = torch.randn(BS, N, D, device=device, dtype=dtype) * 0.1
        self.prev_messages = self._message(self.h)

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
        """Per-neuron modulator: [h, trace_prim, trace_key] → gate, gate_key, decay_mod."""
        tp = _trace_prim if _trace_prim is not None else self.trace_prim
        tk = _trace_key if _trace_key is not None else self.trace_key

        mod_input = torch.cat([h.float(), tp.float(), tk.float()], dim=-1)

        x = torch.einsum('bnd,ndh->bnh', mod_input, self.mod_w1.float()) + self.mod_b1.float()
        x = torch.tanh(x)
        out = torch.einsum('bnh,nho->bno', x, self.mod_w2.float()) + self.mod_b2.float()

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
    # ================================================================
    # Per-neuron compute: integrate + message MLPs
    # ================================================================

    @torch.no_grad()
    def _integrate(self, h: Tensor, received: Tensor) -> Tensor:
        """Per-neuron integrate MLP: [h, received] → new_h.

        Uses residual gating: h_new = sigmoid(gate)*MLP([h,received]) + (1-sigmoid(gate))*h
        """
        x = torch.cat([h.float(), received.float()], dim=-1)  # [BS, N, 2D]
        hidden = torch.einsum('bnd,ndh->bnh', x, self.int_w1.float()) + self.int_b1.float()
        hidden = torch.tanh(hidden)
        mlp_out = torch.einsum('bnh,nhd->bnd', hidden, self.int_w2.float()) + self.int_b2.float()
        gate = torch.sigmoid(self.int_gate.float().unsqueeze(0))  # [1, N, D]
        return (gate * mlp_out + (1 - gate) * h.float()).to(self.dtype)

    @torch.no_grad()
    def _message(self, h: Tensor) -> Tensor:
        """Per-neuron message MLP: h → outgoing message (tanh-bounded)."""
        hidden = torch.einsum('bnd,ndh->bnh', h.float(), self.msg_w1.float()) + self.msg_b1.float()
        hidden = torch.tanh(hidden)
        msg = torch.einsum('bnh,nhd->bnd', hidden, self.msg_w2.float()) + self.msg_b2.float()
        return torch.tanh(msg).to(self.dtype)

    def _compute_effective_key(self):
        """Compute modulated effective key (once per segment)."""
        gate_prim, gate_key, decay_mod = self._modulator_forward(self.h)
        mod_lr = torch.sigmoid(self.mod_lr_logit)

        trace_key_dir = self.trace_key / self.trace_key.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)

        eff_key = (self.key.unsqueeze(0).to(self.dtype)
                   + mod_lr * gate_key.to(self.dtype) * trace_key_dir)
        return eff_key

    @torch.no_grad()
    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment. Uses Python path (Triton needs update for MLPs)."""
        eff_key = self._compute_effective_key()
        # Triton kernel doesn't support integrate/message MLPs yet — Python only
        return self._forward_segment_python(cc_signals, eff_key)

    def _forward_segment_python(self, cc_signals, eff_key):
        """Python reference implementation with per-neuron MLPs."""
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons
        stride = self.config.memory_update_stride

        h = self.h
        prev_msg = self.prev_messages
        output = torch.empty(BS, T_seg, C, D, device=cc_signals.device, dtype=self.dtype)
        received_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=self.dtype)
        msg_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=self.dtype)

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

                # RMSNorm received — consistent magnitude regardless of neighbor activity
                rms = received.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
                received = received / rms

                # CC injection into WRITE neurons
                received[:, self.write_neurons] = received[:, self.write_neurons] + cc_signals[:, t]

                # Integrate MLP: [h, received] → new h (with residual gate)
                h = self._integrate(h, received)

                # Message MLP: h → outgoing message
                prev_msg = self._message(h)

                received_accum += received
                msg_accum += prev_msg

            output[:, t] = prev_msg[:, self.read_neurons]

        self._post_segment_update(h, prev_msg, received_accum, msg_accum, T_seg)
        return output

    def _forward_segment_triton(self, cc_signals, eff_prim, eff_key, eff_decay):
        """Triton-accelerated per-token dynamics with dendritic FC."""
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.K_connections
        stride = self.config.memory_update_stride

        # Prepare eff_decay as f32 [BS, N] for kernel
        eff_decay_f32 = eff_decay.float().contiguous()
        eff_prim_c = eff_prim.contiguous()
        eff_key_c = eff_key.contiguous()
        cc_signals = cc_signals.contiguous()

        output = torch.empty(BS, T_seg, C, D, device=cc_signals.device, dtype=self.dtype)
        recv_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=torch.float32)
        msg_accum = torch.zeros(BS, N, D, device=cc_signals.device, dtype=torch.float32)
        msg_mag_accum = torch.zeros(BS, N, device=cc_signals.device, dtype=torch.float32)

        grid = (BS, N)

        # Routing kernel: eff_key × prev_messages → routing weights
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
            branch_w = eff_prim_c  # dummy ptr (not accessed)
            group_w = eff_prim_c
            use_fc = 0

        # Dummy act_trace (not needed for ES)
        act_trace = msg_mag_accum  # reuse as dummy ptr

        for t in range(0, T_seg, stride):
            memory_graph_step_kernel[grid](
                self.h, self.prev_messages,
                self._conn_idx_i32, routing_w,
                eff_decay_f32, eff_prim_c,
                branch_w, group_w,
                cc_signals, output,
                recv_accum, msg_accum, msg_mag_accum,
                act_trace,
                t,
                BS=BS, N=N, D=D, K=K, C=C, T_seg=T_seg,
                BRANCH_SIZE=branch_size,
                BRANCHES_PER_GROUP=branches_per_group,
                N_GROUPS=n_groups,
                WRITE_ACT_TRACE=0,
                USE_DENDRITE_FC=use_fc)

        # Fill held output slots
        if stride > 1:
            for s in range(1, stride):
                output[:, s::stride] = output[:, ::stride]

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
            'key': self.key.data,
            'int_w1': self.int_w1.data, 'int_b1': self.int_b1.data,
            'int_w2': self.int_w2.data, 'int_b2': self.int_b2.data,
            'int_gate': self.int_gate.data,
            'msg_w1': self.msg_w1.data, 'msg_b1': self.msg_b1.data,
            'msg_w2': self.msg_w2.data, 'msg_b2': self.msg_b2.data,
            'mod_w1': self.mod_w1.data, 'mod_b1': self.mod_b1.data,
            'mod_w2': self.mod_w2.data, 'mod_b2': self.mod_b2.data,
        }
        if self.use_dendritic_tree:
            params['dendrite_branch_w'] = self.dendrite_branch_w.data
            params['dendrite_group_w'] = self.dendrite_group_w.data
        return params

    def get_neuron_es_params(self, neuron_ids: Tensor) -> dict[str, Tensor]:
        """Get ES params for a subset of neurons (for sparse ES)."""
        K = neuron_ids
        params = {
            'key': self.key.data[K],
            'int_w1': self.int_w1.data[K], 'int_b1': self.int_b1.data[K],
            'int_w2': self.int_w2.data[K], 'int_b2': self.int_b2.data[K],
            'int_gate': self.int_gate.data[K],
            'msg_w1': self.msg_w1.data[K], 'msg_b1': self.msg_b1.data[K],
            'msg_w2': self.msg_w2.data[K], 'msg_b2': self.msg_b2.data[K],
            'mod_w1': self.mod_w1.data[K], 'mod_b1': self.mod_b1.data[K],
            'mod_w2': self.mod_w2.data[K], 'mod_b2': self.mod_b2.data[K],
        }
        if self.use_dendritic_tree:
            params['dendrite_branch_w'] = self.dendrite_branch_w.data[K]
            params['dendrite_group_w'] = self.dendrite_group_w.data[K]
        return params

    def apply_es_perturbation(self, neuron_ids: Tensor,
                              noise: dict[str, Tensor], sigma: float):
        """Add scaled noise to K neurons' parameters."""
        K_idx = neuron_ids
        for name, eps in noise.items():
            param = getattr(self, name)
            param.data[K_idx] += sigma * eps

    def apply_es_update(self, neuron_ids: Tensor,
                        weighted_noise: dict[str, Tensor], lr: float):
        """Apply ES gradient estimate to K neurons' parameters."""
        K_idx = neuron_ids
        for name, update in weighted_noise.items():
            param = getattr(self, name)
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
