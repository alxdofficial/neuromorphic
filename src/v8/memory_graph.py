"""Memory Graph — differentiable per-token recurrence with inter-neuron message passing.

v9: Fully differentiable, trained end-to-end via backprop.
Per-neuron modulators predict gate/decay from internal state.
Per-neuron dendritic FC layers (learned branch/group weights).

At each token timestep within a segment:
  1. Receive: dendritic FC gather from K neighbors (+ CC signal for ports)
  2. Integrate: h = decay * h + (1-decay) * received
  3. Message: prev_messages = tanh(h * primitives)

Per-neuron modulator (runs at segment start):
  modulator(h) → gate_prim, gate_key, decay_mod
  effective_prim = primitives + mod_lr * gate_prim * normalize(trace_prim)
  effective_key = key + mod_lr * gate_key * normalize(trace_key)
  effective_decay = sigmoid(decay_logit + decay_mod)

Trained by backprop during training. At inference, modulator still runs
(no gradients needed) → enables adaptation without backprop.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config


class MemoryGraph(nn.Module):
    """Differentiable memory graph with per-neuron learned dynamics.

    nn.Parameters (trained by backprop):
        primitives, key, decay_logit — base neuron properties
        dendrite_branch_w, dendrite_group_w — per-neuron dendritic FC
        fc1_w, fc1_b, fc2_w, fc2_b — per-neuron modulator MLP
        mod_lr_logit — learnable modulation step size

    State tensors (per-batch, detached at segment boundaries):
        h, prev_messages — neuron hidden state
        trace_prim, trace_key — eligibility traces (always detached)
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        N = config.N_neurons
        K_conn = config.K_connections
        D = config.D_mem

        # Fixed topology: random sparse connectivity (vectorized, GPU-friendly)
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')  # exclude self-connections
        K_actual = min(K_conn, N - 1)
        conn_indices = torch.zeros(N, K_conn, dtype=torch.long, device=device)
        conn_mask = torch.ones(N, K_conn, dtype=torch.bool, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        if K_actual < K_conn:
            conn_mask[:, K_actual:] = False

        # Sort for cache locality
        sorted_idx, order = conn_indices.sort(dim=-1)
        conn_indices = sorted_idx
        conn_mask = conn_mask.gather(1, order)

        self.register_buffer('conn_indices', conn_indices)  # [N, K]
        self.register_buffer('conn_mask', conn_mask)         # [N, K]

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
        # Learned parameters
        # ================================================================

        # Per-neuron base properties (RMS-normalized at init)
        self.primitives = nn.Parameter(self._rms_init(N, D))  # [N, D]
        self.key = nn.Parameter(self._rms_init(N, D))          # [N, D]
        self.decay_logit = nn.Parameter(torch.zeros(N))        # [N]

        # Per-neuron dendritic FC layers
        if self.use_dendritic_tree:
            nb = self.n_branches
            bs = self.branch_size
            ng = self.n_groups
            bpg = self.branches_per_group

            # Branch FC: per-neuron, per-branch, per-connection weights
            # Init: 1/branch_size so initial behavior ≈ uniform average
            self.dendrite_branch_w = nn.Parameter(
                torch.full((N, nb, bs, D), 1.0 / bs))

            # Group FC: per-neuron, per-group, per-branch weights
            self.dendrite_group_w = nn.Parameter(
                torch.full((N, ng, bpg, D), 1.0 / bpg))

        # Per-neuron modulator MLP: [h, trace_prim, trace_key, primitives, key] → 3
        hidden = config.modulator_hidden
        mod_input_dim = D * 5  # h + trace_prim + trace_key + primitives + key
        self._mod_input_dim = mod_input_dim
        # Xavier init for fc1, zero init for fc2 (starts as no-op)
        fc1_w = torch.randn(N, mod_input_dim, hidden) * (2.0 / (mod_input_dim + hidden)) ** 0.5
        self.fc1_w = nn.Parameter(fc1_w)            # [N, mod_input_dim, hidden]
        self.fc1_b = nn.Parameter(torch.zeros(N, hidden))  # [N, hidden]
        self.fc2_w = nn.Parameter(torch.zeros(N, hidden, 3))  # [N, hidden, 3]
        self.fc2_b = nn.Parameter(torch.zeros(N, 3))           # [N, 3]

        # Learnable modulation step size (sigmoid → [0, 1])
        self.mod_lr_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

        # CC port assignment
        self.register_buffer('cc_port_idx',
                             torch.arange(config.C, device=device))

        # State tensors (set in initialize_states)
        self._initialized = False

    def _rms_init(self, N: int, D: int) -> Tensor:
        """Random directions, RMS-normalized."""
        raw = torch.randn(N, D, device=self.device, dtype=torch.float32)
        rms = raw.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        return raw / rms

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        """Initialize per-batch state tensors (not nn.Parameters)."""
        N = self.config.N_neurons
        D = self.config.D_mem
        # Use actual device of parameters (may have been moved by .to())
        device = self.primitives.device
        dtype = self.dtype if self.primitives.dtype == torch.float32 else self.primitives.dtype

        self.h = torch.randn(
            BS, N, D, device=device, dtype=dtype) * 0.1
        self.prev_messages = torch.tanh(
            self.h * self.primitives.unsqueeze(0).to(dtype))

        self.trace_prim = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.trace_key = torch.zeros(BS, N, D, device=device, dtype=dtype)

        # Diagnostics accumulators
        self.mean_input = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.mean_output = torch.zeros(BS, N, D, device=device, dtype=dtype)
        self.msg_magnitude = torch.zeros(BS, N, device=device, dtype=dtype)

        self._initialized = True

    # ================================================================
    # Per-neuron modulator
    # ================================================================

    def _modulator_forward(self, h: Tensor,
                           _trace_prim: Tensor | None = None,
                           _trace_key: Tensor | None = None,
                           ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-neuron MLP: [h, trace_prim, trace_key, primitives, key] → 3.

        Each neuron has its own weights. Implemented via einsum for
        batched per-neuron matmul.

        Args:
            h: [BS, N, D_mem] — neuron hidden state (detached)
            _trace_prim: override for trace_prim (for diagnostics with sliced batch)
            _trace_key: override for trace_key (for diagnostics with sliced batch)
        Returns:
            gate_prim: [BS, N, 1] in [-1, 1]
            gate_key:  [BS, N, 1] in [-1, 1]
            decay_mod: [BS, N, 1] unbounded
        """
        BS = h.shape[0]
        tp = _trace_prim if _trace_prim is not None else self.trace_prim
        tk = _trace_key if _trace_key is not None else self.trace_key

        mod_input = torch.cat([
            h.float(),
            tp.float(),
            tk.float(),
            self.primitives.unsqueeze(0).expand(BS, -1, -1).float(),
            self.key.unsqueeze(0).expand(BS, -1, -1).float(),
        ], dim=-1)  # [BS, N, D_mem * 5]

        # fc1: [BS, N, D*5] @ [N, D*5, H] → [BS, N, H]
        x = torch.einsum('bnd,ndh->bnh', mod_input, self.fc1_w) + self.fc1_b
        x = torch.tanh(x)
        # fc2: [BS, N, H] @ [N, H, 3] → [BS, N, 3]
        out = torch.einsum('bnh,nho->bno', x, self.fc2_w) + self.fc2_b

        gate_prim = torch.tanh(out[..., 0:1])   # [-1, 1]
        gate_key = torch.tanh(out[..., 1:2])     # [-1, 1]
        decay_mod = out[..., 2:3]                 # unbounded
        return gate_prim, gate_key, decay_mod

    # ================================================================
    # Dendritic tree with per-neuron FC
    # ================================================================

    def _dendritic_gather_fc(self, weighted: Tensor) -> Tensor:
        """Dendritic tree with per-neuron FC at branch and group levels.

        Args:
            weighted: [BS, N, K, D] — routing-weighted neighbor messages

        Returns:
            received: [BS, N, D]
        """
        BS = weighted.shape[0]
        N = self.config.N_neurons
        D = weighted.shape[-1]
        K = self.config.K_connections

        bsz = self.branch_size
        bpg = self.branches_per_group
        ng = self.n_groups
        nb = self.n_branches

        n_tree = ng * bpg * bsz
        tree_msgs = weighted[:, :, :n_tree].view(BS, N, nb, bsz, D)

        # Branch FC: per-neuron learned weighted sum across connections
        # tree_msgs: [BS, N, nb, bsz, D], branch_w: [N, nb, bsz, D]
        # → weighted sum over bsz → [BS, N, nb, D]
        branch_out = torch.tanh(
            (tree_msgs * self.dendrite_branch_w.unsqueeze(0)).sum(dim=3)
        )

        # Reshape for group level: [BS, N, ng, bpg, D]
        branch_grouped = branch_out.view(BS, N, ng, bpg, D)

        # Group FC: per-neuron learned weighted sum across branches
        # branch_grouped: [BS, N, ng, bpg, D], group_w: [N, ng, bpg, D]
        # → weighted sum over bpg → [BS, N, ng, D]
        group_out = torch.tanh(
            (branch_grouped * self.dendrite_group_w.unsqueeze(0)).sum(dim=3)
        )

        # Soma: average groups
        received = group_out.mean(dim=2)  # [BS, N, D]

        # Handle leftover connections
        if n_tree < K:
            leftover = weighted[:, :, n_tree:].sum(dim=2)
            tree_frac = n_tree / K
            received = tree_frac * received + (1 - tree_frac) * leftover

        return received

    def _flat_gather(self, weighted: Tensor) -> Tensor:
        """Simple weighted sum (no dendritic tree)."""
        return weighted.sum(dim=2)  # [BS, N, D]

    # ================================================================
    # Forward segment (differentiable)
    # ================================================================

    def _forward_segment_inner(self, cc_signals: Tensor, h_prev: Tensor,
                               eff_prim: Tensor, eff_key: Tensor,
                               eff_decay: Tensor,
                               prev_messages: Tensor) -> tuple[Tensor, Tensor]:
        """Pure computation: per-token dynamics. No side effects, no self access.

        All state passed as explicit args for checkpoint reproducibility.
        """
        amp_ctx = torch.autocast(device_type=cc_signals.device.type,
                                 dtype=torch.bfloat16,
                                 enabled=cc_signals.is_cuda)
        with amp_ctx:
            BS, T_seg, C, D = cc_signals.shape
            N = self.config.N_neurons
            stride = self.config.memory_update_stride

            prev_msg = prev_messages
            neighbor_msgs = prev_msg[:, self.conn_indices]
        sim = (eff_key.unsqueeze(2) * neighbor_msgs).sum(dim=-1)
        routing = torch.sigmoid(sim)

        h = h_prev
        eff_decay_expanded = eff_decay.unsqueeze(-1)
        output_list = []

        for t in range(T_seg):
            if t % stride == 0:
                weighted = routing.unsqueeze(-1) * neighbor_msgs

                if self.use_dendritic_tree:
                    received = self._dendritic_gather_fc(weighted)
                else:
                    received = self._flat_gather(weighted)

                received = received.clone()
                received[:, :C] = received[:, :C] + cc_signals[:, t]

                h = eff_decay_expanded * h + (1 - eff_decay_expanded) * received
                prev_msg = torch.tanh(h * eff_prim)
                neighbor_msgs = prev_msg[:, self.conn_indices]
                curr_port_out = prev_msg[:, :C]

            if t % stride == 0:
                output_list.append(curr_port_out)
            else:
                output_list.append(curr_port_out.detach())

        output = torch.stack(output_list, dim=1)
        return output, h

    def forward_segment(self, cc_signals: Tensor,
                        h_prev: Tensor) -> tuple[Tensor, Tensor]:
        """Process one segment. Wraps _forward_segment_inner with side effects.

        Args:
            cc_signals: [BS, T_seg, C, D_mem] from lower scan (detached)
            h_prev: [BS, N, D_mem] detached from previous segment (TBPTT)
        Returns:
            output: [BS, T_seg, C, D_mem] port neuron messages
            h: [BS, N, D_mem] final h
        """
        BS, T_seg, C, D = cc_signals.shape

        # 1. Per-neuron modulator: compute effective parameters
        gate_prim, gate_key, decay_mod = self._modulator_forward(h_prev)
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

        # 2. Inner loop (checkpointable — no side effects, all state as args)
        prev_msgs_detached = self.prev_messages.detach()
        if self.training and torch.is_grad_enabled():
            output, h = torch.utils.checkpoint.checkpoint(
                self._forward_segment_inner, cc_signals, h_prev,
                eff_prim, eff_key, eff_decay, prev_msgs_detached,
                use_reentrant=False)
        else:
            output, h = self._forward_segment_inner(
                cc_signals, h_prev, eff_prim, eff_key, eff_decay,
                prev_msgs_detached)

        # 3. Side effects (outside checkpoint)
        with torch.no_grad():
            prev_msg = torch.tanh(h.detach() * eff_prim.detach())
            self.prev_messages = prev_msg

            td = self.config.trace_decay
            self.trace_prim = (td * self.trace_prim
                               + (1 - td) * h.detach()).to(self.dtype)

            # Approximate mean_input from h dynamics
            received_approx = (h.detach() - eff_decay.unsqueeze(-1).detach() * h_prev.detach()) / \
                              (1 - eff_decay.unsqueeze(-1).detach()).clamp(min=1e-8)
            self.trace_key = (td * self.trace_key
                              + (1 - td) * received_approx).to(self.dtype)

            seg_msg_mag = prev_msg.norm(dim=-1)
            alpha = 0.05
            self.msg_magnitude = (1 - alpha) * self.msg_magnitude + alpha * seg_msg_mag

        return output, h

    def detach_states(self):
        """Detach all state tensors from compute graph (TBPTT boundary)."""
        self.h = self.h.detach()
        self.prev_messages = self.prev_messages.detach()

    def runtime_state_dict(self) -> dict:
        """Export per-batch runtime state for checkpointing."""
        return {
            'h': self.h,
            'prev_messages': self.prev_messages,
            'trace_prim': self.trace_prim,
            'trace_key': self.trace_key,
            'mean_input': self.mean_input,
            'mean_output': self.mean_output,
            'msg_magnitude': self.msg_magnitude,
        }

    def load_runtime_state(self, state: dict):
        """Restore per-batch runtime state from checkpoint."""
        buffer_names = {name for name, _ in self.named_buffers()}
        param_names = {name for name, _ in self.named_parameters()}
        for key, val in state.items():
            if key not in buffer_names and key not in param_names and hasattr(self, key):
                setattr(self, key, val)
        self._initialized = True
