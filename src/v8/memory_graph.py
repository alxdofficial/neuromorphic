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

        self.register_buffer('cc_port_idx', torch.arange(config.C, device=device))
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

        x = torch.einsum('bnd,ndh->bnh', mod_input, self.fc1_w) + self.fc1_b
        x = torch.tanh(x)
        out = torch.einsum('bnh,nho->bno', x, self.fc2_w) + self.fc2_b

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
    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment. No autograd — fast.

        Returns:
            output: [BS, T_seg, C, D_mem] port neuron messages
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons
        stride = self.config.memory_update_stride

        # Per-neuron modulator: compute effective parameters
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
        eff_decay_exp = eff_decay.unsqueeze(-1)

        # Per-token dynamics
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

                received[:, :C] = received[:, :C] + cc_signals[:, t]
                h = eff_decay_exp * h + (1 - eff_decay_exp) * received
                prev_msg = torch.tanh(h * eff_prim)

                received_accum += received
                msg_accum += prev_msg

            output[:, t] = prev_msg[:, :C]

        # Update state
        self.h = h
        self.prev_messages = prev_msg
        n_steps = max(T_seg // stride, 1)
        self.mean_input = received_accum / n_steps
        self.mean_output = msg_accum / n_steps

        # Traces
        td = self.config.trace_decay
        self.trace_prim = (td * self.trace_prim + (1 - td) * h).to(self.dtype)
        self.trace_key = (td * self.trace_key + (1 - td) * self.mean_input).to(self.dtype)

        # Message magnitude
        alpha = 0.05
        seg_mag = prev_msg.norm(dim=-1)
        self.msg_magnitude = (1 - alpha) * self.msg_magnitude + alpha * seg_mag

        return output

    # ================================================================
    # ES utilities
    # ================================================================

    def get_es_params(self) -> dict[str, Tensor]:
        """Get all ES-trainable parameter tensors."""
        params = {
            'primitives': self.primitives.data,
            'key': self.key.data,
            'decay_logit': self.decay_logit.data,
            'fc1_w': self.fc1_w.data,
            'fc1_b': self.fc1_b.data,
            'fc2_w': self.fc2_w.data,
            'fc2_b': self.fc2_b.data,
            'mod_lr_logit': self.mod_lr_logit.data,
        }
        if self.use_dendritic_tree:
            params['dendrite_branch_w'] = self.dendrite_branch_w.data
            params['dendrite_group_w'] = self.dendrite_group_w.data
        return params

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
            if name in ('mod_lr_logit',):
                param.data += sigma * eps  # scalar param
            else:
                param.data[K_idx] += sigma * eps

    def apply_es_update(self, neuron_ids: Tensor,
                        weighted_noise: dict[str, Tensor], lr: float):
        """Apply ES gradient estimate to K neurons' parameters."""
        K_idx = neuron_ids
        for name, update in weighted_noise.items():
            param = getattr(self, name)
            if name in ('mod_lr_logit',):
                param.data += lr * update
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
