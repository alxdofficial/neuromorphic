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
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import V8Config


# ================================================================
# Standalone step function (used by both forward and backward)
# ================================================================

def _dendritic_gather_static(weighted: Tensor,
                             branch_w: Tensor, group_w: Tensor,
                             nb: int, bsz: int, ng: int, bpg: int,
                             K: int) -> Tensor:
    """Dendritic tree FC gather. Standalone for use in autograd backward."""
    BS, N, _, D = weighted.shape
    n_tree = ng * bpg * bsz
    tree_msgs = weighted[:, :, :n_tree].view(BS, N, nb, bsz, D)

    branch_out = torch.tanh(
        (tree_msgs * branch_w.unsqueeze(0)).sum(dim=3))
    branch_grouped = branch_out.view(BS, N, ng, bpg, D)
    group_out = torch.tanh(
        (branch_grouped * group_w.unsqueeze(0)).sum(dim=3))
    received = group_out.mean(dim=2)

    if n_tree < K:
        leftover = weighted[:, :, n_tree:].sum(dim=2)
        tree_frac = n_tree / K
        received = tree_frac * received + (1 - tree_frac) * leftover
    return received


def _one_step(h: Tensor, prev_msg: Tensor,
              eff_prim: Tensor, eff_key: Tensor, eff_decay_exp: Tensor,
              cc_t: Tensor, conn_indices: Tensor,
              branch_w: Tensor | None, group_w: Tensor | None,
              use_tree: bool, tree_params: dict,
              C: int) -> tuple[Tensor, Tensor]:
    """Single token step. Pure function, autograd-friendly."""
    neighbor_msgs = prev_msg[:, conn_indices]
    sim = (eff_key.unsqueeze(2) * neighbor_msgs).sum(dim=-1)
    routing = torch.sigmoid(sim)
    weighted = routing.unsqueeze(-1) * neighbor_msgs

    if use_tree and branch_w is not None:
        received = _dendritic_gather_static(
            weighted, branch_w, group_w,
            tree_params['nb'], tree_params['bsz'],
            tree_params['ng'], tree_params['bpg'],
            tree_params['K'])
    else:
        received = weighted.sum(dim=2)

    received = received.clone()
    received[:, :C] = received[:, :C] + cc_t
    h_new = eff_decay_exp * h + (1 - eff_decay_exp) * received
    msg_new = torch.tanh(h_new * eff_prim)
    return h_new, msg_new


# ================================================================
# Custom autograd function: save only h, recompute in backward
# ================================================================

class _SegmentFunction(torch.autograd.Function):
    """Memory-efficient forward/backward for the per-token recurrence.

    Forward: runs T_seg steps without autograd, saves h at each step.
    Backward: replays each step in reverse with torch.autograd.grad,
    recomputing intermediates from saved h. Only 1 step's worth of
    activation memory at a time.

    VRAM: O(T_seg * BS * N * D) for saved h (~67 MB at tier_a BS=4)
    vs O(T_seg * BS * N * K * D) for naive autograd (~19 GB).
    """

    @staticmethod
    def forward(ctx, cc_signals, h_prev, prev_messages,
                eff_prim, eff_key, eff_decay,
                conn_indices, branch_w, group_w,
                # Non-tensor args
                use_tree, tree_params, stride, C):

        BS, T_seg, C_dim, D = cc_signals.shape
        N = h_prev.shape[1]
        eff_decay_exp = eff_decay.unsqueeze(-1)

        h = h_prev
        prev_msg = prev_messages
        h_saved = [h_prev.detach()]
        output_list = []

        with torch.no_grad():
            for t in range(T_seg):
                if t % stride == 0:
                    h, prev_msg = _one_step(
                        h, prev_msg, eff_prim, eff_key, eff_decay_exp,
                        cc_signals[:, t], conn_indices, branch_w, group_w,
                        use_tree, tree_params, C)
                    h_saved.append(h.detach())
                    output_list.append(prev_msg[:, :C].detach())
                else:
                    output_list.append(output_list[-1])

        output = torch.stack(output_list, dim=1)
        h_all = torch.stack(h_saved)  # [n_steps+1, BS, N, D]

        # Save everything needed for backward (tensors only via save_for_backward)
        ctx.save_for_backward(h_all, cc_signals, prev_messages,
                              eff_prim, eff_key, eff_decay,
                              conn_indices)
        # Save non-parameter tensors separately if they need grad
        ctx.branch_w = branch_w
        ctx.group_w = group_w
        ctx.use_tree = use_tree
        ctx.tree_params = tree_params
        ctx.stride = stride
        ctx.C = C
        ctx.T_seg = T_seg

        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_h_final):
        (h_all, cc_signals, prev_messages_init,
         eff_prim, eff_key, eff_decay,
         conn_indices) = ctx.saved_tensors
        branch_w = ctx.branch_w
        group_w = ctx.group_w
        T_seg = ctx.T_seg
        stride = ctx.stride
        C = ctx.C

        # Accumulate gradients for shared parameters
        grad_eff_prim = torch.zeros_like(eff_prim)
        grad_eff_key = torch.zeros_like(eff_key)
        grad_eff_decay = torch.zeros_like(eff_decay)
        grad_branch_w = torch.zeros_like(branch_w) if branch_w is not None else None
        grad_group_w = torch.zeros_like(group_w) if group_w is not None else None

        eff_decay_exp = eff_decay.unsqueeze(-1)

        # Current gradient on h
        dh = grad_h_final.clone() if grad_h_final is not None else torch.zeros_like(h_all[0])

        # Track which saved-h index we're at
        step_idx = h_all.shape[0] - 2  # last saved h before final

        for t in range(T_seg - 1, -1, -1):
            if t % stride != 0:
                continue

            # Gather output gradient for this step
            dout_t = grad_output[:, t, :, :]  # [BS, C, D]

            # Retrieve saved h_{t-1} and h_t
            h_prev_t = h_all[step_idx].detach().requires_grad_(True)
            step_idx -= 1

            # Determine prev_msg for this step
            if step_idx >= 0:
                # prev_msg comes from previous step's h
                h_for_msg = h_all[step_idx].detach()
            else:
                h_for_msg = None

            # Create leaf tensors for this step's grad computation
            ep = eff_prim.detach().requires_grad_(True)
            ek = eff_key.detach().requires_grad_(True)
            ed = eff_decay.detach().requires_grad_(True)
            ed_exp = ed.unsqueeze(-1)

            bw = branch_w.detach().requires_grad_(True) if branch_w is not None else None
            gw = group_w.detach().requires_grad_(True) if group_w is not None else None

            # Recompute step t with autograd
            with torch.enable_grad():
                if h_for_msg is not None:
                    pm = torch.tanh(h_for_msg * ep)
                else:
                    pm = prev_messages_init.detach()

                h_t, msg_t = _one_step(
                    h_prev_t, pm, ep, ek, ed_exp,
                    cc_signals[:, t], conn_indices, bw, gw,
                    ctx.use_tree, ctx.tree_params, C)

                port_out = msg_t[:, :C]

                # Loss contributions from this step
                targets = []
                grads = []

                # h_t contributes to future steps (dh) and output (dout_t through msg)
                targets.append(h_t)
                grads.append(dh)

                targets.append(port_out)
                grads.append(dout_t)

                # Compute gradients
                inputs = [h_prev_t, ep, ek, ed]
                if bw is not None:
                    inputs.append(bw)
                if gw is not None:
                    inputs.append(gw)

                local_grads = torch.autograd.grad(
                    targets, inputs, grads,
                    allow_unused=True)

            # Unpack and accumulate
            idx = 0
            dh = local_grads[idx] if local_grads[idx] is not None else torch.zeros_like(h_prev_t)
            idx += 1
            if local_grads[idx] is not None:
                grad_eff_prim = grad_eff_prim + local_grads[idx]
            idx += 1
            if local_grads[idx] is not None:
                grad_eff_key = grad_eff_key + local_grads[idx]
            idx += 1
            if local_grads[idx] is not None:
                grad_eff_decay = grad_eff_decay + local_grads[idx]
            idx += 1
            if branch_w is not None and local_grads[idx] is not None:
                grad_branch_w = grad_branch_w + local_grads[idx]
                idx += 1
            if group_w is not None and idx < len(local_grads) and local_grads[idx] is not None:
                grad_group_w = grad_group_w + local_grads[idx]

        # Return gradients in same order as forward inputs
        # (cc_signals, h_prev, prev_messages, eff_prim, eff_key, eff_decay,
        #  conn_indices, branch_w, group_w, + non-tensor args)
        return (None, None, None,  # cc_signals, h_prev, prev_messages
                grad_eff_prim, grad_eff_key, grad_eff_decay,
                None,  # conn_indices
                grad_branch_w, grad_group_w,
                None, None, None, None)  # non-tensor args


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

    def forward_segment(self, cc_signals: Tensor,
                        h_prev: Tensor) -> tuple[Tensor, Tensor]:
        """Process one segment using custom autograd for bounded VRAM.

        Uses _SegmentFunction: forward saves only h at each step (~67 MB),
        backward replays each step with torch.autograd.grad (~151 MB transient).
        Exact gradients through all tokens. Total: ~218 MB vs 19+ GB naive.

        Args:
            cc_signals: [BS, T_seg, C, D_mem] from lower scan (detached)
            h_prev: [BS, N, D_mem] detached from previous segment (TBPTT)
        Returns:
            output: [BS, T_seg, C, D_mem] port neuron messages
            h: [BS, N, D_mem] final h
        """
        BS, T_seg, C, D = cc_signals.shape
        stride = self.config.memory_update_stride

        # 1. Per-neuron modulator (on compute graph)
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

        # 2. Tree params
        tree_params = {}
        if self.use_dendritic_tree:
            tree_params = {
                'nb': self.n_branches, 'bsz': self.branch_size,
                'ng': self.n_groups, 'bpg': self.branches_per_group,
                'K': self.config.K_connections,
            }

        branch_w = self.dendrite_branch_w if self.use_dendritic_tree else None
        group_w = self.dendrite_group_w if self.use_dendritic_tree else None

        # 3. Custom autograd forward/backward
        output, h = _SegmentFunction.apply(
            cc_signals, h_prev, self.prev_messages.detach(),
            eff_prim, eff_key, eff_decay,
            self.conn_indices, branch_w, group_w,
            self.use_dendritic_tree, tree_params, stride, C)

        # 4. Side effects (outside autograd)
        with torch.no_grad():
            self.prev_messages = torch.tanh(
                h.detach() * eff_prim.detach()).detach()

            td = self.config.trace_decay
            self.trace_prim = (td * self.trace_prim
                               + (1 - td) * h.detach()).to(self.dtype)
            self.trace_key = (td * self.trace_key
                              + (1 - td) * self.prev_messages).to(self.dtype)

            seg_msg_mag = self.prev_messages.norm(dim=-1)
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
