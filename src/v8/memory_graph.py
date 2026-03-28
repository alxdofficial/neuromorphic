"""Memory Graph — differentiable per-token recurrence (v9-backprop).

N=4096 neurons, D_neuron=32, K=128 connections. Trained by backprop.

Per-step dynamics (64 steps per segment at stride=2):
  1. Gather neighbor messages via conn_indices, weight by sigmoid(w_conn)
  2. Dendritic tree integration (branch→tanh→group→tanh→soma)
  3. Add CC signal inject (deterministic: neuron n gets H_mid slice n%C_mem)
  4. State MLP: cat(input_vec, h) → Linear → tanh → Linear → tanh → h_new
  5. Message MLP: cat(h_new, primitive) → Linear → tanh → Linear → tanh → msg
  6. Add learnable neuron ID embedding
  7. Accumulate hebbian trace: |msg| * sigmoid(w_conn)

Segment-boundary modulator (runs FIRST each segment):
  mod(hebbian_traces, h, decay, primitive) → new w_conn, decay, primitive

Inject: H_mid [BS,T,D] → replicate → [BS,T,N,D_neuron]. No parameters.
Readout: msgs [BS,T,N,D_neuron] → reshape → mean over replicas → [BS,T,D].

Structural plasticity: at chunk boundaries, rewire connections based on
pairwise co-activation correlation. Non-differentiable.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config

# Try to import Triton kernels
try:
    from .triton_kernels import fused_dendritic_gather as _triton_gather
    from .triton_kernels import fused_neuron_step as _triton_step
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class MemoryGraph(nn.Module):
    """Differentiable memory graph with per-neuron modulator + MLPs.

    nn.Parameters (requires_grad=True, trained by backprop):
        mod_w1, mod_b1, mod_w2, mod_b2 — segment-boundary modulator
        state_w1, state_b1, state_w2, state_b2 — per-step state update MLP
        msg_w1, msg_b1, msg_w2, msg_b2 — per-step message MLP
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
        self.mod_w2 = nn.Parameter(
            torch.randn(N, H_mod, mod_output_dim, device=device) * 0.01)
        self.mod_b2 = nn.Parameter(
            torch.randn(N, mod_output_dim, device=device) * 0.01)

        # --- Per-step state update MLP ---
        # Input: cat(input_vec[D], h[D], decay[1]) = 2D+1
        # w1 layout: [N, H, I] (transposed for contiguous Triton access)
        state_in = 2 * D + 1
        H_state = config.state_mlp_hidden
        self.state_w1 = nn.Parameter(
            torch.randn(N, H_state, state_in, device=device) *
            (2.0 / (state_in + H_state)) ** 0.5)
        self.state_b1 = nn.Parameter(torch.zeros(N, H_state, device=device))
        self.state_w2 = nn.Parameter(
            torch.randn(N, H_state, D, device=device) * 0.01)
        self.state_b2 = nn.Parameter(torch.zeros(N, D, device=device))

        # --- Per-step message MLP ---
        # Input: cat(h_new[D], primitive[D]) = 2D
        # w1 layout: [N, H, I] (transposed)
        H_msg = config.msg_mlp_hidden
        self.msg_w1 = nn.Parameter(
            torch.randn(N, H_msg, 2 * D, device=device) *
            (2.0 / (2 * D + H_msg)) ** 0.5)
        self.msg_b1 = nn.Parameter(torch.zeros(N, H_msg, device=device))
        self.msg_w2 = nn.Parameter(
            torch.randn(N, H_msg, D, device=device) * 0.01)
        self.msg_b2 = nn.Parameter(torch.zeros(N, D, device=device))

        # --- Neuron ID embedding ---
        self.neuron_id = nn.Parameter(
            torch.randn(N, D, device=device) * 0.02)

        # --- Dendritic FC weights ---
        if self.use_dendritic_tree:
            nb, bs = self.n_branches, self.branch_size
            ng, bpg = self.n_groups, self.branches_per_group
            self.dendrite_branch_w = nn.Parameter(
                torch.full((N, nb, bs, D), 1.0 / bs, device=device))
            self.dendrite_group_w = nn.Parameter(
                torch.full((N, ng, bpg, D), 1.0 / bpg, device=device))

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

        # Structural plasticity
        if self.config.structural_plasticity:
            self.co_activation = torch.zeros(N, N, device=device,
                                             dtype=torch.float32)
            self._plasticity_segment_count = 0

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

        dt = self.mod_w1.dtype
        mod_input = torch.cat([
            hebbian_traces.to(dt),      # [BS, N, K]
            h.to(dt),                    # [BS, N, D]
            decay_logit.to(dt).unsqueeze(-1),  # [BS, N, 1]
            primitives.to(dt),           # [BS, N, D]
        ], dim=-1)  # [BS, N, K+2D+1]

        # Per-neuron MLP: einsum over neuron dim
        # w1 layout: [N, H, I] (transposed for Triton contiguous access)
        hidden = torch.einsum(
            'bni,nhi->bnh', mod_input, self.mod_w1
        ) + self.mod_b1  # [BS, N, H]
        hidden = torch.tanh(hidden)

        output = torch.einsum(
            'bnh,nho->bno', hidden, self.mod_w2
        ) + self.mod_b2  # [BS, N, K+1+D]

        new_w_conn = output[..., :K]              # [BS, N, K]
        new_decay_logit = output[..., K]           # [BS, N]
        new_primitives = output[..., K + 1:]       # [BS, N, D]

        return new_w_conn, new_decay_logit, new_primitives

    # ================================================================
    # Per-step MLPs
    # ================================================================

    def _state_mlp(self, input_vec: Tensor, h_prev: Tensor,
                   decay: Tensor) -> Tensor:
        """Per-neuron state update: cat(input, h_prev, decay) → h_new.

        decay (sigmoid of decay_logit) gives the MLP a persistence signal.
        Architecture: Linear → tanh → Linear → tanh (bounded output).
        """
        dt = self.state_w1.dtype
        x = torch.cat([input_vec.to(dt), h_prev.to(dt),
                        decay.to(dt).unsqueeze(-1)], dim=-1)  # [BS, N, 2D+1]
        hidden = torch.einsum(
            'bni,nhi->bnh', x, self.state_w1
        ) + self.state_b1
        hidden = torch.tanh(hidden)
        out = torch.einsum(
            'bnh,nhd->bnd', hidden, self.state_w2
        ) + self.state_b2
        return torch.tanh(out)

    def _msg_mlp(self, h_new: Tensor, primitives: Tensor) -> Tensor:
        """Per-neuron message generation: cat(h_new, primitive) → msg.

        Architecture: Linear → tanh → Linear → tanh (bounded output).
        """
        dt = self.msg_w1.dtype
        x = torch.cat([h_new.to(dt), primitives.to(dt)], dim=-1)  # [BS, N, 2D]
        hidden = torch.einsum(
            'bni,nhi->bnh', x, self.msg_w1
        ) + self.msg_b1
        hidden = torch.tanh(hidden)
        out = torch.einsum(
            'bnh,nhd->bnd', hidden, self.msg_w2
        ) + self.msg_b2
        return torch.tanh(out)

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

    def inject(self, H_mid_seg: Tensor) -> Tensor:
        """H_mid segment → per-neuron CC signals by replication.

        Args:
            H_mid_seg: [BS, T_seg, D_lm] — detached LM hidden states

        Returns:
            inject_bc: [BS, T_seg, N, D_neuron]
        """
        BS, T_seg, D_lm = H_mid_seg.shape
        slices = H_mid_seg.view(BS, T_seg, self.C_mem, self.config.D_neuron)
        return slices.unsqueeze(3).expand(
            -1, -1, -1, self.N_per_slice, -1
        ).reshape(BS, T_seg, self.config.N_neurons, self.config.D_neuron)

    def readout(self, msg_all: Tensor) -> Tensor:
        """All neuron messages → LM hidden dim by averaging replicas.

        Args:
            msg_all: [BS, T_seg, N, D_neuron]

        Returns:
            mem_out: [BS, T_seg, D_lm]
        """
        BS, T_seg = msg_all.shape[:2]
        grouped = msg_all.view(
            BS, T_seg, self.C_mem, self.N_per_slice, self.config.D_neuron)
        averaged = grouped.mean(dim=3)
        return averaged.reshape(BS, T_seg, self.config.D)

    # ================================================================
    # Forward segment (differentiable)
    # ================================================================

    def _step_group(self, h: Tensor, prev_msg: Tensor,
                    inject_group: Tensor, w_conn_sig: Tensor,
                    decay: Tensor, primitives: Tensor,
                    neuron_id: Tensor,
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run a group of token steps. Pure function for gradient checkpointing.

        Args:
            decay: [BS, N] — sigmoid(decay_logit), fed into state MLP

        Returns:
            h: [BS, N, D] — final hidden state
            prev_msg: [BS, N, D] — final messages
            msgs: [BS, G, N, D] — messages at each step
            hebbian_accum: [BS, N, K] — accumulated hebbian traces
        """
        G = inject_group.shape[1]
        BS, N, K = w_conn_sig.shape
        msgs = []
        hebbian_accum = torch.zeros(BS, N, K, device=h.device,
                                    dtype=h.dtype)

        for g in range(G):
            # 1-3. Fused gather + weight + dendritic tree (Triton on CUDA)
            received = self._fused_gather(prev_msg, w_conn_sig)
            # 4. Add CC signal inject
            input_vec = received + inject_group[:, g]
            # 5. State MLP (decay provides persistence signal)
            h = self._state_mlp(input_vec, h, decay)
            # 6. Message MLP
            msg = self._msg_mlp(h, primitives)
            # 7. Add neuron ID embedding
            msg = msg + neuron_id
            prev_msg = msg
            msgs.append(msg)

            # 8. Hebbian trace: |msg| * sigmoid(w_conn)
            msg_mag = msg.norm(dim=-1, keepdim=True)  # [BS, N, 1]
            hebbian_accum = hebbian_accum + msg_mag * w_conn_sig

        return h, prev_msg, torch.stack(msgs, dim=1), hebbian_accum

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment of tokens. Differentiable through modulator + MLPs.

        Uses gradient checkpointing to limit VRAM: the token loop is split into
        groups, each checkpointed.

        Args:
            cc_signals: [BS, T_seg, D_lm] — detached H_mid for this segment

        Returns:
            mem_out: [BS, T_seg, D_lm] — readout signal to inject into LM
        """
        BS = cc_signals.shape[0]
        T_seg = cc_signals.shape[1]
        N = self.config.N_neurons
        K = self.config.K_connections
        stride = self.config.memory_update_stride

        # TBPTT: detach state at segment boundary
        h = self.h.detach()
        prev_msg = self.prev_messages.detach()

        # Run modulator FIRST (on compute graph — this is what backprop trains)
        w_conn, decay_logit, primitives = self._run_modulator(
            h, self.hebbian_traces.detach(),
            self.decay_logit.detach(),
            self.primitives_state.detach())

        # Precompute sigmoid (on graph through w_conn/decay_logit)
        w_conn_sig = torch.sigmoid(w_conn)      # [BS, N, K]
        decay = torch.sigmoid(decay_logit)       # [BS, N]

        # Inject: replicate H_mid → per-neuron signals at stride positions
        inject_bc = self.inject(cc_signals)
        inject_steps = inject_bc[:, ::stride]
        n_steps = inject_steps.shape[1]

        # Run all steps in one group (no gradient checkpointing — VRAM is sufficient)
        msg_groups = []
        total_hebbian = torch.zeros(BS, N, K, device=h.device,
                                    dtype=h.dtype)

        h, prev_msg, all_msgs, total_hebbian = self._step_group(
            h, prev_msg, inject_steps,
            w_conn_sig, decay, primitives, self.neuron_id)
        msg_groups.append(all_msgs)

        # Concatenate all step messages: [BS, n_steps, N, D]
        msg_all_steps = torch.cat(msg_groups, dim=1)

        # If stride > 1, repeat to fill all T_seg positions
        if stride > 1:
            msg_all = msg_all_steps.repeat_interleave(stride, dim=1)[:, :T_seg]
        else:
            msg_all = msg_all_steps

        # Readout: average replicas → [BS, T_seg, D_lm]
        mem_out = self.readout(msg_all)

        # Update persistent state (detached, for next segment)
        with torch.no_grad():
            self.h = h.detach()
            self.prev_messages = prev_msg.detach()
            self.w_conn = w_conn.detach()
            self.primitives_state = primitives.detach()
            self.decay_logit = decay_logit.detach()

            # Hebbian traces: per-segment average
            self.hebbian_traces = (total_hebbian / max(n_steps, 1)).to(self.dtype)

            # Structural plasticity accumulation
            if (self.config.structural_plasticity and
                    hasattr(self, 'co_activation')):
                msg_mag = prev_msg.detach().float().norm(dim=-1)  # [BS, N]
                mag_mean = msg_mag.mean(dim=0)  # [N]
                self.co_activation += torch.outer(mag_mean, mag_mean)
                self._plasticity_segment_count += 1

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

    def rewire_connections(self):
        """Structural plasticity: prune weak connections, create strong ones.

        Called at chunk boundaries. Non-differentiable.
        Modifies conn_indices buffer in-place.
        """
        if not self.config.structural_plasticity:
            return
        if not hasattr(self, 'co_activation'):
            return
        if self._plasticity_segment_count == 0:
            return

        N = self.config.N_neurons
        K = self.config.K_connections
        n_swap = min(self.config.plasticity_n_swap, K)

        # Normalize co-activation by number of segments
        co_act = self.co_activation / self._plasticity_segment_count
        co_act.fill_diagonal_(0.0)  # no self-connections

        conn = self.conn_indices  # [N, K]

        with torch.no_grad():
            for n in range(N):
                current_neighbors = conn[n]  # [K]

                # Strength of current connections
                current_strength = co_act[n, current_neighbors]  # [K]

                # Find weakest current connections
                _, weakest_idx = current_strength.topk(n_swap, largest=False)

                # Mask out current connections
                mask = torch.ones(N, device=co_act.device, dtype=torch.bool)
                mask[current_neighbors] = False
                mask[n] = False

                # Find strongest non-connected neurons
                candidates = co_act[n].clone()
                candidates[~mask] = -float('inf')
                _, strongest_new = candidates.topk(n_swap, largest=True)

                # Swap
                conn[n, weakest_idx] = strongest_new

            # Re-sort for efficient gather
            sorted_idx, _ = conn.sort(dim=-1)
            self.conn_indices.copy_(sorted_idx)

        # Store for diagnostics
        self._last_rewire_swaps = n_swap * N

        # Reset accumulator
        self.co_activation.zero_()
        self._plasticity_segment_count = 0

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
                hasattr(self, 'co_activation')):
            state['co_activation'] = self.co_activation
            state['_plasticity_segment_count'] = self._plasticity_segment_count
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
