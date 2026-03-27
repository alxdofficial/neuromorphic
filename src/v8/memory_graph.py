"""Memory Graph — per-token recurrence with inter-neuron message passing.

At each token timestep within a segment:
  1. Receive: dendritic tree gather from K neighbors (+ CC signal for ports)
     - Connections split into branches, each independently summed + tanh
     - Branches grouped, each group summed + tanh
     - Soma averages groups → received signal
     This creates 3 levels of nonlinear processing (Poirazi 2003).
  2. Integrate: h = decay * h + (1-decay) * received
  3. Message: prev_messages = tanh(h * primitives)

Neurons exchange messages at every timestep, allowing multi-hop
propagation through the graph within a single segment.

State (persistent across chunks, outside autograd):
  primitives: [BS, N, D_mem] — per-neuron message modulation
  key: [BS, N, D_mem] — per-neuron routing selectivity
  decay_logit: [BS, N] — per-neuron decay (pre-sigmoid)
  h: [BS, N, D_mem] — internal state (temporal memory)
  prev_messages: [BS, N, D_mem] — last outgoing messages

Plasticity metrics:
  co_activation_ema: [N, N] — phi coefficient matrix for structural plasticity
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config

try:
    import triton
    from .triton_kernels import memory_graph_routing_kernel, memory_graph_step_kernel
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class MemoryGraph:
    """Per-token recurrent memory graph with sparse message passing.

    Architecture per token within a segment:
        1. received = dendritic_tree_gather(prev_messages) (+ CC signal for ports)
        2. h = decay * h + (1-decay) * received
        3. prev_messages = tanh(h * primitives)
    After full segment → read port neuron messages → mem_signals
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.device = device
        self.dtype = dtype

        N = config.N_neurons
        K_conn = config.K_connections

        # Fixed topology: random sparse connectivity (vectorized, GPU-friendly)
        # For each neuron, sample K_conn random neighbors excluding self.
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')  # exclude self-connections
        K_actual = min(K_conn, N - 1)
        conn_indices = torch.zeros(N, K_conn, dtype=torch.long, device=device)
        conn_mask = torch.ones(N, K_conn, dtype=torch.bool, device=device)
        conn_indices[:, :K_actual] = scores.topk(K_actual, dim=1).indices
        if K_actual < K_conn:
            conn_mask[:, K_actual:] = False

        self.conn_indices = conn_indices  # [N, K_conn]
        self.conn_mask = conn_mask        # [N, K_conn]
        self._sort_conn_indices()  # sort for cache locality in sparse gather

        # Dendritic tree structure (derived from config)
        branch_size = config.dendrite_branch_size
        if branch_size > 0 and K_conn >= branch_size:
            self.n_branches = K_conn // branch_size
            self.branch_size = branch_size
            # Group branches: up to 4 per group, at least 1 group
            self.branches_per_group = min(4, self.n_branches)
            self.n_groups = max(1, self.n_branches // self.branches_per_group)
            # Adjust branches_per_group if n_branches not evenly divisible
            self.branches_per_group = self.n_branches // self.n_groups
            self.use_dendritic_tree = True
        else:
            self.use_dendritic_tree = False

        # CC port assignment: first C neurons are CC ports
        self.cc_port_idx = torch.arange(config.C, device=device)

        # State tensors (set in initialize())
        self._initialized = False

    def _sort_conn_indices(self):
        """Sort conn_indices per neuron for cache-friendly sparse gather."""
        sorted_idx, order = self.conn_indices.sort(dim=-1)  # [N, K]
        self.conn_indices = sorted_idx
        self.conn_mask = self.conn_mask.gather(1, order)

    @torch.no_grad()
    def _compute_routing_weights(self):
        """Compute sigmoid routing weights from key × neighbor messages.

        Each connection is independently gated by sigmoid(key · neighbor_msg).
        Unlike softmax, strong connections don't suppress weak ones — a neuron
        can attend to multiple strong sources simultaneously without dilution.

        Called once per segment. Produces [BS, N, K] scalar weights used
        by the fast sparse Triton kernel for the entire segment.
        """
        BS = self.prev_messages.shape[0]
        N = self.config.N_neurons
        K = self.config.K_connections

        # Gather neighbor messages: [BS, N, K, D]
        neighbor_msgs = self.prev_messages[:, self.conn_indices]

        # Raw dot product: key[n] · neighbor_msg[k] for each connection
        sim = (self.key.unsqueeze(2) * neighbor_msgs).sum(dim=-1)  # [BS, N, K]

        # Sigmoid → per-connection independent gate (no normalization)
        self._routing_weights = torch.sigmoid(sim).to(self.dtype)  # [BS, N, K]

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, BS: int):
        """Initialize persistent state for batch size BS."""
        N = self.config.N_neurons
        D = self.config.D_mem
        K_conn = self.config.K_connections

        # Per-neuron parameters (neuromodulator-controlled)
        # Random directions, RMS-normalized — diverse from the start so
        # messages differ across neurons and routing can differentiate
        prim_raw = torch.randn(BS, N, D, device=self.device, dtype=self.dtype)
        rms = prim_raw.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.primitives = prim_raw / rms

        key_raw = torch.randn(BS, N, D, device=self.device, dtype=self.dtype)
        key_rms = key_raw.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.key = key_raw / key_rms

        self.decay_logit = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Persistent neuron state — small random init so first messages
        # aren't all zero (preserves primitive diversity from step 1)
        self.h = torch.randn(
            BS, N, D, device=self.device, dtype=self.dtype) * 0.1
        self.prev_messages = torch.tanh(
            self.h * self.primitives)

        # Running stats
        self.mean_input = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.mean_output = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)

        # Per-neuron activity
        self.msg_magnitude = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Eligibility traces (Hebbian, accumulated via EMA)
        self.trace_prim = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.trace_key = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)

        # Co-activation matrix
        self.co_activation_ema = torch.zeros(
            N, N, device=self.device, dtype=torch.float32)

        # Plasticity tracking
        self._plasticity_rewires = 0
        self._co_activation_ready = False

        # Triton kernel buffers
        if _HAS_TRITON and self.device.type == 'cuda':
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
            self._decay_f32 = torch.zeros(
                BS, N, device=self.device, dtype=torch.float32)
            self._triton_ready = True
        else:
            self._triton_ready = False

        self._initialized = True

    def _dendritic_gather(self, prev_msg: Tensor, routing_w: Tensor) -> Tensor:
        """Dendritic tree gather: branch → group → soma.

        Connections are organized into a tree:
          Level 0 (distal):   n_branches branches, each sums branch_size connections → tanh
          Level 1 (proximal): n_groups groups, each sums branches_per_group branches → tanh
          Level 2 (soma):     averages n_groups group outputs → received

        This creates 3 levels of nonlinear integration (no trainable params).

        Args:
            prev_msg: [BS, N, D] — previous messages from all neurons
            routing_w: [BS, N, K] — precomputed sigmoid routing weights

        Returns:
            received: [BS, N, D] — dendritic-processed input per neuron
        """
        BS = prev_msg.shape[0]
        N = self.config.N_neurons
        D = prev_msg.shape[2]
        K = self.config.K_connections

        # Gather all neighbor messages: [BS, N, K, D]
        neighbor_msgs = prev_msg[:, self.conn_indices]  # [BS, N, K, D]
        # Apply routing weights: [BS, N, K, 1] * [BS, N, K, D] → weighted messages
        weighted = routing_w.unsqueeze(-1) * neighbor_msgs  # [BS, N, K, D]

        bsz = self.branch_size
        bpg = self.branches_per_group
        ng = self.n_groups

        # Reshape into tree: [BS, N, n_groups, branches_per_group, branch_size, D]
        n_tree = ng * bpg * bsz  # connections covered by tree
        tree_msgs = weighted[:, :, :n_tree].view(BS, N, ng, bpg, bsz, D)

        # Level 0: sum within each branch → tanh
        branch_sums = tree_msgs.sum(dim=4)  # [BS, N, ng, bpg, D]
        branch_out = torch.tanh(branch_sums)

        # Level 1: sum branches within each group → tanh
        group_sums = branch_out.sum(dim=3)  # [BS, N, ng, D]
        group_out = torch.tanh(group_sums)

        # Level 2: average groups → soma
        received = group_out.mean(dim=2)  # [BS, N, D]

        # Handle leftover connections (if K not evenly divisible)
        if n_tree < K:
            leftover = weighted[:, :, n_tree:].sum(dim=2)  # [BS, N, D]
            # Blend leftover with tree output proportionally
            tree_frac = n_tree / K
            received = tree_frac * received + (1 - tree_frac) * leftover

        return received

    @torch.no_grad()
    def forward_segment(self, cc_signals: Tensor,
                        eot_mask: Tensor | None = None,
                        update_co_activation: bool = True) -> Tensor:
        """Process a segment of tokens with per-token neuron dynamics.

        Dispatches to Triton kernel on CUDA, Python fallback on CPU.
        """
        if self._triton_ready and cc_signals.is_cuda:
            return self._forward_segment_triton(cc_signals, eot_mask,
                                                update_co_activation)
        return self._forward_segment_python(cc_signals, eot_mask,
                                            update_co_activation)

    def _forward_segment_python(self, cc_signals: Tensor,
                                eot_mask: Tensor | None = None,
                                update_co_activation: bool = True) -> Tensor:
        """Python reference implementation of per-token neuron dynamics.

        Uses dendritic tree gather when enabled: connections organized into
        branches and groups with tanh nonlinearity at each level.
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons

        decay = torch.sigmoid(self.decay_logit).unsqueeze(-1)  # [BS, N, 1]
        one_minus_decay = 1.0 - decay

        has_eot = None
        if eot_mask is not None and eot_mask.any():
            has_eot = eot_mask.any(dim=0).tolist()

        # Compute routing weights once per segment
        self._compute_routing_weights()

        h = self.h
        prev_msg = self.prev_messages
        output = torch.empty(BS, T_seg, C, D, device=self.device, dtype=self.dtype)

        act_trace = torch.empty(BS, T_seg, N, device=self.device, dtype=torch.float32)
        received_accum = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)
        msg_accum = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)

        stride = self.config.memory_update_stride

        for t in range(T_seg):
            if t % stride == 0:
                # Full neuron dynamics step
                # 1. Receive: dendritic tree or flat gather
                if self.use_dendritic_tree:
                    received = self._dendritic_gather(prev_msg, self._routing_weights)
                else:
                    neighbor_msgs = prev_msg[:, self.conn_indices]  # [BS, N, K, D]
                    received = (self._routing_weights.unsqueeze(-1) * neighbor_msgs).sum(dim=2)

                # Port neurons also receive external CC signal from LM
                received[:, :C] = received[:, :C] + cc_signals[:, t]

                # 2. Integrate
                if has_eot is not None and has_eot[t]:
                    eot_t = eot_mask[:, t].view(BS, 1, 1).to(dtype=h.dtype)
                    d_t = decay * (1.0 - eot_t)
                    omd_t = 1.0 - d_t
                    h = d_t * h + omd_t * received
                else:
                    h = decay * h + one_minus_decay * received

                # 3. Message
                prev_msg = torch.tanh(h * self.primitives)

                received_accum += received
                msg_accum += prev_msg

            # Always record output (held between updates)
            act_trace[:, t] = prev_msg.norm(dim=-1).float()
            output[:, t] = prev_msg[:, :C]

        self.h = h
        self.prev_messages = prev_msg
        n_steps = T_seg // stride
        self.mean_input = received_accum / max(n_steps, 1)
        self.mean_output = msg_accum / max(n_steps, 1)

        self._post_segment_stats(prev_msg, act_trace,
                                 update_co_activation=update_co_activation)
        return output

    def _forward_segment_triton(self, cc_signals: Tensor,
                                eot_mask: Tensor | None = None,
                                update_co_activation: bool = True) -> Tensor:
        """Triton-accelerated per-token neuron dynamics.

        Routing weights computed by a separate Triton kernel (no Python-side
        [BS,N,K,D] gather). Step kernel launched per token with dendritic tree.
        msg_magnitude accumulated in-kernel; act_trace only allocated when
        co-activation phi is needed.
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.K_connections

        self._decay_f32.copy_(torch.sigmoid(self.decay_logit))

        if eot_mask is not None:
            eot_flags = eot_mask.to(dtype=torch.float32).contiguous()
        else:
            eot_flags = torch.zeros(BS, T_seg, device=self.device, dtype=torch.float32)

        cc_signals = cc_signals.contiguous()
        output = torch.empty(BS, T_seg, C, D, device=self.device, dtype=self.dtype)
        grid = (BS, N)

        recv_accum = torch.zeros(BS, N, D, device=self.device, dtype=torch.float32)
        msg_accum = torch.zeros(BS, N, D, device=self.device, dtype=torch.float32)
        msg_mag_accum = torch.zeros(BS, N, device=self.device, dtype=torch.float32)

        # Only allocate act_trace when co-activation needs it
        if update_co_activation:
            act_trace = torch.empty(BS, T_seg, N, device=self.device, dtype=torch.float32)
            act_trace_ptr = act_trace
        else:
            act_trace = None
            # Dummy pointer (kernel won't write to it when WRITE_ACT_TRACE=0)
            act_trace_ptr = msg_mag_accum  # reuse any valid ptr

        # Compute routing weights via Triton kernel
        routing_w = torch.empty(BS, N, K, device=self.device, dtype=self.dtype)
        memory_graph_routing_kernel[grid](
            self.prev_messages, self.key.contiguous(),
            self._conn_idx_i32, routing_w,
            BS=BS, N=N, D=D, K=K,
        )

        # Dendritic tree parameters
        if self.use_dendritic_tree:
            branch_size = self.branch_size
            branches_per_group = self.branches_per_group
            n_groups = self.n_groups
        else:
            branch_size = K
            branches_per_group = 1
            n_groups = 1

        write_trace = 1 if update_co_activation else 0
        stride = self.config.memory_update_stride

        # Only launch kernel at stride intervals
        update_steps = range(0, T_seg, stride)
        for t in update_steps:
            memory_graph_step_kernel[grid](
                self.h, self.prev_messages,
                self._conn_idx_i32, routing_w,
                self._decay_f32, self.primitives,
                cc_signals, eot_flags, output,
                recv_accum, msg_accum, msg_mag_accum,
                act_trace_ptr,
                t,
                BS=BS, N=N, D=D, K=K, C=C, T_seg=T_seg,
                BRANCH_SIZE=branch_size,
                BRANCHES_PER_GROUP=branches_per_group,
                N_GROUPS=n_groups,
                WRITE_ACT_TRACE=write_trace,
            )

        # Fill held output slots by copying from update positions
        if stride > 1:
            for s in range(1, stride):
                output[:, s::stride] = output[:, ::stride]

        # Update stats from accumulators (divided by actual dynamics steps, not T_seg)
        n_steps = T_seg // stride
        self.mean_input = (recv_accum / max(n_steps, 1)).to(self.dtype)
        self.mean_output = (msg_accum / max(n_steps, 1)).to(self.dtype)

        # msg_magnitude EMA from accumulated norms
        stats_alpha = 1.0 - self.config.plasticity_ema_decay
        seg_msg_mag = msg_mag_accum / max(n_steps, 1)
        self.msg_magnitude = ((1 - stats_alpha) * self.msg_magnitude
                              + stats_alpha * seg_msg_mag)

        # Co-activation stats only when needed (avoids quantile + bmm overhead)
        if update_co_activation and act_trace is not None:
            # Fill held act_trace positions (kernel only wrote at stride intervals)
            if stride > 1:
                for s in range(1, stride):
                    act_trace[:, s::stride] = act_trace[:, ::stride]
            self._post_segment_stats(self.prev_messages, act_trace,
                                     update_co_activation=True)

        return output

    def _post_segment_stats(self, prev_msg: Tensor, act_trace: Tensor = None,
                            update_co_activation: bool = True):
        """Compute co-activation phi from act_trace (only when needed).

        msg_magnitude is updated separately: Python path does it here,
        Triton path accumulates in-kernel via msg_mag_accum.
        """
        if act_trace is None:
            return

        BS, T_seg, N = act_trace.shape

        # msg_magnitude EMA (Python path only — Triton handles this in-kernel)
        if not (hasattr(self, '_triton_ready') and self._triton_ready
                and act_trace.is_cuda):
            stats_alpha = 1.0 - self.config.plasticity_ema_decay
            seg_msg_mag = act_trace.mean(dim=1)
            self.msg_magnitude = ((1 - stats_alpha) * self.msg_magnitude
                                  + stats_alpha * seg_msg_mag)

        if not update_co_activation:
            return

        # Co-activation phi: binary firing from 75th percentile threshold
        threshold = torch.quantile(
            act_trace.float(), 0.75, dim=1, keepdim=True)
        fired = (act_trace > threshold).float()

        p_i = fired.mean(dim=1, keepdim=True)
        fired_centered = fired - p_i
        var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)
        cov = torch.bmm(
            fired_centered.transpose(1, 2),
            fired_centered,
        ) / T_seg
        std_i = var_i.sqrt().unsqueeze(2)
        std_j = var_i.sqrt().unsqueeze(1)
        phi = cov / (std_i * std_j).clamp(min=1e-8)

        phi_mean = phi.mean(dim=0).float()
        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = ca_decay * self.co_activation_ema + (1 - ca_decay) * phi_mean
        self._co_activation_ready = True

    @torch.no_grad()
    def compute_eligibility_traces(self):
        """Accumulate Hebbian eligibility traces from current neuron state.

        Called after forward_segment so h and mean_input reflect this segment's
        activity. Traces are EMA-smoothed: consistent signals build up,
        contradictory signals cancel out.

        trace_prim ← decay * trace_prim + (1-decay) * h
            "shift primitives toward what I consistently encode"
        trace_key  ← decay * trace_key  + (1-decay) * mean_input
            "shift key toward what I consistently receive"
        """
        td = self.config.trace_decay
        self.trace_prim = (td * self.trace_prim + (1 - td) * self.h).to(self.dtype)
        self.trace_key = (td * self.trace_key + (1 - td) * self.mean_input).to(self.dtype)

    @torch.no_grad()
    def apply_gated_plasticity(self, gate: Tensor, decay_target: Tensor):
        """Apply neuromod-gated Hebbian update + decay blend.

        Three-factor learning: eligibility trace × neuromod gate → parameter update.
        gate > 0: consolidate (apply Hebbian update — reinforce current patterns)
        gate < 0: explore (reverse Hebbian update — break current patterns)
        gate ≈ 0: maintain (no change)

        Args:
            gate: [BS, N] in [-1, 1] — Hebbian gate per neuron
            decay_target: [BS, N] — target for decay_logit
        """
        lr = self.config.hebbian_lr
        g = gate.unsqueeze(-1)  # [BS, N, 1] for broadcasting over D

        # Normalize traces to unit direction (magnitude encoded separately in obs)
        prim_dir = self.trace_prim / self.trace_prim.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        key_dir = self.trace_key / self.trace_key.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Gated Hebbian update on primitives + RMS normalize
        self.primitives = self.primitives + lr * g * prim_dir
        rms = self.primitives.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.primitives = (self.primitives / rms).to(self.dtype)

        # Gated Hebbian update on key + RMS normalize
        self.key = self.key + lr * g * key_dir
        key_rms = self.key.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.key = (self.key / key_rms).to(self.dtype)

        # Decay: blend toward target
        alpha = 0.1
        self.decay_logit = ((1 - alpha) * self.decay_logit + alpha * decay_target).to(self.dtype)

    @torch.no_grad()
    def structural_plasticity(self):
        """Autonomous co-activation-based structural plasticity.

        Prune connections where phi < 0, grow toward highest phi.
        80% correlation-guided, 20% random exploration.
        """
        N = self.config.N_neurons
        K_conn = self.config.K_connections
        explore_frac = self.config.plasticity_exploration_frac
        phi = self.co_activation_ema

        if not self._co_activation_ready:
            return

        conn_phi = phi[torch.arange(N, device=self.device).unsqueeze(1),
                       self.conn_indices]

        worst_phi, worst_k = conn_phi.min(dim=-1)
        prune_mask = worst_phi < 0
        prune_neurons = prune_mask.nonzero(as_tuple=True)[0]

        if prune_neurons.shape[0] == 0:
            return

        n_prune = prune_neurons.shape[0]

        existing_mask = torch.zeros(n_prune, N, dtype=torch.bool, device=self.device)
        prune_conn_idx = self.conn_indices[prune_neurons]
        existing_mask.scatter_(1, prune_conn_idx, True)
        existing_mask[torch.arange(n_prune, device=self.device), prune_neurons] = True

        phi_masked = phi[prune_neurons].clone()
        phi_masked[existing_mask] = -float('inf')

        best_targets = phi_masked.argmax(dim=-1)
        best_phi = phi_masked[torch.arange(n_prune, device=self.device), best_targets]

        random_targets = torch.randint(0, N, (n_prune,), device=self.device)
        use_random = torch.rand(n_prune, device=self.device) < explore_frac
        use_random = use_random | (best_phi <= 0)

        new_targets = torch.where(use_random, random_targets, best_targets)

        worst_k_for_prune = worst_k[prune_neurons]
        self.conn_indices[prune_neurons, worst_k_for_prune] = new_targets

        self._plasticity_rewires += n_prune
        self._sort_conn_indices()

        if self._triton_ready:
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()

    @torch.no_grad()
    def get_neuron_obs(self) -> Tensor:
        """Build observation tensor for neuromodulator.

        The neuromod sees the neuron's full state to decide:
        - gate: whether the Hebbian trace should be applied
        - decay_target: what temporal persistence this neuron should have
        """
        parts = [
            self.primitives,                                    # [BS, N, D_mem]
            self.key,                                           # [BS, N, D_mem]
            self.mean_input,                                    # [BS, N, D_mem]
            self.mean_output,                                   # [BS, N, D_mem]
            self.msg_magnitude.unsqueeze(-1),                   # [BS, N, 1]
            torch.sigmoid(self.decay_logit).unsqueeze(-1),      # [BS, N, 1]
            self.trace_prim.norm(dim=-1, keepdim=True),         # [BS, N, 1]
            self.trace_key.norm(dim=-1, keepdim=True),          # [BS, N, 1]
        ]
        return torch.cat(parts, dim=-1)

    @property
    def obs_dim(self) -> int:
        # D_mem*4 (prim + key + mean_in + mean_out) + 4 (msg_mag + decay + trace_norms)
        return self.config.D_mem * 4 + 4

    def state_dict(self) -> dict:
        """Export full memory graph state for checkpointing."""
        return {
            'primitives': self.primitives,
            'key': self.key,
            'decay_logit': self.decay_logit,
            'conn_indices': self.conn_indices,
            'conn_mask': self.conn_mask,
            'h': self.h,
            'prev_messages': self.prev_messages,
            'mean_input': self.mean_input,
            'mean_output': self.mean_output,
            'msg_magnitude': self.msg_magnitude,
            'trace_prim': self.trace_prim,
            'trace_key': self.trace_key,
            'co_activation_ema': self.co_activation_ema,
        }

    def load_state_dict(self, state: dict):
        """Restore memory graph state from checkpoint."""
        for key, val in state.items():
            setattr(self, key, val)
        if self._triton_ready:
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
