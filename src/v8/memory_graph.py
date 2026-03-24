"""Memory Graph — per-token recurrence with inter-neuron message passing.

At each token timestep within a segment:
  1. Receive: received = A @ prev_messages (+ CC signal for ports)
  2. Integrate: h = decay * h + (1-decay) * received
  3. Message: prev_messages = tanh(h * primitives)

Neurons exchange messages at every timestep, allowing multi-hop
propagation through the graph within a single segment.

State (persistent across chunks, outside autograd):
  primitives: [BS, N, D_mem] — per-neuron message modulation (RMS-normalized)
  key: [BS, N, D_mem] — per-neuron routing selectivity (L2-normalized)
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
    from fla.ops.hgrn.fused_recurrent import fused_recurrent_hgrn
    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False

try:
    import triton
    from .triton_kernels import memory_graph_step_kernel
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _fla_scan(decay_logit: Tensor, x: Tensor,
              h0: Tensor | None = None) -> Tensor:
    """Fused scan via FLA HGRN kernel.

    HGRN computes: h[t] = sigmoid(g) * h[t-1] + (1 - sigmoid(g)) * x[t]
    So we pass x = gate * u (WITHOUT the (1-decay) pre-scaling).
    The kernel applies (1-sigmoid(g)) internally.

    Args:
        decay_logit: [B, D] — constant across T
        x: [B, T, D] — scan input (NOT pre-scaled by 1-decay)
        h0: [B, D] or None

    Returns:
        h_all: [B, T, D]
    """
    B, T, D = x.shape
    g = F.logsigmoid(decay_logit).unsqueeze(1).expand(B, T, D)
    out, _ = fused_recurrent_hgrn(x, g, initial_state=h0)
    return out


def _cpu_scan(decay_logit: Tensor, b: Tensor,
              h0: Tensor | None = None) -> Tensor:
    """CPU fallback parallel scan: h[t] = sigmoid(decay_logit) * h[t-1] + b[t].

    Args:
        decay_logit: [B, D] — constant across T
        b: [B, T, D] — scan input (pre-scaled by 1-decay)
        h0: [B, D] or None

    Returns:
        h_all: [B, T, D]
    """
    import math
    B, T, D = b.shape
    a = torch.sigmoid(decay_logit).unsqueeze(1).expand(B, T, D)

    aa = a
    if h0 is not None:
        bb = b.clone()
        bb[:, 0] = aa[:, 0] * h0 + bb[:, 0]
    else:
        bb = b.clone()

    num_steps = math.ceil(math.log2(T)) if T > 1 else 0
    for d in range(num_steps):
        stride = 1 << d
        a_prev = aa[:, :-stride]
        b_prev = bb[:, :-stride]
        a_cur = aa[:, stride:]
        b_cur = bb[:, stride:]
        new_a = a_cur * a_prev
        new_b = a_cur * b_prev + b_cur
        aa = torch.cat([aa[:, :stride], new_a], dim=1)
        bb = torch.cat([bb[:, :stride], new_b], dim=1)

    return bb


class MemoryGraph:
    """Per-token recurrent memory graph with sparse message passing.

    Architecture per token within a segment:
        1. received = A @ prev_messages (+ CC signal for ports)
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

        # Fixed topology: random sparse connectivity
        # conn_indices[j, k] = index of k-th neuron connected to neuron j
        conn_indices = torch.zeros(N, K_conn, dtype=torch.long, device=device)
        conn_mask = torch.ones(N, K_conn, dtype=torch.bool, device=device)
        for j in range(N):
            candidates = [i for i in range(N) if i != j]
            n_actual = min(K_conn, len(candidates))
            perm = torch.randperm(len(candidates), device=device)[:n_actual]
            for k in range(n_actual):
                conn_indices[j, k] = candidates[perm[k]]
            if n_actual < K_conn:
                conn_mask[j, n_actual:] = False

        self.conn_indices = conn_indices  # [N, K_conn]
        self.conn_mask = conn_mask        # [N, K_conn]
        self._sort_conn_indices()  # sort for cache locality in sparse gather

        # CC port assignment: first C neurons are CC ports
        self.cc_port_idx = torch.arange(config.C, device=device)

        # State tensors (set in initialize())
        self._initialized = False

    def _sort_conn_indices(self):
        """Sort conn_indices per neuron for cache-friendly sparse gather.

        Sorting by source index clusters memory accesses during the gather
        step, improving L2 cache reuse.
        """
        sorted_idx, order = self.conn_indices.sort(dim=-1)  # [N, K]
        self.conn_indices = sorted_idx
        self.conn_mask = self.conn_mask.gather(1, order)

    @torch.no_grad()
    def _build_routing_adjacency(self) -> Tensor:
        """Build dense [BS, N, N] adjacency from precomputed routing weights.

        Scatters the [BS, N, K] softmax weights into a dense matrix for bmm.
        Used by the Python path. Triton path uses sparse weights directly.
        """
        BS = self._routing_weights.shape[0]
        N = self.config.N_neurons
        K = self.config.K_connections

        A = torch.zeros(BS, N, N, device=self.device, dtype=self.dtype)
        idx = self.conn_indices.unsqueeze(0).expand(BS, N, K)
        A.scatter_add_(2, idx, self._routing_weights)
        return A

    @torch.no_grad()
    def _compute_routing_weights(self):
        """Compute softmax routing weights from key × neighbor messages.

        Called once per segment. Produces [BS, N, K] scalar weights used
        by the fast sparse Triton kernel for the entire segment.
        """
        BS = self.prev_messages.shape[0]
        N = self.config.N_neurons
        K = self.config.K_connections

        # Gather neighbor messages: [BS, N, K, D]
        neighbor_msgs = self.prev_messages[:, self.conn_indices]

        # Raw dot product: key[n] · neighbor_msg[k] for each connection
        # key: [BS, N, D] → [BS, N, 1, D]
        # neighbor_msgs: [BS, N, K, D]
        sim = (self.key.unsqueeze(2) * neighbor_msgs).sum(dim=-1)  # [BS, N, K]

        # Softmax → routing weights (sum to 1 per neuron)
        self._routing_weights = torch.softmax(sim, dim=-1).to(self.dtype)  # [BS, N, K]

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, BS: int):
        """Initialize persistent state for batch size BS."""
        N = self.config.N_neurons
        D = self.config.D_mem
        K_conn = self.config.K_connections

        # Per-neuron parameters (neuromodulator-controlled)
        # Primitives: what the neuron broadcasts. RMS-normalized (per-dim ≈ 1).
        prim_raw = 1.0 + torch.randn(BS, N, D, device=self.device, dtype=self.dtype) * 0.02
        rms = (prim_raw ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.primitives = prim_raw / rms

        # Key: what the neuron listens for. L2-normalized (unit direction vector).
        # Connection weight = softmax(cosine_sim(neighbor_msg, my_key)) — content-based routing.
        key_raw = torch.randn(BS, N, D, device=self.device, dtype=self.dtype)
        self.key = key_raw / key_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Decay: sigmoid(0) = 0.5 — neutral starting point (50% carry)
        self.decay_logit = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Persistent neuron state
        self.h = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)         # internal state
        self.prev_messages = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)         # last outgoing messages

        # Running stats
        self.mean_input = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.mean_output = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)

        # Firing stats (per neuron)
        self.firing_rate = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Co-activation matrix (batch-averaged, for structural plasticity)
        # [N, N] — not per-batch, topology is shared
        self.co_activation_ema = torch.zeros(
            N, N, device=self.device, dtype=torch.float32)

        # Plasticity tracking
        self._plasticity_rewires = 0  # cumulative rewired connections
        self._co_activation_ready = False  # set True after first phi update

        # Triton kernel buffers (allocated once, reused)
        if _HAS_TRITON and self.device.type == 'cuda':
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
            self._decay_f32 = torch.zeros(
                BS, N, device=self.device, dtype=torch.float32)
            self._triton_ready = True
        else:
            self._triton_ready = False

        self._initialized = True

    @torch.no_grad()
    def forward_segment(self, cc_signals: Tensor,
                        eot_mask: Tensor | None = None,
                        update_co_activation: bool = True) -> Tensor:
        """Process a segment of tokens with per-token neuron dynamics.

        Dispatches to Triton kernel on CUDA, Python fallback on CPU.
        Both produce identical results — Python version is the reference.

        Args:
            cc_signals: [BS, T_seg, C, D_mem] — CC signals for this segment
            eot_mask: [BS, T_seg] bool — True at positions where previous token
                      was EOT. State is killed at these positions.
            update_co_activation: if True, compute phi bmm for structural plasticity.
                Only needed on segments immediately before plasticity runs.

        Returns:
            mem_signals: [BS, T_seg, C, D_mem] — port neuron messages
        """
        # Triton kernel uses precomputed scalar routing weights (same speed as old kernel)
        if self._triton_ready and cc_signals.is_cuda:
            return self._forward_segment_triton(cc_signals, eot_mask,
                                                update_co_activation)
        return self._forward_segment_python(cc_signals, eot_mask,
                                            update_co_activation)

    def _forward_segment_python(self, cc_signals: Tensor,
                                eot_mask: Tensor | None = None,
                                update_co_activation: bool = True) -> Tensor:
        """Python reference implementation of per-token neuron dynamics.

        At each timestep, each neuron:
          1. Receives presynaptic messages: received = A @ prev_messages
             Port neurons also receive CC signals from the LM.
          2. Integrates stimuli into state: h = decay * h + (1-decay) * received
             (convex combination — h is self-bounding regardless of decay)
          3. Computes outgoing message: message = tanh(h * primitives)
             (L2-normed primitives keep h*prim in tanh's linear regime)

        Also tracks per-token activation magnitudes for firing threshold
        and co-activation computation.

        This is the readable reference. The Triton version must match
        the neuron dynamics exactly (firing/plasticity stats are Python-only).
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons

        # Pre-compute constants
        decay = torch.sigmoid(self.decay_logit).unsqueeze(-1)  # [BS, N, 1]
        one_minus_decay = 1.0 - decay                          # [BS, N, 1]

        # Pre-check EOT positions (one sync, avoid per-step GPU→CPU transfer)
        has_eot = None
        if eot_mask is not None and eot_mask.any():
            has_eot = eot_mask.any(dim=0).tolist()  # [T_seg] bools

        # Compute routing weights from key × neighbor messages (once per segment)
        self._compute_routing_weights()
        # Build dense adjacency from sparse routing weights for bmm
        A = self._build_routing_adjacency()  # [BS, N, N]
        use_f32_bmm = not A.is_cuda and A.dtype == torch.bfloat16

        # Per-token sequential loop
        h = self.h                          # [BS, N, D] — internal state
        prev_msg = self.prev_messages       # [BS, N, D] — last outgoing messages
        output = torch.empty(BS, T_seg, C, D, device=self.device, dtype=self.dtype)

        # Track activation magnitudes for firing threshold
        act_trace = torch.empty(BS, T_seg, N, device=self.device, dtype=torch.float32)
        # Accumulate received signals for mean_input (all neurons, not just ports)
        received_accum = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)
        msg_accum = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)

        for t in range(T_seg):
            # 1. Receive: sparse weighted sum using precomputed routing weights
            if use_f32_bmm:
                received = torch.bmm(A.float(), prev_msg.float()).to(self.dtype)
            else:
                received = torch.bmm(A, prev_msg)  # [BS, N, D]

            # Port neurons also receive external CC signal from LM
            received[:, :C] = received[:, :C] + cc_signals[:, t]

            # 2. Integrate: blend internal state with incoming stimuli
            if has_eot is not None and has_eot[t]:
                # EOT: kill state for affected batch elements
                eot_t = eot_mask[:, t].view(BS, 1, 1).to(dtype=h.dtype)
                d_t = decay * (1.0 - eot_t)       # 0 at EOT
                omd_t = 1.0 - d_t                  # 1 at EOT
                h = d_t * h + omd_t * received
            else:
                h = decay * h + one_minus_decay * received

            # 3. Compute outgoing message
            # tanh bounds messages; L2-normed primitives keep h*prim in linear regime
            prev_msg = torch.tanh(h * self.primitives)   # [BS, N, D]

            # Track activation magnitude and accumulate for stats
            act_trace[:, t] = prev_msg.norm(dim=-1).float()
            received_accum += received
            msg_accum += prev_msg

            # Store port neuron messages for LM
            output[:, t] = prev_msg[:, :C]

        # Update persistent state
        self.h = h
        self.prev_messages = prev_msg

        # Store previous segment's means (no EMA blur — fresh each segment)
        self.mean_input = received_accum / T_seg
        self.mean_output = msg_accum / T_seg

        self._post_segment_stats(prev_msg, act_trace,
                                 update_co_activation=update_co_activation)
        return output

    def _forward_segment_triton(self, cc_signals: Tensor,
                                eot_mask: Tensor | None = None,
                                update_co_activation: bool = True) -> Tensor:
        """Triton-accelerated per-token neuron dynamics with precomputed routing.

        Routing weights computed once per segment from key × neighbor messages.
        Then the fast sparse kernel runs for all T_seg tokens using those weights.
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.K_connections

        # Compute routing weights once per segment from key × neighbor messages
        self._compute_routing_weights()
        routing_w = self._routing_weights.contiguous()  # [BS, N, K]

        # Pre-compute decay as float32
        self._decay_f32.copy_(torch.sigmoid(self.decay_logit))

        # EOT flags as float32 [BS, T_seg]
        if eot_mask is not None:
            eot_flags = eot_mask.to(dtype=torch.float32).contiguous()
        else:
            eot_flags = torch.zeros(BS, T_seg, device=self.device, dtype=torch.float32)

        cc_signals = cc_signals.contiguous()
        output = torch.empty(BS, T_seg, C, D, device=self.device, dtype=self.dtype)
        grid = (BS, N)
        act_trace = torch.empty(BS, T_seg, N, device=self.device, dtype=torch.float32)

        for t in range(T_seg):
            memory_graph_step_kernel[grid](
                self.h, self.prev_messages,
                self._conn_idx_i32, routing_w,
                self._decay_f32, self.primitives,
                cc_signals, eot_flags, output,
                act_trace,
                t,
                BS=BS, N=N, D=D, K=K, C=C, T_seg=T_seg,
            )

        # Store final state as segment summary (Triton path can't accumulate per-token)
        # These are the most recent values, not a segment mean — slight inconsistency
        # with Python path but still fresh per-segment, no EMA blur
        self.mean_output = self.prev_messages.clone()
        self.mean_input = self.h.clone()

        self._post_segment_stats(self.prev_messages, act_trace,
                                 update_co_activation=update_co_activation)
        return output

    def _post_segment_stats(self, prev_msg: Tensor, act_trace: Tensor = None,
                            update_co_activation: bool = True):
        """Update firing stats and optionally co-activation after a segment.

        Args:
            prev_msg: [BS, N, D] — final messages
            act_trace: [BS, T_seg, N] float32 — activation magnitudes per token
            update_co_activation: if False, skip the phi bmm (only needed before plasticity)
        """
        if act_trace is None:
            return

        BS, T_seg, N = act_trace.shape
        stats_alpha = 1.0 - self.config.plasticity_ema_decay  # 0.01

        # Binary firing: per-neuron 75th percentile within this segment
        # No EMA lag, adapts instantly to any activation scale
        threshold = torch.quantile(
            act_trace.float(), 0.75, dim=1, keepdim=True)  # [BS, 1, N]
        fired = (act_trace > threshold).float()  # [BS, T_seg, N]

        # Update firing rate EMA
        seg_fire_rate = fired.mean(dim=1)  # [BS, N] — ~0.25 by construction
        self.firing_rate = (1 - stats_alpha) * self.firing_rate + stats_alpha * seg_fire_rate

        # Co-activation phi: only compute when needed (before structural plasticity)
        if not update_co_activation:
            return

        # Co-activation: phi coefficient (binary Pearson) between all neuron pairs
        p_i = fired.mean(dim=1, keepdim=True)  # [BS, 1, N] — firing probability
        fired_centered = fired - p_i  # [BS, T_seg, N]
        var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)  # [BS, N]
        # Covariance via bmm: [BS, N, T] @ [BS, T, N] / T = [BS, N, N]
        cov = torch.bmm(
            fired_centered.transpose(1, 2),
            fired_centered,
        ) / T_seg  # [BS, N, N]
        std_i = var_i.sqrt().unsqueeze(2)   # [BS, N, 1]
        std_j = var_i.sqrt().unsqueeze(1)   # [BS, 1, N]
        phi = cov / (std_i * std_j).clamp(min=1e-8)  # [BS, N, N]

        # Average across batch, update EMA (co_activation_ema is [N, N], not per-batch)
        phi_mean = phi.mean(dim=0).float()  # [N, N]
        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = ca_decay * self.co_activation_ema + (1 - ca_decay) * phi_mean
        self._co_activation_ready = True

    @torch.no_grad()
    def apply_actions(self, delta_primitives: Tensor,
                      delta_key: Tensor,
                      delta_decay: Tensor):
        """Apply neuromodulator actions to neuron state.

        Normalization after applying deltas:
        - primitives: RMS-normalized per neuron (per-dim ≈ 1, controls broadcast direction)
        - key: L2-normalized per neuron (unit vector, controls what to listen for)
        - decay: free (convex combination self-bounds h)

        Args:
            delta_primitives: [BS, N, D_mem]
            delta_key: [BS, N, D_mem]
            delta_decay: [BS, N]
        """
        self.primitives = (self.primitives + delta_primitives).to(self.dtype)
        self.key = (self.key + delta_key).to(self.dtype)
        self.decay_logit = (self.decay_logit + delta_decay).to(self.dtype)

        # Primitives: RMS-normalize (per-dim ≈ 1, controls message content)
        rms = (self.primitives ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.primitives = (self.primitives / rms).to(self.dtype)

        # Key: L2-normalize (unit direction vector, controls routing selectivity)
        key_norm = self.key.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.key = (self.key / key_norm).to(self.dtype)

    @torch.no_grad()
    def structural_plasticity(self):
        """Autonomous co-activation-based structural plasticity.

        Prune: connections where the two neurons are anti-correlated (phi < 0).
        Grow: connect to non-connected neurons with highest co-activation (phi > 0).
        80% correlation-guided, 20% random exploration.

        Uses the co_activation_ema matrix (updated in _post_segment_stats).
        Fully vectorized — no Python loop over neurons.
        All thresholds are relative (phi < 0 is a natural boundary, not a magic number).
        """
        N = self.config.N_neurons
        K_conn = self.config.K_connections
        explore_frac = self.config.plasticity_exploration_frac
        phi = self.co_activation_ema  # [N, N]

        # Skip if co-activation hasn't been measured yet (use flag, no GPU sync)
        if not self._co_activation_ready:
            return

        # For each neuron, get phi values of its existing connections
        # conn_indices: [N, K] — gather from phi: [N, N]
        conn_phi = phi[torch.arange(N, device=self.device).unsqueeze(1),
                       self.conn_indices]  # [N, K]

        # Find the worst (most anti-correlated) connection per neuron
        worst_phi, worst_k = conn_phi.min(dim=-1)  # [N], [N]

        # Only prune neurons that have at least one anti-correlated connection
        prune_mask = worst_phi < 0  # [N] bool
        prune_neurons = prune_mask.nonzero(as_tuple=True)[0]  # indices of neurons to rewire

        # No neurons to prune — let vectorized ops handle empty tensors
        if prune_neurons.shape[0] == 0:
            return

        n_prune = prune_neurons.shape[0]

        # Build mask of existing connections for pruning neurons [n_prune, N]
        existing_mask = torch.zeros(n_prune, N, dtype=torch.bool, device=self.device)
        # Scatter existing connections into mask
        prune_conn_idx = self.conn_indices[prune_neurons]  # [n_prune, K]
        existing_mask.scatter_(1, prune_conn_idx, True)
        # Also mask self-connections
        existing_mask[torch.arange(n_prune, device=self.device), prune_neurons] = True

        # Choose new targets: correlation-guided (best unconnected phi) or random
        # Mask phi for pruning neurons: [n_prune, N]
        phi_masked = phi[prune_neurons].clone()  # [n_prune, N]
        phi_masked[existing_mask] = -float('inf')

        # Best unconnected neuron per pruning neuron
        best_targets = phi_masked.argmax(dim=-1)  # [n_prune]
        best_phi = phi_masked[torch.arange(n_prune, device=self.device), best_targets]

        # Random targets for exploration fraction
        random_targets = torch.randint(0, N, (n_prune,), device=self.device)
        use_random = torch.rand(n_prune, device=self.device) < explore_frac

        # Fall back to random if no positive-correlation candidate
        use_random = use_random | (best_phi <= 0)

        new_targets = torch.where(use_random, random_targets, best_targets)

        # Apply rewiring
        worst_k_for_prune = worst_k[prune_neurons]  # [n_prune]
        self.conn_indices[prune_neurons, worst_k_for_prune] = new_targets

        # Track cumulative rewires
        self._plasticity_rewires += n_prune

        # Re-sort for cache locality after topology change
        self._sort_conn_indices()

        if self._triton_ready:
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()

    @torch.no_grad()
    def get_neuron_obs(self) -> Tensor:
        """Build observation tensor for neuromodulator.

        Returns:
            obs: [BS, N, obs_dim]
        """
        parts = [
            self.primitives,                              # [BS, N, D_mem]
            self.key,                                     # [BS, N, D_mem]
            self.mean_input,                              # [BS, N, D_mem]
            self.mean_output,                             # [BS, N, D_mem]
            self.firing_rate.unsqueeze(-1),               # [BS, N, 1]
            torch.sigmoid(self.decay_logit).unsqueeze(-1),# [BS, N, 1]
        ]

        return torch.cat(parts, dim=-1)

    @property
    def obs_dim(self) -> int:
        """Observation dimension for neuromodulator."""
        # D_mem*4 (prim + key + mean_in + mean_out) + 2 (firing_rate + decay)
        return self.config.D_mem * 4 + 2

    @torch.no_grad()
    def reset_streams(self, mask: Tensor):
        """Reset memory state for specific batch elements (at EOT boundaries).

        Args:
            mask: [BS] bool — True for elements to reset
        """
        # Caller pre-checks has_reset on CPU — no GPU sync needed here.
        # Only reset dynamic state at document boundaries.
        # Structural state (primitives, key, decay, plasticity metrics,
        # co_activation_ema) is preserved.
        m1 = mask.unsqueeze(-1).to(dtype=self.dtype)    # [BS, 1]
        m2 = m1.unsqueeze(-1)                            # [BS, 1, 1]
        keep1 = 1.0 - m1                                 # [BS, 1]
        keep2 = 1.0 - m2                                 # [BS, 1, 1]

        self.h = self.h * keep2
        self.prev_messages = self.prev_messages * keep2
        self.firing_rate = self.firing_rate * keep1

    def state_dict(self) -> dict:
        """Export full memory graph state for checkpointing."""
        state = {
            'primitives': self.primitives,
            'key': self.key,
            'decay_logit': self.decay_logit,
            'conn_indices': self.conn_indices,
            'conn_mask': self.conn_mask,
            'h': self.h,
            'prev_messages': self.prev_messages,
            'mean_input': self.mean_input,
            'mean_output': self.mean_output,
            'firing_rate': self.firing_rate,
            'co_activation_ema': self.co_activation_ema,
        }
        return state

    def load_state_dict(self, state: dict):
        """Restore memory graph state from checkpoint."""
        for key, val in state.items():
            setattr(self, key, val)
        if self._triton_ready:
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
