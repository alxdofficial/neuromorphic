"""Memory Graph — per-token recurrence with inter-neuron message passing.

At each token timestep within a segment:
  1. Receive: received = A @ prev_messages (+ CC signal for ports)
  2. Integrate: h = decay * h + (1-decay) * received
  3. Message: prev_messages = tanh(h * primitives)

Neurons exchange messages at every timestep, allowing multi-hop
propagation through the graph within a single segment.

State (persistent across chunks, outside autograd):
  primitives: [BS, N, D_mem] — per-neuron message modulation
  decay_logit: [BS, N] — per-neuron decay (pre-sigmoid)
  conn_weights: [BS, N, K_conn] — connection weights
  h: [BS, N, D_mem] — internal state (temporal memory)
  prev_messages: [BS, N, D_mem] — last outgoing messages

Plasticity metrics:
  flow_ema: [BS, N, K_conn] — EMA of signal flow magnitude
  corr_ema: [BS, N, K_conn] — EMA of co-activation correlation
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
    from .triton_kernels import memory_graph_step_kernel, prepare_sparse_weights_kernel
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

        # CC port assignment: first C neurons are CC ports
        self.cc_port_idx = torch.arange(config.C, device=device)

        # State tensors (set in initialize())
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, BS: int):
        """Initialize persistent state for batch size BS."""
        N = self.config.N_neurons
        D = self.config.D_mem
        K_conn = self.config.K_connections

        # Per-neuron parameters (neuromodulator-controlled)
        # Primitives modulate outgoing messages: message = tanh(h * primitives)
        # Init near 1.0 so messages ≈ tanh(h) at start
        self.primitives = 1.0 + torch.randn(
            BS, N, D, device=self.device, dtype=self.dtype) * 0.02
        # Decay: sigmoid(0) = 0.5 — neutral starting point (50% carry)
        self.decay_logit = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Connection weights (neuromodulator-controlled, L1-normalized per neuron)
        # Init uniform positive, then L1-normalize so sum |w| = 1 per neuron
        raw_w = torch.rand(BS, N, K_conn, device=self.device, dtype=self.dtype) + 0.01
        self.conn_weights = raw_w / raw_w.abs().sum(dim=-1, keepdim=True)

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

        # Adaptive firing threshold stats (per neuron)
        self.activation_ema = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)
        self.activation_std_ema = torch.ones(
            BS, N, device=self.device, dtype=self.dtype) * 0.1  # small positive init
        self.firing_rate = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        # Co-activation matrix (batch-averaged, for structural plasticity)
        # [N, N] — not per-batch, topology is shared
        self.co_activation_ema = torch.zeros(
            N, N, device=self.device, dtype=torch.float32)

        # Per-connection plasticity metrics
        self.flow_ema = torch.zeros(
            BS, N, K_conn, device=self.device, dtype=self.dtype)
        self.corr_ema = torch.zeros(
            BS, N, K_conn, device=self.device, dtype=self.dtype)

        # Cached adjacency matrix (rebuilt when conn_weights change)
        self._adjacency_dirty = True
        self._adjacency_cache = None

        # Triton kernel buffers (allocated once, reused)
        if _HAS_TRITON and self.device.type == 'cuda':
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
            self._conn_w_norm = torch.zeros(
                BS, N, K_conn, device=self.device, dtype=self.dtype)
            self._decay_f32 = torch.zeros(
                BS, N, device=self.device, dtype=torch.float32)
            self._triton_ready = True
        else:
            self._triton_ready = False

        self._initialized = True

    @torch.no_grad()
    def forward_segment(self, cc_signals: Tensor,
                        eot_mask: Tensor | None = None) -> Tensor:
        """Process a segment of tokens with per-token neuron dynamics.

        Dispatches to Triton kernel on CUDA, Python fallback on CPU.
        Both produce identical results — Python version is the reference.

        Args:
            cc_signals: [BS, T_seg, C, D_mem] — CC signals for this segment
            eot_mask: [BS, T_seg] bool — True at positions where previous token
                      was EOT. State is killed at these positions.

        Returns:
            mem_signals: [BS, T_seg, C, D_mem] — port neuron messages
        """
        if self._triton_ready and cc_signals.is_cuda:
            return self._forward_segment_triton(cc_signals, eot_mask)
        return self._forward_segment_python(cc_signals, eot_mask)

    def _forward_segment_python(self, cc_signals: Tensor,
                                eot_mask: Tensor | None = None) -> Tensor:
        """Python reference implementation of per-token neuron dynamics.

        At each timestep, each neuron:
          1. Receives presynaptic messages: received = A @ prev_messages
             Port neurons also receive CC signals from the LM.
          2. Integrates stimuli into state: h = decay * h + (1-decay) * received
          3. Computes outgoing message: message = tanh(h * primitives)

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
        A = self._build_adjacency()                            # [BS, N, N]
        use_f32_bmm = not A.is_cuda and A.dtype == torch.bfloat16

        # Update mean_input stat (CC signal mean, computed outside loop)
        alpha = 0.05
        mean_cc = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)
        mean_cc[:, :C] = cc_signals.mean(dim=1)
        self.mean_input = (1 - alpha) * self.mean_input + alpha * mean_cc

        # Pre-check EOT positions (one sync, avoid per-step GPU→CPU transfer)
        has_eot = None
        if eot_mask is not None and eot_mask.any():
            has_eot = eot_mask.any(dim=0).tolist()  # [T_seg] bools

        # Per-token sequential loop
        h = self.h                          # [BS, N, D] — internal state
        prev_msg = self.prev_messages       # [BS, N, D] — last outgoing messages
        output = torch.empty(BS, T_seg, C, D, device=self.device, dtype=self.dtype)

        # Track activation magnitudes for firing threshold
        act_trace = torch.empty(BS, T_seg, N, device=self.device, dtype=torch.float32)

        for t in range(T_seg):
            # 1. Receive: route presynaptic messages through graph
            if use_f32_bmm:
                received = torch.bmm(A.float(), prev_msg.float()).to(self.dtype)
            else:
                received = torch.bmm(A, prev_msg)       # [BS, N, D]

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
            prev_msg = torch.tanh(h * self.primitives)   # [BS, N, D]

            # Track activation magnitude
            act_trace[:, t] = prev_msg.float().norm(dim=-1)

            # Store port neuron messages for LM
            output[:, t] = prev_msg[:, :C]

        # Update persistent state
        self.h = h
        self.prev_messages = prev_msg

        self._post_segment_stats(prev_msg, act_trace)
        return output

    def _forward_segment_triton(self, cc_signals: Tensor,
                                eot_mask: Tensor | None = None) -> Tensor:
        """Triton-accelerated per-token neuron dynamics.

        Fuses sparse gather + integration + tanh + port output into one kernel
        per token step. Uses sparse conn_indices instead of dense [N,N] adjacency.
        """
        BS, T_seg, C, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.K_connections

        # Pre-compute normalized sparse weights (once per segment)
        if self._adjacency_dirty:
            BLOCK_K = triton.next_power_of_2(K)
            prepare_sparse_weights_kernel[(BS, N)](
                self.conn_weights, self.conn_mask,
                self._conn_w_norm,
                N=N, K=K, BLOCK_K=BLOCK_K,
            )
            self._adjacency_dirty = False

        # Pre-compute decay as float32
        self._decay_f32.copy_(torch.sigmoid(self.decay_logit))

        # EOT flags as float32 [BS, T_seg]
        if eot_mask is not None:
            eot_flags = eot_mask.to(dtype=torch.float32).contiguous()
        else:
            eot_flags = torch.zeros(BS, T_seg, device=self.device, dtype=torch.float32)

        # Update mean_input stat
        alpha = 0.05
        mean_cc = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)
        mean_cc[:, :C] = cc_signals.mean(dim=1)
        self.mean_input = (1 - alpha) * self.mean_input + alpha * mean_cc

        # Ensure cc_signals is contiguous for pointer arithmetic in kernel
        cc_signals = cc_signals.contiguous()

        # Allocate output
        output = torch.empty(BS, T_seg, C, D, device=self.device, dtype=self.dtype)

        # h and prev_messages are modified in-place by the kernel
        grid = (BS, N)

        for t in range(T_seg):
            memory_graph_step_kernel[grid](
                self.h, self.prev_messages,
                self._conn_idx_i32, self._conn_w_norm,
                self._decay_f32, self.primitives,
                cc_signals, eot_flags, output,
                t,
                BS=BS, N=N, D=D, K=K, C=C, T_seg=T_seg,
            )

        self._post_segment_stats(self.prev_messages)
        return output

    def _post_segment_stats(self, prev_msg: Tensor, act_trace: Tensor = None):
        """Update plasticity metrics, firing stats, and co-activation after a segment.

        Args:
            prev_msg: [BS, N, D] — final messages
            act_trace: [BS, T_seg, N] float32 — activation magnitudes per token
                       (None when called from Triton path, which handles stats separately)
        """
        alpha = 0.05

        # Per-connection flow/correlation (from final messages)
        self._update_plasticity_metrics_from_messages(prev_msg)

        # Running output stats
        self.mean_output = (1 - alpha) * self.mean_output + alpha * prev_msg

        if act_trace is None:
            return

        BS, T_seg, N = act_trace.shape

        # Update adaptive firing threshold (per-neuron mean + std of activation)
        seg_mean = act_trace.mean(dim=1)  # [BS, N]
        seg_std = act_trace.std(dim=1, correction=0)  # [BS, N]
        self.activation_ema = (1 - alpha) * self.activation_ema + alpha * seg_mean
        self.activation_std_ema = (1 - alpha) * self.activation_std_ema + alpha * seg_std

        # Binary firing: activation > (mean + std) per neuron
        threshold = (self.activation_ema + self.activation_std_ema).unsqueeze(1)  # [BS, 1, N]
        fired = (act_trace > threshold).float()  # [BS, T_seg, N]

        # Update firing rate
        seg_fire_rate = fired.mean(dim=1)  # [BS, N]
        self.firing_rate = (1 - alpha) * self.firing_rate + alpha * seg_fire_rate

        # Co-activation: phi coefficient (binary Pearson) between all neuron pairs
        # Compute once per segment via bmm
        p_i = fired.mean(dim=1, keepdim=True)  # [BS, 1, N] — firing probability
        # Center the firing indicators
        fired_centered = fired - p_i  # [BS, T_seg, N]
        # Variance per neuron
        var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)  # [BS, N]
        # Covariance via bmm: [BS, N, T] @ [BS, T, N] / T = [BS, N, N]
        cov = torch.bmm(
            fired_centered.transpose(1, 2),
            fired_centered,
        ) / T_seg  # [BS, N, N]
        # Normalize to phi: cov / sqrt(var_i * var_j)
        std_i = var_i.sqrt().unsqueeze(2)   # [BS, N, 1]
        std_j = var_i.sqrt().unsqueeze(1)   # [BS, 1, N]
        phi = cov / (std_i * std_j).clamp(min=1e-8)  # [BS, N, N]

        # Average across batch, update EMA (co_activation_ema is [N, N], not per-batch)
        phi_mean = phi.mean(dim=0).float()  # [N, N]
        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = ca_decay * self.co_activation_ema + (1 - ca_decay) * phi_mean

    def _chunked_scan(self, decay_logit: Tensor, scan_input: Tensor,
                      carry: Tensor, chunk_size: int = 256,
                      use_fla: bool = False,
                      ) -> tuple[Tensor, Tensor]:
        """Run diagonal scan chunked over neurons.

        Args:
            decay_logit: [BS, N] (constant) or [BS, T, N] (time-varying for EOT)
            scan_input: [BS, T, N, D] — gate*u for FLA, (1-decay)*gate*u for CPU
            carry: [BS, N, D]
            use_fla: if True, use FLA HGRN kernel (CUDA only)

        Returns:
            output: [BS, T, N, D]
            new_carry: [BS, N, D]
        """
        BS, T, N, D = scan_input.shape
        time_varying = decay_logit.dim() == 3  # [BS, T, N] vs [BS, N]
        outputs = []
        carries = []

        for n0 in range(0, N, chunk_size):
            n1 = min(n0 + chunk_size, N)
            nc = n1 - n0

            # Reshape input: [BS, T, nc, D] → [BS*nc, T, D]
            chunk = scan_input[:, :, n0:n1].permute(0, 2, 1, 3).reshape(BS * nc, T, D)
            h0_chunk = carry[:, n0:n1].reshape(BS * nc, D)

            if time_varying:
                # [BS, T, nc] → [BS*nc, T] → expand to [BS*nc, T, D]
                dl_chunk = decay_logit[:, :, n0:n1].permute(0, 2, 1).reshape(BS * nc, T)
                dl_chunk = dl_chunk.unsqueeze(-1).expand(BS * nc, T, D)

                if use_fla:
                    # Time-varying decay: FLA needs [B, T, D] gate
                    g = F.logsigmoid(dl_chunk)
                    out, _ = fused_recurrent_hgrn(chunk, g, initial_state=h0_chunk)
                else:
                    # CPU: pre-scale already done, use time-varying a
                    a = torch.sigmoid(dl_chunk)
                    # Sequential fallback for time-varying (parallel scan needs constant a)
                    h = h0_chunk
                    outs = []
                    for t in range(T):
                        h = a[:, t] * h + chunk[:, t]
                        outs.append(h)
                    out = torch.stack(outs, dim=1)
            else:
                # Constant decay — fast path
                dl_chunk = decay_logit[:, n0:n1].reshape(BS * nc)
                dl_chunk = dl_chunk.unsqueeze(-1).expand(BS * nc, D)

                if use_fla:
                    out = _fla_scan(dl_chunk, chunk, h0_chunk)
                else:
                    out = _cpu_scan(dl_chunk, chunk, h0_chunk)

            # Reshape back: [BS*nc, T, D] → [BS, T, nc, D]
            out = out.reshape(BS, nc, T, D).permute(0, 2, 1, 3)
            outputs.append(out)
            carries.append(out[:, -1])

        return torch.cat(outputs, dim=2), torch.cat(carries, dim=1)

    def _build_adjacency(self) -> Tensor:
        """Build dense adjacency matrix from sparse connectivity.

        Weights are already L1-normalized (sum |w| = 1 per neuron), so no
        additional mean-normalization is needed. The adjacency scatters the
        raw L1-normalized weights.

        Returns:
            A: [BS, N, N] — dense weighted adjacency (L1-normalized rows)
        """
        if not self._adjacency_dirty and self._adjacency_cache is not None:
            return self._adjacency_cache

        BS = self.conn_weights.shape[0]
        N = self.config.N_neurons
        K_conn = self.config.K_connections

        w_masked = self.conn_weights * self.conn_mask.unsqueeze(0).to(dtype=self.dtype)

        A = torch.zeros(BS, N, N, device=self.device, dtype=self.dtype)
        idx = self.conn_indices.unsqueeze(0).expand(BS, N, K_conn)
        A.scatter_add_(2, idx, w_masked)

        self._adjacency_cache = A
        self._adjacency_dirty = False
        return A

    def _message_pass(self, x: Tensor) -> Tensor:
        """Sparse graph message passing via dense matmul.

        Builds a [BS, N, N] adjacency matrix and uses bmm for efficiency.
        One matmul replaces 256 gather operations.

        Args:
            x: [BS, T, N, D]

        Returns:
            messages: [BS, T, N, D]
        """
        BS, T, N, D = x.shape

        A = self._build_adjacency()  # [BS, N, N] — ~2MB, tiny

        # Reshape for batched matmul: [BS, N, T*D]
        x_flat = x.permute(0, 2, 1, 3).reshape(BS, N, T * D)

        # Single bmm: [BS, N, N] @ [BS, N, T*D] → [BS, N, T*D]
        # Use float32 for bmm on CPU (bf16 bmm not supported on CPU)
        if not x_flat.is_cuda and x_flat.dtype == torch.bfloat16:
            y_flat = torch.bmm(A.float(), x_flat.float()).to(self.dtype)
        else:
            y_flat = torch.bmm(A, x_flat)

        # Reshape back: [BS, T, N, D]
        return y_flat.reshape(BS, N, T, D).permute(0, 2, 1, 3)

    def _update_plasticity_metrics_from_messages(self, messages: Tensor):
        """Update flow and correlation EMAs from neuron messages.

        Args:
            messages: [BS, N, D] — outgoing messages (last timestep)
        """
        ema_decay = self.config.plasticity_ema_decay

        # Gather presynaptic neuron messages
        neighbor_msg = messages[:, self.conn_indices]  # [BS, N, K_conn, D]
        w = self.conn_weights.unsqueeze(-1)  # [BS, N, K_conn, 1]

        flow = (w * neighbor_msg).abs().mean(dim=-1)  # [BS, N, K_conn]
        self.flow_ema = ema_decay * self.flow_ema + (1 - ema_decay) * flow

        my_msg = messages.unsqueeze(2).expand_as(neighbor_msg)  # [BS, N, K_conn, D]
        corr = (my_msg * neighbor_msg).mean(dim=-1)  # [BS, N, K_conn]
        self.corr_ema = ema_decay * self.corr_ema + (1 - ema_decay) * corr

    @torch.no_grad()
    def apply_actions(self, delta_primitives: Tensor,
                      delta_conn_weights: Tensor,
                      delta_decay: Tensor):
        """Apply neuromodulator actions to neuron/connection state.

        After applying deltas, L1-normalize conn_weights per neuron
        (energy conservation: fixed routing budget of 1.0).

        Args:
            delta_primitives: [BS, N, D_mem]
            delta_conn_weights: [BS, N, K_conn]
            delta_decay: [BS, N]
        """
        self.primitives = self.primitives + delta_primitives
        self.conn_weights = self.conn_weights + delta_conn_weights
        self.decay_logit = self.decay_logit + delta_decay

        # Energy conservation: L1-normalize connection weights per neuron
        # The neuromod controls the distribution, not the total magnitude
        w_abs_sum = self.conn_weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.conn_weights = self.conn_weights / w_abs_sum

        self._adjacency_dirty = True

    @torch.no_grad()
    def structural_plasticity(self):
        """Autonomous co-activation-based structural plasticity.

        Prune: connections where the two neurons are anti-correlated (phi < 0).
        Grow: connect to non-connected neurons with highest co-activation (phi > 0).
        80% correlation-guided, 20% random exploration.

        Uses the co_activation_ema matrix (updated in _post_segment_stats).
        All thresholds are relative (phi < 0 is a natural boundary, not a magic number).
        """
        N = self.config.N_neurons
        K_conn = self.config.K_connections
        explore_frac = self.config.plasticity_exploration_frac
        phi = self.co_activation_ema  # [N, N]

        # Skip if co-activation hasn't been measured yet
        if phi.abs().max() < 1e-10:
            return

        self._adjacency_dirty = True
        rewired = False

        for j in range(N):
            existing_indices = self.conn_indices[j]  # [K_conn]
            existing_set = set(existing_indices.tolist())
            existing_set.add(j)  # no self-connections

            # Get phi values for existing connections
            conn_phi = phi[j, existing_indices]  # [K_conn]

            # Find anti-correlated connections (phi < 0) — candidates for pruning
            anti_corr = (conn_phi < 0).nonzero(as_tuple=True)[0]
            if len(anti_corr) == 0:
                continue

            # Prune the most anti-correlated connection (lowest phi)
            worst_k = anti_corr[conn_phi[anti_corr].argmin()].item()

            # Choose replacement: correlation-guided or random
            if torch.rand(1).item() < explore_frac:
                # Random exploration
                for _ in range(10):
                    new_target = torch.randint(0, N, (1,), device=self.device).item()
                    if new_target not in existing_set:
                        break
                else:
                    continue  # couldn't find a non-existing target
            else:
                # Correlation-guided: find best non-connected neuron
                phi_j = phi[j].clone()  # [N]
                # Mask out existing connections and self
                for idx in existing_set:
                    phi_j[idx] = -float('inf')
                new_target = phi_j.argmax().item()
                if phi_j[new_target] <= 0:
                    # No positive-correlation candidates, skip
                    continue

            # Rewire
            self.conn_indices[j, worst_k] = new_target
            existing_set.add(new_target)

            # Init new weight to median of this neuron's current |weights|
            median_w = self.conn_weights[:, j].abs().median(dim=-1).values  # [BS]
            self.conn_weights[:, j, worst_k] = median_w
            self.flow_ema[:, j, worst_k] = 0
            self.corr_ema[:, j, worst_k] = 0
            rewired = True

        if rewired:
            # Re-normalize weights after topology change
            w_abs_sum = self.conn_weights.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
            self.conn_weights = self.conn_weights / w_abs_sum

            # Update int32 indices for Triton kernel
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
            self.mean_input,                              # [BS, N, D_mem]
            self.mean_output,                             # [BS, N, D_mem]
            self.firing_rate.unsqueeze(-1),               # [BS, N, 1]
            torch.sigmoid(self.decay_logit).unsqueeze(-1),# [BS, N, 1]
        ]

        # Routing entropy from connection weights (L1-normalized, so w_abs sums to 1)
        w_abs = self.conn_weights.abs()
        eps = torch.tensor(1e-4, dtype=self.dtype, device=self.device)
        entropy = -(w_abs * (w_abs + eps).log()).sum(dim=-1, keepdim=True)
        parts.append(entropy)  # [BS, N, 1]

        # Plasticity metrics (use correction=0 for population std, avoids NaN)
        parts.append(self.flow_ema.mean(dim=-1, keepdim=True))                     # [BS, N, 1]
        parts.append(self.flow_ema.std(dim=-1, keepdim=True, correction=0))        # [BS, N, 1]
        parts.append(self.corr_ema.mean(dim=-1, keepdim=True))                     # [BS, N, 1]
        parts.append(self.flow_ema.min(dim=-1, keepdim=True)[0])                   # [BS, N, 1]

        return torch.cat(parts, dim=-1)

    @property
    def obs_dim(self) -> int:
        """Observation dimension for neuromodulator."""
        # D_mem*3 + 2 (firing_rate, decay) + 1 (entropy) + 4 (plasticity metrics)
        return self.config.D_mem * 3 + 7

    @torch.no_grad()
    def reset_streams(self, mask: Tensor):
        """Reset memory state for specific batch elements (at EOT boundaries).

        Args:
            mask: [BS] bool — True for elements to reset
        """
        if not mask.any():
            return

        # Only reset dynamic state at document boundaries.
        # Structural state (primitives, conn_weights, decay, plasticity metrics,
        # co_activation_ema) is preserved.
        m1 = mask.unsqueeze(-1).to(dtype=self.dtype)    # [BS, 1]
        m2 = m1.unsqueeze(-1)                            # [BS, 1, 1]
        keep1 = 1.0 - m1                                 # [BS, 1]
        keep2 = 1.0 - m2                                 # [BS, 1, 1]

        self.h = self.h * keep2
        self.prev_messages = self.prev_messages * keep2
        # Reset firing stats for reset streams (they'll rebuild quickly)
        self.activation_ema = self.activation_ema * keep1
        self.activation_std_ema = self.activation_std_ema * keep1 + m1 * 0.1
        self.firing_rate = self.firing_rate * keep1

    def state_dict(self) -> dict:
        """Export full memory graph state for checkpointing."""
        state = {
            'primitives': self.primitives,
            'decay_logit': self.decay_logit,
            'conn_weights': self.conn_weights,
            'conn_indices': self.conn_indices,
            'conn_mask': self.conn_mask,
            'h': self.h,
            'prev_messages': self.prev_messages,
            'mean_input': self.mean_input,
            'mean_output': self.mean_output,
            'activation_ema': self.activation_ema,
            'activation_std_ema': self.activation_std_ema,
            'firing_rate': self.firing_rate,
            'co_activation_ema': self.co_activation_ema,
            'flow_ema': self.flow_ema,
            'corr_ema': self.corr_ema,
        }
        return state

    def load_state_dict(self, state: dict):
        """Restore memory graph state from checkpoint."""
        for key, val in state.items():
            setattr(self, key, val)
        self._adjacency_dirty = True
        if self._triton_ready:
            self._conn_idx_i32 = self.conn_indices.to(torch.int32).contiguous()
