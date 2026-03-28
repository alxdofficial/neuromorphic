"""Memory Graph — scalar neuron simulation (v10).

N=524,288 scalar neurons in 512 groups of 1024. K=64 sparse connections.
Inspired by fruit fly connectome simulations (FlyWire/Eon, Nature 2024):
simple leaky-integrate-and-fire dynamics + right graph topology = behavior.

Per-step dynamics (128 steps per segment, all element-wise):
  1. Gather: read K neighbors' activations via conn_indices
  2. Weight: multiply by w_conn, sum → received scalar
  3. Inject: add LM signal (one scalar per neuron)
  4. Integrate: V = decay * V + (1 - decay) * (received + inject)
  5. Activate: activation = sigmoid(V - threshold)
  6. Hebbian: trace correlation between firing and neighbor firing

Neuromodulator (once per segment, per group):
  - 512 groups × 1024 neurons/group
  - Each group has its own MLP weights
  - Processes neurons one-at-a-time (batched): input=[70] → output=[66]
  - Predicts new w_conn, decay, threshold for each neuron

Inject: H_mid [BS,T,2048] → repeat_interleave → [BS,T,N] scalars
Readout: activation [BS,T,N] → reshape → mean over replicas → [BS,T,2048]
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config


class SparseWeightedSum(torch.autograd.Function):
    """Fused gather → multiply → sum that never materializes [BS, N, K] for autograd.

    Computes: received[b,n] = sum_k(activation[b, conn[n,k]] * w_conn[b,n,k])

    Standard PyTorch would save the [BS, N, K] intermediate from the multiply
    for w_conn's gradient. At N=524K, K=64, that's 134MB × 128 steps = 17GB.

    This function saves only activation [BS, N] (2MB) and recomputes the
    gather during backward.
    """

    @staticmethod
    def forward(ctx, activation, conn_indices, w_conn):
        # Compute: gather → multiply → sum
        neighbor_act = activation[:, conn_indices]  # [BS, N, K]
        received = (neighbor_act * w_conn).sum(dim=-1)  # [BS, N]

        # Save only the small tensors for backward
        ctx.save_for_backward(activation, conn_indices, w_conn)
        # Also save neighbor_act for hebbian (will be accessed via ctx)
        ctx.neighbor_act = neighbor_act.detach()
        return received

    @staticmethod
    def backward(ctx, grad_received):
        activation, conn_indices, w_conn = ctx.saved_tensors
        BS, N = activation.shape

        # Recompute the gather (cheap — just index lookups)
        neighbor_act = activation[:, conn_indices]  # [BS, N, K]

        # grad_w_conn: d(received)/d(w_conn) = neighbor_act, scaled by grad_received
        # grad_w_conn[b,n,k] = grad_received[b,n] * neighbor_act[b,n,k]
        grad_w_conn = grad_received.unsqueeze(-1) * neighbor_act  # [BS, N, K]

        # grad_activation: scatter_add w_conn-weighted grad back to source neurons
        # d(received[b,n])/d(activation[b,src]) = w_conn[b,n,k] where conn[n,k]=src
        weighted_grad = grad_received.unsqueeze(-1) * w_conn  # [BS, N, K]
        grad_activation = torch.zeros(BS, N, device=grad_received.device,
                                      dtype=grad_received.dtype)
        idx_expanded = conn_indices.unsqueeze(0).expand(BS, -1, -1)
        grad_activation.scatter_add_(1, idx_expanded.reshape(BS, -1),
                                     weighted_grad.reshape(BS, -1))

        return grad_activation, None, grad_w_conn


def sparse_weighted_sum(activation, conn_indices, w_conn):
    """Memory-efficient: gather → weight → sum without saving [BS,N,K] per step."""
    return SparseWeightedSum.apply(activation, conn_indices, w_conn)


class MemoryGraph(nn.Module):
    """Scalar neuron memory graph with grouped neuromodulators.

    nn.Parameters (trained by backprop):
        mod_w1, mod_b1, mod_w2, mod_b2 — per-group modulator MLP

    Runtime state (per-batch, set by modulator):
        V, activation — neuron dynamics
        w_conn, decay, threshold — neuron properties
        hebbian — per-connection correlation traces
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K = config.K_connections
        G = config.n_groups
        GS = config.group_size

        # ================================================================
        # Sparse connectivity [N, K] — fixed-degree adjacency list
        # ================================================================
        conn_indices = self._init_connectivity(N, K, G, GS,
                                                config.min_intra_connections,
                                                device)
        self.register_buffer('conn_indices', conn_indices)

        # ================================================================
        # Fixed neuron identity — sin/cos of global index
        # ================================================================
        global_pos = torch.arange(N, device=device, dtype=torch.float32)
        neuron_id = torch.stack([
            torch.sin(global_pos * (2 * math.pi / N)),
            torch.cos(global_pos * (2 * math.pi / N)),
        ], dim=-1)  # [N, 2]
        self.register_buffer('neuron_id', neuron_id)

        # ================================================================
        # Neuromodulator MLP (per-group weights)
        # ================================================================
        # Input per neuron: activation(1) + hebbian(K) + decay(1) + threshold(1) + id(2) = K+5
        mod_input_dim = K + 5
        # Output per neuron: new_w_conn(K) + new_decay(1) + new_threshold(1) = K+2
        mod_output_dim = K + 2
        H = config.neuromod_hidden

        self.mod_w1 = nn.Parameter(
            torch.randn(G, mod_input_dim, H, device=device) *
            (2.0 / (mod_input_dim + H)) ** 0.5)
        self.mod_b1 = nn.Parameter(torch.zeros(G, H, device=device))
        self.mod_w2 = nn.Parameter(
            torch.randn(G, H, mod_output_dim, device=device) * 0.01)
        self.mod_b2 = nn.Parameter(
            torch.randn(G, mod_output_dim, device=device) * 0.01)

        # Inject/readout constants
        self.replicas = config.replicas_per_dim

        self._initialized = False

    @staticmethod
    def _init_connectivity(N, K, G, GS, min_intra, device):
        """Initialize sparse connectivity with intra-group guarantees."""
        conn = torch.zeros(N, K, dtype=torch.long, device=device)

        for n in range(N):
            group_start = (n // GS) * GS
            group_end = group_start + GS

            # Intra-group connections (excluding self)
            group_members = [i for i in range(group_start, group_end) if i != n]
            n_intra = min(min_intra, len(group_members))
            intra_idx = torch.randperm(len(group_members), device=device)[:n_intra]
            intra = torch.tensor([group_members[i] for i in intra_idx],
                                 device=device)

            # Remaining connections: random across all neurons (excluding self)
            n_external = K - n_intra
            # Simple approach: random, filter self
            external = torch.randint(0, N, (n_external * 2,), device=device)
            external = external[external != n][:n_external]
            if len(external) < n_external:
                # Rare edge case: pad with random
                extra = torch.randint(0, N, (n_external - len(external),),
                                      device=device)
                external = torch.cat([external, extra])

            all_conn = torch.cat([intra, external[:n_external]])
            conn[n] = all_conn.sort().values

        return conn

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        """Initialize runtime state for batch size BS."""
        device = self.mod_w1.device
        N = self.config.N_neurons
        K = self.config.K_connections
        dt = self.dtype

        # Neuron state
        self.V = torch.randn(BS, N, device=device, dtype=dt) * 0.1
        self.activation = torch.sigmoid(self.V)

        # Neuron properties (set by modulator)
        self.w_conn = torch.zeros(BS, N, K, device=device, dtype=dt)
        self.decay = torch.full((BS, N), 0.9, device=device, dtype=dt)
        self.threshold = torch.zeros(BS, N, device=device, dtype=dt)

        # Hebbian traces
        self.hebbian = torch.zeros(BS, N, K, device=device, dtype=dt)

        # Diagnostics
        self.activation_magnitude = torch.zeros(BS, N, device=device, dtype=dt)

        self._initialized = True

    # ================================================================
    # Neuromodulator
    # ================================================================

    def _run_modulator(self):
        """Per-group modulator: predict w_conn, decay, threshold for each neuron.

        Uses batched matmul (bmm) across all 512 groups simultaneously.
        Processes each neuron independently within a group (but batched).
        """
        BS = self.V.shape[0]
        G = self.config.n_groups
        GS = self.config.group_size
        K = self.config.K_connections

        # Assemble per-neuron input: [BS, N, K+5]
        # Mean hebbian per neuron or full hebbian vector
        mod_input = torch.cat([
            self.activation.unsqueeze(-1),                    # [BS, N, 1]
            self.hebbian,                                      # [BS, N, K]
            self.decay.unsqueeze(-1),                          # [BS, N, 1]
            self.threshold.unsqueeze(-1),                      # [BS, N, 1]
            self.neuron_id.unsqueeze(0).expand(BS, -1, -1),   # [BS, N, 2]
        ], dim=-1)  # [BS, N, K+5]

        # Reshape by group: [BS, G, GS, input_dim]
        mod_input = mod_input.view(BS, G, GS, -1)

        # Flatten batch into group dim for bmm: [BS*G, GS, input_dim]
        x = mod_input.reshape(BS * G, GS, -1)

        # Expand weights for batch: [G, in, H] → [BS*G, in, H]
        w1 = self.mod_w1.repeat(BS, 1, 1)  # [BS*G, in, H]
        b1 = self.mod_b1.repeat(BS, 1)      # [BS*G, H]
        w2 = self.mod_w2.repeat(BS, 1, 1)  # [BS*G, H, out]
        b2 = self.mod_b2.repeat(BS, 1)      # [BS*G, out]

        # MLP: [BS*G, GS, in] @ [BS*G, in, H] → [BS*G, GS, H]
        hidden = torch.bmm(x, w1) + b1.unsqueeze(1)
        hidden = torch.tanh(hidden)

        # [BS*G, GS, H] @ [BS*G, H, out] → [BS*G, GS, out]
        output = torch.bmm(hidden, w2) + b2.unsqueeze(1)

        # Reshape back: [BS, G, GS, out] → [BS, N, out]
        output = output.view(BS, G * GS, -1)

        # Split output
        new_w_conn = output[..., :K]           # [BS, N, K]
        new_decay = torch.sigmoid(output[..., K])  # [BS, N] — sigmoid for [0,1]
        new_threshold = output[..., K + 1]     # [BS, N]

        return new_w_conn, new_decay, new_threshold

    # ================================================================
    # Inject / Readout (parameter-free, scalar)
    # ================================================================

    def inject(self, H_mid_seg: Tensor) -> Tensor:
        """H_mid [BS, T_seg, D] → [BS, T_seg, N] by repeating each dim.

        Each of the D=2048 LM dims gets replicated to `replicas` neurons.
        """
        # [BS, T_seg, D] → [BS, T_seg, N] via repeat_interleave
        return H_mid_seg.repeat_interleave(self.replicas, dim=-1)

    def readout(self, act: Tensor) -> Tensor:
        """activation [BS, T_seg, N] → [BS, T_seg, D] by averaging replicas."""
        BS, T_seg, N = act.shape
        # [BS, T_seg, D, replicas] → mean → [BS, T_seg, D]
        return act.view(BS, T_seg, self.config.D, self.replicas).mean(dim=-1)

    # ================================================================
    # Forward segment
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment of tokens with scalar neuron dynamics.

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
        V = self.V.detach()
        activation = self.activation.detach()

        # Run modulator FIRST (on compute graph)
        w_conn, decay, threshold = self._run_modulator()

        # Reset hebbian for this segment
        hebbian = torch.zeros(BS, N, K, device=V.device, dtype=V.dtype)

        # Per-step dynamics with gradient checkpointing.
        # Memory optimizations:
        #   - inject computed per-chunk (not precomputed for all T)
        #   - readout accumulated incrementally as [BS, D] (not stored as [BS, T, N])
        #   - hebbian accumulated detached
        from torch.utils.checkpoint import checkpoint

        chunk_size = 8
        replicas = self.replicas
        D = self.config.D

        def run_chunk(V, activation, cc_chunk, w_conn, decay, threshold,
                      conn_indices, replicas, D):
            """Run chunk_size steps. Returns per-step readout [BS, C, D]."""
            C = cc_chunk.shape[1]
            inject_chunk = cc_chunk.repeat_interleave(replicas, dim=-1)
            step_readouts = []
            for i in range(C):
                neighbor_act = activation[:, conn_indices]
                received = (neighbor_act * w_conn).sum(dim=-1)
                received = received + inject_chunk[:, i]
                V = decay * V + (1.0 - decay) * received
                activation = torch.sigmoid(V - threshold)
                # Per-step readout: [BS, N] → [BS, D] (tiny)
                step_readouts.append(
                    activation.view(-1, D, replicas).mean(dim=-1))
            return V, activation, torch.stack(step_readouts, dim=1)

        chunk_readouts = []
        for c_start in range(0, T_seg, chunk_size):
            c_end = min(c_start + chunk_size, T_seg)
            cc_chunk = cc_signals[:, c_start:c_end]

            V, activation, chunk_readout = checkpoint(
                run_chunk, V, activation, cc_chunk,
                w_conn, decay, threshold, self.conn_indices, replicas, D,
                use_reentrant=False)
            chunk_readouts.append(chunk_readout)  # [BS, chunk_size, D]

            # Hebbian (detached, outside checkpoint)
            with torch.no_grad():
                neighbor_act = activation.detach()[:, self.conn_indices]
                hebbian = hebbian + activation.detach().unsqueeze(-1) * neighbor_act

        # [BS, T_seg, D] — per-token readout
        mem_out = torch.cat(chunk_readouts, dim=1)  # [BS, T_seg, D]

        # Update persistent state (detached)
        with torch.no_grad():
            self.V = V.detach()
            self.activation = activation.detach()
            self.w_conn = w_conn.detach()
            self.decay = decay.detach()
            self.threshold = threshold.detach()
            self.hebbian = (hebbian / max(T_seg, 1)).detach()

            # Diagnostics: EMA of activation magnitude
            alpha = 0.05
            self.activation_magnitude = (
                (1 - alpha) * self.activation_magnitude +
                alpha * self.activation.abs()
            ).to(self.dtype)

        return mem_out

    # ================================================================
    # Structural plasticity
    # ================================================================

    def rewire_connections(self):
        """Replace weakest connections with random new ones.

        Based on hebbian trace magnitude — weak correlations get pruned.
        No N² matrix needed.
        """
        if not self.config.structural_plasticity:
            return
        if not self._initialized:
            return

        N = self.config.N_neurons
        K = self.config.K_connections
        n_swap = min(self.config.plasticity_n_swap, K)

        with torch.no_grad():
            # Mean hebbian across batch
            heb_strength = self.hebbian.abs().mean(dim=0)  # [N, K]

            # Find weakest connections per neuron
            _, weakest_idx = heb_strength.topk(n_swap, dim=-1, largest=False)

            # Random new targets (biased 50% toward same group)
            GS = self.config.group_size
            group_starts = (torch.arange(N, device=heb_strength.device) // GS) * GS

            # Half intra-group, half random
            n_intra = n_swap // 2
            n_random = n_swap - n_intra

            new_targets = torch.zeros(N, n_swap, dtype=torch.long,
                                      device=heb_strength.device)
            if n_intra > 0:
                intra_offsets = torch.randint(0, GS, (N, n_intra),
                                              device=heb_strength.device)
                new_targets[:, :n_intra] = group_starts.unsqueeze(1) + intra_offsets
            if n_random > 0:
                new_targets[:, n_intra:] = torch.randint(
                    0, N, (N, n_random), device=heb_strength.device)

            # Replace weakest with new targets
            self.conn_indices.scatter_(1, weakest_idx, new_targets)

            # Re-sort for efficient gather
            sorted_idx, _ = self.conn_indices.sort(dim=-1)
            self.conn_indices.copy_(sorted_idx)

        self._last_rewire_swaps = n_swap * N

    # ================================================================
    # State management
    # ================================================================

    def detach_states(self):
        if not self._initialized:
            return
        self.V = self.V.detach()
        self.activation = self.activation.detach()

    def runtime_state_dict(self) -> dict:
        state = {
            'V': self.V, 'activation': self.activation,
            'w_conn': self.w_conn, 'decay': self.decay,
            'threshold': self.threshold,
            'hebbian': self.hebbian,
            'activation_magnitude': self.activation_magnitude,
        }
        return state

    def load_runtime_state(self, state: dict):
        for key, val in state.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if isinstance(current, Tensor) and isinstance(val, Tensor):
                if current.shape != val.shape:
                    raise ValueError(
                        f"Runtime state shape mismatch for '{key}': "
                        f"expected {current.shape}, got {val.shape}.")
            setattr(self, key, val)
        self._initialized = True
