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

        # Inject: [BS, T_seg, D] → [BS, T_seg, N]
        inject_all = self.inject(cc_signals)

        # Reset hebbian for this segment
        hebbian = torch.zeros(BS, N, K, device=V.device, dtype=V.dtype)

        # Per-step dynamics (sequential, all element-wise)
        readout_steps = []
        for t in range(T_seg):
            # 1. Gather neighbors' activations
            neighbor_act = activation[:, self.conn_indices]  # [BS, N, K]

            # 2. Weighted sum of incoming signals
            received = (neighbor_act * w_conn).sum(dim=-1)  # [BS, N]

            # 3. Add inject signal
            received = received + inject_all[:, t]  # [BS, N]

            # 4. Leaky integration
            V = decay * V + (1.0 - decay) * received  # [BS, N]

            # 5. Activation
            activation = torch.sigmoid(V - threshold)  # [BS, N]

            # 6. Hebbian trace: correlation of this neuron's firing with neighbors'
            hebbian = hebbian + activation.unsqueeze(-1) * neighbor_act  # [BS, N, K]

            readout_steps.append(activation)

        # Stack readout: [BS, T_seg, N]
        act_all = torch.stack(readout_steps, dim=1)

        # Readout: [BS, T_seg, N] → [BS, T_seg, D]
        mem_out = self.readout(act_all)

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
