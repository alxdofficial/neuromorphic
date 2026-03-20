"""Neural Memory Graph — persistent neuron network outside autograd.

A graph of neurons with energy-conserving signal routing. Each neuron has a
primitive value (stored information) and energy thresholds (routing weights).
Signal flows: receive → modulate → route. Organized in blocks with full
intra-block and sparse inter-block connectivity.

Runs every token via step(). Not in the autograd graph — the neuromodulator
(trained by PPO) is the only way memory learns.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config


class MemoryGraph:
    """Neural memory graph — persistent, runs every token, no autograd.

    Neurons are organized into C blocks of M neurons each. CCs attach at
    block port neurons (index 0 of each block).

    State (per stream in batch):
      primitives:  [BS, N_neurons, D_mem]  — stored information per neuron
      thresholds:  [BS, N_neurons, max_conn] — energy thresholds for routing
      activations: [BS, N_neurons, D_mem]  — current step accumulation buffer
      prev_output: [BS, N_neurons, D_mem]  — previous step output (read-only)

    Connectivity (fixed topology, shared across batch):
      conn_indices: [N_neurons, max_conn] — index of connected neuron per slot
      conn_mask:    [N_neurons, max_conn] — bool, True if connection exists
      cc_port_idx:  [C] — neuron index of each CC port (one per block)
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        self.config = config
        self.device = device
        self.dtype = dtype

        C = config.C
        M = config.M_per_block
        N = config.N_neurons
        D = config.D_mem
        max_conn = config.max_connections
        inter_k = config.inter_block_k

        # --- Build fixed connectivity ---
        # conn_indices[i, j] = index of neuron connected to neuron i at slot j
        # First M slots: intra-block (fully connected within block)
        # Last inter_k slots: random inter-block connections
        conn_indices = torch.zeros(N, max_conn, dtype=torch.long, device=device)
        conn_mask = torch.zeros(N, max_conn, dtype=torch.bool, device=device)

        for b in range(C):
            block_start = b * M
            block_neurons = list(range(block_start, block_start + M))
            other_neurons = [i for i in range(N)
                             if i < block_start or i >= block_start + M]

            for local_i in range(M):
                neuron_i = block_start + local_i

                # Intra-block: connect to all M neurons in same block
                for j, target in enumerate(block_neurons):
                    conn_indices[neuron_i, j] = target
                    conn_mask[neuron_i, j] = True

                # Inter-block: random sparse connections
                if other_neurons and inter_k > 0:
                    perm = torch.randperm(len(other_neurons), device=device)[:inter_k]
                    for j, p in enumerate(perm):
                        conn_indices[neuron_i, M + j] = other_neurons[p.item()]
                        conn_mask[neuron_i, M + j] = True

        self.conn_indices = conn_indices  # [N_neurons, max_conn]
        self.conn_mask = conn_mask        # [N_neurons, max_conn]

        # CC port neurons: first neuron of each block
        self.cc_port_idx = torch.arange(C, device=device) * M  # [C]

        # --- Shared modulation function ---
        # W_mod: [2*D_mem, D_mem] — shared across all neurons
        # Trained by PPO (not backprop), so stored here as a plain tensor
        self.W_mod = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.02
        self.b_mod = torch.zeros(D, device=device, dtype=dtype)

        # --- State (allocated on initialize) ---
        self.primitives = None   # [BS, N_neurons, D_mem]
        self.thresholds = None   # [BS, N_neurons, max_conn]
        self.activations = None  # [BS, N_neurons, D_mem]
        self.prev_output = None  # [BS, N_neurons, D_mem]

        # Running stats for neuromodulator observations
        self.mean_input = None   # [BS, N_neurons, D_mem]
        self.mean_output = None  # [BS, N_neurons, D_mem]
        self.usage_count = None  # [BS, N_neurons]
        self._ema_decay = 0.95

    def initialize(self, BS: int):
        """Allocate state tensors."""
        N = self.config.N_neurons
        D = self.config.D_mem
        max_conn = self.config.max_connections
        dev, dt = self.device, self.dtype

        self.primitives = torch.randn(BS, N, D, device=dev, dtype=dt) * 0.1
        self.thresholds = torch.zeros(BS, N, max_conn, device=dev, dtype=dt)
        self.activations = torch.zeros(BS, N, D, device=dev, dtype=dt)
        self.prev_output = torch.zeros(BS, N, D, device=dev, dtype=dt)

        self.mean_input = torch.zeros(BS, N, D, device=dev, dtype=dt)
        self.mean_output = torch.zeros(BS, N, D, device=dev, dtype=dt)
        self.usage_count = torch.zeros(BS, N, device=dev, dtype=dt)

    def is_initialized(self) -> bool:
        return self.primitives is not None

    @torch.no_grad()
    def step(self, cc_signals: Tensor) -> Tensor:
        """One graph timestep. All neurons: receive → modulate → route.

        Args:
            cc_signals: [BS, C, D_mem] — signals from cortical columns

        Returns:
            mem_signals: [BS, C, D_mem] — signals to cortical columns
        """
        BS = self.primitives.shape[0]
        N = self.config.N_neurons
        D = self.config.D_mem
        max_conn = self.config.max_connections

        # 1. Inject CC signals into port neurons' activation buffer
        # cc_port_idx: [C], cc_signals: [BS, C, D_mem]
        for c in range(self.config.C):
            self.activations[:, self.cc_port_idx[c]] += cc_signals[:, c]

        # 2. Gather inputs: each neuron sums prev_output of connected neurons
        # conn_indices: [N, max_conn] → gather from prev_output [BS, N, D]
        # Expand for gather: [BS, N, max_conn, D]
        idx = self.conn_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, N, max_conn, D
        )  # [BS, N, max_conn, D]
        gathered = torch.gather(
            self.prev_output.unsqueeze(2).expand(BS, N, max_conn, D),
            dim=1,
            index=idx.clamp(max=N - 1),
        )  # Nah, this doesn't work with gather on dim=1

        # Simpler: index prev_output with conn_indices
        # prev_output: [BS, N, D], conn_indices: [N, max_conn]
        flat_idx = self.conn_indices.reshape(-1)  # [N * max_conn]
        gathered_flat = self.prev_output[:, flat_idx]  # [BS, N*max_conn, D]
        gathered = gathered_flat.reshape(BS, N, max_conn, D)

        # Mask invalid connections
        mask = self.conn_mask.unsqueeze(0).unsqueeze(-1)  # [1, N, max_conn, 1]
        gathered = gathered * mask.float()

        inputs = gathered.sum(dim=2)  # [BS, N, D]

        # 3. Modulate: output = silu(W_mod @ [input; primitive] + b_mod)
        combined = torch.cat([inputs, self.primitives], dim=-1)  # [BS, N, 2*D]
        outputs = F.silu(combined @ self.W_mod + self.b_mod)     # [BS, N, D]

        # Update running stats
        alpha = 1.0 - self._ema_decay
        self.mean_input = self._ema_decay * self.mean_input + alpha * inputs
        self.mean_output = self._ema_decay * self.mean_output + alpha * outputs
        output_mag = outputs.norm(dim=-1)  # [BS, N]
        self.usage_count = self._ema_decay * self.usage_count + alpha * (output_mag > 0.01).float()

        # 4. Route: divide output among connections by energy thresholds
        # Lower threshold = easier to send = higher weight
        route_logits = -self.thresholds / max(self.config.mem_temperature, 1e-6)
        # Mask invalid connections
        route_logits = route_logits.masked_fill(~self.conn_mask.unsqueeze(0), float('-inf'))

        route_weights = F.softmax(route_logits, dim=-1)  # [BS, N, max_conn]
        route_weights = route_weights.nan_to_num(0.0)

        # Sparsity: zero out bottom fraction of routing weights
        if self.config.mem_sparsity > 0:
            k_keep = max(1, int(max_conn * (1.0 - self.config.mem_sparsity)))
            topk_vals, _ = route_weights.topk(k_keep, dim=-1)
            threshold = topk_vals[:, :, -1:]  # [BS, N, 1]
            route_weights = route_weights * (route_weights >= threshold).float()
            # Renormalize
            route_sum = route_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            route_weights = route_weights / route_sum

        # Energy conservation: output magnitude is preserved
        output_norm = outputs.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [BS, N, 1]
        output_dir = outputs / output_norm  # [BS, N, D] unit direction

        # Scatter: for each neuron i, for each connection j:
        #   activations[conn_indices[i,j]] += route_weights[i,j] * output_norm[i] * output_dir[i]
        # = route_weights[i,j] * outputs[i]
        weighted = route_weights.unsqueeze(-1) * outputs.unsqueeze(2)  # [BS, N, max_conn, D]

        # Reset activations buffer, then scatter-add
        self.activations.zero_()
        flat_idx = self.conn_indices.reshape(-1)  # [N * max_conn]
        # Expand to [BS, N*max_conn, D]
        weighted_flat = weighted.reshape(BS, N * max_conn, D)
        # Scatter add into activations
        idx_expanded = flat_idx.unsqueeze(0).unsqueeze(-1).expand(BS, -1, D)
        self.activations.scatter_add_(1, idx_expanded, weighted_flat)

        # Re-inject CC signals (they were added before routing, add again for readout)
        # Actually, CC ports read from activations AFTER routing
        # So CC signals from this step are part of the activation buffer

        # 5. Read CC port activations for return signal
        mem_signals = self.activations[:, self.cc_port_idx]  # [BS, C, D]

        # 6. Swap buffers
        self.prev_output = outputs.clone()

        return mem_signals

    @torch.no_grad()
    def apply_actions(self, delta_primitives: Tensor, delta_thresholds: Tensor):
        """Apply neuromodulator actions to neuron state.

        Args:
            delta_primitives:  [BS, N_neurons, D_mem]
            delta_thresholds:  [BS, N_neurons, max_conn]
        """
        self.primitives = self.primitives + delta_primitives
        self.thresholds = self.thresholds + delta_thresholds

    @torch.no_grad()
    def get_neuron_obs(self, cc_surprise: Tensor | None = None) -> Tensor:
        """Build observation tensor for neuromodulator.

        Args:
            cc_surprise: [BS, C, D_cc] — per-CC surprise (optional)

        Returns:
            obs: [BS, N_neurons, obs_dim]
        """
        BS = self.primitives.shape[0]
        N = self.config.N_neurons
        M = self.config.M_per_block
        C = self.config.C

        parts = [
            self.primitives,      # [BS, N, D_mem]
            self.mean_input,      # [BS, N, D_mem]
            self.mean_output,     # [BS, N, D_mem]
            self.usage_count.unsqueeze(-1),  # [BS, N, 1]
        ]

        # Routing entropy: how spread out is each neuron's routing?
        route_logits = -self.thresholds / max(self.config.mem_temperature, 1e-6)
        route_logits = route_logits.masked_fill(~self.conn_mask.unsqueeze(0), float('-inf'))
        route_probs = F.softmax(route_logits, dim=-1).nan_to_num(0.0)
        entropy = -(route_probs * (route_probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        parts.append(entropy)  # [BS, N, 1]

        # Per-neuron CC surprise: each neuron gets its block's CC surprise
        if cc_surprise is not None:
            # cc_surprise: [BS, C, D_cc] → expand to [BS, N, D_cc]
            # Each block's neurons get the same CC surprise
            surprise_expanded = cc_surprise.unsqueeze(2).expand(
                BS, C, M, -1
            ).reshape(BS, N, -1)  # [BS, N, D_cc]
            parts.append(surprise_expanded)
        else:
            parts.append(torch.zeros(BS, N, self.config.D_cc,
                                     device=self.device, dtype=self.dtype))

        return torch.cat(parts, dim=-1)  # [BS, N, obs_dim]

    @property
    def obs_dim(self) -> int:
        """Observation dimension for neuromodulator."""
        # D_mem*3 (prim + mean_in + mean_out) + 1 (usage) + 1 (entropy) + D_cc (surprise)
        return self.config.D_mem * 3 + 2 + self.config.D_cc

    @torch.no_grad()
    def reset_streams(self, mask: Tensor):
        """Reset memory for masked streams (doc boundary).

        Args:
            mask: [BS] bool — True for streams to reset
        """
        if not mask.any():
            return
        m = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
        m2 = mask.unsqueeze(-1)               # [BS, 1]

        fresh_prim = torch.randn_like(self.primitives[0:1]) * 0.1
        fresh_thresh = torch.zeros_like(self.thresholds[0:1])
        fresh_act = torch.zeros_like(self.activations[0:1])

        self.primitives = torch.where(m, fresh_prim, self.primitives)
        self.thresholds = torch.where(m, fresh_thresh, self.thresholds)
        self.activations = torch.where(m.squeeze(-1).unsqueeze(-1).expand_as(self.activations),
                                        fresh_act, self.activations)
        self.prev_output = torch.where(m.squeeze(-1).unsqueeze(-1).expand_as(self.prev_output),
                                        fresh_act, self.prev_output)
        self.mean_input = torch.where(m.squeeze(-1).unsqueeze(-1).expand_as(self.mean_input),
                                       torch.zeros_like(self.mean_input), self.mean_input)
        self.mean_output = torch.where(m.squeeze(-1).unsqueeze(-1).expand_as(self.mean_output),
                                        torch.zeros_like(self.mean_output), self.mean_output)
        self.usage_count = torch.where(m2.expand_as(self.usage_count),
                                        torch.zeros_like(self.usage_count), self.usage_count)
