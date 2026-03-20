"""Neural Memory Graph — persistent neuron network outside autograd.

A graph of neurons with energy-conserving signal routing. Each neuron has a
primitive value (stored information) and energy thresholds (routing weights).
Signal flows: receive → modulate → route.

Organized in N_blocks blocks with random sparse connectivity (K_intra within
block, K_inter across blocks). CCs attach via port neurons — each block has
CCs_per_block ports. Blocks are independent of CC count.

Runs every token via step(). Not in the autograd graph — the neuromodulator
(trained by PPO) is the only way memory learns.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config


class MemoryGraph:
    """Neural memory graph — persistent, runs every token, no autograd.

    State (per stream in batch):
      primitives:  [BS, N_neurons, D_mem]  — stored information per neuron
      thresholds:  [BS, N_neurons, max_conn] — energy thresholds for routing
      temperature: [BS, N_neurons] — per-neuron routing sharpness
      decay:       [BS, N_neurons] — per-neuron activation persistence
      activations: [BS, N_neurons, D_mem]  — current step accumulation buffer
      prev_output: [BS, N_neurons, D_mem]  — previous step output (read-only)

    Connectivity (fixed topology, shared across batch):
      conn_indices: [N_neurons, max_conn] — index of connected neuron per slot
      conn_mask:    [N_neurons, max_conn] — bool, True if connection exists
      cc_port_idx:  [C] — neuron index of each CC's port
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        self.config = config
        self.device = device
        self.dtype = dtype

        C = config.C
        N_blocks = config.N_blocks
        M = config.M_per_block
        N = config.N_neurons
        D = config.D_mem
        K_intra = config.K_intra
        K_inter = config.K_inter
        max_conn = config.max_connections
        ccs_per_block = config.CCs_per_block

        # --- Build fixed connectivity ---
        conn_indices = torch.zeros(N, max_conn, dtype=torch.long, device=device)
        conn_mask = torch.zeros(N, max_conn, dtype=torch.bool, device=device)

        for b in range(N_blocks):
            block_start = b * M
            block_neurons = list(range(block_start, block_start + M))
            other_neurons = [i for i in range(N)
                             if i < block_start or i >= block_start + M]

            for local_i in range(M):
                neuron_i = block_start + local_i

                # Intra-block: K_intra random connections within block
                candidates = [n for n in block_neurons if n != neuron_i]
                perm = torch.randperm(len(candidates), device=device)[:K_intra]
                for j, p in enumerate(perm):
                    conn_indices[neuron_i, j] = candidates[p.item()]
                    conn_mask[neuron_i, j] = True

                # Inter-block: K_inter random connections to other blocks
                if other_neurons and K_inter > 0:
                    perm = torch.randperm(len(other_neurons), device=device)[:K_inter]
                    for j, p in enumerate(perm):
                        conn_indices[neuron_i, K_intra + j] = other_neurons[p.item()]
                        conn_mask[neuron_i, K_intra + j] = True

        self.conn_indices = conn_indices  # [N_neurons, max_conn]
        self.conn_mask = conn_mask        # [N_neurons, max_conn]

        # CC port neurons: ccs_per_block ports per block, evenly spaced
        # CC c maps to block (c // ccs_per_block), port index (c % ccs_per_block)
        cc_port_idx = []
        for c in range(C):
            block_idx = c // ccs_per_block
            port_within_block = c % ccs_per_block
            # Use first ccs_per_block neurons of each block as ports
            cc_port_idx.append(block_idx * M + port_within_block)
        self.cc_port_idx = torch.tensor(cc_port_idx, device=device, dtype=torch.long)

        # No learned weights in the memory graph.
        # Neuron modulation is element-wise: output = silu(input * primitive).
        # The neuromodulator controls behavior entirely through primitives + thresholds.

        # --- State (allocated on initialize) ---
        self.primitives = None   # [BS, N_neurons, D_mem]
        self.thresholds = None   # [BS, N_neurons, max_conn]
        self.temperature = None  # [BS, N_neurons] — per-neuron routing sharpness
        self.decay = None        # [BS, N_neurons] — per-neuron activation persistence
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
        self.temperature = torch.ones(BS, N, device=dev, dtype=dt)
        self.decay = torch.full((BS, N), -2.0, device=dev, dtype=dt)  # sigmoid(-2)≈0.12, mostly reactive
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

        # 1. Save pre-inject prev_output for decay (avoid CC signal bleed)
        prev_output_for_decay = self.prev_output.clone()

        # 2. Inject CC signals into port neurons' prev_output (for gather)
        for c in range(self.config.C):
            self.prev_output[:, self.cc_port_idx[c]] = (
                self.prev_output[:, self.cc_port_idx[c]] + cc_signals[:, c]
            )

        # 3. Gather inputs: each neuron sums prev_output of connected neurons
        flat_idx = self.conn_indices.reshape(-1)
        gathered_flat = self.prev_output[:, flat_idx]
        gathered = gathered_flat.reshape(BS, N, max_conn, D)

        mask = self.conn_mask.unsqueeze(0).unsqueeze(-1)
        gathered = gathered * mask.float()

        inputs = gathered.sum(dim=2)  # [BS, N, D]

        # 3. Modulate: element-wise gating by primitive
        outputs = F.silu(inputs * self.primitives)  # [BS, N, D]

        # Update running stats
        alpha = 1.0 - self._ema_decay
        self.mean_input = self._ema_decay * self.mean_input + alpha * inputs
        self.mean_output = self._ema_decay * self.mean_output + alpha * outputs
        output_mag = outputs.norm(dim=-1)
        self.usage_count = self._ema_decay * self.usage_count + alpha * (output_mag > 0.01).float()

        # 4. Route with per-neuron temperature
        neuron_temp = self.temperature.unsqueeze(-1).clamp(min=0.01)
        route_logits = -self.thresholds / neuron_temp
        route_logits = route_logits.masked_fill(~self.conn_mask.unsqueeze(0), float('-inf'))

        route_weights = F.softmax(route_logits, dim=-1)
        route_weights = route_weights.nan_to_num(0.0)

        # Sparsity: zero out bottom fraction of routing weights
        if self.config.mem_sparsity > 0:
            k_keep = max(1, int(max_conn * (1.0 - self.config.mem_sparsity)))
            topk_vals, _ = route_weights.topk(k_keep, dim=-1)
            threshold = topk_vals[:, :, -1:]
            route_weights = route_weights * (route_weights >= threshold).float()
            route_sum = route_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            route_weights = route_weights / route_sum

        # Scatter routed signals
        weighted = route_weights.unsqueeze(-1) * outputs.unsqueeze(2)

        self.activations.zero_()
        flat_idx = self.conn_indices.reshape(-1)
        weighted_flat = weighted.reshape(BS, N * max_conn, D)
        idx_expanded = flat_idx.unsqueeze(0).unsqueeze(-1).expand(BS, -1, D)
        self.activations.scatter_add_(1, idx_expanded, weighted_flat)

        # 5. Read CC port activations (clone to avoid aliasing)
        mem_signals = self.activations[:, self.cc_port_idx].clone()

        # 7. Swap with decay (use pre-inject prev_output to avoid CC signal bleed)
        d = torch.sigmoid(self.decay).unsqueeze(-1)
        self.prev_output = d * prev_output_for_decay + (1 - d) * outputs

        return mem_signals

    @torch.no_grad()
    def apply_actions(self, delta_primitives: Tensor, delta_thresholds: Tensor,
                      delta_temperature: Tensor, delta_decay: Tensor):
        """Apply neuromodulator actions to neuron state.

        Args:
            delta_primitives:   [BS, N_neurons, D_mem]
            delta_thresholds:   [BS, N_neurons, max_conn]
            delta_temperature:  [BS, N_neurons]
            delta_decay:        [BS, N_neurons]
        """
        self.primitives = self.primitives + delta_primitives
        self.thresholds = self.thresholds + delta_thresholds
        self.temperature = (self.temperature + delta_temperature).clamp(min=0.01)
        self.decay = self.decay + delta_decay

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
        N_blocks = self.config.N_blocks
        ccs_per_block = self.config.CCs_per_block

        parts = [
            self.primitives,                         # [BS, N, D_mem]
            self.mean_input,                         # [BS, N, D_mem]
            self.mean_output,                        # [BS, N, D_mem]
            self.usage_count.unsqueeze(-1),          # [BS, N, 1]
            self.temperature.unsqueeze(-1),          # [BS, N, 1]
            torch.sigmoid(self.decay).unsqueeze(-1), # [BS, N, 1]
        ]

        # Routing entropy
        neuron_temp = self.temperature.unsqueeze(-1).clamp(min=0.01)
        route_logits = -self.thresholds / neuron_temp
        route_logits = route_logits.masked_fill(~self.conn_mask.unsqueeze(0), float('-inf'))
        route_probs = F.softmax(route_logits, dim=-1).nan_to_num(0.0)
        entropy = -(route_probs * (route_probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        parts.append(entropy)  # [BS, N, 1]

        # Per-neuron CC surprise: each neuron gets its block's mean CC surprise
        if cc_surprise is not None:
            # cc_surprise: [BS, C, D_cc]
            # Average surprise across CCs in each block → [BS, N_blocks, D_cc]
            block_surprise = cc_surprise.view(BS, N_blocks, ccs_per_block, -1).mean(dim=2)
            # Expand to all neurons in each block → [BS, N, D_cc]
            surprise_expanded = block_surprise.unsqueeze(2).expand(
                BS, N_blocks, M, -1
            ).reshape(BS, N, -1)
            parts.append(surprise_expanded)
        else:
            parts.append(torch.zeros(BS, N, self.config.D_cc,
                                     device=self.device, dtype=self.dtype))

        return torch.cat(parts, dim=-1)

    @property
    def obs_dim(self) -> int:
        """Observation dimension for neuromodulator."""
        # D_mem*3 (prim + mean_in + mean_out) + 4 (usage + temp + decay + entropy) + D_cc (surprise)
        return self.config.D_mem * 3 + 4 + self.config.D_cc

    @torch.no_grad()
    def reset_streams(self, mask: Tensor):
        """Reset memory for masked streams (doc boundary)."""
        if not mask.any():
            return
        m3 = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
        m2 = mask.unsqueeze(-1)                 # [BS, 1]

        fresh_prim = torch.randn_like(self.primitives[0:1]) * 0.1
        fresh_thresh = torch.zeros_like(self.thresholds[0:1])
        fresh_zero = torch.zeros_like(self.activations[0:1])

        self.primitives = torch.where(m3, fresh_prim, self.primitives)
        self.thresholds = torch.where(m3, fresh_thresh, self.thresholds)
        self.activations = torch.where(m3, fresh_zero, self.activations)
        self.prev_output = torch.where(m3, fresh_zero, self.prev_output)
        self.mean_input = torch.where(m3, torch.zeros_like(self.mean_input[0:1]), self.mean_input)
        self.mean_output = torch.where(m3, torch.zeros_like(self.mean_output[0:1]), self.mean_output)
        self.usage_count = torch.where(m2, torch.zeros_like(self.usage_count[0:1]), self.usage_count)
        self.temperature = torch.where(m2, torch.ones_like(self.temperature[0:1]), self.temperature)
        self.decay = torch.where(m2, torch.full_like(self.decay[0:1], -2.0), self.decay)
