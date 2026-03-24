"""Memory graph with backprop training (Phase 1).

Same neuron dynamics as the RL-trained MemoryGraph, but:
- primitives, conn_weights, decay_logit are nn.Parameters (no batch dim)
- No neuromodulator — parameters trained directly by backprop
- Gradient flows through the graph via K-step windows (detach every K steps)
- Uses Python path only (autograd-compatible, no Triton in-place ops)
- Normalizations applied after optimizer step, not in the loop

Per-token dynamics (unchanged from RL version):
    1. received = A @ prev_messages (+ CC signal for ports)
    2. h = decay * h + (1-decay) * received
    3. prev_messages = tanh(h * primitives)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config


class MemoryGraphBackprop(nn.Module):
    """Backprop-trainable memory graph.

    Parameters (nn.Parameters, no batch dim):
        primitives: [N, D_mem] — RMS-normalized per neuron
        conn_weights: [N, K] — L1-normalized per neuron
        decay_logit: [N] — clamped so sigmoid in [decay_min, decay_max]

    State (persistent across chunks, detached at chunk boundaries):
        h: [BS, N, D_mem] — internal neuron state
        prev_messages: [BS, N, D_mem] — last outgoing messages
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

        # Decay bounds
        self._logit_min = math.log(config.decay_min / (1 - config.decay_min))
        self._logit_max = math.log(config.decay_max / (1 - config.decay_max))

        # Fixed topology: random sparse connectivity [N, K]
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

        self.register_buffer('conn_indices', conn_indices)
        self.register_buffer('conn_mask', conn_mask)
        self._sort_conn_indices()
        self.register_buffer('cc_port_idx',
                             torch.arange(config.C, device=device))

        # Learnable parameters (no batch dim — shared across batch)
        prim_raw = 1.0 + torch.randn(N, D, device=device, dtype=dtype) * 0.02
        rms = (prim_raw ** 2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.primitives = nn.Parameter(prim_raw / rms)

        raw_w = torch.rand(N, K_conn, device=device, dtype=dtype) + 0.01
        self.conn_weights = nn.Parameter(
            raw_w / raw_w.abs().sum(dim=-1, keepdim=True))

        self.decay_logit = nn.Parameter(
            torch.zeros(N, device=device, dtype=dtype))

        # State (set in initialize, NOT parameters)
        self.h = None
        self.prev_messages = None
        self._initialized = False

        # Stats for plasticity and diagnostics (not parameters)
        self.mean_input = None
        self.mean_output = None
        self.activation_ema = None
        self.activation_std_ema = None
        self.firing_rate = None
        self.co_activation_ema = torch.zeros(
            N, N, device=device, dtype=torch.float32)
        self._plasticity_rewires = 0
        self._co_activation_ready = False

        # Adjacency cache
        self._adjacency_dirty = True
        self._adjacency_cache = None

    def _sort_conn_indices(self):
        sorted_idx, order = self.conn_indices.sort(dim=-1)
        self.conn_indices.copy_(sorted_idx)
        self.conn_mask.copy_(self.conn_mask.gather(1, order))

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, BS: int):
        """Initialize per-batch state (h, prev_messages, running stats)."""
        N = self.config.N_neurons
        D = self.config.D_mem

        self.h = torch.zeros(BS, N, D, device=self.device, dtype=self.dtype)
        self.prev_messages = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)

        self.mean_input = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.mean_output = torch.zeros(
            BS, N, D, device=self.device, dtype=self.dtype)
        self.activation_ema = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)
        self.activation_std_ema = torch.ones(
            BS, N, device=self.device, dtype=self.dtype) * self.config.activation_std_init
        self.firing_rate = torch.zeros(
            BS, N, device=self.device, dtype=self.dtype)

        self._initialized = True
        self._adjacency_dirty = True

    def _build_adjacency(self) -> Tensor:
        """Build dense [N, N] adjacency from sparse connectivity.

        No batch dim — conn_weights is [N, K], expanded to [BS, N, N] via scatter.
        """
        if not self._adjacency_dirty and self._adjacency_cache is not None:
            return self._adjacency_cache

        N = self.config.N_neurons
        K_conn = self.config.K_connections

        # conn_weights is [N, K], need to scatter into [N, N]
        w_masked = self.conn_weights * self.conn_mask.to(dtype=self.dtype)
        A = torch.zeros(N, N, device=self.device, dtype=self.dtype)
        A.scatter_add_(1, self.conn_indices, w_masked)

        self._adjacency_cache = A
        self._adjacency_dirty = False
        return A

    def forward_chunk(self, cc_signals: Tensor,
                      eot_mask: Tensor | None = None,
                      update_co_activation: bool = False) -> Tensor:
        """Process a full chunk of tokens with K-step gradient windows.

        Args:
            cc_signals: [BS, T, C, D_mem] — CC signals (NOT detached)
            eot_mask: [BS, T] bool — True where previous token was EOT
            update_co_activation: if True, compute phi for structural plasticity

        Returns:
            mem_signals: [BS, T, C, D_mem] — port neuron messages
        """
        BS, T, C, D = cc_signals.shape
        N = self.config.N_neurons
        K = self.config.grad_window

        # Normalize CC signals to unit norm (match graph internal scale)
        cc_norm = cc_signals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cc_signals = cc_signals / cc_norm

        # Build adjacency [N, N] — differentiable w.r.t. conn_weights
        A = self._build_adjacency()  # [N, N]
        # Expand for batch bmm: [BS, N, N]
        A_batch = A.unsqueeze(0).expand(BS, N, N)

        decay = torch.sigmoid(self.decay_logit).unsqueeze(-1)  # [N, 1]
        one_minus_decay = 1.0 - decay  # [N, 1]
        # Expand for batch: [BS, N, 1]
        decay_batch = decay.unsqueeze(0).expand(BS, N, 1)
        omd_batch = one_minus_decay.unsqueeze(0).expand(BS, N, 1)
        # Primitives: [N, D] → [BS, N, D]
        prim_batch = self.primitives.unsqueeze(0).expand(BS, N, D)

        h = self.h
        prev_msg = self.prev_messages
        output = torch.empty(BS, T, C, D, device=self.device, dtype=self.dtype)

        # Track activation magnitudes for firing/co-activation stats
        act_norms = []

        # Handle EOT
        has_eot = None
        if eot_mask is not None:
            with torch.no_grad():
                has_eot = eot_mask.any(dim=0).tolist()  # [T] bools (CPU)

        for t in range(T):
            # K-step gradient window: detach every K steps
            if t % K == 0:
                prev_msg = prev_msg.detach()
                h = h.detach()

            # 1. Receive: route presynaptic messages through graph
            # A_batch is [BS, N, N] but derived from [N, N] — gradient flows to conn_weights
            received = torch.bmm(A_batch, prev_msg)  # [BS, N, D]

            # Port neurons receive CC signal
            received[:, :C] = received[:, :C] + cc_signals[:, t]

            # 2. Integrate
            if has_eot is not None and has_eot[t]:
                eot_t = eot_mask[:, t].view(BS, 1, 1).to(dtype=h.dtype)
                d_t = decay_batch * (1.0 - eot_t)
                omd_t = 1.0 - d_t
                h = d_t * h + omd_t * received
            else:
                h = decay_batch * h + omd_batch * received

            # 3. Compute outgoing message
            prev_msg = torch.tanh(h * prim_batch)

            # Store port neuron output
            output[:, t] = prev_msg[:, :C]

            # Track activation norm (detached, for stats only)
            with torch.no_grad():
                act_norms.append(prev_msg.detach().norm(dim=-1))

        # Save state for next chunk
        self.h = h.detach()
        self.prev_messages = prev_msg.detach()

        # Update running stats (detached, for diagnostics/plasticity)
        with torch.no_grad():
            stats_alpha = 1.0 - self.config.plasticity_ema_decay
            self.mean_output = ((1 - stats_alpha) * self.mean_output
                                + stats_alpha * prev_msg.detach())
            self.mean_input = ((1 - stats_alpha) * self.mean_input
                               + stats_alpha * h.detach())

            # Firing stats from activation norms
            act_trace = torch.stack(act_norms, dim=1)  # [BS, T, N]
            self._update_firing_stats(act_trace, update_co_activation)

        return output

    @torch.no_grad()
    def _update_firing_stats(self, act_trace: Tensor,
                             update_co_activation: bool = False):
        """Update firing threshold, firing rate, and optionally co-activation."""
        stats_alpha = 1.0 - self.config.plasticity_ema_decay
        BS, T, N = act_trace.shape

        seg_mean = act_trace.mean(dim=1)
        seg_std = act_trace.std(dim=1, correction=0)
        self.activation_ema = (
            (1 - stats_alpha) * self.activation_ema + stats_alpha * seg_mean)
        self.activation_std_ema = (
            (1 - stats_alpha) * self.activation_std_ema + stats_alpha * seg_std)

        threshold = (self.activation_ema + self.activation_std_ema).unsqueeze(1)
        fired = (act_trace > threshold).float()
        seg_fire_rate = fired.mean(dim=1)
        self.firing_rate = (
            (1 - stats_alpha) * self.firing_rate + stats_alpha * seg_fire_rate)

        if not update_co_activation:
            return

        p_i = fired.mean(dim=1, keepdim=True)
        fired_centered = fired - p_i
        var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)
        cov = torch.bmm(
            fired_centered.transpose(1, 2), fired_centered) / T
        std_i = var_i.sqrt().unsqueeze(2)
        std_j = var_i.sqrt().unsqueeze(1)
        phi = cov / (std_i * std_j).clamp(min=1e-8)

        phi_mean = phi.mean(dim=0).float()
        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = (
            ca_decay * self.co_activation_ema + (1 - ca_decay) * phi_mean)
        self._co_activation_ready = True

    @torch.no_grad()
    def structural_plasticity(self):
        """Co-activation-based structural plasticity (same as RL version)."""
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

        existing_mask = torch.zeros(
            n_prune, N, dtype=torch.bool, device=self.device)
        prune_conn_idx = self.conn_indices[prune_neurons]
        existing_mask.scatter_(1, prune_conn_idx, True)
        existing_mask[torch.arange(n_prune, device=self.device),
                      prune_neurons] = True

        phi_masked = phi[prune_neurons].clone()
        phi_masked[existing_mask] = -float('inf')
        best_targets = phi_masked.argmax(dim=-1)
        best_phi = phi_masked[
            torch.arange(n_prune, device=self.device), best_targets]

        random_targets = torch.randint(0, N, (n_prune,), device=self.device)
        use_random = torch.rand(n_prune, device=self.device) < explore_frac
        use_random = use_random | (best_phi <= 0)
        new_targets = torch.where(use_random, random_targets, best_targets)

        worst_k_for_prune = worst_k[prune_neurons]
        self.conn_indices[prune_neurons, worst_k_for_prune] = new_targets

        # Init new connection weight to median of existing weights
        with torch.no_grad():
            median_w = self.conn_weights[prune_neurons].abs().median(
                dim=-1).values
            self.conn_weights.data[prune_neurons, worst_k_for_prune] = median_w

        self._plasticity_rewires += n_prune
        self._adjacency_dirty = True
        self._sort_conn_indices()

    @torch.no_grad()
    def project_params(self):
        """Project parameters back to constraints. Call after optimizer.step()."""
        # RMS-normalize primitives
        rms = (self.primitives.data ** 2).mean(
            dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        self.primitives.data.div_(rms)

        # L1-normalize conn_weights
        w_sum = self.conn_weights.data.abs().sum(
            dim=-1, keepdim=True).clamp(min=1e-8)
        self.conn_weights.data.div_(w_sum)

        # Clamp decay_logit
        self.conn_weights.data.clamp_(min=-1.0)  # prevent large negatives
        self.decay_logit.data.clamp_(self._logit_min, self._logit_max)

        self._adjacency_dirty = True

    @torch.no_grad()
    def reset_streams(self, mask: Tensor):
        """Reset state for batch elements at document boundaries."""
        m1 = mask.unsqueeze(-1).to(dtype=self.dtype)
        m2 = m1.unsqueeze(-1)
        keep1 = 1.0 - m1
        keep2 = 1.0 - m2

        self.h = self.h * keep2
        self.prev_messages = self.prev_messages * keep2
        self.activation_ema = self.activation_ema * keep1
        self.activation_std_ema = (
            self.activation_std_ema * keep1
            + m1 * self.config.activation_std_init)
        self.firing_rate = self.firing_rate * keep1

    def detach_state(self):
        """Detach persistent state at chunk boundaries."""
        if self.h is not None:
            self.h = self.h.detach()
        if self.prev_messages is not None:
            self.prev_messages = self.prev_messages.detach()

    @torch.no_grad()
    def get_neuron_obs(self) -> Tensor:
        """Build observation tensor (for diagnostics, same format as RL version)."""
        parts = [
            self.primitives.unsqueeze(0).expand_as(self.mean_input),
            self.mean_input,
            self.mean_output,
            self.firing_rate.unsqueeze(-1),
            torch.sigmoid(self.decay_logit).unsqueeze(0).expand(
                self.firing_rate.shape[0], -1).unsqueeze(-1),
        ]
        w_abs = self.conn_weights.abs()
        eps = torch.tensor(1e-8, dtype=self.dtype, device=self.device)
        entropy = -(w_abs * (w_abs + eps).log()).sum(dim=-1, keepdim=True)
        parts.append(entropy.unsqueeze(0).expand(
            self.firing_rate.shape[0], -1, -1))
        return torch.cat(parts, dim=-1)

    @property
    def obs_dim(self) -> int:
        return self.config.D_mem * 3 + 3
