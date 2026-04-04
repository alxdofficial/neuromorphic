"""Memory Graph — sequential differentiable neuron simulation.

N neurons with D-dimensional state, K sparse random connections each.
Shared-weight MLPs conditioned on per-neuron identity. Per-neuron
modulator predicts deltas (not raw values) with depth-scaled init.
Fully sequential: gather reads LIVE messages every token step.

Per-token step:
  1. GATHER: received = Σ sigmoid(w_conn[k]) × msg[neighbor_k]
  2. INJECT: inject neurons receive LM signal
  3. STATE MLP (shared): update = tanh(MLP(received+inject, prim, id, decay))
  4. TEMPORAL: h = decay × h + (1-decay) × update
  5. MSG MLP (shared): msg = tanh(MLP(h, prim, id)) + neuron_id
  6. READOUT: average readout neurons → D_lm

Per-neuron modulator (once per segment):
  Predicts DELTAS for w_conn, decay, primitives. Added to previous values.
  Depth-scaled W2 init (0.01×) so initial deltas ≈ 0.
  RMS-normalized primitives after update to prevent drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config


class MemoryGraph(nn.Module):
    """Flat neuron graph with shared MLPs and per-neuron modulator.

    Parameters (backprop-trained):
        state_w1, state_b1, state_w2, state_b2 — shared state MLP
        msg_w1, msg_b1, msg_w2, msg_b2 — shared message MLP
        mod_w1, mod_b1, mod_w2, mod_b2 — per-neuron modulator [N, ...]
        neuron_id — per-neuron identity [N, D]

    Runtime state (per-batch):
        h, prev_messages, w_conn, primitives_state, decay_logit,
        hebbian_traces, msg_magnitude
    """

    def __init__(self, config: V8Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        N = config.N_neurons
        K = config.K_connections
        D = config.D_neuron

        # ---- Random sparse connectivity ----
        all_idx = torch.arange(N, device=device)
        scores = torch.rand(N, N, device=device)
        scores[all_idx, all_idx] = -float('inf')
        conn = scores.topk(K, dim=1).indices
        conn, _ = conn.sort(dim=-1)
        self.register_buffer('conn_indices', conn)  # [N, K]

        # ---- Inject/readout neuron indices ----
        C_mem = config.C_mem
        N_per_slice = config.N_per_slice
        alpha = min(4, N_per_slice)
        self.alpha = alpha
        inject_idx = []
        readout_idx = []
        for s in range(C_mem):
            base = s * N_per_slice
            inject_idx.extend(range(base, base + alpha))
            readout_idx.extend(range(base + N_per_slice - alpha, base + N_per_slice))
        self.register_buffer('inject_indices',
                             torch.tensor(inject_idx, device=device, dtype=torch.long))
        self.register_buffer('readout_indices',
                             torch.tensor(readout_idx, device=device, dtype=torch.long))
        self.N_inject = len(inject_idx)
        self.N_readout = len(readout_idx)

        # ================================================================
        # Shared-weight MLPs
        # ================================================================

        state_in = 3 * D + 1
        H_s = config.state_mlp_hidden
        self.state_w1 = nn.Parameter(
            torch.randn(H_s, state_in, device=device) *
            (2.0 / (state_in + H_s)) ** 0.5)
        self.state_b1 = nn.Parameter(torch.zeros(H_s, device=device))
        self.state_w2 = nn.Parameter(
            torch.randn(D, H_s, device=device) *
            (2.0 / (H_s + D)) ** 0.5)
        self.state_b2 = nn.Parameter(torch.zeros(D, device=device))

        msg_in = 3 * D
        H_m = config.msg_mlp_hidden
        self.msg_w1 = nn.Parameter(
            torch.randn(H_m, msg_in, device=device) *
            (2.0 / (msg_in + H_m)) ** 0.5)
        self.msg_b1 = nn.Parameter(torch.zeros(H_m, device=device))
        self.msg_w2 = nn.Parameter(
            torch.randn(D, H_m, device=device) *
            (2.0 / (H_m + D)) ** 0.5)
        self.msg_b2 = nn.Parameter(torch.zeros(D, device=device))

        # ---- Per-neuron modulator (delta prediction) ----
        mod_in = K + 3 * D + 1
        mod_out = K + 1 + D
        H_mod = config.neuromod_hidden

        self.mod_w1 = nn.Parameter(
            torch.randn(N, H_mod, mod_in, device=device) *
            (2.0 / (mod_in + H_mod)) ** 0.5)
        self.mod_b1 = nn.Parameter(torch.zeros(N, H_mod, device=device))
        self.mod_w2 = nn.Parameter(
            torch.randn(N, H_mod, mod_out, device=device) *
            (2.0 / (H_mod + mod_out)) ** 0.5 * 0.01)
        self.mod_b2 = nn.Parameter(torch.zeros(N, mod_out, device=device))

        # ---- Neuron identity embedding ----
        self.neuron_id = nn.Parameter(
            torch.randn(N, D, device=device) * (1.0 / D ** 0.5))

        self.C_mem = C_mem
        self.N_per_slice = N_per_slice
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        device = self.neuron_id.device
        N, D, K = self.config.N_neurons, self.config.D_neuron, self.config.K_connections
        dt = self.dtype

        self.h = torch.randn(BS, N, D, device=device, dtype=dt) * 0.1
        self.prev_messages = torch.zeros(BS, N, D, device=device, dtype=dt)
        self.w_conn = torch.zeros(BS, N, K, device=device, dtype=dt)
        self.decay_logit = torch.zeros(BS, N, device=device, dtype=dt)
        self.primitives_state = torch.zeros(BS, N, D, device=device, dtype=dt)
        self.hebbian_traces = torch.zeros(BS, N, K, device=device, dtype=dt)
        self.msg_magnitude = torch.zeros(BS, N, device=device, dtype=dt)

        if self.config.structural_plasticity:
            self.co_activation_ema = torch.zeros(N, N, device=device, dtype=torch.float32)
            self._co_activation_ready = False

        self._initialized = True

    def detach_states(self):
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
        if self.config.structural_plasticity and hasattr(self, 'co_activation_ema'):
            state['co_activation_ema'] = self.co_activation_ema
            state['_co_activation_ready'] = self._co_activation_ready
        return state

    def load_runtime_state(self, state: dict):
        for key, val in state.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if isinstance(current, Tensor) and isinstance(val, Tensor):
                if current.shape != val.shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': expected {current.shape}, got {val.shape}.")
            setattr(self, key, val)
        self._initialized = True

    # ================================================================
    # Modulator (delta prediction, once per segment)
    # ================================================================

    def _run_modulator(self, h, hebbian_traces, decay_logit, primitives):
        """Per-neuron modulator: predicts deltas added to previous values."""
        BS, N, K, D = h.shape[0], self.config.N_neurons, self.config.K_connections, self.config.D_neuron
        idt = h.dtype

        mod_input = torch.cat([
            hebbian_traces,
            h,
            decay_logit.unsqueeze(-1),
            primitives,
            self.neuron_id.to(idt).unsqueeze(0).expand(BS, -1, -1),
        ], dim=-1)

        hidden = torch.einsum(
            'bni,nhi->bnh', mod_input, self.mod_w1.to(idt)
        ) + self.mod_b1.to(idt)
        hidden = torch.tanh(hidden)
        delta = torch.einsum(
            'bnh,nho->bno', hidden, self.mod_w2.to(idt)
        ) + self.mod_b2.to(idt)

        new_w_conn = self.w_conn + delta[..., :K]
        new_decay = decay_logit + delta[..., K]
        new_prim = primitives + delta[..., K+1:]
        rms = new_prim.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        new_prim = new_prim * rms

        return new_w_conn, new_decay, new_prim

    # ================================================================
    # Inject / Readout
    # ================================================================

    def _inject(self, H_mid_t: Tensor) -> Tensor:
        """Per-token inject: [BS, D_lm] → [BS, N_inject, D]."""
        BS = H_mid_t.shape[0]
        D = self.config.D_neuron
        sliced = H_mid_t.view(BS, self.C_mem, D)
        return sliced.unsqueeze(2).expand(-1, -1, self.alpha, -1).reshape(
            BS, self.N_inject, D)

    def _readout(self, msg: Tensor) -> Tensor:
        """Per-token readout: msg[BS, N, D] → [BS, D_lm]."""
        BS = msg.shape[0]
        D = self.config.D_neuron
        readout_msgs = msg[:, self.readout_indices]
        grouped = readout_msgs.view(BS, self.C_mem, self.alpha, D)
        return grouped.mean(dim=2).reshape(BS, self.config.D)

    # ================================================================
    # Forward segment (sequential, live messages)
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Sequential per-token processing with live message propagation."""
        BS, T_seg = cc_signals.shape[0], cc_signals.shape[1]
        D = self.config.D_neuron

        if not self._initialized:
            raise RuntimeError("Call initialize_states(BS) first.")

        h = self.h.detach()
        msg = self.prev_messages.detach()

        w_conn, decay_logit, primitives = self._run_modulator(
            h, self.hebbian_traces.detach(),
            self.decay_logit.detach(),
            self.primitives_state.detach())

        w_conn_sig = torch.sigmoid(w_conn)
        dt = h.dtype

        nid = self.neuron_id.to(dt)
        state_w_in = self.state_w1[:, :D].to(dt)
        state_w2 = self.state_w2.to(dt)
        state_b2 = self.state_b2.to(dt)
        msg_w_h = self.msg_w1[:, :D].to(dt)
        msg_w2 = self.msg_w2.to(dt)
        msg_b2 = self.msg_b2.to(dt)

        state_const = (
            F.linear(primitives, self.state_w1[:, D:2*D].to(dt)) +
            F.linear(nid, self.state_w1[:, 2*D:3*D].to(dt)).unsqueeze(0) +
            F.linear(decay_logit.unsqueeze(-1), self.state_w1[:, 3*D:].to(dt)) +
            self.state_b1.to(dt)
        )
        msg_const = (
            F.linear(primitives, self.msg_w1[:, D:2*D].to(dt)) +
            F.linear(nid, self.msg_w1[:, 2*D:].to(dt)).unsqueeze(0) +
            self.msg_b1.to(dt)
        )

        decay = torch.sigmoid(decay_logit)
        d = decay.unsqueeze(-1)
        omd = 1 - d

        readouts = []
        act_norms = []
        hebbian_accum = torch.zeros_like(self.hebbian_traces)

        for t in range(T_seg):
            received = msg[:, self.conn_indices]  # [BS, N, K, D]
            received = (w_conn_sig.unsqueeze(-1) * received).sum(dim=2)

            inject_signal = cc_signals[:, t].view(BS, self.C_mem, D)
            inject_signal = inject_signal.unsqueeze(2).expand(
                -1, -1, self.alpha, -1).reshape(BS, self.N_inject, D)
            received[:, self.inject_indices] = (
                received[:, self.inject_indices] + inject_signal)

            hidden = torch.tanh(F.linear(received, state_w_in) + state_const)
            update = torch.tanh(F.linear(hidden, state_w2, state_b2))
            h = d * h + omd * update

            msg_hidden = torch.tanh(F.linear(h, msg_w_h) + msg_const)
            msg = torch.tanh(F.linear(msg_hidden, msg_w2, msg_b2)) + nid

            readout_msgs = msg[:, self.readout_indices]
            grouped = readout_msgs.view(BS, self.C_mem, self.alpha, D)
            readouts.append(grouped.mean(dim=2).reshape(BS, self.config.D))

            with torch.no_grad():
                msg_norms_t = msg.detach().norm(dim=-1)
                hebbian_accum += msg_norms_t.unsqueeze(-1) * w_conn_sig.detach()
                act_norms.append(msg_norms_t.float())

        mem_out = torch.stack(readouts, dim=1)

        with torch.no_grad():
            msg_norms = msg.detach().norm(dim=-1)
            self.h = h.detach().to(self.dtype)
            self.prev_messages = msg.detach().to(self.dtype)
            self.w_conn = w_conn.detach().to(self.dtype)
            self.primitives_state = primitives.detach().to(self.dtype)
            self.decay_logit = decay_logit.detach().to(self.dtype)
            # Hebbian: true per-segment average across all T tokens
            self.hebbian_traces = (
                hebbian_accum / max(T_seg, 1)
            ).to(self.dtype)
            self.msg_magnitude = (
                0.95 * self.msg_magnitude + 0.05 * msg_norms
            ).to(self.dtype)

            if (self.config.structural_plasticity and
                    hasattr(self, 'co_activation_ema') and act_norms):
                act_trace = torch.stack(act_norms, dim=1)  # [BS, T, N]
                self._update_phi(act_trace)

        return mem_out

    # ================================================================
    # Structural plasticity
    # ================================================================

    @torch.no_grad()
    def _update_phi(self, act_trace: Tensor):
        BS, T_seg, N = act_trace.shape
        threshold = torch.quantile(act_trace, 0.75, dim=1, keepdim=True)
        fired = (act_trace > threshold).float()
        p_i = fired.mean(dim=1, keepdim=True)
        fired_centered = fired - p_i
        var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)
        cov = torch.bmm(fired_centered.transpose(1, 2), fired_centered) / T_seg
        std_i = var_i.sqrt().unsqueeze(2)
        std_j = var_i.sqrt().unsqueeze(1)
        phi = cov / (std_i * std_j).clamp(min=1e-8)
        ca_decay = self.config.co_activation_ema_decay
        self.co_activation_ema = (
            ca_decay * self.co_activation_ema + (1 - ca_decay) * phi.mean(dim=0))
        self._co_activation_ready = True

    def rewire_connections(self):
        if not self.config.structural_plasticity:
            return
        if not hasattr(self, 'co_activation_ema') or not self._co_activation_ready:
            return
        N, K = self.config.N_neurons, self.config.K_connections
        phi = self.co_activation_ema
        phi.fill_diagonal_(0.0)
        device = phi.device
        conn = self.conn_indices
        n_prune = max(1, int(N * K * self.config.plasticity_pct))

        with torch.no_grad():
            conn_phi = phi[torch.arange(N, device=device).unsqueeze(1), conn]
            _, prune_idx = conn_phi.reshape(-1).topk(n_prune, largest=False)
            prune_n, prune_k = prune_idx // K, prune_idx % K

            conn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            conn_mask.scatter_(1, conn, True)
            conn_mask.fill_diagonal_(True)
            phi_cand = phi.clone()
            phi_cand[conn_mask] = -float('inf')
            _, grow_idx = phi_cand.reshape(-1).topk(n_prune, largest=True)
            grow_target = grow_idx % N

            # Random exploration: sample targets excluding self-connections
            rand_t = torch.randint(0, N - 1, (n_prune,), device=device)
            # Shift past self: if rand_t >= prune_n, add 1 to skip self
            rand_t = rand_t + (rand_t >= prune_n).long()
            use_rand = torch.rand(n_prune, device=device) < self.config.plasticity_exploration_frac
            grow_target = torch.where(use_rand, rand_t, grow_target)

            conn[prune_n, prune_k] = grow_target
            self.conn_indices.copy_(conn.sort(dim=-1).values)
