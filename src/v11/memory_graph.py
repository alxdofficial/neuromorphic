"""Cell-Based Memory Graph — flat world of cells with thin neurons.

N_cells cells of C_neurons neurons each. D_neuron=8. All connections cell-local
except border neurons which have K_border inter-cell connections.
R message-passing rounds per token step. Dedicated inject/readout port neurons.

Per-token processing (×R rounds):
  1. GATHER: intra-cell (K=16) + inter-cell border (K_border=4)
  2. INJECT: inject neurons receive LM signal
  3. STATE MLP (shared F.linear): update from (received+inject, prim, id, decay)
  4. TEMPORAL: h = decay × h + (1-decay) × update
  5. MSG MLP (shared F.linear): msg from (h, prim, id) + neuron_id
  6. READOUT (last round): average readout neurons → D_lm

Per-neuron modulator (once per segment): per-neuron einsum for w_conn, decay, prim.
Structural plasticity: within-cell co-activation rewiring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V11Config


class CellMemoryGraph(nn.Module):
    """Cell-based memory graph with shared MLPs and per-neuron modulator.

    Parameters (backprop-trained):
        state_w1, state_b1, state_w2, state_b2 — shared state MLP
        msg_w1, msg_b1, msg_w2, msg_b2 — shared message MLP
        mod_w1, mod_b1, mod_w2, mod_b2 — per-neuron modulator [N_total, ...]
        neuron_id — per-neuron identity [NC, C, D]

    Runtime state (per-batch):
        h, prev_messages, w_conn, w_conn_border, primitives_state,
        decay_logit, hebbian_traces, hebbian_traces_border, msg_magnitude
    """

    def __init__(self, config: V11Config, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype = dtype

        NC = config.N_cells
        C = config.C_neurons
        D = config.D_neuron
        K = config.K_connections
        alpha = config.alpha
        B = config.N_border_per_cell
        K_b = config.K_border

        # ---- Fixed cell-local connectivity ----
        conn = torch.zeros(NC, C, K, dtype=torch.long, device=device)
        for cell in range(NC):
            for n in range(C):
                scores = torch.rand(C, device=device)
                scores[n] = -float('inf')
                conn[cell, n] = scores.topk(K).indices
        self.register_buffer('conn_indices', conn)

        # ---- Fixed neuron role indices per cell ----
        # Layout: [inject(alpha) | border(B) | interneurons | readout(alpha)]
        # FIX #4: .clone() instead of .expand() to avoid shared-memory views
        inject_idx = torch.arange(alpha, device=device).unsqueeze(0).expand(NC, -1).clone()
        border_idx = torch.arange(alpha, alpha + B, device=device).unsqueeze(0).expand(NC, -1).clone()
        readout_idx = torch.arange(C - alpha, C, device=device).unsqueeze(0).expand(NC, -1).clone()
        self.register_buffer('inject_indices', inject_idx)
        self.register_buffer('border_indices', border_idx)
        self.register_buffer('readout_indices', readout_idx)

        # ---- Inter-cell border connectivity ----
        N_border_total = NC * B
        border_conn = torch.zeros(NC, B, K_b, dtype=torch.long, device=device)
        for cell in range(NC):
            for b in range(B):
                scores = torch.rand(N_border_total, device=device)
                for own_b in range(B):
                    scores[cell * B + own_b] = -float('inf')
                border_conn[cell, b] = scores.topk(K_b).indices
        self.register_buffer('border_conn_indices', border_conn)

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

        # ---- Per-neuron modulator ----
        mod_in = K + 3 * D + 1
        mod_out = K + K_b + 1 + D
        H_mod = config.cell_mod_hidden
        N_total = NC * C

        self.mod_w1 = nn.Parameter(
            torch.randn(N_total, H_mod, mod_in, device=device) *
            (2.0 / (mod_in + H_mod)) ** 0.5)
        self.mod_b1 = nn.Parameter(torch.zeros(N_total, H_mod, device=device))
        self.mod_w2 = nn.Parameter(
            torch.randn(N_total, H_mod, mod_out, device=device) *
            (2.0 / (H_mod + mod_out)) ** 0.5)
        self.mod_b2 = nn.Parameter(torch.zeros(N_total, mod_out, device=device))

        # ---- Neuron identity embedding ----
        self.neuron_id = nn.Parameter(
            torch.randn(NC, C, D, device=device) * (1.0 / D ** 0.5))

        self._initialized = False

    # ================================================================
    # State management
    # ================================================================

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize_states(self, BS: int):
        device = self.neuron_id.device
        NC = self.config.N_cells
        C = self.config.C_neurons
        D = self.config.D_neuron
        K = self.config.K_connections
        B = self.config.N_border_per_cell
        K_b = self.config.K_border
        dt = self.dtype

        self.h = torch.randn(BS, NC, C, D, device=device, dtype=dt) * 0.1
        self.prev_messages = torch.zeros(BS, NC, C, D, device=device, dtype=dt)
        self.w_conn = torch.zeros(BS, NC, C, K, device=device, dtype=dt)
        self.w_conn_border = torch.zeros(BS, NC, B, K_b, device=device, dtype=dt)
        self.decay_logit = torch.zeros(BS, NC, C, device=device, dtype=dt)
        self.primitives_state = torch.zeros(BS, NC, C, D, device=device, dtype=dt)
        self.hebbian_traces = torch.zeros(BS, NC, C, K, device=device, dtype=dt)
        self.hebbian_traces_border = torch.zeros(BS, NC, B, K_b, device=device, dtype=dt)
        self.msg_magnitude = torch.zeros(BS, NC, C, device=device, dtype=dt)

        if self.config.structural_plasticity:
            self.co_activation_ema = torch.zeros(
                NC, C, C, device=device, dtype=torch.float32)
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
            'w_conn': self.w_conn, 'w_conn_border': self.w_conn_border,
            'primitives_state': self.primitives_state,
            'decay_logit': self.decay_logit,
            'hebbian_traces': self.hebbian_traces,
            'hebbian_traces_border': self.hebbian_traces_border,
            'msg_magnitude': self.msg_magnitude,
        }
        if (self.config.structural_plasticity and
                hasattr(self, 'co_activation_ema')):
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
                        f"Runtime state shape mismatch for '{key}': "
                        f"expected {current.shape}, got {val.shape}.")
            setattr(self, key, val)
        self._initialized = True

    # ================================================================
    # Per-neuron modulator (segment boundary)
    # ================================================================

    def _run_modulator(self, h, hebbian_traces, decay_logit, primitives):
        """Per-neuron modulator: each neuron has its own MLP weights."""
        BS = h.shape[0]
        NC = self.config.N_cells
        C = self.config.C_neurons
        K = self.config.K_connections
        K_b = self.config.K_border
        D = self.config.D_neuron
        N = NC * C
        B = self.config.N_border_per_cell
        idt = h.dtype

        nid = self.neuron_id
        mod_input = torch.cat([
            hebbian_traces,
            h,
            decay_logit.unsqueeze(-1),
            primitives,
            nid.to(idt).unsqueeze(0).expand(BS, -1, -1, -1),
        ], dim=-1)

        flat_input = mod_input.reshape(BS, N, -1)
        hidden = torch.einsum(
            'bni,nhi->bnh', flat_input, self.mod_w1.to(idt)
        ) + self.mod_b1.to(idt)
        hidden = torch.tanh(hidden)
        output = torch.einsum(
            'bnh,nho->bno', hidden, self.mod_w2.to(idt)
        ) + self.mod_b2.to(idt)
        output = output.reshape(BS, NC, C, -1)

        new_w_conn = output[..., :K]
        new_w_conn_border_all = output[..., K:K+K_b]
        new_decay = output[..., K+K_b]
        new_prim = output[..., K+K_b+1:]

        # Extract border neurons' border w_conn
        idx = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, K_b)
        new_w_conn_border = torch.gather(new_w_conn_border_all, 2, idx)

        return new_w_conn, new_w_conn_border, new_decay, new_prim

    # ================================================================
    # Gather operations
    # ================================================================

    def _cell_gather(self, msg, w_conn_sig):
        """Cell-local weighted gather: [BS, NC, C, D]."""
        BS, NC, C, D = msg.shape
        batch_idx = torch.arange(BS, device=msg.device)[:, None, None, None]
        cell_idx = torch.arange(NC, device=msg.device)[None, :, None, None]
        conn_expanded = self.conn_indices.unsqueeze(0).expand(BS, -1, -1, -1)
        neighbor_msgs = msg[batch_idx, cell_idx, conn_expanded]
        return (w_conn_sig.unsqueeze(-1) * neighbor_msgs).sum(dim=3)

    def _border_gather(self, msg, w_conn_border_sig):
        """Inter-cell gather: border neurons read from border neurons in other cells.
        FIX #7: Vectorized, no Python for-loop.
        """
        BS, NC, C, D = msg.shape
        B = self.config.N_border_per_cell
        K_b = self.config.K_border

        # Extract border neurons' messages → flat global buffer
        border_local_idx = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, D)
        border_msgs = torch.gather(msg, 2, border_local_idx)  # [BS, NC, B, D]
        border_flat = border_msgs.reshape(BS, NC * B, D)

        # Vectorized gather: [NC, B, K_b] → [NC*B, K_b] → index into border_flat
        conn_flat = self.border_conn_indices.reshape(NC * B, K_b)
        conn_exp = conn_flat.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D)
        border_flat_exp = border_flat.unsqueeze(2).expand(-1, -1, K_b, -1)
        # Gather: for each (border_neuron, k), read border_flat[conn[k]]
        batch_idx = torch.arange(BS, device=msg.device)[:, None, None, None]
        neighbor_msgs = border_flat[batch_idx.squeeze(-1),
                                     conn_flat.unsqueeze(0).expand(BS, -1, -1)]
        # neighbor_msgs: [BS, NC*B, K_b, D]
        neighbor_msgs = neighbor_msgs.reshape(BS, NC, B, K_b, D)
        return (w_conn_border_sig.unsqueeze(-1) * neighbor_msgs).sum(dim=3)

    # ================================================================
    # Inject / Readout
    # ================================================================

    def _inject(self, cc_signals_t):
        """Per-token inject: [BS, D_lm] → [BS, NC, alpha, D]."""
        BS = cc_signals_t.shape[0]
        D = self.config.D_neuron
        alpha = self.config.alpha
        sliced = cc_signals_t.view(BS, self.config.N_cells, D)
        return sliced.unsqueeze(2).expand(-1, -1, alpha, -1)

    def _readout(self, msg):
        """Per-token readout: msg[BS, NC, C, D] → [BS, D_lm]."""
        BS = msg.shape[0]
        D = self.config.D_neuron
        alpha = self.config.alpha
        idx = self.readout_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, D)
        readout_msgs = torch.gather(msg, 2, idx)
        averaged = readout_msgs.mean(dim=2)
        return averaged.reshape(BS, self.config.D)

    # ================================================================
    # Shared MLPs (via F.linear — cuBLAS GEMM)
    # ================================================================

    def _state_update(self, input_vec, primitives, decay_logit):
        """Shared state MLP: (input_vec, prim, neuron_id, decay) → update."""
        D = self.config.D_neuron
        dt = input_vec.dtype
        nid = self.neuron_id.to(dt)
        w_in = self.state_w1[:, :D].to(dt)
        w_prim = self.state_w1[:, D:2*D].to(dt)
        w_id = self.state_w1[:, 2*D:3*D].to(dt)
        w_decay = self.state_w1[:, 3*D:].to(dt)
        hidden = (
            F.linear(input_vec, w_in) +
            F.linear(primitives, w_prim) +
            F.linear(nid, w_id).unsqueeze(0) +
            F.linear(decay_logit.unsqueeze(-1), w_decay) +
            self.state_b1.to(dt)
        )
        return torch.tanh(F.linear(torch.tanh(hidden), self.state_w2.to(dt),
                                    self.state_b2.to(dt)))

    def _msg_output(self, h, primitives):
        """Shared message MLP: (h, prim, neuron_id) → msg + neuron_id."""
        D = self.config.D_neuron
        dt = h.dtype
        nid = self.neuron_id.to(dt)
        w_h = self.msg_w1[:, :D].to(dt)
        w_prim = self.msg_w1[:, D:2*D].to(dt)
        w_id = self.msg_w1[:, 2*D:].to(dt)
        hidden = (
            F.linear(h, w_h) +
            F.linear(primitives, w_prim) +
            F.linear(nid, w_id).unsqueeze(0) +
            self.msg_b1.to(dt)
        )
        return torch.tanh(F.linear(torch.tanh(hidden), self.msg_w2.to(dt),
                                    self.msg_b2.to(dt))) + nid

    # ================================================================
    # Forward segment
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment with cell-local + border message passing.

        Uses shared F.linear (cuBLAS GEMM) for MLPs. No fused Triton kernel.
        Works on both CPU and GPU.
        """
        BS = cc_signals.shape[0]
        T_seg = cc_signals.shape[1]
        NC = self.config.N_cells
        C = self.config.C_neurons
        D = self.config.D_neuron
        R = self.config.R_rounds

        # FIX #8: Raise if uninitialized
        if not self._initialized:
            raise RuntimeError(
                "CellMemoryGraph.forward_segment() called before "
                "initialize_states(). Call model.initialize_states(BS) first.")

        # TBPTT: detach at segment boundary
        h = self.h.detach()
        msg = self.prev_messages.detach()

        # Run per-neuron modulator
        w_conn, w_conn_border, decay_logit, primitives = self._run_modulator(
            h, self.hebbian_traces.detach(),
            self.decay_logit.detach(),
            self.primitives_state.detach())

        w_conn_sig = torch.sigmoid(w_conn)
        w_conn_border_sig = torch.sigmoid(w_conn_border)
        decay = torch.sigmoid(decay_logit)
        d = decay.unsqueeze(-1)
        omd = 1 - d

        total_hebbian = torch.zeros_like(self.hebbian_traces)
        readouts = []
        act_norms_all = []

        # Pre-expand indices (avoid repeated expand in loop)
        border_idx_exp = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, D)
        inject_idx_exp = self.inject_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, D)

        def _one_token_step(h_in, msg_in, inject_sig):
            """Process one token: R rounds of message passing."""
            h_t, msg_t = h_in, msg_in
            for r in range(R):
                received = self._cell_gather(msg_t, w_conn_sig)
                border_received = self._border_gather(msg_t, w_conn_border_sig)
                received = received.scatter_add(
                    2, border_idx_exp, border_received.to(received.dtype))

                inject_addend = torch.zeros_like(received)
                inject_addend.scatter_(2, inject_idx_exp,
                                       inject_sig.to(received.dtype))
                input_vec = received + inject_addend

                update = self._state_update(input_vec, primitives, decay_logit)
                h_t = d * h_t + omd * update
                msg_t = self._msg_output(h_t, primitives)
            return h_t, msg_t

        use_checkpoint = self.training and T_seg > 1

        for t in range(T_seg):
            inject_signal = self._inject(cc_signals[:, t])

            if use_checkpoint:
                h, msg = torch.utils.checkpoint.checkpoint(
                    _one_token_step, h, msg, inject_signal,
                    use_reentrant=False)
            else:
                h, msg = _one_token_step(h, msg, inject_signal)

            readouts.append(self._readout(msg))

            with torch.no_grad():
                msg_norms = msg.detach().norm(dim=-1)
                total_hebbian += msg_norms.unsqueeze(-1) * w_conn_sig.detach()
                act_norms_all.append(msg_norms.float())

        mem_out = torch.stack(readouts, dim=1)

        # Update persistent state
        with torch.no_grad():
            self.h = h.detach().to(self.dtype)
            self.prev_messages = msg.detach().to(self.dtype)
            self.w_conn = w_conn.detach().to(self.dtype)
            self.w_conn_border = w_conn_border.detach().to(self.dtype)
            self.primitives_state = primitives.detach().to(self.dtype)
            self.decay_logit = decay_logit.detach().to(self.dtype)
            self.hebbian_traces = (
                total_hebbian / max(T_seg, 1)).to(self.dtype)

            # FIX #5: Update co-activation for structural plasticity
            if (self.config.structural_plasticity and
                    hasattr(self, 'co_activation_ema') and act_norms_all):
                act_trace = torch.stack(act_norms_all, dim=1)  # [BS, T, NC, C]
                self._update_phi(act_trace)

            alpha_diag = 0.05
            seg_mag = msg.detach().norm(dim=-1)
            self.msg_magnitude = (
                (1 - alpha_diag) * self.msg_magnitude + alpha_diag * seg_mag
            ).to(self.dtype)

        return mem_out

    # ================================================================
    # Structural plasticity (within-cell)
    # ================================================================

    @torch.no_grad()
    def _update_phi(self, act_trace: Tensor):
        """Compute per-cell Pearson phi from activity traces, EMA-smooth.

        FIX #5: Actually implements the co-activation update.
        """
        BS, T_seg, NC, C = act_trace.shape

        for cell in range(NC):
            cell_act = act_trace[:, :, cell, :]  # [BS, T, C]

            threshold = torch.quantile(cell_act, 0.75, dim=1, keepdim=True)
            fired = (cell_act > threshold).float()
            p_i = fired.mean(dim=1, keepdim=True)
            fired_centered = fired - p_i
            var_i = (p_i * (1 - p_i)).squeeze(1).clamp(min=1e-8)
            cov = torch.bmm(fired_centered.transpose(1, 2), fired_centered) / T_seg
            std_i = var_i.sqrt().unsqueeze(2)
            std_j = var_i.sqrt().unsqueeze(1)
            phi = cov / (std_i * std_j).clamp(min=1e-8)
            phi_mean = phi.mean(dim=0)

            ca_decay = self.config.co_activation_ema_decay
            self.co_activation_ema[cell] = (
                ca_decay * self.co_activation_ema[cell] +
                (1 - ca_decay) * phi_mean)

        self._co_activation_ready = True

    def rewire_connections(self):
        """Within-cell structural plasticity: prune/grow by phi rank."""
        if not self.config.structural_plasticity:
            return
        if not hasattr(self, 'co_activation_ema'):
            return
        if not self._co_activation_ready:
            return

        NC = self.config.N_cells
        C = self.config.C_neurons
        K = self.config.K_connections
        explore_frac = self.config.plasticity_exploration_frac
        device = self.conn_indices.device

        total_per_cell = C * K
        n_prune = max(1, int(total_per_cell * self.config.plasticity_pct))

        with torch.no_grad():
            for cell in range(NC):
                phi = self.co_activation_ema[cell].clone()
                phi.fill_diagonal_(0.0)
                conn = self.conn_indices[cell]  # [C, K]

                conn_phi = phi[torch.arange(C, device=device).unsqueeze(1), conn]
                flat_phi = conn_phi.reshape(-1)
                _, prune_flat_idx = flat_phi.topk(n_prune, largest=False)
                prune_n = prune_flat_idx // K
                prune_k = prune_flat_idx % K

                conn_mask = torch.zeros(C, C, dtype=torch.bool, device=device)
                conn_mask.scatter_(1, conn, True)
                conn_mask.fill_diagonal_(True)

                phi_candidates = phi.clone()
                phi_candidates[conn_mask] = -float('inf')
                flat_candidates = phi_candidates.reshape(-1)
                _, grow_flat_idx = flat_candidates.topk(n_prune, largest=True)
                grow_target = grow_flat_idx % C

                rand_targets = torch.randint(0, C, (n_prune,), device=device)
                use_random = torch.rand(n_prune, device=device) < explore_frac
                grow_target = torch.where(use_random, rand_targets, grow_target)

                conn[prune_n, prune_k] = grow_target
                sorted_idx, _ = conn.sort(dim=-1)
                self.conn_indices[cell].copy_(sorted_idx)

        self._last_rewire_swaps = n_prune
