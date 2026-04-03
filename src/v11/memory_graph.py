"""Cell-Based Memory Graph — flat world of cells with thin neurons.

N_cells cells of C_neurons neurons each. D_neuron=8. All connections cell-local.
R message-passing rounds per token step. Dedicated inject/readout port neurons.

Per-token processing (×R rounds within each cell):
  1. INJECT: inject neurons receive LM signal
  2. GATHER: received = Σ sigmoid(w_conn[k]) × msg[neighbor_k]  (cell-local)
  3. STATE MLP (shared): update from (received+inject, prim, id, decay)
  4. TEMPORAL: h = decay × h + (1-decay) × update
  5. MSG MLP (shared): msg from (h, prim, id) + neuron_id
  6. READOUT (last round): average readout neurons → D_lm

Segment-boundary: per-cell modulator outputs w_conn, decay, primitives.
Structural plasticity: within-cell co-activation rewiring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V11Config

try:
    from .triton_kernels import fused_cell_forward as _fused_cell_fwd
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class CellMemoryGraph(nn.Module):
    """Cell-based memory graph with shared MLPs and per-cell modulator.

    Parameters (backprop-trained):
        state_w1, state_b1, state_w2, state_b2 — shared state MLP
        msg_w1, msg_b1, msg_w2, msg_b2 — shared message MLP
        mod_w1, mod_b1, mod_w2, mod_b2 — per-cell modulator [N_cells, ...]
        neuron_id — per-neuron identity [N_total, D]

    Runtime state (per-batch):
        h, prev_messages, w_conn, primitives_state, decay_logit,
        hebbian_traces, msg_magnitude
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

        # ---- Fixed cell-local connectivity ----
        # conn_indices[cell, neuron, k] = local index (0..C-1) of k-th neighbor
        conn = torch.zeros(NC, C, K, dtype=torch.long, device=device)
        for cell in range(NC):
            all_idx = torch.arange(C, device=device)
            for n in range(C):
                scores = torch.rand(C, device=device)
                scores[n] = -float('inf')  # no self-connections
                conn[cell, n] = scores.topk(K).indices
        self.register_buffer('conn_indices', conn)

        # ---- Fixed neuron role indices per cell ----
        # Layout: [inject(alpha) | border(B) | interneurons | readout(alpha)]
        B = config.N_border_per_cell
        inject_idx = torch.arange(alpha, device=device).unsqueeze(0).expand(NC, -1)
        border_idx = torch.arange(alpha, alpha + B, device=device).unsqueeze(0).expand(NC, -1)
        readout_idx = torch.arange(C - alpha, C, device=device).unsqueeze(0).expand(NC, -1)
        self.register_buffer('inject_indices', inject_idx)    # [NC, alpha]
        self.register_buffer('border_indices', border_idx)    # [NC, B]
        self.register_buffer('readout_indices', readout_idx)  # [NC, alpha]

        # ---- Inter-cell border connectivity ----
        # Each border neuron has K_border connections to border neurons in OTHER cells.
        # Stored as (cell_idx, border_slot) pairs → flattened global border neuron index.
        # Total border neurons = NC * B. Each border neuron connects to K_border of them.
        K_b = config.K_border
        N_border_total = NC * B
        border_conn = torch.zeros(NC, B, K_b, dtype=torch.long, device=device)
        for cell in range(NC):
            for b in range(B):
                # Global index of this border neuron: cell * B + b
                self_global = cell * B + b
                scores = torch.rand(N_border_total, device=device)
                # Exclude all border neurons in own cell
                for own_b in range(B):
                    scores[cell * B + own_b] = -float('inf')
                border_conn[cell, b] = scores.topk(K_b).indices
        self.register_buffer('border_conn_indices', border_conn)  # [NC, B, K_border]

        # Border connection weights (learned via modulator, same as intra-cell w_conn)
        # Stored separately since they index into a different space

        # ================================================================
        # Shared-weight MLPs
        # ================================================================

        # State MLP: input = [input_vec(D) + primitive(D) + neuron_id(D) + decay(1)]
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

        # Message MLP: input = [h(D) + primitive(D) + neuron_id(D)]
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
        # Each neuron has its own modulator weights. Runs once per segment
        # via per-neuron einsum — cost is negligible vs T×R inner loop.
        # Input: hebbian[K] + h[D] + decay[1] + primitives[D] + neuron_id[D]
        mod_in = K + 3 * D + 1
        # Output: w_conn_delta[K] + w_conn_border_delta[K_border] + decay_delta[1] + prim_delta[D]
        K_b = config.K_border
        mod_out = K + K_b + 1 + D
        H_mod = config.cell_mod_hidden
        N_total = NC * C

        # w1 layout: [N_total, H_mod, mod_in] for per-neuron einsum
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
        dt = self.dtype

        B = self.config.N_border_per_cell
        K_b = self.config.K_border

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
            # Per-cell co-activation: [NC, C, C]
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
            'w_conn': self.w_conn, 'primitives_state': self.primitives_state,
            'decay_logit': self.decay_logit,
            'hebbian_traces': self.hebbian_traces,
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
    # Per-cell modulator (segment boundary)
    # ================================================================

    def _run_modulator(self, h: Tensor, hebbian_traces: Tensor,
                       decay_logit: Tensor,
                       primitives: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per-neuron modulator: each neuron has its own MLP weights.

        Runs once per segment via per-neuron einsum. Each neuron gets a
        personalized w_conn, decay, and primitive adjustment based on its
        own state AND its own learned modulation weights.
        """
        BS = h.shape[0]
        NC = self.config.N_cells
        C = self.config.C_neurons
        K = self.config.K_connections
        D = self.config.D_neuron
        N = NC * C

        nid = self.neuron_id  # [NC, C, D]
        idt = h.dtype

        # Per-neuron input: cat(hebbian, h, decay, primitives, neuron_id)
        mod_input = torch.cat([
            hebbian_traces,                    # [BS, NC, C, K]
            h,                                 # [BS, NC, C, D]
            decay_logit.unsqueeze(-1),         # [BS, NC, C, 1]
            primitives,                        # [BS, NC, C, D]
            nid.to(idt).unsqueeze(0).expand(BS, -1, -1, -1),  # [BS, NC, C, D]
        ], dim=-1)  # [BS, NC, C, mod_in]

        # Flatten cell dims for per-neuron einsum: [BS, N, mod_in]
        flat_input = mod_input.reshape(BS, N, -1)

        # Per-neuron einsum: each neuron has its own W1, b1, W2, b2
        # mod_w1: [N, H_mod, mod_in], mod_b1: [N, H_mod]
        hidden = torch.einsum(
            'bni,nhi->bnh', flat_input, self.mod_w1.to(idt)
        ) + self.mod_b1.to(idt)
        hidden = torch.tanh(hidden)
        output = torch.einsum(
            'bnh,nho->bno', hidden, self.mod_w2.to(idt)
        ) + self.mod_b2.to(idt)

        # Reshape back to [BS, NC, C, mod_out]
        output = output.reshape(BS, NC, C, -1)

        # Split into per-neuron adjustments
        K_b = self.config.K_border
        new_w_conn = output[..., :K]                    # [BS, NC, C, K]
        new_w_conn_border = output[..., K:K+K_b]        # [BS, NC, C, K_b]
        new_decay = output[..., K+K_b]                   # [BS, NC, C]
        new_prim = output[..., K+K_b+1:]                 # [BS, NC, C, D]

        # Extract border neurons' border w_conn
        B = self.config.N_border_per_cell
        border_idx = self.border_indices  # [NC, B]
        idx = border_idx.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, K_b)
        new_w_conn_border_neurons = torch.gather(
            new_w_conn_border, 2, idx)  # [BS, NC, B, K_b]

        return new_w_conn, new_w_conn_border_neurons, new_decay, new_prim

    # ================================================================
    # Inject / Readout with port neurons
    # ================================================================

    def _inject(self, cc_signals_t: Tensor) -> Tensor:
        """Per-token inject: [BS, D_lm] → additive signal for inject neurons.

        Returns: [BS, NC, alpha, D] to be added to inject neurons' input.
        """
        BS = cc_signals_t.shape[0]
        D = self.config.D_neuron
        alpha = self.config.alpha
        # [BS, D_lm] → [BS, NC, D] → replicate alpha times
        sliced = cc_signals_t.view(BS, self.config.N_cells, D)
        return sliced.unsqueeze(2).expand(-1, -1, alpha, -1)  # [BS, NC, alpha, D]

    def _readout(self, msg: Tensor) -> Tensor:
        """Per-token readout: msg[BS, NC, C, D] → [BS, D_lm].

        Reads only the readout neurons, averages alpha replicas.
        """
        BS = msg.shape[0]
        NC = self.config.N_cells
        D = self.config.D_neuron
        alpha = self.config.alpha

        # Gather readout neurons: [BS, NC, alpha, D]
        # readout_indices: [NC, alpha] → expand to [BS, NC, alpha, D]
        idx = self.readout_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, D)  # [BS, NC, alpha, D]
        readout_msgs = torch.gather(msg, 2, idx)  # [BS, NC, alpha, D]

        # Average over alpha replicas: [BS, NC, D]
        averaged = readout_msgs.mean(dim=2)

        # Reshape to D_lm: [BS, NC * D] = [BS, D_lm]
        return averaged.reshape(BS, self.config.D)

    # ================================================================
    # Cell-local gather
    # ================================================================

    def _cell_gather(self, msg: Tensor, w_conn_sig: Tensor) -> Tensor:
        """Cell-local weighted gather: each neuron reads K neighbors in its cell.

        Args:
            msg: [BS, NC, C, D] — current messages
            w_conn_sig: [BS, NC, C, K] — sigmoid connection weights

        Returns:
            received: [BS, NC, C, D] — weighted sum of neighbor messages
        """
        BS, NC, C, D = msg.shape
        K = self.config.K_connections

        # conn_indices: [NC, C, K] → expand to [BS, NC, C, K, D]
        idx = self.conn_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, -1, D)  # [BS, NC, C, K, D]

        # Gather neighbor messages: [BS, NC, C, K, D]
        msg_expanded = msg.unsqueeze(3).expand(-1, -1, -1, K, -1)
        # Actually need to gather from cell dim (dim=2)
        neighbor_msgs = torch.gather(
            msg.unsqueeze(2).expand(-1, -1, C, -1, -1),
            3,
            idx)  # This doesn't work cleanly with gather

        # Simpler: reshape and index
        # msg: [BS, NC, C, D] → for each (cell, neuron, k): msg[:, cell, conn[cell,n,k], :]
        # Use advanced indexing
        batch_idx = torch.arange(BS, device=msg.device)[:, None, None, None]
        cell_idx = torch.arange(NC, device=msg.device)[None, :, None, None]
        conn_expanded = self.conn_indices.unsqueeze(0).expand(BS, -1, -1, -1)
        neighbor_msgs = msg[batch_idx, cell_idx, conn_expanded]  # [BS, NC, C, K, D]

        # Weighted sum
        weighted = w_conn_sig.unsqueeze(-1) * neighbor_msgs  # [BS, NC, C, K, D]
        return weighted.sum(dim=3)  # [BS, NC, C, D]

    # ================================================================
    # Inter-cell border gather
    # ================================================================

    def _border_gather(self, msg: Tensor, w_conn_border_sig: Tensor) -> Tensor:
        """Inter-cell gather: border neurons read from border neurons in other cells.

        Args:
            msg: [BS, NC, C, D] — current messages (all neurons)
            w_conn_border_sig: [BS, NC, B, K_border] — sigmoid border weights

        Returns:
            border_received: [BS, NC, B, D] — weighted sum of remote border messages
        """
        BS, NC, C, D = msg.shape
        B = self.config.N_border_per_cell
        K_b = self.config.K_border

        # Extract all border neurons' messages into a flat buffer
        # border_indices: [NC, B] — cell-local indices of border neurons
        border_local_idx = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, D)  # [BS, NC, B, D]
        border_msgs = torch.gather(msg, 2, border_local_idx)  # [BS, NC, B, D]

        # Flatten to global border buffer: [BS, NC*B, D]
        border_flat = border_msgs.reshape(BS, NC * B, D)

        # border_conn_indices: [NC, B, K_border] — global indices into [NC*B]
        conn = self.border_conn_indices.unsqueeze(0).unsqueeze(-1).expand(
            BS, -1, -1, -1, D)  # [BS, NC, B, K_b, D]
        conn_flat = conn.reshape(BS, NC * B, K_b, D)

        # Gather from global border buffer
        # For each border neuron's K_b connections, read the target's message
        batch_idx = torch.arange(BS, device=msg.device)[:, None, None]
        conn_indices_flat = self.border_conn_indices.reshape(NC * B, K_b)
        conn_exp = conn_indices_flat.unsqueeze(0).expand(BS, -1, -1)
        # [BS, NC*B, K_b, D]
        neighbor_msgs = border_flat[:, :, None, :].expand(
            -1, -1, K_b, -1).clone()
        for k in range(K_b):
            neighbor_msgs[:, :, k, :] = border_flat[
                batch_idx.squeeze(-1), conn_exp[:, :, k]]

        # Reshape back to [BS, NC, B, K_b, D] and weight
        neighbor_msgs = neighbor_msgs.reshape(BS, NC, B, K_b, D)
        weighted = w_conn_border_sig.unsqueeze(-1) * neighbor_msgs
        return weighted.sum(dim=3)  # [BS, NC, B, D]

    # ================================================================
    # Shared MLPs
    # ================================================================

    def _state_update(self, input_vec: Tensor, primitives: Tensor,
                      decay_logit: Tensor) -> Tensor:
        """Shared state MLP: (input_vec, prim, neuron_id, decay) → update.

        Args:
            input_vec: [BS, NC, C, D]
            primitives: [BS, NC, C, D]
            decay_logit: [BS, NC, C]

        Returns:
            update: [BS, NC, C, D]
        """
        D = self.config.D_neuron
        dt = input_vec.dtype
        nid = self.neuron_id.to(dt)  # [NC, C, D]

        w_in = self.state_w1[:, :D].to(dt)
        w_prim = self.state_w1[:, D:2*D].to(dt)
        w_id = self.state_w1[:, 2*D:3*D].to(dt)
        w_decay = self.state_w1[:, 3*D:].to(dt)
        b1 = self.state_b1.to(dt)
        w2 = self.state_w2.to(dt)
        b2 = self.state_b2.to(dt)

        hidden = (
            F.linear(input_vec, w_in) +
            F.linear(primitives, w_prim) +
            F.linear(nid, w_id).unsqueeze(0) +
            F.linear(decay_logit.unsqueeze(-1), w_decay) +
            b1
        )
        return torch.tanh(F.linear(torch.tanh(hidden), w2, b2))

    def _msg_output(self, h: Tensor, primitives: Tensor) -> Tensor:
        """Shared message MLP: (h, prim, neuron_id) → msg + neuron_id.

        Args:
            h: [BS, NC, C, D]
            primitives: [BS, NC, C, D]

        Returns:
            msg: [BS, NC, C, D]
        """
        D = self.config.D_neuron
        dt = h.dtype
        nid = self.neuron_id.to(dt)

        w_h = self.msg_w1[:, :D].to(dt)
        w_prim = self.msg_w1[:, D:2*D].to(dt)
        w_id = self.msg_w1[:, 2*D:].to(dt)
        b1 = self.msg_b1.to(dt)
        w2 = self.msg_w2.to(dt)
        b2 = self.msg_b2.to(dt)

        hidden = (
            F.linear(h, w_h) +
            F.linear(primitives, w_prim) +
            F.linear(nid, w_id).unsqueeze(0) +
            b1
        )
        return torch.tanh(F.linear(torch.tanh(hidden), w2, b2)) + nid

    # ================================================================
    # Forward segment
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment with cell-local message passing.

        Args:
            cc_signals: [BS, T_seg, D_lm] — detached H_mid

        Returns:
            mem_out: [BS, T_seg, D_lm]
        """
        BS = cc_signals.shape[0]
        T_seg = cc_signals.shape[1]
        NC = self.config.N_cells
        C = self.config.C_neurons
        D = self.config.D_neuron
        K = self.config.K_connections
        R = self.config.R_rounds
        alpha = self.config.alpha

        # TBPTT: detach at segment boundary
        h = self.h.detach()
        msg = self.prev_messages.detach()

        B = self.config.N_border_per_cell
        K_b = self.config.K_border

        # Run shared modulator (per-neuron input → per-neuron output)
        w_conn, w_conn_border, decay_logit, primitives = self._run_modulator(
            h, self.hebbian_traces.detach(),
            self.decay_logit.detach(),
            self.primitives_state.detach())

        w_conn_sig = torch.sigmoid(w_conn)          # [BS, NC, C, K]
        w_conn_border_sig = torch.sigmoid(w_conn_border)  # [BS, NC, B, K_b]
        decay = torch.sigmoid(decay_logit)           # [BS, NC, C]

        # ---- Fused Triton path (GPU) or Python fallback (CPU) ----
        if _HAS_TRITON and h.is_cuda:
            h_final, msg_final, mem_out, hebb = _fused_cell_fwd(
                h, msg, w_conn_sig, decay, primitives,
                self.conn_indices, self.neuron_id,
                self.inject_indices, self.readout_indices,
                cc_signals,
                self.state_w1, self.state_b1, self.state_w2, self.state_b2,
                self.msg_w1, self.msg_b1, self.msg_w2, self.msg_b2,
                self.config)
            h = h_final
            msg = msg_final
            total_hebbian = hebb
        else:
            # Python fallback (slow, for CPU tests)
            d = decay.unsqueeze(-1)
            omd = 1 - d
            total_hebbian = torch.zeros_like(self.hebbian_traces)
            readouts = []

            for t in range(T_seg):
                inject_signal = self._inject(cc_signals[:, t])

                # R intra-cell rounds
                for r in range(R):
                    received = self._cell_gather(msg, w_conn_sig)

                    # Inter-cell border gather: border neurons receive
                    # from border neurons in other cells
                    border_received = self._border_gather(
                        msg, w_conn_border_sig)  # [BS, NC, B, D]
                    # Add border signal to border neurons' received
                    border_local = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(
                        BS, -1, -1, D)
                    received.scatter_add_(2, border_local,
                                          border_received.to(received.dtype))

                    # Inject to inject neurons
                    inject_idx = self.inject_indices.unsqueeze(0).unsqueeze(-1).expand(
                        BS, -1, -1, D)
                    inject_addend = torch.zeros_like(received)
                    inject_addend.scatter_(2, inject_idx,
                                           inject_signal.to(received.dtype))
                    input_vec = received + inject_addend

                    update = self._state_update(input_vec, primitives, decay_logit)
                    h = d * h + omd * update
                    msg = self._msg_output(h, primitives)

                readouts.append(self._readout(msg))
                with torch.no_grad():
                    msg_norms = msg.detach().norm(dim=-1)
                    total_hebbian += msg_norms.unsqueeze(-1) * w_conn_sig.detach()

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

            alpha_diag = 0.05
            seg_mag = msg.detach().norm(dim=-1)
            self.msg_magnitude = (
                (1 - alpha_diag) * self.msg_magnitude + alpha_diag * seg_mag
            ).to(self.dtype)

        return mem_out

    # ================================================================
    # Structural plasticity (within-cell)
    # ================================================================

    def rewire_connections(self):
        """Within-cell structural plasticity."""
        if not self.config.structural_plasticity:
            return
        if not hasattr(self, 'co_activation_ema'):
            return
        if not self._co_activation_ready:
            return
        # TODO: implement per-cell rewiring
        pass
