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
from ..model.scan import fused_scan

try:
    from .triton_kernels import combined_cell_border_gather as _combined_cell_border_gather
    _HAS_TRITON_GATHER = True
except ImportError:
    _HAS_TRITON_GATHER = False

try:
    from .triton_kernels import fused_token_step as _fused_token_step
    _HAS_TRITON_TOKEN_STEP = True
except ImportError:
    _HAS_TRITON_TOKEN_STEP = False


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

    def _cell_chunk_size(self, BS: int, T_seg: int, dtype: torch.dtype) -> int:
        """Choose a training-friendly cell chunk size for the scan/message path."""
        NC = self.config.N_cells
        if NC <= 32:
            return NC
        # Empirically, smaller chunks reduce backward/checkpoint overhead more
        # than the extra Python loop hurts. A 32-cell chunk is the best tested
        # point on the 4090 for BS in {8, 16, 32}, while still fitting BS=32.
        return 32

    def _run_round_chunk(
        self,
        received_chunk: Tensor,
        primitives_chunk: Tensor,
        decay_chunk: Tensor,
        h_chunk: Tensor,
        inject_chunk: Tensor,
        nid_chunk: Tensor,
        inject_idx_chunk: Tensor,
        readout_idx_chunk: Tensor,
        need_readout: bool,
        need_act_trace: bool,
        state_w_in: Tensor,
        state_w_prim: Tensor,
        state_w_id: Tensor,
        state_w_decay: Tensor,
        state_w2: Tensor,
        state_b1: Tensor,
        state_b2: Tensor,
        msg_w_h: Tensor,
        msg_w_prim: Tensor,
        msg_w_id: Tensor,
        msg_w2: Tensor,
        msg_b1: Tensor,
        msg_b2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one scan round for a contiguous block of cells."""
        BS, cell_chunk, C, D = received_chunk.shape
        T_seg = inject_chunk.shape[1]
        H_state = state_b1.shape[0]
        alpha = self.config.alpha

        base_hidden = (
            F.linear(received_chunk, state_w_in) +
            F.linear(primitives_chunk, state_w_prim) +
            F.linear(nid_chunk, state_w_id).unsqueeze(0) +
            F.linear(decay_chunk.unsqueeze(-1), state_w_decay) +
            state_b1
        )

        inject_proj = F.linear(inject_chunk, state_w_in)
        inject_src = inject_proj.unsqueeze(3).expand(-1, -1, -1, alpha, -1)
        inject_idx_hidden = inject_idx_chunk.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        inject_idx_hidden = inject_idx_hidden.expand(BS, T_seg, -1, -1, H_state)

        hidden_all = base_hidden.unsqueeze(1).expand(BS, T_seg, cell_chunk, C, H_state).clone()
        hidden_all = hidden_all.scatter_add(3, inject_idx_hidden, inject_src)
        hidden_all = torch.tanh(hidden_all)

        update_all = torch.tanh(F.linear(hidden_all, state_w2, state_b2))

        B_flat = BS * cell_chunk * C
        decay_flat = decay_chunk.reshape(B_flat, 1, 1).expand(-1, T_seg, D)
        omd_flat = (1 - torch.sigmoid(decay_chunk)).unsqueeze(-1).reshape(B_flat, 1, 1)
        update_seq = update_all.permute(0, 2, 3, 1, 4).reshape(B_flat, T_seg, D)
        b_flat = omd_flat * update_seq
        h0_flat = h_chunk.reshape(B_flat, D)

        scan_chunk = 16384
        if B_flat <= scan_chunk or not h0_flat.is_cuda:
            h_all_flat = fused_scan(decay_flat, b_flat, h0_flat)
        else:
            parts = []
            for i in range(0, B_flat, scan_chunk):
                j = min(i + scan_chunk, B_flat)
                parts.append(fused_scan(
                    decay_flat[i:j], b_flat[i:j], h0_flat[i:j]))
            h_all_flat = torch.cat(parts, dim=0)

        h_seq = h_all_flat.reshape(BS, cell_chunk, C, T_seg, D)
        h_last = h_seq[:, :, :, -1, :]

        base_msg = (
            F.linear(primitives_chunk, msg_w_prim) +
            F.linear(nid_chunk, msg_w_id).unsqueeze(0) +
            msg_b1
        )
        nid_full = nid_chunk.unsqueeze(0)
        msg_hidden_last = torch.tanh(base_msg + F.linear(h_last, msg_w_h))
        msg_last = torch.tanh(F.linear(msg_hidden_last, msg_w2, msg_b2)) + nid_full

        if need_act_trace:
            msg_hidden = torch.tanh(
                base_msg.unsqueeze(3) + F.linear(h_seq, msg_w_h)
            )
            msg_seq = (
                torch.tanh(F.linear(msg_hidden, msg_w2, msg_b2)) +
                nid_full.unsqueeze(3)
            )
            act_chunk = msg_seq.detach().norm(dim=-1).permute(0, 3, 1, 2).float()
        else:
            msg_seq = None
            act_chunk = h_chunk.new_empty((0,), dtype=torch.float32)

        if need_readout:
            idx_h = readout_idx_chunk.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            idx_h = idx_h.expand(BS, -1, -1, T_seg, D)
            h_read = torch.gather(h_seq, 2, idx_h)

            H_msg = msg_b1.shape[0]
            idx_msg = readout_idx_chunk.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, H_msg)
            base_msg_read = torch.gather(base_msg, 2, idx_msg)
            nid_read = torch.gather(
                nid_full.expand(BS, -1, -1, D),
                2,
                readout_idx_chunk.unsqueeze(0).unsqueeze(-1).expand(BS, -1, -1, D),
            )
            read_hidden = torch.tanh(
                base_msg_read.unsqueeze(3) + F.linear(h_read, msg_w_h)
            )
            readout_msgs = (
                torch.tanh(F.linear(read_hidden, msg_w2, msg_b2)) +
                nid_read.unsqueeze(3)
            )
            readout_chunk = readout_msgs.mean(dim=2).permute(0, 2, 1, 3).contiguous()
        else:
            readout_chunk = h_chunk.new_empty((0,), dtype=h_chunk.dtype)

        return h_last, msg_last, readout_chunk, act_chunk

    # ================================================================
    # Forward segment
    # ================================================================

    def forward_segment(self, cc_signals: Tensor) -> Tensor:
        """Process one segment via scan-batched cell message passing.

        With R=1, all T tokens' MLPs are batched into single F.linear calls,
        and temporal integration uses fused_scan. No Python loop over T.

        For R>1, falls back to a loop over R rounds (each round is scan-batched
        over T).
        """
        BS = cc_signals.shape[0]
        T_seg = cc_signals.shape[1]
        NC = self.config.N_cells
        C = self.config.C_neurons
        D = self.config.D_neuron
        R = self.config.R_rounds
        alpha = self.config.alpha

        if not self._initialized:
            raise RuntimeError(
                "CellMemoryGraph.forward_segment() called before "
                "initialize_states(). Call model.initialize_states(BS) first.")

        # TBPTT: detach at segment boundary
        h = self.h.detach()
        msg = self.prev_messages.detach()

        # Run per-neuron modulator (once per segment)
        w_conn, w_conn_border, decay_logit, primitives = self._run_modulator(
            h, self.hebbian_traces.detach(),
            self.decay_logit.detach(),
            self.primitives_state.detach())

        w_conn_sig = torch.sigmoid(w_conn)
        w_conn_border_sig = torch.sigmoid(w_conn_border)
        dt = h.dtype

        # Pre-cast shared weights (avoid repeated .to() calls)
        nid = self.neuron_id.to(dt)
        state_w_in = self.state_w1[:, :D].to(dt)
        state_w_prim = self.state_w1[:, D:2*D].to(dt)
        state_w_id = self.state_w1[:, 2*D:3*D].to(dt)
        state_w_decay = self.state_w1[:, 3*D:].to(dt)
        state_w2 = self.state_w2.to(dt)
        state_b1 = self.state_b1.to(dt)
        state_b2 = self.state_b2.to(dt)

        msg_w_h = self.msg_w1[:, :D].to(dt)
        msg_w_prim = self.msg_w1[:, D:2*D].to(dt)
        msg_w_id = self.msg_w1[:, 2*D:].to(dt)
        msg_w2 = self.msg_w2.to(dt)
        msg_b1 = self.msg_b1.to(dt)
        msg_b2 = self.msg_b2.to(dt)

        inject_all = cc_signals.reshape(BS, T_seg, NC, D)
        cell_chunk = self._cell_chunk_size(BS, T_seg, dt)
        use_chunk_checkpoint = torch.is_grad_enabled() and cell_chunk < NC
        act_trace = None

        for r in range(R):
            # --- 1. GATHER (one gather, messages frozen for all T) ---
            if _HAS_TRITON_GATHER and msg.is_cuda:
                received = _combined_cell_border_gather(
                    msg, w_conn_sig, self.conn_indices,
                    w_conn_border_sig, self.border_conn_indices, alpha)
            else:
                received = self._cell_gather(msg, w_conn_sig)
                border_received = self._border_gather(msg, w_conn_border_sig)
                border_idx_exp = self.border_indices.unsqueeze(0).unsqueeze(-1).expand(
                    BS, -1, -1, D)
                received = received.scatter_add(
                    2, border_idx_exp, border_received.to(received.dtype))

            h_chunks = []
            msg_chunks = []
            readout_chunks = []
            act_chunks = [] if self.config.structural_plasticity else None

            for c0 in range(0, NC, cell_chunk):
                c1 = min(c0 + cell_chunk, NC)
                chunk_args = (
                    received[:, c0:c1],
                    primitives[:, c0:c1],
                    decay_logit[:, c0:c1],
                    h[:, c0:c1],
                    inject_all[:, :, c0:c1],
                    nid[c0:c1],
                    self.inject_indices[c0:c1],
                    self.readout_indices[c0:c1],
                    r == R - 1,
                    self.config.structural_plasticity and r == R - 1,
                    state_w_in,
                    state_w_prim,
                    state_w_id,
                    state_w_decay,
                    state_w2,
                    state_b1,
                    state_b2,
                    msg_w_h,
                    msg_w_prim,
                    msg_w_id,
                    msg_w2,
                    msg_b1,
                    msg_b2,
                )

                if use_chunk_checkpoint:
                    h_chunk, msg_chunk, readout_chunk, act_chunk = (
                        torch.utils.checkpoint.checkpoint(
                            self._run_round_chunk,
                            *chunk_args,
                            use_reentrant=False,
                        )
                    )
                else:
                    h_chunk, msg_chunk, readout_chunk, act_chunk = self._run_round_chunk(
                        *chunk_args
                    )

                h_chunks.append(h_chunk)
                msg_chunks.append(msg_chunk)
                if r == R - 1:
                    readout_chunks.append(readout_chunk)
                    if act_chunks is not None:
                        act_chunks.append(act_chunk)

            h = torch.cat(h_chunks, dim=1)
            msg = torch.cat(msg_chunks, dim=1)
            if r == R - 1:
                mem_out = torch.cat(readout_chunks, dim=2).reshape(BS, T_seg, NC * D)
                if act_chunks is not None:
                    act_trace = torch.cat(act_chunks, dim=2)

        # --- Diagnostics ---
        with torch.no_grad():
            msg_norms = msg.detach().norm(dim=-1)  # [BS, NC, C]
            total_hebbian = msg_norms.unsqueeze(-1) * w_conn_sig.detach()

        # Update persistent state
        with torch.no_grad():
            self.h = h.detach().to(self.dtype)
            self.prev_messages = msg.detach().to(self.dtype)
            self.w_conn = w_conn.detach().to(self.dtype)
            self.w_conn_border = w_conn_border.detach().to(self.dtype)
            self.primitives_state = primitives.detach().to(self.dtype)
            self.decay_logit = decay_logit.detach().to(self.dtype)
            self.hebbian_traces = total_hebbian.to(self.dtype)

            if (self.config.structural_plasticity and
                    hasattr(self, 'co_activation_ema') and act_trace is not None):
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
        act_trace = act_trace.float()
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
