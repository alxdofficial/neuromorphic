"""Dense-W cell-grid memory graph.

Connectivity within each cell is a dense N×N weight matrix W, updated by
a neuromodulator with low-rank deltas. Message passing is a single batched
matmul per step. No sparse gather, no Triton kernels in the hot path.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config


class MemoryGraph(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.D_n = config.D_n
        self.N_cells = config.N_cells
        self.C_n = config.neurons_per_cell
        self.alpha = config.alpha
        self.border_per_cell = config.border_per_cell
        self.grid_h = config.grid_h
        self.grid_w = config.grid_w

        self.input_lo = 0
        self.input_hi = self.alpha
        self.output_lo = self.input_hi
        self.output_hi = self.output_lo + self.alpha
        self.border_lo = self.output_hi
        self.border_hi = self.border_lo + self.border_per_cell

        # Group mapping
        cell_to_group = torch.arange(self.N_cells) % config.mlp_groups
        self.register_buffer("cell_to_group", cell_to_group)

        # Border exchange neighbor lookup
        self.register_buffer(
            "border_neighbor_cell",
            self._build_border_neighbor_cells(),
            persistent=False,
        )
        # Per-neuron identity
        self.neuron_id = nn.Parameter(
            torch.randn(self.N_cells, self.C_n, self.D_n) * 0.02)

        G = config.mlp_groups
        Hs = config.state_mlp_hidden
        Hm = config.msg_mlp_hidden
        Hmod = config.cell_mod_hidden
        N = self.C_n

        # Shared state MLP: input = cat(received, h) = 2*D_n
        self.state_w1 = nn.Parameter(torch.empty(Hs, config.state_in))
        self.state_b1 = nn.Parameter(torch.zeros(Hs))
        self.state_w2 = nn.Parameter(torch.empty(self.D_n, Hs))
        self.state_b2 = nn.Parameter(torch.zeros(self.D_n))

        # Shared message MLP: input = h = D_n
        self.msg_w1 = nn.Parameter(torch.empty(Hm, config.msg_in))
        self.msg_b1 = nn.Parameter(torch.zeros(Hm))
        self.msg_w2 = nn.Parameter(torch.empty(self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(self.D_n))

        # Inject projection (per-group)
        self.inject_w = nn.Parameter(
            torch.empty(G, self.alpha * self.D_n, self.D_n))
        self.inject_b = nn.Parameter(torch.zeros(G, self.alpha * self.D_n))

        # Per-cell modulator
        # Input: h_mean + msg_mean + ctx + W_row_norms + decay_mean
        self.mod_w1 = nn.Parameter(torch.empty(self.N_cells, config.mod_in, Hmod))
        self.mod_b1 = nn.Parameter(torch.zeros(self.N_cells, Hmod))
        self.mod_w2 = nn.Parameter(torch.empty(self.N_cells, Hmod, config.mod_out))
        self.mod_b2 = nn.Parameter(torch.zeros(self.N_cells, config.mod_out))

        # Init weights
        for weight in (
            self.state_w1, self.state_w2, self.msg_w1, self.msg_w2,
            self.mod_w1, self.mod_w2,
        ):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        with torch.no_grad():
            self.inject_b.zero_()
            for group in range(G):
                for port in range(self.alpha):
                    block = torch.empty(self.D_n, self.D_n)
                    nn.init.orthogonal_(block)
                    start = port * self.D_n
                    self.inject_w[group, start:start + self.D_n].copy_(block)
            self.state_w2.mul_(0.1)
            self.msg_w2.mul_(0.1)
            self.mod_w2.mul_(0.01)

        self._initialized = False
        self._compiled = False

    def _build_border_neighbor_cells(self) -> Tensor:
        neighbors = torch.full(
            (self.N_cells, self.border_per_cell), -1, dtype=torch.long)
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                cell = row * self.grid_w + col
                if row > 0:
                    neighbors[cell, 0] = (row - 1) * self.grid_w + col
                if row + 1 < self.grid_h:
                    neighbors[cell, 1] = (row + 1) * self.grid_w + col
                if col > 0:
                    neighbors[cell, 2] = row * self.grid_w + (col - 1)
                if col + 1 < self.grid_w:
                    neighbors[cell, 3] = row * self.grid_w + (col + 1)
        return neighbors

    # ================================================================
    # State management
    # ================================================================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _identity(self, BS, dtype, device):
        return self.neuron_id.to(device=device, dtype=dtype).unsqueeze(0).expand(BS, -1, -1, -1)

    def initialize_states(self, BS: int, device: torch.device):
        dt = torch.bfloat16
        NC, N, D_n = self.N_cells, self.C_n, self.D_n
        K_init = self.config.K  # initial sparse connections

        self.h = torch.randn(BS, NC, N, D_n, device=device, dtype=dt) * 0.01
        self.msg = torch.zeros(BS, NC, N, D_n, device=device, dtype=dt)
        self.decay_logit = torch.zeros(BS, NC, N, device=device, dtype=dt)
        self.cell_context = torch.zeros(BS, NC, D_n, device=device, dtype=dt)
        self.border_gate_logit = torch.zeros(
            BS, NC, self.border_per_cell, device=device, dtype=dt)

        # Initialize W as sparse: K random nonzero connections per neuron
        W = torch.zeros(BS, NC, N, N, device=device, dtype=dt)
        for cell in range(NC):
            for neuron in range(N):
                neighbors = torch.randperm(N, device=device)[:K_init]
                W[:, cell, neuron, neighbors] = 0.1
            # Zero out self-connections
            W[:, cell].diagonal(dim1=-2, dim2=-1).zero_()
        self.W = W

        # Surprise state (scalar per batch — memory-head prediction signal).
        # s_mem(t) is the NEGATIVE target-logit under the memory head (lower = less
        # surprised). s_mem_live is what the modulator reads inside the loop; the
        # fast/slow EMAs track it and their difference is learning progress.
        self.s_mem_live = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_ema_fast = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_ema_slow = torch.zeros(BS, device=device, dtype=dt)
        # Per-cell previous readout — for computing readout_drift (local surprise).
        self.prev_readout_cell = torch.zeros(BS, NC, D_n, device=device, dtype=dt)
        # Full D-dim previous readout — fed to the memory head for predicting
        # the next token (memory head uses readout[t-1] to predict x_t).
        self.prev_readout = torch.zeros(BS, self.config.D, device=device, dtype=dt)

        self._initialized = True

    def detach_states(self):
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.msg = self.msg.detach()
        self.W = self.W.detach()
        self.decay_logit = self.decay_logit.detach()
        self.cell_context = self.cell_context.detach()
        self.border_gate_logit = self.border_gate_logit.detach()
        self.prev_readout = self.prev_readout.detach()
        self.prev_readout_cell = self.prev_readout_cell.detach()
        # s_mem_* are already detached (no_grad EMAs)

    def reset_states(self, mask: Tensor):
        """Retained for API compatibility; memory is intended to be lifelong."""
        pass

    def runtime_state_dict(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized": True,
            "h": self.h.clone(),
            "msg": self.msg.clone(),
            "W": self.W.clone(),
            "decay_logit": self.decay_logit.clone(),
            "cell_context": self.cell_context.clone(),
            "border_gate_logit": self.border_gate_logit.clone(),
            "s_mem_live": self.s_mem_live.clone(),
            "s_mem_ema_fast": self.s_mem_ema_fast.clone(),
            "s_mem_ema_slow": self.s_mem_ema_slow.clone(),
            "prev_readout": self.prev_readout.clone(),
            "prev_readout_cell": self.prev_readout_cell.clone(),
        }

    def load_runtime_state(self, state: dict):
        if not state or not state.get("initialized", False):
            self._initialized = False
            return
        device = self.neuron_id.device
        self.h = state["h"].to(device)
        self.msg = state["msg"].to(device)
        self.W = state["W"].to(device)
        self.decay_logit = state["decay_logit"].to(device)
        self.cell_context = state["cell_context"].to(device)
        self.border_gate_logit = state["border_gate_logit"].to(device)
        BS = self.h.shape[0]
        dt = self.h.dtype
        zero_b = torch.zeros(BS, device=device, dtype=dt)
        self.s_mem_live = state.get("s_mem_live", zero_b).to(device)
        self.s_mem_ema_fast = state.get("s_mem_ema_fast", zero_b).to(device)
        self.s_mem_ema_slow = state.get("s_mem_ema_slow", zero_b).to(device)
        self.prev_readout = state.get("prev_readout", torch.zeros(
            BS, self.config.D, device=device, dtype=dt)).to(device)
        self.prev_readout_cell = state.get("prev_readout_cell", torch.zeros(
            BS, self.N_cells, self.D_n, device=device, dtype=dt)).to(device)
        self._initialized = True

    # ================================================================
    # Per-step components
    # ================================================================

    def _receive(self, msg: Tensor, W: Tensor) -> Tensor:
        """Dense matmul message passing: received = W @ msg."""
        return torch.matmul(W, msg)  # [BS, NC, N, D_n]

    def _inject(self, received, H_aug_t, inject_w, inject_b):
        cell_slice = H_aug_t.reshape(H_aug_t.shape[0], self.N_cells, self.D_n)
        inject = torch.einsum("bni,noi->bno", cell_slice, inject_w)
        inject = inject + inject_b.unsqueeze(0)
        inject = inject.reshape(
            H_aug_t.shape[0], self.N_cells, self.alpha, self.D_n)
        received[:, :, self.input_lo:self.input_hi].add_(inject.to(received.dtype))
        return received

    def _border_exchange(self, msg, border_gate):
        """Fixed geometric border exchange using grid reshape + slice shifts."""
        BS = msg.shape[0]
        border = msg[:, :, self.border_lo:self.border_hi]  # [BS, NC, 4, D_n]

        # Reshape to grid: [BS, H, W, 4, D_n]
        border = border.reshape(BS, self.grid_h, self.grid_w, self.border_per_cell, self.D_n)
        incoming = torch.zeros_like(border)

        # 0=north reads south(1) from cell above, 1=south reads north(0) from cell below
        # 2=west reads east(3) from cell left, 3=east reads west(2) from cell right
        if self.grid_h > 1:
            incoming[:, 1:, :, 0] = border[:, :-1, :, 1]   # north
            incoming[:, :-1, :, 1] = border[:, 1:, :, 0]    # south
        if self.grid_w > 1:
            incoming[:, :, 1:, 2] = border[:, :, :-1, 3]    # west
            incoming[:, :, :-1, 3] = border[:, :, 1:, 2]     # east

        incoming = incoming.reshape(BS, self.N_cells, self.border_per_cell, self.D_n)
        return border_gate * incoming

    def _mlp2(self, x, w1, b1, w2, b2):
        """Two-layer MLP: F.linear → tanh → F.linear → tanh."""
        flat = x.reshape(-1, x.shape[-1])
        hidden = torch.tanh(F.linear(flat, w1, b1))
        out = torch.tanh(F.linear(hidden, w2, b2))
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    def _state_update(self, received, h, decay, one_minus_decay, identity,
                      w1_recv, w1_h, b1, w2, b2):
        """State MLP with split first layer — avoids cat([received, h])."""
        flat_recv = received.reshape(-1, received.shape[-1])
        flat_h = h.reshape(-1, h.shape[-1])
        hidden = torch.tanh(
            F.linear(flat_recv, w1_recv) + F.linear(flat_h, w1_h, b1))
        candidate = torch.tanh(F.linear(hidden, w2, b2))
        candidate = candidate.reshape(h.shape)
        return decay * h + one_minus_decay * candidate

    def _emit_message(self, h, identity, w1, b1, w2, b2):
        msg_new = self._mlp2(h, w1, b1, w2, b2)
        return msg_new + identity

    def _readout(self, msg):
        out_ports = msg[:, :, self.output_lo:self.output_hi]
        readout = out_ports.sum(dim=2) * (self.alpha ** -0.5)
        return readout.reshape(msg.shape[0], -1)

    def _modulate_cells(self, h, msg, W, decay_logit, cell_context,
                        border_gate_logit,
                        readout_drift, s_mem_live, s_mem_ema_fast, s_progress,
                        mod_w1, mod_b1, mod_w2, mod_b2):
        """Per-cell neuromodulator step.

        Surprise inputs:
          - readout_drift   : [BS, NC, 1] — per-cell local state churn
          - s_mem_live      : [BS]        — current memory-head negative target-logit
          - s_mem_ema_fast  : [BS]        — fast EMA of s_mem_live
          - s_progress      : [BS]        — slow - fast (positive = improving)
        The three scalars are broadcast to all cells.
        """
        NC, N = self.N_cells, self.C_n
        D_n, B = self.D_n, self.border_per_cell
        BS = h.shape[0]

        h_mean = h.mean(dim=2)
        msg_mean = msg.mean(dim=2)
        W_stats = W.abs().mean(dim=-1).mean(dim=2, keepdim=True)  # [BS, NC, 1]
        decay_mean = decay_logit.mean(dim=2, keepdim=True)  # [BS, NC, 1]

        # Broadcast global scalars to every cell.
        s1 = s_mem_live.view(BS, 1, 1).expand(BS, NC, 1).to(h_mean.dtype)
        s2 = s_mem_ema_fast.view(BS, 1, 1).expand(BS, NC, 1).to(h_mean.dtype)
        s3 = s_progress.view(BS, 1, 1).expand(BS, NC, 1).to(h_mean.dtype)

        mod_input = torch.cat([
            h_mean, msg_mean, cell_context,    # 3*D_n
            W_stats, decay_mean,               # 2
            readout_drift,                      # 1
            s1, s2, s3,                         # 3
        ], dim=-1)
        hidden = torch.tanh(
            torch.einsum("bni,nih->bnh", mod_input, mod_w1) + mod_b1.unsqueeze(0))
        output = torch.einsum("bnh,nho->bno", hidden, mod_w2) + mod_b2.unsqueeze(0)

        # Unpack: direct delta_W, delta_decay, delta_ctx, delta_border
        idx = 0
        delta_W = output[..., idx:idx + N * N].reshape(BS, NC, N, N)
        idx += N * N
        delta_decay = output[..., idx:idx + N].reshape(BS, NC, N)
        idx += N
        delta_ctx = output[..., idx:idx + D_n]
        idx += D_n
        delta_border = output[..., idx:idx + B]

        return (
            W + delta_W,
            decay_logit + delta_decay,
            cell_context + delta_ctx,
            border_gate_logit + delta_border,
        )

    def _step(self, h, msg, W, decay, one_minus_decay, border_gate,
              H_aug_t, identity, inject_w, inject_b,
              st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
              mg_w1, mg_b1, mg_w2, mg_b2):
        received = self._receive(msg, W)
        received = self._inject(received, H_aug_t, inject_w, inject_b)
        received[:, :, self.border_lo:self.border_hi] += self._border_exchange(
            msg, border_gate)

        h = self._state_update(received, h, decay, one_minus_decay, identity,
                               st_w1_recv, st_w1_h, st_b1, st_w2, st_b2)
        msg = self._emit_message(h, identity, mg_w1, mg_b1, mg_w2, mg_b2)
        readout = self._readout(msg)
        return h, msg, readout

    def _run_block(self, h, msg, W, decay_logit, cell_context,
                   border_gate_logit,
                   s_mem_live, s_mem_ema_fast, s_mem_ema_slow,
                   prev_readout_cell, prev_readout_full,
                   block_H_mid, block_input_ids,
                   start_t, identity, inject_w, inject_b,
                   st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   mod_w1, mod_b1, mod_w2, mod_b2,
                   lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                   w_decay_rate, gain_fast, gain_slow):
        """Run one block of memory steps with live in-block surprise signals.

        Live surprise: at each step t, use prev_readout_full (readout at t-1) to
        predict input_ids[t] via the weight-tied memory head. The negative
        target logit becomes s_mem_live(t); EMAs update per-token. The modulator
        at the next modulation step sees the fresh signal.
        """
        block_T = block_H_mid.shape[1]
        BS = h.shape[0]
        D = self.config.D
        NC, D_n = self.N_cells, self.D_n
        readouts = torch.empty(BS, block_T, D, device=block_H_mid.device, dtype=h.dtype)

        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        one_minus_decay = 1.0 - decay
        border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)
        readout_drift = torch.zeros(BS, NC, 1, device=h.device, dtype=h.dtype)

        for offset in range(block_T):
            t = start_t + offset
            H_mid_t = block_H_mid[:, offset]  # [BS, D]
            tok_t = block_input_ids[:, offset]  # [BS]

            # TBPTT detach
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                W = W.detach()
                decay_logit = decay_logit.detach()
                cell_context = cell_context.detach()
                border_gate_logit = border_gate_logit.detach()
                prev_readout_full = prev_readout_full.detach()
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            # --- Live memory-head surprise (no_grad — detached signal) ---
            with torch.no_grad():
                x = prev_readout_full
                if proj_down_w is not None:
                    x = F.linear(x, proj_down_w, proj_down_b)
                x = F.layer_norm(x, (x.shape[-1],), ln_final_w, ln_final_b)
                target_emb = lm_head_w[tok_t]  # [BS, D_embed]
                target_logit = (x * target_emb).sum(dim=-1)  # [BS]
                s_mem_live = (-target_logit).to(h.dtype)
                s_mem_ema_fast = (1 - gain_fast) * s_mem_ema_fast + gain_fast * s_mem_live
                s_mem_ema_slow = (1 - gain_slow) * s_mem_ema_slow + gain_slow * s_mem_live

            # --- Modulate every M tokens using the FRESH surprise ---
            if t % self.config.modulation_interval == 0:
                s_progress = s_mem_ema_slow - s_mem_ema_fast
                W, decay_logit, cell_context, border_gate_logit = \
                    self._modulate_cells(
                        h, msg, W, decay_logit, cell_context,
                        border_gate_logit,
                        readout_drift, s_mem_live, s_mem_ema_fast, s_progress,
                        mod_w1, mod_b1, mod_w2, mod_b2)
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            # --- Memory step ---
            h, msg, readout = self._step(
                h, msg, W, decay, one_minus_decay, border_gate,
                H_mid_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2)

            readouts[:, offset] = readout

            # Update per-cell drift (for next modulation).
            with torch.no_grad():
                new_cell = readout.reshape(BS, NC, D_n)
                readout_drift = (new_cell - prev_readout_cell).abs().mean(
                    dim=-1, keepdim=True).to(h.dtype)
                prev_readout_cell = new_cell.detach()

            # Advance prev_readout_full for next-token memory-head prediction.
            prev_readout_full = readout.detach()

            # Soft W decay (on-graph so later-token loss can train modulator).
            W = W * (1.0 - w_decay_rate)

        return (h, msg, W, decay_logit, cell_context, border_gate_logit,
                s_mem_live, s_mem_ema_fast, s_mem_ema_slow,
                prev_readout_cell, prev_readout_full, readouts)

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_mid: Tensor, input_ids: Tensor, lm):
        """Process T tokens through the memory graph.

        Args:
            H_mid: [BS, T, D] — lower-scan output (detached from LM graph)
            input_ids: [BS, T] — input tokens; used to compute the per-token
                memory-head target-logit surprise signal, where the memory
                head uses readout[t-1] to predict the token at position t.
            lm: LM module — provides mem_head_target_logit and mem_head_logits
                (weight-tied to the main lm_head).

        Returns:
            readouts:      [BS, T, D] — memory readout per token
            mem_pred_loss: scalar      — CE of memory head against input_ids
                                          (used as auxiliary training loss)
        """
        BS, T, _ = H_mid.shape
        if not self._initialized:
            self.initialize_states(BS, H_mid.device)
        if not self._compiled and H_mid.is_cuda:
            self._run_block = torch.compile(
                self._run_block, mode="default", fullgraph=False)
            self._compiled = True

        h = self.h
        msg = self.msg
        W = self.W
        decay_logit = self.decay_logit
        cell_context = self.cell_context
        border_gate_logit = self.border_gate_logit
        s_mem_live = self.s_mem_live
        s_mem_ema_fast = self.s_mem_ema_fast
        s_mem_ema_slow = self.s_mem_ema_slow
        prev_readout_cell = self.prev_readout_cell
        prev_readout_full = self.prev_readout  # full-D, for memory head at block start
        segment_start_prev_readout = prev_readout_full.detach().clone()
        H_mid = H_mid.to(h.dtype)

        identity = self._identity(BS, h.dtype, h.device)

        group_idx = self.cell_to_group
        dt = h.dtype
        st_w1_full = self.state_w1.to(dt)
        st_w1_recv = st_w1_full[:, :self.D_n].contiguous()
        st_w1_h = st_w1_full[:, self.D_n:].contiguous()
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)
        inject_w = self.inject_w[group_idx].to(dt)
        inject_b = self.inject_b[group_idx].to(dt)
        mod_w1 = self.mod_w1.to(dt)
        mod_b1 = self.mod_b1.to(dt)
        mod_w2 = self.mod_w2.to(dt)
        mod_b2 = self.mod_b2.to(dt)
        # LM head weights for the in-block memory head (bf16 compute copies)
        lm_head_w = lm.lm_head.weight.to(dt)
        if lm.proj_down is not None:
            proj_down_w = lm.proj_down.weight.to(dt)
            proj_down_b = lm.proj_down.bias.to(dt)
        else:
            proj_down_w = None
            proj_down_b = None
        ln_final_w = lm.ln_final.weight.to(dt)
        ln_final_b = lm.ln_final.bias.to(dt)

        readouts = torch.empty(BS, T, self.config.D, device=H_mid.device, dtype=dt)
        block_size = max(1, self.config.checkpoint_every)
        w_decay_rate = self.config.w_decay_rate
        gain_fast = self.config.gain_ema_fast
        gain_slow = self.config.gain_ema_slow

        use_ckpt = self.training and self.config.checkpoint_memory
        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_mid = H_mid[:, start_t:end_t]
            block_input_ids = input_ids[:, start_t:end_t]

            block_args = (
                h, msg, W, decay_logit, cell_context, border_gate_logit,
                s_mem_live, s_mem_ema_fast, s_mem_ema_slow,
                prev_readout_cell, prev_readout_full,
                block_H_mid, block_input_ids,
                start_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                mod_w1, mod_b1, mod_w2, mod_b2,
                lm_head_w, proj_down_w, proj_down_b, ln_final_w, ln_final_b,
                w_decay_rate, gain_fast, gain_slow)
            if use_ckpt:
                result = torch.utils.checkpoint.checkpoint(
                    self._run_block, *block_args, use_reentrant=False)
            else:
                result = self._run_block(*block_args)

            (h, msg, W, decay_logit, cell_context, border_gate_logit,
             s_mem_live, s_mem_ema_fast, s_mem_ema_slow,
             prev_readout_cell, prev_readout_full, block_out) = result

            readouts[:, start_t:end_t] = block_out

        # Save state
        self.h = h
        self.msg = msg
        self.W = W
        self.decay_logit = decay_logit
        self.cell_context = cell_context
        self.border_gate_logit = border_gate_logit
        self.s_mem_live = s_mem_live.detach()
        self.s_mem_ema_fast = s_mem_ema_fast.detach()
        self.s_mem_ema_slow = s_mem_ema_slow.detach()
        self.prev_readout = prev_readout_full.detach()
        self.prev_readout_cell = prev_readout_cell.detach()

        # --- Segment-level mem_pred_loss, chunked for VRAM ---
        # Memory head uses readout[t-1] to predict token at position t.
        # Chunk the time axis to avoid materializing [BS, T, V] all at once.
        shifted_all = torch.cat([
            segment_start_prev_readout.unsqueeze(1).to(readouts.dtype),  # [BS, 1, D]
            readouts[:, :-1],                                             # [BS, T-1, D]
        ], dim=1)  # [BS, T, D]

        loss_sum = torch.zeros((), device=H_mid.device, dtype=torch.float32)
        count = 0
        chunk = block_size  # reuse the same chunk size (8)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_readout = shifted_all[:, s:e]
            sub_target = input_ids[:, s:e]
            sub_logits = lm.mem_head_logits(sub_readout)  # [BS, chunk, V]
            loss_sum = loss_sum + F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                sub_target.reshape(-1),
                reduction="sum",
            )
            count += sub_target.numel()
        mem_pred_loss = loss_sum / max(count, 1)

        return readouts, mem_pred_loss
