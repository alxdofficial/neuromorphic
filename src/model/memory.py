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
from .pcm import BatchedPCM


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
        self.mod_rank = config.mod_rank
        self.C = config.C
        self.D_cc = config.D_cc

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
        r = self.mod_rank

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

        # PCM (interleaved, per-token)
        if config.pcm_enabled:
            self.pcm = BatchedPCM(config.C, config.D_cc, hidden=config.pcm_hidden)
        else:
            self.pcm = None

        # Surprise projection: compress surprise_ema + readout_ema for modulator
        proj_dim = config.surprise_proj_dim
        self.surprise_proj_w = nn.Parameter(torch.empty(proj_dim, 2 * config.D_n))
        self.surprise_proj_b = nn.Parameter(torch.zeros(proj_dim))
        nn.init.kaiming_uniform_(self.surprise_proj_w, a=math.sqrt(5))

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

        # Surprise state (NEW)
        self.surprise_ema = torch.zeros(BS, NC, D_n, device=device, dtype=dt)
        self.readout_ema = torch.zeros(BS, NC, D_n, device=device, dtype=dt)
        self.prev_readout = torch.zeros(BS, self.config.D, device=device, dtype=dt)
        self.prev_delta_hat = None  # set on first PCM call

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
        # surprise_ema, readout_ema, prev_readout are already detached (EMA / no_grad)
        if self.prev_delta_hat is not None:
            self.prev_delta_hat = self.prev_delta_hat.detach()

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
            "surprise_ema": self.surprise_ema.clone(),
            "readout_ema": self.readout_ema.clone(),
            "prev_readout": self.prev_readout.clone(),
            "prev_delta_hat": (
                self.prev_delta_hat.clone() if self.prev_delta_hat is not None else None),
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
        self.surprise_ema = state.get("surprise_ema", torch.zeros_like(self.h[:, :, 0, :])).to(device)
        self.readout_ema = state.get("readout_ema", torch.zeros_like(self.h[:, :, 0, :])).to(device)
        self.prev_readout = state.get("prev_readout", torch.zeros(
            self.h.shape[0], self.config.D, device=device, dtype=self.h.dtype)).to(device)
        pdt = state.get("prev_delta_hat")
        self.prev_delta_hat = pdt.to(device) if pdt is not None else None
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
                        border_gate_logit, surprise_compressed,
                        mod_w1, mod_b1, mod_w2, mod_b2):
        NC, N, r = self.N_cells, self.C_n, self.mod_rank
        D_n, B = self.D_n, self.border_per_cell

        h_mean = h.mean(dim=2)
        msg_mean = msg.mean(dim=2)
        W_stats = W.abs().mean(dim=-1).mean(dim=2, keepdim=True)  # [BS, NC, 1]
        decay_mean = decay_logit.mean(dim=2, keepdim=True)  # [BS, NC, 1]

        mod_input = torch.cat([
            h_mean, msg_mean, cell_context, W_stats, decay_mean,
            surprise_compressed,  # [BS, NC, proj_dim]
        ], dim=-1)
        hidden = torch.tanh(
            torch.einsum("bni,nih->bnh", mod_input, mod_w1) + mod_b1.unsqueeze(0))
        output = torch.einsum("bnh,nho->bno", hidden, mod_w2) + mod_b2.unsqueeze(0)

        # Unpack: low-rank delta_W (u, v), delta_decay, delta_ctx, delta_border
        idx = 0
        u = output[..., idx:idx + N * r].reshape(-1, NC, N, r)
        idx += N * r
        v = output[..., idx:idx + N * r].reshape(-1, NC, N, r)
        idx += N * r
        delta_decay = output[..., idx:idx + N].reshape(-1, NC, N)
        idx += N
        delta_ctx = output[..., idx:idx + D_n]
        idx += D_n
        delta_border = output[..., idx:idx + B]

        # Low-rank W update
        delta_W = torch.matmul(u, v.transpose(-1, -2))  # [BS, NC, N, N]

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
                   border_gate_logit, surprise_ema, readout_ema,
                   prev_readout, prev_delta_hat, prev_H_mid_cols,
                   block_H_mid, augment_fn,
                   start_t, identity, inject_w, inject_b,
                   st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                   mg_w1, mg_b1, mg_w2, mg_b2,
                   mod_w1, mod_b1, mod_w2, mod_b2,
                   surprise_proj_w, surprise_proj_b,
                   w_decay_rate):
        block_T = block_H_mid.shape[1]
        BS = h.shape[0]
        D = self.config.D
        C, D_cc = self.C, self.D_cc
        NC, D_n = self.N_cells, self.D_n
        readouts = torch.empty(BS, block_T, D, device=block_H_mid.device, dtype=h.dtype)
        pcm_loss_accum = torch.tensor(0.0, device=block_H_mid.device)
        pcm_count = 0

        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        one_minus_decay = 1.0 - decay
        border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)
        ema_decay = self.config.surprise_ema_decay

        for offset in range(block_T):
            t = start_t + offset
            H_mid_t = block_H_mid[:, offset]  # [BS, D]

            # TBPTT detach
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                W = W.detach()
                decay_logit = decay_logit.detach()
                cell_context = cell_context.detach()
                border_gate_logit = border_gate_logit.detach()
                if prev_delta_hat is not None:
                    prev_delta_hat = prev_delta_hat.detach()
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            # --- PCM (interleaved) ---
            if self.pcm is not None:
                H_mid_cols = H_mid_t.reshape(BS, C, D_cc)
                prev_readout_cols = prev_readout.reshape(BS, C, D_cc)
                delta_hat = self.pcm.predict(H_mid_cols, prev_readout_cols)

                if prev_delta_hat is not None and prev_H_mid_cols is not None:
                    delta_actual = H_mid_cols - prev_H_mid_cols
                    # PCM loss: gradient flows to PCM params only
                    pcm_loss_accum = pcm_loss_accum + (
                        prev_delta_hat - delta_actual.detach()).pow(2).mean()
                    pcm_count += 1
                    # Surprise for augment: DETACHED from PCM graph
                    # (CE loss should not reach PCM through this path)
                    surprise_t = (prev_delta_hat - delta_actual).detach()
                else:
                    surprise_t = torch.zeros(BS, C, D_cc, device=h.device, dtype=h.dtype)

                prev_delta_hat = delta_hat
                prev_H_mid_cols = H_mid_cols.detach()

                # Accumulate surprise EMA (no grad)
                surprise_cell = surprise_t.reshape(BS, NC, D_n)
                with torch.no_grad():
                    surprise_ema = ema_decay * surprise_ema + (1 - ema_decay) * surprise_cell

                # Augment H_mid with surprise (surprise is detached)
                surprise_flat = surprise_t.reshape(BS, D)
                H_aug_t = augment_fn(H_mid_t, surprise_flat)
            else:
                H_aug_t = H_mid_t

            # --- Modulate (with surprise_compressed) ---
            if t % self.config.modulation_interval == 0:
                surp_input = torch.cat([
                    surprise_ema.to(surprise_proj_w.dtype),
                    readout_ema.to(surprise_proj_w.dtype),
                ], dim=-1)
                surprise_compressed = F.linear(
                    surp_input, surprise_proj_w, surprise_proj_b)
                W, decay_logit, cell_context, border_gate_logit = \
                    self._modulate_cells(
                        h, msg, W, decay_logit, cell_context,
                        border_gate_logit, surprise_compressed,
                        mod_w1, mod_b1, mod_w2, mod_b2)
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                one_minus_decay = 1.0 - decay
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            # --- Memory step ---
            h, msg, readout = self._step(
                h, msg, W, decay, one_minus_decay, border_gate,
                H_aug_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2)

            readouts[:, offset] = readout

            # Update readout EMA
            readout_cell = readout.reshape(BS, NC, D_n)
            with torch.no_grad():
                readout_ema = ema_decay * readout_ema + (1 - ema_decay) * readout_cell.detach()
            prev_readout = readout.detach()

            # Soft W decay
            with torch.no_grad():
                W = W * (1.0 - w_decay_rate)

        pcm_loss = pcm_loss_accum / max(pcm_count, 1)
        return (h, msg, W, decay_logit, cell_context, border_gate_logit,
                surprise_ema, readout_ema, prev_readout, prev_delta_hat,
                prev_H_mid_cols, readouts, pcm_loss)

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_mid: Tensor, augment_fn=None):
        """Process T tokens with interleaved PCM + memory.

        Args:
            H_mid: [BS, T, D] — lower scan output (detached from LM graph)
            augment_fn: callable(H_mid_t, surprise_t) → H_aug_t, or None

        Returns:
            readouts: [BS, T, D] — memory readout per token
            pcm_loss: scalar — PCM prediction loss (for aux_loss)
        """
        BS, T, _ = H_mid.shape
        if not self._initialized:
            self.initialize_states(BS, H_mid.device)
        if not self._compiled and H_mid.is_cuda:
            self._step = torch.compile(self._step, mode="default", fullgraph=False)
            self._compiled = True

        h = self.h
        msg = self.msg
        W = self.W
        decay_logit = self.decay_logit
        cell_context = self.cell_context
        border_gate_logit = self.border_gate_logit
        surprise_ema = self.surprise_ema
        readout_ema = self.readout_ema
        prev_readout = self.prev_readout
        prev_delta_hat = self.prev_delta_hat
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
        surprise_proj_w = self.surprise_proj_w.to(dt)
        surprise_proj_b = self.surprise_proj_b.to(dt)

        if augment_fn is None:
            augment_fn = lambda h_mid_t, surp_t: h_mid_t

        readouts = torch.empty(BS, T, self.config.D, device=H_mid.device, dtype=dt)
        block_size = max(1, self.config.checkpoint_every)
        w_decay_rate = self.config.w_decay_rate
        total_pcm_loss = torch.tensor(0.0, device=H_mid.device)
        prev_H_mid_cols = getattr(self, '_prev_H_mid_cols', None)
        n_blocks = 0

        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_mid = H_mid[:, start_t:end_t]

            result = self._run_block(
                h, msg, W, decay_logit, cell_context, border_gate_logit,
                surprise_ema, readout_ema, prev_readout, prev_delta_hat,
                prev_H_mid_cols,
                block_H_mid, augment_fn,
                start_t, identity, inject_w, inject_b,
                st_w1_recv, st_w1_h, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                mod_w1, mod_b1, mod_w2, mod_b2,
                surprise_proj_w, surprise_proj_b,
                w_decay_rate)

            (h, msg, W, decay_logit, cell_context, border_gate_logit,
             surprise_ema, readout_ema, prev_readout, prev_delta_hat,
             prev_H_mid_cols, block_out, pcm_loss) = result

            readouts[:, start_t:end_t] = block_out
            total_pcm_loss = total_pcm_loss + pcm_loss
            n_blocks += 1

        # Save state
        self.h = h
        self.msg = msg
        self.W = W
        self.decay_logit = decay_logit
        self.cell_context = cell_context
        self.border_gate_logit = border_gate_logit
        self.surprise_ema = surprise_ema
        self.readout_ema = readout_ema
        self.prev_readout = prev_readout
        self.prev_delta_hat = prev_delta_hat
        self._prev_H_mid_cols = prev_H_mid_cols

        avg_pcm_loss = total_pcm_loss / max(n_blocks, 1)
        return readouts, avg_pcm_loss
