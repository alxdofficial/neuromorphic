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
        self.mod_rank = config.mod_rank

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
        self.register_buffer(
            "border_src_port",
            torch.tensor([1, 0, 3, 2], dtype=torch.long),
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
        self.state_gs1 = nn.Parameter(torch.ones(G, Hs))
        self.state_gb1 = nn.Parameter(torch.zeros(G, Hs))
        self.state_w2 = nn.Parameter(torch.empty(self.D_n, Hs))
        self.state_b2 = nn.Parameter(torch.zeros(self.D_n))
        self.state_gs2 = nn.Parameter(torch.ones(G, self.D_n))
        self.state_gb2 = nn.Parameter(torch.zeros(G, self.D_n))

        # Shared message MLP: input = h = D_n
        self.msg_w1 = nn.Parameter(torch.empty(Hm, config.msg_in))
        self.msg_b1 = nn.Parameter(torch.zeros(Hm))
        self.msg_gs1 = nn.Parameter(torch.ones(G, Hm))
        self.msg_gb1 = nn.Parameter(torch.zeros(G, Hm))
        self.msg_w2 = nn.Parameter(torch.empty(self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(self.D_n))
        self.msg_gs2 = nn.Parameter(torch.ones(G, self.D_n))
        self.msg_gb2 = nn.Parameter(torch.zeros(G, self.D_n))

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

    def _grouped_mlp(self, x, w1, b1, gs1, gb1, w2, b2, gs2, gb2):
        BS, NC, Cn, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])
        hidden = F.linear(flat, w1, b1)
        hidden = hidden.reshape(BS, NC, Cn, -1)
        hidden = hidden * gs1.unsqueeze(0).unsqueeze(2) + gb1.unsqueeze(0).unsqueeze(2)
        hidden = torch.tanh(hidden)
        flat2 = hidden.reshape(-1, hidden.shape[-1])
        out = F.linear(flat2, w2, b2)
        out = out.reshape(BS, NC, Cn, -1)
        out = out * gs2.unsqueeze(0).unsqueeze(2) + gb2.unsqueeze(0).unsqueeze(2)
        return torch.tanh(out)

    def _state_update(self, received, h, decay, identity,
                      w1, b1, gs1, gb1, w2, b2, gs2, gb2):
        state_input = torch.cat([received, h], dim=-1)
        candidate = self._grouped_mlp(state_input, w1, b1, gs1, gb1, w2, b2, gs2, gb2)
        return decay * h + (1.0 - decay) * candidate

    def _emit_message(self, h, identity, w1, b1, gs1, gb1, w2, b2, gs2, gb2):
        msg_new = self._grouped_mlp(h, w1, b1, gs1, gb1, w2, b2, gs2, gb2)
        return msg_new + identity

    def _readout(self, msg):
        out_ports = msg[:, :, self.output_lo:self.output_hi]
        readout = out_ports.sum(dim=2) * (self.alpha ** -0.5)
        return readout.reshape(msg.shape[0], -1)

    def _modulate_cells(self, h, msg, W, decay_logit, cell_context,
                        border_gate_logit, mod_w1, mod_b1, mod_w2, mod_b2):
        NC, N, r = self.N_cells, self.C_n, self.mod_rank
        D_n, B = self.D_n, self.border_per_cell

        h_mean = h.mean(dim=2)
        msg_mean = msg.mean(dim=2)
        W_row_norms = W.abs().mean(dim=-1).mean(dim=2)  # [BS, NC] → expand to [BS, NC, N] → mean → [BS, NC]
        # Actually: W is [BS, NC, N, N], mean over last dim → [BS, NC, N], mean over N → [BS, NC]
        # But we want per-cell stats. Let's use row-wise L1 mean: [BS, NC, N] → mean → [BS, NC, 1]
        W_stats = W.abs().mean(dim=-1).mean(dim=2, keepdim=True)  # [BS, NC, 1]
        decay_mean = decay_logit.mean(dim=2, keepdim=True)  # [BS, NC, 1]

        mod_input = torch.cat([h_mean, msg_mean, cell_context, W_stats, decay_mean], dim=-1)
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

    def _step(self, h, msg, W, decay, border_gate,
              H_aug_t, identity, inject_w, inject_b,
              st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
              mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2):
        received = self._receive(msg, W)
        received = self._inject(received, H_aug_t, inject_w, inject_b)
        received[:, :, self.border_lo:self.border_hi] += self._border_exchange(
            msg, border_gate)

        h = self._state_update(received, h, decay, identity,
                               st_w1, st_b1, st_gs1, st_gb1,
                               st_w2, st_b2, st_gs2, st_gb2)
        msg = self._emit_message(h, identity,
                                 mg_w1, mg_b1, mg_gs1, mg_gb1,
                                 mg_w2, mg_b2, mg_gs2, mg_gb2)
        readout = self._readout(msg)
        return h, msg, readout

    def _run_block(self, h, msg, W, decay_logit, cell_context,
                   border_gate_logit,
                   block_H_aug, start_t, identity, inject_w, inject_b,
                   st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
                   mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2,
                   mod_w1, mod_b1, mod_w2, mod_b2, w_decay_rate):
        block_T = block_H_aug.shape[1]
        readouts = torch.empty(
            h.shape[0], block_T, self.config.D,
            device=block_H_aug.device, dtype=h.dtype)

        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

        for offset in range(block_T):
            t = start_t + offset

            # TBPTT detach
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                W = W.detach()
                decay_logit = decay_logit.detach()
                cell_context = cell_context.detach()
                border_gate_logit = border_gate_logit.detach()
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            # Modulate
            if t % self.config.modulation_interval == 0:
                W, decay_logit, cell_context, border_gate_logit = \
                    self._modulate_cells(
                        h, msg, W, decay_logit, cell_context,
                        border_gate_logit,
                        mod_w1, mod_b1, mod_w2, mod_b2)
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            # Step
            h, msg, readout = self._step(
                h, msg, W, decay, border_gate,
                block_H_aug[:, offset], identity, inject_w, inject_b,
                st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
                mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2)

            readouts[:, offset] = readout

            # Soft W decay (outside autograd)
            with torch.no_grad():
                W = W * (1.0 - w_decay_rate)

        return h, msg, W, decay_logit, cell_context, border_gate_logit, readouts

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_aug: Tensor) -> Tensor:
        BS, T, _ = H_aug.shape
        if not self._initialized:
            self.initialize_states(BS, H_aug.device)

        h = self.h
        msg = self.msg
        W = self.W
        decay_logit = self.decay_logit
        cell_context = self.cell_context
        border_gate_logit = self.border_gate_logit
        H_aug = H_aug.to(h.dtype)

        identity = self._identity(BS, h.dtype, h.device)

        group_idx = self.cell_to_group
        dt = h.dtype
        st_w1 = self.state_w1.to(dt)
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)
        st_gs1 = self.state_gs1[group_idx].to(dt)
        st_gb1 = self.state_gb1[group_idx].to(dt)
        st_gs2 = self.state_gs2[group_idx].to(dt)
        st_gb2 = self.state_gb2[group_idx].to(dt)
        mg_gs1 = self.msg_gs1[group_idx].to(dt)
        mg_gb1 = self.msg_gb1[group_idx].to(dt)
        mg_gs2 = self.msg_gs2[group_idx].to(dt)
        mg_gb2 = self.msg_gb2[group_idx].to(dt)
        inject_w = self.inject_w[group_idx].to(dt)
        inject_b = self.inject_b[group_idx].to(dt)
        mod_w1 = self.mod_w1.to(dt)
        mod_b1 = self.mod_b1.to(dt)
        mod_w2 = self.mod_w2.to(dt)
        mod_b2 = self.mod_b2.to(dt)

        readouts = torch.empty(BS, T, self.config.D, device=H_aug.device, dtype=dt)
        block_size = max(1, self.config.checkpoint_every)
        w_decay_rate = self.config.w_decay_rate

        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_aug = H_aug[:, start_t:end_t]

            run_block = lambda h_, msg_, W_, dec_, ctx_, border_: self._run_block(
                h_, msg_, W_, dec_, ctx_, border_,
                block_H_aug, start_t, identity, inject_w, inject_b,
                st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
                mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2,
                mod_w1, mod_b1, mod_w2, mod_b2, w_decay_rate)

            if self.training and block_size > 1:
                h, msg, W, decay_logit, cell_context, border_gate_logit, block_out = (
                    torch.utils.checkpoint.checkpoint(
                        run_block,
                        h, msg, W, decay_logit, cell_context, border_gate_logit,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        determinism_check="none",
                    ))
            else:
                h, msg, W, decay_logit, cell_context, border_gate_logit, block_out = (
                    run_block(h, msg, W, decay_logit, cell_context, border_gate_logit))

            readouts[:, start_t:end_t] = block_out

        self.h = h
        self.msg = msg
        self.W = W
        self.decay_logit = decay_logit
        self.cell_context = cell_context
        self.border_gate_logit = border_gate_logit

        return readouts
