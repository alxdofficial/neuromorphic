"""Cell-grid memory graph.

The memory graph is laid out as an 8x8 grid of cells. Each cell owns one
32-dim LM slice and contains a contiguous block of neurons with mostly
within-cell connectivity plus fixed directional border exchange.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import Config
from .triton_kernels import (
    border_exchange_activated,
    hebbian_ema_update,
    local_receive,
    local_receive_activated,
)


class MemoryGraph(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.D_n = config.D_n
        self.N_cells = config.N_cells
        self.C_n = config.neurons_per_cell
        self.K = config.K
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

        conn = torch.empty(self.N_cells, self.C_n, self.K, dtype=torch.long)
        for cell in range(self.N_cells):
            for neuron in range(self.C_n):
                scores = torch.rand(self.C_n)
                scores[neuron] = -float("inf")
                conn[cell, neuron] = scores.topk(self.K).indices.sort().values
        self.register_buffer("conn_idx", conn)
        conn_idx_i32, src_edge_offsets, src_edge_dst, src_edge_slot = self._build_src_edge_buffers(conn)
        self.register_buffer("conn_idx_i32", conn_idx_i32, persistent=False)
        self.register_buffer("src_edge_offsets", src_edge_offsets, persistent=False)
        self.register_buffer("src_edge_dst", src_edge_dst, persistent=False)
        self.register_buffer("src_edge_slot", src_edge_slot, persistent=False)

        cell_to_group = torch.arange(self.N_cells) % config.mlp_groups
        self.register_buffer("cell_to_group", cell_to_group)
        self.register_buffer("border_neighbor_cell", self._build_border_neighbor_cells(), persistent=False)
        self.register_buffer(
            "border_src_port",
            torch.tensor([1, 0, 3, 2], dtype=torch.int32),
            persistent=False,
        )

        self.neuron_id = nn.Parameter(torch.randn(self.N_cells, self.C_n, self.D_n) * 0.02)

        Hs = config.state_mlp_hidden
        Hm = config.msg_mlp_hidden
        Hmod = config.cell_mod_hidden

        self.state_w1 = nn.Parameter(torch.empty(config.mlp_groups, Hs, config.state_in))
        self.state_b1 = nn.Parameter(torch.zeros(config.mlp_groups, Hs))
        self.state_w2 = nn.Parameter(torch.empty(config.mlp_groups, self.D_n, Hs))
        self.state_b2 = nn.Parameter(torch.zeros(config.mlp_groups, self.D_n))

        self.msg_w1 = nn.Parameter(torch.empty(config.mlp_groups, Hm, config.msg_in))
        self.msg_b1 = nn.Parameter(torch.zeros(config.mlp_groups, Hm))
        self.msg_w2 = nn.Parameter(torch.empty(config.mlp_groups, self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(config.mlp_groups, self.D_n))

        self.inject_w = nn.Parameter(
            torch.empty(config.mlp_groups, self.alpha * self.D_n, self.D_n))
        self.inject_b = nn.Parameter(torch.zeros(config.mlp_groups, self.alpha * self.D_n))

        self.mod_w1 = nn.Parameter(torch.empty(self.N_cells, config.mod_in, Hmod))
        self.mod_b1 = nn.Parameter(torch.zeros(self.N_cells, Hmod))
        self.mod_w2 = nn.Parameter(torch.empty(self.N_cells, Hmod, config.mod_out))
        self.mod_b2 = nn.Parameter(torch.zeros(self.N_cells, config.mod_out))

        for weight in (
            self.state_w1, self.state_w2, self.msg_w1, self.msg_w2,
            self.mod_w1, self.mod_w2,
        ):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        with torch.no_grad():
            self.inject_b.zero_()
            for group in range(config.mlp_groups):
                for port in range(self.alpha):
                    block = torch.empty(self.D_n, self.D_n)
                    nn.init.orthogonal_(block)
                    start = port * self.D_n
                    end = start + self.D_n
                    self.inject_w[group, start:end].copy_(block)

        with torch.no_grad():
            self.state_w2.mul_(0.1)
            self.msg_w2.mul_(0.1)
            self.mod_w2.mul_(0.01)

        self._initialized = False

    def _rebuild_connectivity_buffers(self):
        conn_idx_i32, src_edge_offsets, src_edge_dst, src_edge_slot = self._build_src_edge_buffers(
            self.conn_idx)
        device = self.conn_idx.device
        self.conn_idx_i32 = conn_idx_i32.to(device)
        self.src_edge_offsets = src_edge_offsets.to(device)
        self.src_edge_dst = src_edge_dst.to(device)
        self.src_edge_slot = src_edge_slot.to(device)

    @staticmethod
    def _build_src_edge_buffers(conn_idx: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        conn_cpu = conn_idx.to(device="cpu", dtype=torch.long)
        nc, cn, k = conn_cpu.shape
        edge_count = cn * k
        offsets = torch.zeros(nc, cn + 1, dtype=torch.int32)
        dst = torch.empty(nc, edge_count, dtype=torch.int32)
        slot = torch.empty(nc, edge_count, dtype=torch.int32)

        for cell in range(nc):
            counts = torch.bincount(conn_cpu[cell].reshape(-1), minlength=cn).to(torch.int32)
            offsets[cell, 1:] = counts.cumsum(dim=0)
            cursor = offsets[cell, :-1].clone()
            for dst_idx in range(cn):
                for slot_idx in range(k):
                    src_idx = int(conn_cpu[cell, dst_idx, slot_idx].item())
                    edge_idx = int(cursor[src_idx].item())
                    dst[cell, edge_idx] = dst_idx
                    slot[cell, edge_idx] = slot_idx
                    cursor[src_idx] += 1

        return conn_cpu.to(dtype=torch.int32), offsets, dst, slot

    def _build_border_neighbor_cells(self) -> Tensor:
        neighbors = torch.full((self.N_cells, self.border_per_cell), -1, dtype=torch.int32)
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

    @property
    def identity(self) -> Tensor:
        if not self._initialized:
            raise RuntimeError("MemoryGraph identity requested before initialization")
        return self._identity(self.h.shape[0], self.h.dtype, self.h.device)

    def _identity(self, BS: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.neuron_id.to(device=device, dtype=dtype).unsqueeze(0).expand(BS, -1, -1, -1)

    def initialize_states(self, BS: int, device: torch.device):
        dt = torch.bfloat16
        shape = (BS, self.N_cells, self.C_n)

        self.h = torch.randn(*shape, self.D_n, device=device, dtype=dt) * 0.01
        self.msg = torch.zeros(*shape, self.D_n, device=device, dtype=dt)
        self.w_conn = torch.zeros(*shape, self.K, device=device, dtype=dt)
        self.decay_logit = torch.zeros(*shape, device=device, dtype=dt)
        self.cell_context = torch.zeros(BS, self.N_cells, self.D_n, device=device, dtype=dt)
        self.border_gate_logit = torch.zeros(
            BS, self.N_cells, self.border_per_cell, device=device, dtype=dt)
        self.hebbian_traces = torch.zeros(*shape, self.K, device=device, dtype=dt)
        self._initialized = True

    def detach_states(self):
        if not self._initialized:
            return
        self.h = self.h.detach()
        self.msg = self.msg.detach()
        self.w_conn = self.w_conn.detach()
        self.decay_logit = self.decay_logit.detach()
        self.cell_context = self.cell_context.detach()
        self.border_gate_logit = self.border_gate_logit.detach()
        self.hebbian_traces = self.hebbian_traces.detach()

    def runtime_state_dict(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        return {
            "initialized": True,
            "h": self.h.clone(),
            "msg": self.msg.clone(),
            "w_conn": self.w_conn.clone(),
            "decay_logit": self.decay_logit.clone(),
            "cell_context": self.cell_context.clone(),
            "border_gate_logit": self.border_gate_logit.clone(),
            "hebbian_traces": self.hebbian_traces.clone(),
        }

    def load_runtime_state(self, state: dict):
        if not state or not state.get("initialized", False):
            self._initialized = False
            return
        device = self.neuron_id.device
        self.h = state["h"].to(device)
        self.msg = state["msg"].to(device)
        self.w_conn = state["w_conn"].to(device)
        self.decay_logit = state["decay_logit"].to(device)
        self.cell_context = state["cell_context"].to(device)
        self.border_gate_logit = state["border_gate_logit"].to(device)
        self.hebbian_traces = state["hebbian_traces"].to(device)
        self._initialized = True

    def reset_states(self, mask: Tensor):
        """Retained for API compatibility; memory is intended to be lifelong."""
        if not self._initialized:
            return
        keep = (~mask).to(self.h.dtype)
        k4 = keep[:, None, None, None]
        k3 = keep[:, None, None]
        with torch.no_grad():
            self.h = self.h * k4
            self.msg = self.msg * k4
            self.w_conn = self.w_conn * k4
            self.decay_logit = self.decay_logit * k3
            self.cell_context = self.cell_context * keep[:, None, None]
            self.border_gate_logit = self.border_gate_logit * keep[:, None, None]
            self.hebbian_traces = self.hebbian_traces * k4

    # ================================================================
    # Per-token step components
    # ================================================================

    def _modulate_cells(
        self,
        h: Tensor,
        msg: Tensor,
        hebbian: Tensor,
        decay_logit: Tensor,
        cell_context: Tensor,
        border_gate_logit: Tensor,
        w_conn: Tensor,
        mod_w1: Tensor,
        mod_b1: Tensor,
        mod_w2: Tensor,
        mod_b2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h_mean = h.mean(dim=2)
        msg_mean = msg.mean(dim=2)
        hebb_mean = hebbian.mean(dim=2)
        decay_mean = decay_logit.mean(dim=2, keepdim=True)

        mod_input = torch.cat([h_mean, msg_mean, cell_context, hebb_mean, decay_mean], dim=-1)
        hidden = torch.tanh(torch.einsum("bni,nih->bnh", mod_input, mod_w1) + mod_b1.unsqueeze(0))
        output = torch.einsum("bnh,nho->bno", hidden, mod_w2) + mod_b2.unsqueeze(0)

        split0 = self.C_n * self.K
        split1 = split0 + self.C_n
        split2 = split1 + self.D_n

        dw = output[..., :split0].reshape(h.shape[0], self.N_cells, self.C_n, self.K)
        ddecay = output[..., split0:split1].reshape(h.shape[0], self.N_cells, self.C_n)
        dctx = output[..., split1:split2]
        dborder = output[..., split2:split2 + self.border_per_cell]

        return (
            w_conn + dw,
            decay_logit + ddecay,
            cell_context + dctx,
            border_gate_logit + dborder,
        )

    def _receive_local(self, msg: Tensor, w_conn: Tensor) -> Tensor:
        return local_receive(
            msg,
            w_conn,
            self.conn_idx,
            self.src_edge_offsets,
            self.src_edge_dst,
            self.src_edge_slot,
        )

    def _receive_local_activated(self, msg: Tensor, w_conn: Tensor) -> Tensor:
        return local_receive_activated(
            msg,
            w_conn,
            self.conn_idx,
            self.src_edge_offsets,
            self.src_edge_dst,
            self.src_edge_slot,
        )

    def _inject(self, received: Tensor, H_aug_t: Tensor, inject_w: Tensor, inject_b: Tensor) -> Tensor:
        cell_slice = H_aug_t.reshape(H_aug_t.shape[0], self.N_cells, self.D_n)
        inject = torch.einsum("bni,noi->bno", cell_slice, inject_w)
        inject = inject + inject_b.unsqueeze(0)
        inject = inject.reshape(H_aug_t.shape[0], self.N_cells, self.alpha, self.D_n)
        received[:, :, self.input_lo:self.input_hi].add_(inject.to(received.dtype))
        return received

    def _border_exchange(self, msg: Tensor, border_gate_logit: Tensor) -> Tensor:
        BS = msg.shape[0]
        border = msg[:, :, self.border_lo:self.border_hi]
        border = border.reshape(BS, self.grid_h, self.grid_w, self.border_per_cell, self.D_n)
        incoming = torch.zeros_like(border)

        # 0=north, 1=south, 2=west, 3=east
        incoming[:, 1:, :, 0] = border[:, :-1, :, 1]
        incoming[:, :-1, :, 1] = border[:, 1:, :, 0]
        incoming[:, :, 1:, 2] = border[:, :, :-1, 3]
        incoming[:, :, :-1, 3] = border[:, :, 1:, 2]

        incoming = incoming.reshape(BS, self.N_cells, self.border_per_cell, self.D_n)
        gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)
        return gate * incoming

    def _border_exchange_from_gate(self, msg: Tensor, border_gate: Tensor) -> Tensor:
        border = msg[:, :, self.border_lo:self.border_hi]
        return border_exchange_activated(
            border,
            border_gate,
            self.border_neighbor_cell,
            self.border_src_port,
        )

    def _state_update(
        self,
        received: Tensor,
        h: Tensor,
        decay_logit: Tensor,
        identity: Tensor,
        cell_context: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        ctx = cell_context.unsqueeze(2).expand(-1, -1, self.C_n, -1)
        state_input = torch.cat(
            [received, h, identity, ctx, decay_logit.unsqueeze(-1)], dim=-1)
        hidden = torch.tanh(
            torch.einsum("bnci,nhi->bnch", state_input, w1) + b1.unsqueeze(0).unsqueeze(2))
        candidate = torch.tanh(
            torch.einsum("bnch,nih->bnci", hidden, w2) + b2.unsqueeze(0).unsqueeze(2))
        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        return decay * h + (1.0 - decay) * candidate

    def _state_update_from_decay(
        self,
        received: Tensor,
        h: Tensor,
        decay: Tensor,
        identity: Tensor,
        cell_context: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        ctx = cell_context.unsqueeze(2).expand(-1, -1, self.C_n, -1)
        state_input = torch.cat([received, h, identity, ctx, decay], dim=-1)
        hidden = torch.tanh(
            torch.einsum("bnci,nhi->bnch", state_input, w1) + b1.unsqueeze(0).unsqueeze(2))
        candidate = torch.tanh(
            torch.einsum("bnch,nih->bnci", hidden, w2) + b2.unsqueeze(0).unsqueeze(2))
        return decay * h + (1.0 - decay) * candidate

    def _emit_message(
        self,
        h: Tensor,
        identity: Tensor,
        cell_context: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
    ) -> Tensor:
        ctx = cell_context.unsqueeze(2).expand(-1, -1, self.C_n, -1)
        msg_input = torch.cat([h, identity, ctx], dim=-1)
        hidden = torch.tanh(
            torch.einsum("bnci,nhi->bnch", msg_input, w1) + b1.unsqueeze(0).unsqueeze(2))
        msg_new = torch.tanh(
            torch.einsum("bnch,nih->bnci", hidden, w2) + b2.unsqueeze(0).unsqueeze(2))
        return msg_new + identity

    def _readout(self, msg: Tensor) -> Tensor:
        out_ports = msg[:, :, self.output_lo:self.output_hi]
        readout = out_ports.sum(dim=2) * (self.alpha ** -0.5)
        return readout.reshape(msg.shape[0], -1)

    @staticmethod
    def _hebbian_next(
        msg_prev: Tensor,
        msg_new: Tensor,
        hebbian: Tensor,
        conn_idx: Tensor,
        decay: float = 0.995,
    ) -> Tensor:
        return hebbian_ema_update(msg_prev, msg_new, hebbian, conn_idx, decay)

    @staticmethod
    def _update_hebbian(
        msg_prev: Tensor,
        msg_new: Tensor,
        hebbian: Tensor,
        conn_idx: Tensor,
        decay: float = 0.995,
    ):
        with torch.no_grad():
            hebbian.copy_(MemoryGraph._hebbian_next(msg_prev, msg_new, hebbian, conn_idx, decay))

    def _step(
        self,
        h: Tensor,
        msg: Tensor,
        w_conn: Tensor,
        decay_logit: Tensor,
        cell_context: Tensor,
        border_gate_logit: Tensor,
        hebbian: Tensor,
        w_conn_act: Tensor,
        decay: Tensor,
        border_gate: Tensor,
        H_aug_t: Tensor,
        identity: Tensor,
        inject_w: Tensor,
        inject_b: Tensor,
        st_w1: Tensor,
        st_b1: Tensor,
        st_w2: Tensor,
        st_b2: Tensor,
        mg_w1: Tensor,
        mg_b1: Tensor,
        mg_w2: Tensor,
        mg_b2: Tensor,
    ):
        received = self._receive_local_activated(msg, w_conn_act)
        received = self._inject(received, H_aug_t, inject_w, inject_b)
        received[:, :, self.border_lo:self.border_hi] += self._border_exchange_from_gate(msg, border_gate)

        h = self._state_update_from_decay(
            received, h, decay, identity, cell_context,
            st_w1, st_b1, st_w2, st_b2)
        msg = self._emit_message(
            h, identity, cell_context,
            mg_w1, mg_b1, mg_w2, mg_b2)
        readout = self._readout(msg)
        return h, msg, readout

    def _run_block(
        self,
        h: Tensor,
        msg: Tensor,
        w_conn: Tensor,
        decay_logit: Tensor,
        cell_context: Tensor,
        border_gate_logit: Tensor,
        hebbian: Tensor,
        block_H_aug: Tensor,
        start_t: int,
        identity: Tensor,
        inject_w: Tensor,
        inject_b: Tensor,
        st_w1: Tensor,
        st_b1: Tensor,
        st_w2: Tensor,
        st_b2: Tensor,
        mg_w1: Tensor,
        mg_b1: Tensor,
        mg_w2: Tensor,
        mg_b2: Tensor,
        mod_w1: Tensor,
        mod_b1: Tensor,
        mod_w2: Tensor,
        mod_b2: Tensor,
        ema_decay: float,
    ):
        block_T = block_H_aug.shape[1]
        readouts = torch.empty(
            h.shape[0], block_T, self.config.D, device=block_H_aug.device, dtype=h.dtype)
        w_conn_act = torch.sigmoid(w_conn)
        decay = torch.sigmoid(decay_logit).unsqueeze(-1)
        border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

        for offset in range(block_T):
            t = start_t + offset
            if t > 0 and (t % self.config.tbptt_block == 0):
                h = h.detach()
                msg = msg.detach()
                w_conn = w_conn.detach()
                decay_logit = decay_logit.detach()
                cell_context = cell_context.detach()
                border_gate_logit = border_gate_logit.detach()
                w_conn_act = torch.sigmoid(w_conn)
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            if t % self.config.modulation_interval == 0:
                w_conn, decay_logit, cell_context, border_gate_logit = self._modulate_cells(
                    h, msg, hebbian, decay_logit, cell_context, border_gate_logit, w_conn,
                    mod_w1, mod_b1, mod_w2, mod_b2,
                )
                w_conn_act = torch.sigmoid(w_conn)
                decay = torch.sigmoid(decay_logit).unsqueeze(-1)
                border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

            msg_prev = msg
            h, msg, readout = self._step(
                h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian,
                w_conn_act, decay, border_gate,
                block_H_aug[:, offset], identity,
                inject_w, inject_b,
                st_w1, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
            )
            hebbian = self._hebbian_next(msg_prev, msg, hebbian, self.conn_idx_i32, ema_decay)
            readouts[:, offset] = readout

        return h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian, readouts

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_aug: Tensor) -> Tensor:
        BS, T, _ = H_aug.shape
        if not self._initialized:
            self.initialize_states(BS, H_aug.device)

        h = self.h
        msg = self.msg
        w_conn = self.w_conn
        decay_logit = self.decay_logit
        cell_context = self.cell_context
        border_gate_logit = self.border_gate_logit
        hebbian = self.hebbian_traces
        H_aug = H_aug.to(h.dtype)

        identity = self._identity(BS, h.dtype, h.device)

        group_idx = self.cell_to_group
        st_w1 = self.state_w1[group_idx].to(h.dtype)
        st_b1 = self.state_b1[group_idx].to(h.dtype)
        st_w2 = self.state_w2[group_idx].to(h.dtype)
        st_b2 = self.state_b2[group_idx].to(h.dtype)
        mg_w1 = self.msg_w1[group_idx].to(h.dtype)
        mg_b1 = self.msg_b1[group_idx].to(h.dtype)
        mg_w2 = self.msg_w2[group_idx].to(h.dtype)
        mg_b2 = self.msg_b2[group_idx].to(h.dtype)
        inject_w = self.inject_w[group_idx].to(h.dtype)
        inject_b = self.inject_b[group_idx].to(h.dtype)
        mod_w1 = self.mod_w1.to(h.dtype)
        mod_b1 = self.mod_b1.to(h.dtype)
        mod_w2 = self.mod_w2.to(h.dtype)
        mod_b2 = self.mod_b2.to(h.dtype)

        readouts = torch.empty(BS, T, self.config.D, device=H_aug.device, dtype=h.dtype)
        ema_decay = self.config.hebbian_ema_decay
        block_size = max(1, self.config.checkpoint_every)

        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_aug = H_aug[:, start_t:end_t]

            run_block = lambda h_, msg_, w_conn_, decay_, ctx_, border_, hebb_: self._run_block(
                h_, msg_, w_conn_, decay_, ctx_, border_, hebb_,
                block_H_aug, start_t, identity,
                inject_w, inject_b,
                st_w1, st_b1, st_w2, st_b2,
                mg_w1, mg_b1, mg_w2, mg_b2,
                mod_w1, mod_b1, mod_w2, mod_b2,
                ema_decay,
            )

            if self.training and block_size > 1:
                h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian, block_out = (
                    torch.utils.checkpoint.checkpoint(
                        run_block,
                        h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        determinism_check="none",
                    )
                )
            else:
                h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian, block_out = (
                    run_block(h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian)
                )

            readouts[:, start_t:end_t] = block_out

        self.h = h
        self.msg = msg
        self.w_conn = w_conn
        self.decay_logit = decay_logit
        self.cell_context = cell_context
        self.border_gate_logit = border_gate_logit
        self.hebbian_traces = hebbian

        return readouts

    # ================================================================
    # Structural plasticity
    # ================================================================

    def rewire_connections(self):
        if not self.config.structural_plasticity or not self._initialized:
            return

        NC, Cn, K = self.N_cells, self.C_n, self.K
        n_swap = max(1, int(NC * Cn * K * self.config.plasticity_pct))
        explore_frac = self.config.plasticity_exploration_frac
        device = self.conn_idx.device

        with torch.no_grad():
            hebb = self.hebbian_traces.mean(dim=0)
            flat_hebb = hebb.reshape(-1)
            _, prune_flat = flat_hebb.topk(n_swap, largest=False)
            prune_cell = prune_flat // (Cn * K)
            rem = prune_flat % (Cn * K)
            prune_n = rem // K
            prune_k = rem % K
            rewired_mask = torch.zeros(NC, Cn, K, dtype=torch.bool, device=device)
            rewired_mask[prune_cell, prune_n, prune_k] = True

            exists = torch.zeros(NC, Cn, Cn, dtype=torch.bool, device=device)
            row_idx = torch.arange(Cn, device=device)[None, :, None].expand(NC, -1, K)
            cell_idx = torch.arange(NC, device=device)[:, None, None].expand_as(self.conn_idx)
            exists[cell_idx, row_idx, self.conn_idx] = True

            msg_mag = self.msg.detach().float().norm(dim=-1).mean(dim=0)
            n_exploit = n_swap - int(n_swap * explore_frac)
            new_targets = torch.empty(n_swap, dtype=torch.long, device=device)

            for i in range(n_swap):
                cell = prune_cell[i].item()
                neuron = prune_n[i].item()
                old_target = self.conn_idx[cell, neuron, prune_k[i]]
                exists[cell, neuron, old_target] = False

                if i < n_exploit:
                    scores = msg_mag[cell].clone()
                    scores[exists[cell, neuron]] = -float("inf")
                    scores[neuron] = -float("inf")
                    target = scores.argmax().item()
                else:
                    candidates = (~exists[cell, neuron]).nonzero(as_tuple=True)[0]
                    candidates = candidates[candidates != neuron]
                    if len(candidates) == 0:
                        target = old_target.item()
                    else:
                        target = candidates[torch.randint(len(candidates), (1,), device=device)].item()

                new_targets[i] = target
                exists[cell, neuron, target] = True

            self.conn_idx[prune_cell, prune_n, prune_k] = new_targets

            modified_cells = prune_cell.unique()
            for cell in modified_cells.tolist():
                for neuron in prune_n[prune_cell == cell].unique().tolist():
                    sorted_conn, order = self.conn_idx[cell, neuron].sort()
                    self.conn_idx[cell, neuron] = sorted_conn
                    self.w_conn[:, cell, neuron] = self.w_conn[:, cell, neuron][:, order]
                    self.hebbian_traces[:, cell, neuron] = self.hebbian_traces[:, cell, neuron][:, order]
                    row_rewired = rewired_mask[cell, neuron][order]
                    self.w_conn[:, cell, neuron, row_rewired] = 0
                    self.hebbian_traces[:, cell, neuron, row_rewired] = 0

            self._rebuild_connectivity_buffers()
