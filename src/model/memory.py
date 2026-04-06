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
        self.neuron_key = nn.Parameter(torch.randn(self.N_cells, self.C_n, self.D_n) * 0.02)
        self.attn_heads = 4  # D_n=32 / 4 = 8 per head

        G = config.mlp_groups
        Hs = config.state_mlp_hidden
        Hm = config.msg_mlp_hidden
        Hmod = config.cell_mod_hidden

        # Shared weights (one GEMM for all neurons) + per-group scale/bias
        self.state_w1 = nn.Parameter(torch.empty(Hs, config.state_in))
        self.state_b1 = nn.Parameter(torch.zeros(Hs))
        self.state_gs1 = nn.Parameter(torch.ones(G, Hs))   # group scale
        self.state_gb1 = nn.Parameter(torch.zeros(G, Hs))   # group bias
        self.state_w2 = nn.Parameter(torch.empty(self.D_n, Hs))
        self.state_b2 = nn.Parameter(torch.zeros(self.D_n))
        self.state_gs2 = nn.Parameter(torch.ones(G, self.D_n))
        self.state_gb2 = nn.Parameter(torch.zeros(G, self.D_n))

        self.msg_w1 = nn.Parameter(torch.empty(Hm, config.msg_in))
        self.msg_b1 = nn.Parameter(torch.zeros(Hm))
        self.msg_gs1 = nn.Parameter(torch.ones(G, Hm))
        self.msg_gb1 = nn.Parameter(torch.zeros(G, Hm))
        self.msg_w2 = nn.Parameter(torch.empty(self.D_n, Hm))
        self.msg_b2 = nn.Parameter(torch.zeros(self.D_n))
        self.msg_gs2 = nn.Parameter(torch.ones(G, self.D_n))
        self.msg_gb2 = nn.Parameter(torch.zeros(G, self.D_n))

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
            for group in range(G):
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
        self._compiled = False

    def _maybe_compile(self):
        """Compile _step with torch.compile if configured and not yet done."""
        if self._compiled or not self.config.compile_step:
            return
        if not torch.cuda.is_available():
            return
        self._step = torch.compile(self._step, mode="default", fullgraph=False)
        self._compiled = True

    def _rebuild_connectivity_buffers(self):
        device = self.conn_idx.device
        self.conn_idx_i32 = self.conn_idx.to(device=device, dtype=torch.int32)

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

    @staticmethod
    def _plan_rewire_cpu(
        conn_idx: Tensor,
        hebb: Tensor,
        msg_mag: Tensor,
        n_swap: int,
        n_exploit: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        conn_cpu = conn_idx.to(device="cpu", dtype=torch.long)
        hebb_cpu = hebb.to(device="cpu", dtype=torch.float32)
        msg_mag_cpu = msg_mag.to(device="cpu", dtype=torch.float32)

        nc, cn, k = conn_cpu.shape
        flat_hebb = hebb_cpu.reshape(-1)
        _, prune_flat = flat_hebb.topk(n_swap, largest=False)
        prune_cell = prune_flat // (cn * k)
        rem = prune_flat % (cn * k)
        prune_n = rem // k
        prune_k = rem % k

        rewired_mask = torch.zeros(nc, cn, k, dtype=torch.bool)
        rewired_mask[prune_cell, prune_n, prune_k] = True

        exists = torch.zeros(nc, cn, cn, dtype=torch.bool)
        row_idx = torch.arange(cn)[None, :, None].expand(nc, -1, k)
        cell_idx = torch.arange(nc)[:, None, None].expand_as(conn_cpu)
        exists[cell_idx, row_idx, conn_cpu] = True
        exists[prune_cell, prune_n, conn_cpu[prune_cell, prune_n, prune_k]] = False

        row_ids = prune_cell * cn + prune_n
        sort_idx = row_ids.argsort()
        row_ids_sorted = row_ids[sort_idx]
        prune_cell_sorted = prune_cell[sort_idx]
        prune_n_sorted = prune_n[sort_idx]
        prune_k_sorted = prune_k[sort_idx]
        is_exploit_sorted = (sort_idx < n_exploit)
        sorted_global = sort_idx

        new_targets_sorted = torch.empty(n_swap, dtype=torch.long)
        unique_rows, counts = row_ids_sorted.unique_consecutive(return_counts=True)
        offset = 0

        for count in counts.tolist():
            row_slice = slice(offset, offset + count)
            cell = int(prune_cell_sorted[row_slice.start].item())
            neuron = int(prune_n_sorted[row_slice.start].item())
            row_targets = new_targets_sorted[row_slice]

            allowed = ~exists[cell, neuron]
            allowed[neuron] = False
            exploit_mask = is_exploit_sorted[row_slice]
            n_row_exploit = int(exploit_mask.sum().item())

            if n_row_exploit > 0:
                scores = msg_mag_cpu[cell].masked_fill(~allowed, -float("inf"))
                exploit_targets = scores.topk(n_row_exploit).indices
                row_targets[exploit_mask] = exploit_targets
                allowed[exploit_targets] = False

            n_row_explore = count - n_row_exploit
            if n_row_explore > 0:
                candidates = allowed.nonzero(as_tuple=True)[0]
                if candidates.numel() < n_row_explore:
                    fallback = conn_cpu[cell, neuron, prune_k_sorted[row_slice][~exploit_mask]]
                    explore_targets = fallback
                elif n_row_explore == 1:
                    rand_idx = torch.randint(candidates.numel(), (1,))
                    explore_targets = candidates[rand_idx]
                else:
                    explore_targets = candidates[torch.randperm(candidates.numel())[:n_row_explore]]
                row_targets[~exploit_mask] = explore_targets

            offset += count

        new_targets = torch.empty_like(new_targets_sorted)
        new_targets[sorted_global] = new_targets_sorted
        conn_cpu[prune_cell, prune_n, prune_k] = new_targets

        modified_rows = unique_rows
        row_cell = modified_rows // cn
        row_neuron = modified_rows % cn
        sorted_conn, order = conn_cpu[row_cell, row_neuron].sort(dim=-1)
        conn_cpu[row_cell, row_neuron] = sorted_conn
        row_rewired = rewired_mask[row_cell, row_neuron].gather(-1, order)

        return conn_cpu, row_cell, row_neuron, order, row_rewired

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

    def _key(self, BS: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        return self.neuron_key.to(device=device, dtype=dtype).unsqueeze(0).expand(BS, -1, -1, -1)

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
        )

    def _receive_local_activated(self, msg: Tensor, w_conn: Tensor) -> Tensor:
        return local_receive_activated(
            msg,
            w_conn,
            self.conn_idx,
        )

    def _receive_attention(self, msg: Tensor, identity: Tensor, key: Tensor) -> Tensor:
        """Within-cell attention: identity=Q, key=K, msg=V.

        Uses F.scaled_dot_product_attention (Flash Attention) for fused fwd+bwd.
        Each cell is an independent 128-length sequence with 4 heads of dim 8.
        """
        BS, NC, Cn, Dn = msg.shape
        nh = self.attn_heads
        hd = Dn // nh  # head_dim

        # Reshape to [BS*NC, num_heads, Cn, head_dim]
        q = identity.reshape(BS * NC, Cn, nh, hd).permute(0, 2, 1, 3)
        k = key.reshape(BS * NC, Cn, nh, hd).permute(0, 2, 1, 3)
        v = msg.reshape(BS * NC, Cn, nh, hd).permute(0, 2, 1, 3)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.permute(0, 2, 1, 3).reshape(BS, NC, Cn, Dn)

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

    def _grouped_mlp(
        self,
        x: Tensor,
        w1: Tensor,
        b1: Tensor,
        gs1: Tensor,
        gb1: Tensor,
        w2: Tensor,
        b2: Tensor,
        gs2: Tensor,
        gb2: Tensor,
    ) -> Tensor:
        """Two-layer MLP: flat F.linear + per-group scale/bias conditioning.

        Args:
            x: [BS, NC, Cn, D_in]
            w1: [H, D_in], b1: [H] — shared first layer
            gs1: [NC, H], gb1: [NC, H] — per-cell group scale/bias (pre-indexed by cell_to_group)
            w2: [D_out, H], b2: [D_out] — shared second layer
            gs2: [NC, D_out], gb2: [NC, D_out]
        Returns:
            [BS, NC, Cn, D_out]
        """
        BS, NC, Cn, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])                     # [BS*NC*Cn, D_in]
        hidden = torch.nn.functional.linear(flat, w1, b1)      # [BS*NC*Cn, H]
        hidden = hidden.reshape(BS, NC, Cn, -1)                # [BS, NC, Cn, H]
        hidden = hidden * gs1.unsqueeze(0).unsqueeze(2) + gb1.unsqueeze(0).unsqueeze(2)
        hidden = torch.tanh(hidden)
        flat2 = hidden.reshape(-1, hidden.shape[-1])           # [BS*NC*Cn, H]
        out = torch.nn.functional.linear(flat2, w2, b2)        # [BS*NC*Cn, D_out]
        out = out.reshape(BS, NC, Cn, -1)                      # [BS, NC, Cn, D_out]
        out = out * gs2.unsqueeze(0).unsqueeze(2) + gb2.unsqueeze(0).unsqueeze(2)
        return torch.tanh(out)

    def _state_update_from_decay(
        self,
        received: Tensor,
        h: Tensor,
        decay: Tensor,
        identity: Tensor,
        w1, b1, gs1, gb1, w2, b2, gs2, gb2,
    ) -> Tensor:
        state_input = torch.cat([received, h], dim=-1)  # [BS, NC, Cn, 2*D_n]
        candidate = self._grouped_mlp(state_input, w1, b1, gs1, gb1, w2, b2, gs2, gb2)
        return decay * h + (1.0 - decay) * candidate

    def _emit_message(
        self,
        h: Tensor,
        identity: Tensor,
        w1, b1, gs1, gb1, w2, b2, gs2, gb2,
    ) -> Tensor:
        msg_new = self._grouped_mlp(h, w1, b1, gs1, gb1, w2, b2, gs2, gb2)
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
        h, msg, w_conn_act, decay, border_gate,
        H_aug_t, identity, inject_w, inject_b,
        st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
        mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2,
    ):
        received = self._receive_local_activated(msg, w_conn_act)
        received = self._inject(received, H_aug_t, inject_w, inject_b)
        received[:, :, self.border_lo:self.border_hi] += self._border_exchange_from_gate(msg, border_gate)

        h = self._state_update_from_decay(
            received, h, decay, identity,
            st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2)
        msg = self._emit_message(
            h, identity,
            mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2)
        readout = self._readout(msg)
        return h, msg, readout

    def _run_block(
        self,
        h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian,
        block_H_aug, start_t, identity, inject_w, inject_b,
        st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
        mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2,
        mod_w1, mod_b1, mod_w2, mod_b2,
        ema_decay,
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
                h, msg, w_conn_act, decay, border_gate,
                block_H_aug[:, offset], identity, inject_w, inject_b,
                st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
                mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2,
            )
            hebbian = self._hebbian_next(msg_prev, msg, hebbian, self.conn_idx, ema_decay)
            readouts[:, offset] = readout

        return h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian, readouts

    # ================================================================
    # Main forward
    # ================================================================

    def forward_segment(self, H_aug: Tensor) -> Tensor:
        BS, T, _ = H_aug.shape
        if not self._initialized:
            self.initialize_states(BS, H_aug.device)
        self._maybe_compile()

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
        dt = h.dtype
        # Shared weights (single GEMM)
        st_w1 = self.state_w1.to(dt)
        st_b1 = self.state_b1.to(dt)
        st_w2 = self.state_w2.to(dt)
        st_b2 = self.state_b2.to(dt)
        mg_w1 = self.msg_w1.to(dt)
        mg_b1 = self.msg_b1.to(dt)
        mg_w2 = self.msg_w2.to(dt)
        mg_b2 = self.msg_b2.to(dt)
        # Per-group scale/bias (pre-indexed by cell)
        st_gs1 = self.state_gs1[group_idx].to(dt)
        st_gb1 = self.state_gb1[group_idx].to(dt)
        st_gs2 = self.state_gs2[group_idx].to(dt)
        st_gb2 = self.state_gb2[group_idx].to(dt)
        mg_gs1 = self.msg_gs1[group_idx].to(dt)
        mg_gb1 = self.msg_gb1[group_idx].to(dt)
        mg_gs2 = self.msg_gs2[group_idx].to(dt)
        mg_gb2 = self.msg_gb2[group_idx].to(dt)
        # Inject (still per-group indexed)
        inject_w = self.inject_w[group_idx].to(dt)
        inject_b = self.inject_b[group_idx].to(dt)
        # Modulator (per-cell)
        mod_w1 = self.mod_w1.to(dt)
        mod_b1 = self.mod_b1.to(dt)
        mod_w2 = self.mod_w2.to(dt)
        mod_b2 = self.mod_b2.to(dt)

        readouts = torch.empty(BS, T, self.config.D, device=H_aug.device, dtype=h.dtype)
        ema_decay = self.config.hebbian_ema_decay
        block_size = max(1, self.config.checkpoint_every)

        for start_t in range(0, T, block_size):
            end_t = min(start_t + block_size, T)
            block_H_aug = H_aug[:, start_t:end_t]

            run_block = lambda h_, msg_, w_conn_, decay_, ctx_, border_, hebb_: self._run_block(
                h_, msg_, w_conn_, decay_, ctx_, border_, hebb_,
                block_H_aug, start_t, identity, inject_w, inject_b,
                st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2,
                mg_w1, mg_b1, mg_gs1, mg_gb1, mg_w2, mg_b2, mg_gs2, mg_gb2,
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
        n_exploit = n_swap - int(n_swap * explore_frac)
        device = self.conn_idx.device

        with torch.no_grad():
            hebb = self.hebbian_traces.mean(dim=0)
            msg_mag = self.msg.detach().float().norm(dim=-1).mean(dim=0)
            conn_cpu, row_cell, row_neuron, order_cpu, row_rewired_cpu = self._plan_rewire_cpu(
                self.conn_idx,
                hebb,
                msg_mag,
                n_swap,
                n_exploit,
            )

            row_cell = row_cell.to(device=device, dtype=torch.long)
            row_neuron = row_neuron.to(device=device, dtype=torch.long)
            order = order_cpu.to(device=device, dtype=torch.long)
            row_rewired = row_rewired_cpu.to(device=device, dtype=torch.bool)

            old_w = self.w_conn[:, row_cell, row_neuron]
            old_hebb = self.hebbian_traces[:, row_cell, row_neuron]
            gather_idx = order.unsqueeze(0).expand(old_w.shape[0], -1, -1)
            new_w = old_w.gather(-1, gather_idx)
            new_hebb = old_hebb.gather(-1, gather_idx)
            rewired_mask = row_rewired.unsqueeze(0).expand_as(new_w)
            new_w.masked_fill_(rewired_mask, 0)
            new_hebb.masked_fill_(rewired_mask, 0)

            self.conn_idx.copy_(conn_cpu.to(device=device, dtype=self.conn_idx.dtype))
            self.w_conn[:, row_cell, row_neuron] = new_w
            self.hebbian_traces[:, row_cell, row_neuron] = new_hebb

            self._rebuild_connectivity_buffers()
