"""ColumnGraphMemory — the memory substrate module.

See docs/column_graph.md for the full design.

Public interface:
    mem = ColumnGraphMemory(cfg)
    mem.begin_segment(B, device)     # reset persistent state
    for token_t in sequence:
        logits = mem.step(token_t)   # [B, K_horizons, V]
        loss = ... (from logits)
    # TBPTT: detach state at block boundary
    mem.detach_state()

The module owns: column state s, E_bias, pred_buf, surprise_ema,
input_ctx_ema, mag_ema, var_ema, traffic_ema, counters. These are
*buffers*, not parameters.

Trainable parameters: update_MLP, content_MLP, delta_MLP, score_MLP,
column identity table, decay_proj, cross-attn input/output projections,
tied token_emb (owned by the StandaloneLM wrapper, passed in at step()),
MultiHorizonReadout (horizon_emb + pred_head), neuromod trunk + heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.topology import Topology, build_topology
from src.column_graph.neuromod import Neuromod
from src.column_graph.plasticity import (
    compute_activities,
    hebbian_update_incoming,
    update_input_ctx_ema,
    update_tile_stats,
)
from src.column_graph.readout import (
    MultiHorizonReadout,
    init_prediction_buffer,
    multi_horizon_surprise,
    write_prediction_buffer,
)


def _rmsnorm(dim: int) -> nn.Module:
    from src.column_graph.readout import _FallbackRMSNorm
    return _FallbackRMSNorm(dim)


# =====================================================================
# Column-local MLPs
# =====================================================================


class ColumnMLPs(nn.Module):
    """update / content / delta / score — shared across all columns and
    all plasticity steps. All are standard GELU MLPs, batched over N or N·K.
    """

    def __init__(self, cfg: ColumnGraphConfig) -> None:
        super().__init__()
        D_s = cfg.D_s
        D_id = cfg.D_id

        # update_MLP: (D_s_norm + D_id + D_s_incoming) → 4·D_s → D_s
        self.update_norm_s = _rmsnorm(D_s)
        self.update_norm_in = _rmsnorm(D_s)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * D_s + D_id, cfg.ffn_mult_update * D_s),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult_update * D_s, D_s),
        )

        # content_MLP: (D_s_norm + D_id) → 2·D_s → D_s
        self.content_norm = _rmsnorm(D_s)
        self.content_mlp = nn.Sequential(
            nn.Linear(D_s + D_id, cfg.ffn_mult_content * D_s),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult_content * D_s, D_s),
        )

        # delta_MLP: (D_s_content + D_id_dest) → 1·D_s → D_s (residual, zero-init)
        self.delta_mlp = nn.Sequential(
            nn.Linear(D_s + D_id, cfg.ffn_mult_delta * D_s),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult_delta * D_s, D_s),
        )
        # Zero-init final delta so initial m_edge = m_out for every edge.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

        # score_MLP: (D_s_content + D_id_src + D_id_dst) → hidden → 1
        self.score_mlp = nn.Sequential(
            nn.Linear(D_s + 2 * D_id, cfg.score_hidden),
            nn.GELU(),
            nn.Linear(cfg.score_hidden, 1),
        )
        # Zero-init score so initial w_out = sigmoid(0 + 0) = 0.5 uniform.
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

        # Depth-scaled init on the first layer of each MLP: std = 1/sqrt(depth).
        # Depth here = 2 (2-layer MLPs). Using 0.02 * 1/sqrt(2) ≈ 0.014.
        for module in [self.update_mlp[0], self.content_mlp[0], self.delta_mlp[0],
                       self.score_mlp[0]]:
            nn.init.normal_(module.weight, mean=0.0, std=0.014)


# =====================================================================
# ColumnGraphMemory
# =====================================================================


@dataclass
class MemoryReadout:
    """Return bundle from step()."""
    logits: torch.Tensor                  # [B, K_horizons, V]
    surprise_ema: torch.Tensor            # [B, K_horizons]
    eta_global: torch.Tensor | None        # [B] scalar, only on mod_period steps


class ColumnGraphMemory(nn.Module):
    def __init__(self, cfg: ColumnGraphConfig, tied_token_emb: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.tied_token_emb = tied_token_emb   # stored as a reference, not copied

        # Topology (CPU buffers built once at init, moved to device on first step).
        topo = build_topology(
            plane_rows=cfg.plane_rows,
            plane_cols=cfg.plane_cols,
            L=cfg.L,
            K=cfg.K,
            p_rewire=cfg.p_rewire,
            K_intra_fraction=cfg.K_intra_fraction,
            num_tiles_per_plane_dim=cfg.num_tiles_per_plane_dim,
            seed=cfg.topology_seed,
        )
        self.register_buffer("out_nbrs", topo.out_nbrs, persistent=False)
        self.register_buffer("edge_src", topo.edge_src, persistent=False)
        self.register_buffer("edge_dst", topo.edge_dst, persistent=False)
        self.register_buffer("tile_ids", topo.tile_ids, persistent=False)
        self.register_buffer("plane_ids", topo.plane_ids, persistent=False)
        self.register_buffer("input_positions", topo.input_positions, persistent=False)
        self.register_buffer("output_positions", topo.output_positions, persistent=False)

        # Column identity table (learned)
        torch.manual_seed(cfg.init_seed)
        self.col_id = nn.Parameter(torch.randn(cfg.N, cfg.D_id) * 0.02)

        # Per-column learned decay α = sigmoid(decay_proj(id))
        # decay_proj: D_id → 1, bias initialized to logit(0.5) = 0
        self.decay_proj = nn.Linear(cfg.D_id, 1)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.decay_proj.bias)

        # Column compute MLPs
        self.mlps = ColumnMLPs(cfg)

        # Cross-attention: input injection (single multi-head pass)
        # We project the token embedding (= h_input of size D_s) to per-head
        # K and V, then each input-plane column computes a scalar gate via
        # dot(q, k) and adds gate·v to its state. Simple, fast.
        self.in_norm = _rmsnorm(cfg.D_s)
        self.in_q_proj = nn.Linear(cfg.D_s + cfg.D_id, cfg.D_s, bias=False)
        self.in_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.in_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        # Zero-init v_proj so inject is a no-op at day 0 (state stays bounded).
        nn.init.zeros_(self.in_v_proj.weight)

        # Cross-attention: output readout (single motor query over N_out columns)
        self.out_norm = _rmsnorm(cfg.D_s)
        self.motor_query = nn.Parameter(torch.randn(cfg.D_s) * 0.02)
        self.out_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.out_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)

        # Multi-horizon readout
        self.readout = MultiHorizonReadout(cfg)

        # Neuromod
        self.neuromod = Neuromod(cfg)

        # --- Persistent state (buffers, not parameters) ---
        # Allocated lazily in begin_segment() because they depend on B and device.
        self._state_initialized = False
        self.s: torch.Tensor                # [B, N, D_s]
        self.E_bias_flat: torch.Tensor       # [N*K] fp32
        self.pred_buf: torch.Tensor          # [B, K_buf, K_horizons, V]
        self.surprise_ema: torch.Tensor      # [B, K_horizons]
        self.surprise_prev: torch.Tensor     # [B, K_horizons]  for Δsurprise
        self.input_ctx_ema: torch.Tensor     # [B, D_s]
        self.mag_ema: torch.Tensor           # [B, num_tiles]
        self.var_ema: torch.Tensor           # [B, num_tiles]
        self.traffic_ema: torch.Tensor       # [B, num_tiles]
        self.pred_cursor: int = 0
        self.pred_filled: int = 0
        self.tick_counter: int = 0

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def begin_segment(self, B: int, device: torch.device) -> None:
        """Reset persistent state for a new segment. Call at segment start."""
        cfg = self.cfg
        dtype_fast = self._fast_dtype(device)

        # State init: h[c] = id_proj(id[c]) — warm start from identity, not zero.
        # But keep it simple: zero init in v1, let plasticity ramp up.
        self.s = torch.zeros(B, cfg.N, cfg.D_s, device=device, dtype=dtype_fast)

        self.E_bias_flat = torch.zeros(
            cfg.num_edges, device=device, dtype=torch.float32
        )
        self.pred_buf = init_prediction_buffer(
            B, cfg.K_buf, cfg.K_horizons, cfg.vocab_size, device, dtype_fast
        )
        self.surprise_ema = torch.zeros(
            B, cfg.K_horizons, device=device, dtype=torch.float32
        )
        self.surprise_prev = torch.zeros_like(self.surprise_ema)
        self.input_ctx_ema = torch.zeros(B, cfg.D_s, device=device, dtype=dtype_fast)
        self.mag_ema = torch.zeros(B, cfg.num_tiles, device=device, dtype=torch.float32)
        self.var_ema = torch.zeros(B, cfg.num_tiles, device=device, dtype=torch.float32)
        self.traffic_ema = torch.zeros(
            B, cfg.num_tiles, device=device, dtype=torch.float32
        )
        self.pred_cursor = 0
        self.pred_filled = 0
        self.tick_counter = 0
        self._state_initialized = True

    def detach_state(self) -> None:
        """TBPTT: detach all persistent state from the autograd graph."""
        if not self._state_initialized:
            return
        self.s = self.s.detach()
        self.E_bias_flat = self.E_bias_flat.detach()
        self.pred_buf = self.pred_buf.detach()
        self.surprise_ema = self.surprise_ema.detach()
        self.surprise_prev = self.surprise_prev.detach()
        self.input_ctx_ema = self.input_ctx_ema.detach()
        self.mag_ema = self.mag_ema.detach()
        self.var_ema = self.var_ema.detach()
        self.traffic_ema = self.traffic_ema.detach()

    def _fast_dtype(self, device: torch.device) -> torch.dtype:
        choice = self.cfg.state_dtype
        if choice == "bf16":
            return torch.bfloat16
        if choice == "fp32":
            return torch.float32
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    # -----------------------------------------------------------------
    # One token step
    # -----------------------------------------------------------------

    def _hot_forward(self, token_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compilable tensor-only hot path: inject → propagate → readout → logits.

        Returns (logits, m_out, w_out_flat, incoming). No Python integer
        state touched here. Callable from torch.compile without recompiles.
        """
        h_input = self.tied_token_emb(token_id)
        self._inject(h_input)
        m_out, w_out_flat, incoming = self._propagate()
        motor = self._readout_motor()
        logits = self.readout(motor, self.tied_token_emb.weight)
        return logits, m_out, w_out_flat, incoming

    def step(self, token_id: torch.Tensor) -> MemoryReadout:
        """One propagation round for one token. Returns logits + surprise."""
        assert self._state_initialized, "call begin_segment() first"
        cfg = self.cfg
        B = self.s.shape[0]

        logits, m_out, w_out_flat, incoming = self._hot_forward(token_id)

        # Python bookkeeping (ring buffer cursor + surprise EMA)
        self._surprise_and_buffer_bookkeeping(token_id, logits)
        h_input = self.tied_token_emb(token_id)

        # 8. Input-ctx EMA (fp32 under autocast-disabled)
        with torch.autocast(device_type=h_input.device.type, enabled=False):
            self.input_ctx_ema = update_input_ctx_ema(
                self.input_ctx_ema.float(),
                h_input.detach().float(),
                cfg.alpha_input_ctx,
            ).to(self.input_ctx_ema.dtype)

            # Update tile stats on every step (cheap EMA).
            self.mag_ema, self.var_ema = update_tile_stats(
                self.mag_ema,
                self.var_ema,
                self.s.detach(),
                self.tile_ids,
                cfg.num_tiles,
                cfg.alpha_tile_stats,
            )

            # Traffic EMA: sum of |w_out_flat| per tile (source side)
            traffic_per_col = self._sum_edge_abs_per_source(
                w_out_flat.detach().float()
            )                                                 # [B, N]
            counts = torch.bincount(self.tile_ids, minlength=cfg.num_tiles).float()
            sum_tile = torch.zeros(
                B, cfg.num_tiles, device=self.s.device, dtype=torch.float32
            )
            sum_tile.index_add_(1, self.tile_ids, traffic_per_col)
            mean_tile = sum_tile / counts.clamp(min=1)
            self.traffic_ema = (
                (1 - cfg.alpha_tile_stats) * self.traffic_ema
                + cfg.alpha_tile_stats * mean_tile
            )

        # 9. Plasticity step (every mod_period tokens)
        eta_global = None
        self.tick_counter += 1
        if self.tick_counter % cfg.mod_period == 0:
            eta_global = self._plasticity_step(m_out, incoming, w_out_flat)

        return MemoryReadout(
            logits=logits, surprise_ema=self.surprise_ema, eta_global=eta_global
        )

    # -----------------------------------------------------------------
    # Inject / propagate / readout
    # -----------------------------------------------------------------

    def _inject(self, h_input: torch.Tensor) -> None:
        """Per-column gated additive injection at input-plane columns."""
        B = h_input.shape[0]
        in_cols = self.input_positions                       # [N_in]
        s_in = self.s[:, in_cols]                            # [B, N_in, D_s]
        id_in = self.col_id[in_cols]                         # [N_in, D_id]
        id_in_exp = id_in.unsqueeze(0).expand(B, -1, -1).to(s_in.dtype)

        norm_s = self.in_norm(s_in).to(s_in.dtype)
        q = self.in_q_proj(torch.cat([norm_s, id_in_exp], dim=-1))  # [B, N_in, D_s]
        k = self.in_k_proj(h_input)                          # [B, D_s]
        v = self.in_v_proj(h_input)                          # [B, D_s]

        # Gate: sigmoid(q · k / sqrt(D_s)), per (B, N_in)
        scale = 1.0 / (self.cfg.D_s ** 0.5)
        gate = torch.sigmoid(
            torch.sum(q * k.unsqueeze(1), dim=-1) * scale
        )                                                    # [B, N_in]
        delta = (gate.unsqueeze(-1) * v.unsqueeze(1)).to(s_in.dtype)

        # Additive in-place update of input-plane slice. Keep storage dtype.
        self.s = self.s.index_copy(
            1,
            in_cols,
            (self.s.index_select(1, in_cols) + delta).to(self.s.dtype),
        )

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One round: compute m_out, w_out, send via scatter-add, update state.

        Processes per-edge work (delta_MLP, score_MLP, msg·weight) in K-chunks
        of `k_chunk` edges at a time, accumulating into `incoming` and
        `w_out_flat`. This caps peak memory to ~1/K_CHUNKS of the naïve
        all-K-at-once version, which is critical for TBPTT because the
        saved activations per tick dominate backward memory.

        Returns (m_out, w_out_flat, incoming) for use by plasticity.
        """
        cfg = self.cfg
        B, N, K, D_s, D_id = self.s.shape[0], cfg.N, cfg.K, cfg.D_s, cfg.D_id
        state_dtype = self.s.dtype

        # --- Content: m_out[c] = content_MLP(norm(s[c]), id[c]). Compute once.
        s_norm = self.mlps.content_norm(self.s).to(state_dtype)
        id_exp = self.col_id.unsqueeze(0).expand(B, -1, -1).to(state_dtype)
        content_in = torch.cat([s_norm, id_exp], dim=-1)     # [B, N, D_s + D_id]
        m_out = self.mlps.content_mlp(content_in)            # [B, N, D_s]

        # --- Per-chunk edge work
        # k_chunk=8 was the sweet spot empirically: smaller chunks save less
        # activation memory but more Python-loop overhead; larger chunks save
        # more activations which hurts backward. Tune per scale if needed.
        k_chunk = 8 if K >= 16 else K
        incoming = torch.zeros_like(self.s)                  # [B, N, D_s]
        w_out_chunks: list[torch.Tensor] = []

        E_bias_2d = self.E_bias_flat.view(N, K).to(state_dtype)

        for k_start in range(0, K, k_chunk):
            k_end = min(k_start + k_chunk, K)
            k_size = k_end - k_start

            out_nbrs_chunk = self.out_nbrs[:, k_start:k_end]          # [N, k_size]
            id_dst_chunk = self.col_id[out_nbrs_chunk].to(state_dtype)  # [N, k_size, D_id]
            id_dst_exp = id_dst_chunk.unsqueeze(0).expand(B, N, k_size, D_id)

            # delta_MLP input — flatten (B*N*k_size, D_s + D_id)
            m_out_src_exp = m_out.unsqueeze(2).expand(B, N, k_size, D_s)
            delta_in = torch.cat([m_out_src_exp, id_dst_exp], dim=-1)
            delta_out = self.mlps.delta_mlp(
                delta_in.reshape(B * N * k_size, D_s + D_id)
            ).reshape(B, N, k_size, D_s)
            m_edge = m_out_src_exp + delta_out                          # [B, N, k_size, D_s]

            # score_MLP
            s_norm_src_exp = s_norm.unsqueeze(2).expand(B, N, k_size, D_s)
            id_src_exp = id_exp.unsqueeze(2).expand(B, N, k_size, D_id)
            score_in = torch.cat([s_norm_src_exp, id_src_exp, id_dst_exp], dim=-1)
            score = self.mlps.score_mlp(
                score_in.reshape(B * N * k_size, D_s + 2 * D_id)
            ).reshape(B, N, k_size)

            E_bias_chunk = E_bias_2d[:, k_start:k_end].unsqueeze(0)     # [1, N, k_size]
            w_out = torch.sigmoid(score + E_bias_chunk)                 # [B, N, k_size]
            w_out_chunks.append(w_out)

            # Send + scatter-add into incoming
            msg = (w_out.unsqueeze(-1) * m_edge).reshape(B, N * k_size, D_s)
            # edge_dst for this chunk: flatten out_nbrs[:, k_start:k_end]
            edge_dst_chunk = out_nbrs_chunk.reshape(-1)                 # [N * k_size]
            incoming = incoming.index_add(1, edge_dst_chunk, msg)

        w_out_flat = torch.cat(w_out_chunks, dim=-1).reshape(B, N * K)  # [B, N*K]

        # --- Residual update: gated-tanh
        s_norm_for_update = self.mlps.update_norm_s(self.s).to(state_dtype)
        in_norm = self.mlps.update_norm_in(incoming).to(state_dtype)
        update_in = torch.cat([s_norm_for_update, id_exp, in_norm], dim=-1)
        update = torch.tanh(self.mlps.update_mlp(update_in))
        alpha = torch.sigmoid(self.decay_proj(self.col_id)).squeeze(-1)  # [N]
        alpha = alpha.unsqueeze(0).unsqueeze(-1).to(state_dtype)         # [1, N, 1]
        self.s = (alpha * self.s + (1.0 - alpha) * update.to(state_dtype)).to(state_dtype)

        return m_out, w_out_flat, incoming

    def _readout_motor(self) -> torch.Tensor:
        """Cross-attn motor vector from output-plane column states."""
        cfg = self.cfg
        B = self.s.shape[0]
        state_dtype = self.s.dtype
        out_cols = self.output_positions                     # [N_out]
        s_out = self.out_norm(self.s[:, out_cols]).to(state_dtype)

        k = self.out_k_proj(s_out)                           # [B, N_out, D_s]
        v = self.out_v_proj(s_out)                           # [B, N_out, D_s]
        q = self.motor_query.to(state_dtype).unsqueeze(0).expand(B, -1)  # [B, D_s]

        # Dot-product attention
        scale = 1.0 / (cfg.D_s ** 0.5)
        scores = torch.sum(k * q.unsqueeze(1), dim=-1) * scale   # [B, N_out]
        attn = F.softmax(scores, dim=-1)                          # [B, N_out]
        motor = torch.sum(attn.unsqueeze(-1) * v, dim=1)          # [B, D_s]
        return motor

    # -----------------------------------------------------------------
    # Plasticity
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    def _surprise_and_buffer_bookkeeping(
        self, token_id: torch.Tensor, logits: torch.Tensor
    ) -> None:
        """Kept outside the compile region because the ring-buffer cursor is a
        Python integer and Dynamo would recompile per-cursor-value otherwise."""
        cfg = self.cfg
        self.surprise_prev = self.surprise_ema.detach().clone()
        self.surprise_ema = multi_horizon_surprise(
            self.pred_buf,
            self.pred_cursor,
            cfg.K_buf,
            self.pred_filled,
            token_id,
            self.surprise_ema,
            cfg.alpha_gamma_s,
        )
        write_prediction_buffer(self.pred_buf, self.pred_cursor, logits)
        self.pred_cursor = (self.pred_cursor + 1) % cfg.K_buf
        self.pred_filled = min(self.pred_filled + 1, cfg.K_buf)

    def _sum_edge_abs_per_source(self, w_out_flat: torch.Tensor) -> torch.Tensor:
        """Sum |w_out| over each source column's K out-edges. [B, N]"""
        B = w_out_flat.shape[0]
        N, K = self.cfg.N, self.cfg.K
        return w_out_flat.abs().view(B, N, K).sum(dim=-1)

    def _plasticity_step(
        self,
        m_out: torch.Tensor,
        incoming: torch.Tensor,
        w_out_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Neuromod forward + Hebbian update on E_bias. Returns η_global [B]."""
        cfg = self.cfg
        with torch.autocast(device_type=self.s.device.type, enabled=False):
            # --- Activities
            pre, post = compute_activities(m_out, incoming)    # [B, N] each

            # --- Per-edge pre gathered at in-neighbours (for neuromod head features)
            B, N, K = self.s.shape[0], cfg.N, cfg.K
            # w_in[c, k] = the w_out on the edge that terminates at c, per column.
            # We don't directly have "in-edges per column" unless we invert.
            # For the neuromod features, we use a simpler proxy: for each column,
            # look at the K *outgoing* edges and their weights as local features.
            # This is a small simplification: it tells the head "how strongly is
            # this column broadcasting?" instead of "how strongly is it receiving?"
            # The Hebbian math itself is post-synaptic-correct; only the feature
            # is a proxy. Cheaper than building an in-edge-per-column table.
            w_col_K = w_out_flat.view(B, N, K).float()         # [B, N, K]
            pre_at_out_nbrs = pre[:, self.out_nbrs].float()    # [B, N, K] (senders=srces of rev edges, approx)

            # Surprise delta
            surprise_delta = (self.surprise_ema - self.surprise_prev).float()

            eta_global, eta, beta = self.neuromod(
                self.surprise_ema.float(),
                surprise_delta,
                self.input_ctx_ema.float(),
                self.mag_ema,
                self.var_ema,
                self.traffic_ema,
                self.col_id.float(),
                post.float(),
                w_col_K,
                pre_at_out_nbrs,
            )
            # Average per-batch eta_global into a scalar for the shared E_bias.
            eta_global_scalar = eta_global.mean()
            # Post-synaptic eta and beta: average over batch to get per-column
            # scalars (plastic state is shared across the cohort).
            eta_post = eta.mean(dim=0)                         # [N]
            beta_post = beta.mean(dim=0)                       # [N]

            # --- Hebbian update
            self.E_bias_flat = hebbian_update_incoming(
                E_bias_flat=self.E_bias_flat,
                w_out_flat=w_out_flat.float(),
                edge_src=self.edge_src,
                edge_dst=self.edge_dst,
                pre=pre.float(),
                post=post.float(),
                eta_global=eta_global_scalar,
                eta=eta_post,
                beta=beta_post,
                E_max=cfg.E_bias_max,
            )
        return eta_global
