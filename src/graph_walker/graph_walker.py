"""GraphWalkerMemory — trajectory-routed plastic concept graph.

Hot path per token:
  1. Softmax over input-plane columns → per-head starting column
  2. Inject token content at starting columns
  3. For L hops: content_mlp at current col, score K out-edges, Gumbel
     top-1 pick next col, record trajectory + outgoing message
  4. Aggregate messages across heads/hops per destination col
  5. LIF-integrate messages into visited columns' states (others frozen)
  6. Cross-attn readout over H·L visited column states → motor → logits

Every mod_period tokens: neuromod + Hebbian update to E_bias based on
accumulated per-edge traversal counts over the window.

See docs/graph_walker.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.readout import (
    MultiHorizonReadout,
    _FallbackRMSNorm,
    init_prediction_buffer,
    multi_horizon_surprise,
    write_prediction_buffer,
)
from src.graph_walker.routing import gumbel_top1_softmax, gumbel_schedule
from src.graph_walker.topology import build_topology


def _rmsnorm(dim: int) -> nn.Module:
    return _FallbackRMSNorm(dim)


# =====================================================================
# Column-level shared compute
# =====================================================================


class ColumnCompute(nn.Module):
    """Shared-weight content_mlp, q_proj, k_proj — applied per visited column.

    Unlike column_graph, these fire ONLY on visited columns (H·L = 16 at
    default config), not on all N=1024 columns. So the per-column work
    per token is dominated by H·L·MLP_size, not N·MLP_size.
    """

    def __init__(self, cfg: GraphWalkerConfig) -> None:
        super().__init__()
        D_s, D_id = cfg.D_s, cfg.D_id
        H, D_q = cfg.n_score_heads, cfg.D_q_per_head

        self.content_norm = _rmsnorm(D_s)
        self.content_mlp = nn.Sequential(
            nn.Linear(D_s + D_id, cfg.ffn_mult_content * D_s),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult_content * D_s, D_s),
        )
        self.q_proj = nn.Sequential(
            nn.Linear(D_s + D_id, 2 * H * D_q),
            nn.GELU(),
            nn.Linear(2 * H * D_q, H * D_q),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(D_id, 2 * H * D_q),
            nn.GELU(),
            nn.Linear(2 * H * D_q, H * D_q),
        )

        # Small-random init on final layers of q/k. We want initial routing
        # nearly-uniform (small scores) but NOT exactly zero, because
        # zero-init would make the bilinear product q·k have zero gradient
        # (grad to q requires k ≠ 0 and vice versa — classic dead product).
        # At std=0.02 and D_q=64, initial scores are O(0.02² × √64) ≈ 0.003
        # — nearly uniform softmax, and gradients flow freely.
        for module in (self.q_proj[-1], self.k_proj[-1]):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)

        # Depth-scaled init on first layers.
        for module in (self.content_mlp[0], self.q_proj[0], self.k_proj[0]):
            nn.init.normal_(module.weight, mean=0.0, std=0.014)

        self.n_heads = H
        self.D_q = D_q


# =====================================================================
# GraphWalkerMemory
# =====================================================================


@dataclass
class WalkerReadout:
    """Return bundle from step()."""
    motor: torch.Tensor                   # [B, D_s]
    logits: torch.Tensor                  # [B, K_horizons, V]
    surprise_ema: torch.Tensor            # [B, K_horizons]
    visit_freq_step: torch.Tensor | None  # [N] fractional visits (if tracked)
    load_balance_loss: torch.Tensor       # scalar — aux loss term


class GraphWalkerMemory(nn.Module):
    def __init__(
        self, cfg: GraphWalkerConfig, tied_token_emb: nn.Embedding
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tied_token_emb = tied_token_emb

        # Topology
        topo = build_topology(
            plane_rows=cfg.plane_rows, plane_cols=cfg.plane_cols,
            L=cfg.L, K=cfg.K, p_rewire=cfg.p_rewire,
            K_intra_fraction=cfg.K_intra_fraction, seed=cfg.topology_seed,
        )
        self.register_buffer("out_nbrs", topo.out_nbrs, persistent=False)
        self.register_buffer("edge_src", topo.edge_src, persistent=False)
        self.register_buffer("edge_dst", topo.edge_dst, persistent=False)
        self.register_buffer("plane_ids", topo.plane_ids, persistent=False)
        self.register_buffer(
            "input_positions", topo.input_positions, persistent=False,
        )
        self.register_buffer(
            "output_positions", topo.output_positions, persistent=False,
        )

        # Column identity
        torch.manual_seed(cfg.init_seed)
        self.col_id = nn.Parameter(torch.randn(cfg.N, cfg.D_id) * 0.02)

        # Per-column decay α = σ(decay_proj(id))
        self.decay_proj = nn.Linear(cfg.D_id, 1)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.decay_proj.bias)

        # Shared column compute
        self.cols = ColumnCompute(cfg)

        # Input plane: per-head softmax over input-plane columns.
        # q_in: [B, D_s] -> [B, H, D_q_in]
        self.input_q_proj = nn.Linear(
            cfg.D_s, cfg.n_heads * cfg.D_q_in, bias=False,
        )
        self.input_k_proj = nn.Linear(cfg.D_id, cfg.D_q_in, bias=False)
        self.input_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        # Zero-init v so initial token-injection is no-op (state stays bounded).
        nn.init.zeros_(self.input_v_proj.weight)

        # Plastic per-head input bias over input-plane columns: [H, N_in]
        # fp32 plastic state, not a parameter.
        self._N_in = cfg.N_per_plane
        self.register_buffer(
            "input_E_bias",
            torch.zeros(cfg.n_heads, cfg.N_per_plane, dtype=torch.float32),
            persistent=False,
        )

        # Output readout: cross-attn over H·L visited column states.
        self.out_norm = _rmsnorm(cfg.D_s)
        self.motor_query = nn.Parameter(torch.randn(cfg.D_s) * 0.02)
        self.out_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.out_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)

        # Multi-horizon readout (reused from column-graph)
        self.readout = MultiHorizonReadout(cfg)

        # --- Persistent state (buffers) ---
        self._state_initialized = False
        self.s: torch.Tensor                 # [B, N, D_s]
        self.E_bias_flat: torch.Tensor        # [N*K] fp32
        self.pred_buf: torch.Tensor           # [B, K_buf, K_h, V]
        self.surprise_ema: torch.Tensor       # [B, K_h]
        self.surprise_prev: torch.Tensor      # [B, K_h]
        self.pred_cursor: int = 0
        self.pred_filled: int = 0
        self.tick_counter: int = 0

        # Window-accumulated stats for plasticity (reset every mod_period)
        self.co_visit_flat: torch.Tensor | None = None  # [N*K] fp32
        self.window_len: int = 0

        # Visit-frequency count across the current training step (for
        # load-balance aux loss). Reset externally per step.
        self.visit_count: torch.Tensor | None = None     # [N] fp32

        # Training / routing scheduling — caller sets these externally.
        self.training_step: int = 0

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def begin_segment(self, B: int, device: torch.device) -> None:
        """Reset working memory (per-document). E_bias persists."""
        cfg = self.cfg
        dtype_fast = self._fast_dtype(device)

        self.s = torch.zeros(B, cfg.N, cfg.D_s, device=device, dtype=dtype_fast)
        self.pred_buf = init_prediction_buffer(
            B, cfg.K_buf, cfg.K_horizons, cfg.vocab_size, device, dtype_fast,
        )
        self.surprise_ema = torch.zeros(
            B, cfg.K_horizons, device=device, dtype=torch.float32,
        )
        self.surprise_prev = torch.zeros_like(self.surprise_ema)
        self.pred_cursor = 0
        self.pred_filled = 0
        self.tick_counter = 0
        self.window_len = 0

        # Lazy-init plastic state on first segment
        if not hasattr(self, "E_bias_flat") or self.E_bias_flat is None:
            self.reset_plastic_memory(device)

        # Fresh per-segment counters
        self.co_visit_flat = torch.zeros(
            cfg.num_edges, device=device, dtype=torch.float32,
        )
        self.visit_count = torch.zeros(cfg.N, device=device, dtype=torch.float32)

        self._state_initialized = True

    def reset_plastic_memory(self, device: torch.device) -> None:
        """Hard reset of long-term plastic E_bias + input_E_bias."""
        self.E_bias_flat = torch.zeros(
            self.cfg.num_edges, device=device, dtype=torch.float32,
        )
        self.input_E_bias = self.input_E_bias.to(device).zero_()

    def detach_state(self) -> None:
        """TBPTT boundary: preserve values, sever gradient graph."""
        if not self._state_initialized:
            return
        self.s = self.s.detach()
        self.E_bias_flat = self.E_bias_flat.detach()
        self.pred_buf = self.pred_buf.detach()
        self.surprise_ema = self.surprise_ema.detach()
        self.surprise_prev = self.surprise_prev.detach()
        if self.co_visit_flat is not None:
            self.co_visit_flat = self.co_visit_flat.detach()
        if self.visit_count is not None:
            self.visit_count = self.visit_count.detach()

    def _fast_dtype(self, device: torch.device) -> torch.dtype:
        choice = self.cfg.state_dtype
        if choice == "bf16":
            return torch.bfloat16
        if choice == "fp32":
            return torch.float32
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    # -----------------------------------------------------------------
    # Step: one token's forward + bookkeeping
    # -----------------------------------------------------------------

    def step(self, token_id: torch.Tensor) -> WalkerReadout:
        """One token → H trajectories × L hops → readout + plasticity."""
        assert self._state_initialized, "call begin_segment() first"
        cfg = self.cfg
        B = self.s.shape[0]
        device = self.s.device

        # Current routing parameters (annealed by caller via training_step)
        tau = gumbel_schedule(
            self.training_step, cfg.gumbel_tau_start, cfg.gumbel_tau_end,
            cfg.gumbel_anneal_steps,
        )
        epsilon = gumbel_schedule(
            self.training_step, cfg.epsilon_start, cfg.epsilon_end,
            cfg.epsilon_anneal_steps,
        )
        is_training = self.training

        # 1. Embed token
        h_input = self.tied_token_emb(token_id)                     # [B, D_s]

        # 2. Per-head start-column softmax over input plane
        #    q[B, H, D_q_in] · k[N_in, D_q_in].T = [B, H, N_in]
        query_flat = self.input_q_proj(h_input).view(
            B, cfg.n_heads, cfg.D_q_in,
        )                                                            # [B, H, D_q_in]
        input_ids = self.col_id[self.input_positions]                # [N_in, D_id]
        input_keys = self.input_k_proj(input_ids)                    # [N_in, D_q_in]
        # Scale
        scale_in = 1.0 / (cfg.D_q_in ** 0.5)
        scores_in = torch.einsum(
            "bhd,nd->bhn", query_flat, input_keys,
        ) * scale_in                                                 # [B, H, N_in]
        scores_in = scores_in + self.input_E_bias.unsqueeze(0).to(scores_in.dtype)

        # Gumbel top-1 per head
        # Flatten H into batch for routing: [B*H, N_in]
        scores_in_flat = scores_in.reshape(B * cfg.n_heads, self._N_in)
        rout_in = gumbel_top1_softmax(
            scores_in_flat, tau=tau, epsilon=epsilon, training=is_training,
        )
        # [B, H] col-index into input_positions → [B, H] global col-index
        start_local = rout_in.selected_idx.view(B, cfg.n_heads)       # [B, H]
        start_cols = self.input_positions[start_local]                # [B, H]

        # 3. Inject token content at starting columns
        v_inject = self.input_v_proj(h_input)                         # [B, D_s]
        # For each (b, h), update s[b, start_cols[b, h]] additively.
        # State update: s[start] = α * s + (1-α) * tanh(v_inject)
        alpha = torch.sigmoid(self.decay_proj(self.col_id)).squeeze(-1)  # [N]

        # We'll do a single aggregation pass that handles both injection AND
        # trajectory messages to avoid multiple scatter operations.
        # Accumulate additive contributions to s_new[b, c] as weighted tanh(incoming).
        # For injection: weight=1, incoming = v_inject at start_cols.

        # 4. Walk L hops per head (read-only on self.s)
        # We build `trajectory[B, H, L] = col idx` and collect messages.
        trajectory = torch.zeros(
            B, cfg.n_heads, cfg.n_hops, dtype=torch.long, device=device,
        )
        trajectory[:, :, 0] = start_cols

        # messages[b, h, t] is the m_out emitted at hop t (before routing to hop t+1).
        # weights[b, h, t] is the straight-through weight on the chosen edge at
        # hop t (used to weight the message sent to hop t+1's column). In forward
        # this weight is 1.0 (one-hot selected); in backward it carries gradient
        # to q_proj, k_proj, E_bias via the softmax distribution.
        msg_stack: list[torch.Tensor] = []
        weight_stack: list[torch.Tensor] = []

        # Also need the input-plane routing ste weight (for gradient to input_q_proj)
        # Pick the ste weight at the chosen start col per head.
        input_ste_weights = torch.gather(
            rout_in.ste_weights, 1, rout_in.selected_idx.unsqueeze(1),
        ).squeeze(1)                                                    # [B*H]

        # Trajectory state is simply "current col per (b, h)" — start cols.
        cur = start_cols.reshape(B * cfg.n_heads)                     # [B*H]

        cfg_n_heads = cfg.n_heads
        BH = B * cfg_n_heads
        state_dtype = self.s.dtype

        # Running load-balance counts
        visit_count_step = torch.zeros(cfg.N, device=device, dtype=torch.float32)

        # Traversed-edge indices (for plasticity accumulation later): list of [B*H] flat-edge indices
        traversed_edges: list[torch.Tensor] = []

        # State at each visited column, broadcast across B*H. We'll gather s[cur] on demand.
        for hop_t in range(cfg.n_hops):
            # s at current columns: [B*H, D_s]
            # Note: self.s is [B, N, D_s]; we need s[b, cur[b_h]] for each (b, h).
            # cur spans B*H; reshape to per-batch gather.
            cur_bh = cur.view(B, cfg_n_heads)                         # [B, H]
            s_cur = torch.gather(
                self.s,
                1,
                cur_bh.unsqueeze(-1).expand(B, cfg_n_heads, cfg.D_s),
            ).reshape(BH, cfg.D_s)                                     # [B*H, D_s]

            # id at current cols
            id_cur = self.col_id[cur]                                  # [B*H, D_id]

            # content_mlp → m_out [B*H, D_s]
            cat_content = torch.cat([
                self.cols.content_norm(s_cur).to(state_dtype),
                id_cur.to(state_dtype),
            ], dim=-1)
            m_out = self.cols.content_mlp(cat_content)                 # [B*H, D_s]

            # Store for aggregation (the NEXT column receives this)
            msg_stack.append(m_out.view(B, cfg_n_heads, cfg.D_s))

            if hop_t == cfg.n_hops - 1:
                # Last hop: don't pick next; trajectory done.
                break

            # q_proj at current cols
            q = self.cols.q_proj(cat_content).view(
                BH, self.cols.n_heads, self.cols.D_q,
            )                                                          # [B*H, H_score, D_q]

            # K neighbors of each current col
            nbrs_of_cur = self.out_nbrs[cur]                           # [B*H, K]
            nbr_ids = self.col_id[nbrs_of_cur]                          # [B*H, K, D_id]
            k_nbrs = self.cols.k_proj(
                nbr_ids.reshape(-1, cfg.D_id)
            ).view(BH, cfg.K, self.cols.n_heads, self.cols.D_q)        # [B*H, K, H_score, D_q]

            # Bilinear score: Σ_h q · k (per-head sum)
            # Output: [B*H, K]
            scale = 1.0 / (self.cols.D_q ** 0.5)
            scores = torch.einsum("bhd,bkhd->bk", q, k_nbrs) * scale

            # Add plastic E_bias for these (cur, k) flat edges
            # Edge flat index: edge_flat[cur, k] = cur * K + k
            k_range = torch.arange(cfg.K, device=device)
            edge_flat = cur.unsqueeze(1) * cfg.K + k_range.unsqueeze(0)  # [B*H, K]
            E_vals = self.E_bias_flat[edge_flat].to(scores.dtype)       # [B*H, K]
            scores = scores + E_vals

            # Gumbel top-1 softmax
            rout = gumbel_top1_softmax(
                scores, tau=tau, epsilon=epsilon, training=is_training,
            )
            next_local = rout.selected_idx                             # [B*H]
            next_col = torch.gather(
                nbrs_of_cur, 1, next_local.unsqueeze(-1),
            ).squeeze(-1)                                              # [B*H]
            # The flat edge taken (for plasticity)
            edge_taken_flat = cur * cfg.K + next_local                 # [B*H]

            # Pick the straight-through weight at the chosen edge.
            # Forward: 1.0 (one-hot). Backward: soft_probs via STE.
            chosen_weight = torch.gather(
                rout.ste_weights, 1, next_local.unsqueeze(1),
            ).squeeze(1)                                               # [B*H]
            weight_stack.append(chosen_weight)

            # Record trajectory
            trajectory[:, :, hop_t + 1] = next_col.view(B, cfg_n_heads)
            traversed_edges.append(edge_taken_flat)

            # Track visit frequency (each trajectory step counts the dest col)
            visit_count_step.scatter_add_(
                0, next_col, torch.ones_like(next_col, dtype=torch.float32),
            )

            # Advance
            cur = next_col

        # Also count start columns in visit frequency
        visit_count_step.scatter_add_(
            0, start_cols.reshape(-1),
            torch.ones(BH, device=device, dtype=torch.float32),
        )
        if self.visit_count is not None:
            with torch.no_grad():
                self.visit_count = self.visit_count + visit_count_step

        # 5. Aggregate messages per visited destination column
        #    For each head and each hop t in 1..L-1:
        #      dest = trajectory[:, h, t]
        #      message arriving at dest = m_out from hop t-1
        #    Plus: start-column injection = v_inject weighted to start_cols
        # Build incoming [B, N, D_s]
        incoming = torch.zeros_like(self.s)

        # Injection contribution to start columns — weighted by input STE weight
        # so gradient flows back to input_q_proj.
        inject_msg = v_inject.unsqueeze(1).expand(B, cfg_n_heads, cfg.D_s).reshape(BH, cfg.D_s)
        inject_msg = inject_msg * input_ste_weights.unsqueeze(-1).to(inject_msg.dtype)
        start_cols_flat = start_cols.reshape(BH)                       # [B*H]
        batch_idx = torch.arange(B, device=device).repeat_interleave(cfg_n_heads)  # [B*H]
        incoming_flat = incoming.view(B * cfg.N, cfg.D_s)
        dest_flat_inject = batch_idx * cfg.N + start_cols_flat
        incoming_flat.index_add_(0, dest_flat_inject, inject_msg.to(incoming_flat.dtype))

        # Trajectory messages: hop t's m_out goes to trajectory[:, :, t+1]
        # weighted by the straight-through edge weight (gradient flows to q/k/E_bias).
        for hop_t in range(cfg.n_hops - 1):
            m = msg_stack[hop_t]                                      # [B, H, D_s]
            dest = trajectory[:, :, hop_t + 1]                        # [B, H]
            w = weight_stack[hop_t]                                   # [B*H]
            m_flat = m.reshape(BH, cfg.D_s) * w.unsqueeze(-1).to(m.dtype)
            dest_flat = batch_idx * cfg.N + dest.reshape(BH)
            incoming_flat.index_add_(0, dest_flat, m_flat.to(incoming_flat.dtype))

        incoming = incoming_flat.view(B, cfg.N, cfg.D_s)

        # 6. LIF integrate at visited columns (non-visited stay unchanged).
        # Build visited mask per batch: [B, N] {0, 1}
        visited_mask = torch.zeros(B, cfg.N, device=device, dtype=state_dtype)
        # Start cols
        visited_mask.scatter_(1, start_cols, 1.0)
        # Trajectory cols per hop
        for hop_t in range(1, cfg.n_hops):
            visited_mask.scatter_(1, trajectory[:, :, hop_t], 1.0)

        alpha_exp = alpha.unsqueeze(0).unsqueeze(-1).to(state_dtype)   # [1, N, 1]
        # Where visited: s_new = α·s + (1-α)·tanh(incoming)
        # Where not visited: s_new = s  (unchanged)
        s_visited_update = alpha_exp * self.s + (1.0 - alpha_exp) * torch.tanh(incoming)
        self.s = torch.where(
            visited_mask.unsqueeze(-1) > 0.5, s_visited_update, self.s,
        )

        # 7. Readout: cross-attn over H·L visited column states
        # Gather visited states: [B, H*L, D_s]
        traj_flat = trajectory.reshape(B, cfg_n_heads * cfg.n_hops)    # [B, H*L]
        traj_states = torch.gather(
            self.s, 1, traj_flat.unsqueeze(-1).expand(B, -1, cfg.D_s),
        )                                                              # [B, H*L, D_s]

        s_traj = self.out_norm(traj_states).to(state_dtype)
        k = self.out_k_proj(s_traj)                                    # [B, H*L, D_s]
        v = self.out_v_proj(s_traj)
        q_motor = self.motor_query.to(state_dtype).unsqueeze(0).expand(B, -1)
        scale_out = 1.0 / (cfg.D_s ** 0.5)
        attn_scores = torch.sum(k * q_motor.unsqueeze(1), dim=-1) * scale_out  # [B, H*L]
        attn = F.softmax(attn_scores, dim=-1)
        motor = torch.sum(attn.unsqueeze(-1) * v, dim=1)               # [B, D_s]

        unemb = self.tied_token_emb.weight
        logits = self.readout(motor, unemb)                             # [B, K_h, V]

        # 8. Surprise + ring buffer (same as column_graph)
        self._surprise_and_buffer_bookkeeping(token_id, logits)

        # 9. Plasticity window accumulation — co-visitation counts
        with torch.no_grad():
            if traversed_edges:
                # Stack: [L-1, B*H] flat edge indices
                all_edges = torch.cat(traversed_edges, dim=0)           # [(L-1)*B*H]
                # Count visits per edge
                self.co_visit_flat = self.co_visit_flat + torch.bincount(
                    all_edges,
                    minlength=cfg.num_edges,
                ).to(torch.float32)
        self.window_len += 1

        # 10. Plasticity fires every mod_period
        if self.tick_counter + 1 >= cfg.mod_period and (self.tick_counter + 1) % cfg.mod_period == 0:
            self._plasticity_step()
        self.tick_counter += 1

        # 11. Load-balance aux loss: KL(visit_freq, uniform) over visited cols
        # We use soft_probs from input routing + any soft routing stats in walk.
        # For v1: simple KL from uniform on visit_count_step.
        with torch.no_grad():
            total = visit_count_step.sum().clamp(min=1.0)
            freq = visit_count_step / total
        # KL(freq, uniform) = sum freq * log(freq * N), guarding zeros
        # Gradient does not flow through visit_count (it's discrete), so this
        # aux loss is informational for now. The true load-balance signal in
        # MoE relies on soft_probs; for simplicity we skip the gradient
        # pathway in v1 and use ε-exploration + E_bias plasticity for
        # routing diversification.
        load_balance_loss = torch.tensor(0.0, device=device)

        return WalkerReadout(
            motor=motor, logits=logits,
            surprise_ema=self.surprise_ema,
            visit_freq_step=freq.detach() if cfg.lambda_balance > 0 else None,
            load_balance_loss=load_balance_loss,
        )

    # -----------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    def _surprise_and_buffer_bookkeeping(
        self, token_id: torch.Tensor, logits: torch.Tensor,
    ) -> None:
        cfg = self.cfg
        self.surprise_prev = self.surprise_ema.detach().clone()
        self.surprise_ema = multi_horizon_surprise(
            self.pred_buf, self.pred_cursor, cfg.K_buf, self.pred_filled,
            token_id, self.surprise_ema, cfg.alpha_gamma_s,
        )
        write_prediction_buffer(self.pred_buf, self.pred_cursor, logits)
        self.pred_cursor = (self.pred_cursor + 1) % cfg.K_buf
        self.pred_filled = min(self.pred_filled + 1, cfg.K_buf)

    @torch._dynamo.disable
    def _plasticity_step(self) -> None:
        """Hebbian on co-visitation counts. Fires every mod_period ticks."""
        cfg = self.cfg
        device = self.s.device

        with torch.autocast(device_type=device.type, enabled=False):
            # Normalize by window length
            window = max(self.window_len, 1)
            co_visit_norm = (self.co_visit_flat.float() / window) if self.co_visit_flat is not None \
                else torch.zeros_like(self.E_bias_flat)

            # Simple Hebbian rule v1: strengthen edges traversed, decay untouched
            # η_global from surprise magnitude (higher surprise → faster learning)
            surprise_scalar = self.surprise_ema.mean().float()
            eta_global = 0.1 * torch.sigmoid(surprise_scalar - 1.0)

            # Global step: E_bias += η * (co_visit - decay * E_bias)
            beta_decay = 0.1
            delta = eta_global * (co_visit_norm - beta_decay * self.E_bias_flat)
            self.E_bias_flat = (self.E_bias_flat + delta).clamp(
                -cfg.E_bias_max, cfg.E_bias_max,
            )

        # Reset window counters
        self.co_visit_flat = torch.zeros_like(self.co_visit_flat)
        self.window_len = 0
