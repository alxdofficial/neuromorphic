"""GraphWalkerMemory — trajectory-routed plastic concept graph.

Hot path per token:
  1. Softmax over input-plane columns → per-head starting column
  2. Inject token content at starting columns
  3. Advance each persistent walker one hop: content_mlp at current col,
     score K out-edges, Gumbel top-1 pick next col
  4. Aggregate sparse messages at touched destination columns
  5. LIF-integrate messages into visited columns' states (others frozen)
  6. Cross-attn readout over H walker endpoints → motor_state

Every mod_period tokens: batch-compute exact surprise over the accumulated
window, then do surprise-gated Hebbian on traversed-edge counts (scalar-eta v1).

See docs/graph_walker.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.readout import MultiHorizonReadout, _FallbackRMSNorm
from src.graph_walker.routing import gumbel_top1_softmax, gumbel_schedule
from src.graph_walker.topology import build_topology
from src.graph_walker.triton_sparse_update import sparse_lif_update


def _rmsnorm(dim: int) -> nn.Module:
    return _FallbackRMSNorm(dim)


# =====================================================================
# Column-level shared compute
# =====================================================================


class DeepContentMLP(nn.Module):
    """Deep residual FFN stack with shared weights.

    Replaces the old 2-layer content_mlp with:
        in_proj: (D_s + D_id) → D_s
        n_layers × ResidualFFN(D_s, D_hid = ffn_mult · D_s)

    Each ResidualFFN block is `x + down(gelu(up(norm(x))))`. The stack lives
    in the hot per-hop path, so this width/depth is now kept smaller than the
    external model width; extra capacity is added later in model space.
    This keeps throughput sane —
    one big batched matmul per sublayer, not L small ones.
    """

    def __init__(self, D_in: int, D_s: int, D_hid: int, n_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(D_in, D_s)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.014)
        self.blocks = nn.ModuleList([
            _ResidualFFN(D_s, D_hid) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return x


class _ResidualFFN(nn.Module):
    def __init__(self, D_s: int, D_hid: int) -> None:
        super().__init__()
        self.norm = _rmsnorm(D_s)
        self.up = nn.Linear(D_s, D_hid)
        self.down = nn.Linear(D_hid, D_s)
        # Depth-scaled init on final projection so sum of residuals stays bounded.
        nn.init.normal_(self.down.weight, mean=0.0, std=(2.0 / D_hid) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(self.norm(x))))


class PerPlaneMLP(nn.Module):
    """2-layer FFN with L independent weight sets, dispatched by plane index.

    Each plane gets its own (W1, b1, W2, b2). Forward segment-matmuls by
    plane: loops L times, each iteration runs F.linear on the subset of
    walkers currently at that plane. Adds L kernel launches per layer
    (vs one shared matmul) in exchange for L× more params and per-plane
    specialisation.

    When all walkers happen to be in a single plane the remaining L-1
    iterations early-exit via mask.any() == False — cheap.
    """

    def __init__(self, L: int, D_in: int, D_hid: int, D_out: int) -> None:
        super().__init__()
        self.L = L
        self.D_in = D_in
        self.D_hid = D_hid
        self.D_out = D_out

        # Layer 1: D_in → D_hid
        self.W1 = nn.Parameter(torch.empty(L, D_hid, D_in))
        self.b1 = nn.Parameter(torch.zeros(L, D_hid))
        # Layer 2: D_hid → D_out
        self.W2 = nn.Parameter(torch.empty(L, D_out, D_hid))
        self.b2 = nn.Parameter(torch.zeros(L, D_out))

        # Depth-scaled init on first layer, small on second.
        nn.init.normal_(self.W1, mean=0.0, std=0.014)
        nn.init.normal_(self.W2, mean=0.0, std=(2.0 / D_hid) ** 0.5)

    def forward(
        self, x: torch.Tensor, plane_idx: torch.Tensor,
    ) -> torch.Tensor:
        """x: [N_walkers, D_in]; plane_idx: [N_walkers] long in [0, L)."""
        # Scatter into output via mask per plane. L kernel launches each
        # layer but each slice is small — negligible at B·H~100s of walkers.
        out = torch.empty(
            x.shape[0], self.D_out, device=x.device, dtype=x.dtype,
        )
        for p in range(self.L):
            mask = plane_idx == p
            if not mask.any():
                continue
            x_p = x[mask]
            h_p = F.linear(x_p, self.W1[p], self.b1[p])
            h_p = F.gelu(h_p)
            o_p = F.linear(h_p, self.W2[p], self.b2[p])
            out[mask] = o_p
        return out


class ColumnCompute(nn.Module):
    """content_mlp (optionally per-plane) + shared q_proj + shared k_proj.

    Unlike column_graph, these fire ONLY on visited columns (roughly H current
    walker positions per token), not on all N columns. So the per-column work
    per token is dominated by sparse visited-column compute, not N·MLP_size.
    """

    def __init__(self, cfg: GraphWalkerConfig) -> None:
        super().__init__()
        D_s, D_id = cfg.D_s, cfg.D_id
        H, D_q = cfg.n_score_heads, cfg.D_q_per_head

        self.content_norm = _rmsnorm(D_s)
        self.per_plane_content = cfg.per_plane_content_mlp
        if cfg.per_plane_content_mlp:
            self.content_mlp = PerPlaneMLP(
                L=cfg.L, D_in=D_s + D_id,
                D_hid=cfg.ffn_mult_content * D_s, D_out=D_s,
            )
        elif cfg.content_mlp_depth > 1:
            # Deep shared MLP stack — residual FFN blocks. Scales params
            # to GPT-2-small class while keeping one batched matmul per
            # sublayer call (throughput-friendly).
            self.content_mlp = DeepContentMLP(
                D_in=D_s + D_id, D_s=D_s,
                D_hid=cfg.ffn_mult_content * D_s,
                n_layers=cfg.content_mlp_depth,
            )
        else:
            self.content_mlp = nn.Sequential(
                nn.Linear(D_s + D_id, cfg.ffn_mult_content * D_s),
                nn.GELU(),
                nn.Linear(cfg.ffn_mult_content * D_s, D_s),
            )
            nn.init.normal_(
                self.content_mlp[0].weight, mean=0.0, std=0.014,
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
        for module in (self.q_proj[0], self.k_proj[0]):
            nn.init.normal_(module.weight, mean=0.0, std=0.014)

        self.n_heads = H
        self.D_q = D_q


# =====================================================================
# GraphWalkerMemory
# =====================================================================


@dataclass
class WalkerReadout:
    """Return bundle from public step()."""
    motor: torch.Tensor                   # [B, D_model]
    motor_state: torch.Tensor             # [B, D_s]
    logits: torch.Tensor                  # [B, K_horizons, V]
    surprise_ema: torch.Tensor            # [B, K_horizons]
    visit_freq_step: torch.Tensor | None  # [N] fractional visits (if tracked)
    load_balance_loss: torch.Tensor       # scalar — aux loss term


@dataclass
class WalkerCoreReadout:
    """Return bundle from the hot persistent-walker core."""
    motor_state: torch.Tensor             # [B, D_s]
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
        # token_to_state keeps the graph hot path at D_s even when the lexical
        # model width is larger.
        self.token_to_state = nn.Linear(cfg.D_model, cfg.D_s, bias=False)
        nn.init.normal_(self.token_to_state.weight, mean=0.0, std=0.014)

        # q_in: [B, D_s] -> [B, H, D_q_in]
        self.input_q_proj = nn.Linear(
            cfg.D_s, cfg.n_heads * cfg.D_q_in, bias=False,
        )
        self.input_k_proj = nn.Linear(cfg.D_id, cfg.D_q_in, bias=False)
        self.input_v_proj = nn.Linear(cfg.D_model, cfg.D_s, bias=False)
        # Zero-init v so initial token-injection is no-op (state stays bounded).
        nn.init.zeros_(self.input_v_proj.weight)

        # Prev-token motor feeds into the start-col query (only): lets the
        # start position depend on recent output direction, not just the
        # current token's identity. Zero-init so day-1 behaviour is unchanged
        # and the model learns to use the chain through the routing gradient.
        # Autograd follows the chain within a TBPTT block; detach_state cuts
        # it at block boundaries.
        self.prev_motor_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        nn.init.zeros_(self.prev_motor_proj.weight)

        self._N_in = cfg.N_per_plane

        # Output readout: cross-attn over the H persistent walker endpoints.
        self.out_norm = _rmsnorm(cfg.D_s)
        self.motor_query = nn.Parameter(torch.randn(cfg.D_s) * 0.02)
        self.out_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.out_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.state_to_model = nn.Linear(cfg.D_s, cfg.D_model, bias=False)
        nn.init.normal_(self.state_to_model.weight, mean=0.0, std=0.014)

        # Multi-horizon readout (reused from column-graph)
        self.readout = MultiHorizonReadout(cfg)

        # --- Persistent state (buffers) ---
        self._state_initialized = False
        self.s: torch.Tensor                 # [B, N, D_s]
        self.prev_motor: torch.Tensor        # [B, D_s] — chained into next step
        self.walker_pos: torch.Tensor        # [B, H] long — persistent walker positions
        self.E_bias_flat: torch.Tensor       # [N*K] fp32
        self.surprise_ema: torch.Tensor       # [B, K_h]
        self.surprise_prev: torch.Tensor      # [B, K_h]
        self.surprise_motor_window: torch.Tensor  # [B, mod_period, D_s]
        self.surprise_token_window: torch.Tensor  # [B, mod_period]
        self.surprise_tail_motor: torch.Tensor    # [B, K_h-1, D_s]
        self.surprise_tail_len: int = 0
        self.tick_counter: int = 0

        # Window-accumulated stats for plasticity (reset every mod_period)
        self.co_visit_flat: torch.Tensor | None = None  # [N*K] fp32
        self.window_len: int = 0

        # Visit-frequency count across the current training step (for
        # load-balance aux loss). Reset externally per step.
        self.visit_count: torch.Tensor | None = None     # [N] fp32

        # Training / routing scheduling — caller sets these externally.
        self.training_step: int = 0

        # Per-block caches for values that depend only on Parameters (so
        # they are constant within a TBPTT block). Invalidated on
        # detach_state / begin_segment so gradient still flows correctly
        # to the underlying Parameters each backward pass.
        self._horizon_logits_cache: torch.Tensor | None = None
        self._alpha_cache: torch.Tensor | None = None              # [N]
        self._input_keys_cache: torch.Tensor | None = None         # [N_in, D_q_in]
        self._k_all_cache: torch.Tensor | None = None              # [N, H_score*D_q]

        # Filled when compile_step() is called; the hot graph core then routes
        # through the compiled version. Readout / surprise bookkeeping stays
        # outside that region so the compiled path only pays the true fast
        # recurrent work.
        self._compiled_step = None

    # -----------------------------------------------------------------
    # Block-level caches (static per forward within a TBPTT block)
    # -----------------------------------------------------------------

    def compile_step(self, mode: str = "default") -> None:
        """Compile the hot per-token graph core. Must be called after .cuda().

        Uses fullgraph=False so dynamo can graph-break around the Python
        bookkeeping that stays outside the compiled step. Also enables
        capture_dynamic_output_shape_ops so
        torch.unique / torch.bincount (data-dependent shapes) don't
        trigger their own graph breaks. On Triton 3.6 we avoid the
        TritonGPURemoveLayoutConversions randint-in-fusion crash by
        using rand+argmax for exploration sampling (see routing.py).
        """
        # Let unique/bincount stay in the compiled graph
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        # Let integer module attributes stay dynamic instead
        # of triggering a recompile every time they change.
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        # Room for a few recompile variants (requires_grad cycles, etc.)
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64,
        )
        self._compiled_step = torch.compile(
            self._step_core_impl, mode=mode, fullgraph=False,
        )

    @torch._dynamo.disable
    def _ensure_block_caches(self, unembedding: torch.Tensor) -> None:
        """Populate all per-block caches if not already set. Kept out of
        dynamo's compiled region so the `is None` checks don't trigger
        recompiles each time the cache status flips across a block.
        """
        if self._horizon_logits_cache is None:
            self._horizon_logits_cache = torch.matmul(
                self.readout.horizon_emb, unembedding.t(),
            )
        if self._alpha_cache is None:
            self._alpha_cache = torch.sigmoid(
                self.decay_proj(self.col_id),
            ).squeeze(-1)
        if self._input_keys_cache is None:
            self._input_keys_cache = self.input_k_proj(
                self.col_id[self.input_positions],
            )
        if self._k_all_cache is None:
            self._k_all_cache = self.cols.k_proj(self.col_id)

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def begin_segment(self, B: int, device: torch.device) -> None:
        """Reset working memory (per-document). E_bias persists."""
        cfg = self.cfg
        dtype_fast = self._fast_dtype(device)

        self.s = torch.zeros(B, cfg.N, cfg.D_s, device=device, dtype=dtype_fast)
        # prev_motor: last step's motor output, chained into next step's
        # start-col query. Starts at zero each segment.
        self.prev_motor = torch.zeros(B, cfg.D_s, device=device, dtype=dtype_fast)
        self.walker_pos = torch.zeros(
            B, cfg.n_heads, device=device, dtype=torch.long,
        )
        self.surprise_ema = torch.zeros(
            B, cfg.K_horizons, device=device, dtype=torch.float32,
        )
        self.surprise_prev = torch.zeros_like(self.surprise_ema)
        self.surprise_motor_window = torch.zeros(
            B, cfg.mod_period, cfg.D_s, device=device, dtype=dtype_fast,
        )
        self.surprise_token_window = torch.zeros(
            B, cfg.mod_period, device=device, dtype=torch.long,
        )
        tail_len = max(cfg.K_horizons - 1, 1)
        self.surprise_tail_motor = torch.zeros(
            B, tail_len, cfg.D_s, device=device, dtype=dtype_fast,
        )
        self.surprise_tail_len = 0
        self.tick_counter = 0
        self.window_len = 0

        # Invalidate block-level caches at segment boundary.
        self._horizon_logits_cache = None
        self._alpha_cache = None
        self._input_keys_cache = None
        self._k_all_cache = None

        # Lazy-init plastic state on first segment
        if not hasattr(self, "E_bias_flat") or self.E_bias_flat is None:
            self.reset_plastic_memory(device)

        # Fresh per-segment counters
        self.co_visit_flat = torch.zeros(
            cfg.num_edges, device=device, dtype=torch.float32,
        )
        self.visit_count = torch.zeros(cfg.N, device=device, dtype=torch.float32)

        # Per-segment scratch constants (recomputed every token in the old
        # hot path; now built once per segment). BH stays fixed within a
        # segment, so these are safe to cache.
        H = cfg.n_heads
        self._batch_idx = (
            torch.arange(B, device=device).repeat_interleave(H)
        )                                                             # [B*H]
        self._k_range = torch.arange(cfg.K, device=device)            # [K]
        self._ones_bh = torch.ones(B * H, device=device, dtype=torch.float32)

        self._state_initialized = True

    def reset_plastic_memory(self, device: torch.device) -> None:
        """Hard reset of long-term plastic E_bias."""
        self.E_bias_flat = torch.zeros(
            self.cfg.num_edges, device=device, dtype=torch.float32,
        )

    def detach_state(self) -> None:
        """TBPTT boundary: preserve values, sever gradient graph."""
        if not self._state_initialized:
            return
        self.s = self.s.detach()
        self.prev_motor = self.prev_motor.detach()
        self.E_bias_flat = self.E_bias_flat.detach()
        self.walker_pos = self.walker_pos.detach()
        self.surprise_ema = self.surprise_ema.detach()
        self.surprise_prev = self.surprise_prev.detach()
        self.surprise_motor_window = self.surprise_motor_window.detach()
        self.surprise_token_window = self.surprise_token_window.detach()
        self.surprise_tail_motor = self.surprise_tail_motor.detach()
        if self.co_visit_flat is not None:
            self.co_visit_flat = self.co_visit_flat.detach()
        if self.visit_count is not None:
            self.visit_count = self.visit_count.detach()
        # Invalidate block caches — next step rebuilds them with fresh
        # autograd refs tied to the new block's forward computation.
        self._horizon_logits_cache = None
        self._alpha_cache = None
        self._input_keys_cache = None
        self._k_all_cache = None

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

    def _schedule_tensors(self, token_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        tau = torch.tensor(
            gumbel_schedule(
                self.training_step, cfg.gumbel_tau_start, cfg.gumbel_tau_end,
                cfg.gumbel_anneal_steps,
            ),
            device=token_id.device, dtype=torch.float32,
        )
        epsilon = torch.tensor(
            gumbel_schedule(
                self.training_step, cfg.epsilon_start, cfg.epsilon_end,
                cfg.epsilon_anneal_steps,
            ),
            device=token_id.device, dtype=torch.float32,
        )
        return tau, epsilon

    def step_core(self, token_id: torch.Tensor) -> WalkerCoreReadout:
        """Hot persistent-walker core: sparse graph dynamics only."""
        self._ensure_block_caches(self.tied_token_emb.weight)
        tau, epsilon = self._schedule_tensors(token_id)
        if self._compiled_step is not None:
            return self._compiled_step(token_id, tau, epsilon)
        return self._step_core_impl(token_id, tau, epsilon)

    def step(self, token_id: torch.Tensor) -> WalkerReadout:
        """Public single-token API: graph core + immediate readout.

        Training should prefer `step_core()` plus blockwise readout so the
        large model-space stack stays off the token clock.
        """
        core = self.step_core(token_id)
        motor = self.state_to_model(core.motor_state)
        logits = self.readout(
            motor, self.tied_token_emb.weight,
            horizon_logits=self._horizon_logits_cache,
        )
        self._record_surprise_token(token_id, core.motor_state.detach())
        self._maybe_finalize_surprise_and_plasticity()
        return WalkerReadout(
            motor=motor,
            motor_state=core.motor_state,
            logits=logits,
            surprise_ema=self.surprise_ema,
            visit_freq_step=core.visit_freq_step,
            load_balance_loss=core.load_balance_loss,
        )

    def _step_core_impl(
        self, token_id: torch.Tensor, tau: torch.Tensor, epsilon: torch.Tensor,
    ) -> WalkerCoreReadout:
        """One token of sparse graph dynamics with persistent walkers.

        This runs only the graph-side work that is truly on the token clock:
        token anchoring on the input plane, one hop of persistent-walker
        movement, sparse state update, and graph-state readout.
        """
        assert self._state_initialized, "call begin_segment() first"
        cfg = self.cfg
        B = self.s.shape[0]
        device = self.s.device

        is_training = self.training
        cfg_n_heads = cfg.n_heads
        BH = B * cfg_n_heads
        state_dtype = self.s.dtype

        # 1. Embed token in model space, then project into graph-state space.
        h_input_model = self.tied_token_emb(token_id)                 # [B, D_model]
        h_input = self.token_to_state(h_input_model)                  # [B, D_s]

        # 2. Token-conditioned anchoring on the input plane.
        query_input = h_input + self.prev_motor_proj(
            self.prev_motor.to(h_input.dtype),
        )
        query_flat = self.input_q_proj(query_input).view(
            B, cfg_n_heads, cfg.D_q_in,
        )
        input_keys = self._input_keys_cache
        scale_in = 1.0 / (cfg.D_q_in ** 0.5)
        scores_in = torch.einsum("bhd,nd->bhn", query_flat, input_keys) * scale_in

        scores_in_flat = scores_in.reshape(BH, self._N_in)
        rout_in = gumbel_top1_softmax(
            scores_in_flat, tau=tau, epsilon=epsilon, training=is_training,
        )
        start_local = rout_in.selected_idx.view(B, cfg_n_heads)
        start_cols = self.input_positions[start_local]                # [B, H]

        # Input routing still happens each token, but the walkers now persist
        # through graph space instead of relaunching an H×L walk from scratch.
        cur_bh = (
            start_cols.reshape(BH)
            if self.tick_counter == 0
            else self.walker_pos.reshape(BH)
        )

        # Injection anchors current token content into graph state.
        v_inject = self.input_v_proj(h_input_model)                   # [B, D_s]
        alpha = self._alpha_cache                                     # [N]
        input_ste_weights = torch.gather(
            rout_in.ste_weights, 1, rout_in.selected_idx.unsqueeze(1),
        ).squeeze(1)                                                  # [B*H]

        # Load-balance accounting.
        visit_count_step = torch.zeros(cfg.N, device=device, dtype=torch.float32)
        lb_dtype = torch.float32
        P_mass = torch.zeros(cfg.N, device=device, dtype=lb_dtype)
        p_in_per_col = rout_in.soft_probs.to(lb_dtype).sum(dim=0)     # [N_in]
        P_mass = P_mass.index_add(0, self.input_positions, p_in_per_col)

        # 3. Advance each persistent walker by one hop.
        cur = cur_bh.view(B, cfg_n_heads)
        s_cur = torch.gather(
            self.s, 1, cur.unsqueeze(-1).expand(B, cfg_n_heads, cfg.D_s),
        ).reshape(BH, cfg.D_s)
        id_cur = self.col_id[cur_bh]
        cat_content = torch.cat([
            self.cols.content_norm(s_cur).to(state_dtype),
            id_cur.to(state_dtype),
        ], dim=-1)
        if self.cols.per_plane_content:
            plane_of_cur = self.plane_ids[cur_bh]
            m_out = self.cols.content_mlp(cat_content, plane_of_cur)
        else:
            m_out = self.cols.content_mlp(cat_content)

        q = self.cols.q_proj(cat_content).view(
            BH, self.cols.n_heads, self.cols.D_q,
        )
        nbrs_of_cur = self.out_nbrs[cur_bh]                           # [B*H, K]
        k_all = self._k_all_cache
        k_nbrs = k_all[nbrs_of_cur].view(
            BH, cfg.K, self.cols.n_heads, self.cols.D_q,
        )
        scale = 1.0 / (self.cols.D_q ** 0.5)
        scores = torch.einsum("bhd,bkhd->bk", q, k_nbrs) * scale

        edge_flat = cur_bh.unsqueeze(1) * cfg.K + self._k_range.unsqueeze(0)
        E_vals = self.E_bias_flat[edge_flat].to(scores.dtype)
        scores = scores + E_vals

        rout = gumbel_top1_softmax(
            scores, tau=tau, epsilon=epsilon, training=is_training,
        )
        next_local = rout.selected_idx                                # [B*H]
        next_col = torch.gather(
            nbrs_of_cur, 1, next_local.unsqueeze(-1),
        ).squeeze(-1)                                                 # [B*H]
        chosen_weight = torch.gather(
            rout.ste_weights, 1, next_local.unsqueeze(1),
        ).squeeze(1)                                                  # [B*H]
        edge_taken_flat = cur_bh * cfg.K + next_local                 # [B*H]

        P_mass = P_mass.index_add(
            0, nbrs_of_cur.reshape(-1),
            rout.soft_probs.to(lb_dtype).reshape(-1),
        )

        visit_count_step.scatter_add_(0, start_cols.reshape(-1), self._ones_bh)
        visit_count_step.scatter_add_(0, next_col, self._ones_bh)
        if self.visit_count is not None:
            with torch.no_grad():
                self.visit_count = self.visit_count + visit_count_step

        # 4. Sparse aggregation and state update at only the touched rows.
        batch_idx = self._batch_idx
        inject_msg = (
            v_inject.unsqueeze(1).expand(B, cfg_n_heads, cfg.D_s)
            .reshape(BH, cfg.D_s)
        )
        inject_msg = inject_msg * input_ste_weights.unsqueeze(-1).to(inject_msg.dtype)
        next_msg = m_out * chosen_weight.unsqueeze(-1).to(m_out.dtype)

        all_dests = torch.cat([
            batch_idx * cfg.N + start_cols.reshape(BH),
            batch_idx * cfg.N + next_col,
        ], dim=0).contiguous()
        all_msgs = torch.cat(
            [inject_msg, next_msg], dim=0,
        ).to(state_dtype).contiguous()

        # Fused Triton op: segment-sum messages into unique destinations,
        # gather s_old, LIF-blend with alpha, scatter back in place.
        # Operates only on touched rows (O(U)) rather than the full
        # [B*N, D_s] state. Returns s_flat aliased to self.s's storage.
        s_flat = self.s.view(B * cfg.N, cfg.D_s)
        s_flat = sparse_lif_update(s_flat, all_msgs, all_dests, alpha, cfg.N)
        self.s = s_flat.view(B, cfg.N, cfg.D_s)

        # 5. Read out only from the new walker endpoints.
        self.walker_pos = next_col.view(B, cfg_n_heads)
        end_states = torch.gather(
            self.s, 1, self.walker_pos.unsqueeze(-1).expand(B, -1, cfg.D_s),
        )                                                              # [B, H, D_s]
        s_traj = self.out_norm(end_states).to(state_dtype)
        k = self.out_k_proj(s_traj)
        v = self.out_v_proj(s_traj)
        q_motor = self.motor_query.to(state_dtype).unsqueeze(0).expand(B, -1)
        scale_out = 1.0 / (cfg.D_s ** 0.5)
        attn_scores = torch.sum(k * q_motor.unsqueeze(1), dim=-1) * scale_out
        attn = F.softmax(attn_scores, dim=-1)
        motor_state = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        self.prev_motor = motor_state

        # 6. Plasticity-window bookkeeping stays on the fast clock only for
        # sparse edge counts. Exact lexical surprise is deferred to the
        # plasticity clock.
        with torch.no_grad():
            self.co_visit_flat = self.co_visit_flat + torch.bincount(
                edge_taken_flat, minlength=cfg.num_edges,
            ).to(torch.float32)
        self.window_len += 1
        self.tick_counter += 1

        n_decisions = float(BH * 2)
        P_mean = P_mass / n_decisions
        f_mean = (visit_count_step / n_decisions).detach()
        load_balance_loss = cfg.N * (P_mean * f_mean).sum()

        return WalkerCoreReadout(
            motor_state=motor_state,
            visit_freq_step=f_mean if cfg.lambda_balance > 0 else None,
            load_balance_loss=load_balance_loss,
        )

    def readout_from_state_block(self, motor_state: torch.Tensor) -> torch.Tensor:
        """Project graph-state readouts into model space and logits in batch."""
        self._ensure_block_caches(self.tied_token_emb.weight)
        if not torch.is_autocast_enabled():
            motor_state = motor_state.to(self.state_to_model.weight.dtype)
        motor = self.state_to_model(motor_state)
        return self.readout(
            motor, self.tied_token_emb.weight,
            horizon_logits=self._horizon_logits_cache,
        )

    # -----------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    def _record_surprise_token(
        self, token_id: torch.Tensor, motor_state: torch.Tensor,
    ) -> None:
        idx = self.window_len - 1
        self.surprise_motor_window[:, idx] = motor_state
        self.surprise_token_window[:, idx] = token_id.detach()

    @torch._dynamo.disable
    def _finalize_surprise_window(self) -> None:
        """Compute exact multi-horizon surprise only on the plasticity clock."""
        if self.window_len == 0:
            return

        cfg = self.cfg
        B = self.s.shape[0]
        W = self.window_len
        tail_len = min(self.surprise_tail_len, cfg.K_horizons - 1)

        if tail_len > 0:
            motor_all = torch.cat([
                self.surprise_tail_motor[:, :tail_len],
                self.surprise_motor_window[:, :W],
            ], dim=1)
        else:
            motor_all = self.surprise_motor_window[:, :W]

        with torch.autocast(device_type=self.s.device.type, enabled=False):
            logits_all = self.readout_from_state_block(motor_all).float()   # [B, tail+W, K_h, V]
            new_ema = self.surprise_ema.float().clone()
            self.surprise_prev = self.surprise_ema.detach().clone()
            # Vectorized per-horizon closed-form EMA.
            # Old path: O(W × K_h) Python iterations with per-step
            # `logits_all[:, idx-k, k-1]` selects + F.cross_entropy calls,
            # each backward-allocating a full-shape gradient zeros tensor.
            # New path: K_h slice-views + K_h CE calls + closed-form EMA
            # recurrence per horizon. 128× fewer autograd nodes.
            alpha_g = cfg.alpha_gamma_s
            device = self.s.device
            V_vocab = logits_all.shape[-1]
            for k in range(1, cfg.K_horizons + 1):
                # Valid i range for this k: i such that (tail_len + i - k) >= 0.
                # Equivalently, i >= max(0, k - tail_len). Always contiguous
                # from that starting point through W-1.
                i_first_valid = max(0, k - tail_len)
                if i_first_valid >= W:
                    continue
                # Slice past logits: position p = tail_len + i - k for valid i.
                # p ranges over [tail_len + i_first_valid - k, tail_len + W - 1 - k).
                p_start = tail_len + i_first_valid - k
                valid_count = W - i_first_valid
                past_slice = logits_all[:, p_start : p_start + valid_count, k - 1, :]
                target_slice = self.surprise_token_window[:, i_first_valid:W]
                ce_k = F.cross_entropy(
                    past_slice.reshape(-1, V_vocab),
                    target_slice.reshape(-1),
                    reduction="none",
                ).reshape(past_slice.shape[0], valid_count)
                # Closed-form EMA: final = (1-α)^V * init + α Σ (1-α)^(V-1-s) ce[s]
                weights = (1.0 - alpha_g) ** torch.arange(
                    valid_count - 1, -1, -1, device=device, dtype=torch.float32,
                )
                weighted_sum = (ce_k * weights.unsqueeze(0)).sum(dim=1)
                decay = (1.0 - alpha_g) ** valid_count
                new_ema[:, k - 1] = (
                    decay * new_ema[:, k - 1] + alpha_g * weighted_sum
                )
            self.surprise_ema = new_ema

        tail_keep = min(cfg.K_horizons - 1, motor_all.shape[1])
        if tail_keep > 0:
            self.surprise_tail_motor[:, :tail_keep] = motor_all[:, -tail_keep:].detach()
        self.surprise_tail_len = tail_keep

    @torch._dynamo.disable
    def _maybe_finalize_surprise_and_plasticity(self) -> None:
        if self.window_len < self.cfg.mod_period:
            return
        self._finalize_surprise_window()
        self._plasticity_step()

    @torch._dynamo.disable
    def _plasticity_step(self) -> None:
        """Hebbian on traversed-edge counts. Fires every mod_period ticks."""
        cfg = self.cfg
        device = self.s.device

        with torch.autocast(device_type=device.type, enabled=False):
            # Normalize by window length
            window = max(self.window_len, 1)
            co_visit_norm = (self.co_visit_flat.float() / window) if self.co_visit_flat is not None \
                else torch.zeros_like(self.E_bias_flat)

            # Scalar-eta Hebbian v1: surprise-gated global learning rate.
            # Higher surprise → faster learning. A trunk+head neuromod MLP
            # could replace this with per-column/per-head rates later; for
            # now a single scalar is sufficient and keeps the hot path cheap.
            surprise_scalar = self.surprise_ema.mean().float()
            eta_global = cfg.plast_eta * torch.sigmoid(
                surprise_scalar - cfg.plast_surprise_bias,
            )

            # Global step: E_bias += η * (co_visit - decay * E_bias)
            delta = eta_global * (co_visit_norm - cfg.plast_decay * self.E_bias_flat)
            self.E_bias_flat = (self.E_bias_flat + delta).clamp(
                -cfg.E_bias_max, cfg.E_bias_max,
            )

        # Reset window counters
        self.co_visit_flat = torch.zeros_like(self.co_visit_flat)
        self.window_len = 0
