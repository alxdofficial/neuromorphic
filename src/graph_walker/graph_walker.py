"""GraphWalkerMemory — trajectory-routed plastic concept graph.

Hot path per token (write-first-then-route, no anchoring — the walker
just walks continuously):
  1. Walker at its current column emits a message from (col_state,
     col_id, walker_state, token_embed) → content_mlp.
  2. Sparse LIF deposit: walker writes m_out at its CURRENT column.
  3. Re-read updated col state at cur_bh (walker's own contribution is now
     in it). Steering query uses (s_cur_new, col_id, walker_state,
     token_embed). Score K out-edges, Gumbel top-1 → next_col.
  4. Endpoint readout = s_cur_new + Σ_k ste[k] · nbr_id_to_s(col_id[nbrs[k]]).
     First term carries gradient to content_mlp / walker_state / token;
     second term is the STE bridge from routing → loss.
  5. Cross-attn over H endpoints → motor_state.
  6. Walker state update: w_new = σ(α_h) · w_old + (1-σ(α_h)) · m_out.

Every mod_period tokens the TBPTT block closes (enforced alignment
`tbptt_block == mod_period`): flush computes per-horizon CE via the
factorized path (no [B,T,K_h,V] broadcast), streams it into
`surprise_ema`, fires Hebbian + neuromod updates on `E_bias_flat`, then
backprops + detaches.

See docs/graph_walker.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.neuromod import (
    NeuromodGraphTransformer,
    build_adjacency_bias,
    enumerate_touched_edges,
)
from src.graph_walker.readout import MultiHorizonReadout, _FallbackRMSNorm
from src.graph_walker.routing import (
    StepRoutingChoices,
    gumbel_schedule,
    gumbel_top1_softmax,
    route_or_replay,
)
from src.graph_walker.topology import build_topology
from src.graph_walker.triton.lif import sparse_lif_update


class _NullCtx:
    """Minimal no-op context for the autocast-defensive wrap in walk_segment."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


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


class PerHeadDeepContentMLP(nn.Module):
    """Per-head content MLP: H independent weight sets, one per walker index.

    Each walker h uses its own (in_proj_h, ResidualFFN_h) — H disjoint sets
    of parameters. Per-walker compute is identical to shared-weight
    DeepContentMLP; only the total parameter count scales with H. This is
    MoE-style capacity gain at constant per-token FLOPs.

    Batched across H via einsum so we do one cuBLAS call per matmul, not H.
    Each matmul has batch dim H, row dim B (not B·H), so M drops from
    `B·H` to `B` per matmul. For small H this costs Tensor Core utilization;
    for H=32 with D_hid=2048 the per-matmul workload is still large enough
    to amortize launch overhead.
    """

    def __init__(
        self, H: int, D_in: int, D_s: int, D_hid: int, n_layers: int,
    ) -> None:
        super().__init__()
        self.H = H
        self.D_s = D_s
        self.D_hid = D_hid
        self.n_layers = n_layers

        # in_proj: [H, D_in, D_s], b: [H, D_s]
        self.in_proj_w = nn.Parameter(torch.empty(H, D_in, D_s))
        self.in_proj_b = nn.Parameter(torch.zeros(H, D_s))

        # Per-head ResidualFFN block tensors stacked along H and n_layers.
        # norm weight (RMSNorm gain): [H, n_layers, D_s]
        self.norm_w = nn.Parameter(torch.ones(H, n_layers, D_s))
        # up: [H, n_layers, D_s, D_hid]; bias: [H, n_layers, D_hid]
        self.up_w = nn.Parameter(torch.empty(H, n_layers, D_s, D_hid))
        self.up_b = nn.Parameter(torch.zeros(H, n_layers, D_hid))
        # down: [H, n_layers, D_hid, D_s]; bias: [H, n_layers, D_s]
        self.down_w = nn.Parameter(torch.empty(H, n_layers, D_hid, D_s))
        self.down_b = nn.Parameter(torch.zeros(H, n_layers, D_s))

        nn.init.normal_(self.in_proj_w, mean=0.0, std=0.014)
        nn.init.normal_(self.up_w, mean=0.0, std=0.014)
        # Depth-scaled init on final projection so sum of residuals stays bounded.
        nn.init.normal_(
            self.down_w, mean=0.0, std=(2.0 / D_hid) ** 0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*H, D_in] (flat, from the rest of the hot path).
        Returns [B*H, D_s]."""
        BH = x.shape[0]
        assert BH % self.H == 0, (
            f"input first dim {BH} not divisible by H={self.H}"
        )
        B = BH // self.H
        x = x.view(B, self.H, -1)                                  # [B, H, D_in]

        # in_proj: [B, H, D_in] × [H, D_in, D_s] → [B, H, D_s]
        x = torch.einsum("bhd,hde->bhe", x, self.in_proj_w)
        x = x + self.in_proj_b.unsqueeze(0)                        # broadcast over B

        for i in range(self.n_layers):
            # Per-head RMSNorm (scale-invariant so compute in input dtype).
            rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            x_n = x * rms * self.norm_w[:, i].to(x.dtype).unsqueeze(0)
            # up: [B, H, D_s] × [H, D_s, D_hid] → [B, H, D_hid]
            hid = torch.einsum("bhd,hde->bhe", x_n, self.up_w[:, i])
            hid = hid + self.up_b[:, i].unsqueeze(0)
            hid = F.gelu(hid)
            # down: [B, H, D_hid] × [H, D_hid, D_s] → [B, H, D_s]
            out = torch.einsum("bhd,hde->bhe", hid, self.down_w[:, i])
            out = out + self.down_b[:, i].unsqueeze(0)
            x = x + out

        return x.reshape(BH, self.D_s)


class ColumnCompute(nn.Module):
    """content_mlp (optionally per-plane) + shared q_proj + shared k_proj.

    Unlike column_graph, these fire ONLY on visited columns (roughly H current
    walker positions per token), not on all N columns. So the per-column work
    per token is dominated by sparse visited-column compute, not N·MLP_size.

    Steering input (fed to both content_mlp and q_proj) is:
        cat(s_cur, id_cur, walker_state, token_embed)
    with dims (D_s, D_id, D_s, D_s) → D_steer = 3·D_s + D_id. This unifies
    "where should I go" as a function of column state, column identity,
    walker's own running state, and the current token's content. Gradient
    through the routing decision flows back into all four in one hop.
    """

    def __init__(self, cfg: GraphWalkerConfig) -> None:
        super().__init__()
        D_s, D_id = cfg.D_s, cfg.D_id
        H, D_q = cfg.n_score_heads, cfg.D_q_per_head
        D_steer = 3 * D_s + D_id

        self.content_norm = _rmsnorm(D_s)
        # Decoupled content_mlp hidden width: explicit cfg.D_hid_content when
        # set, else legacy ffn_mult_content × D_s. Pinning D_hid_content to
        # a fixed value (e.g. 1024) makes content_mlp cost O(D_s · D_hid)
        # instead of O(D_s²); see config.py for rationale.
        D_hid_content = (
            cfg.D_hid_content
            if cfg.D_hid_content is not None
            else cfg.ffn_mult_content * D_s
        )
        if cfg.per_head_content_mlp:
            # MoE-style: H independent content_mlp copies, one per walker
            # index. Total params scale with H while per-token compute is
            # identical to the shared path (each walker still runs one
            # forward pass, just with its own weights).
            self.content_mlp = PerHeadDeepContentMLP(
                H=cfg.n_heads,
                D_in=D_steer, D_s=D_s,
                D_hid=D_hid_content,
                n_layers=cfg.content_mlp_depth,
            )
        else:
            # Deep shared MLP stack — residual FFN blocks. Handles both
            # shallow (depth=1) and deeper cases uniformly; keeps the
            # RMSNorm + residual + GELU structure.
            self.content_mlp = DeepContentMLP(
                D_in=D_steer, D_s=D_s,
                D_hid=D_hid_content,
                n_layers=cfg.content_mlp_depth,
            )
        self.q_proj = nn.Sequential(
            nn.Linear(D_steer, 2 * H * D_q),
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


@dataclass
class WalkerStepOutput:
    """Return bundle from the pure step-core compute.

    Pure variant: all outputs are returned explicitly instead of being
    written to self.* so the compute can be wrapped in
    torch.utils.checkpoint.checkpoint, whose forward-recompute requires
    side-effect-free functions. The step_core wrapper applies the returned
    state and accumulators to self.*.
    """
    s_new: torch.Tensor                   # [B, N, D_s]
    walker_pos_new: torch.Tensor          # [B, H]
    walker_state_new: torch.Tensor        # [B, H, D_s]
    motor_state: torch.Tensor             # [B, D_s]
    co_visit_delta: torch.Tensor          # [N*K] fp32 — add to self.co_visit_flat
    visit_count_delta: torch.Tensor       # [N] fp32 — add to self.visit_count
    visit_freq_step: torch.Tensor | None  # [N] fractional visits (optional)
    load_balance_loss: torch.Tensor       # scalar — aux loss term
    log_pi_step: torch.Tensor | None      # [B] phase-2 log π over routing
                                          # decisions made this step,
                                          # summed over walker heads.
    routing_choices: StepRoutingChoices | None = None
    # Captured `selected_idx` per routing call this step. Populated when
    # the step samples (any phase); None when this step replayed pre-saved
    # choices via `replay_choices=...`. Used by phase-2 GRPO to record a
    # rollout trace under no-grad sampling, which a later teacher-forced
    # replay re-runs with grad enabled to compute per-action log-π × A.
                                          # None outside phase 2.


@dataclass
class WalkBlockOutput:
    """Return bundle from `walk_block`: T_block sequential steps.

    All forward state (s, walker_pos, walker_state) is returned
    explicitly so the caller can thread it through autograd or write it
    back to self. Non-grad accumulators (co_visit, visit_count) are
    summed within the block; caller adds them to module-level totals.
    """
    s_new: torch.Tensor                  # [B, N, D_s]
    walker_pos_new: torch.Tensor         # [B, H]
    walker_state_new: torch.Tensor       # [B, H, D_s]
    motor_states_bt: torch.Tensor        # [B, T_block, D_s] — fed to readout
    co_visit_total: torch.Tensor         # [N*K] fp32, summed across T_block
    visit_count_total: torch.Tensor      # [N] fp32, summed across T_block
    load_balance_loss: torch.Tensor      # scalar — Σ over T_block, mean later


class GraphWalkerMemory(nn.Module):
    def __init__(
        self, cfg: GraphWalkerConfig, tied_token_emb: nn.Embedding
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tied_token_emb = tied_token_emb

        # Topology — single flat substrate of N columns
        topo = build_topology(
            grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols,
            K=cfg.K, p_rewire=cfg.p_rewire,
            seed=cfg.topology_seed, radius=cfg.radius,
        )
        self.register_buffer("out_nbrs", topo.out_nbrs, persistent=False)
        self.register_buffer("edge_src", topo.edge_src, persistent=False)
        self.register_buffer("edge_dst", topo.edge_dst, persistent=False)

        # Column identity
        torch.manual_seed(cfg.init_seed)
        self.col_id = nn.Parameter(torch.randn(cfg.N, cfg.D_id) * 0.02)

        # Per-column decay α = σ(decay_proj(id))
        self.decay_proj = nn.Linear(cfg.D_id, 1)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.zeros_(self.decay_proj.bias)

        # Shared column compute
        self.cols = ColumnCompute(cfg)

        # token_to_state keeps the graph hot path at D_s even when the lexical
        # model width is larger. Used only by the standalone walker (token-id
        # driven path); the integration feeds h_mem in D_s directly.
        self.token_to_state = nn.Linear(cfg.D_model, cfg.D_s, bias=False)
        nn.init.normal_(self.token_to_state.weight, mean=0.0, std=0.014)

        # Walker state: each of H walkers carries a persistent D_s-dim state
        # across tokens (separate from the column state it happens to sit
        # on). Updated as an EMA of the walker's own content_mlp output:
        #     w_new = σ(α_h) · w_old + (1 − σ(α_h)) · m_out
        # α_h is per-walker learnable; init so σ(α_h) ≈ 0.9 (slow decay).
        # Walker state feeds back into the steering input, so walkers carry
        # their own running summary of where they've been — the graph's
        # column states are the shared memory, walker_state is the private
        # memory.
        self.walker_state_alpha = nn.Parameter(
            torch.full((cfg.n_heads,), 2.2)
        )

        # Output readout: cross-attn over the H persistent walker endpoints.
        self.out_norm = _rmsnorm(cfg.D_s)
        self.motor_query = nn.Parameter(torch.randn(cfg.D_s) * 0.02)
        self.out_k_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.out_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        self.state_to_model = nn.Linear(cfg.D_s, cfg.D_model, bias=False)
        nn.init.normal_(self.state_to_model.weight, mean=0.0, std=0.014)

        # Neighbor-id projection used in the endpoint readout to carry the
        # routing straight-through gradient. end_state_h adds a term
        # Σ_k ste_weights[k] · nbr_id_to_s(col_id[nbrs[cur_bh,k]]), which in
        # forward equals nbr_id_to_s(col_id[next_col]) (ste is one-hot) and
        # in backward differentiates through ste → scores → q/k/E_bias.
        # Small-random init so the dot product carries gradient from step 0;
        # zero-init would make ∂loss/∂ste[k] ≡ 0 and dead-product routing.
        self.nbr_id_to_s = nn.Linear(cfg.D_id, cfg.D_s, bias=False)
        nn.init.normal_(self.nbr_id_to_s.weight, mean=0.0, std=0.02)

        # Multi-horizon readout (reused from column-graph)
        self.readout = MultiHorizonReadout(cfg)

        # Optional graph-transformer neuromodulator (see src/graph_walker/neuromod.py).
        # Disabled by default so existing configs and tests are unchanged.
        # When enabled, acts at the start of each plasticity window on a
        # detached snapshot of the previous window's touched-column stats.
        #
        # Per-col features: state + id + visit_count, plus a scalar surprise
        # broadcast in neuromod_only mode (so neuromod conditions plastic
        # updates on how surprised the LM was).
        # Per-edge features: in neuromod_only mode, log(co_visit+1) and
        # E_bias_old at the edge feed the edge MLP — neuromod sees the
        # Hebbian-flavored stats it now replaces.
        if cfg.use_neuromod:
            extra_col_feats = (
                1 if cfg.plasticity_mode == "neuromod_only" else 0
            )
            D_feat = cfg.D_s + cfg.D_id + 1 + extra_col_feats
            D_per_edge_extra = (
                2 if cfg.plasticity_mode == "neuromod_only" else 0
            )  # log(co_visit+1) + E_bias_old
            self.neuromod = NeuromodGraphTransformer(
                D_feat=D_feat,
                D_mod=cfg.neuromod_D_mod,
                n_layers=cfg.neuromod_n_layers,
                n_heads=cfg.neuromod_n_heads,
                edge_hidden=cfg.neuromod_edge_hidden,
                E_bias_max=cfg.E_bias_max,
                D_per_edge_extra=D_per_edge_extra,
            )
        else:
            self.neuromod = None

        # --- Persistent state (buffers) ---
        self._state_initialized = False
        self.s: torch.Tensor                 # [B, N, D_s]
        self.walker_pos: torch.Tensor        # [B, H] long — persistent walker positions
        self.walker_state: torch.Tensor      # [B, H, D_s] — walker's private running state
        # E_bias_flat is the long-term plastic state. Registered as a buffer
        # so it appears in state_dict() (survives checkpoint/resume) and
        # follows .to(device) / .cuda() moves with the module.
        self.register_buffer(
            "E_bias_flat",
            torch.zeros(cfg.num_edges, dtype=torch.float32),
            persistent=True,
        )
        self.surprise_ema: torch.Tensor       # [B, K_h]
        self.surprise_prev: torch.Tensor      # [B, K_h] — snapshot at window close
        self.tick_counter: int = 0

        # Window-accumulated stats for plasticity (reset every mod_period)
        self.co_visit_flat: torch.Tensor | None = None  # [N*K] fp32
        self.window_len: int = 0

        # Neuromod plastic-window state. `_active_neuromod_delta` is the grad-
        # carrying E_bias delta for the CURRENT plasticity window; it is
        # computed at window start from the snapshot of the PREVIOUS window,
        # detached and merged into E_bias_flat at window close. None when
        # there is no previous snapshot (bootstrapping) or neuromod off.
        self._active_neuromod_delta: torch.Tensor | None = None
        # Snapshot of the most recently closed window's touched columns —
        # consumed at the next window's start to produce _active_neuromod_delta.
        self._neuromod_input_ids: torch.Tensor | None = None
        self._neuromod_input_feats: torch.Tensor | None = None
        # Per-edge co_visit snapshot from the just-closed window. Captured
        # in `_snapshot_touched_columns` BEFORE `_plasticity_step` resets
        # `co_visit_flat`, so the next window's neuromod sees the activity
        # pattern that produced the snapshot. Only populated in
        # neuromod_only mode (legacy mode reads co_visit live in the
        # Hebbian path before the reset).
        self._neuromod_input_co_visit_flat: torch.Tensor | None = None

        # Visit-frequency count across the current training step (for
        # load-balance aux loss). Reset externally per step.
        self.visit_count: torch.Tensor | None = None     # [N] fp32

        # Training / routing scheduling — caller sets these externally.
        self.training_step: int = 0

        # Phase indicator for routing. "phase1" = Gumbel-soft + STE (backprop);
        # "phase2" = hard Categorical sampling + log_pi (REINFORCE / GRPO).
        # Set externally by `wrapper.current_phase` for pretrained training.
        self.phase: str = "phase1"
        # Accumulated log π over routing decisions in the current segment.
        # Reset at `begin_segment`. Read by GRPO via `consume_log_pi_mean()`
        # (returns the sum divided by the count of accumulated steps,
        # which is the proper normalization for REINFORCE — without it,
        # log_pi_sum scales with trajectory length and produces gradient
        # magnitudes thousands of times larger than necessary).
        # None when no phase-2 decisions have been recorded.
        # DeepSeek-style routing-trace capture buffer. None = not armed
        # (capture disabled). When armed via `start_capturing_routes()`,
        # `_writeback_step_state` appends each step's `StepRoutingChoices` to
        # this list. The list is consumed (and cleared) by
        # `consume_routing_trace()`.
        self._captured_routes: list | None = None
        # DeepSeek-style replay stash. When non-None, the next
        # `walk_segment` consumes it as the per-step routing trace
        # (length must equal segment length) and runs the per-token
        # path with `replay_choices` per step. Cleared after one
        # walk_segment call. Set via `arm_replay_trace(trace)`.
        self._next_replay_trace: list | None = None
        self._log_pi_sum: torch.Tensor | None = None
        # Companion counter for the running sum: the number of `_writeback_step_state`
        # calls that have contributed a non-None `log_pi_step`. Each
        # contribution is itself a sum across H walkers and across (anchor +
        # n_hops) routing decisions per step, but for normalization purposes
        # we use step-count as the denominator — fixed-topology runs have a
        # constant ratio of decisions-per-step, so step-count and
        # decision-count differ only by a constant scale that's absorbed
        # into the learning rate.
        self._log_pi_count: int = 0

        # Per-block caches for values that depend only on Parameters (so
        # they are constant within a TBPTT block). Invalidated on
        # detach_state / begin_segment so gradient still flows correctly
        # to the underlying Parameters each backward pass.
        self._horizon_logits_cache: torch.Tensor | None = None
        self._alpha_cache: torch.Tensor | None = None              # [N]
        self._k_all_cache: torch.Tensor | None = None              # [N, H_score*D_q]

        # Filled when compile_step() is called; the hot graph core then routes
        # through the compiled version. Readout / surprise bookkeeping stays
        # outside that region so the compiled path only pays the true fast
        # recurrent work.
        self._compiled_step = None

        # Filled when compile_walk_block_from_h() is called; walk_segment
        # routes a whole tbptt block through the compiled version when set,
        # replacing T_block per-token walker_step_from_h calls.
        self._compiled_walk_block_from_h = None

        # Filled when compile_walk_block() is called (standalone path that drives
        # walk_block with token ids). Initialized here so callers can
        # check `lm.memory._compiled_walk_block is not None` without an
        # AttributeError when compile_walk_block was never invoked.
        self._compiled_walk_block = None

    # -----------------------------------------------------------------
    # Block-level caches (static per forward within a TBPTT block)
    # -----------------------------------------------------------------

    def compile_step(
        self, mode: str = "default", *, dynamic: bool | None = False,
    ) -> None:
        """Compile the hot per-token graph core. Must be called after .cuda().

        Uses fullgraph=False so dynamo can graph-break around the Python
        bookkeeping that stays outside the compiled step. Also enables
        capture_dynamic_output_shape_ops so
        torch.unique / torch.bincount (data-dependent shapes) don't
        trigger their own graph breaks. On Triton 3.6 we avoid the
        TritonGPURemoveLayoutConversions randint-in-fusion crash by
        using rand+argmax for exploration sampling (see routing.py).

        ``dynamic``: passed through to ``torch.compile``. ``False`` is
        static-specialize (best per-iter, recompile per shape).
        ``None`` is auto-detect (recompiles after the 2nd shape with a
        shape-polymorphic kernel — useful for BS sweeps).
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
            self._walker_step, mode=mode, fullgraph=False, dynamic=dynamic,
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
        if self._k_all_cache is None:
            self._k_all_cache = self.cols.k_proj(self.col_id)

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def begin_segment(
        self, B: int, device: torch.device,
        *, clear_neuromod_carryover: bool = False,
    ) -> None:
        """Reset working memory for a new segment.

        Args:
            B: batch size for the new segment.
            device: target device.
            clear_neuromod_carryover: if True, also clear `_neuromod_input_*`
                and `_active_neuromod_delta`. Use this for INDEPENDENT-document
                batches (e.g. shuffled pretrained training) — without
                clearing, the previous batch's last-window snapshot
                bleeds into the new batch's first window's neuromod
                target, polluting credit assignment across documents.
                Default False preserves the streaming-document semantics
                of the standalone training loop.
        """
        cfg = self.cfg
        dtype_fast = self._fast_dtype(device)
        if clear_neuromod_carryover:
            self._neuromod_input_ids = None
            self._neuromod_input_feats = None
            self._neuromod_input_co_visit_flat = None
            self._active_neuromod_delta = None

        self.s = torch.zeros(B, cfg.N, cfg.D_s, device=device, dtype=dtype_fast)
        # Walkers start at deterministic spread positions across the
        # substrate at segment start. Each of the H heads gets its own
        # starting column so that the H walkers diverge from step 1 instead
        # of all colliding at column 0. After a few hops the LM-driven
        # routing takes over and starting positions matter little.
        head_starts = torch.arange(
            cfg.n_heads, device=device, dtype=torch.long,
        ) * (cfg.N // max(cfg.n_heads, 1))
        self.walker_pos = head_starts.unsqueeze(0).expand(B, cfg.n_heads).contiguous()
        # Walker's private running state, zero at segment start.
        self.walker_state = torch.zeros(
            B, cfg.n_heads, cfg.D_s, device=device, dtype=dtype_fast,
        )
        self.surprise_ema = torch.zeros(
            B, cfg.K_horizons, device=device, dtype=torch.float32,
        )
        self.surprise_prev = torch.zeros_like(self.surprise_ema)
        self.tick_counter = 0
        self.window_len = 0

        # Phase-2 log_pi accumulator: reset per segment so phase-2 rollouts
        # see a clean slate. Stays None until the first phase-2 routing
        # decision (so phase-1 forwards don't allocate an extra tensor).
        self._log_pi_sum = None
        self._log_pi_count = 0

        # Invalidate block-level caches at segment boundary.
        self._horizon_logits_cache = None
        self._alpha_cache = None
        self._k_all_cache = None

        # E_bias_flat is a registered buffer initialized in __init__; it
        # persists across segments by design. No re-init here.

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

        # Neuromod plastic-window initialization. If we have a snapshot
        # from a prior window (carried across segments), consume it to
        # produce the grad-carrying delta_nm that routing will use during
        # this window. Otherwise, start with no delta.
        self._begin_plastic_window()

        self._state_initialized = True

    def reset_plastic_memory(self, device: torch.device | None = None) -> None:
        """Hard reset of long-term plastic E_bias (in place, buffer-preserving).

        `device` is kept as an optional argument for backwards-compatible
        callers; `E_bias_flat` is now a registered buffer and lives on
        whatever device the module was moved to.
        """
        self.E_bias_flat.zero_()
        # Also clear any carried-over neuromod snapshot.
        self._neuromod_input_ids = None
        self._neuromod_input_feats = None
        self._active_neuromod_delta = None

    # -----------------------------------------------------------------
    # Neuromodulator plumbing
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    def _begin_plastic_window(self) -> None:
        """Called at the start of each plasticity window.

        Consumes the snapshot of the previous window's touched columns and
        produces `_active_neuromod_delta` — a grad-carrying delta added to
        `E_bias_flat.detach()` during routing this window.
        """
        if self.neuromod is None or self._neuromod_input_feats is None:
            self._active_neuromod_delta = None
            return
        touched_ids = self._neuromod_input_ids
        feats = self._neuromod_input_feats
        if touched_ids.numel() < 2:
            # Need at least 2 touched columns for any edge between them.
            self._active_neuromod_delta = None
            return

        adj_bias = build_adjacency_bias(touched_ids, self.out_nbrs)
        edge_src_local, edge_dst_local, edge_flat = enumerate_touched_edges(
            touched_ids, self.out_nbrs, self.cfg.K,
        )
        if edge_flat.numel() == 0:
            self._active_neuromod_delta = None
            return

        # In neuromod_only mode, hand the neuromod the per-edge Hebbian-
        # flavored stats (log co_visit, current E_bias_old) so its edge MLP
        # has direct access to "how often was this edge fired" and "where is
        # this bias now" when predicting the new target. These were the
        # signals the old Hebbian path used; under the redesign neuromod
        # decides what to do with them.
        #
        # `co_visit` is read from the snapshot taken in
        # `_snapshot_touched_columns` BEFORE `_plasticity_step` resets the
        # live `co_visit_flat`. Without the snapshot the neuromod would
        # see all-zero co_visit (the just-reset live buffer) and never
        # learn to use the per-edge attention bias path.
        if self.cfg.plasticity_mode == "neuromod_only":
            if self._neuromod_input_co_visit_flat is None:
                # No co_visit snapshot yet (planted features without one,
                # or first window after reset_plastic_memory). Without it
                # we can't compute per_edge_extras and the neuromod's
                # edge MLP / per-head bias path is undefined. Skip this
                # window's plasticity entirely; it will resume on the
                # next properly-snapshotted close.
                self._active_neuromod_delta = None
                return
            with torch.no_grad():
                co_visit_at_edges = (
                    self._neuromod_input_co_visit_flat[edge_flat]
                )
                e_bias_at_edges = self.E_bias_flat.detach()[edge_flat]
                # `window_len` was reset by the just-closed `_plasticity_step`,
                # so use `cfg.mod_period` (which equals window length under
                # the single-knob clock invariant) to normalize.
                window_norm = max(self.cfg.mod_period, 1)
                co_visit_log = (
                    co_visit_at_edges.float() / window_norm + 1.0
                ).log().unsqueeze(-1)                           # [E, 1]
                e_bias_feat = e_bias_at_edges.float().unsqueeze(-1)  # [E, 1]
                per_edge_extras = torch.cat(
                    [co_visit_log, e_bias_feat], dim=-1,
                )                                               # [E, 2]
        else:
            per_edge_extras = None

        targets = self.neuromod(
            feats, adj_bias, edge_src_local, edge_dst_local,
            per_edge_extras=per_edge_extras,
        )                                                   # [E] tanh-clamped targets
        # Convert target → delta via EMA blend gated by γ:
        #   active_E_bias = (1 - γ)·E_bias_base + γ·target
        #                 = E_bias_base + γ·(target - E_bias_base)
        # so _active_neuromod_delta[edge] = γ · (target[edge] - E_bias_base[edge]).
        # E_bias_base is detached (no grad back into the persistent buffer)
        # but γ and target both carry grad, so the neuromod still trains
        # from the window's routing gradient.
        gamma = self.neuromod.gamma                         # scalar ∈ (0,1)
        base_at_edges = self.E_bias_flat.detach()[edge_flat].float()
        delta_edge = gamma * (targets.float() - base_at_edges)
        # Scatter into [N*K] dense layout. Force fp32 (matches E_bias_flat).
        dense = torch.zeros(
            self.cfg.num_edges, device=delta_edge.device, dtype=torch.float32,
        )
        # `.float()` on delta_edge preserves grad_fn through the dtype cast.
        self._active_neuromod_delta = dense.index_copy(0, edge_flat, delta_edge)

    def _routing_e_bias(self) -> torch.Tensor:
        """E_bias tensor used by routing this window.

        = E_bias_flat.detach() + η_nm · delta_nm
        where delta_nm is grad-carrying (so loss → routing → delta_nm →
        neuromod params is a live gradient path).

        `E_bias_flat` is always detached here: even when the neuromod is
        off, the persistent buffer can carry a grad_fn from
        `_plasticity_step`'s arithmetic (it's non-leaf after the first
        update), and we never want that entering the compiled step.
        """
        base = self.E_bias_flat.detach()
        if self._active_neuromod_delta is None:
            return base
        return base + self.cfg.neuromod_eta * self._active_neuromod_delta

    @torch._dynamo.disable
    def _snapshot_touched_columns(self) -> None:
        """Snapshot stats for columns visited in the window just closed.

        Stored in `_neuromod_input_ids`, `_neuromod_input_feats` (both detached)
        for consumption by the NEXT window's `_begin_plastic_window`.
        Called at window close, before counters are reset.

        Per-col features:
          - mean state across batch (D_s)
          - column identity (D_id)
          - log(visit_count+1) (1)
          - In neuromod_only mode: scalar surprise broadcast (1) — every
            touched col gets the same surprise_ema.mean() value, exposing
            the LM's overall struggle level as a feature the neuromod can
            condition on. Per-col surprise (visit-weighted mean of per-token
            CE) would be richer but requires per-token visit tracking we
            don't keep today.
        """
        if self.neuromod is None or self.visit_count is None:
            return

        # Touched columns (visited at least once this window)
        touched_mask = self.visit_count > 0
        touched_ids = torch.nonzero(touched_mask, as_tuple=False).squeeze(-1)
        if touched_ids.numel() == 0:
            self._neuromod_input_ids = None
            self._neuromod_input_feats = None
            self._neuromod_input_co_visit_flat = None
            return

        with torch.no_grad():
            s_mean = self.s.float().mean(dim=0)                        # [N, D_s]
            s_feats = s_mean[touched_ids]                              # [U, D_s]
            id_feats = self.col_id[touched_ids].float()                # [U, D_id]
            vc_feats = (self.visit_count[touched_ids] + 1.0).log().unsqueeze(-1)
            parts = [s_feats, id_feats, vc_feats]
            if self.cfg.plasticity_mode == "neuromod_only":
                # Scalar surprise broadcast across touched cols.
                surprise_scalar = self.surprise_ema.mean().float()
                surprise_feat = surprise_scalar.expand(touched_ids.shape[0], 1)
                parts.append(surprise_feat)
            feats = torch.cat(parts, dim=-1)

        self._neuromod_input_ids = touched_ids.detach()
        self._neuromod_input_feats = feats.detach()
        # Snapshot co_visit BEFORE `_plasticity_step` resets it. The next
        # window's neuromod needs the just-closed window's per-edge
        # activity to predict targeted plastic updates. Stored as the full
        # [N*K] flat layout for cheap gather at edge_flat indices later.
        if self.cfg.plasticity_mode == "neuromod_only":
            self._neuromod_input_co_visit_flat = (
                self.co_visit_flat.detach().clone()
                if self.co_visit_flat is not None else None
            )
        else:
            self._neuromod_input_co_visit_flat = None

    def detach_state(self) -> None:
        """TBPTT boundary: preserve values, sever gradient graph."""
        if not self._state_initialized:
            return
        self.s = self.s.detach()
        self.E_bias_flat = self.E_bias_flat.detach()
        self.walker_pos = self.walker_pos.detach()
        self.walker_state = self.walker_state.detach()
        self.surprise_ema = self.surprise_ema.detach()
        self.surprise_prev = self.surprise_prev.detach()
        if self.co_visit_flat is not None:
            self.co_visit_flat = self.co_visit_flat.detach()
        if self.visit_count is not None:
            self.visit_count = self.visit_count.detach()
        # Invalidate block caches — next step rebuilds them with fresh
        # autograd refs tied to the new block's forward computation.
        self._horizon_logits_cache = None
        self._alpha_cache = None
        self._k_all_cache = None
        # Neuromod delta must also get a fresh grad_fn for the next block.
        # The underlying snapshot is already detached, so rebuilding
        # `_active_neuromod_delta` from it gives a fresh grad path while the
        # value remains identical.
        self._active_neuromod_delta = None
        self._begin_plastic_window()

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
        """Hot persistent-walker core: sparse graph dynamics only.

        Under normal training/eval this mutates self.* state in place. See
        step_core_checkpointed for the pure-functional entry point used by
        torch.utils.checkpoint in Phase 1 training.
        """
        self._ensure_block_caches(self.tied_token_emb.weight)
        tau, epsilon = self._schedule_tensors(token_id)
        # Active E_bias = frozen persistent base + grad-carrying neuromod
        # delta (if any). When the neuromod is off, this is just E_bias_flat.
        e_bias_active = self._routing_e_bias()
        if self._compiled_step is not None:
            out = self._compiled_step(
                self.s, self.walker_pos, self.walker_state,
                e_bias_active, token_id, tau, epsilon,
            )
        else:
            out = self._walker_step(
                self.s, self.walker_pos, self.walker_state,
                e_bias_active, token_id, tau, epsilon,
            )
        self._writeback_step_state(out)
        return WalkerCoreReadout(
            motor_state=out.motor_state,
            visit_freq_step=out.visit_freq_step,
            load_balance_loss=out.load_balance_loss,
        )

    # NOTE: A block-level checkpoint helper used to live here. It was
    # removed because (a) it did not call _record_surprise_token or
    # _maybe_close_plasticity_window, so plasticity and surprise
    # EMA would silently freeze on that path, and (b) wall-time showed
    # checkpointing cost more than it saved in this model (autograd
    # still traversed the recomputed per-token graph). See commit log
    # for the original version if a re-introduction is ever warranted.

    def _writeback_step_state(self, out: WalkerStepOutput) -> None:
        """Write pure-step outputs into self.* state (non-grad accumulators)."""
        self.s = out.s_new
        self.walker_pos = out.walker_pos_new
        self.walker_state = out.walker_state_new
        with torch.no_grad():
            self.co_visit_flat = self.co_visit_flat + out.co_visit_delta
            if self.visit_count is not None:
                self.visit_count = self.visit_count + out.visit_count_delta
        self.window_len += 1
        self.tick_counter += 1
        # Phase-2 log_pi accumulation: kept outside the pure step so a
        # compiled or checkpointed _walker_step stays side-effect-free.
        # Gradient must remain connected (no detach), so the addition runs
        # outside the no_grad block above.
        if out.log_pi_step is not None:
            self._log_pi_sum = (
                out.log_pi_step if self._log_pi_sum is None
                else self._log_pi_sum + out.log_pi_step
            )
            self._log_pi_count += 1
        # Routing-trace capture: if a buffer was armed via
        # `start_capturing_routes()`, append this step's choices to it.
        # Used by phase-2 GRPO to record a no-grad sampling trace that a
        # later teacher-forced replay re-evaluates with grad enabled.
        # Captured choices are detached (saved selected_idx are int64
        # tensors with no autograd), so they're safe to keep across the
        # gen pass without holding the autograd graph alive.
        if (
            self._captured_routes is not None
            and out.routing_choices is not None
        ):
            self._captured_routes.append(out.routing_choices)

    def start_capturing_routes(self) -> None:
        """Arm the per-step routing-choice capture buffer.

        After this call, every subsequent step records its routing
        choices into `self._captured_routes`. Call `consume_routing_trace()`
        to retrieve and clear the buffer.

        Used by phase-2 GRPO: sample the rollout under no-grad with
        capture armed, then replay teacher-forced with grad enabled
        using the captured trace as `replay_choices` per step.
        """
        self._captured_routes = []

    def consume_routing_trace(self) -> list[StepRoutingChoices] | None:
        """Pop and return the captured routing trace, or None if not armed.

        After this call, `_captured_routes` is reset to None — capture
        must be re-armed for another sweep.
        """
        out = self._captured_routes
        self._captured_routes = None
        return out

    def arm_replay_trace(self, trace: list[StepRoutingChoices]) -> None:
        """Stash a routing trace for the NEXT `walk_segment` call.

        Used by phase-2 GRPO replay path: after sampling under no-grad
        with capture armed, the trainer calls
        `wrapper.memory.arm_replay_trace(captured_trace)` immediately
        before the with-grad teacher-forced re-forward. `walk_segment`
        consumes the stash, runs the per-token path with `replay_choices`
        per step, and clears the stash to None on entry.
        """
        self._next_replay_trace = trace

    def step(self, token_id: torch.Tensor) -> WalkerReadout:
        """Public single-token API: graph core + immediate readout.

        This is a smoke/debug path, not the training path. It does NOT
        accumulate surprise (that only happens via `accumulate_block_ce`
        from `phase1_step`) and does NOT fire plasticity (firing here
        would both use a stale/zero `surprise_ema` AND rebuild the
        neuromod's `_active_neuromod_delta` without the subsequent
        `detach_state` rescue, leaving it grad-free for the next
        window). Training code must use `step_core()` + `phase1_step`
        flush instead.
        """
        core = self.step_core(token_id)
        motor = self.state_to_model(core.motor_state)
        logits = self.readout(
            motor, self.tied_token_emb.weight,
            horizon_logits=self._horizon_logits_cache,
        )
        return WalkerReadout(
            motor=motor,
            motor_state=core.motor_state,
            logits=logits,
            surprise_ema=self.surprise_ema,
            visit_freq_step=core.visit_freq_step,
            load_balance_loss=core.load_balance_loss,
        )

    def consume_log_pi_sum(self) -> torch.Tensor | None:
        """Return the accumulated phase-2 log π SUM and clear the buffer.

        Returns None when no phase-2 routing decisions were recorded since
        the last `begin_segment` call.

        Note: callers training with REINFORCE should prefer
        `consume_log_pi_mean()`. The raw sum scales with trajectory length
        (T_pre × n_hops × n_heads decisions per rollout = ~thousands of
        log probabilities summed), producing huge gradient magnitudes
        without a corresponding learning-rate adjustment. Mean
        normalization keeps the per-step gradient scale roughly invariant
        to T_pre.
        """
        out = self._log_pi_sum
        self._log_pi_sum = None
        self._log_pi_count = 0
        return out

    def consume_log_pi_mean(self) -> torch.Tensor | None:
        """Return the accumulated phase-2 log π MEAN (sum / step count).

        This is the right normalization for REINFORCE: trajectory length
        affects the gradient direction the same way as the unnormalized
        sum, but the magnitude is now per-step-bounded (~|log p| ≈ a few
        nats) regardless of T_pre. Without this normalization at our
        n_hops=4, n_heads=4, T_pre=512 setup, the loss magnitude is
        ~22,000× larger than per-step.

        Returns None when no phase-2 decisions were recorded since the
        last `begin_segment` call. Clears the buffer + counter.
        """
        if self._log_pi_sum is None or self._log_pi_count == 0:
            self._log_pi_sum = None
            self._log_pi_count = 0
            return None
        mean = self._log_pi_sum / float(self._log_pi_count)
        self._log_pi_sum = None
        self._log_pi_count = 0
        return mean

    # -----------------------------------------------------------------
    # Block-level functional API (compile target for whole-block CUDA graphs)
    # -----------------------------------------------------------------

    def walk_block(
        self,
        s_in: torch.Tensor,                 # [B, N, D_s]
        walker_pos_in: torch.Tensor,        # [B, H]
        walker_state_in: torch.Tensor,      # [B, H, D_s]
        e_bias_in: torch.Tensor,            # [N*K] fp32
        tokens_block: torch.Tensor,         # [B, T_block]
        tau: torch.Tensor,                  # scalar fp32
        epsilon: torch.Tensor,              # scalar fp32
    ) -> WalkBlockOutput:
        """Run T_block sequential _walker_step calls in one autograd graph.

        Compiling THIS function with torch.compile (instead of compiling
        per-step) lets inductor fuse across step boundaries and produces
        ONE forward + ONE backward graph for the whole T_block window.

        State threads through return values — no self.* mutations during
        the loop body — so the compiled graph is free of side effects.
        Caller is responsible for writing the returned state back to
        self.* (or keeping it as a local var for next walk_block call).
        """
        cfg = self.cfg
        _, T_block = tokens_block.shape

        s = s_in
        walker_pos = walker_pos_in
        walker_state = walker_state_in

        motor_list: list[torch.Tensor] = []
        co_visit_total = torch.zeros(
            cfg.num_edges, device=s_in.device, dtype=torch.float32,
        )
        visit_count_total = torch.zeros(
            cfg.N, device=s_in.device, dtype=torch.float32,
        )
        lb_loss_total = torch.zeros(
            (), device=s_in.device, dtype=torch.float32,
        )

        for t in range(T_block):
            out = self._walker_step(
                s, walker_pos, walker_state,
                e_bias_in,
                tokens_block[:, t], tau, epsilon,
            )
            s = out.s_new
            walker_pos = out.walker_pos_new
            walker_state = out.walker_state_new
            motor_list.append(out.motor_state)
            co_visit_total = co_visit_total + out.co_visit_delta
            visit_count_total = visit_count_total + out.visit_count_delta
            lb_loss_total = lb_loss_total + out.load_balance_loss

        motor_states_bt = torch.stack(motor_list, dim=1)  # [B, T_block, D_s]

        return WalkBlockOutput(
            s_new=s,
            walker_pos_new=walker_pos,
            walker_state_new=walker_state,
            motor_states_bt=motor_states_bt,
            co_visit_total=co_visit_total,
            visit_count_total=visit_count_total,
            load_balance_loss=lb_loss_total,
        )

    def compile_walk_block(
        self, mode: str = "default", fullgraph: bool = True,
    ) -> None:
        """Compile `walk_block` for whole-block CUDA-graph capture.

        Replaces the per-step `compile_step()` path with a single compiled
        function that covers an entire mod_period-token block. Inductor
        unrolls the T_block loop, fuses across step boundaries, and
        produces one forward + one backward graph — much higher fusion
        opportunity than per-step.

        ``mode="default"`` is the safe choice (~3.7× over eager).
        ``mode="reduce-overhead"`` adds CUDA-graph capture (additional ~2×
        on top) but requires stable input pointers across replays —
        callers must reuse the same state buffers.
        """
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64,
        )
        self._compiled_walk_block = torch.compile(
            self.walk_block, mode=mode, fullgraph=fullgraph,
        )

    def walk_block_from_h(
        self,
        s_in: torch.Tensor,                 # [B, N, D_s]
        walker_pos_in: torch.Tensor,        # [B, H]
        walker_state_in: torch.Tensor,      # [B, H, D_s]
        e_bias_in: torch.Tensor,            # [N*K] fp32
        h_mem_block: torch.Tensor,          # [B, T_block, D_s]
        tau: torch.Tensor,                  # scalar fp32
        epsilon: torch.Tensor,              # scalar fp32
    ) -> WalkBlockOutput:
        """Block-level analog of `walk_block` driven by an externally-
        supplied [B, T_block, D_s] hidden-state stream instead of token ids.
        Used by the pretrained-LM integration (`walk_segment`).

        State threads through return values — no self.* mutation in the
        loop body — so the compiled graph is free of side effects. Caller
        writes the returned state back to self.* after the call.
        """
        cfg = self.cfg
        B, T_block, _ = h_mem_block.shape
        device = s_in.device

        s = s_in
        walker_pos = walker_pos_in
        walker_state = walker_state_in

        # Dummy token_id — `_walker_step` ignores it when
        # `h_input_override` is provided. Allocated once outside the loop
        # so dynamo doesn't see a fresh tensor each step.
        dummy_tok = torch.zeros(B, dtype=torch.int64, device=device)

        motor_list: list[torch.Tensor] = []
        co_visit_total = torch.zeros(
            cfg.num_edges, device=device, dtype=torch.float32,
        )
        visit_count_total = torch.zeros(
            cfg.N, device=device, dtype=torch.float32,
        )
        lb_loss_total = torch.zeros(
            (), device=device, dtype=torch.float32,
        )

        for t in range(T_block):
            out = self._walker_step(
                s, walker_pos, walker_state,
                e_bias_in,
                dummy_tok, tau, epsilon,
                h_input_override=h_mem_block[:, t],
            )
            s = out.s_new
            walker_pos = out.walker_pos_new
            walker_state = out.walker_state_new
            motor_list.append(out.motor_state)
            co_visit_total = co_visit_total + out.co_visit_delta
            visit_count_total = visit_count_total + out.visit_count_delta
            lb_loss_total = lb_loss_total + out.load_balance_loss

        motor_states_bt = torch.stack(motor_list, dim=1)  # [B, T_block, D_s]

        return WalkBlockOutput(
            s_new=s,
            walker_pos_new=walker_pos,
            walker_state_new=walker_state,
            motor_states_bt=motor_states_bt,
            co_visit_total=co_visit_total,
            visit_count_total=visit_count_total,
            load_balance_loss=lb_loss_total,
        )

    def _walk_block_from_h_ckpt(
        self,
        s_in: torch.Tensor,
        walker_pos_in: torch.Tensor,
        walker_state_in: torch.Tensor,
        e_bias_in: torch.Tensor,
        h_mem_block: torch.Tensor,
        tau: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> WalkBlockOutput:
        """Activation-checkpointed wrapper around `_compiled_walk_block_from_h`.

        Internally calls the compiled block under
        `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, returning
        the WalkBlockOutput's tensor fields as a tuple so the checkpointing
        machinery's "output must be tensors" requirement is satisfied
        cleanly. We unpack back into WalkBlockOutput for the caller.
        """
        from torch.utils.checkpoint import checkpoint

        def _fn(s, wp, ws, eb, hm, t_, e_):
            r = self._compiled_walk_block_from_h(
                s, wp, ws, eb, hm, t_, e_,
            )
            # Return as tuple of tensors for checkpoint compatibility.
            return (
                r.s_new, r.walker_pos_new, r.walker_state_new,
                r.motor_states_bt,
                r.co_visit_total, r.visit_count_total, r.load_balance_loss,
            )

        out_tuple = checkpoint(
            _fn,
            s_in, walker_pos_in, walker_state_in,
            e_bias_in, h_mem_block, tau, epsilon,
            use_reentrant=False,
        )
        return WalkBlockOutput(
            s_new=out_tuple[0],
            walker_pos_new=out_tuple[1],
            walker_state_new=out_tuple[2],
            motor_states_bt=out_tuple[3],
            co_visit_total=out_tuple[4],
            visit_count_total=out_tuple[5],
            load_balance_loss=out_tuple[6],
        )

    def compile_walk_block_from_h(
        self, mode: str = "default", fullgraph: bool = True,
        *, dynamic: bool | None = False,
    ) -> None:
        """Compile `walk_block_from_h` for whole-block fusion in the
        pretrained-LM integration path. ``walk_segment`` routes through
        `_compiled_walk_block_from_h` when set — one compiled call per
        `tbptt_block` window instead of T_block per-token calls.

        ``mode="default"`` (~3.7× over eager) is the safe choice for the
        Llama integration: gives inductor's cross-step fusion without
        cudagraph's stable-pointer constraints. The walker is embedded in
        Llama's autograd graph, so the cudagraph variant
        (``"reduce-overhead"``) would conflict with Llama's dynamic
        activation addresses.

        ``dynamic``: see ``compile_step``. NOTE first-compile cost on
        the whole-block path is 10-15 min at T=256 production scale
        because Dynamo unrolls the T_block loop into one giant FX graph
        (op count multiplied by T). For dev iteration where compile
        cost dominates, use the regional path
        (``IntegratedLM.compile_walker_block(regional=True)``) instead
        — same correctness, ~10× faster compile, ~5-15% lower per-iter
        throughput.
        """
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64,
        )
        self._compiled_walk_block_from_h = torch.compile(
            self.walk_block_from_h, mode=mode, fullgraph=fullgraph,
            dynamic=dynamic,
        )

    def walker_step_from_h(
        self,
        h_input: torch.Tensor,
        *,
        replay_choices: StepRoutingChoices | None = None,
    ) -> WalkerCoreReadout:
        """Hot core driven by an externally-supplied `[B, D_s]` vector
        instead of a `token_id`. Used by the pretrained-LM integration
        (`walk_segment`) where h_mem_t = W_in(llama_hidden_state_t)
        arrives already in graph-state dim.

        Routes through `_compiled_step` if `compile_step()` has been
        called — fuses the per-step kernels just like the token-id path.
        torch.compile recompiles for the new (h_input_override=Tensor)
        input shape on first call, then reuses that graph for the rest
        of the segment.
        """
        self._ensure_block_caches(self.tied_token_emb.weight)
        # Dummy token_id argument (ignored when h_input_override is passed).
        # We still need SOMETHING with a .device so schedule_tensors and
        # downstream ops don't choke. Any scalar int will do.
        dummy_tok = torch.zeros(
            h_input.shape[0], dtype=torch.int64, device=h_input.device,
        )
        tau, epsilon = self._schedule_tensors(dummy_tok)
        e_bias_active = self._routing_e_bias()
        # Replay mode bypasses the compiled step (which doesn't take a
        # `replay_choices` arg). Replay is a phase-2 GRPO-only path; the
        # compile speedup it would lose is irrelevant since phase-2
        # already disables the compiled-block path (see walk_segment).
        if replay_choices is not None:
            out = self._walker_step(
                self.s, self.walker_pos, self.walker_state,
                e_bias_active,
                dummy_tok, tau, epsilon,
                h_input_override=h_input,
                replay_choices=replay_choices,
            )
        else:
            step_fn = self._compiled_step if self._compiled_step is not None \
                      else self._walker_step
            out = step_fn(
                self.s, self.walker_pos, self.walker_state,
                e_bias_active,
                dummy_tok, tau, epsilon,
                h_input_override=h_input,
            )
        self._writeback_step_state(out)
        return WalkerCoreReadout(
            motor_state=out.motor_state,
            visit_freq_step=out.visit_freq_step,
            load_balance_loss=out.load_balance_loss,
        )

    def walk_segment(
        self,
        h_mem: torch.Tensor,                     # [B, T, D_s]
        input_ids: torch.Tensor | None = None,   # accepted for back-compat; unused
        adapter: object | None = None,           # accepted for back-compat; unused
        *,
        preserve_graph: bool = False,
    ) -> torch.Tensor:
        """Process one segment of pretrained-LM hidden states through the
        walker. Returns per-token walker readouts in graph-state space.

        The walker is a vocab-agnostic memory module. It takes Llama hidden
        states in (`h_mem`, shape `[B, T, D_s]`) and produces readouts out
        (`[B, T, D_s]`) that the trainer feeds back into Llama via
        `MemInjectLayer.W_out`. The walker has no LM head, no aux loss, no
        token-level supervision of its own — it learns purely via gradient
        flowing back through `W_out` from Llama's primary CE.

        State contract:
        - Caller is responsible for `begin_segment(B, device)` (via
          `wrapper.begin_segment(bs)`) before the first segment.
        - TBPTT: state is detached every `cfg.tbptt_block` tokens. Skipped
          when `preserve_graph=True` (AR unroll).
        - **Plasticity does NOT fire inside this method.** Surprise is
          supplied externally — the trainer computes Llama's per-token CE
          after `loss.backward()` and calls `update_plasticity(per_token_ce)`,
          which folds CE into `surprise_ema` and runs the plasticity step
          once per training step. See `update_plasticity` for details.

        `input_ids` and `adapter` are accepted but ignored — kept in the
        signature for back-compat with older callers. They will be removed
        in a future cleanup.
        """
        B, T, D_s = h_mem.shape
        assert D_s == self.cfg.D_s, (
            f"h_mem last-dim {D_s} must match cfg.D_s {self.cfg.D_s}"
        )
        assert self._state_initialized, (
            "call begin_segment(B, device) before walk_segment()"
        )

        device = h_mem.device
        cfg = self.cfg
        tbptt = cfg.tbptt_block

        # Walker hot path runs in `state_dtype` (bf16 on CUDA, fp32 on CPU).
        # If the caller runs us on CUDA WITHOUT autocast, the bf16 column
        # state vs fp32 walker weights would mismatch in `content_mlp`,
        # `q_proj`, etc. Defensively enter autocast at the top of the
        # processing loop so inference / benchmark callers don't need to
        # know about the requirement. When autocast is already active
        # (training harnesses set it), this nests cleanly.
        # IMPORTANT: this re-entry check must come BEFORE we consume
        # `_next_replay_trace`. Otherwise the outer call would clear the
        # stash, then the inner (recursive) call would find it None and
        # silently skip replay — manifesting only on CUDA without an
        # external autocast region (CPU tests would not catch it).
        if device.type == "cuda" and not torch.is_autocast_enabled():
            return self._walk_segment_with_autocast(
                h_mem, preserve_graph=preserve_graph,
            )

        # Consume the replay stash exactly once. If non-None, length must
        # equal the segment length T; the per-token loop will pull
        # `replay_choices` from it per step. Set None up-front so a
        # nested / re-entrant call doesn't re-use the same trace.
        replay_trace = self._next_replay_trace
        self._next_replay_trace = None
        if replay_trace is not None:
            assert len(replay_trace) == T, (
                f"replay trace length {len(replay_trace)} doesn't match "
                f"segment T={T}"
            )

        motor_blocks: list[torch.Tensor] = []

        # Block-stride loop. Whole-block path replaces T_block per-token
        # `walker_step_from_h` calls with one `walk_block_from_h` call —
        # inductor fuses across step boundaries, giving ~3.7× over the
        # eager per-token path standalone (the speedup we lose if we run
        # the per-token loop inside Llama's autograd graph). Falls back
        # to per-token for the partial last block (when T isn't divisible
        # by tbptt), or when the block compile hasn't been set up yet.
        block_start = 0
        while block_start < T:
            block_end = min(block_start + tbptt, T)
            ticks = block_end - block_start
            use_block_path = (
                ticks == tbptt and self._compiled_walk_block_from_h is not None
            )
            # Phase-2 GRPO needs `log_pi_step` from `_walker_step` to
            # accumulate `_log_pi_sum` for `consume_log_pi_sum()`. The
            # block path drops `log_pi_step` (WalkBlockOutput has no field
            # for it), so silently falling into the block path during
            # phase 2 would null the policy gradient. Force the per-
            # token fallback in that case so log_pi accumulation stays
            # alive via `_writeback_step_state`.
            if use_block_path and self.phase == "phase2":
                use_block_path = False
            # Replay path uses per-step `replay_choices` which the compiled
            # block kernel doesn't support. Force per-token fallback.
            if use_block_path and replay_trace is not None:
                use_block_path = False

            if use_block_path:
                # Set up block-static caches once per block. _walker_step
                # reads these from self.* (they're not arguments).
                self._ensure_block_caches(self.tied_token_emb.weight)
                e_bias_active = self._routing_e_bias()
                # Dummy token_id satisfies _walker_step's signature; it
                # ignores it when h_input_override is supplied.
                dummy_tok = torch.zeros(B, dtype=torch.int64, device=device)
                tau, epsilon = self._schedule_tensors(dummy_tok)

                # Activation checkpointing on the whole-block forward.
                # Disabled in eval / when explicitly opted out.
                ckpt_block = (
                    self.training
                    and getattr(self, "_checkpoint_block", True)
                )
                if ckpt_block:
                    out = self._walk_block_from_h_ckpt(
                        self.s, self.walker_pos, self.walker_state,
                        e_bias_active,
                        h_mem[:, block_start:block_end],
                        tau, epsilon,
                    )
                else:
                    out = self._compiled_walk_block_from_h(
                        self.s, self.walker_pos, self.walker_state,
                        e_bias_active,
                        h_mem[:, block_start:block_end],
                        tau, epsilon,
                    )

                # State writeback — walk_block_from_h is pure-functional
                # so callers must thread state explicitly. Mirrors what
                # _writeback_step_state does for the per-token path, but for
                # the whole block in one shot.
                self.s = out.s_new
                self.walker_pos = out.walker_pos_new
                self.walker_state = out.walker_state_new
                with torch.no_grad():
                    self.co_visit_flat = self.co_visit_flat + out.co_visit_total
                    if self.visit_count is not None:
                        self.visit_count = (
                            self.visit_count + out.visit_count_total
                        )
                self.window_len += ticks
                self.tick_counter += ticks
                # NOTE: walk_block_from_h drops log_pi_step. Phase-2 GRPO
                # callers stay on the per-token preserve_graph=True path
                # (T=1 per call), which keeps log_pi accumulation alive
                # via _writeback_step_state. If phase-2 ever wants block-path
                # speedup, WalkBlockOutput needs a log_pi_total field.

                motor_bt = out.motor_states_bt                          # [B, ticks, D_s]
            else:
                # Per-token fallback (partial last block, AR unroll, or
                # compile not set up). Mutates self.* via walker_step_from_h.
                block_motor_list: list[torch.Tensor] = []
                for t in range(block_start, block_end):
                    rc = (
                        replay_trace[t] if replay_trace is not None else None
                    )
                    r = self.walker_step_from_h(
                        h_mem[:, t], replay_choices=rc,
                    )
                    block_motor_list.append(r.motor_state)
                motor_bt = torch.stack(block_motor_list, dim=1)         # [B, ticks, D_s]

            motor_blocks.append(motor_bt)

            # TBPTT detach at block boundary (unless AR unroll preserving graph).
            if not preserve_graph and block_end < T:
                self.detach_state()

            block_start = block_end

        readouts = torch.cat(motor_blocks, dim=1)                       # [B, T, D_s]
        return readouts

    def _walk_segment_with_autocast(
        self,
        h_mem: torch.Tensor,
        *,
        preserve_graph: bool,
    ) -> torch.Tensor:
        """Defensive wrapper that enters bf16 autocast on CUDA before
        recursing into `walk_segment`. Used when a caller runs us on
        CUDA without first establishing an autocast region — without it,
        the bf16 column state would mismatch the fp32 walker weights in
        `content_mlp` / `q_proj`."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.walk_segment(
                h_mem, preserve_graph=preserve_graph,
            )

    def _walker_step(
        self,
        s_in: torch.Tensor,              # [B, N, D_s]
        walker_pos_in: torch.Tensor,     # [B, H]
        walker_state_in: torch.Tensor,   # [B, H, D_s]
        e_bias_flat_in: torch.Tensor,    # [N*K] fp32 — snapshot of plastic bias
        token_id: torch.Tensor,          # [B] int64 (ignored if h_input_override given)
        tau: torch.Tensor,               # scalar
        epsilon: torch.Tensor,           # scalar
        h_input_override: torch.Tensor | None = None,   # [B, D_s] — bypass token embed
        replay_choices: StepRoutingChoices | None = None,  # if set: replay
                                          # routing instead of sampling. Used
                                          # by phase-2 GRPO teacher-forced
                                          # re-evaluation. Captured per-step
                                          # output is None when this kwarg is
                                          # supplied (we can't capture what we
                                          # didn't sample).
    ) -> WalkerStepOutput:
        """Pure-functional one-token walker update.

        Flow (write-first-then-route, no anchoring — the walker walks
        continuously across the substrate):
          1. Token embed + per-walker broadcast.
          2. Steering input (pre-update): cat(s_cur_old, id_cur,
             walker_state, token_per_walker). Feed content_mlp → m_out.
          3. Sparse deposit: walker writes at CURRENT column cur_bh.
          4. Re-read s[cur_bh] (post-update).
          5. Steering input (post-update): cat(s_cur_new, id_cur,
             walker_state, token_per_walker). Feed q_proj → hop scores.
             Gumbel top-1 over the K out-neighbors picks next_col.
          6. STE-gated endpoint readout: end_state =
             Σ_k ste_weights[k] · s_new[nbrs[cur,k]]. Routing gradient
             bridges to loss through this soft sum.
          7. Cross-attn over endpoints → motor_state.
          8. Walker state update: EMA of m_out (per-walker decay).

        Takes current state (s, walker_pos, walker_state) as explicit
        inputs and returns new state plus non-grad accumulator deltas.
        No mutation of self.* here; caller's job.
        """
        assert self._state_initialized, "call begin_segment() first"
        cfg = self.cfg
        B = s_in.shape[0]
        device = s_in.device

        is_training = self.training
        cfg_n_heads = cfg.n_heads
        BH = B * cfg_n_heads
        state_dtype = s_in.dtype
        lb_dtype = torch.float32
        alpha = self._alpha_cache                                     # [N]

        # 1. Embed token and project to graph-state space.
        #    When h_input_override is provided (pretrained-LM path), skip
        #    the token_emb lookup and use the supplied vector directly.
        if h_input_override is None:
            h_input_model = self.tied_token_emb(token_id)             # [B, D_model]
            h_input = self.token_to_state(h_input_model)              # [B, D_s]
        else:
            h_input = h_input_override                                # [B, D_s]

        cur_bh = walker_pos_in.reshape(BH)
        P_mass = torch.zeros(cfg.N, device=device, dtype=lb_dtype)
        log_pi_step: torch.Tensor | None = None

        # 2. Walker message from OLD column state + walker state + token.
        # ``s_in.detach()`` here cuts the gather backward into [B, N, D_s]
        # zeros + scatter (~50% of CUDA time at large N per kineto profile).
        # The substrate read becomes a constant input feature for content_mlp;
        # gradient still flows to content_mlp's params via this step's loss
        # via the s_cur_new (post-LIF) gather below — that one is left
        # ATTACHED so the LIF chain backward fires. Cross-step substrate
        # gradient is preserved via walker_state EMA + LIF α-channel
        # through s_cur_new.
        cur = cur_bh.view(B, cfg_n_heads)
        s_cur_old = torch.gather(
            s_in.detach(), 1,
            cur.unsqueeze(-1).expand(B, cfg_n_heads, cfg.D_s),
        ).reshape(BH, cfg.D_s)
        id_cur = self.col_id[cur_bh]                                  # [B*H, D_id]
        walker_state_flat = walker_state_in.reshape(BH, cfg.D_s)
        token_per_walker = (
            h_input.unsqueeze(1)
            .expand(B, cfg_n_heads, cfg.D_s)
            .reshape(BH, cfg.D_s)
        )
        cat_pre = torch.cat([
            self.cols.content_norm(s_cur_old).to(state_dtype),
            id_cur.to(state_dtype),
            walker_state_flat.to(state_dtype),
            token_per_walker.to(state_dtype),
        ], dim=-1)
        m_out = self.cols.content_mlp(cat_pre)                        # [B*H, D_s]

        # 3. Sparse deposit. Walker always writes at its current column
        # (no STE gating on m_out — routing's gradient comes through the
        # endpoint readout below). Single-source write per walker; the
        # Triton LIF kernel sees a fixed grid shape of BH writes per step.
        batch_idx = self._batch_idx
        all_dests = (batch_idx * cfg.N + cur_bh).contiguous()
        all_msgs = m_out.to(state_dtype).contiguous()

        s_flat = s_in.reshape(B * cfg.N, cfg.D_s)
        s_flat_new = sparse_lif_update(s_flat, all_msgs, all_dests, alpha, cfg.N)
        s_new = s_flat_new.view(B, cfg.N, cfg.D_s)

        # 4. Re-read walker's current column post-update.
        s_cur_new = torch.gather(
            s_new, 1, cur.unsqueeze(-1).expand(B, cfg_n_heads, cfg.D_s),
        ).reshape(BH, cfg.D_s)

        # 5. Routing query uses POST-UPDATE state.
        cat_post = torch.cat([
            self.cols.content_norm(s_cur_new).to(state_dtype),
            id_cur.to(state_dtype),
            walker_state_flat.to(state_dtype),
            token_per_walker.to(state_dtype),
        ], dim=-1)
        q = self.cols.q_proj(cat_post).view(
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
        E_vals = e_bias_flat_in[edge_flat].to(scores.dtype)
        scores = scores + E_vals

        saved_edge = (
            replay_choices.edge_idx if replay_choices is not None else None
        )
        rout = route_or_replay(
            scores, tau=tau, epsilon=epsilon, training=is_training,
            phase=self.phase, saved_idx=saved_edge,
        )
        if rout.log_pi is not None:
            log_pi_step = rout.log_pi.view(B, cfg_n_heads).sum(dim=1)   # [B]
        next_local = rout.selected_idx                                # [B*H]
        captured_edge_idx = (
            rout.selected_idx if replay_choices is None else None
        )
        next_col = torch.gather(
            nbrs_of_cur, 1, next_local.unsqueeze(-1),
        ).squeeze(-1)                                                 # [B*H]
        edge_taken_flat = cur_bh * cfg.K + next_local                 # [B*H]

        P_mass = P_mass.index_add(
            0, nbrs_of_cur.reshape(-1),
            rout.soft_probs.to(lb_dtype).reshape(-1),
        )

        # visit_count delta (non-grad). Trajectory footprint fed to the
        # neuromod at window close. Only the next-col landing contributes
        # (no anchors).
        visit_count_delta = torch.zeros(cfg.N, device=device, dtype=torch.float32)
        visit_count_delta.scatter_add_(0, next_col, self._ones_bh)

        # 6. Endpoint readout = walker's own current-col content + STE-gated
        # neighbor-id embedding. The first term (s_cur_new) is always
        # non-zero in forward because the walker just wrote there; this
        # carries gradient to content_mlp, LIF α, walker_state, token
        # embedding. The second term is an STE bridge for routing: it's
        # Σ_k ste[k] · nbr_id_to_s(col_id[nbrs[k]]). Forward it equals
        # nbr_id_to_s(col_id[next_col]) (ste is one-hot); backward gives
        # routing its gradient path.
        walker_pos_new = next_col.view(B, cfg_n_heads)
        s_cur_new_bhh = s_cur_new.view(B, cfg_n_heads, cfg.D_s)
        id_nbrs = self.col_id[nbrs_of_cur].view(
            B, cfg_n_heads, cfg.K, cfg.D_id,
        )                                                             # [B, H, K, D_id]
        nbr_id_embed = self.nbr_id_to_s(id_nbrs)                      # [B, H, K, D_s]
        ste_bhk = rout.ste_weights.view(B, cfg_n_heads, cfg.K)
        nbr_signal = (
            ste_bhk.to(nbr_id_embed.dtype).unsqueeze(-1) * nbr_id_embed
        ).sum(dim=2)                                                  # [B, H, D_s]
        end_states = s_cur_new_bhh + nbr_signal.to(s_cur_new_bhh.dtype)

        # 7. Cross-attn over endpoint states → motor_state.
        s_traj = self.out_norm(end_states).to(state_dtype)
        k = self.out_k_proj(s_traj)
        v = self.out_v_proj(s_traj)
        q_motor = self.motor_query.to(state_dtype).unsqueeze(0).expand(B, -1)
        scale_out = 1.0 / (cfg.D_s ** 0.5)
        attn_scores = torch.sum(k * q_motor.unsqueeze(1), dim=-1) * scale_out
        attn = F.softmax(attn_scores, dim=-1)
        motor_state = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        # 8. Walker state update: EMA of m_out per walker.
        alpha_w = torch.sigmoid(self.walker_state_alpha)              # [H]
        alpha_w_view = alpha_w.view(1, cfg_n_heads, 1).to(walker_state_in.dtype)
        m_out_view = m_out.view(B, cfg_n_heads, cfg.D_s).to(walker_state_in.dtype)
        walker_state_new = (
            alpha_w_view * walker_state_in
            + (1.0 - alpha_w_view) * m_out_view
        )

        # 9. co_visit delta (non-grad)
        co_visit_delta = torch.bincount(
            edge_taken_flat, minlength=cfg.num_edges,
        ).to(torch.float32)

        # Single routing decision per step (BH decisions). Normalize
        # P_mass and visit counts by the matching divisor.
        n_decisions = float(BH)
        P_mean = P_mass / n_decisions
        f_mean = (visit_count_delta / n_decisions).detach()
        load_balance_loss = cfg.N * (P_mean * f_mean).sum()

        if replay_choices is None:
            routing_choices_out: StepRoutingChoices | None = StepRoutingChoices(
                edge_idx=captured_edge_idx,
            )
        else:
            routing_choices_out = None

        return WalkerStepOutput(
            s_new=s_new,
            walker_pos_new=walker_pos_new,
            walker_state_new=walker_state_new,
            motor_state=motor_state,
            co_visit_delta=co_visit_delta,
            visit_count_delta=visit_count_delta,
            visit_freq_step=f_mean if cfg.lambda_balance > 0 else None,
            load_balance_loss=load_balance_loss,
            log_pi_step=log_pi_step,
            routing_choices=routing_choices_out,
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

    def readout_ce_block(
        self,
        motor_state: torch.Tensor,   # [B, T, D_s]
        targets: torch.Tensor,       # [B, T, K_h] int64
        valid: torch.Tensor,         # [B, T, K_h] bool
    ) -> torch.Tensor:
        """Factorized cross-entropy: never materializes [B, T, K_h, V].

        Replaces the `readout_from_state_block → F.cross_entropy(logits_flat,
        targets_flat)` path used for training loss. Memory-efficient at
        large batch: at BS=88, T=48, K_h=8, V=32000 this frees ~4 GB that
        the naive broadcast would spend on the big logit tensor and its
        log_softmax save-for-backward.

        Returns `[B, T, K_h]` float32 CE per (position, horizon), masked.
        """
        self._ensure_block_caches(self.tied_token_emb.weight)
        if not torch.is_autocast_enabled():
            motor_state = motor_state.to(self.state_to_model.weight.dtype)
        motor = self.state_to_model(motor_state)                      # [B, T, D_model]
        return self.readout.cross_entropy_factorized(
            motor, self.tied_token_emb.weight,
            targets, valid,
            horizon_logits=None,  # recomputed for fp32 precision inside CE
        )

    # -----------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    @torch.no_grad()
    def accumulate_block_ce(
        self,
        ce_block: torch.Tensor,      # [B, T_block, K_h] — per-position per-horizon CE
        valid_mask: torch.Tensor,    # [T_block, K_h] — True where target exists
    ) -> None:
        """Stream the training flush's CE tensor into `surprise_ema`.

        Replaces the old "re-run readout at window close" path. The training
        loop already computes CE for every valid (position, horizon) pair in
        the just-finished TBPTT block; that is exactly what surprise needs.
        Per horizon, apply the EMA recurrence
            surprise_ema[:, k] ← (1-α)·surprise_ema[:, k] + α·ce_block[:, i, k]
        in token order, skipping positions where the target was masked out
        (segment-boundary tail). T_block is small (≤ mod_period = 48–128),
        so the Python loop overhead is negligible vs one dense readout call.
        """
        alpha = self.cfg.alpha_gamma_s
        T_block, K_h = valid_mask.shape
        assert ce_block.shape[1:] == (T_block, K_h), (
            f"ce_block shape {tuple(ce_block.shape)} mismatches valid_mask "
            f"shape {tuple(valid_mask.shape)}"
        )
        ema = self.surprise_ema.float()
        ce_fp32 = ce_block.float()
        valid_bool = valid_mask.bool()
        for i in range(T_block):
            valid_i = valid_bool[i].unsqueeze(0)            # [1, K_h] bool
            candidate = (1.0 - alpha) * ema + alpha * ce_fp32[:, i, :]
            ema = torch.where(valid_i, candidate, ema)
        self.surprise_ema = ema

    @torch._dynamo.disable
    def update_plasticity(
        self,
        per_token_surprise: torch.Tensor | None,
    ) -> None:
        """Externally-driven plasticity update. Call AFTER `loss.backward()`.

        Critical grad-scope contract (do NOT add @torch.no_grad here): the
        helper methods this calls each have their own no_grad scope where
        needed:
          - `accumulate_block_ce`        @torch.no_grad
          - `_finalize_surprise_window`  @torch.no_grad
          - `_plasticity_step`           @torch.no_grad
        The final `_begin_plastic_window()` call MUST run with gradients
        enabled — it runs the neuromod forward to build the next segment's
        `_active_neuromod_delta`, and that delta MUST carry grad_fn so the next
        forward's loss can train the neuromod. Wrapping this whole function
        in @torch.no_grad would silently kill the neuromod training signal
        (caught by Codex audit 2026-05-04 — was a real bug introduced by
        the external-surprise refactor).

        The walker is vocab-agnostic: it does not compute its own next-token
        CE. The trainer is responsible for supplying surprise as Llama's
        per-token CE (or any other ground-truth-aware signal).

        Modes:
          - ``per_token_surprise=None``: AR free-generation / inference. Skip
            plasticity entirely. Walker forward state still evolves; only
            the structural plastic update (`E_bias_flat`) is frozen.
          - ``per_token_surprise: [B, T] float``: training. Fold per-token
            CE into ``surprise_ema``, fire one plasticity window (commit the
            current ``_active_neuromod_delta`` into ``E_bias_flat``, snapshot,
            reset window counters), then build the next window's neuromod
            delta with a fresh forward pass.

        Why a single fire per call rather than a fire per ``mod_period``
        block: under external-surprise mode all ``T`` forward steps in the
        segment shared the SAME ``_active_neuromod_delta`` — there is exactly one
        delta to commit per segment forward. Multiple commits would commit
        deltas that no forward step ever read, attaching no gradient to
        their neuromod params. Effective plasticity rate becomes once per
        training step (vs. T/mod_period under the old in-forward path).
        """
        if per_token_surprise is None:
            # No ground truth → no surprise → no plastic update. Reset the
            # window counters so the next training-step call sees a clean
            # window even if forward steps incremented `window_len` past
            # `mod_period` during the AR rollout.
            self.window_len = 0
            if self.co_visit_flat is not None:
                self.co_visit_flat = torch.zeros_like(self.co_visit_flat)
            if self.visit_count is not None:
                self.visit_count = torch.zeros_like(self.visit_count)
            return

        assert per_token_surprise.ndim == 2, (
            f"per_token_surprise must be [B, T]; got "
            f"{tuple(per_token_surprise.shape)}"
        )
        B_in, T_in = per_token_surprise.shape
        K_h = self.cfg.K_horizons
        # Broadcast the horizon-1 Llama CE across the surprise_ema's K_h dim.
        # `surprise_ema.mean()` is the only consumer (see `_plasticity_step`),
        # so duplicating across horizons is mathematically a no-op vs a
        # collapsed-shape EMA — cheaper than reshaping the EMA buffer.
        ce_block = (
            per_token_surprise.detach()
            .unsqueeze(-1)
            .expand(B_in, T_in, K_h)
            .float()
        )
        valid_mask = torch.ones(
            T_in, K_h, dtype=torch.bool, device=per_token_surprise.device,
        )
        # Stream into surprise_ema in token order.
        self.accumulate_block_ce(ce_block, valid_mask)
        # Snapshot, fire plasticity, build next delta.
        # `_plasticity_step` resets `window_len`, `co_visit_flat`, and
        # `visit_count`; `_begin_plastic_window` produces a fresh
        # grad-carrying `_active_neuromod_delta` for the next segment.
        self._finalize_surprise_window()
        self._plasticity_step()
        self._begin_plastic_window()

    @torch._dynamo.disable
    @torch.no_grad()
    def _finalize_surprise_window(self) -> None:
        """Snapshot the current `surprise_ema` into `surprise_prev`.

        The EMA itself was already streamed in via `accumulate_block_ce`
        during each TBPTT flush, so this is a cheap snapshot now — no
        readout re-run, no dense logit materialization. Keeps the
        `surprise_prev` buffer available for downstream Δ-surprise consumers.
        """
        self.surprise_prev = self.surprise_ema.detach().clone()

    @torch._dynamo.disable
    def _maybe_close_plasticity_window(self) -> None:
        if self.window_len < self.cfg.mod_period:
            return
        self._finalize_surprise_window()
        self._plasticity_step()
        # Start the next window's neuromod delta OUTSIDE the no_grad scope
        # of _plasticity_step so the delta carries a live grad_fn and
        # routing loss can reach the neuromod parameters.
        self._begin_plastic_window()

    @torch._dynamo.disable
    @torch.no_grad()
    def _plasticity_step(self) -> None:
        """Plasticity update on window close. Runs:
          1. Scalar-eta Hebbian on traversed-edge counts (legacy).
          2. If neuromod is enabled: bake the grad-carrying `_active_neuromod_delta`
             into `E_bias_flat` (detached), then take a new snapshot of the
             just-closed window and produce the next window's delta.

        Wrapped in `torch.no_grad` so the non-diff update can't accidentally
        attach a grad_fn to `E_bias_flat` across repeated calls.
        """
        cfg = self.cfg
        device = self.s.device

        with torch.autocast(device_type=device.type, enabled=False):
            # Hebbian path is only computed in "hebbian_plus_neuromod" mode.
            # In "neuromod_only" mode, the Hebbian-flavored stats (co_visit,
            # E_bias_old) are inputs to the neuromod's edge MLP instead, so
            # the neuromod itself decides what plastic update to apply.
            if cfg.plasticity_mode == "hebbian_plus_neuromod":
                window = max(self.window_len, 1)
                co_visit_norm = (
                    self.co_visit_flat.float() / window
                    if self.co_visit_flat is not None
                    else torch.zeros_like(self.E_bias_flat)
                )
                surprise_scalar = self.surprise_ema.mean().float()
                eta_global = cfg.plast_eta * torch.sigmoid(
                    surprise_scalar - cfg.plast_surprise_bias,
                )
                delta_hebb = eta_global * (
                    co_visit_norm - cfg.plast_decay * self.E_bias_flat
                )
            else:
                delta_hebb = torch.zeros_like(self.E_bias_flat)

            # Neuromod contribution (if any), detached and scaled.
            delta_nm_commit = torch.zeros_like(self.E_bias_flat)
            if self._active_neuromod_delta is not None:
                delta_nm_commit = (
                    cfg.neuromod_eta * self._active_neuromod_delta.detach()
                )

            self.E_bias_flat = (
                self.E_bias_flat + delta_hebb + delta_nm_commit
            ).clamp(-cfg.E_bias_max, cfg.E_bias_max)

        # Snapshot the just-closed window BEFORE we reset counters — this
        # becomes the observation for the NEXT window's neuromod fire.
        self._snapshot_touched_columns()

        # Reset window-scoped counters (co_visit, window_len, visit_count,
        # active_delta_nm — the delta is now baked into E_bias_flat).
        self.co_visit_flat = torch.zeros_like(self.co_visit_flat)
        self.window_len = 0
        if self.visit_count is not None:
            self.visit_count = torch.zeros_like(self.visit_count)
        self._active_neuromod_delta = None

        # NOTE: `_begin_plastic_window()` deliberately does NOT run here.
        # This method is @torch.no_grad()'d — running the neuromod forward
        # inside that scope would produce an `_active_neuromod_delta` without a
        # grad_fn, and routing in the next window would see a grad-free
        # delta, silently killing the neuromod's training signal. The
        # caller (`_maybe_close_plasticity_window`) runs
        # `_begin_plastic_window()` outside this no_grad scope.
