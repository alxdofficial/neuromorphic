"""GraphWalkerMemory — trajectory-routed plastic concept graph.

Hot path per token (write-first-then-route):
  1. If `is_new_window` (first token of a new plasticity window):
     input-plane Gumbel softmax picks H anchor columns and STE-gates an
     injection of the current token's content. Walker positions get
     teleported to the anchor cols. Within a window this step is SKIPPED
     and walkers just roam.
  2. Walker at its current column emits a message from (col_state,
     col_id, walker_state, token_embed) → content_mlp.
  3. Sparse LIF deposit: walker writes m_out at its CURRENT column (not
     destination). On window-start steps the anchor injection is stacked
     into the same LIF kernel call at the anchor columns (which are also
     the walker's current position on those steps).
  4. Re-read updated col state at cur_bh (walker's own contribution is now
     in it). Steering query uses (s_cur_new, col_id, walker_state,
     token_embed). Score K out-edges, Gumbel top-1 → next_col.
  5. Endpoint readout = s_cur_new + Σ_k ste[k] · nbr_id_to_s(col_id[nbrs[k]]).
     First term carries gradient to content_mlp / walker_state / token;
     second term is the STE bridge from routing → loss.
  6. Cross-attn over H endpoints → motor_state.
  7. Walker state update: w_new = σ(α_h) · w_old + (1-σ(α_h)) · m_out.

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
from src.graph_walker.routing import gumbel_top1_softmax, gumbel_schedule
from src.graph_walker.topology import build_topology
from src.graph_walker.triton.lif import sparse_lif_update


class _NullCtx:
    """Minimal no-op context for the autocast-defensive wrap in forward_segment."""
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
        self.per_plane_content = cfg.per_plane_content_mlp
        # Decoupled content_mlp hidden width: explicit cfg.D_hid_content when
        # set, else legacy ffn_mult_content × D_s. Pinning D_hid_content to
        # a fixed value (e.g. 1024) makes content_mlp cost O(D_s · D_hid)
        # instead of O(D_s²); see config.py for rationale.
        D_hid_content = (
            cfg.D_hid_content
            if cfg.D_hid_content is not None
            else cfg.ffn_mult_content * D_s
        )
        if cfg.per_plane_content_mlp:
            self.content_mlp = PerPlaneMLP(
                L=cfg.L, D_in=D_steer,
                D_hid=D_hid_content, D_out=D_s,
            )
        elif cfg.per_head_content_mlp:
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
class WalkerCorePureOutput:
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
    prev_motor_new: torch.Tensor          # [B, D_s]
    motor_state: torch.Tensor             # [B, D_s]
    co_visit_delta: torch.Tensor          # [N*K] fp32 — add to self.co_visit_flat
    visit_count_delta: torch.Tensor       # [N] fp32 — add to self.visit_count
    visit_freq_step: torch.Tensor | None  # [N] fractional visits (optional)
    load_balance_loss: torch.Tensor       # scalar — aux loss term
    log_pi_step: torch.Tensor | None      # [B] phase-2 log π over routing
                                          # decisions made this step (anchor +
                                          # per-token), summed over walker heads.
                                          # None outside phase 2.


@dataclass
class BlockOutput:
    """Return bundle from `block_forward`: T_block sequential steps.

    All forward state (s, walker_pos, walker_state, prev_motor) is returned
    explicitly so the caller can thread it through autograd or write it
    back to self. Non-grad accumulators (co_visit, visit_count) are
    summed within the block; caller adds them to module-level totals.
    """
    s_new: torch.Tensor                  # [B, N, D_s]
    walker_pos_new: torch.Tensor         # [B, H]
    walker_state_new: torch.Tensor       # [B, H, D_s]
    prev_motor_new: torch.Tensor         # [B, D_s]
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

        # Topology
        topo = build_topology(
            plane_rows=cfg.plane_rows, plane_cols=cfg.plane_cols,
            L=cfg.L, K=cfg.K, p_rewire=cfg.p_rewire,
            K_intra_fraction=cfg.K_intra_fraction, seed=cfg.topology_seed,
            K_inter_bwd_fraction=cfg.K_inter_bwd_fraction,
            intra_radius=cfg.intra_radius, inter_radius=cfg.inter_radius,
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
        # Small-random init (NOT zero). Zero-init would make `inject_msg`
        # identically zero regardless of the STE-gated anchor softmax
        # output, which zeros out input_q_proj's gradient on step 0 (dead
        # bilinear product: grad on ste flows as inject_msg · grad_upstream,
        # and inject_msg·anything = 0 if inject_msg = 0). Small std keeps
        # initial injection small enough not to destabilise the LIF state
        # but large enough to light up the gradient to anchor routing.
        nn.init.normal_(self.input_v_proj.weight, mean=0.0, std=0.014)

        # Parallel v_inject projection for the pretrained-LM integration
        # path, where h_input arrives already in D_s dim (from W_in applied
        # to a frozen-Llama hidden state) and there is no D_model-dim
        # h_input_model available. Same small-init as input_v_proj.
        self.mem_input_v_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        nn.init.normal_(self.mem_input_v_proj.weight, mean=0.0, std=0.014)

        # Prev-token motor feeds into the start-col query (only): lets the
        # start position depend on recent output direction, not just the
        # current token's identity. Zero-init so day-1 behaviour is unchanged
        # and the model learns to use the chain through the routing gradient.
        # Autograd follows the chain within a TBPTT block; detach_state cuts
        # it at block boundaries.
        self.prev_motor_proj = nn.Linear(cfg.D_s, cfg.D_s, bias=False)
        nn.init.zeros_(self.prev_motor_proj.weight)

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

        self._N_in = cfg.N_per_plane

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
        if cfg.use_neuromod:
            D_feat = cfg.D_s + cfg.D_id + 1  # state + id + visit_count
            self.neuromod = NeuromodGraphTransformer(
                D_feat=D_feat,
                D_mod=cfg.neuromod_D_mod,
                n_layers=cfg.neuromod_n_layers,
                n_heads=cfg.neuromod_n_heads,
                edge_hidden=cfg.neuromod_edge_hidden,
                E_bias_max=cfg.E_bias_max,
            )
        else:
            self.neuromod = None

        # --- Persistent state (buffers) ---
        self._state_initialized = False
        self.s: torch.Tensor                 # [B, N, D_s]
        self.prev_motor: torch.Tensor        # [B, D_s] — chained into next step
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

        # Neuromod plastic-window state. `_active_delta_nm` is the grad-
        # carrying E_bias delta for the CURRENT plasticity window; it is
        # computed at window start from the snapshot of the PREVIOUS window,
        # detached and merged into E_bias_flat at window close. None when
        # there is no previous snapshot (bootstrapping) or neuromod off.
        self._active_delta_nm: torch.Tensor | None = None
        # Snapshot of the most recently closed window's touched columns —
        # consumed at the next window's start to produce _active_delta_nm.
        self._prev_snapshot_ids: torch.Tensor | None = None
        self._prev_snapshot_feats: torch.Tensor | None = None

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
        # Reset at `begin_segment`. Read by GRPO via `consume_log_pi_sum()`.
        # None when no phase-2 decisions have been recorded.
        self._log_pi_sum: torch.Tensor | None = None

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

        # Filled when compile_block_from_h() is called; forward_segment
        # routes a whole tbptt block through the compiled version when set,
        # replacing T_block per-token step_core_from_h calls.
        self._compiled_block_from_h = None

        # Filled when compile_block() is called (standalone path that drives
        # block_forward with token ids). Initialized here so callers can
        # check `lm.memory._compiled_block is not None` without an
        # AttributeError when compile_block was never invoked.
        self._compiled_block = None

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
            self._step_core_pure, mode=mode, fullgraph=False,
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

    def begin_segment(
        self, B: int, device: torch.device,
        *, clear_neuromod_carryover: bool = False,
    ) -> None:
        """Reset working memory for a new segment.

        Args:
            B: batch size for the new segment.
            device: target device.
            clear_neuromod_carryover: if True, also clear `_prev_snapshot_*`
                and `_active_delta_nm`. Use this for INDEPENDENT-document
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
            self._prev_snapshot_ids = None
            self._prev_snapshot_feats = None
            self._active_delta_nm = None

        self.s = torch.zeros(B, cfg.N, cfg.D_s, device=device, dtype=dtype_fast)
        # prev_motor: last step's motor output, chained into next step's
        # start-col query. Starts at zero each segment.
        self.prev_motor = torch.zeros(B, cfg.D_s, device=device, dtype=dtype_fast)
        self.walker_pos = torch.zeros(
            B, cfg.n_heads, device=device, dtype=torch.long,
        )
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

        # Invalidate block-level caches at segment boundary.
        self._horizon_logits_cache = None
        self._alpha_cache = None
        self._input_keys_cache = None
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
        self._prev_snapshot_ids = None
        self._prev_snapshot_feats = None
        self._active_delta_nm = None

    # -----------------------------------------------------------------
    # Neuromodulator plumbing
    # -----------------------------------------------------------------

    @torch._dynamo.disable
    def _begin_plastic_window(self) -> None:
        """Called at the start of each plasticity window.

        Consumes the snapshot of the previous window's touched columns and
        produces `_active_delta_nm` — a grad-carrying delta added to
        `E_bias_flat.detach()` during routing this window.
        """
        if self.neuromod is None or self._prev_snapshot_feats is None:
            self._active_delta_nm = None
            return
        touched_ids = self._prev_snapshot_ids
        feats = self._prev_snapshot_feats
        if touched_ids.numel() < 2:
            # Need at least 2 touched columns for any edge between them.
            self._active_delta_nm = None
            return

        adj_bias = build_adjacency_bias(touched_ids, self.out_nbrs)
        edge_src_local, edge_dst_local, edge_flat = enumerate_touched_edges(
            touched_ids, self.out_nbrs, self.cfg.K,
        )
        if edge_flat.numel() == 0:
            self._active_delta_nm = None
            return

        targets = self.neuromod(
            feats, adj_bias, edge_src_local, edge_dst_local,
        )                                                   # [E] tanh-clamped targets
        # Convert target → delta via EMA blend gated by γ:
        #   active_E_bias = (1 - γ)·E_bias_base + γ·target
        #                 = E_bias_base + γ·(target - E_bias_base)
        # so _active_delta_nm[edge] = γ · (target[edge] - E_bias_base[edge]).
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
        self._active_delta_nm = dense.index_copy(0, edge_flat, delta_edge)

    def _active_e_bias(self) -> torch.Tensor:
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
        if self._active_delta_nm is None:
            return base
        return base + self.cfg.neuromod_eta * self._active_delta_nm

    @torch._dynamo.disable
    def _snapshot_touched_columns(self) -> None:
        """Snapshot stats for columns visited in the window just closed.

        Stored in `_prev_snapshot_ids`, `_prev_snapshot_feats` (both detached)
        for consumption by the NEXT window's `_begin_plastic_window`.
        Called at window close, before counters are reset.
        """
        if self.neuromod is None or self.visit_count is None:
            return

        # Touched columns (visited at least once this window)
        touched_mask = self.visit_count > 0
        touched_ids = torch.nonzero(touched_mask, as_tuple=False).squeeze(-1)
        if touched_ids.numel() == 0:
            self._prev_snapshot_ids = None
            self._prev_snapshot_feats = None
            return

        # Per-column features: mean state across batch, column identity,
        # visit count (log-scaled for sane dynamic range).
        with torch.no_grad():
            s_mean = self.s.float().mean(dim=0)                        # [N, D_s]
            s_feats = s_mean[touched_ids]                              # [U, D_s]
            id_feats = self.col_id[touched_ids].float()                # [U, D_id]
            vc_feats = (self.visit_count[touched_ids] + 1.0).log().unsqueeze(-1)
            feats = torch.cat([s_feats, id_feats, vc_feats], dim=-1)

        self._prev_snapshot_ids = touched_ids.detach()
        self._prev_snapshot_feats = feats.detach()

    def detach_state(self) -> None:
        """TBPTT boundary: preserve values, sever gradient graph."""
        if not self._state_initialized:
            return
        self.s = self.s.detach()
        self.prev_motor = self.prev_motor.detach()
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
        self._input_keys_cache = None
        self._k_all_cache = None
        # Neuromod delta must also get a fresh grad_fn for the next block.
        # The underlying snapshot is already detached, so rebuilding
        # `_active_delta_nm` from it gives a fresh grad path while the
        # value remains identical.
        self._active_delta_nm = None
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
        # New-window flag: True at segment start and on the token right
        # after a plasticity fire. That's when we re-anchor walkers. Within
        # a window walkers just roam using walker_state + token steering.
        is_new_window = self.window_len == 0
        # Active E_bias = frozen persistent base + grad-carrying neuromod
        # delta (if any). When the neuromod is off, this is just E_bias_flat.
        e_bias_active = self._active_e_bias()
        if self._compiled_step is not None:
            out = self._compiled_step(
                self.s, self.walker_pos, self.walker_state,
                self.prev_motor, e_bias_active,
                token_id, tau, epsilon, is_new_window,
            )
        else:
            out = self._step_core_pure(
                self.s, self.walker_pos, self.walker_state,
                self.prev_motor, e_bias_active,
                token_id, tau, epsilon, is_new_window,
            )
        self._apply_step_state(out)
        return WalkerCoreReadout(
            motor_state=out.motor_state,
            visit_freq_step=out.visit_freq_step,
            load_balance_loss=out.load_balance_loss,
        )

    # NOTE: A block-level checkpoint helper used to live here. It was
    # removed because (a) it did not call _record_surprise_token or
    # _maybe_finalize_surprise_and_plasticity, so plasticity and surprise
    # EMA would silently freeze on that path, and (b) wall-time showed
    # checkpointing cost more than it saved in this model (autograd
    # still traversed the recomputed per-token graph). See commit log
    # for the original version if a re-introduction is ever warranted.

    def _apply_step_state(self, out: WalkerCorePureOutput) -> None:
        """Write pure-step outputs into self.* state (non-grad accumulators)."""
        self.s = out.s_new
        self.walker_pos = out.walker_pos_new
        self.walker_state = out.walker_state_new
        self.prev_motor = out.prev_motor_new
        with torch.no_grad():
            self.co_visit_flat = self.co_visit_flat + out.co_visit_delta
            if self.visit_count is not None:
                self.visit_count = self.visit_count + out.visit_count_delta
        self.window_len += 1
        self.tick_counter += 1
        # Phase-2 log_pi accumulation: kept outside the pure step so a
        # compiled or checkpointed _step_core_pure stays side-effect-free.
        # Gradient must remain connected (no detach), so the addition runs
        # outside the no_grad block above.
        if out.log_pi_step is not None:
            self._log_pi_sum = (
                out.log_pi_step if self._log_pi_sum is None
                else self._log_pi_sum + out.log_pi_step
            )

    def step(self, token_id: torch.Tensor) -> WalkerReadout:
        """Public single-token API: graph core + immediate readout.

        This is a smoke/debug path, not the training path. It does NOT
        accumulate surprise (that only happens via `accumulate_block_ce`
        from `phase1_step`) and does NOT fire plasticity (firing here
        would both use a stale/zero `surprise_ema` AND rebuild the
        neuromod's `_active_delta_nm` without the subsequent
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
        """Return the accumulated phase-2 log π sum and clear the buffer.

        Used by `grpo_step` after the prefix pass. Returns None when no
        phase-2 routing decisions were recorded since the last
        `begin_segment` call.
        """
        out = self._log_pi_sum
        self._log_pi_sum = None
        return out

    # -----------------------------------------------------------------
    # Block-level functional API (compile target for whole-block CUDA graphs)
    # -----------------------------------------------------------------

    def block_forward(
        self,
        s_in: torch.Tensor,                 # [B, N, D_s]
        walker_pos_in: torch.Tensor,        # [B, H]
        walker_state_in: torch.Tensor,      # [B, H, D_s]
        prev_motor_in: torch.Tensor,        # [B, D_s]
        e_bias_in: torch.Tensor,            # [N*K] fp32
        tokens_block: torch.Tensor,         # [B, T_block]
        tau: torch.Tensor,                  # scalar fp32
        epsilon: torch.Tensor,              # scalar fp32
        anchor_at_t0: bool,                 # True if t=0 of this block is a new plasticity window
    ) -> BlockOutput:
        """Run T_block sequential _step_core_pure calls in one autograd graph.

        Compiling THIS function with torch.compile (instead of compiling
        per-step) lets inductor fuse across step boundaries and produces
        ONE forward + ONE backward graph for the whole T_block window.
        That's the difference between 1.92× (per-step compile) and
        ~3.7× (whole-block compile) over eager — used by the standalone
        cudagraph trainer (no Llama). The pretrained-LM integration
        path uses ``step_core_from_h`` per token instead.

        State threads through return values — no self.* mutations during
        the loop body — so the compiled graph is free of side effects.
        Caller is responsible for writing the returned state back to
        self.* (or keeping it as a local var for next block_forward call).

        anchor_at_t0 controls whether step 0 is a window-start anchor pick
        (True at the start of every plasticity window) or a regular
        interior step. dynamo specialises the boolean automatically and
        produces two compiled subgraphs as needed.
        """
        cfg = self.cfg
        _, T_block = tokens_block.shape

        s = s_in
        walker_pos = walker_pos_in
        walker_state = walker_state_in
        prev_motor = prev_motor_in

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
            is_new = (t == 0) and anchor_at_t0
            out = self._step_core_pure(
                s, walker_pos, walker_state, prev_motor,
                e_bias_in,
                tokens_block[:, t], tau, epsilon, is_new,
            )
            s = out.s_new
            walker_pos = out.walker_pos_new
            walker_state = out.walker_state_new
            prev_motor = out.prev_motor_new
            motor_list.append(out.motor_state)
            co_visit_total = co_visit_total + out.co_visit_delta
            visit_count_total = visit_count_total + out.visit_count_delta
            lb_loss_total = lb_loss_total + out.load_balance_loss

        motor_states_bt = torch.stack(motor_list, dim=1)  # [B, T_block, D_s]

        return BlockOutput(
            s_new=s,
            walker_pos_new=walker_pos,
            walker_state_new=walker_state,
            prev_motor_new=prev_motor,
            motor_states_bt=motor_states_bt,
            co_visit_total=co_visit_total,
            visit_count_total=visit_count_total,
            load_balance_loss=lb_loss_total,
        )

    def compile_block(
        self, mode: str = "default", fullgraph: bool = True,
    ) -> None:
        """Compile `block_forward` for whole-block CUDA-graph capture.

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
        self._compiled_block = torch.compile(
            self.block_forward, mode=mode, fullgraph=fullgraph,
        )

    def block_forward_from_h(
        self,
        s_in: torch.Tensor,                 # [B, N, D_s]
        walker_pos_in: torch.Tensor,        # [B, H]
        walker_state_in: torch.Tensor,      # [B, H, D_s]
        prev_motor_in: torch.Tensor,        # [B, D_s]
        e_bias_in: torch.Tensor,            # [N*K] fp32
        h_mem_block: torch.Tensor,          # [B, T_block, D_s]
        tau: torch.Tensor,                  # scalar fp32
        epsilon: torch.Tensor,              # scalar fp32
        anchor_at_t0: bool,
    ) -> BlockOutput:
        """Block-level analog of `block_forward` driven by an externally-
        supplied [B, T_block, D_s] hidden-state stream instead of token ids.
        Used by the pretrained-LM integration (`forward_segment`): one
        compiled call per `tbptt_block` window replaces T_block per-token
        `step_core_from_h` calls. Inductor fuses across step boundaries,
        reproducing the standalone whole-block speedup (~3.7× over eager).

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
        prev_motor = prev_motor_in

        # Dummy token_id — `_step_core_pure` ignores it when
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
            is_new = (t == 0) and anchor_at_t0
            out = self._step_core_pure(
                s, walker_pos, walker_state, prev_motor,
                e_bias_in,
                dummy_tok, tau, epsilon, is_new,
                h_input_override=h_mem_block[:, t],
            )
            s = out.s_new
            walker_pos = out.walker_pos_new
            walker_state = out.walker_state_new
            prev_motor = out.prev_motor_new
            motor_list.append(out.motor_state)
            co_visit_total = co_visit_total + out.co_visit_delta
            visit_count_total = visit_count_total + out.visit_count_delta
            lb_loss_total = lb_loss_total + out.load_balance_loss

        motor_states_bt = torch.stack(motor_list, dim=1)  # [B, T_block, D_s]

        return BlockOutput(
            s_new=s,
            walker_pos_new=walker_pos,
            walker_state_new=walker_state,
            prev_motor_new=prev_motor,
            motor_states_bt=motor_states_bt,
            co_visit_total=co_visit_total,
            visit_count_total=visit_count_total,
            load_balance_loss=lb_loss_total,
        )

    def _block_forward_from_h_ckpt(
        self,
        s_in: torch.Tensor,
        walker_pos_in: torch.Tensor,
        walker_state_in: torch.Tensor,
        prev_motor_in: torch.Tensor,
        e_bias_in: torch.Tensor,
        h_mem_block: torch.Tensor,
        tau: torch.Tensor,
        epsilon: torch.Tensor,
        anchor_at_t0: bool,
    ) -> BlockOutput:
        """Activation-checkpointed wrapper around `_compiled_block_from_h`.

        Internally calls the compiled block under
        `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, returning
        the BlockOutput's tensor fields as a tuple so the checkpointing
        machinery's "output must be tensors" requirement is satisfied
        cleanly. We unpack back into BlockOutput for the caller.

        Without this wrapper, the compiled block's per-step intermediates
        (saved-for-backward of every `_step_core_pure` call across the
        block) stay alive until segment-end backward — empirically the
        largest memory contributor at integration scale.
        """
        from torch.utils.checkpoint import checkpoint

        def _fn(s, wp, ws, pm, eb, hm, t_, e_, isnew_):
            r = self._compiled_block_from_h(
                s, wp, ws, pm, eb, hm, t_, e_, isnew_,
            )
            # Return as tuple of tensors for checkpoint compatibility.
            return (
                r.s_new, r.walker_pos_new, r.walker_state_new,
                r.prev_motor_new, r.motor_states_bt,
                r.co_visit_total, r.visit_count_total, r.load_balance_loss,
            )

        out_tuple = checkpoint(
            _fn,
            s_in, walker_pos_in, walker_state_in, prev_motor_in,
            e_bias_in, h_mem_block, tau, epsilon, anchor_at_t0,
            use_reentrant=False,
        )
        return BlockOutput(
            s_new=out_tuple[0],
            walker_pos_new=out_tuple[1],
            walker_state_new=out_tuple[2],
            prev_motor_new=out_tuple[3],
            motor_states_bt=out_tuple[4],
            co_visit_total=out_tuple[5],
            visit_count_total=out_tuple[6],
            load_balance_loss=out_tuple[7],
        )

    def compile_block_from_h(
        self, mode: str = "default", fullgraph: bool = True,
    ) -> None:
        """Compile `block_forward_from_h` for whole-block fusion in the
        pretrained-LM integration path. ``forward_segment`` routes through
        `_compiled_block_from_h` when set — one compiled call per
        `tbptt_block` window instead of T_block per-token calls.

        ``mode="default"`` (~3.7× over eager) is the safe choice for the
        Llama integration: gives inductor's cross-step fusion without
        cudagraph's stable-pointer constraints. The walker is embedded in
        Llama's autograd graph, so the cudagraph variant
        (``"reduce-overhead"``) would conflict with Llama's dynamic
        activation addresses.
        """
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64,
        )
        self._compiled_block_from_h = torch.compile(
            self.block_forward_from_h, mode=mode, fullgraph=fullgraph,
        )

    def step_core_from_h(self, h_input: torch.Tensor) -> WalkerCoreReadout:
        """Hot core driven by an externally-supplied `[B, D_s]` vector
        instead of a `token_id`. Used by the pretrained-LM integration
        (`forward_segment`) where h_mem_t = W_in(llama_hidden_state_t)
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
        is_new_window = self.window_len == 0
        e_bias_active = self._active_e_bias()
        step_fn = self._compiled_step if self._compiled_step is not None \
                  else self._step_core_pure
        out = step_fn(
            self.s, self.walker_pos, self.walker_state,
            self.prev_motor, e_bias_active,
            dummy_tok, tau, epsilon, is_new_window,
            h_input_override=h_input,
        )
        self._apply_step_state(out)
        return WalkerCoreReadout(
            motor_state=out.motor_state,
            visit_freq_step=out.visit_freq_step,
            load_balance_loss=out.load_balance_loss,
        )

    def forward_segment(
        self,
        h_mem: torch.Tensor,                     # [B, T, D_s]
        input_ids: torch.Tensor,                 # [B, T] int64 (targets + surprise)
        adapter: "MemAdapter | None" = None,     # Llama adapter for aux CE
        *,
        compute_aux_loss: bool = True,
        preserve_graph: bool = False,
        walker_aux_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Process one segment of pretrained-LM hidden states through the
        walker. Returns per-token walker readouts in graph-state space,
        plus an optional aux CE (motor_state → adapter.mem_head_logits →
        CE vs shifted input_ids).

        Semantics match `v2`'s `MemoryGraph.forward_segment` so the same
        `MemInjectLayer` + cycle-loop scaffolding in `src/pretrained/`
        can wire this in with minimal changes.

        State contract:
        - Caller is responsible for `begin_segment(B, device)` (via
          `wrapper.reset_memory(bs)`) before the first segment. This
          function does NOT re-init state — it only processes the segment.
        - TBPTT: state is detached every `cfg.tbptt_block` tokens within
          the segment. Skipped when `preserve_graph=True` (AR unroll).
        - Plasticity + surprise fold fire at `cfg.mod_period` cadence,
          using the walker's OWN multi-horizon CE against upcoming
          `input_ids` (not Llama's CE). Cheap, local, avoids a Llama-side
          dependency for plasticity.

        Returned aux loss combines (when `compute_aux_loss=True`):
          1. Walker-side multi-horizon CE (gradient flows to state_to_model,
             walker hot path). Always included when compute_aux_loss.
             Scaled by `walker_aux_weight`.
          2. Llama-side horizon-1 CE via `adapter.mem_head_logits` (gradient
             flows to W_out, Llama norm/lm_head are frozen). Included only
             when adapter is provided.
        """
        B, T, D_s = h_mem.shape
        assert D_s == self.cfg.D_s, (
            f"h_mem last-dim {D_s} must match cfg.D_s {self.cfg.D_s}"
        )
        assert input_ids.shape == (B, T), (
            f"input_ids shape {tuple(input_ids.shape)} must be [B, T]={(B, T)}"
        )
        assert self._state_initialized, (
            "call begin_segment(B, device) before forward_segment()"
        )

        device = h_mem.device
        cfg = self.cfg
        tbptt = cfg.tbptt_block
        K_h = cfg.K_horizons

        # Walker hot path runs in `state_dtype` (bf16 on CUDA, fp32 on CPU).
        # If the caller runs us on CUDA WITHOUT autocast, the bf16 column
        # state vs fp32 walker weights would mismatch in `content_mlp`,
        # `q_proj`, etc. Defensively enter autocast at the top of the
        # processing loop so inference / benchmark callers don't need to
        # know about the requirement. When autocast is already active
        # (training harnesses set it), this nests cleanly.
        if device.type == "cuda" and not torch.is_autocast_enabled():
            return self._forward_segment_with_autocast(
                h_mem, input_ids, adapter,
                compute_aux_loss=compute_aux_loss,
                preserve_graph=preserve_graph,
                walker_aux_weight=walker_aux_weight,
            )

        motor_blocks: list[torch.Tensor] = []
        # Horizon weights mirror `phase1_step`: horizon-1 is primary (1.0),
        # rest (0.2). Keeps walker-side aux semantically aligned with
        # standalone training.
        horizon_weights = torch.full(
            (K_h,), 0.2, device=device, dtype=torch.float32,
        )
        horizon_weights[0] = 1.0
        # Accumulators for the two aux-loss components (gradient-carrying).
        walker_ce_sum = torch.zeros(K_h, device=device, dtype=torch.float32)
        walker_ce_count = torch.zeros(K_h, device=device, dtype=torch.float32)
        llama_ce_sum = torch.zeros((), device=device, dtype=torch.float32)
        llama_ce_count = 0

        # Block-stride loop. Whole-block path replaces T_block per-token
        # `step_core_from_h` calls with one `block_forward_from_h` call —
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
                ticks == tbptt and self._compiled_block_from_h is not None
            )
            # Phase-2 GRPO needs `log_pi_step` from `_step_core_pure` to
            # accumulate `_log_pi_sum` for `consume_log_pi_sum()`. The
            # block path drops `log_pi_step` (BlockOutput has no field
            # for it), so silently falling into the block path during
            # phase 2 would null the policy gradient. Force the per-
            # token fallback in that case so log_pi accumulation stays
            # alive via `_apply_step_state`.
            if use_block_path and self.phase == "phase2":
                use_block_path = False

            if use_block_path:
                # Set up block-static caches once per block. _step_core_pure
                # reads these from self.* (they're not arguments).
                self._ensure_block_caches(self.tied_token_emb.weight)
                is_new_window = self.window_len == 0
                e_bias_active = self._active_e_bias()
                # Dummy token_id satisfies _step_core_pure's signature; it
                # ignores it when h_input_override is supplied.
                dummy_tok = torch.zeros(B, dtype=torch.int64, device=device)
                tau, epsilon = self._schedule_tensors(dummy_tok)

                # Activation checkpointing on the whole-block forward.
                # Without this the compiled `block_forward_from_h` saves
                # per-step intermediates across all `tbptt_block` token steps
                # for backward — at production scale that's the largest
                # single contributor to peak memory (~7 GB at BS=4 from
                # the memprof). With checkpoint, the block forward gets
                # re-run during backward instead. ~10-20% wall-time cost on
                # the walker portion, ~5-6 GB freed at BS=4 (linear in B).
                # Disabled in eval / when explicitly opted out.
                ckpt_block = (
                    self.training
                    and getattr(self, "_checkpoint_block", True)
                )
                if ckpt_block:
                    out = self._block_forward_from_h_ckpt(
                        self.s, self.walker_pos, self.walker_state,
                        self.prev_motor, e_bias_active,
                        h_mem[:, block_start:block_end],
                        tau, epsilon, is_new_window,
                    )
                else:
                    out = self._compiled_block_from_h(
                        self.s, self.walker_pos, self.walker_state,
                        self.prev_motor, e_bias_active,
                        h_mem[:, block_start:block_end],
                        tau, epsilon, is_new_window,
                    )

                # State writeback — block_forward_from_h is pure-functional
                # so callers must thread state explicitly. Mirrors what
                # _apply_step_state does for the per-token path, but for
                # the whole block in one shot.
                self.s = out.s_new
                self.walker_pos = out.walker_pos_new
                self.walker_state = out.walker_state_new
                self.prev_motor = out.prev_motor_new
                with torch.no_grad():
                    self.co_visit_flat = self.co_visit_flat + out.co_visit_total
                    if self.visit_count is not None:
                        self.visit_count = (
                            self.visit_count + out.visit_count_total
                        )
                self.window_len += ticks
                self.tick_counter += ticks
                # NOTE: block_forward_from_h drops log_pi_step. Phase-2 GRPO
                # callers stay on the per-token preserve_graph=True path
                # (T=1 per call), which keeps log_pi accumulation alive
                # via _apply_step_state. If phase-2 ever wants block-path
                # speedup, BlockOutput needs a log_pi_total field.

                motor_bt = out.motor_states_bt                          # [B, ticks, D_s]
            else:
                # Per-token fallback (partial last block, AR unroll, or
                # compile not set up). Mutates self.* via step_core_from_h.
                block_motor_list: list[torch.Tensor] = []
                for t in range(block_start, block_end):
                    r = self.step_core_from_h(h_mem[:, t])
                    block_motor_list.append(r.motor_state)
                motor_bt = torch.stack(block_motor_list, dim=1)         # [B, ticks, D_s]

            motor_blocks.append(motor_bt)

            # --- Walker's own multi-horizon CE.
            # Shape logic matches phase1_step.flush(); uses the walker's
            # internal tied_token_emb readout, NOT Llama's. This same CE
            # drives (a) the walker-side aux loss, gradient-carrying, and
            # (b) the surprise EMA fold, detached.
            i_idx = torch.arange(ticks, device=device)
            k_idx = torch.arange(1, K_h + 1, device=device)
            t_idx = block_start + i_idx.unsqueeze(1) + k_idx.unsqueeze(0)  # [ticks, K_h]
            valid_tk = t_idx < T
            block_has_targets = bool(valid_tk.any().item())
            # Targetless block (typical: T=1 generation step inside an
            # autoregressive rollout, where there are no upcoming tokens
            # in the segment to predict). Skip the dense vocab readout
            # AND the surprise / plasticity finalize. Otherwise we'd
            # both waste a [B, T, K_h, V] CE compute on an all-False
            # mask, and pollute E_bias by firing plasticity off
            # accumulated co-visit with stale (or zero) surprise EMA.
            if block_has_targets:
                t_idx_clamped = t_idx.clamp(max=T - 1)
                targets = input_ids.index_select(
                    1, t_idx_clamped.reshape(-1),
                ).reshape(B, ticks, K_h)
                valid_btk = valid_tk.unsqueeze(0).expand(B, -1, -1)

                with torch.autocast(device_type=device.type, enabled=False):
                    ce_masked = self.readout_ce_block(
                        motor_bt, targets, valid_btk,
                    )                                              # [B, ticks, K_h]
                if compute_aux_loss:
                    walker_ce_sum = walker_ce_sum + ce_masked.sum(dim=(0, 1))
                    walker_ce_count = (
                        walker_ce_count + valid_tk.float().sum(dim=0) * B
                    )
                self.accumulate_block_ce(ce_masked.detach(), valid_tk.detach())
                self._maybe_finalize_surprise_and_plasticity()

            # --- Llama-side aux CE (horizon-1 against adapter.mem_head_logits).
            if compute_aux_loss and adapter is not None:
                aux_valid_i = (block_start + i_idx + 1) < T
                if aux_valid_i.any():
                    motor_flat = motor_bt.reshape(B * ticks, D_s)
                    mem_logits = adapter.mem_head_logits(motor_flat)   # [B*ticks, vocab_lm]
                    aux_t_idx = (block_start + i_idx + 1).clamp(max=T - 1)
                    tgt = input_ids.index_select(1, aux_t_idx)         # [B, ticks]
                    tgt_flat = tgt.reshape(B * ticks)
                    valid_flat = aux_valid_i.unsqueeze(0).expand(B, -1).reshape(-1)
                    ce_all = F.cross_entropy(
                        mem_logits.float(), tgt_flat, reduction="none",
                    )                                                  # [B*ticks]
                    llama_ce_sum = llama_ce_sum + (ce_all * valid_flat.float()).sum()
                    llama_ce_count += int(valid_flat.sum().item())

            # TBPTT detach at block boundary (unless AR unroll preserving graph).
            if not preserve_graph and block_end < T:
                self.detach_state()

            block_start = block_end

        readouts = torch.cat(motor_blocks, dim=1)                       # [B, T, D_s]

        aux_loss: torch.Tensor | None = None
        if compute_aux_loss:
            # Walker-side: horizon-weighted mean CE (gradient-carrying).
            has_counts = walker_ce_count > 0
            per_h_mean = walker_ce_sum / walker_ce_count.clamp(min=1)
            denom = horizon_weights[has_counts].sum().clamp(min=1)
            walker_mh_ce = (
                per_h_mean * horizon_weights * has_counts.float()
            ).sum() / denom
            aux_loss = walker_aux_weight * walker_mh_ce
            # Llama-side: horizon-1 mean (gradient to W_out, walker).
            if adapter is not None and llama_ce_count > 0:
                aux_loss = aux_loss + (llama_ce_sum / llama_ce_count)
        return readouts, aux_loss

    def _forward_segment_with_autocast(
        self,
        h_mem: torch.Tensor,
        input_ids: torch.Tensor,
        adapter,
        *,
        compute_aux_loss: bool,
        preserve_graph: bool,
        walker_aux_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Defensive wrapper that enters bf16 autocast on CUDA before
        recursing into `forward_segment`. Used when a caller runs us on
        CUDA without first establishing an autocast region — without it,
        the bf16 column state would mismatch the fp32 walker weights in
        `content_mlp` / `q_proj`."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.forward_segment(
                h_mem, input_ids, adapter,
                compute_aux_loss=compute_aux_loss,
                preserve_graph=preserve_graph,
                walker_aux_weight=walker_aux_weight,
            )

    def _step_core_pure(
        self,
        s_in: torch.Tensor,              # [B, N, D_s]
        walker_pos_in: torch.Tensor,     # [B, H]
        walker_state_in: torch.Tensor,   # [B, H, D_s]
        prev_motor_in: torch.Tensor,     # [B, D_s]
        e_bias_flat_in: torch.Tensor,    # [N*K] fp32 — snapshot of plastic bias
        token_id: torch.Tensor,          # [B] int64 (ignored if h_input_override given)
        tau: torch.Tensor,               # scalar
        epsilon: torch.Tensor,           # scalar
        is_new_window: bool,             # True at segment start AND after plasticity fires
        h_input_override: torch.Tensor | None = None,   # [B, D_s] — bypass token embed
    ) -> WalkerCorePureOutput:
        """Pure-functional one-token walker update.

        Flow (write-first-then-route):
          1. Token embed + per-walker broadcast.
          2. If `is_new_window`: re-pick H anchor columns on the input plane
             via Gumbel STE, teleport walkers there, and add one STE-gated
             injection of token content (the gradient path for input_q_proj).
             Otherwise walkers stay at `walker_pos_in` and no anchor
             injection happens this step.
          3. Steering input (pre-update): cat(s_cur_old, id_cur,
             walker_state, token_per_walker). Feed content_mlp → m_out.
          4. Sparse deposit: walker writes at CURRENT column cur_bh. On a
             new-window step the anchor injection is stacked in the same
             LIF call (both land at the anchor cols).
          5. Re-read s[cur_bh] (post-update).
          6. Steering input (post-update): cat(s_cur_new, id_cur,
             walker_state, token_per_walker). Feed q_proj → hop scores.
             Gumbel top-1 picks next_col.
          7. STE-gated endpoint readout: end_state =
             Σ_k ste_weights[k] · s_new[nbrs[cur,k]]. Routing gradient
             bridges to loss through this soft sum.
          8. Cross-attn over endpoints → motor_state.
          9. Walker state update: EMA of m_out (per-walker decay).

        Takes current state (s, walker_pos, walker_state, prev_motor) as
        explicit inputs and returns new state plus non-grad accumulator
        deltas. No mutation of self.* here; caller's job.
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
        #    In that case h_input_model is None — anchor v_inject falls
        #    back to mem_input_v_proj (see block 2 below).
        if h_input_override is None:
            h_input_model = self.tied_token_emb(token_id)             # [B, D_model]
            h_input = self.token_to_state(h_input_model)              # [B, D_s]
        else:
            h_input_model = None
            h_input = h_input_override                                # [B, D_s]

        # 2. Window-boundary anchor pick (only when a new plasticity window
        # is starting). Within a window, walkers roam using their persistent
        # walker_pos + walker_state; the anchor mechanism is dormant.
        if is_new_window:
            query_input = h_input + self.prev_motor_proj(
                prev_motor_in.to(h_input.dtype),
            )
            query_flat = self.input_q_proj(query_input).view(
                B, cfg_n_heads, cfg.D_q_in,
            )
            input_keys = self._input_keys_cache
            scale_in = 1.0 / (cfg.D_q_in ** 0.5)
            scores_in = torch.einsum(
                "bhd,nd->bhn", query_flat, input_keys,
            ) * scale_in
            scores_in_flat = scores_in.reshape(BH, self._N_in)
            rout_in = gumbel_top1_softmax(
                scores_in_flat, tau=tau, epsilon=epsilon, training=is_training,
                phase=self.phase,
            )
            if rout_in.log_pi is not None:
                # Sum per-batch-element by reducing the H walker dim. Returned
                # via WalkerCorePureOutput.log_pi_step so the side effect lives
                # in `_apply_step_state`, keeping `_step_core_pure` truly pure
                # (compatible with torch.compile and checkpoint recompute).
                log_pi_step = rout_in.log_pi.view(B, cfg_n_heads).sum(dim=1)  # [B]
            else:
                log_pi_step = None
            start_local = rout_in.selected_idx.view(B, cfg_n_heads)
            start_cols = self.input_positions[start_local]            # [B, H]
            cur_bh = start_cols.reshape(BH)                           # teleport

            if h_input_model is not None:
                v_inject = self.input_v_proj(h_input_model)           # [B, D_s]
            else:
                # Pretrained-LM path: no D_model-dim h; take v_inject
                # from the D_s-dim Llama-projected h via mem_input_v_proj.
                v_inject = self.mem_input_v_proj(h_input)             # [B, D_s]
            input_ste_weights = torch.gather(
                rout_in.ste_weights, 1, rout_in.selected_idx.unsqueeze(1),
            ).squeeze(1)                                              # [B*H]
            inject_msg = (
                v_inject.unsqueeze(1).expand(B, cfg_n_heads, cfg.D_s)
                .reshape(BH, cfg.D_s)
            )
            inject_msg = (
                inject_msg * input_ste_weights.unsqueeze(-1).to(inject_msg.dtype)
            )

            # Load-balance mass from the anchor softmax (over input-plane cols).
            P_mass = torch.zeros(cfg.N, device=device, dtype=lb_dtype)
            p_in_per_col = rout_in.soft_probs.to(lb_dtype).sum(dim=0)
            P_mass = P_mass.index_add(0, self.input_positions, p_in_per_col)
        else:
            cur_bh = walker_pos_in.reshape(BH)
            start_cols = None
            inject_msg = None
            P_mass = torch.zeros(cfg.N, device=device, dtype=lb_dtype)
            log_pi_step = None

        # 3. Walker message from OLD column state + walker state + token.
        # ``s_in.detach()`` here cuts the gather backward into [B, N, D_s]
        # zeros + scatter (~50% of CUDA time at large N per kineto profile).
        # The substrate read becomes a constant input feature for content_mlp;
        # gradient still flows to content_mlp's params via this step's loss
        # via the s_cur_new (post-LIF) gather below — that one is left
        # ATTACHED so the LIF chain backward fires (decay_proj, input_v_proj
        # need that path). Cross-step substrate gradient is preserved via
        # walker_state EMA + LIF α-channel through s_cur_new. See Phase 1
        # detach-hack rationale in graph-walker branch commit c3d7e25.
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
        if self.cols.per_plane_content:
            plane_of_cur = self.plane_ids[cur_bh]
            m_out = self.cols.content_mlp(cat_pre, plane_of_cur)
        else:
            m_out = self.cols.content_mlp(cat_pre)                    # [B*H, D_s]

        # 4. Sparse deposit. Walker always writes at its current column
        # (no STE gating on m_out — routing's gradient comes through the
        # endpoint readout below). On a new-window step the anchor
        # injection is stacked in the same LIF call; both message sets land
        # at the anchor cols, which are also the walker's current position.
        # Always pad to static M_max = 2*BH with sentinel BN for unused
        # dest slots so the Triton LIF kernel sees a fixed grid shape across
        # interior + window-start steps (required for cudagraph capture).
        batch_idx = self._batch_idx
        BN = B * cfg.N
        if is_new_window:
            all_dests = torch.cat([
                batch_idx * cfg.N + start_cols.reshape(BH),
                batch_idx * cfg.N + cur_bh,
            ], dim=0).contiguous()
            all_msgs = torch.cat(
                [inject_msg, m_out], dim=0,
            ).to(state_dtype).contiguous()
        else:
            real_dests = batch_idx * cfg.N + cur_bh
            pad_dests = torch.full(
                (BH,), BN, dtype=real_dests.dtype, device=device,
            )
            all_dests = torch.cat([real_dests, pad_dests], dim=0).contiguous()
            pad_msgs = torch.zeros(
                BH, cfg.D_s, dtype=state_dtype, device=device,
            )
            all_msgs = torch.cat(
                [m_out.to(state_dtype), pad_msgs], dim=0,
            ).contiguous()

        s_flat = s_in.reshape(B * cfg.N, cfg.D_s)
        s_flat_new = sparse_lif_update(s_flat, all_msgs, all_dests, alpha, cfg.N)
        s_new = s_flat_new.view(B, cfg.N, cfg.D_s)

        # 5. Re-read walker's current column post-update.
        s_cur_new = torch.gather(
            s_new, 1, cur.unsqueeze(-1).expand(B, cfg_n_heads, cfg.D_s),
        ).reshape(BH, cfg.D_s)

        # 6. Routing query uses POST-UPDATE state.
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

        rout = gumbel_top1_softmax(
            scores, tau=tau, epsilon=epsilon, training=is_training,
            phase=self.phase,
        )
        if rout.log_pi is not None:
            lp = rout.log_pi.view(B, cfg_n_heads).sum(dim=1)            # [B]
            log_pi_step = lp if log_pi_step is None else log_pi_step + lp
        next_local = rout.selected_idx                                # [B*H]
        next_col = torch.gather(
            nbrs_of_cur, 1, next_local.unsqueeze(-1),
        ).squeeze(-1)                                                 # [B*H]
        edge_taken_flat = cur_bh * cfg.K + next_local                 # [B*H]

        P_mass = P_mass.index_add(
            0, nbrs_of_cur.reshape(-1),
            rout.soft_probs.to(lb_dtype).reshape(-1),
        )

        # visit_count delta (non-grad). Trajectory footprint fed to the
        # neuromod at window close. Anchors only contribute on window-start
        # steps (once per 128 tokens, not per token).
        visit_count_delta = torch.zeros(cfg.N, device=device, dtype=torch.float32)
        if is_new_window:
            visit_count_delta.scatter_add_(0, start_cols.reshape(-1), self._ones_bh)
        visit_count_delta.scatter_add_(0, next_col, self._ones_bh)

        # 7. Endpoint readout = walker's own current-col content + STE-gated
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

        # 8. Cross-attn over endpoint states → motor_state.
        s_traj = self.out_norm(end_states).to(state_dtype)
        k = self.out_k_proj(s_traj)
        v = self.out_v_proj(s_traj)
        q_motor = self.motor_query.to(state_dtype).unsqueeze(0).expand(B, -1)
        scale_out = 1.0 / (cfg.D_s ** 0.5)
        attn_scores = torch.sum(k * q_motor.unsqueeze(1), dim=-1) * scale_out
        attn = F.softmax(attn_scores, dim=-1)
        motor_state = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        prev_motor_new = motor_state

        # 9. Walker state update: EMA of m_out per walker.
        alpha_w = torch.sigmoid(self.walker_state_alpha)              # [H]
        alpha_w_view = alpha_w.view(1, cfg_n_heads, 1).to(walker_state_in.dtype)
        m_out_view = m_out.view(B, cfg_n_heads, cfg.D_s).to(walker_state_in.dtype)
        walker_state_new = (
            alpha_w_view * walker_state_in
            + (1.0 - alpha_w_view) * m_out_view
        )

        # 10. co_visit delta (non-grad)
        co_visit_delta = torch.bincount(
            edge_taken_flat, minlength=cfg.num_edges,
        ).to(torch.float32)

        # On window-start steps both anchor and walker softmaxes contribute
        # mass (2·BH decisions). On interior steps only walker routing
        # contributes (BH decisions). Normalize P_mass and visit counts by
        # the matching divisor.
        n_decisions = float(BH * 2 if is_new_window else BH)
        P_mean = P_mass / n_decisions
        f_mean = (visit_count_delta / n_decisions).detach()
        load_balance_loss = cfg.N * (P_mean * f_mean).sum()

        return WalkerCorePureOutput(
            s_new=s_new,
            walker_pos_new=walker_pos_new,
            walker_state_new=walker_state_new,
            prev_motor_new=prev_motor_new,
            motor_state=motor_state,
            co_visit_delta=co_visit_delta,
            visit_count_delta=visit_count_delta,
            visit_freq_step=f_mean if cfg.lambda_balance > 0 else None,
            load_balance_loss=load_balance_loss,
            log_pi_step=log_pi_step,
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
    def _maybe_finalize_surprise_and_plasticity(self) -> None:
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
          2. If neuromod is enabled: bake the grad-carrying `_active_delta_nm`
             into `E_bias_flat` (detached), then take a new snapshot of the
             just-closed window and produce the next window's delta.

        Wrapped in `torch.no_grad` so the non-diff update can't accidentally
        attach a grad_fn to `E_bias_flat` across repeated calls.
        """
        cfg = self.cfg
        device = self.s.device

        with torch.autocast(device_type=device.type, enabled=False):
            # Normalize by window length
            window = max(self.window_len, 1)
            co_visit_norm = (self.co_visit_flat.float() / window) if self.co_visit_flat is not None \
                else torch.zeros_like(self.E_bias_flat)

            # Scalar-eta Hebbian: surprise-gated global learning rate.
            # Higher surprise → faster learning. Additive to the neuromod.
            surprise_scalar = self.surprise_ema.mean().float()
            eta_global = cfg.plast_eta * torch.sigmoid(
                surprise_scalar - cfg.plast_surprise_bias,
            )

            delta_hebb = eta_global * (
                co_visit_norm - cfg.plast_decay * self.E_bias_flat
            )
            # Neuromod contribution (if any), detached and scaled.
            delta_nm_commit = torch.zeros_like(self.E_bias_flat)
            if self._active_delta_nm is not None:
                delta_nm_commit = (
                    cfg.neuromod_eta * self._active_delta_nm.detach()
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
        self._active_delta_nm = None

        # NOTE: `_begin_plastic_window()` deliberately does NOT run here.
        # This method is @torch.no_grad()'d — running the neuromod forward
        # inside that scope would produce an `_active_delta_nm` without a
        # grad_fn, and routing in the next window would see a grad-free
        # delta, silently killing the neuromod's training signal. The
        # caller (`_maybe_finalize_surprise_and_plasticity`) runs
        # `_begin_plastic_window()` outside this no_grad scope.
