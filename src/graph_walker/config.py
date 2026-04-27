"""GraphWalkerConfig — hyperparams for the trajectory-routed variant.

See docs/graph_walker.md for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphWalkerConfig:
    # --- Graph topology (same spirit as column_graph) ---
    plane_rows: int = 32                # was 24, originally 16. → 1024 cols/plane,
    plane_cols: int = 32                # 4 planes = 4096 total cols. The graph
                                        # is the model's vocabulary of concepts;
                                        # N is the vocab size. Per-token compute
                                        # is invariant to N (walkers visit only
                                        # B·H cols per step regardless of N), so
                                        # N is pure capacity at no runtime cost.
    L: int = 4                          # number of planes (0 = input, L-1 = output)
    K: int = 96                         # out-edges per column (was 64, orig 32).
                                        # With K_intra_fraction=0.5,
                                        # K_inter_bwd_fraction=0.5 → K_intra=48,
                                        # K_inter_fwd=24, K_inter_bwd=24. At
                                        # radius=3, max_intra=48 (exact fit) and
                                        # max_inter=49 (24/49 → comfortable).
                                        # Per-step routing scoring stays in the
                                        # noise relative to content_mlp.
    # p_rewire: Watts-Strogatz rewiring probability. Each edge's destination
    # is, with probability p_rewire, replaced by a uniform-random column
    # anywhere in the graph (any plane, any (r, c)). 0.0 = pure local grid.
    #
    # Default 0.3 (well past the small-world transition at ~0.1).
    # Rationale: under the "graph as syntactic substrate for an emergent
    # high-dimensional language" framing, the topology is the grammar's
    # possibility space — what trajectories the model is even allowed to
    # encode. A pure grid forces local-relations-only "sentences" (Moore
    # radius-2 + plane progression), syntactically impoverished. Heavy
    # rewiring gives the language long-distance syntactic moves at init,
    # before plasticity has had to discover them. Trade-off: weaker spatial
    # prior at init, more work for plasticity to learn which random edges
    # carry meaning. ε-exploration in routing.py provides per-decision
    # stochasticity on top.
    p_rewire: float = 0.3
    K_intra_fraction: float = 0.5
    # Of the K_inter = K - K_intra inter-plane edges, what fraction goes
    # BACKWARD (to plane p-1) vs FORWARD (to plane p+1). Default 0.5 means
    # equal split → no structural feed-forward bias under your "graph as
    # syntactic substrate" framing. Set 0.0 to recover legacy forward-only.
    K_inter_bwd_fraction: float = 0.5
    # Moore-radius for intra/inter-plane neighbour candidate sampling.
    # Radius 2 → 24 intra candidates, 25 inter candidates.
    # Radius 3 → 48 intra candidates, 49 inter candidates (needed for K ≥ ~50
    # since we now split inter into fwd+bwd halves: K_inter_dir ≤ max_inter).
    intra_radius: int = 3               # was 2; bumped to support K=48
    inter_radius: int = 3               # was 2

    # --- Per-plane weight specialisation ---
    # When True, content_mlp has L independent copies (one per plane).
    # Slower on GPU (L small matmuls instead of one big batched matmul)
    # but architecturally principled. Disabled by default in favour of
    # `content_mlp_depth` (deep shared blocks) which hits the same param
    # budget with better throughput.
    per_plane_content_mlp: bool = False
    # When True, content_mlp has H independent copies (one per walker head).
    # MoE-style capacity: total params scale with H, but per-token compute
    # is identical to the shared path since each walker still does one
    # forward pass, just with its own weights.
    per_head_content_mlp: bool = False
    content_mlp_depth: int = 4          # residual FFN blocks at D_s=1024,
                                        # D_hid=4·D_s. 4 stacked nonlinear
                                        # composition steps per walker per
                                        # token. 8.4M params per block.

    # --- Widths ---
    # D_model is the external lexical/readout width (token embedding,
    # tied unembedding, post-readout capacity). D_s is the internal graph
    # state width paid in the per-hop recurrent hot path.
    D_model: int = 1024
    D_s: int = 768                      # column state dim. Original 512;
                                        # tried 1024 (Option A, ~106M params,
                                        # B=4 cudagraph OOM); settled on 768
                                        # (Option B, ~75M params) so we keep
                                        # most of the representational gain
                                        # without the quadratic cost — D_s²
                                        # is what makes content_mlp dominant.
    D_id: int = 512                     # column identity vector, was 32. The
                                        # identity should be a rich semantic
                                        # vector ("what concept does this
                                        # column carry"), not a tiny ID tag.
    ffn_mult_content: int = 4           # content_mlp: D_s + D_id → 4·D_s → D_s
    # Post-readout model-space capacity. These blocks run once per token,
    # not once per hop, so they are a cheaper place to keep params than
    # the walker hot loop. BUT under the "memory graph IS the model" framing
    # we want most parameters in the substrate + plasticity, not in a
    # standard FFN bolted on after the graph is done thinking. Reduced from
    # 7 to 3 to free ~32M params for graph-side capacity.
    post_model_depth: int = 2           # was 7. The thesis is that the
                                        # graph IS the model — the cold
                                        # readout is just a thin lexical
                                        # projection head, not a deep
                                        # transformer FFN tower.
    post_model_ffn_mult: int = 4
    n_score_heads: int = 4              # multi-head bilinear edge scoring
    D_q_per_head: int = 64

    # --- Graph-walker specifics ---
    n_heads: int = 4                    # H — parallel persistent walkers
    # Legacy trajectory-depth knob kept for compatibility with older configs
    # and future history-aware variants. In the current persistent-walker
    # design the fast path advances one hop per token.
    n_hops: int = 4
    D_q_in: int = 64                    # input-plane start-column query dim

    # --- Multi-horizon readout ---
    K_horizons: int = 8
    # Legacy compatibility field from the earlier token-clock surprise path.
    # Current delayed-surprise code keeps the same lower-bound contract so old
    # configs do not silently under-allocate history.
    K_buf: int = 8

    # --- Clocks ---
    # `mod_period` is the plasticity window (and the anchor re-pick cadence
    # since `is_new_window` is derived from it). `tbptt_block` is the
    # gradient-detach cadence. They should match: the neuromod's delta
    # lives for exactly one window, and we want its one gradient event per
    # window to attribute the full window-loss back to its action. Splitting
    # them breaks the walker-trajectory gradient chain mid-window and leaves
    # the neuromod credited only for short-range effects.
    # If activation memory blows up, shorten both together rather than
    # decoupling.
    mod_period: int = 128               # plasticity fires every N tokens
    tbptt_block: int = 128              # gradient detach cadence (== mod_period)

    # --- Plasticity (scalar-eta v1: global Hebbian on co-visit counts) ---
    E_bias_max: float = 4.0
    alpha_gamma_s: float = 0.1          # surprise EMA rate
    plast_eta: float = 0.1              # base Hebbian learning rate
    plast_decay: float = 0.1            # E_bias decay per plasticity tick
    plast_surprise_bias: float = 1.0    # σ(surprise_ema - bias) gates eta

    # --- Neuromodulator (graph transformer on touched columns) ---
    # A small graph transformer runs at the start of each plasticity
    # window, observing a detached per-touched-column feature snapshot
    # from the previous window and emitting a grad-carrying delta to
    # E_bias_flat. The delta is live during the current window (gradient
    # flows back via routing → active_E_bias → neuromod params) and is
    # detached + folded into the persistent E_bias at window close.
    # Enabled by default — this is the thesis's intended plasticity path.
    # Zero-init output projections make day-0 behaviour identical to
    # scalar-eta Hebbian-only; disable with `use_neuromod=False` to
    # ablate.
    use_neuromod: bool = True
    # Neuromod sizing: this is the "learns how to learn" component — for a
    # lifelong-learning agent it needs enough capacity to encode meta-rules
    # over the graph, not just be a regulariser. Originally 128/2/4/64
    # (~0.5M, 0.4% of model); bumped to 384/4/8/256 (~8M, ~9% of model);
    # bumped again to 512/6/8/384 (~20M, ~20% of model) since neuromod is
    # FREE in per-token compute (fires once per mod_period window, runs
    # outside the captured graph) — pure meta-learning capacity gain.
    neuromod_D_mod: int = 512           # was 384, originally 128
    neuromod_n_layers: int = 6          # was 4, originally 2 — depth for graph reasoning
    neuromod_n_heads: int = 8           # was 4
    # Hidden dim of the per-edge target MLP (cat(x_src, x_dst) → scalar).
    # Replaces the old low-rank bilinear decoder's `neuromod_rank`.
    neuromod_edge_hidden: int = 384     # was 256, originally 64
    # Extra live/commit scale on top of the neuromod's own learnable blend
    # rate γ = σ(blend_logit). With γ as the primary knob this should be 1;
    # kept as a config knob for conservative sweeps or quick ablation.
    neuromod_eta: float = 1.0

    # --- Routing (Gumbel + exploration) ---
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_anneal_steps: int = 10_000
    epsilon_start: float = 0.05
    epsilon_end: float = 0.01
    epsilon_anneal_steps: int = 10_000
    lambda_balance: float = 0.01        # load-balance aux loss weight

    # --- Vocab ---
    vocab_size: int = 32_000

    # --- Training ---
    segment_T: int = 1024               # was 256. 8 windows per phase1_step
                                        # call → 8 plasticity fires per opt
                                        # step. Block memory unchanged
                                        # (depends only on mod_period).
    lr: float = 1e-4
    compile_on_train: bool = True
    # torch.compile mode for compile_block.
    #   "default":         ~3.7× over eager via inductor fusion (no cudagraphs)
    #   "reduce-overhead": adds CUDA-graph capture (additional ~2× target)
    #   "none":            disables compile entirely (dev iteration)
    compile_mode: str = "default"

    # --- Seeds ---
    topology_seed: int = 42
    init_seed: int = 43

    # --- Precision ---
    state_dtype: str = "auto"           # "auto" | "bf16" | "fp32"

    def __post_init__(self) -> None:
        if self.plane_rows * self.plane_cols < 1:
            raise ValueError("plane_rows and plane_cols must be positive")
        if self.L < 2:
            raise ValueError("L must be >= 2")
        if self.K < 2:
            raise ValueError("K must be >= 2")
        if not 0.0 <= self.p_rewire <= 1.0:
            raise ValueError("p_rewire must be in [0, 1]")
        if self.n_hops < 1:
            raise ValueError("n_hops must be >= 1")
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if self.D_model < 1 or self.D_s < 1:
            raise ValueError("D_model and D_s must be positive")
        if self.K_buf < self.K_horizons:
            raise ValueError(
                f"K_buf ({self.K_buf}) must be >= K_horizons ({self.K_horizons}) "
                "— delayed multi-horizon surprise still needs at least one "
                "history slot per horizon."
            )
        if self.segment_T % self.mod_period != 0:
            raise ValueError(
                f"segment_T ({self.segment_T}) must be a multiple of "
                f"mod_period ({self.mod_period}). Otherwise the final "
                f"{self.segment_T % self.mod_period}-token fragment in each "
                "segment gets CE training but no plasticity fire, silently "
                "dropping its Hebbian co-visits and neuromod snapshot."
            )
        if self.tbptt_block != self.mod_period:
            raise ValueError(
                f"tbptt_block ({self.tbptt_block}) must equal mod_period "
                f"({self.mod_period}). Each TBPTT block must close exactly "
                "one plasticity window — tbptt < mod_period fragments the "
                "neuromod's gradient mid-window, and tbptt > mod_period "
                "would silently skip plasticity firings because the flush "
                "code only runs _maybe_finalize... after backward."
            )
        # Validate the K split against the topology's local-neighborhood
        # candidate counts. build_topology samples without replacement via
        # torch.randperm; if K_intra/K_inter_fwd/K_inter_bwd exceed their
        # candidate counts, randperm silently truncates and the assignment
        # `out_nbrs[src] = ...` shape-mismatches and crashes. Guard up front.
        if not 0.0 <= self.K_inter_bwd_fraction <= 1.0:
            raise ValueError("K_inter_bwd_fraction must be in [0, 1]")
        if self.intra_radius < 1 or self.inter_radius < 1:
            raise ValueError("radii must be >= 1")
        K_intra = max(1, int(round(self.K * self.K_intra_fraction)))
        K_inter_total = self.K - K_intra
        K_inter_bwd = int(round(K_inter_total * self.K_inter_bwd_fraction))
        K_inter_fwd = K_inter_total - K_inter_bwd
        # Moore neighbourhood sizes: (2r+1)^2, minus self for intra-plane.
        max_intra = (2 * self.intra_radius + 1) ** 2 - 1
        max_inter = (2 * self.inter_radius + 1) ** 2
        if K_intra > max_intra:
            raise ValueError(
                f"K * K_intra_fraction = {K_intra} exceeds the "
                f"intra-plane Moore-radius-{self.intra_radius} candidate "
                f"count ({max_intra}). Increase intra_radius, reduce K "
                f"(currently {self.K}), or reduce K_intra_fraction "
                f"(currently {self.K_intra_fraction})."
            )
        for label, k_dir in (("forward", K_inter_fwd), ("backward", K_inter_bwd)):
            if k_dir > max_inter:
                raise ValueError(
                    f"K_inter_{label} = {k_dir} exceeds the inter-plane "
                    f"Moore-radius-{self.inter_radius} candidate count "
                    f"({max_inter}). Increase inter_radius, reduce K "
                    f"(currently {self.K}), or rebalance K_intra_fraction / "
                    f"K_inter_bwd_fraction."
                )

    @property
    def N_per_plane(self) -> int:
        return self.plane_rows * self.plane_cols

    @property
    def N(self) -> int:
        return self.L * self.N_per_plane

    @property
    def num_edges(self) -> int:
        return self.N * self.K
