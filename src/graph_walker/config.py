"""GraphWalkerConfig — hyperparams for the trajectory-routed variant.

See docs/graph_walker.md for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphWalkerConfig:
    # --- Graph topology (single flat substrate, no planes) ---
    grid_rows: int = 32                 # 32×32 = 1024 columns. Sized to the
    grid_cols: int = 32                 # ~1500-2000-word active vocabulary that
                                        # is the model's vocabulary of concepts;
                                        # N is the vocab size. Per-token compute
                                        # is invariant to N (walkers visit only
                                        # B·H cols per step regardless of N), so
                                        # N is pure capacity at no runtime cost.
    K: int = 16                         # out-edges per column. With N=1024,
                                        # K=16 gives a sparse small-world graph.
                                        # Per-step routing scoring stays in the
                                        # noise relative to content_mlp.
    # p_rewire: Watts-Strogatz rewiring probability. Each edge's destination
    # is, with probability p_rewire, replaced by a uniform-random column
    # anywhere in the graph. 0.0 = pure local grid.
    #
    # Default 0.3 (well past the small-world transition at ~0.1).
    # Rationale: under the "graph as syntactic substrate for an emergent
    # high-dimensional language" framing, the topology is the grammar's
    # possibility space — what trajectories the model is even allowed to
    # encode. A pure grid forces local-relations-only "sentences"
    # (Moore-radius only), syntactically impoverished. Heavy rewiring
    # gives the language long-distance syntactic moves at init, before
    # plasticity has had to discover them. Trade-off: weaker spatial
    # prior at init, more work for plasticity to learn which random edges
    # carry meaning. ε-exploration in routing.py provides per-decision
    # stochasticity on top.
    p_rewire: float = 0.3
    # Moore-radius for neighbour candidate sampling.
    # Radius 3 → 48 candidates, comfortable for K up to ~48.
    radius: int = 3

    # --- Per-walker weight specialisation ---
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
    D_s: int = 256                      # column state dim. With D_id=512 the
                                        # tried 1024 (Option A, ~106M params,
                                        # B=4 cudagraph OOM); settled on 768
                                        # (Option B, ~75M params) so we keep
                                        # most of the representational gain
                                        # without the quadratic cost — D_s²
                                        # is what makes content_mlp dominant.
    D_id: int = 512                     # column identity vector. The
                                        # identity should be a rich semantic
                                        # vector ("what concept does this
                                        # column carry"), not a tiny ID tag.
    ffn_mult_content: int = 4           # content_mlp: D_s + D_id → 4·D_s → D_s
                                        # ONLY used when D_hid_content is None
                                        # (legacy / small-D_s configs).
    # Hidden width of every content_mlp ResidualFFN block. Decoupled from
    # D_s — when None the legacy `ffn_mult_content * D_s` formula applies.
    D_hid_content: int | None = 1024
    # Post-readout model-space capacity. These blocks run once per token,
    # not once per hop, so they are a cheaper place to keep params than
    # the walker hot loop. The thesis is that the graph IS the model — the
    # cold readout is just a thin lexical projection head, not a deep
    # transformer FFN tower.
    post_model_depth: int = 2
    post_model_ffn_mult: int = 4
    n_score_heads: int = 4              # multi-head bilinear edge scoring
    D_q_per_head: int = 64

    # --- Graph-walker specifics ---
    n_heads: int = 4                    # H — parallel persistent walkers
    # Legacy trajectory-depth knob kept for compatibility with older configs
    # and future history-aware variants. In the current persistent-walker
    # design the fast path advances one hop per token.
    n_hops: int = 4

    # --- Multi-horizon readout ---
    K_horizons: int = 8
    # Legacy compatibility field from the earlier token-clock surprise path.
    # Current delayed-surprise code keeps the same lower-bound contract so old
    # configs do not silently under-allocate history.
    K_buf: int = 8

    # --- Clocks ---
    # `mod_period` is the plasticity window. `tbptt_block` is the
    # gradient-detach cadence. They should match: the neuromod's delta
    # lives for exactly one window, and we want its one gradient event per
    # window to attribute the full window-loss back to its action.
    mod_period: int = 128               # plasticity fires every N tokens
    tbptt_block: int = 128              # gradient detach cadence (== mod_period)

    # --- Plasticity ---
    E_bias_max: float = 4.0
    alpha_gamma_s: float = 0.1          # surprise EMA rate
    # plasticity_mode controls the plastic-update rule. Two values:
    #
    #   "hebbian_plus_neuromod" (default, used by standalone walker):
    #     E_bias += δ_hebb + δ_neuromod
    #     where δ_hebb = η_global · (co_visit_norm - decay·E_bias),
    #     η_global = plast_eta · σ(surprise_ema.mean() - plast_surprise_bias).
    #
    #   "neuromod_only" (used by integration via PretrainedGWConfig):
    #     E_bias += δ_neuromod only. Hebbian-flavored stats (co_visit,
    #     E_bias_old) become per-edge inputs to neuromod's edge MLP.
    plasticity_mode: str = "hebbian_plus_neuromod"
    plast_eta: float = 0.1              # base Hebbian learning rate
    plast_decay: float = 0.1            # E_bias decay per plasticity tick
    plast_surprise_bias: float = 1.0    # σ(surprise_ema - bias) gates eta

    # --- Neuromodulator (graph transformer on touched columns) ---
    # A small graph transformer runs at the start of each plasticity
    # window, observing a detached per-touched-column feature snapshot
    # from the previous window and emitting a grad-carrying delta to
    # E_bias_flat.
    use_neuromod: bool = True
    neuromod_D_mod: int = 512
    neuromod_n_layers: int = 6
    neuromod_n_heads: int = 8
    neuromod_edge_hidden: int = 384
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
    segment_T: int = 1024
    lr: float = 1e-4
    compile_on_train: bool = True
    compile_mode: str = "default"

    # --- Seeds ---
    topology_seed: int = 42
    init_seed: int = 43

    # --- Precision ---
    state_dtype: str = "auto"           # "auto" | "bf16" | "fp32"

    def __post_init__(self) -> None:
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid_rows and grid_cols must be positive")
        if self.K < 2:
            raise ValueError("K must be >= 2")
        if not 0.0 <= self.p_rewire <= 1.0:
            raise ValueError("p_rewire must be in [0, 1]")
        if self.radius < 1:
            raise ValueError("radius must be >= 1")
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
                "one plasticity window."
            )
        if self.plasticity_mode not in (
            "hebbian_plus_neuromod", "neuromod_only",
        ):
            raise ValueError(
                f"plasticity_mode={self.plasticity_mode!r} must be one of "
                "{'hebbian_plus_neuromod', 'neuromod_only'}."
            )
        if self.plasticity_mode == "neuromod_only" and not self.use_neuromod:
            raise ValueError(
                "plasticity_mode='neuromod_only' requires use_neuromod=True; "
                "otherwise nothing updates E_bias."
            )
        max_local = (2 * self.radius + 1) ** 2 - 1
        if self.K > max_local:
            raise ValueError(
                f"K={self.K} exceeds Moore-radius-{self.radius} candidate "
                f"count ({max_local}). Increase radius or reduce K."
            )

    @property
    def N(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def num_edges(self) -> int:
        return self.N * self.K
