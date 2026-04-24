"""GraphWalkerConfig — hyperparams for the trajectory-routed variant.

See docs/graph_walker.md for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphWalkerConfig:
    # --- Graph topology (same spirit as column_graph) ---
    plane_rows: int = 16
    plane_cols: int = 16
    L: int = 4                          # number of planes (0 = input, L-1 = output)
    K: int = 32                         # out-edges per column
    # p_rewire: Watts-Strogatz rewiring probability. 0.0 = pure local grid
    # (every edge bounded to Moore radius 2 in same plane or next plane).
    # Non-zero replaces that fraction of edges with uniform-random long-
    # range destinations — gives log-diameter small-world reach at the
    # cost of spatially-arbitrary teleports. Disabled by default; walks
    # rely on grid+plane hierarchy for retrieval, plasticity (E_bias) for
    # learned emphasis on useful local routes.
    p_rewire: float = 0.0
    K_intra_fraction: float = 0.5

    # --- Per-plane weight specialisation ---
    # When True, content_mlp has L independent copies (one per plane).
    # Slower on GPU (L small matmuls instead of one big batched matmul)
    # but architecturally principled. Disabled by default in favour of
    # `content_mlp_depth` (deep shared blocks) which hits the same param
    # budget with better throughput.
    per_plane_content_mlp: bool = False
    content_mlp_depth: int = 4          # residual FFN blocks (shared weights)

    # --- Widths ---
    # D_model is the external lexical/readout width (token embedding,
    # tied unembedding, post-readout capacity). D_s is the internal graph
    # state width paid in the per-hop recurrent hot path.
    D_model: int = 1024
    D_s: int = 512                      # column state dim
    D_id: int = 32
    ffn_mult_content: int = 4           # content_mlp: D_s + D_id → 4·D_s → D_s
    # Post-readout model-space capacity. These blocks run once per token,
    # not once per hop, so they are a cheaper place to keep params than
    # the walker hot loop.
    post_model_depth: int = 7
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
    mod_period: int = 128               # plasticity fires every N tokens
    tbptt_block: int = 16               # gradient detach cadence

    # --- Plasticity (scalar-eta v1: global Hebbian on co-visit counts) ---
    E_bias_max: float = 4.0
    alpha_gamma_s: float = 0.1          # surprise EMA rate
    plast_eta: float = 0.1              # base Hebbian learning rate
    plast_decay: float = 0.1            # E_bias decay per plasticity tick
    plast_surprise_bias: float = 1.0    # σ(surprise_ema - bias) gates eta

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
    segment_T: int = 256
    lr: float = 1e-4
    compile_on_train: bool = True

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

    @property
    def N_per_plane(self) -> int:
        return self.plane_rows * self.plane_cols

    @property
    def N(self) -> int:
        return self.L * self.N_per_plane

    @property
    def num_edges(self) -> int:
        return self.N * self.K
