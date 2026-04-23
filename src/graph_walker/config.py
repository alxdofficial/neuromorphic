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
    p_rewire: float = 0.30
    K_intra_fraction: float = 0.5

    # --- Column internals ---
    D_s: int = 512                      # column state dim
    D_id: int = 32
    ffn_mult_content: int = 2           # content_mlp: D_s + D_id → 2·D_s → D_s
    n_score_heads: int = 4              # multi-head bilinear edge scoring
    D_q_per_head: int = 64

    # --- Graph-walker specifics ---
    n_heads: int = 4                    # H — parallel trajectories per token
    n_hops: int = 4                     # L_walk — hops per trajectory
    D_q_in: int = 64                    # input-plane start-column query dim

    # --- Multi-horizon readout ---
    K_horizons: int = 8
    K_buf: int = 8

    # --- Clocks ---
    mod_period: int = 128               # plasticity fires every N tokens
    tbptt_block: int = 16               # gradient detach cadence

    # --- Plasticity ---
    E_bias_max: float = 4.0
    alpha_gamma_s: float = 0.1          # surprise EMA rate

    # --- Neuromod ---
    D_trunk: int = 128
    trunk_hidden: int = 256
    head_hidden: int = 64

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

    @property
    def N_per_plane(self) -> int:
        return self.plane_rows * self.plane_cols

    @property
    def N(self) -> int:
        return self.L * self.N_per_plane

    @property
    def num_edges(self) -> int:
        return self.N * self.K
