"""ColumnGraphConfig — all architecture + training hyperparameters.

See docs/column_graph.md for the rationale behind each default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnGraphConfig:
    # --- Graph topology ---
    # L planes, each a 2D grid of shape plane_rows x plane_cols.
    plane_rows: int = 32
    plane_cols: int = 32
    L: int = 4                              # number of planes (0=input, L-1=output)
    K: int = 32                             # out-edges per column
    p_rewire: float = 0.30                  # Watts-Strogatz shuffle probability
    # Split of K out-edges between intra-plane local vs inter-plane local
    # (the remaining p_rewire fraction is uniformly shuffled).
    K_intra_fraction: float = 0.5           # half of pre-shuffle edges stay in-plane

    # --- Column internals ---
    D_s: int = 256                          # column state dim
    D_id: int = 32                          # column identity dim
    ffn_mult_update: int = 4                # update_MLP: D_s + D_id + D_s → 4·D_s → D_s
    ffn_mult_content: int = 2               # content_MLP: D_s + D_id → 2·D_s → D_s
    ffn_mult_delta: int = 1                 # delta_MLP: D_s + D_id → D_s → D_s (residual)
    score_hidden: int = 64                  # score_MLP hidden

    # --- Dynamics ---
    T: int = 1                              # rounds per token (depth from T_seq, not T)

    # --- Multi-horizon readout ---
    K_horizons: int = 8
    K_buf: int = 8

    # --- Clocks ---
    mod_period: int = 16                    # neuromod + plasticity firing cadence
    tbptt_block: int = 32                   # detach persistent state every N tokens

    # --- Plasticity ---
    E_bias_max: float = 4.0
    E_bias_init_scale: float = 0.0          # start at zero, let plasticity write
    alpha_gamma_s: float = 0.1              # surprise EMA rate
    alpha_input_ctx: float = 0.05           # input-ctx EMA rate
    alpha_tile_stats: float = 0.05          # mag/var EMA rate for tile features

    # --- Neuromod ---
    D_trunk: int = 384
    trunk_hidden: int = 768
    head_hidden: int = 64
    num_tiles_per_plane_dim: int = 4        # each plane tiled 4×4 = 16 tiles per plane

    # --- Cross-attention I/O ---
    n_attn_heads_in: int = 4                # multi-head for input injection
    n_attn_heads_out: int = 4               # multi-head for output readout

    # --- Vocab ---
    vocab_size: int = 32000

    # --- Training ---
    segment_T: int = 256
    lr: float = 1e-4

    # --- Seeds (topology / init determinism) ---
    topology_seed: int = 42
    init_seed: int = 43

    # --- Precision ---
    # bf16 autocast on CUDA, fp32 fallback on CPU. Plasticity / surprise /
    # neuromod-feature math always forced fp32 via autocast(enabled=False).
    state_dtype: str = "auto"               # "auto" | "bf16" | "fp32"

    def __post_init__(self) -> None:
        if self.plane_rows * self.plane_cols < 1:
            raise ValueError("plane_rows and plane_cols must be positive")
        if self.L < 2:
            raise ValueError("L must be >= 2 (input plane + output plane)")
        if self.K < 2:
            raise ValueError("K must be >= 2")
        if not 0.0 <= self.p_rewire <= 1.0:
            raise ValueError("p_rewire must be in [0, 1]")
        if self.plane_rows % self.num_tiles_per_plane_dim != 0:
            raise ValueError(
                f"plane_rows ({self.plane_rows}) must be divisible by "
                f"num_tiles_per_plane_dim ({self.num_tiles_per_plane_dim})"
            )
        if self.plane_cols % self.num_tiles_per_plane_dim != 0:
            raise ValueError(
                f"plane_cols ({self.plane_cols}) must be divisible by "
                f"num_tiles_per_plane_dim ({self.num_tiles_per_plane_dim})"
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

    @property
    def num_tiles_per_plane(self) -> int:
        return self.num_tiles_per_plane_dim ** 2

    @property
    def num_tiles(self) -> int:
        return self.L * self.num_tiles_per_plane

    @property
    def tile_rows(self) -> int:
        return self.plane_rows // self.num_tiles_per_plane_dim

    @property
    def tile_cols(self) -> int:
        return self.plane_cols // self.num_tiles_per_plane_dim

    @property
    def tile_size(self) -> int:
        return self.tile_rows * self.tile_cols
