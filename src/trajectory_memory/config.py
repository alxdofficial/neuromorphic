"""TrajMemConfig — single source of truth for trajectory-memory hyperparameters.

Tier presets (small / medium / large) match the table in §4.4 + Appendix B
of docs/plan_trajectory_memory.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrajMemConfig:
    """Hyperparameters for the trajectory-memory module.

    The key axes (see Appendix B in the plan):
    - `N`, `D_concept` set the manifold capacity.
    - `J`, `K_read`, `K_write` set retrieval/encoding breadth.
    - `K_max_neighbors`, `p_rewire`, `radius` set graph topology.
    - `D` (TBPTT depth) sets cross-window gradient horizon.
    - The remaining knobs are operational (LR, dtype, etc.) and live on
      the trainer, not here.
    """

    # ── manifold capacity ─────────────────────────────────────────────
    N: int = 2048
    D_concept: int = 256
    K_max_neighbors: int = 32

    # ── topology (small-world ring rewire) ────────────────────────────
    # 1D ring with each concept's K_max nearest neighbors as the local
    # zone; `radius` is the half-width of the local zone (so the local
    # candidate pool has 2*radius slots, must be >= K_max).
    # graph_walker's `radius=4` referred to a 2D Moore neighborhood (80
    # candidates for K=64) — for our 1D ring topology we need
    # radius >= K_max/2.
    p_rewire: float = 0.5
    radius: int = 16

    # ── trajectory shape ──────────────────────────────────────────────
    J: int = 4
    K_read: int = 8
    K_write: int = 8

    # ── window structure ──────────────────────────────────────────────
    T_window: int = 256

    # ── llama integration ────────────────────────────────────────────
    d_lm: int = 2048              # llama-3.2-1B hidden dim; overridden post-load
    inject_layer: int = 8         # mid-stack
    bridge_hidden: int | None = 2048  # MemInjectLayer bridge MLP hidden dim

    # ── training (cross-window TBPTT) ────────────────────────────────
    D: int = 4                    # tbptt depth in windows
    effective_lm_context: int = 2048  # hard-truncate LM input

    # ── trajectory mechanics ─────────────────────────────────────────
    mutation_init_scale: float = 0.1   # `new = state + 0.1 * MLP(...)`
    gumbel_tau: float = 1.0            # Gumbel-softmax temperature
    pos_enc_scale: float = 1.0         # multiplier on positional encoding

    # ── init seeds ────────────────────────────────────────────────────
    seed_topology: int = 0
    seed_concepts: int = 0

    # Note on dtype policy: memory params (concept_ids, concept_states,
    # MLPs) always stay in fp32 — bf16 weight updates round small Adam
    # steps to zero (same lesson as graph_walker / pretrained). Llama's
    # backbone runs in bf16; MemInjectLayer's W_in/W_out bridge handles
    # the cross-dtype add. No knob — fp32-only by design.

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        assert self.N > 0
        assert self.D_concept > 0
        assert 1 <= self.K_max_neighbors <= self.N
        assert 0.0 <= self.p_rewire <= 1.0
        assert self.radius >= 1
        # Local 1D-ring zone has 2*radius candidates; K_max can't exceed.
        assert self.K_max_neighbors <= 2 * self.radius, (
            f"K_max_neighbors={self.K_max_neighbors} > 2*radius={2*self.radius} "
            f"local candidates. Increase radius to >= {self.K_max_neighbors // 2}."
        )
        assert self.J >= 1
        assert self.K_read >= 1
        assert self.K_write >= 1
        assert self.T_window > 0
        assert self.D >= 2, "TBPTT depth < 2 starves the write module"
        assert self.effective_lm_context >= self.T_window

    # ── factory presets ───────────────────────────────────────────────

    @classmethod
    def small(cls) -> "TrajMemConfig":
        """Smoke / debugging preset. Fits comfortably on CPU."""
        return cls(
            N=1024, D_concept=128, K_max_neighbors=8, radius=4,
            J=2, K_read=4, K_write=4,
            T_window=128, D=2,
            d_lm=512, inject_layer=4, bridge_hidden=512,
            effective_lm_context=512,
        )

    @classmethod
    def medium(cls) -> "TrajMemConfig":
        """v1 default — see plan §4.4."""
        return cls()

    @classmethod
    def large(cls) -> "TrajMemConfig":
        """Post-v1 scale-up. Same shape as medium but bigger."""
        return cls(
            N=4096, D_concept=256, K_max_neighbors=32, radius=16,
            J=8, K_read=16, K_write=16,
            T_window=256, D=8,
            d_lm=3072, inject_layer=16, bridge_hidden=3072,  # llama-3b-ish
            effective_lm_context=4096,
        )
