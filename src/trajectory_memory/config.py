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
    N: int = 4096
    D_concept: int = 1024
    K_max_neighbors: int = 64

    # ── topology (small-world ring rewire) ────────────────────────────
    # 1D ring with each concept's K_max nearest neighbors as the local
    # zone; `radius` is the half-width of the local zone (so the local
    # candidate pool has 2*radius slots, must be >= K_max).
    # graph_walker's `radius=4` referred to a 2D Moore neighborhood (80
    # candidates for K=64) — for our 1D ring topology we need
    # radius >= K_max/2.
    p_rewire: float = 0.5
    radius: int = 32

    # ── trajectory shape ──────────────────────────────────────────────
    J: int = 4
    K_read: int = 8
    K_write: int = 8

    # ── window structure ──────────────────────────────────────────────
    T_window: int = 256

    # ── llama integration ────────────────────────────────────────────
    d_lm: int = 2048              # llama-3.2-1B hidden dim; overridden post-load
    # `inject_layer` is the absolute layer index. Setting it directly is
    # backward-compatible. Prefer `inject_layer_frac` to keep mid-stack
    # placement under different Llama sizes — set frac to non-None and it
    # OVERRIDES inject_layer at IntegratedLM construction with
    # int(frac × num_hidden_layers). Default 8 ≈ mid for Llama-3.2-1B
    # (16 layers × 0.5).
    inject_layer: int = 8
    inject_layer_frac: float | None = None
    bridge_hidden: int | None = 2048  # MemInjectLayer bridge MLP hidden dim

    # ── training (cross-window TBPTT) ────────────────────────────────
    D: int = 4                    # tbptt depth in windows
    effective_lm_context: int = 2048  # hard-truncate LM input

    # ── trajectory mechanics ─────────────────────────────────────────
    # mutation_init_scale: retained only for backward-compat with old
    # checkpoints. The write module now uses an nn.GRUCell (target-form
    # + bounded tanh + per-element update gate), so there's no candidate-
    # scale knob. Kept here so loading an old config doesn't crash.
    mutation_init_scale: float = 0.1
    gumbel_tau: float = 1.0            # legacy — unused after switch to softmax-STE
    pos_enc_scale: float = 1.0         # multiplier on positional encoding

    # Learnable logit-scale for cosine routing (CLIP-style).
    # Effective scale = exp(logit_scale_init).clamp(max=20). Init at 1.5
    # → exp(1.5) ≈ 4.5 → cosine logits scaled to [-4.5, 4.5] before softmax.
    # Without this, cosine logits ∈ [-1, 1] are dominated by the Gumbel(0,1)
    # noise (std 1.28) that the older routing used, so routing was random.
    # With softmax-STE (no Gumbel), the scale governs softmax temperature
    # directly: too small → flat softmax → no specialization; too large →
    # peaked softmax → gradient only flows to top-1.
    logit_scale_init: float = 1.5

    # ── magnitude bounding ────────────────────────────────────────────
    # Apply normalization at consumption boundaries so unbounded parameter
    # drift doesn't translate into unbounded state magnitudes (the
    # state-drift bug from Wave 1 run @ step ~14500). Four sites:
    #   1. concept_states     — GRUCell (in write_module) bounds via tanh.
    #   2. state_init         — L2-normalize at reset_states() consumption.
    #   3. concept_ids        — L2-normalize at every routing dot-product.
    #   4. MemInjectLayer.scale — tanh-clamp at consumption.
    # `state_init_norm` sets the target L2 norm for the reset-state.
    state_init_norm: float = 1.0       # target L2 norm for state_init at consumption

    # ── init seeds ────────────────────────────────────────────────────
    seed_topology: int = 0
    seed_concepts: int = 0

    # ── Architectural ablation ─────────────────────────────────────────
    # flat_bank=True swaps the trajectory read/write modules for simpler
    # top-K cell-attention modules over the same 4096×D_concept manifold.
    # Used to test whether the trajectory + graph machinery adds value
    # over a flat memory bank at same capacity.
    flat_bank: bool = False

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
