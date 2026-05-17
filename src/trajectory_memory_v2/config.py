"""Configuration for the vocabulary-trajectory architecture (v2).

Separate from `src/trajectory_memory/config.py` so the two architectures
can coexist. Most names match v1 where the meaning is the same.

See docs/design_vocabulary_trajectory.md for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrajMemV2Config:
    # ── Vocabulary (frozen N, learnable embeddings) ───────────────────────
    # N is the abstract-vocabulary size. Each concept_id is a D-dim vector.
    # In contrast to v1, nodes have NO mutable per-cell state — only the
    # frozen-N embeddings exist. Content lives on edges.
    N: int = 4096
    D_concept: int = 1024  # alias D for backward-compat readers

    # ── Edge memory (sparse, bounded fan-out) ─────────────────────────────
    # Each node has up to K_max outgoing edges. Edge state is a D-dim
    # vector accumulated via EMA across writes that traverse the edge.
    K_max: int = 32

    # ── Trajectory walk ───────────────────────────────────────────────────
    # J parallel trajectories per window, each of length K (entry + K-1 hops).
    J: int = 4
    K_read: int = 8
    K_write: int = 8

    # ── Window / LM context ───────────────────────────────────────────────
    T_window: int = 256
    d_lm: int = 2048  # Llama-3.2-1B hidden size

    # ── EMA edge update ───────────────────────────────────────────────────
    # α = max(α_base / (1 + log(1 + visit_count)), α_min)
    # α_min prevents silent freeze on heavily-used edges.
    #
    # At α_base=0.1, the EMA at visit=1 is ~0.14 (gentle imprint), at visit=10
    # ~0.04 (slow drift), at visit=100 ~0.022 (mostly stable). This gives
    # "old writes leave a trace, new writes overlay on top" — graceful
    # dilution rather than catastrophic overwrite.
    ema_alpha_base: float = 0.1
    ema_alpha_min: float = 0.01

    # ── Eviction policy: W-TinyLFU read/write ratio ───────────────────────
    # Each edge tracks two EMAs (decayed each step by `touch_ema_decay`):
    #   write_touches_ema — incremented when the walker writes to this edge
    #   read_touches_ema  — incremented when a read trajectory chose this
    #                       edge's dst (= "the model actually retrieved it")
    #
    # Effectiveness = (reads + α) / (writes + β) with Laplace smoothing:
    #   α = small prior on reads (so brand-new edges don't sit at 0)
    #   β = prior on writes (set so a 0-read/1-write edge isn't instantly
    #       evicted)
    # Evict argmin(effectiveness). Long-horizon retention works correctly:
    # an edge can sit idle for 10K steps and survive if its read/write
    # ratio stays high — what matters is "did retrieval find this useful?",
    # not "was it visited recently."
    touch_ema_decay: float = 0.999     # ~1000-step memory at decay = 0.999
    evict_smoothing_alpha: float = 1.0  # prior reads
    evict_smoothing_beta: float = 4.0   # prior writes

    # ── Specificity (diagnostic only — not used in eviction policy) ──────
    # EMA of cosine distance between successive write signatures and the
    # current edge state. High spec = novel writes; low = refining.
    # Reported in edge_stats() for monitoring; no longer drives eviction.
    spec_ema_beta: float = 0.1
    spec_ref: float = 1.0

    # ── Eviction protection (single age floor) ────────────────────────────
    # Brand-new edges (age < protect_min_age) are protected from eviction —
    # they need time to accumulate read/write evidence. If too many slots
    # are protected (lockup risk), the floor is bypassed for that source.
    protect_min_age: int = 50
    protect_max_frac: float = 0.30  # fraction of K_max that can be age-protected

    # ── Reuse bonus (encourages routing toward existing edges) ────────────
    # Walker adds `reuse_bonus * existing_dst_mask` to the combined routing
    # logits. Pulls the walker toward already-allocated cells unless the
    # vocab score for a non-edge cell is enough higher to overcome it.
    # Learnable scalar so the model can tune the pressure during training.
    # Default 1.0 ≈ one std-dev of vocab logits (significant but not
    # dominant — a much better non-edge cell can still win).
    reuse_bonus_init: float = 1.0

    # ── Routing score composition ─────────────────────────────────────────
    # combined_score = vocab_score + λ · edge_score
    # edge_score is zero-centered cosine (RMS-norm both sides).
    # λ is a learnable scalar, initialized below.
    lambda_edge_init: float = 0.5

    # Running-cue leaky integrator decay (in walker step loop).
    # cue_D ← cue_decay · cue_D + cue_proj(next_embed)
    # 0 = no history (only current visit); 1 = unbounded sum (the old bug).
    # 0.7 → ~3.3-step effective memory; preserves recency-weighted
    # magnitude info across hops without unbounded blow-up.
    cue_decay: float = 0.7

    # ── SimVQ reparameterization of concept_ids ───────────────────────────
    # concept_ids = id_proj(id_basis), with id_proj init to identity +
    # a small Gaussian perturbation. The perturbation breaks the perfect
    # symmetry that would otherwise make every concept_id rotate in
    # lock-step under early gradients (since they all share id_proj.weight).
    simvq_init_std: float = 0.02       # std for id_basis init
    id_proj_perturb_std: float = 0.001  # off-identity noise on id_proj.weight

    # ── Routing aux losses ────────────────────────────────────────────────
    # NOTE: lower than v1's 1e-2 / 1e-3 because v2 routes over the FULL
    # N=4096 at EVERY hop (not just at entries), while v1 routed over
    # K_max ≈ 16-32 graph neighbors at hops. Switch load_balance scales
    # with N for collapsed routing, so v2's aux_lb at random init is
    # ~400× v1's; compensate by scaling coefficients down to keep total
    # contribution to loss in the same range as v1 (~0.1 nats).
    load_balance_coef: float = 1e-4
    z_loss_coef: float = 1e-4

    # ── Contrastive losses (Wave 1 only) ──────────────────────────────────
    contrast_coef: float = 0.1
    contrast_temperature: float = 0.07
    per_step_contrast_coef: float = 0.05  # on step_queries between matched read/write
    per_step_contrast_temperature: float = 0.07

    # ── Tripwire thresholds for telemetry ─────────────────────────────────
    tripwire_min_alpha: float = 0.005  # warn if min(α) over edges falls below
    tripwire_max_eviction_rate: float = 0.20  # warn if >20% of edges evicted/step

    @classmethod
    def small(cls) -> "TrajMemV2Config":
        """Tiny config for unit tests."""
        return cls(N=64, D_concept=128, K_max=8, J=2, K_read=4, K_write=4)

    @classmethod
    def medium(cls) -> "TrajMemV2Config":
        """Standard config; matches v1 medium where comparable."""
        return cls()

    @classmethod
    def large(cls) -> "TrajMemV2Config":
        return cls(N=8192, D_concept=1024, K_max=32)

    def validate(self) -> None:
        assert self.N > 0, "N must be positive"
        assert self.D_concept > 0, "D_concept must be positive"
        assert self.K_max > 0, "K_max must be positive"
        assert self.K_read > 0 and self.K_write > 0
        assert self.J > 0
        assert 0.0 < self.ema_alpha_base <= 1.0
        assert 0.0 <= self.ema_alpha_min <= self.ema_alpha_base
        assert 0.0 <= self.protect_max_frac < 1.0

    # Convenience accessor for code that uses cfg.D in v1 style.
    @property
    def D(self) -> int:
        return self.D_concept
