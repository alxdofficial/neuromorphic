"""VocabularyManifold — vocabulary embeddings + sparse edge memory.

Key differences from v1's `Manifold`:

- **No `current_states`**. Nodes have no mutable per-cell state. Only
  the learnable `concept_ids` exist as per-node tensors.
- **Sparse edge buffer** with bounded fan-out (K_max per source).
  Edge state is a (non-parameter) buffer that accumulates via EMA
  across writes. Gradient flows THROUGH it (for signature_fn training)
  but optimizer doesn't update it directly.
- **Per-edge metadata**: visit_count, last_visit_step, alloc_step,
  specificity — used for EMA stability and eviction policy.

Design doc: docs/design_vocabulary_trajectory.md
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.trajectory_memory_v2.config import TrajMemV2Config


def rms_norm_last(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMS-norm over the last dim. Unit per-element variance."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms


class VocabularyManifold(nn.Module):
    """N learnable vocabulary embeddings + sparse edge state buffer.

    State layout:
        concept_ids:    [N, D]      learnable (via SimVQ reparam)
        edge_state:     [N, K_max, D]  buffer (EMA-driven, not optimized)
        edge_dst:       [N, K_max]  buffer (long, -1 for unused slot)
        edge_active:    [N, K_max]  buffer (bool)
        visit_count:    [N, K_max]  buffer (long)
        last_visit:     [N, K_max]  buffer (long, step counter)
        alloc_step:     [N, K_max]  buffer (long, -1 for unused)
        specificity:    [N, K_max]  buffer (float, EMA-decayed surprise)

    The per-node-dense layout (rather than CSR) costs ~1MB of metadata
    at N=4096, K_max=32 and is far easier to batch.
    """

    def __init__(self, cfg: TrajMemV2Config):
        super().__init__()
        self.cfg = cfg
        N, D, K_max = cfg.N, cfg.D_concept, cfg.K_max

        # ── Vocabulary: SimVQ reparameterization (carried over from v1) ──
        # concept_ids = id_proj(id_basis). id_proj initialized to identity
        # so concept_ids ≡ id_basis at init. Gradient flows through both
        # id_basis (per-row) and id_proj.weight (shared). This is what
        # prevents codebook collapse.
        ids_init = torch.randn(N, D) * cfg.simvq_init_std
        self.id_basis = nn.Parameter(ids_init)
        self.id_proj = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            # Identity + small perturbation. Pure identity init means
            # routing gradients perturb id_proj.weight identically for
            # every concept_id (since concept_ids = id_proj @ id_basis
            # shares one weight matrix). The perturbation breaks that
            # exact symmetry, so different cells start individuating
            # immediately rather than after id_basis catches up.
            self.id_proj.weight.copy_(
                torch.eye(D) + cfg.id_proj_perturb_std * torch.randn(D, D)
            )

        # ── Edge memory: buffers, not parameters ──
        # edge_state evolves via EMA from write traversals. Gradient
        # flows through it during backward (so signature_fn gets trained)
        # but the optimizer does not apply gradient steps to it.
        self.register_buffer("edge_state", torch.zeros(N, K_max, D))
        self.register_buffer(
            "edge_dst",
            torch.full((N, K_max), -1, dtype=torch.long),
        )
        self.register_buffer(
            "edge_active",
            torch.zeros(N, K_max, dtype=torch.bool),
        )
        self.register_buffer(
            "visit_count",
            torch.zeros(N, K_max, dtype=torch.long),
        )
        self.register_buffer(
            "last_visit",
            torch.zeros(N, K_max, dtype=torch.long),
        )
        self.register_buffer(
            "alloc_step",
            torch.full((N, K_max), -1, dtype=torch.long),
        )
        self.register_buffer(
            "specificity",
            torch.zeros(N, K_max),
        )
        # Global step counter — incremented externally by the trainer.
        # Used for staleness computation and alloc_step grace-period.
        self.register_buffer(
            "step_counter",
            torch.zeros((), dtype=torch.long),
        )

    # ── Vocabulary access ───────────────────────────────────────────────

    @property
    def concept_ids(self) -> Tensor:
        """Effective vocab embeddings: id_proj(id_basis).

        SimVQ ensures gradient flows back through the shared id_proj,
        which means routing into one cell trains the routing keys of
        ALL cells (via id_proj.weight). Prevents disjoint-codebook
        collapse.
        """
        return self.id_proj(self.id_basis)

    @property
    def concept_ids_normed(self) -> Tensor:
        """RMS-normed concept_ids for SDPA routing."""
        return rms_norm_last(self.concept_ids)

    # ── Edge lookup (batched read access) ───────────────────────────────

    def lookup_edges(self, src_nodes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Gather outgoing edges for a batch of source nodes.

        Args:
            src_nodes: [...] long — source node indices.

        Returns:
            edge_states: [..., K_max, D]    — per-slot edge state
            edge_dsts:   [..., K_max]  long — destination node (-1 for unused)
            edge_active: [..., K_max]  bool — slot occupancy

        Inactive slots have zero state, dst=-1, active=False.
        """
        # edge_state shape: [N, K_max, D] → index by src_nodes
        edge_states = self.edge_state[src_nodes]    # [..., K_max, D]
        edge_dsts = self.edge_dst[src_nodes]        # [..., K_max]
        edge_active = self.edge_active[src_nodes]   # [..., K_max]
        return edge_states, edge_dsts, edge_active

    # ── Edge update (write path) ────────────────────────────────────────

    @torch.no_grad()
    def update_edges(
        self,
        src_batch: Tensor,           # [M] long — source nodes
        dst_batch: Tensor,           # [M] long — destination nodes
        signature_batch: Tensor,     # [M, D] — signatures to write
    ) -> Tensor:
        """EMA-update (or allocate) edges for a batch of writes.

        Sequential implementation: processes one write at a time. For
        v1 this is fine; profile and parallelize if it becomes a
        bottleneck.

        IMPORTANT: this runs UNDER torch.no_grad() — the EMA update of
        the edge_state buffer is NOT in the autograd graph. This means
        **gradient does NOT flow back through edge_state to the
        signature_fn that produced the signatures**. To train any
        learnable parameters in the signature path, the trainer must
        apply a SEPARATE direct supervision (e.g., a contrastive loss
        on signatures vs. read-time queries). Without that, signatures
        are effectively a fixed, non-learnable function of their
        inputs.

        For v1, the recommended simplification is signature = step_query
        (no separate signature_fn), so this constraint doesn't bite —
        step_queries already get gradient from the routing decisions.

        Returns:
            slot_indices: [M] long — slot index per write (-1 if alloc failed)
        """
        M = src_batch.shape[0]
        slot_indices = torch.full((M,), -1, dtype=torch.long,
                                  device=src_batch.device)
        step = int(self.step_counter.item())

        for i in range(M):
            src = int(src_batch[i].item())
            dst = int(dst_batch[i].item())
            sig = signature_batch[i]

            slot = self._find_edge_slot(src, dst)
            if slot >= 0:
                # Existing edge — EMA update
                self._update_existing_slot(slot, src, sig, step)
            else:
                # No existing edge — try to allocate
                slot = self._allocate_or_evict(src, dst, sig, step)

            slot_indices[i] = slot

        return slot_indices

    def _find_edge_slot(self, src: int, dst: int) -> int:
        """Return the slot index of edge (src, dst), or -1 if not present."""
        slots = self.edge_dst[src]                     # [K_max]
        active = self.edge_active[src]                 # [K_max]
        match = (slots == dst) & active
        if match.any():
            return int(match.nonzero(as_tuple=False)[0].item())
        return -1

    def _update_existing_slot(self, slot: int, src: int, sig: Tensor, step: int) -> None:
        """EMA-update an existing edge slot.

        Specificity = EMA-tracked **cosine distance** between the new
        signature and the current edge state. Cosine is scale-invariant,
        so a high-specificity edge means "successive writes here keep
        rotating the state" (genuinely novel content), while low-spec
        means "writes keep landing on the same direction" (refining or
        redundant). Bounded in [0, 2] per update, EMA-smoothed.
        """
        cfg = self.cfg
        old_state = self.edge_state[src, slot]
        visit = int(self.visit_count[src, slot].item())

        # α = max(α_base / (1 + log(1 + visit)), α_min) — EMA on edge_state
        alpha = cfg.ema_alpha_base / (1.0 + math.log(1.0 + visit))
        alpha = max(alpha, cfg.ema_alpha_min)

        # Specificity: cosine-distance-based novelty, EMA-smoothed.
        if old_state.norm().item() > 1e-6:
            cos_sim = float(F.cosine_similarity(
                sig.unsqueeze(0), old_state.unsqueeze(0), dim=-1,
            ).item())
            cos_dist = 1.0 - cos_sim  # ∈ [0, 2]
        else:
            cos_dist = 1.0  # zero prior — treat first overlay as moderately novel
        new_spec = (
            (1.0 - cfg.spec_ema_beta) * float(self.specificity[src, slot].item())
            + cfg.spec_ema_beta * cos_dist
        )

        # In-place updates
        self.edge_state[src, slot] = (1 - alpha) * old_state + alpha * sig
        self.visit_count[src, slot] += 1
        self.last_visit[src, slot] = step
        self.specificity[src, slot] = new_spec

    def _allocate_or_evict(
        self, src: int, dst: int, sig: Tensor, step: int,
    ) -> int:
        """Allocate a new edge (src, dst). Evict from src if at K_max.

        Returns the new slot index, or -1 if allocation failed (all slots
        protected and we can't evict).
        """
        # Find an empty slot for this src
        active = self.edge_active[src]
        empty_mask = ~active
        if empty_mask.any():
            slot = int(empty_mask.nonzero(as_tuple=False)[0].item())
            self._initialize_slot(src, slot, dst, sig, step)
            return slot

        # No empty slot — need to evict
        victim_slot = self._select_eviction_victim(src, step)
        if victim_slot < 0:
            # All slots protected; cannot evict. Skip this allocation.
            return -1

        # Free the victim, then allocate into its slot
        self._free_slot(src, victim_slot)
        self._initialize_slot(src, victim_slot, dst, sig, step)
        return victim_slot

    def _initialize_slot(
        self, src: int, slot: int, dst: int, sig: Tensor, step: int,
    ) -> None:
        """Initialize a freshly-allocated slot with the new edge's data."""
        self.edge_state[src, slot] = sig
        self.edge_dst[src, slot] = dst
        self.edge_active[src, slot] = True
        self.visit_count[src, slot] = 1
        self.last_visit[src, slot] = step
        self.alloc_step[src, slot] = step
        # Initial specificity = 1.0 (moderately novel by default —
        # a brand-new edge has no prior to compare against, so we set
        # it mid-range. Subsequent updates EMA this toward true novelty).
        self.specificity[src, slot] = 1.0

    def _free_slot(self, src: int, slot: int) -> None:
        """Reset a slot to empty (called during eviction)."""
        self.edge_state[src, slot].zero_()
        self.edge_dst[src, slot] = -1
        self.edge_active[src, slot] = False
        self.visit_count[src, slot] = 0
        self.last_visit[src, slot] = 0
        self.alloc_step[src, slot] = -1
        self.specificity[src, slot] = 0.0

    def _select_eviction_victim(self, src: int, step: int) -> int:
        """Pick the highest eviction-score slot in src's outgoing list.
        Respect protection floors. Returns -1 if all are protected."""
        cfg = self.cfg
        K_max = cfg.K_max

        active = self.edge_active[src]                       # [K_max] bool
        if not active.any():
            return -1

        # ── Protection floors (per-slot) ──
        ages = step - self.alloc_step[src].clamp_min(0)
        ages = ages.masked_fill(~active, 0)
        # Unit-scale state norm (D-invariant). Raw L2 of a D-dim vector
        # scales as O(√D); dividing by √D gives a scale comparable across
        # different D_concept and to the [0, 1] thresholds below.
        sqrt_D = math.sqrt(float(cfg.D_concept))
        state_norm_unit = self.edge_state[src].norm(dim=-1) / sqrt_D  # [K_max]

        protected_mask = (
            active
            & (ages >= cfg.protect_min_age)
            & (self.specificity[src] >= cfg.protect_min_spec)
            & (state_norm_unit >= cfg.protect_min_norm)
        )

        # Cap protected fraction — if too many are protected, allow evicting
        # from the protected set (so the system can't lock up).
        n_protected = int(protected_mask.sum().item())
        if n_protected > cfg.protect_max_frac * K_max:
            evictable_mask = active
        else:
            evictable_mask = active & ~protected_mask

        if not evictable_mask.any():
            return -1

        # ── 4-feature eviction score (high score = more evictable) ──
        # Each term is monotonic-bounded so all four actually discriminate
        # (the previous `clamp(0, 1)` on raw L2 zeroed-out the norm term
        # for any edge with norm ≥ 1, which is every active edge).
        visit_term = 1.0 / (self.visit_count[src].float() + 1.0)             # [0, 0.5]
        stale_term = (step - self.last_visit[src]).clamp_min(0).float() / cfg.evict_horizon
        stale_term = stale_term.clamp(0, 1)                                   # [0, 1]
        # Smooth, non-saturating: ~1 for empty edges, → 0 as state grows
        norm_term = 1.0 / (1.0 + state_norm_unit)                             # (0, 1]
        # spec ∈ [0, 2] (cosine-distance EMA); clamp at spec_ref to bound term
        spec_term = self.specificity[src].clamp(0, cfg.spec_ref) / cfg.spec_ref  # [0, 1]

        scores = (
            cfg.evict_w_visit * visit_term
            + cfg.evict_w_stale * stale_term
            + cfg.evict_w_norm * norm_term
            - cfg.evict_w_spec * spec_term
        )

        scores = scores.masked_fill(~evictable_mask, float("-inf"))
        victim = int(scores.argmax().item())
        return victim

    # ── Step counter management ─────────────────────────────────────────

    @torch.no_grad()
    def advance_step(self, n: int = 1) -> None:
        """Increment the global step counter. Called by the trainer."""
        self.step_counter += n

    # ── Diagnostics ─────────────────────────────────────────────────────

    @torch.no_grad()
    def edge_stats(self) -> dict:
        """Diagnostic statistics on edge buffer health."""
        cfg = self.cfg
        active = self.edge_active
        n_active = int(active.sum().item())
        n_total = active.numel()

        # Per-source fan-out
        fan_out = active.long().sum(dim=-1).float()        # [N]

        # Edge state norms (active edges only)
        norms = self.edge_state.norm(dim=-1)[active]       # flat
        spec = self.specificity[active]
        vc = self.visit_count[active].float()
        ages = (self.step_counter - self.alloc_step.clamp_min(0))[active].float()

        return {
            "n_active_edges": n_active,
            "active_fraction": n_active / max(n_total, 1),
            "mean_fan_out": float(fan_out.mean()),
            "max_fan_out": int(fan_out.max()),
            "min_fan_out_nonempty": int(fan_out[fan_out > 0].min())
            if (fan_out > 0).any() else 0,
            "mean_state_norm": float(norms.mean()) if n_active else 0.0,
            "mean_specificity": float(spec.mean()) if n_active else 0.0,
            "mean_visit_count": float(vc.mean()) if n_active else 0.0,
            "mean_age": float(ages.mean()) if n_active else 0.0,
        }

    @torch.no_grad()
    def snapshot_edge_state(self) -> dict:
        """Clone every per-edge buffer. Use this around eval to prevent val
        writes from contaminating the training manifold. Pair with
        `restore_edge_state(snap)` in a try/finally.
        """
        return {
            "edge_state": self.edge_state.clone(),
            "edge_dst": self.edge_dst.clone(),
            "edge_active": self.edge_active.clone(),
            "visit_count": self.visit_count.clone(),
            "last_visit": self.last_visit.clone(),
            "alloc_step": self.alloc_step.clone(),
            "specificity": self.specificity.clone(),
            "step_counter": self.step_counter.clone(),
        }

    @torch.no_grad()
    def restore_edge_state(self, snap: dict) -> None:
        self.edge_state.copy_(snap["edge_state"])
        self.edge_dst.copy_(snap["edge_dst"])
        self.edge_active.copy_(snap["edge_active"])
        self.visit_count.copy_(snap["visit_count"])
        self.last_visit.copy_(snap["last_visit"])
        self.alloc_step.copy_(snap["alloc_step"])
        self.specificity.copy_(snap["specificity"])
        self.step_counter.copy_(snap["step_counter"])

    @torch.no_grad()
    def reset_edge_memory(self) -> None:
        """Reset all edges to empty. Useful for restart-from-scratch
        scenarios; does NOT reset concept_ids."""
        self.edge_state.zero_()
        self.edge_dst.fill_(-1)
        self.edge_active.fill_(False)
        self.visit_count.zero_()
        self.last_visit.zero_()
        self.alloc_step.fill_(-1)
        self.specificity.zero_()
        # Keep step_counter — it represents training progress, not memory state
