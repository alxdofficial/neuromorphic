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
        # W-TinyLFU read/write touch EMAs — the eviction signal.
        # Both decay each step by `touch_ema_decay`; touch events
        # increment by 1. See `_select_eviction_victim`.
        self.register_buffer(
            "write_touches_ema",
            torch.zeros(N, K_max, dtype=torch.float32),
        )
        self.register_buffer(
            "read_touches_ema",
            torch.zeros(N, K_max, dtype=torch.float32),
        )
        # Global step counter — incremented externally by the trainer.
        # Used for age-based eviction protection.
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
        # W-TinyLFU: record a write touch on this edge.
        self.write_touches_ema[src, slot] += 1.0

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
        # Initial specificity = 1.0 (moderately novel by default).
        self.specificity[src, slot] = 1.0
        # W-TinyLFU: fresh slot. 1 write so far, 0 reads.
        self.write_touches_ema[src, slot] = 1.0
        self.read_touches_ema[src, slot] = 0.0

    def _free_slot(self, src: int, slot: int) -> None:
        """Reset a slot to empty (called during eviction)."""
        self.edge_state[src, slot].zero_()
        self.edge_dst[src, slot] = -1
        self.edge_active[src, slot] = False
        self.visit_count[src, slot] = 0
        self.last_visit[src, slot] = 0
        self.alloc_step[src, slot] = -1
        self.specificity[src, slot] = 0.0
        self.write_touches_ema[src, slot] = 0.0
        self.read_touches_ema[src, slot] = 0.0

    def _select_eviction_victim(self, src: int, step: int) -> int:
        """W-TinyLFU: evict the slot with the lowest read/write ratio.

        Effectiveness = (reads + α) / (writes + β). Higher = more retrieval
        bang per write event = keep. Lower = wrote often but rarely
        retrieved = evict. This works for long-horizon memory because the
        ratio doesn't decay with idle time, only with new touches — an
        edge can sit untouched for thousands of steps and still survive if
        its ratio is high.

        Brand-new edges (age < protect_min_age) are unconditionally
        protected — they haven't had enough chances to accumulate
        evidence. If too many slots are age-protected, the floor is
        bypassed (lockup prevention).
        """
        cfg = self.cfg
        K_max = cfg.K_max

        active = self.edge_active[src]                       # [K_max] bool
        if not active.any():
            return -1

        ages = step - self.alloc_step[src].clamp_min(0)
        ages = ages.masked_fill(~active, 0)
        young = ages < cfg.protect_min_age                   # [K_max] bool

        # Bypass age protection if too many slots are young (lockup risk).
        n_young = int((young & active).sum().item())
        if n_young > cfg.protect_max_frac * K_max:
            evictable_mask = active
        else:
            evictable_mask = active & ~young

        if not evictable_mask.any():
            return -1

        # W-TinyLFU effectiveness ratio with Laplace smoothing.
        reads = self.read_touches_ema[src]
        writes = self.write_touches_ema[src]
        effectiveness = (reads + cfg.evict_smoothing_alpha) / (
            writes + cfg.evict_smoothing_beta
        )                                                    # [K_max]
        # Pick argmin among evictable slots
        effectiveness = effectiveness.masked_fill(~evictable_mask, float("inf"))
        victim = int(effectiveness.argmin().item())
        return victim

    # ── Step counter management ─────────────────────────────────────────

    @torch.no_grad()
    def advance_step(self, n: int = 1) -> None:
        """Increment the global step counter AND decay touch EMAs.

        Decaying both EMAs once per step gives the W-TinyLFU eviction
        signal its memory horizon (set by `touch_ema_decay` — at 0.999,
        ~1000-step effective memory).
        """
        self.step_counter += n
        decay = float(self.cfg.touch_ema_decay) ** n
        self.write_touches_ema.mul_(decay)
        self.read_touches_ema.mul_(decay)

    @torch.no_grad()
    def record_read_touch(self, src: Tensor, dst: Tensor) -> None:
        """Record a 'read used this edge' event for each (src, dst) pair
        where an active edge from src already points at dst.

        Called from the walker after each READ-mode hop (write_mode=False).
        Vectorized via scatter_add on the flattened buffer.

        Args:
            src: [M] long — source node ids (one per traversal-hop event)
            dst: [M] long — chosen next_idx for each event
        """
        if src.numel() == 0:
            return
        K_max = self.cfg.K_max
        # For each m: which slot of src[m]'s edge list has edge_dst == dst[m]?
        slots = self.edge_dst[src]              # [M, K_max]
        active = self.edge_active[src]          # [M, K_max]
        matches = (slots == dst.unsqueeze(-1)) & active  # [M, K_max]
        if not matches.any():
            return
        # Flat indices into the [N, K_max] buffer; scatter_add ones.
        m_idx, k_idx = matches.nonzero(as_tuple=True)        # [num_matches] each
        src_for_match = src[m_idx]
        flat_idx = src_for_match * K_max + k_idx
        flat_buf = self.read_touches_ema.view(-1)
        flat_buf.scatter_add_(
            0, flat_idx,
            torch.ones_like(flat_idx, dtype=flat_buf.dtype),
        )

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
        reads = self.read_touches_ema[active]
        writes = self.write_touches_ema[active]
        # Effectiveness (the eviction signal). Reported in stats so we can
        # watch the distribution shift over training.
        effectiveness = (
            (reads + cfg.evict_smoothing_alpha)
            / (writes + cfg.evict_smoothing_beta)
        )

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
            "mean_read_touches": float(reads.mean()) if n_active else 0.0,
            "mean_write_touches": float(writes.mean()) if n_active else 0.0,
            "mean_effectiveness": float(effectiveness.mean()) if n_active else 0.0,
            "min_effectiveness": float(effectiveness.min()) if n_active else 0.0,
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
            "write_touches_ema": self.write_touches_ema.clone(),
            "read_touches_ema": self.read_touches_ema.clone(),
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
        self.write_touches_ema.copy_(snap["write_touches_ema"])
        self.read_touches_ema.copy_(snap["read_touches_ema"])
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
        self.write_touches_ema.zero_()
        self.read_touches_ema.zero_()
        # Keep step_counter — it represents training progress, not memory state
