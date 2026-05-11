"""Manifold — N concepts (id + state vectors) on a sparse small-world graph.

Three tensors, all fixed size at construction:

    concept_ids    : [N, D_concept]   stable routing keys, only updated via backprop
    concept_states : [N, D_concept]   volatile content, mutates on writes
    edge_indices   : [N, K_max]       sparse adjacency, fixed at init

`concept_states` is held as a non-persistent buffer — a regular Tensor
(not an nn.Parameter) that receives gradient via paths that consume it,
and can be reset to `state_init` via `reset_states()`. Mutations during
training writes go through the **functional** `scatter_mean_states`,
which returns a NEW tensor — leaves the buffer intact for cross-window
autograd safety.

`state_init` is a learnable `[N, D_concept]` parameter that's the reset
target — not zeros — so concept_i has SOME content correlated with its
identity even at session start.

See docs/plan_trajectory_memory.md §2.1, §4.4, Appendix B.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig


def init_small_world_ring(
    N: int,
    K_max: int,
    p_rewire: float,
    radius: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Build a 1D-ring small-world adjacency tensor.

    Concepts laid out at positions 0..N-1 on a ring (wraparound). Each
    concept's default neighbors are the K_max closest positions within
    ±radius. Then with probability `p_rewire`, each edge gets its
    destination replaced with a uniformly random other concept (the
    Watts-Strogatz shuffle).

    Returns int64 [N, K_max] of neighbor concept IDs.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    # 2*radius local candidates, excluding self.
    if 2 * radius < K_max:
        raise ValueError(
            f"K_max={K_max} > 2*radius={2*radius} local candidates; "
            f"increase radius or reduce K_max."
        )

    # Default neighbor offsets: ±1, ±2, ..., ±radius. Pick K_max randomly
    # from the 2*radius candidates (without replacement) per concept.
    offsets_pool = torch.tensor(
        [d for d in range(-radius, radius + 1) if d != 0],
        dtype=torch.int64,
    )

    edge_indices = torch.empty(N, K_max, dtype=torch.int64)
    positions = torch.arange(N, dtype=torch.int64)

    for i in range(N):
        idx = torch.randperm(len(offsets_pool), generator=generator)[:K_max]
        chosen_offsets = offsets_pool[idx]
        edge_indices[i] = (positions[i] + chosen_offsets) % N

    # Watts-Strogatz rewiring: each edge has probability p_rewire of being
    # replaced. The replacement target must be (a) not the source row
    # itself (no self-loops) and (b) not already a neighbor of that row
    # (no duplicates within a row's neighbor list — these would reduce
    # effective branching and waste trajectory hops). We iterate per row
    # since the "no duplicates" constraint is per-row.
    rewire_mask = torch.rand(N, K_max, generator=generator) < p_rewire
    if rewire_mask.any():
        for r in range(N):
            cols = rewire_mask[r].nonzero(as_tuple=True)[0]
            if len(cols) == 0:
                continue
            # Forbidden = self + the K_max - len(cols) neighbors that
            # we're keeping (i.e., not rewiring this round).
            keep_mask = ~rewire_mask[r]
            forbidden = set(edge_indices[r][keep_mask].tolist()) | {r}
            # Rewire each marked column with a target that's not in
            # forbidden and not equal to any column we've already
            # rewired in this row.
            for c in cols.tolist():
                # Bounded retries — at default scale (N=2048, K_max=32)
                # this almost always succeeds in the first sample.
                for _ in range(50):
                    new = int(torch.randint(0, N, (1,), generator=generator).item())
                    if new not in forbidden:
                        edge_indices[r, c] = new
                        forbidden.add(new)
                        break
                # If all 50 retries fail (only possible when N is small
                # relative to K_max), leave the original local-zone
                # neighbor — this falls back to a non-rewired edge,
                # which is still a valid (local) connection.

    return edge_indices


def scatter_mean_states(
    prev_states: Tensor,
    visited_ids: Tensor,
    visited_states: Tensor,
) -> Tensor:
    """Functional scatter_mean — returns a NEW tensor, doesn't mutate prev.

    For each `i` in visited_ids, gather all visited_states with that id
    and average them. Concepts NOT in visited_ids are passed through
    unchanged from prev_states.

    Args:
        prev_states:    [N, D_concept]  manifold state going into this write
        visited_ids:    [M] or [BS, M]  concept IDs visited (M = J*K_write)
        visited_states: [M, D] or [BS, M, D]  proposed new states per visit

    Returns:
        new_states:     same shape as prev_states

    Implementation: scatter-add the visited states + counts, divide where
    any visit happened, leave prev_states untouched elsewhere.

    Backprops into both `prev_states` (for unvisited) and `visited_states`
    (for visited, weighted by 1/count). No in-place mutation — safe across
    TBPTT windows.
    """
    if visited_ids.dim() == 1:
        # Single-batch path. Add a leading dim.
        return scatter_mean_states(
            prev_states.unsqueeze(0),
            visited_ids.unsqueeze(0),
            visited_states.unsqueeze(0),
        ).squeeze(0)

    BS, M = visited_ids.shape
    N, D = prev_states.shape[-2:]
    assert prev_states.shape == (BS, N, D), (
        f"prev_states {prev_states.shape} != ({BS}, {N}, {D})"
    )
    assert visited_states.shape == (BS, M, D), (
        f"visited_states {visited_states.shape} != ({BS}, {M}, {D})"
    )

    # Sum of proposed states per concept ID, plus count.
    idx_d = visited_ids.unsqueeze(-1).expand(-1, -1, D)        # [BS, M, D]
    sum_buf = torch.zeros_like(prev_states)                    # [BS, N, D]
    sum_buf.scatter_add_(dim=1, index=idx_d, src=visited_states)

    ones = torch.ones(BS, M, 1, dtype=prev_states.dtype, device=prev_states.device)
    count_buf = torch.zeros(BS, N, 1, dtype=prev_states.dtype, device=prev_states.device)
    count_buf.scatter_add_(dim=1, index=visited_ids.unsqueeze(-1), src=ones)

    # mean = sum / count; for concepts not visited, leave prev_states unchanged.
    visited_mask = (count_buf > 0).to(prev_states.dtype)       # [BS, N, 1]
    safe_count = count_buf.clamp_min(1.0)
    mean_visited = sum_buf / safe_count
    new_states = visited_mask * mean_visited + (1.0 - visited_mask) * prev_states
    return new_states


class Manifold(nn.Module):
    """The concept manifold: ids, states, sparse edges, reset machinery.

    `concept_ids` and `state_init` are nn.Parameters (backprop-trainable).
    `concept_states` is a plain Tensor held as a non-persistent buffer —
    it's activation-like state that:
      - receives gradient through paths that consume it (cross-attn keys,
        history attention, etc.),
      - is reset in place to `state_init` by `reset_states()` for new
        sequences / inference sessions,
      - is mutated *functionally* by `scatter_mean_states` during writes
        (returns a new tensor, leaves the buffer intact for cross-window
        autograd safety).

    `edge_indices` is a registered buffer (int64 adjacency, fixed at init).

    **Dtype caveat.** Per plan §4.4, memory params are policy fp32 (bf16
    weight updates round small Adam steps to zero). PyTorch's `nn.Module
    .to(dtype)` walks all submodules and casts every floating buffer +
    parameter — calling it on the IntegratedLM containing this manifold
    would silently violate the fp32 policy. Use `.to(device)` / `.cuda()`
    only; load Llama in bf16 via `torch_dtype=` at construction time.
    """

    def __init__(self, cfg: TrajMemConfig):
        super().__init__()
        self.cfg = cfg
        N, D = cfg.N, cfg.D_concept

        # concept_ids: routing keys. Trainable. Init N(0, 1/sqrt(D)) per
        # plan §2.1 — keeps QK matmul magnitudes well-scaled at init.
        gen_c = torch.Generator().manual_seed(cfg.seed_concepts)
        ids_std = 1.0 / (D ** 0.5)
        ids_init = torch.empty(N, D)
        with torch.no_grad():
            ids_init.normal_(0.0, ids_std, generator=gen_c)
        self.concept_ids = nn.Parameter(ids_init)

        # state_init: learnable "good seed" the manifold resets to. Init at
        # the same Glorot-style scale as concept_ids (std=1/sqrt(D)) so the
        # per-concept state norm at init is O(1) and scale-invariant across
        # D_concept choices. The write_module's decay-gate equilibrium
        # bounds further drift; what state_init controls is the early-
        # training trajectory, not the steady-state magnitude.
        state_init_std = 1.0 / (D ** 0.5)
        state_init = torch.empty(N, D)
        with torch.no_grad():
            state_init.normal_(0.0, state_init_std, generator=gen_c)
        self.state_init = nn.Parameter(state_init)

        # concept_states: activation-like, starts at state_init's value.
        # Held as a non-persistent buffer so state-dict checkpoints don't
        # save activation snapshots; only state_init persists.
        self.register_buffer(
            "concept_states", state_init.detach().clone(),
            persistent=False,
        )

        # edge_indices: small-world ring rewire. Fixed at init.
        gen_t = torch.Generator().manual_seed(cfg.seed_topology)
        edges = init_small_world_ring(
            N=N,
            K_max=cfg.K_max_neighbors,
            p_rewire=cfg.p_rewire,
            radius=cfg.radius,
            generator=gen_t,
        )
        self.register_buffer("edge_indices", edges, persistent=True)

        # Per-concept usage EMA — tracks how often each concept gets selected
        # during routing. Updated by record_visits() each forward window;
        # consumed by revive_dead_concepts() to find dead concepts. Persists
        # in state_dict so a resumed run keeps its usage history.
        self.register_buffer(
            "usage_ema", torch.zeros(N), persistent=True,
        )

    # ── reset semantics ──────────────────────────────────────────────

    def reset_states(self, batch_size: int | None = None) -> Tensor:
        """Reset `concept_states` to `state_init`. Called at sequence start.

        If batch_size is provided, returns a per-batch state tensor of
        shape [BS, N, D_concept] — used inside TBPTT where each batch
        element's manifold evolves independently.

        If batch_size is None, mutates the buffer in place (useful for
        single-stream inference) and returns it.
        """
        if batch_size is None:
            with torch.no_grad():
                self.concept_states.copy_(self.state_init)
            return self.concept_states
        else:
            # Broadcast to [BS, N, D]; keep gradient flowing through state_init.
            return self.state_init.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    # ── usage tracking + dead-code revival ────────────────────────────

    @torch.no_grad()
    def record_visits(self, visited_ids: Tensor, ema_decay: float = 0.99) -> None:
        """Update per-concept usage EMA from a tensor of visit IDs.

        Standard VQ-VAE codebook-collapse mitigation pattern: each step,
        EMA-blend a per-concept selection histogram into a long-running
        usage tracker. `revive_dead_concepts()` then uses this to identify
        and re-initialize permanently dead concepts.

        Args:
            visited_ids: int64 of any shape — flatten counted as visits.
            ema_decay:   smoothing factor (0.99 ≈ 100-step horizon).
        """
        flat = visited_ids.reshape(-1)
        if flat.numel() == 0:
            return
        counts = torch.bincount(flat, minlength=self.cfg.N).to(self.usage_ema.dtype)
        # Normalize per-step counts to fraction so EMA is unit-scale
        # regardless of how many visits we tracked this call.
        counts = counts / counts.sum().clamp_min(1.0)
        self.usage_ema.mul_(ema_decay).add_(counts, alpha=1.0 - ema_decay)

    @torch.no_grad()
    def revive_dead_concepts(
        self, threshold: float = 1e-5, jitter: float = 0.02,
    ) -> int:
        """Re-initialize dead concepts (usage_ema < threshold).

        Replaces each dead concept's concept_id and state_init row with a
        randomly-chosen ACTIVE concept's row + small noise. This is the
        VQ-VAE codebook revival pattern: rescue dead codes by seeding them
        near a high-frequency code so they have a chance to take over a
        sub-region of the embedding space.

        Returns:
            Number of concepts that were revived.
        """
        usage = self.usage_ema
        N = self.cfg.N
        D = self.cfg.D_concept
        dead_mask = usage < threshold                                  # [N]
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0
        # Active concepts (above threshold). If none active yet (very
        # early training), do nothing — there's nothing to copy from.
        active_idx = (~dead_mask).nonzero(as_tuple=False).squeeze(-1)  # [n_active]
        if active_idx.numel() == 0:
            return 0
        # For each dead concept, pick a random active concept to seed from.
        dead_idx = dead_mask.nonzero(as_tuple=False).squeeze(-1)       # [n_dead]
        seed_pick = active_idx[
            torch.randint(0, active_idx.numel(), (n_dead,), device=usage.device)
        ]                                                              # [n_dead]
        # Replace concept_ids (the address keys) — this is what biases
        # routing toward the seed concept's neighborhood.
        noise_ids = torch.randn_like(self.concept_ids[dead_idx]) * jitter
        self.concept_ids.data[dead_idx] = (
            self.concept_ids.data[seed_pick] + noise_ids
        )
        # Replace state_init too (the per-concept default state) so reads
        # at revived concepts return sensible content, not zeros.
        noise_state = torch.randn_like(self.state_init[dead_idx]) * jitter
        self.state_init.data[dead_idx] = (
            self.state_init.data[seed_pick] + noise_state
        )
        # Reset usage so revived concepts get a fresh chance (don't
        # immediately re-revive next step).
        self.usage_ema[dead_idx] = threshold * 2.0
        return n_dead

    # ── neighbor lookup ──────────────────────────────────────────────

    def get_neighbor_ids(self, concept_id: Tensor) -> Tensor:
        """Look up neighbor id_vecs for given concept(s).

        Args:
            concept_id: int64 of any shape — concept indices to look up.

        Returns:
            neighbor id_vecs of shape (*concept_id.shape, K_max, D_concept).
        """
        nbr_idx = self.edge_indices[concept_id]              # (..., K_max)
        return self.concept_ids[nbr_idx]                     # (..., K_max, D)

    def get_neighbor_indices(self, concept_id: Tensor) -> Tensor:
        """Like `get_neighbor_ids` but returns the int64 indices, not vectors.

        Useful for indexing into the per-batch concept_states tensor.
        """
        return self.edge_indices[concept_id]                 # (..., K_max)

    # ── state access (per-batch) ──────────────────────────────────────

    def gather_states(
        self, states: Tensor, concept_id: Tensor,
    ) -> Tensor:
        """Gather concept states by id. Per-batch aware.

        Args:
            states:     [BS, N, D] per-batch concept_states tensor
            concept_id: int64 of any shape ending without the BS dim, e.g.
                        [BS, J] (current concept per trajectory) or
                        [BS, J, K_max] (neighbors of current).

        Returns:
            gathered states with shape concept_id.shape + (D,).
        """
        BS, N, D = states.shape
        # Flatten leading dims, gather, reshape back.
        flat_idx = concept_id.reshape(BS, -1)                    # [BS, M]
        flat_idx_d = flat_idx.unsqueeze(-1).expand(-1, -1, D)    # [BS, M, D]
        gathered = torch.gather(states, dim=1, index=flat_idx_d) # [BS, M, D]
        return gathered.reshape(*concept_id.shape, D)

    # ── update entry-point: functional scatter_mean ───────────────────

    def write_states(
        self,
        prev_states: Tensor,
        visited_ids: Tensor,
        visited_states: Tensor,
    ) -> Tensor:
        """Functional state update via scatter_mean. See `scatter_mean_states`."""
        return scatter_mean_states(prev_states, visited_ids, visited_states)
