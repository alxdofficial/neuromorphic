"""Column graph topology — static, seeded, built once at init.

What we build:
  * column_index[p, r, c]  — flat column index given (plane, row, col)
  * position[n]            — inverse: (plane, row, col) for flat index n
  * tile_ids[n]            — tile index in [0, num_tiles) for each column
  * input_positions / output_positions — column indices on plane 0 / L-1
  * out_nbrs[N, K]         — directed out-edge destinations per source
  * edge_src[N·K]          — flat source column per directed edge
  * edge_dst[N·K]          — flat destination column per directed edge

Design (from docs/column_graph.md §4.2):
  - Per column: K fixed out-edges drawn from
      * ~half: intra-plane Moore-radius-2 neighbours (25 candidates)
      * ~half: inter-plane next-plane-same-neighbourhood (25 candidates
        around the (row, col) on plane p+1 wrapped)
  - Then Watts-Strogatz rewire: each edge with probability p_rewire is
    replaced with a uniformly-random destination anywhere in the graph.
  - Static after init; only edge *biases* E_bias change during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Topology:
    """Bundled topology buffers — all tensors live on CPU at build time,
    call `.to(device)` via `Topology.move_to` before using on GPU."""

    out_nbrs: torch.Tensor        # [N, K] int64
    edge_src: torch.Tensor        # [N*K] int64 — source column for each directed edge
    edge_dst: torch.Tensor        # [N*K] int64 — destination column for each directed edge
    position: torch.Tensor        # [N, 3] int64 — (plane, row, col)
    plane_ids: torch.Tensor       # [N] int64
    input_positions: torch.Tensor  # [N_per_plane] int64
    output_positions: torch.Tensor  # [N_per_plane] int64

    def move_to(self, device: torch.device) -> "Topology":
        return Topology(
            out_nbrs=self.out_nbrs.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            position=self.position.to(device),
            plane_ids=self.plane_ids.to(device),
            input_positions=self.input_positions.to(device),
            output_positions=self.output_positions.to(device),
        )


def _moore_offsets(radius: int) -> list[tuple[int, int]]:
    """Moore-radius-R neighbourhood offsets including (0,0). Length = (2R+1)²."""
    offs = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            offs.append((dr, dc))
    return offs


def _intra_plane_candidates(
    rows: int, cols: int, r: int, c: int, radius: int
) -> list[tuple[int, int]]:
    """Torus-wrapped Moore neighbours within a plane, excluding self."""
    cands = []
    for dr, dc in _moore_offsets(radius):
        if dr == 0 and dc == 0:
            continue
        nr = (r + dr) % rows
        nc = (c + dc) % cols
        cands.append((nr, nc))
    return cands


def _inter_plane_candidates(
    rows: int, cols: int, r: int, c: int, radius: int
) -> list[tuple[int, int]]:
    """Same (row, col) on next plane plus Moore-radius neighbourhood around it."""
    cands = []
    for dr, dc in _moore_offsets(radius):
        nr = (r + dr) % rows
        nc = (c + dc) % cols
        cands.append((nr, nc))
    return cands  # includes (r, c) itself — the "straight-up" connection


def build_topology(
    plane_rows: int,
    plane_cols: int,
    L: int,
    K: int,
    p_rewire: float,
    K_intra_fraction: float,
    seed: int,
) -> Topology:
    """Build the static topology. All indices are flat column indices in [0, N)."""
    N_per_plane = plane_rows * plane_cols
    N = L * N_per_plane

    g = torch.Generator().manual_seed(seed)

    # ------- Column flat-index helpers -------
    def to_flat(p: int, r: int, c: int) -> int:
        return p * N_per_plane + r * plane_cols + c

    position = torch.empty(N, 3, dtype=torch.int64)
    plane_ids = torch.empty(N, dtype=torch.int64)
    for p in range(L):
        for r in range(plane_rows):
            for c in range(plane_cols):
                n = to_flat(p, r, c)
                position[n] = torch.tensor([p, r, c])
                plane_ids[n] = p

    # ------- I/O plane positions -------
    input_positions = torch.tensor(
        [to_flat(0, r, c) for r in range(plane_rows) for c in range(plane_cols)],
        dtype=torch.int64,
    )
    output_positions = torch.tensor(
        [to_flat(L - 1, r, c) for r in range(plane_rows) for c in range(plane_cols)],
        dtype=torch.int64,
    )

    # ------- Build out-edges -------
    K_intra = max(1, int(round(K * K_intra_fraction)))
    K_inter = K - K_intra  # rest from next-plane neighbourhood
    intra_radius = 2  # Moore-radius-2 = 25 candidates (exclude self = 24)
    inter_radius = 2  # same neighbourhood on next plane = 25 candidates

    out_nbrs = torch.empty(N, K, dtype=torch.int64)

    for p in range(L):
        for r in range(plane_rows):
            for c in range(plane_cols):
                src = to_flat(p, r, c)

                # Intra-plane candidates (exclude self)
                intra_cands = _intra_plane_candidates(
                    plane_rows, plane_cols, r, c, intra_radius
                )
                # Sample K_intra without replacement
                idx_perm = torch.randperm(len(intra_cands), generator=g)[:K_intra]
                chosen_intra = [intra_cands[int(i)] for i in idx_perm]
                intra_flat = [to_flat(p, nr, nc) for (nr, nc) in chosen_intra]

                # Inter-plane candidates (next plane, wrap at L-1 → 0)
                p_next = (p + 1) % L
                inter_cands = _inter_plane_candidates(
                    plane_rows, plane_cols, r, c, inter_radius
                )
                idx_perm = torch.randperm(len(inter_cands), generator=g)[:K_inter]
                chosen_inter = [inter_cands[int(i)] for i in idx_perm]
                inter_flat = [to_flat(p_next, nr, nc) for (nr, nc) in chosen_inter]

                edges = intra_flat + inter_flat
                assert len(edges) == K, (len(edges), K)
                out_nbrs[src] = torch.tensor(edges, dtype=torch.int64)

    # ------- Watts-Strogatz rewiring -------
    # For each edge, with probability p_rewire, replace destination with
    # uniform random column (allowing any plane).
    rand = torch.rand(N, K, generator=g)
    rewire_mask = rand < p_rewire
    num_rewires = int(rewire_mask.sum().item())
    if num_rewires > 0:
        new_dests = torch.randint(0, N, (num_rewires,), generator=g)
        out_nbrs[rewire_mask] = new_dests
    # Note: we allow self-loops after rewire (low probability; harmless —
    # the column just gets weighted echo of itself).

    # ------- Flat edge arrays -------
    edge_src = (
        torch.arange(N, dtype=torch.int64)
        .unsqueeze(1)
        .expand(N, K)
        .reshape(-1)
        .contiguous()
    )
    edge_dst = out_nbrs.reshape(-1).contiguous()

    return Topology(
        out_nbrs=out_nbrs,
        edge_src=edge_src,
        edge_dst=edge_dst,
        position=position,
        plane_ids=plane_ids,
        input_positions=input_positions,
        output_positions=output_positions,
    )
