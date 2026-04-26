"""Graph topology — static after init. Same grid + Watts-Strogatz as
column-graph, but WITHOUT inverse adjacency (graph-walker doesn't need
scatter-gather; it walks).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Topology:
    out_nbrs: torch.Tensor        # [N, K] int64 — destination col for each out-edge
    edge_src: torch.Tensor        # [N*K] int64 — source col per flat-edge-idx
    edge_dst: torch.Tensor        # [N*K] int64 — dest col per flat-edge-idx
    plane_ids: torch.Tensor       # [N] int64
    input_positions: torch.Tensor  # [N_per_plane] int64 — plane 0 cols

    def move_to(self, device: torch.device) -> "Topology":
        return Topology(
            out_nbrs=self.out_nbrs.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            plane_ids=self.plane_ids.to(device),
            input_positions=self.input_positions.to(device),
        )


def _moore_offsets(radius: int) -> list[tuple[int, int]]:
    offs = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            offs.append((dr, dc))
    return offs


def _intra_plane_candidates(rows, cols, r, c, radius):
    cands = []
    for dr, dc in _moore_offsets(radius):
        if dr == 0 and dc == 0:
            continue
        nr = (r + dr) % rows
        nc = (c + dc) % cols
        cands.append((nr, nc))
    return cands


def _inter_plane_candidates(rows, cols, r, c, radius):
    cands = []
    for dr, dc in _moore_offsets(radius):
        nr = (r + dr) % rows
        nc = (c + dc) % cols
        cands.append((nr, nc))
    return cands


def build_topology(
    plane_rows: int,
    plane_cols: int,
    L: int,
    K: int,
    p_rewire: float,
    K_intra_fraction: float,
    seed: int,
    K_inter_bwd_fraction: float = 0.5,
    intra_radius: int = 2,
    inter_radius: int = 2,
) -> Topology:
    """Build the graph-walker topology.

    Each column gets ``K`` outgoing edges, partitioned into three groups:
      - ``K_intra``    edges to the same plane (Moore radius ``intra_radius``)
      - ``K_inter_fwd`` edges to plane ``(p + 1) % L`` (forward in depth)
      - ``K_inter_bwd`` edges to plane ``(p - 1) % L`` (backward in depth)

    Where:
      ``K_intra        = round(K · K_intra_fraction)``
      ``K_inter_total  = K − K_intra``
      ``K_inter_bwd    = round(K_inter_total · K_inter_bwd_fraction)``
      ``K_inter_fwd    = K_inter_total − K_inter_bwd``

    With defaults (K_intra_fraction=0.5, K_inter_bwd_fraction=0.5), edges
    split roughly into thirds across (same plane, forward, backward) — no
    structural feed-forward bias. Set ``K_inter_bwd_fraction=0.0`` for the
    legacy forward-only behaviour.

    On top of this structured topology, ``p_rewire`` fraction of edges have
    their destination replaced by a uniform-random global column.
    """
    N_per_plane = plane_rows * plane_cols
    N = L * N_per_plane

    g = torch.Generator().manual_seed(seed)

    def to_flat(p, r, c):
        return p * N_per_plane + r * plane_cols + c

    plane_ids = torch.empty(N, dtype=torch.int64)
    for p in range(L):
        for r in range(plane_rows):
            for c in range(plane_cols):
                plane_ids[to_flat(p, r, c)] = p

    input_positions = torch.tensor(
        [to_flat(0, r, c) for r in range(plane_rows) for c in range(plane_cols)],
        dtype=torch.int64,
    )

    # Out-edge budget: K = K_intra + K_inter_fwd + K_inter_bwd.
    K_intra = max(1, int(round(K * K_intra_fraction)))
    K_inter_total = K - K_intra
    K_inter_bwd = int(round(K_inter_total * K_inter_bwd_fraction))
    K_inter_fwd = K_inter_total - K_inter_bwd

    out_nbrs = torch.empty(N, K, dtype=torch.int64)

    for p in range(L):
        for r in range(plane_rows):
            for c in range(plane_cols):
                src = to_flat(p, r, c)

                # Intra-plane neighbours (same plane p).
                intra_cands = _intra_plane_candidates(
                    plane_rows, plane_cols, r, c, intra_radius,
                )
                idx = torch.randperm(len(intra_cands), generator=g)[:K_intra]
                intra_flat = [to_flat(p, *intra_cands[int(i)]) for i in idx]

                # Forward inter-plane neighbours (plane p+1).
                inter_cands = _inter_plane_candidates(
                    plane_rows, plane_cols, r, c, inter_radius,
                )
                if K_inter_fwd > 0:
                    p_fwd = (p + 1) % L
                    idx = torch.randperm(len(inter_cands), generator=g)[:K_inter_fwd]
                    inter_fwd_flat = [
                        to_flat(p_fwd, *inter_cands[int(i)]) for i in idx
                    ]
                else:
                    inter_fwd_flat = []

                # Backward inter-plane neighbours (plane p-1).
                if K_inter_bwd > 0:
                    p_bwd = (p - 1) % L
                    idx = torch.randperm(len(inter_cands), generator=g)[:K_inter_bwd]
                    inter_bwd_flat = [
                        to_flat(p_bwd, *inter_cands[int(i)]) for i in idx
                    ]
                else:
                    inter_bwd_flat = []

                out_nbrs[src] = torch.tensor(
                    intra_flat + inter_fwd_flat + inter_bwd_flat, dtype=torch.int64,
                )

    # Watts-Strogatz rewiring
    rand = torch.rand(N, K, generator=g)
    rewire_mask = rand < p_rewire
    if rewire_mask.any():
        new_dests = torch.randint(0, N, (int(rewire_mask.sum().item()),), generator=g)
        out_nbrs[rewire_mask] = new_dests

    edge_src = (
        torch.arange(N, dtype=torch.int64)
        .unsqueeze(1).expand(N, K).reshape(-1).contiguous()
    )
    edge_dst = out_nbrs.reshape(-1).contiguous()

    return Topology(
        out_nbrs=out_nbrs,
        edge_src=edge_src,
        edge_dst=edge_dst,
        plane_ids=plane_ids,
        input_positions=input_positions,
    )
