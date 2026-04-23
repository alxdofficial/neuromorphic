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
    output_positions: torch.Tensor  # [N_per_plane] int64 — plane L-1 cols

    def move_to(self, device: torch.device) -> "Topology":
        return Topology(
            out_nbrs=self.out_nbrs.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            plane_ids=self.plane_ids.to(device),
            input_positions=self.input_positions.to(device),
            output_positions=self.output_positions.to(device),
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
) -> Topology:
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
    output_positions = torch.tensor(
        [to_flat(L - 1, r, c) for r in range(plane_rows) for c in range(plane_cols)],
        dtype=torch.int64,
    )

    # Out-edges
    K_intra = max(1, int(round(K * K_intra_fraction)))
    K_inter = K - K_intra
    intra_radius = 2
    inter_radius = 2

    out_nbrs = torch.empty(N, K, dtype=torch.int64)

    for p in range(L):
        for r in range(plane_rows):
            for c in range(plane_cols):
                src = to_flat(p, r, c)

                intra_cands = _intra_plane_candidates(plane_rows, plane_cols, r, c, intra_radius)
                idx = torch.randperm(len(intra_cands), generator=g)[:K_intra]
                intra_flat = [to_flat(p, *intra_cands[int(i)]) for i in idx]

                p_next = (p + 1) % L
                inter_cands = _inter_plane_candidates(plane_rows, plane_cols, r, c, inter_radius)
                idx = torch.randperm(len(inter_cands), generator=g)[:K_inter]
                inter_flat = [to_flat(p_next, *inter_cands[int(i)]) for i in idx]

                out_nbrs[src] = torch.tensor(intra_flat + inter_flat, dtype=torch.int64)

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
        output_positions=output_positions,
    )
