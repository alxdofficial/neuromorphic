"""Graph topology — single flat substrate of N columns laid out on a 2D
torus, with K out-edges per column drawn from a Moore-radius local
neighborhood and Watts-Strogatz random rewiring.

There are no planes — every column is structurally equal. Walkers
traverse this single homogeneous concept space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Topology:
    out_nbrs: torch.Tensor        # [N, K] int64 — destination col for each out-edge
    edge_src: torch.Tensor        # [N*K] int64 — source col per flat-edge-idx
    edge_dst: torch.Tensor        # [N*K] int64 — dest col per flat-edge-idx

    def move_to(self, device: torch.device) -> "Topology":
        return Topology(
            out_nbrs=self.out_nbrs.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
        )


def _moore_offsets(radius: int) -> list[tuple[int, int]]:
    offs = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            offs.append((dr, dc))
    return offs


def build_topology(
    grid_rows: int,
    grid_cols: int,
    K: int,
    p_rewire: float,
    seed: int,
    radius: int = 3,
    verbose: bool = False,
) -> Topology:
    """Build a single-grid Watts-Strogatz topology.

    Each of the ``N = grid_rows * grid_cols`` columns gets ``K`` outgoing
    edges. Initial neighbors are sampled (without replacement) from the
    local Moore neighborhood of radius ``radius`` (so each column has
    ``(2*radius+1)^2 - 1`` candidates). Then with probability ``p_rewire``
    each edge has its destination replaced by a uniform-random column
    anywhere in the graph — the Watts-Strogatz small-world shuffle.

    With ``radius=3`` we have ``48`` candidates per column, comfortably
    above ``K=16`` so the local-neighbor sampling is well-defined.

    With ``p_rewire=0.3`` the graph is well past the small-world transition
    (~0.1) and has plenty of long-range shortcuts so any column is reachable
    from any other in ~log(N) hops.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not 0.0 <= p_rewire <= 1.0:
        raise ValueError("p_rewire must be in [0, 1]")
    if radius < 1:
        raise ValueError("radius must be >= 1")

    N = grid_rows * grid_cols
    max_local = (2 * radius + 1) ** 2 - 1
    if K > max_local:
        raise ValueError(
            f"K={K} exceeds Moore-radius-{radius} candidate count "
            f"({max_local}). Increase radius or reduce K."
        )

    g = torch.Generator().manual_seed(seed)

    def to_flat(r: int, c: int) -> int:
        return r * grid_cols + c

    offsets = _moore_offsets(radius)
    out_nbrs = torch.empty(N, K, dtype=torch.int64)

    import time as _time
    t_start = _time.perf_counter()
    last_print = t_start
    cols_done = 0

    for r in range(grid_rows):
        for c in range(grid_cols):
            src = to_flat(r, c)
            if verbose:
                cols_done += 1
                now = _time.perf_counter()
                if now - last_print > 2.0:
                    pct = 100.0 * cols_done / N
                    rate = cols_done / (now - t_start)
                    eta = (N - cols_done) / max(rate, 1e-6)
                    print(
                        f"  [topology] {cols_done}/{N} cols ({pct:.0f}%), "
                        f"{rate:.0f} cols/s, ETA {eta:.0f}s",
                        flush=True,
                    )
                    last_print = now

            cands = [
                to_flat((r + dr) % grid_rows, (c + dc) % grid_cols)
                for dr, dc in offsets
            ]
            idx = torch.randperm(len(cands), generator=g)[:K]
            out_nbrs[src] = torch.tensor(
                [cands[int(i)] for i in idx], dtype=torch.int64,
            )

    # Watts-Strogatz rewiring
    rand = torch.rand(N, K, generator=g)
    rewire_mask = rand < p_rewire
    if rewire_mask.any():
        new_dests = torch.randint(0, N, (int(rewire_mask.sum().item()),), generator=g)
        out_nbrs[rewire_mask] = new_dests

    if verbose:
        print(
            f"  [topology] done in {_time.perf_counter() - t_start:.1f}s",
            flush=True,
        )

    edge_src = (
        torch.arange(N, dtype=torch.int64)
        .unsqueeze(1).expand(N, K).reshape(-1).contiguous()
    )
    edge_dst = out_nbrs.reshape(-1).contiguous()

    return Topology(
        out_nbrs=out_nbrs,
        edge_src=edge_src,
        edge_dst=edge_dst,
    )
