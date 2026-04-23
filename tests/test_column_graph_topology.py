"""Topology invariants for ColumnGraph."""

from __future__ import annotations

import pytest
import torch

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.topology import build_topology


def _build(**overrides):
    cfg = ColumnGraphConfig(**overrides) if overrides else ColumnGraphConfig()
    return cfg, build_topology(
        plane_rows=cfg.plane_rows,
        plane_cols=cfg.plane_cols,
        L=cfg.L,
        K=cfg.K,
        p_rewire=cfg.p_rewire,
        K_intra_fraction=cfg.K_intra_fraction,
        num_tiles_per_plane_dim=cfg.num_tiles_per_plane_dim,
        seed=cfg.topology_seed,
    )


def test_sizes():
    cfg, topo = _build()
    assert topo.out_nbrs.shape == (cfg.N, cfg.K)
    assert topo.edge_src.shape == (cfg.N * cfg.K,)
    assert topo.edge_dst.shape == (cfg.N * cfg.K,)
    assert topo.input_positions.shape == (cfg.N_per_plane,)
    assert topo.output_positions.shape == (cfg.N_per_plane,)
    assert topo.tile_ids.shape == (cfg.N,)


def test_destinations_in_range():
    cfg, topo = _build()
    assert topo.out_nbrs.min() >= 0
    assert topo.out_nbrs.max() < cfg.N


def test_input_output_planes_distinct():
    cfg, topo = _build()
    # Input plane is 0, output plane is L-1.
    assert torch.all(topo.plane_ids[topo.input_positions] == 0)
    assert torch.all(topo.plane_ids[topo.output_positions] == cfg.L - 1)


def test_tile_ids_partition_columns():
    cfg, topo = _build()
    # Every column gets a tile_id in [0, num_tiles)
    assert topo.tile_ids.min() >= 0
    assert topo.tile_ids.max() < cfg.num_tiles
    # Every tile contains exactly tile_size columns
    counts = torch.bincount(topo.tile_ids, minlength=cfg.num_tiles)
    expected = cfg.tile_size
    assert torch.all(counts == expected), f"tile size mismatch: {counts}"


def test_edge_src_dst_consistency():
    cfg, topo = _build()
    # edge_src[i*K + k] should equal i, edge_dst[i*K + k] should equal out_nbrs[i, k]
    N, K = cfg.N, cfg.K
    assert torch.all(topo.edge_src == torch.arange(N).unsqueeze(1).expand(N, K).reshape(-1))
    assert torch.all(topo.edge_dst == topo.out_nbrs.reshape(-1))


def test_watts_strogatz_produces_cross_plane_skip_connections():
    """At p_rewire=0.30 and N ~ 4K, we expect many input→output direct shortcuts."""
    cfg, topo = _build()
    input_cols_set = set(topo.input_positions.tolist())
    output_cols_set = set(topo.output_positions.tolist())
    direct_shortcuts = 0
    for src in input_cols_set:
        for k in range(cfg.K):
            if int(topo.out_nbrs[src, k]) in output_cols_set:
                direct_shortcuts += 1
                break
    fraction = direct_shortcuts / len(input_cols_set)
    # Theory says ~93% at p=0.30, N=4096, K=32. Allow wide margin for
    # small-graph randomness.
    assert fraction > 0.5, f"only {fraction:.2f} input cols have direct output shortcut"


def test_determinism():
    cfg, topo_a = _build()
    _, topo_b = _build()
    assert torch.all(topo_a.out_nbrs == topo_b.out_nbrs)
    assert torch.all(topo_a.tile_ids == topo_b.tile_ids)


def test_config_validates():
    with pytest.raises(ValueError):
        ColumnGraphConfig(plane_rows=33, num_tiles_per_plane_dim=4)
    with pytest.raises(ValueError):
        ColumnGraphConfig(L=1)
    with pytest.raises(ValueError):
        ColumnGraphConfig(p_rewire=1.5)
