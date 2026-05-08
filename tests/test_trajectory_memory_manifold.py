"""Unit tests for src.trajectory_memory.manifold."""

from __future__ import annotations

import pytest
import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import (
    Manifold,
    init_small_world_ring,
    scatter_mean_states,
)


# ── topology ────────────────────────────────────────────────────────────


def test_small_world_ring_shape():
    edges = init_small_world_ring(N=64, K_max=8, p_rewire=0.0, radius=4)
    assert edges.shape == (64, 8)
    assert edges.dtype == torch.int64


def test_small_world_ring_no_rewire_local():
    """With p_rewire=0, all edges should be within ±radius."""
    N, K, R = 64, 8, 4
    edges = init_small_world_ring(N=N, K_max=K, p_rewire=0.0, radius=R)
    positions = torch.arange(N, dtype=torch.int64).unsqueeze(1)   # [N, 1]
    # signed distance with wraparound
    signed_dist = (edges - positions + N // 2) % N - N // 2
    assert (signed_dist.abs() <= R).all(), (
        f"some edges outside radius={R} when p_rewire=0"
    )
    # No self-loops with offsets ±1..±R only
    assert (edges != positions).all()


def test_small_world_ring_full_rewire():
    """With p_rewire=1, edges should be uniform-random (mostly outside radius)."""
    N, K, R = 256, 8, 4
    edges = init_small_world_ring(N=N, K_max=K, p_rewire=1.0, radius=R)
    positions = torch.arange(N, dtype=torch.int64).unsqueeze(1)
    signed_dist = (edges - positions + N // 2) % N - N // 2
    # Most edges should be outside radius (only 2*R/N chance of falling inside)
    inside = (signed_dist.abs() <= R).float().mean()
    assert inside < 0.2, f"too many edges still local at p_rewire=1: {inside:.2%}"


def test_small_world_ring_K_too_large_raises():
    with pytest.raises(ValueError):
        # 2*radius=4 candidates, K=8 → impossible
        init_small_world_ring(N=64, K_max=8, p_rewire=0.5, radius=2)


# ── scatter_mean_states ─────────────────────────────────────────────────


def test_scatter_mean_passthrough_unvisited():
    """Concepts not in visited_ids should be unchanged."""
    BS, N, D = 1, 8, 4
    prev = torch.randn(BS, N, D)
    # Visit concept 3 only
    visited_ids = torch.tensor([[3]], dtype=torch.int64)          # [1, 1]
    visited_states = torch.zeros(BS, 1, D)
    new = scatter_mean_states(prev, visited_ids, visited_states)
    # All concepts except #3 unchanged
    mask = torch.ones(N, dtype=torch.bool)
    mask[3] = False
    assert torch.allclose(new[0, mask], prev[0, mask])
    # Concept #3 replaced with the mean of visited (just one value: zeros)
    assert torch.allclose(new[0, 3], torch.zeros(D))


def test_scatter_mean_averages_collisions():
    """Multiple visits to the same concept should average."""
    BS, N, D = 1, 8, 4
    prev = torch.zeros(BS, N, D)
    # 3 visits to concept 5 with values [1, 2, 3] (broadcast to D)
    visited_ids = torch.tensor([[5, 5, 5]], dtype=torch.int64)
    vals = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1).expand(BS, 3, D)
    new = scatter_mean_states(prev, visited_ids, vals)
    # Mean is 2.0
    assert torch.allclose(new[0, 5], torch.full((D,), 2.0))


def test_scatter_mean_no_inplace_mutation():
    """The original prev_states tensor should NOT be mutated."""
    BS, N, D = 2, 16, 8
    prev = torch.randn(BS, N, D)
    prev_clone = prev.clone()
    visited_ids = torch.randint(0, N, (BS, 4))
    visited_states = torch.randn(BS, 4, D)
    _ = scatter_mean_states(prev, visited_ids, visited_states)
    assert torch.allclose(prev, prev_clone), "scatter_mean mutated prev_states"


def test_scatter_mean_gradient_flows_to_visited():
    """Gradient should flow from new_states back to visited_states (for visited
    concepts) and to prev_states (for unvisited concepts)."""
    BS, N, D = 1, 8, 4
    prev = torch.randn(BS, N, D, requires_grad=True)
    visited_ids = torch.tensor([[0, 0, 5]], dtype=torch.int64)
    visited_states = torch.randn(BS, 3, D, requires_grad=True)
    new = scatter_mean_states(prev, visited_ids, visited_states)
    new.sum().backward()
    assert prev.grad is not None
    assert visited_states.grad is not None
    # Concept 0 was visited twice → grad concentrated there in visited_states
    # Unvisited concepts (1,2,3,4,6,7) should receive grad on prev (passthrough)
    # Visited concept (0, 5) should receive ZERO grad from prev (overwritten)
    unvisited_mask = torch.ones(N, dtype=torch.bool)
    unvisited_mask[[0, 5]] = False
    assert prev.grad[0, unvisited_mask].abs().sum() > 0
    assert prev.grad[0, ~unvisited_mask].abs().sum() == 0


# ── Manifold class ──────────────────────────────────────────────────────


def test_manifold_construct_and_shapes():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    assert m.concept_ids.shape == (cfg.N, cfg.D_concept)
    assert m.concept_states.shape == (cfg.N, cfg.D_concept)
    assert m.state_init.shape == (cfg.N, cfg.D_concept)
    assert m.edge_indices.shape == (cfg.N, cfg.K_max_neighbors)


def test_manifold_concept_ids_trainable():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    assert m.concept_ids.requires_grad
    assert m.state_init.requires_grad


def test_manifold_concept_states_is_buffer_not_param():
    """concept_states is a non-persistent buffer, not nn.Parameter."""
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    # Should be in buffers, not in parameters
    buffer_names = {n for n, _ in m.named_buffers()}
    param_names = {n for n, _ in m.named_parameters()}
    assert "concept_states" in buffer_names
    assert "concept_states" not in param_names


def test_manifold_reset_states_per_batch():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    BS = 3
    states = m.reset_states(batch_size=BS)
    assert states.shape == (BS, cfg.N, cfg.D_concept)
    # All batch elements should be initialized identically (broadcast)
    assert torch.allclose(states[0], states[1])
    assert torch.allclose(states[0], m.state_init)


def test_manifold_reset_states_grad_flows_to_state_init():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    states = m.reset_states(batch_size=2)
    states.sum().backward()
    assert m.state_init.grad is not None
    assert m.state_init.grad.abs().sum() > 0


def test_manifold_get_neighbor_ids_shapes():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    BS, J = 2, 3
    concept_id = torch.randint(0, cfg.N, (BS, J))
    nbr_ids = m.get_neighbor_ids(concept_id)
    assert nbr_ids.shape == (BS, J, cfg.K_max_neighbors, cfg.D_concept)


def test_manifold_gather_states():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    BS = 2
    states = m.reset_states(batch_size=BS)
    concept_id = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    gathered = m.gather_states(states, concept_id)
    assert gathered.shape == (BS, 3, cfg.D_concept)
    assert torch.allclose(gathered[0, 0], m.state_init[0])
    assert torch.allclose(gathered[1, 2], m.state_init[5])


def test_manifold_write_states_returns_new_tensor():
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    BS = 2
    prev = m.reset_states(batch_size=BS).clone()  # ensure not view
    visited_ids = torch.randint(0, cfg.N, (BS, 4))
    visited_states = torch.randn(BS, 4, cfg.D_concept)
    new = m.write_states(prev, visited_ids, visited_states)
    assert new.shape == prev.shape
    assert new is not prev  # new tensor
    # And m.concept_states (the buffer) should NOT have been touched.
    assert torch.allclose(m.concept_states, m.state_init.detach())
