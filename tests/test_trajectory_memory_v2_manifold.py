"""Unit tests for VocabularyManifold (v2 Phase 1)."""

from __future__ import annotations

import torch

from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold


def _small_cfg() -> TrajMemV2Config:
    cfg = TrajMemV2Config.small()  # N=64, D_concept=128, K_max=8
    cfg.protect_min_age = 2
    cfg.protect_min_spec = 0.5
    cfg.protect_min_norm = 0.1
    cfg.validate()
    return cfg


def test_init_shapes():
    """Manifold initializes with the right buffer shapes."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    assert m.id_basis.shape == (cfg.N, cfg.D_concept)
    assert m.edge_state.shape == (cfg.N, cfg.K_max, cfg.D_concept)
    assert m.edge_dst.shape == (cfg.N, cfg.K_max)
    assert m.edge_active.shape == (cfg.N, cfg.K_max)
    assert m.visit_count.shape == (cfg.N, cfg.K_max)
    # No edges active at init
    assert not m.edge_active.any()
    # Step counter starts at 0
    assert int(m.step_counter.item()) == 0


def test_concept_ids_is_id_proj_of_basis():
    """SimVQ: concept_ids = id_proj(id_basis); near-equal at init.

    id_proj is identity + small Gaussian perturbation (std=0.001), so
    concept_ids ≈ id_basis with a small symmetry-breaking offset. The
    perturbation prevents lock-step rotation of all concepts under
    shared id_proj.weight gradients early in training.
    """
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    cids = m.concept_ids
    assert cids.shape == (cfg.N, cfg.D_concept)
    # Match within a tolerance comparable to perturb_std × ||id_basis_row||
    # (≈ 0.001 × 0.02 × sqrt(D) ≈ 0.001 × 0.02 × ~11 ≈ 2e-4 for D=128).
    assert torch.allclose(cids, m.id_basis, atol=1e-2)


def test_concept_ids_gradient_flows_through_id_proj():
    """Gradient on any single concept_id row reaches id_proj.weight."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    # Pick row 5; sum it; backprop
    loss = m.concept_ids[5].sum()
    loss.backward()
    # id_proj.weight should have gradient because all rows go through it
    assert m.id_proj.weight.grad is not None
    assert m.id_proj.weight.grad.abs().sum() > 0


def test_lookup_edges_at_init_returns_inactive():
    """At init, looking up any source returns all-inactive slots."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    src = torch.tensor([0, 5, 13], dtype=torch.long)
    states, dsts, active = m.lookup_edges(src)
    assert states.shape == (3, cfg.K_max, cfg.D_concept)
    assert dsts.shape == (3, cfg.K_max)
    assert active.shape == (3, cfg.K_max)
    assert not active.any()
    assert (dsts == -1).all()
    assert torch.allclose(states, torch.zeros_like(states))


def test_update_allocates_new_edge():
    """First write to (src, dst) allocates a new edge."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    sig = torch.randn(1, cfg.D_concept)
    src = torch.tensor([3], dtype=torch.long)
    dst = torch.tensor([7], dtype=torch.long)
    slot_idx = m.update_edges(src, dst, sig)
    # One edge should be allocated
    assert slot_idx[0] >= 0
    assert m.edge_active[3].sum() == 1
    assert m.edge_dst[3, slot_idx[0]] == 7
    # State should equal the signature (initial alloc, no EMA blending)
    assert torch.allclose(m.edge_state[3, slot_idx[0]], sig[0])
    # Visit count = 1
    assert int(m.visit_count[3, slot_idx[0]]) == 1


def test_update_existing_edge_ema():
    """Re-write to same edge does EMA update, not new allocation."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    sig1 = torch.randn(1, cfg.D_concept)
    sig2 = torch.randn(1, cfg.D_concept)
    src = torch.tensor([3], dtype=torch.long)
    dst = torch.tensor([7], dtype=torch.long)

    slot1 = m.update_edges(src, dst, sig1)
    slot2 = m.update_edges(src, dst, sig2)

    # Same slot reused
    assert slot1[0] == slot2[0]
    # Only one edge active
    assert m.edge_active[3].sum() == 1
    # Visit count = 2
    assert int(m.visit_count[3, slot1[0]]) == 2
    # State is somewhere between sig1 and sig2 (EMA)
    state = m.edge_state[3, slot1[0]]
    assert not torch.allclose(state, sig1[0], atol=1e-3)
    assert not torch.allclose(state, sig2[0], atol=1e-3)


def test_fan_out_cap_triggers_eviction():
    """Allocating beyond K_max evicts an existing edge."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    src = 5
    # Allocate K_max edges from this src to distinct dests
    for dst in range(cfg.K_max):
        sig = torch.randn(1, cfg.D_concept)
        m.update_edges(
            torch.tensor([src], dtype=torch.long),
            torch.tensor([dst], dtype=torch.long),
            sig,
        )
        # Advance step counter so MIN_AGE doesn't protect everything
        m.advance_step()
    # All K_max slots should be active
    assert int(m.edge_active[src].sum()) == cfg.K_max

    # Advance well past min_age so some edges are evictable
    m.advance_step(100)

    # Now try to allocate one more — eviction must happen
    new_dst = cfg.K_max + 5
    sig = torch.randn(1, cfg.D_concept)
    slot = m.update_edges(
        torch.tensor([src], dtype=torch.long),
        torch.tensor([new_dst], dtype=torch.long),
        sig,
    )
    # Still K_max active (one was evicted, new one took its place)
    assert int(m.edge_active[src].sum()) == cfg.K_max
    # The new dst is present
    assert (m.edge_dst[src] == new_dst).any()


def test_eviction_picks_lowest_visit_count_under_equal_other_features():
    """Eviction prefers low-visit-count slots when other features are equal."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    src = 0
    # Fill all slots with edges at increasing visit counts
    for dst in range(cfg.K_max):
        sig = torch.randn(1, cfg.D_concept)
        m.update_edges(
            torch.tensor([src], dtype=torch.long),
            torch.tensor([dst], dtype=torch.long),
            sig,
        )
        # Visit some edges more than others — give dst=3 many extra updates
        for _ in range(dst):  # dst=0 gets 1 visit, dst=K_max-1 gets K_max visits
            m.update_edges(
                torch.tensor([src], dtype=torch.long),
                torch.tensor([dst], dtype=torch.long),
                torch.randn(1, cfg.D_concept),
            )
    # Advance step counter past min_age
    m.advance_step(100)

    # Now allocate one more; the edge with lowest visit count
    # (dst=0, visited once) should be the victim
    sig = torch.randn(1, cfg.D_concept)
    m.update_edges(
        torch.tensor([src], dtype=torch.long),
        torch.tensor([99], dtype=torch.long),
        sig,
    )

    # dst=0 should be gone; dst=99 should be present
    assert not (m.edge_dst[src] == 0).any()
    assert (m.edge_dst[src] == 99).any()


def test_alpha_floor_prevents_silent_freeze():
    """Heavy visit counts still allow some EMA learning (α >= α_min)."""
    cfg = _small_cfg()
    cfg.ema_alpha_min = 0.05
    m = VocabularyManifold(cfg)
    src = torch.tensor([0], dtype=torch.long)
    dst = torch.tensor([1], dtype=torch.long)
    # Allocate
    sig_init = torch.randn(1, cfg.D_concept)
    m.update_edges(src, dst, sig_init)
    # Many updates to drive visit_count high
    for _ in range(500):
        m.update_edges(src, dst, torch.zeros(1, cfg.D_concept))
    # After many updates with zero signature, state should be near zero
    # (because α_min=0.05 still allows the EMA to pull state toward zero)
    state_norm = m.edge_state[0].norm(dim=-1).max().item()
    assert state_norm < sig_init.norm().item() * 0.2, (
        "α_min floor not working: state didn't decay to near-zero after "
        "500 zero-signature updates"
    )


def test_specificity_protection():
    """Edges with high specificity resist eviction even when stale."""
    cfg = _small_cfg()
    cfg.protect_min_age = 1
    cfg.protect_min_spec = 0.5
    cfg.protect_min_norm = 0.1
    m = VocabularyManifold(cfg)
    src = 0
    # Allocate one edge with a strong, distinct signature (high specificity)
    strong_sig = torch.randn(1, cfg.D_concept) * 5.0
    m.update_edges(
        torch.tensor([src], dtype=torch.long),
        torch.tensor([42], dtype=torch.long),
        strong_sig,
    )
    # Fill remaining slots with weaker, redundant signatures
    weak_sig = torch.zeros(1, cfg.D_concept) + 0.001
    for dst in range(1, cfg.K_max):
        m.update_edges(
            torch.tensor([src], dtype=torch.long),
            torch.tensor([dst], dtype=torch.long),
            weak_sig.clone(),
        )

    # Advance past min_age
    m.advance_step(10)

    # Now write to a new dst — eviction must occur
    sig = torch.randn(1, cfg.D_concept)
    m.update_edges(
        torch.tensor([src], dtype=torch.long),
        torch.tensor([99], dtype=torch.long),
        sig,
    )

    # The strong-signature edge (dst=42) should NOT be evicted
    assert (m.edge_dst[src] == 42).any(), (
        "Specificity protection failed: strong-signature edge was evicted"
    )


def test_step_counter_persists():
    """Step counter is in state_dict for ckpt compatibility."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    m.advance_step(42)
    sd = m.state_dict()
    assert "step_counter" in sd
    assert int(sd["step_counter"].item()) == 42

    m2 = VocabularyManifold(cfg)
    m2.load_state_dict(sd)
    assert int(m2.step_counter.item()) == 42


def test_edge_stats_sanity():
    """edge_stats() produces sensible values."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    # Empty manifold
    stats = m.edge_stats()
    assert stats["n_active_edges"] == 0
    assert stats["active_fraction"] == 0.0

    # Allocate some edges
    for src in range(5):
        for dst in range(3):
            m.update_edges(
                torch.tensor([src], dtype=torch.long),
                torch.tensor([dst + 10], dtype=torch.long),
                torch.randn(1, cfg.D_concept),
            )
        m.advance_step()
    stats = m.edge_stats()
    assert stats["n_active_edges"] == 15  # 5 sources × 3 dests
    assert stats["mean_fan_out"] > 0


def test_reset_edge_memory_keeps_vocab():
    """Reset clears edges but preserves concept_ids and step_counter."""
    cfg = _small_cfg()
    m = VocabularyManifold(cfg)
    cids_before = m.concept_ids.clone()
    step_before = int(m.step_counter.item())

    # Allocate some edges
    for dst in range(5):
        m.update_edges(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([dst], dtype=torch.long),
            torch.randn(1, cfg.D_concept),
        )
    m.advance_step(10)

    assert m.edge_active.any()

    m.reset_edge_memory()
    assert not m.edge_active.any()
    assert torch.allclose(m.concept_ids, cids_before)
    # step_counter is preserved
    assert int(m.step_counter.item()) == step_before + 10
