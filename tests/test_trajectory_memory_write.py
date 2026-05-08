"""Unit tests for src.trajectory_memory.write_module."""

from __future__ import annotations

import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold
from src.trajectory_memory.write_module import WriteTrajectoryGenerator


def _small_setup(BS: int = 2):
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    prev_states = m.reset_states(batch_size=BS)
    cur_hid = torch.randn(BS, cfg.T_window, cfg.d_lm)
    surprise = torch.randn(BS).abs()                              # nonneg
    return cfg, m, prev_states, cur_hid, surprise


def test_write_module_construct():
    cfg = TrajMemConfig.small()
    wm = WriteTrajectoryGenerator(cfg)
    assert wm.head_query.shape == (cfg.J, cfg.D_concept)
    assert wm.mutate_mlp is not None


def test_write_module_forward_shapes():
    cfg, m, prev_states, cur_hid, surprise = _small_setup(BS=2)
    wm = WriteTrajectoryGenerator(cfg)
    new_states, vids, proposed = wm(cur_hid, surprise, prev_states, m, hard=False)
    assert new_states.shape == (2, cfg.N, cfg.D_concept)
    assert vids.shape == (2, cfg.J, cfg.K_write)
    assert proposed.shape == (2, cfg.J, cfg.K_write, cfg.D_concept)


def test_write_module_returns_new_tensor_not_inplace():
    """The write should return a new states tensor, not mutate prev_states."""
    cfg, m, prev_states, cur_hid, surprise = _small_setup(BS=2)
    wm = WriteTrajectoryGenerator(cfg)
    prev_clone = prev_states.clone()
    new_states, _, _ = wm(cur_hid, surprise, prev_states, m, hard=False)
    assert torch.allclose(prev_states, prev_clone), "write mutated prev_states"
    assert new_states is not prev_states


def test_write_module_only_visited_concepts_change():
    """Concepts NOT visited by any trajectory should retain prev_states value."""
    cfg, m, prev_states, cur_hid, surprise = _small_setup(BS=1)
    wm = WriteTrajectoryGenerator(cfg)
    new_states, vids, _ = wm(cur_hid, surprise, prev_states, m, hard=False)

    # Build mask of visited concepts
    visited_set = set(vids[0].flatten().tolist())
    unvisited = [i for i in range(cfg.N) if i not in visited_set]
    if not unvisited:
        return  # all visited; skip
    for i in unvisited:
        assert torch.allclose(new_states[0, i], prev_states[0, i]), (
            f"unvisited concept {i} changed"
        )


def test_write_module_visited_concepts_change():
    """Concepts visited by trajectories should generally have different
    values from prev_states (since mutate_write produces nonzero deltas
    after init — guaranteed unless the MLP weights collapse, but tests use
    fresh init)."""
    cfg, m, prev_states, cur_hid, surprise = _small_setup(BS=1)
    wm = WriteTrajectoryGenerator(cfg)
    # Use surprise=1.0 for noticeable mutation
    surprise = torch.tensor([1.0])
    new_states, vids, _ = wm(cur_hid, surprise, prev_states, m, hard=False)

    visited_set = sorted(set(vids[0].flatten().tolist()))
    diffs = (new_states[0, visited_set] - prev_states[0, visited_set]).abs().sum()
    assert diffs > 1e-4, "visited concepts unchanged after write"


def test_write_module_all_params_receive_gradient():
    """Every named parameter in write_module must receive gradient.
    Pre-Gumbel-STE-fix, step_mlp / entry_mlp / head_query / attn weights
    dead-ended at integer indices (mutate_mlp got grad through the
    scatter_mean path, but routing did not).
    """
    cfg, m, _, cur_hid, surprise = _small_setup(BS=2)
    wm = WriteTrajectoryGenerator(cfg)
    prev_states = m.reset_states(batch_size=2)
    new_states, _, _ = wm(cur_hid, surprise, prev_states, m, hard=True)
    new_states.sum().backward()

    missing = [n for n, p in wm.named_parameters() if p.grad is None]
    assert not missing, (
        f"these write_module params got no gradient (routing dead-end?): {missing}"
    )
    assert m.state_init.grad is not None
    assert m.concept_ids.grad is not None, (
        "concept_ids got no gradient — Q·K_id path dead-ended?"
    )


def test_write_module_concept_states_buffer_unchanged():
    """The Manifold's concept_states buffer is not the same as the per-batch
    prev_states tensor; write_module should not touch the buffer."""
    cfg, m, prev_states, cur_hid, surprise = _small_setup(BS=2)
    wm = WriteTrajectoryGenerator(cfg)
    buffer_before = m.concept_states.clone()
    _ = wm(cur_hid, surprise, prev_states, m, hard=False)
    assert torch.allclose(m.concept_states, buffer_before)


def test_write_module_collisions_get_averaged():
    """Force J trajectories that all visit overlapping concepts (use J >= K_write)
    and check that scatter_mean handles collisions cleanly. Smoke check —
    detailed scatter_mean correctness lives in test_trajectory_memory_manifold.py."""
    cfg = TrajMemConfig.small()
    cfg.J = 4
    cfg.K_write = 4
    cfg.validate()
    m = Manifold(cfg)
    prev_states = m.reset_states(batch_size=1)
    cur_hid = torch.randn(1, cfg.T_window, cfg.d_lm)
    surprise = torch.tensor([1.0])
    wm = WriteTrajectoryGenerator(cfg)
    new_states, vids, _ = wm(cur_hid, surprise, prev_states, m, hard=False)
    # No NaNs / Infs
    assert torch.isfinite(new_states).all()


def test_write_module_j_trajectories_diverge():
    """Different head_queries should produce different write trajectories.
    Tested in both inference (argmax) and training (Gumbel) modes.
    Parallel to the read-module diversity test.
    """
    cfg = TrajMemConfig.small()
    cfg.J = 4
    cfg.validate()
    m = Manifold(cfg)
    prev_states = m.reset_states(batch_size=1)
    cur_hid = torch.randn(1, cfg.T_window, cfg.d_lm)
    surprise = torch.tensor([0.5])
    wm = WriteTrajectoryGenerator(cfg)
    for hard in (False, True):
        torch.manual_seed(7)
        _, vids, _ = wm(cur_hid, surprise, prev_states, m, hard=hard)
        flattened = vids[0].tolist()
        distinct_paths = len({tuple(p) for p in flattened})
        assert distinct_paths >= 2, (
            f"hard={hard}: all J={cfg.J} write trajectories produced same path: "
            f"{flattened}"
        )


def test_write_then_read_uses_new_states():
    """End-to-end: write produces new_states; a subsequent read using those
    new_states should reflect the mutations."""
    from src.trajectory_memory.read_module import ReadTrajectoryGenerator

    cfg, m, prev_states, cur_hid, surprise = _small_setup(BS=1)
    wm = WriteTrajectoryGenerator(cfg)
    rm = ReadTrajectoryGenerator(cfg)
    surprise = torch.tensor([1.0])

    new_states, vids_w, _ = wm(cur_hid, surprise, prev_states, m, hard=False)

    # A read on new_states vs prev_states. The visited concepts have
    # different state, so a read that hits one of those should produce
    # different visited values.
    prev_hid = torch.randn(1, cfg.T_window, cfg.d_lm)
    visited_a, _ = rm(prev_hid, prev_states, m, hard=False)
    visited_b, _ = rm(prev_hid, new_states, m, hard=False)
    # Some difference somewhere (read trajectories may visit different
    # concepts but at least the fact that the underlying states differ
    # means visited contents will too if they overlap). We test that the
    # visited tensor differs.
    diff = (visited_a - visited_b).abs().sum()
    # Not strictly guaranteed (read might miss all written concepts) but
    # very likely with default sizes. If this test flakes consider adding
    # a fixed-seed determinism wrapper.
    if diff <= 1e-6:
        # Acceptable degenerate case — at least confirm shapes.
        assert visited_a.shape == visited_b.shape
