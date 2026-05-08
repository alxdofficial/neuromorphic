"""Unit tests for src.trajectory_memory.read_module."""

from __future__ import annotations

import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold
from src.trajectory_memory.read_module import (
    ReadTrajectoryGenerator,
    gumbel_top1_ste,
    make_pos_enc,
)


# ── helpers ─────────────────────────────────────────────────────────────


def _small_setup(BS: int = 2) -> tuple[TrajMemConfig, Manifold, torch.Tensor, torch.Tensor]:
    cfg = TrajMemConfig.small()
    m = Manifold(cfg)
    states = m.reset_states(batch_size=BS)
    prev_hid = torch.randn(BS, cfg.T_window, cfg.d_lm)
    return cfg, m, states, prev_hid


# ── pos_enc ──────────────────────────────────────────────────────────────


def test_pos_enc_shape_and_values():
    pe = make_pos_enc(K=4, D=8)
    assert pe.shape == (4, 8)
    # Position 0 should be sin(0)=0, cos(0)=1 alternating
    expected_p0 = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    assert torch.allclose(pe[0], expected_p0)


# ── gumbel_top1_ste ─────────────────────────────────────────────────────


def test_gumbel_top1_inference_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    one_hot, idx = gumbel_top1_ste(logits, tau=1.0, hard=False)
    assert idx.tolist() == [1]
    assert one_hot[0, 1] == 1.0


def test_gumbel_top1_train_one_hot_forward():
    logits = torch.tensor([[1.0, 5.0, 2.0]], requires_grad=True)
    one_hot, idx = gumbel_top1_ste(logits, tau=1.0, hard=True)
    # Forward should be one-hot (sums to 1, exactly one slot at 1.0)
    assert torch.allclose(one_hot.sum(-1), torch.ones(1))
    assert ((one_hot == 0) | (one_hot == 1)).all()


def test_gumbel_top1_train_grad_flows():
    logits = torch.randn(4, 8, requires_grad=True)
    one_hot, _ = gumbel_top1_ste(logits, tau=1.0, hard=True)
    one_hot.sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


# ── ReadTrajectoryGenerator ─────────────────────────────────────────────


def test_read_module_construct():
    cfg = TrajMemConfig.small()
    rm = ReadTrajectoryGenerator(cfg)
    # head_query, MLPs, attn modules all present
    assert rm.head_query.shape == (cfg.J, cfg.D_concept)
    assert rm.entry_mlp is not None
    assert rm.history_attn is not None
    assert rm.cross_attn is not None
    assert rm.step_mlp is not None
    # pos_enc buffer
    assert rm.pos_enc.shape == (cfg.K_read + 1, cfg.D_concept)


def test_read_module_forward_shapes_inference():
    cfg, m, states, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    visited, vids = rm(prev_hid, states, m, hard=False)
    assert visited.shape == (2, cfg.J, cfg.K_read, cfg.D_concept)
    assert vids.shape == (2, cfg.J, cfg.K_read)
    assert vids.dtype == torch.int64
    # All visited ids should be valid concept indices
    assert (vids >= 0).all() and (vids < cfg.N).all()


def test_read_module_forward_shapes_train():
    cfg, m, states, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    visited, vids = rm(prev_hid, states, m, hard=True)
    assert visited.shape == (2, cfg.J, cfg.K_read, cfg.D_concept)


def test_read_module_grad_flows_to_states_and_module():
    """Gradient should flow to: read_module params, manifold.concept_ids,
    and the per-batch states tensor."""
    cfg, m, _, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    states = m.reset_states(batch_size=2)  # carries grad to state_init

    visited, _ = rm(prev_hid, states, m, hard=True)
    visited.sum().backward()

    # rm parameters should have gradient
    rm_grads = [p.grad for p in rm.parameters() if p.grad is not None]
    assert len(rm_grads) > 0, "no gradient flowed into read module"
    # concept_ids may or may not have gradient (only flows via QK matmul);
    # state_init definitely should (states gather → visited)
    assert m.state_init.grad is not None


def test_read_module_does_not_mutate_concept_states():
    """Read should be non-destructive: manifold's concept_states buffer
    should be unchanged after a forward pass."""
    cfg, m, _, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    before = m.concept_states.clone()
    states = m.reset_states(batch_size=2)
    _ = rm(prev_hid, states, m, hard=True)
    assert torch.allclose(m.concept_states, before)


def test_read_module_visited_states_match_states_at_visited_ids():
    """visited[b, j, t] should equal states[b, visited_ids[b, j, t]]."""
    cfg, m, states, prev_hid = _small_setup(BS=2)
    rm = ReadTrajectoryGenerator(cfg)
    visited, vids = rm(prev_hid, states, m, hard=False)
    # In read path, visited list contains the RAW pre-mutation states.
    BS = 2
    for b in range(BS):
        for j in range(cfg.J):
            for t in range(cfg.K_read):
                expected = states[b, vids[b, j, t]]
                assert torch.allclose(visited[b, j, t], expected, atol=1e-5), (
                    f"visited[{b},{j},{t}] doesn't match states at id={vids[b, j, t]}"
                )


def test_read_module_j_trajectories_diverge():
    """Different head_queries should produce different trajectories
    (mostly). With J=2 and small N, some collision is possible but not all
    visits should be identical."""
    cfg = TrajMemConfig.small()
    cfg.J = 4  # bump to make collision unlikely
    cfg.validate()
    m = Manifold(cfg)
    states = m.reset_states(batch_size=1)
    prev_hid = torch.randn(1, cfg.T_window, cfg.d_lm)
    rm = ReadTrajectoryGenerator(cfg)
    _, vids = rm(prev_hid, states, m, hard=False)
    # vids: [1, J, K]. Across the J trajectories, at least some hops
    # should differ.
    flattened = vids[0].tolist()                                   # [J, K]
    distinct_paths = len({tuple(p) for p in flattened})
    assert distinct_paths >= 2, (
        f"all J={cfg.J} trajectories produced the same path: {flattened}"
    )
