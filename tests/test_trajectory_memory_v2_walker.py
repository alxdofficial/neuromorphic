"""Unit tests for TrajectoryWalker + Read/Write modules."""

from __future__ import annotations

import torch

from src.trajectory_memory.read_module import EntryProjector
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold
from src.trajectory_memory_v2.read_module import ReadModule
from src.trajectory_memory_v2.walker import TrajectoryWalker
from src.trajectory_memory_v2.write_module import WriteModule


def _small_cfg() -> TrajMemV2Config:
    cfg = TrajMemV2Config.small()
    cfg.protect_min_age = 1
    cfg.protect_min_spec = 0.1
    cfg.protect_min_norm = 0.05
    cfg.validate()
    return cfg


def test_walker_output_shapes():
    """Walker produces correctly-shaped outputs."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    walker = TrajectoryWalker(cfg, K=cfg.K_read)

    BS = 2
    T = cfg.T_window
    window = torch.randn(BS, T, cfg.d_lm)

    result = walker.forward(
        window_hiddens=window,
        entry_proj=entry_proj,
        manifold=manifold,
        write_mode=False,
    )

    assert result.visited_ids.shape == (BS, cfg.J, cfg.K_read)
    assert result.visited_embeds.shape == (BS, cfg.J, cfg.K_read, cfg.D_concept)
    assert result.step_queries.shape == (BS, cfg.J, cfg.K_read, cfg.D_concept)
    # Aux losses are scalars
    assert result.aux_lb.numel() == 1
    assert result.aux_z.numel() == 1


def test_read_does_not_modify_edges():
    """Read trajectories don't update edge state."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    read_module = ReadModule(cfg, entry_proj=entry_proj)

    edges_before = manifold.edge_active.sum().item()
    window = torch.randn(2, cfg.T_window, cfg.d_lm)
    _ = read_module.forward(window, manifold, hard=True)
    edges_after = manifold.edge_active.sum().item()

    assert edges_before == edges_after == 0


def test_write_creates_edges():
    """Write trajectories create new edges."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    edges_before = manifold.edge_active.sum().item()
    window = torch.randn(2, cfg.T_window, cfg.d_lm)
    _ = write_module.forward(window, manifold, hard=True)
    edges_after = manifold.edge_active.sum().item()

    # At cold start, every hop creates a new edge. K-1 hops × J trajectories × BS
    # = 3 × 2 × 2 = 12 max possible (some may collide on same (src,dst))
    assert edges_after > 0, "Write didn't create any edges"


def test_gradient_flows_to_trainable_modules():
    """Gradient from a downstream loss reaches step_mlp, cross_attn, entry_proj,
    lambda_edge, and concept_ids (id_basis, id_proj)."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    walker = TrajectoryWalker(cfg, K=cfg.K_read)

    BS = 2
    window = torch.randn(BS, cfg.T_window, cfg.d_lm)
    result = walker.forward(
        window_hiddens=window,
        entry_proj=entry_proj,
        manifold=manifold,
        write_mode=False,
    )
    # Simple loss: mean over visited embeddings
    loss = result.visited_embeds.mean() + result.aux_lb + result.aux_z
    loss.backward()

    # Trainable parameters that should have grad:
    assert entry_proj.entry_mlp[0].weight.grad is not None, "entry_proj.entry_mlp has no grad"
    assert entry_proj.entry_mlp[0].weight.grad.abs().sum() > 0

    assert walker.step_mlp[0].weight.grad is not None, "step_mlp has no grad"
    # First two visited_embeds (entry + first hop) involve step_mlp; should accumulate

    assert walker.lambda_edge.grad is not None, "lambda_edge has no grad"

    assert manifold.id_proj.weight.grad is not None, "id_proj has no grad"
    assert manifold.id_proj.weight.grad.abs().sum() > 0

    assert manifold.id_basis.grad is not None, "id_basis has no grad"
    # Note: id_basis grad may be very sparse (only rows visited get gradient
    # via direct gather), but at least *some* rows should have nonzero grad.
    assert manifold.id_basis.grad.abs().sum() > 0


def test_edge_state_buffer_no_optimizer_step():
    """Edge state is a buffer; it should NOT be in named_parameters."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    names = {n for n, _ in manifold.named_parameters()}
    # Only id_basis + id_proj.weight should be parameters
    assert "edge_state" not in names
    assert "edge_dst" not in names
    assert "visit_count" not in names
    assert "id_basis" in names
    assert "id_proj.weight" in names


def test_write_mode_updates_edge_state_in_place():
    """After a write, edge_state contains nonzero entries."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    # Initially zero
    assert manifold.edge_state.abs().sum() == 0

    window = torch.randn(2, cfg.T_window, cfg.d_lm)
    _ = write_module.forward(window, manifold, hard=True)

    # After a write, some edges should be non-zero
    assert manifold.edge_state.abs().sum() > 0


def test_repeated_write_does_ema_update():
    """Multiple writes through the same edge accumulate via EMA."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    # Run the SAME input twice. Should hit similar (src, dst) edges (the
    # entry routing is deterministic in hard mode), and visit_count
    # should accumulate.
    torch.manual_seed(42)
    window = torch.randn(2, cfg.T_window, cfg.d_lm)

    _ = write_module.forward(window, manifold, hard=True)
    visits_after_1 = manifold.visit_count.clone()
    _ = write_module.forward(window, manifold, hard=True)
    visits_after_2 = manifold.visit_count.clone()

    # Same trajectory → same edges → visit_count increases
    # At least some edges should have visit_count >= 2
    assert (visits_after_2 > 1).any(), (
        f"No edges had visit_count > 1 after 2 identical writes. "
        f"max visit_count = {visits_after_2.max().item()}"
    )


def test_read_after_write_uses_edges():
    """After writes establish edges, a read uses them in routing."""
    cfg = _small_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    read_module = ReadModule(cfg, entry_proj=entry_proj)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    torch.manual_seed(0)
    window = torch.randn(2, cfg.T_window, cfg.d_lm)

    # Cold start: read with no edges
    r_cold = read_module.forward(window, manifold, hard=True)
    assert r_cold.edge_score_active_frac == 0.0

    # Write to establish edges
    _ = write_module.forward(window, manifold, hard=True)
    # Advance step so MIN_AGE doesn't break things
    manifold.advance_step(10)

    # Read again — should now have active edges
    r_warm = read_module.forward(window, manifold, hard=True)
    assert r_warm.edge_score_active_frac > 0.0, (
        "After writes, reads should encounter active edges from current node"
    )
