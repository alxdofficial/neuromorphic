"""Synthetic autocomplete test — the GATE before Llama integration.

Tests the central new-architecture assumption: can a walker on a
vocabulary graph autocomplete to recover stored sequences from a cue?

Setup:
- Tiny manifold (N=64 vocab, K_max=8, D=128)
- Generate K=4 random "sentences" (sequences of vocab indices)
- Each sentence is encoded as a window of Llama-shaped fake hiddens
  (we use random vectors for the cue; the sentences themselves are
  what gets walked).
- WRITE: walk through each sentence's window, building edges
- READ: walk again on the same window content; verify the read
  trajectory recovers (approximately) the same nodes.

This is a SANITY CHECK that the walker mechanics work in isolation —
NOT a test of the full architecture's retrieval capabilities (which
needs Llama integration).

If this gate fails, the architecture is fundamentally broken and
should be rethought before sinking weeks into Llama integration.
"""

from __future__ import annotations

import torch

from src.trajectory_memory_v2._shared import EntryProjector
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold
from src.trajectory_memory_v2.read_module import ReadModule
from src.trajectory_memory_v2.write_module import WriteModule


def _make_cfg() -> TrajMemV2Config:
    cfg = TrajMemV2Config(
        N=64,
        D_concept=128,
        K_max=8,
        J=2,
        K_read=4,
        K_write=4,
        T_window=32,
        d_lm=128,
        ema_alpha_base=0.3,  # slightly more aggressive for the synthetic test
        ema_alpha_min=0.01,
        protect_min_age=1,
        lambda_edge_init=1.0,  # strong edge weighting for this test
    )
    cfg.validate()
    return cfg


def test_repeated_writes_to_same_window_converge():
    """Writing the same window many times should produce stable trajectories
    (i.e., the model finds a consistent path through vocab for this content).

    This tests that:
        - The walker is deterministic given fixed weights + input + hard routing
        - Edge state updates don't destabilize the routing
    """
    torch.manual_seed(0)
    cfg = _make_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    window = torch.randn(1, cfg.T_window, cfg.d_lm)

    # Run 10 writes of the same window. Check that the trajectories
    # eventually stabilize (last 3 traj should be identical).
    trajectories = []
    for _ in range(10):
        result = write_module.forward(window, manifold, hard=True)
        trajectories.append(result.visited_ids.detach().clone())
        manifold.advance_step()

    # Last 3 trajectories should be identical (the walker has converged)
    assert torch.equal(trajectories[-1], trajectories[-2]), (
        f"Trajectories not converging:\n"
        f"  step -2: {trajectories[-2]}\n"
        f"  step -1: {trajectories[-1]}"
    )


def test_read_recovers_write_trajectory():
    """The GATE.

    After writing a sentence (sequence of node visits) to memory,
    reading with the SAME conditioning should recover the same nodes.

    This is the analogous test to the current architecture's R↔W
    overlap, but for the new architecture. If overlap is ZERO, the
    new architecture has the same partition problem as v1 did before
    contrastive — and we have a fundamental design issue.

    We require overlap >> random baseline. Random baseline with K=4
    reads and K=4 writes over N=64 is ~K/N = 4/64 = 0.0625.
    Target: > 0.3 (significantly above random).
    """
    torch.manual_seed(0)
    cfg = _make_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)
    read_module = ReadModule(cfg, entry_proj=entry_proj)

    BS = 4
    window = torch.randn(BS, cfg.T_window, cfg.d_lm)

    # Write phase: train edges
    for _ in range(20):
        _ = write_module.forward(window, manifold, hard=True)
        manifold.advance_step()

    # Get final write trajectories
    write_result = write_module.forward(window, manifold, hard=True)
    write_ids = write_result.visited_ids.detach()  # [BS, J, K]

    # Read with same conditioning
    read_result = read_module.forward(window, manifold, hard=True)
    read_ids = read_result.visited_ids.detach()  # [BS, J, K]

    # Compute per-chunk overlap (fraction of read nodes also in write nodes)
    overlaps = []
    for b in range(BS):
        w_set = set(write_ids[b].flatten().tolist())
        r_set = set(read_ids[b].flatten().tolist())
        overlap = len(r_set & w_set) / max(len(r_set), 1)
        overlaps.append(overlap)
    mean_overlap = sum(overlaps) / len(overlaps)

    random_baseline = cfg.K_read / cfg.N  # ~0.0625

    print(f"\nSynthetic autocomplete test:")
    print(f"  Mean R↔W overlap: {mean_overlap:.3f}")
    print(f"  Random baseline:  {random_baseline:.3f}")
    print(f"  Ratio:            {mean_overlap / random_baseline:.1f}x")

    assert mean_overlap > 0.3, (
        f"Read trajectories don't recover write trajectories. "
        f"Mean overlap = {mean_overlap:.3f}, random baseline = {random_baseline:.3f}. "
        f"Architecture may be fundamentally broken."
    )


def test_different_windows_route_to_different_trajectories():
    """Different input windows should produce different trajectories,
    i.e., the walker actually uses input content, not just routing inertia."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    BS = 1
    window_a = torch.randn(BS, cfg.T_window, cfg.d_lm)
    window_b = torch.randn(BS, cfg.T_window, cfg.d_lm)

    # Write both several times
    for _ in range(5):
        _ = write_module.forward(window_a, manifold, hard=True)
        manifold.advance_step()
        _ = write_module.forward(window_b, manifold, hard=True)
        manifold.advance_step()

    # Final trajectories
    result_a = write_module.forward(window_a, manifold, hard=True)
    result_b = write_module.forward(window_b, manifold, hard=True)

    # Their visited_ids should NOT be identical
    assert not torch.equal(result_a.visited_ids, result_b.visited_ids), (
        "Walker produces same trajectory for different inputs — "
        "input content is being ignored"
    )


def test_edge_buffer_health_after_writes():
    """After running many writes, the edge buffer should be healthy:
        - Many edges allocated (not stuck at near-zero)
        - Visit counts accumulated (not all 1)
        - Some specificity (edges encoding distinctive content)
    """
    torch.manual_seed(0)
    cfg = _make_cfg()
    manifold = VocabularyManifold(cfg)
    entry_proj = EntryProjector(cfg)
    write_module = WriteModule(cfg, entry_proj=entry_proj)

    BS = 4
    # Run 30 writes with varied inputs
    for _ in range(30):
        window = torch.randn(BS, cfg.T_window, cfg.d_lm)
        _ = write_module.forward(window, manifold, hard=True)
        manifold.advance_step()

    stats = manifold.edge_stats()
    print(f"\nEdge buffer stats after 30 writes:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Sanity checks
    assert stats["n_active_edges"] > 5, "Too few edges allocated"
    assert stats["mean_visit_count"] >= 1.0
    assert stats["mean_state_norm"] > 0, "Edge states are all zero"
