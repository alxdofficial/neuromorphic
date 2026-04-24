"""Tests for the graph-transformer neuromodulator path.

Covers:
- Subgraph helpers (enumerate_touched_edges, build_adjacency_bias)
- Zero-init safety: before any training, delta_nm starts as zero so
  enabling the neuromod doesn't change behavior.
- End-to-end: a phase1 training step with neuromod=True runs, produces
  finite stats, and yields non-zero gradients on the neuromod's params.
- E_bias evolution: after a plasticity fire, E_bias_flat incorporates
  the neuromod's contribution (non-zero where delta_nm contributed).
"""

from __future__ import annotations

import pytest
import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.neuromod import (
    NeuromodGraphTransformer,
    build_adjacency_bias,
    enumerate_touched_edges,
)
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def _tiny_cfg(use_neuromod: bool = True, **overrides) -> GraphWalkerConfig:
    base = dict(
        plane_rows=8, plane_cols=8, L=2,
        K=8, D_model=64, D_s=64, D_id=16,
        n_heads=2, n_hops=3,
        D_q_in=16, D_q_per_head=16, n_score_heads=2,
        K_horizons=4, K_buf=4,
        vocab_size=256,
        mod_period=4, tbptt_block=8,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0,
        use_neuromod=use_neuromod,
        neuromod_D_mod=32, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_rank=8, neuromod_eta=0.5,
    )
    base.update(overrides)
    return GraphWalkerConfig(**base)


# ----- subgraph helpers -----


def test_enumerate_touched_edges_small():
    # Tiny graph: 4 columns, K=2 neighbors each.
    out_nbrs = torch.tensor([
        [1, 2],  # col 0 → 1, 2
        [0, 3],  # col 1 → 0, 3
        [3, 0],  # col 2 → 3, 0
        [1, 2],  # col 3 → 1, 2
    ], dtype=torch.int64)
    # Touch cols {0, 1, 3}.
    touched_ids = torch.tensor([0, 1, 3], dtype=torch.int64)
    src_local, dst_local, edge_flat = enumerate_touched_edges(
        touched_ids, out_nbrs, K=2,
    )
    # Expected touched-touched edges:
    #   0→1 (src_local=0, dst_local=1, global_idx=0*2+0=0)
    #   1→0 (src_local=1, dst_local=0, global_idx=1*2+0=2)
    #   1→3 (src_local=1, dst_local=2, global_idx=1*2+1=3)
    #   3→1 (src_local=2, dst_local=1, global_idx=3*2+0=6)
    assert src_local.tolist() == [0, 1, 1, 2]
    assert dst_local.tolist() == [1, 0, 2, 1]
    assert edge_flat.tolist() == [0, 2, 3, 6]


def test_enumerate_touched_edges_empty():
    out_nbrs = torch.tensor([[1, 2], [0, 3]], dtype=torch.int64)
    touched_ids = torch.tensor([], dtype=torch.int64)
    src, dst, flat = enumerate_touched_edges(touched_ids, out_nbrs, K=2)
    assert src.numel() == 0
    assert dst.numel() == 0
    assert flat.numel() == 0


def test_build_adjacency_bias_small():
    out_nbrs = torch.tensor([
        [1, 2],
        [0, 3],
        [3, 0],
        [1, 2],
    ], dtype=torch.int64)
    touched_ids = torch.tensor([0, 1, 3], dtype=torch.int64)
    adj = build_adjacency_bias(touched_ids, out_nbrs)
    # Expected:
    #   row 0 (col 0): 1.0 at dst_local=1 (col 1)  — since 0→1 exists, 0→3 doesn't via these nbrs
    #   row 1 (col 1): 1.0 at dst_local=0 (col 0), 1.0 at dst_local=2 (col 3)
    #   row 2 (col 3): 1.0 at dst_local=1 (col 1)
    expected = torch.tensor([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    assert torch.equal(adj, expected)


# ----- neuromod module -----


def test_neuromod_zero_init_produces_zero_deltas():
    """With src/dst output projections zero-initialized, the neuromod must
    output zeros regardless of input. This is the 'safe bootstrap' property."""
    torch.manual_seed(0)
    U = 6
    D_feat, D_mod, n_layers, n_heads, rank = 48, 32, 1, 2, 8
    nm = NeuromodGraphTransformer(D_feat, D_mod, n_layers, n_heads, rank)
    features = torch.randn(U, D_feat)
    adj = (torch.rand(U, U) > 0.5).float()
    # Pick some arbitrary subset edges
    src = torch.tensor([0, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 3], dtype=torch.int64)
    deltas = nm(features, adj, src, dst)
    assert deltas.shape == (3,)
    assert torch.allclose(deltas, torch.zeros_like(deltas))


def test_neuromod_nonzero_output_after_perturbing_weights():
    torch.manual_seed(1)
    nm = NeuromodGraphTransformer(D_feat=32, D_mod=16, n_layers=1, n_heads=2, rank=4)
    # Break zero-init
    with torch.no_grad():
        nm.src_proj.weight.normal_(std=0.1)
        nm.dst_proj.weight.normal_(std=0.1)
    features = torch.randn(4, 32)
    adj = torch.zeros(4, 4)
    src = torch.tensor([0, 1, 2], dtype=torch.int64)
    dst = torch.tensor([1, 2, 3], dtype=torch.int64)
    deltas = nm(features, adj, src, dst)
    assert deltas.abs().sum() > 0


# ----- end-to-end integration -----


def test_neuromod_on_preserves_first_segment_behavior():
    """At day-0 (zero-init output), delta_nm should be zero for segment 0,
    so the first segment's logits with neuromod=True should match
    neuromod=False. Uses .train(False) to disable Gumbel stochasticity so
    the comparison is deterministic."""
    torch.manual_seed(0)
    cfg_off = _tiny_cfg(use_neuromod=False)
    cfg_on = _tiny_cfg(use_neuromod=True)
    torch.manual_seed(0)
    lm_off = StandaloneLM(cfg_off).cpu().train(False)
    torch.manual_seed(0)
    lm_on = StandaloneLM(cfg_on).cpu().train(False)
    # Copy shared parameters from off to on so they have the same values
    # (the two models have identical shared params; only lm_on has the extra
    # neuromod — whose outputs should be zero anyway).
    shared_off = dict(lm_off.named_parameters())
    for name, p_on in lm_on.named_parameters():
        if name in shared_off:
            with torch.no_grad():
                p_on.copy_(shared_off[name])

    tokens = torch.randint(0, 256, (2, 8))
    lm_off.memory.begin_segment(2, tokens.device)
    lm_on.memory.begin_segment(2, tokens.device)

    for t in range(8):
        r_off = lm_off.memory.step(tokens[:, t])
        r_on = lm_on.memory.step(tokens[:, t])
        # Logits should match (within float roundoff) because delta_nm=0.
        assert torch.allclose(r_off.logits, r_on.logits, atol=1e-5), (
            f"t={t}: logits diverged with neuromod=on vs off even at zero-init "
            f"(max diff {(r_off.logits - r_on.logits).abs().max().item():.2e})"
        )


def test_phase1_step_with_neuromod_runs_and_trains_neuromod():
    """Running a phase1 step with neuromod enabled, after perturbing its
    output weights so delta_nm is nonzero, must produce gradients on the
    neuromod parameters. Validates the full gradient path
    loss → routing → active_E_bias → delta_nm → neuromod params."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_neuromod=True, mod_period=4)  # Fire twice in T=8
    lm = StandaloneLM(cfg).cpu()
    # Break zero-init so delta_nm is non-trivial from the start (so grads
    # through it are non-zero even before any training has happened).
    with torch.no_grad():
        lm.memory.neuromod.src_proj.weight.normal_(std=0.1)
        lm.memory.neuromod.dst_proj.weight.normal_(std=0.1)

    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    tokens = torch.randint(0, 256, (2, 8))

    # Pre-seed the "previous snapshot" by running one segment first.
    # Without a prior snapshot, the first segment has delta_nm=None
    # and the neuromod params get no gradient.
    phase1_step(lm, opt, tokens, tbptt_block=4, amp_dtype=None, training_step=0)

    # Now run the actual measured step.
    params_before = {
        n: p.detach().clone() for n, p in lm.memory.neuromod.named_parameters()
    }
    stats = phase1_step(lm, opt, tokens, tbptt_block=4, amp_dtype=None, training_step=1)
    assert torch.isfinite(torch.tensor(stats.loss))

    # src_proj / dst_proj should have moved at least a bit from backward
    # having flowed through them.
    any_moved = False
    for n, p in lm.memory.neuromod.named_parameters():
        if "src_proj" in n or "dst_proj" in n:
            if not torch.equal(p, params_before[n]):
                any_moved = True
                break
    assert any_moved, "neuromod src/dst_proj did not update — grad path broken"


def test_neuromod_delta_snapshot_roundtrip():
    """The snapshot / begin_plastic_window cycle should be consistent:
    after a plasticity fire, the next window has a fresh delta_nm that
    depends on the snapshot taken from the just-closed window."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_neuromod=True, mod_period=4)
    lm = StandaloneLM(cfg).cpu()
    # Non-zero output weights so deltas will be non-trivial.
    with torch.no_grad():
        lm.memory.neuromod.src_proj.weight.normal_(std=0.1)
        lm.memory.neuromod.dst_proj.weight.normal_(std=0.1)

    tokens = torch.randint(0, 256, (2, 8))
    lm.memory.begin_segment(2, tokens.device)

    # Initial window: no prior snapshot, delta_nm should be None.
    assert lm.memory._active_delta_nm is None

    for t in range(4):
        lm.memory.step(tokens[:, t])
    # Plasticity should have fired at t=3 (window_len hit mod_period=4).
    # After the fire: snapshot taken, next window's delta_nm computed.
    assert lm.memory._prev_snapshot_ids is not None
    assert lm.memory._active_delta_nm is not None
    assert lm.memory._active_delta_nm.abs().sum() > 0, (
        "post-fire delta_nm is identically zero — the neuromod didn't contribute"
    )
