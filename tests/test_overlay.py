"""Tests for the active-row overlay (src/graph_walker/overlay.py).

The overlay is a two-tier state representation:
- s_base: dense detached [B, N, D_s]
- (active_flat_idx, active_val): sorted sparse differentiable overlay

Correctness oracle: a "dense clone" implementation that maintains a single
[B, N, D_s] tensor updated via the same LIF formula. Overlay reads and
updates must match the dense reference exactly.
"""

from __future__ import annotations

import torch

from src.graph_walker.overlay import (
    commit_overlay_to_base,
    empty_overlay,
    overlay_gather,
    overlay_lif_update,
)


def _lif_reference(
    s_dense: torch.Tensor,        # [B, N, D_s]
    messages: torch.Tensor,       # [M, D_s]
    dests: torch.Tensor,          # [M] flat indices into B·N
    alpha: torch.Tensor,          # [N]
    N: int,
) -> torch.Tensor:
    """Dense reference: clone, aggregate messages, blend, return."""
    if messages.numel() == 0:
        return s_dense.clone()
    D_s = s_dense.shape[-1]
    s_flat = s_dense.reshape(-1, D_s).clone()
    unique_dests, inverse = torch.unique(dests, return_inverse=True)
    incoming = torch.zeros(
        unique_dests.shape[0], D_s, device=s_dense.device, dtype=torch.float32,
    )
    incoming.index_add_(0, inverse, messages.float())
    s_old = s_flat[unique_dests].float()
    alpha_u = alpha[unique_dests % N].float().unsqueeze(-1)
    s_new = alpha_u * s_old + (1.0 - alpha_u) * torch.tanh(incoming)
    s_flat.index_copy_(0, unique_dests, s_new.to(s_flat.dtype))
    return s_flat.view(s_dense.shape)


def _materialize_overlay(
    s_base: torch.Tensor,
    active_flat_idx: torch.Tensor,
    active_val: torch.Tensor,
) -> torch.Tensor:
    """Combine (base, overlay) into a dense [B, N, D_s] for equality checks."""
    D_s = s_base.shape[-1]
    dense = s_base.clone().view(-1, D_s)
    if active_flat_idx.numel() > 0:
        dense[active_flat_idx] = active_val.detach()
    return dense.view(s_base.shape)


def test_gather_empty_overlay_returns_base():
    torch.manual_seed(0)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)
    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    query = torch.tensor([0, 5, 10], dtype=torch.int64)
    out = overlay_gather(idx, val, s_base, query)
    expected = s_base.view(-1, D_s)[query]
    assert torch.equal(out, expected)


def test_gather_hits_overlay_when_present():
    torch.manual_seed(1)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)
    # Put rows 3 and 10 in the overlay with known values
    idx = torch.tensor([3, 10], dtype=torch.int64)
    val = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])
    query = torch.tensor([0, 3, 10, 15], dtype=torch.int64)
    out = overlay_gather(idx, val, s_base, query)
    # Positions 0 and 15 should come from base; 3 and 10 from overlay.
    assert torch.equal(out[0], s_base.view(-1, D_s)[0])
    assert torch.equal(out[1], val[0])
    assert torch.equal(out[2], val[1])
    assert torch.equal(out[3], s_base.view(-1, D_s)[15])


def test_single_lif_update_matches_reference():
    """One update step from an empty overlay produces the same state."""
    torch.manual_seed(2)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)
    alpha = torch.sigmoid(torch.randn(N))
    messages = torch.randn(3, D_s)
    dests = torch.tensor([1, 5, 12], dtype=torch.int64)

    # Reference (dense)
    s_ref = _lif_reference(s_base, messages, dests, alpha, N)

    # Overlay
    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    idx, val = overlay_lif_update(idx, val, s_base, messages, dests, alpha, N)

    s_overlay = _materialize_overlay(s_base, idx, val)
    assert torch.allclose(s_ref, s_overlay, atol=1e-6)


def test_multi_step_lif_matches_reference():
    """Three consecutive updates on overlapping rows should match dense."""
    torch.manual_seed(3)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)
    alpha = torch.sigmoid(torch.randn(N))

    # Step 1: touch rows 2, 5, 11
    msg1 = torch.randn(3, D_s)
    d1 = torch.tensor([2, 5, 11], dtype=torch.int64)
    # Step 2: touch rows 2 (revisit), 7, 14
    msg2 = torch.randn(3, D_s)
    d2 = torch.tensor([2, 7, 14], dtype=torch.int64)
    # Step 3: touch rows 5 (revisit), 11 (revisit), 3
    msg3 = torch.randn(3, D_s)
    d3 = torch.tensor([5, 11, 3], dtype=torch.int64)

    # Dense reference: threading through multi-step updates requires that
    # each step reads from the LATEST state. So we apply LIF sequentially.
    s_ref = s_base.clone()
    s_ref = _lif_reference(s_ref, msg1, d1, alpha, N)
    s_ref = _lif_reference(s_ref, msg2, d2, alpha, N)
    s_ref = _lif_reference(s_ref, msg3, d3, alpha, N)

    # Overlay
    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    idx, val = overlay_lif_update(idx, val, s_base, msg1, d1, alpha, N)
    idx, val = overlay_lif_update(idx, val, s_base, msg2, d2, alpha, N)
    idx, val = overlay_lif_update(idx, val, s_base, msg3, d3, alpha, N)

    s_overlay = _materialize_overlay(s_base, idx, val)
    assert torch.allclose(s_ref, s_overlay, atol=1e-6)

    # Overlay should contain exactly the union of touched rows.
    expected_touched = torch.unique(torch.cat([d1, d2, d3])).sort().values
    assert torch.equal(idx, expected_touched)


def test_collision_same_step_sums_messages():
    """Two messages to the same dest in one step should sum before LIF."""
    torch.manual_seed(4)
    B, N, D_s = 2, 8, 4
    s_base = torch.zeros(B, N, D_s)                  # zero base keeps math easy
    alpha = torch.full((N,), 0.5)
    msg_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    msg_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    dest = torch.tensor([5, 5], dtype=torch.int64)
    msgs = torch.cat([msg_a, msg_b], dim=0)

    # Reference
    s_ref = _lif_reference(s_base, msgs, dest, alpha, N)

    # Overlay
    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    idx, val = overlay_lif_update(idx, val, s_base, msgs, dest, alpha, N)

    s_overlay = _materialize_overlay(s_base, idx, val)
    assert torch.allclose(s_ref, s_overlay, atol=1e-6)
    # Only row 5 should be in the overlay.
    assert idx.tolist() == [5]
    # Row 5 value: α·0 + (1-α)·tanh(1 + 1e_2) = 0.5 · tanh([1, 1, 0, 0])
    expected = 0.5 * torch.tanh(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(val[0], expected, atol=1e-6)


def test_backward_reaches_messages():
    """Gradient from overlay_val should flow back to input messages."""
    torch.manual_seed(5)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)                  # detached, as intended
    alpha = torch.full((N,), 0.3)
    messages = torch.randn(3, D_s, requires_grad=True)
    dests = torch.tensor([1, 5, 12], dtype=torch.int64)

    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    idx, val = overlay_lif_update(idx, val, s_base, messages, dests, alpha, N)

    loss = val.sum()
    loss.backward()

    assert messages.grad is not None
    # All three messages should receive non-zero gradient.
    for i in range(3):
        assert messages.grad[i].abs().sum().item() > 0, (
            f"message {i} got zero gradient"
        )


def test_backward_through_multi_step_revisit():
    """Revisiting a row means the new value depends on the previous write.
    Gradient should flow through the whole chain."""
    torch.manual_seed(6)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)
    alpha = torch.full((N,), 0.5)
    m1 = torch.randn(1, D_s, requires_grad=True)
    m2 = torch.randn(1, D_s, requires_grad=True)
    d = torch.tensor([3], dtype=torch.int64)

    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    idx, val = overlay_lif_update(idx, val, s_base, m1, d, alpha, N)
    idx, val = overlay_lif_update(idx, val, s_base, m2, d, alpha, N)

    loss = val.sum()
    loss.backward()

    assert m1.grad is not None and m1.grad.abs().sum().item() > 0, (
        "gradient did not flow back to first-step message"
    )
    assert m2.grad is not None and m2.grad.abs().sum().item() > 0, (
        "gradient did not flow back to second-step message"
    )


def test_commit_folds_overlay_into_base():
    torch.manual_seed(7)
    B, N, D_s = 2, 8, 4
    s_base = torch.randn(B, N, D_s)
    base_before = s_base.clone()
    alpha = torch.sigmoid(torch.randn(N))
    messages = torch.randn(3, D_s)
    dests = torch.tensor([2, 9, 14], dtype=torch.int64)

    idx, val = empty_overlay(D_s, s_base.device, s_base.dtype)
    idx, val = overlay_lif_update(idx, val, s_base, messages, dests, alpha, N)

    # Dense materialization before commit
    s_merged = _materialize_overlay(s_base, idx, val)

    # Commit in place
    commit_overlay_to_base(s_base, idx, val)

    # Base now equals the merged view; untouched rows unchanged.
    assert torch.equal(s_base, s_merged)
    untouched_mask = torch.ones(B * N, dtype=torch.bool)
    untouched_mask[dests] = False
    assert torch.equal(
        s_base.view(-1, D_s)[untouched_mask],
        base_before.view(-1, D_s)[untouched_mask],
    )
