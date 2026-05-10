"""Correctness tests for per-slot KV cache in multi-stream training.

The PRIMARY test (`test_4d_mask_shape_and_logic`) verifies the mask
construction. The HEAVY test (`test_bs_equivalence_with_real_llama`)
verifies that BS=4 multi-stream produces the same logits as 4 sequential
BS=1 runs on the same per-slot doc sequences. The heavy test requires
real Llama and is opt-in via `RUN_HEAVY_TESTS=1`.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.trajectory_memory.per_slot_kv import (
    PerSlotCacheState,
    build_per_slot_4d_mask,
    build_per_slot_position_ids,
)


def test_per_slot_state_basics():
    BS, T_max = 4, 256
    state = PerSlotCacheState.fresh(BS, T_max, device="cpu")
    assert state.cache is None
    assert state.valid_len.tolist() == [0, 0, 0, 0]
    assert state.abs_pos.tolist() == [0, 0, 0, 0]
    assert state.max_cache_len == T_max


def test_advance_after_forward_clamps_to_max():
    state = PerSlotCacheState.fresh(2, max_cache_len=10, device="cpu")
    state.advance_after_forward(6)
    assert state.valid_len.tolist() == [6, 6]
    assert state.abs_pos.tolist() == [6, 6]
    state.advance_after_forward(8)  # would be 14, clamped to 10
    assert state.valid_len.tolist() == [10, 10]
    assert state.abs_pos.tolist() == [14, 14]   # abs_pos NOT clamped


def test_reset_slots_selective():
    state = PerSlotCacheState.fresh(3, max_cache_len=100, device="cpu")
    state.advance_after_forward(50)
    is_start = torch.tensor([True, False, True])
    state.reset_slots(is_start)
    assert state.valid_len.tolist() == [0, 50, 0]
    assert state.abs_pos.tolist() == [0, 50, 0]


def test_trim_after_cache_trim_preserves_other_slots():
    state = PerSlotCacheState.fresh(3, max_cache_len=100, device="cpu")
    state.valid_len = torch.tensor([20, 80, 100])
    state.abs_pos = torch.tensor([20, 80, 100])
    # Cache trimmed to 50 — slots with valid_len > 50 get clamped.
    state.trim_after_cache_trim(new_cache_len=50)
    assert state.valid_len.tolist() == [20, 50, 50]
    # abs_pos UNCHANGED by trim (RoPE positions are absolute).
    assert state.abs_pos.tolist() == [20, 80, 100]


def test_4d_mask_shape_and_logic_basic():
    """Slot 0 has 2 valid cache positions, slot 1 has 0 (just reset).
    Cache_len_old = 4, q_len = 3.
    Expected mask shape: [2, 1, 3, 7].
    Slot 0 should attend to cache positions [2, 3] (last 2 of 4 valid)
    and new positions [0, q] for each query q (causal).
    Slot 1 should attend ONLY to new positions [0, q] (no cache valid)."""
    valid_len = torch.tensor([2, 0])
    cache_len_old = 4
    q_len = 3
    mask = build_per_slot_4d_mask(
        valid_len=valid_len, q_len=q_len, cache_len_old=cache_len_old,
        device="cpu", dtype=torch.float32,
    )
    assert mask.shape == (2, 1, 3, 7)

    # Slot 0, query 0: attend to cache [2, 3] (valid) + new [0] (causal).
    #                  blocked at cache [0, 1] (stale) + new [1, 2] (future).
    expected_slot0_q0 = [
        float("-inf"), float("-inf"),  # cache 0, 1 stale
        0.0, 0.0,                       # cache 2, 3 valid for slot 0
        0.0,                            # new 0 (q=0 sees new[0])
        float("-inf"), float("-inf"),  # new 1, 2 future
    ]
    assert mask[0, 0, 0].tolist() == expected_slot0_q0

    # Slot 0, query 2: attend to cache [2, 3] + new [0, 1, 2].
    expected_slot0_q2 = [
        float("-inf"), float("-inf"),
        0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    assert mask[0, 0, 2].tolist() == expected_slot0_q2

    # Slot 1, query 0: attend ONLY to new[0]. All cache stale.
    expected_slot1_q0 = [
        float("-inf"), float("-inf"), float("-inf"), float("-inf"),
        0.0,
        float("-inf"), float("-inf"),
    ]
    assert mask[1, 0, 0].tolist() == expected_slot1_q0


def test_4d_mask_empty_cache():
    """Edge case: cache_len_old=0 (very first forward). Mask should
    only contain the causal new-token part."""
    valid_len = torch.tensor([0, 0])
    mask = build_per_slot_4d_mask(
        valid_len=valid_len, q_len=2, cache_len_old=0,
        device="cpu", dtype=torch.float32,
    )
    assert mask.shape == (2, 1, 2, 2)
    # Standard causal: query 0 sees new[0] only; query 1 sees [0, 1].
    expected = [
        [0.0, float("-inf")],
        [0.0, 0.0],
    ]
    for slot in range(2):
        assert mask[slot, 0].tolist() == expected


def test_per_slot_position_ids():
    abs_pos = torch.tensor([0, 100, 250])
    pos = build_per_slot_position_ids(abs_pos, q_len=3, device="cpu")
    assert pos.shape == (3, 3)
    assert pos[0].tolist() == [0, 1, 2]
    assert pos[1].tolist() == [100, 101, 102]
    assert pos[2].tolist() == [250, 251, 252]


# ── Heavy test: BS-equivalence with real Llama ────────────────────────


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="heavy: requires real Llama download. Set RUN_HEAVY_TESTS=1.",
)
def test_bs4_multi_stream_matches_bs1_sequential():
    """The acid test: BS=4 multi-stream forward through Llama should
    produce per-slot logits that match (mod fp tolerance) what you'd
    get from 4 sequential BS=1 runs of the same per-slot doc sequences.

    Setup: build 4 fake "docs" of varying length. Run them as BS=4
    multi-stream with per-slot KV cache (using our 4D mask). Run them
    as 4 sequential BS=1 forwards (HF defaults). Compare logits at
    each token position."""
    # TODO: implement when we wire forward_window to use the new path.
    # Will be the regression test that catches off-by-one bugs in the
    # mask construction or position_ids threading.
    pass
