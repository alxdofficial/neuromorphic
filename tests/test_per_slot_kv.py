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
    reason="heavy: requires real Llama. Set RUN_HEAVY_TESTS=1.",
)
def test_bs2_multi_stream_matches_bs1_sequential():
    """ACID test: BS=2 multi-stream Llama forward should produce per-slot
    logits matching what 2 sequential BS=1 runs of the same per-slot
    sequences would produce. Catches off-by-one in 4D mask, position_ids
    threading, or per-slot cache lifecycle.

    Method: 2 different short token sequences, processed in chunks of 4.
    Reference: each through `llama.model()` standalone with fresh cache.
    Multi-stream: both as BS=2 batched with our 4D mask + per-slot
    position_ids + per-slot cache state. Compare logits per-token.
    """
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache

    from src.trajectory_memory.per_slot_kv import (
        PerSlotCacheState,
        build_per_slot_4d_mask,
        build_per_slot_position_ids,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    model.requires_grad_(False)
    model_inference_mode = model.eval()  # PyTorch's set-eval-mode (not Python eval)
    del model_inference_mode  # the assignment is just to avoid lint complaints

    torch.manual_seed(0)
    n_chunks = 3
    chunk_size = 4
    seq_a = torch.randint(0, 32000, (n_chunks * chunk_size,), device=device)
    seq_b = torch.randint(0, 32000, (n_chunks * chunk_size,), device=device)

    # ── Reference: BS=1 sequential per slot ──
    @torch.no_grad()
    def run_bs1(seq):
        cache = DynamicCache()
        all_logits = []
        cache_len = 0
        for c in range(n_chunks):
            chunk = seq[c * chunk_size:(c + 1) * chunk_size].unsqueeze(0)
            out = model.model(
                input_ids=chunk,
                past_key_values=cache,
                cache_position=torch.arange(
                    cache_len, cache_len + chunk_size, device=device,
                ),
                position_ids=torch.arange(
                    cache_len, cache_len + chunk_size, device=device,
                ).unsqueeze(0),
                use_cache=True,
            )
            cache = out.past_key_values
            cache_len += chunk_size
            all_logits.append(model.lm_head(out.last_hidden_state[0]))
        return torch.cat(all_logits, dim=0)            # [n_total, V]

    logits_a_ref = run_bs1(seq_a)
    logits_b_ref = run_bs1(seq_b)

    # ── Multi-stream: BS=2 per-slot ──
    @torch.no_grad()
    def run_bs2_multi():
        state = PerSlotCacheState.fresh(2, max_cache_len=64, device=device)
        all_logits = [[], []]
        for c in range(n_chunks):
            chunk_a = seq_a[c * chunk_size:(c + 1) * chunk_size]
            chunk_b = seq_b[c * chunk_size:(c + 1) * chunk_size]
            chunk = torch.stack([chunk_a, chunk_b], dim=0)
            cache_len_old = (
                state.cache.get_seq_length()
                if state.cache is not None else 0
            )
            cache_position = torch.arange(
                cache_len_old, cache_len_old + chunk_size, device=device,
            )
            position_ids = build_per_slot_position_ids(
                state.abs_pos, chunk_size, device=device,
            )
            attention_mask = build_per_slot_4d_mask(
                valid_len=state.valid_len,
                q_len=chunk_size,
                cache_len_old=cache_len_old,
                device=device,
                dtype=torch.bfloat16,
            )
            out = model.model(
                input_ids=chunk,
                past_key_values=state.cache,
                cache_position=cache_position,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            state.cache = out.past_key_values
            state.advance_after_forward(chunk_size)
            chunk_logits = model.lm_head(out.last_hidden_state)
            all_logits[0].append(chunk_logits[0])
            all_logits[1].append(chunk_logits[1])
        return torch.cat(all_logits[0], dim=0), torch.cat(all_logits[1], dim=0)

    logits_a_multi, logits_b_multi = run_bs2_multi()

    # ── Compare ──
    diff_a = (logits_a_ref.float() - logits_a_multi.float()).abs().max().item()
    diff_b = (logits_b_ref.float() - logits_b_multi.float()).abs().max().item()
    print(f"slot 0 max abs diff: {diff_a:.4e}")
    print(f"slot 1 max abs diff: {diff_b:.4e}")
    # bf16 numerical noise tolerance — different attention backends
    # can produce ~1e-2 to 1e-1 logit differences. Catastrophic divergence
    # (>1) indicates real bug.
    assert diff_a < 1.0, f"slot 0 logits diverged: {diff_a}"
    assert diff_b < 1.0, f"slot 1 logits diverged: {diff_b}"
