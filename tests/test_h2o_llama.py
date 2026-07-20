"""Offline correctness tests for the Llama-3.1 H2O adaptation."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.memory.eval.h2o_llama import (
    H2OCache,
    H2OLlamaEngine,
    _chunk_causal_mask,
)


def _config(*, layers: int = 1) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=0,
    )


def _kv(start: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(start, start + count, dtype=torch.float32)
    keys = positions.view(1, 1, count, 1).expand(1, 2, count, 1).clone()
    return keys, keys + 100


def _uniform_probs(query_heads: int, query_tokens: int, key_tokens: int) -> torch.Tensor:
    return torch.full((1, query_heads, query_tokens, key_tokens), 1.0 / key_tokens)


def test_chunk_mask_allows_past_and_masks_future():
    mask = _chunk_causal_mask(past_length=3, query_length=3, dtype=torch.float32, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 3, 6)
    assert torch.equal(mask[0, 0, :, :3], torch.zeros(3, 3))
    assert mask[0, 0, 0, 4] < -1e20
    assert mask[0, 0, 1, 4] == 0


def test_chunk_mask_covers_retained_cache_plus_incoming_chunk():
    past = 2048
    mask = _chunk_causal_mask(
        past_length=past,
        query_length=128,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape[-1] == 2176
    assert mask[0, 0, 0, 2049] < -1e20
    assert mask[0, 0, 1, 2049] == 0


@pytest.mark.parametrize("head_mode,stored_heads", [("query_head", 4), ("kv_head", 2)])
def test_h2o_post_attention_prunes_to_capacity(head_mode, stored_heads):
    cache = H2OCache(_config(), heavy_size=2, recent_size=2, head_mode=head_mode)

    for start in (0, 2):
        keys, values = _kv(start, 2)
        cache.update(keys, values, 0)
        cache.record_attention(0, _uniform_probs(4, 2, start + 2))

    # Force deterministic heavy hitters. The next 2-token update is visible to
    # attention before the cache prunes back to four entries.
    forced = torch.tensor([10.0, 1.0, 9.0, 1.0]).view(1, 1, 4)
    cache.scores[0] = forced.expand(1, stored_heads, 4).clone()
    keys, values = _kv(4, 2)
    cached_keys, _ = cache.update(keys, values, 0)

    assert cached_keys.shape == (1, stored_heads, 6, 1)
    cache.record_attention(0, _uniform_probs(4, 2, 6))
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 2.0, 4.0, 5.0]
    assert cache.get_seq_length() == 4
    assert cache.max_unpruned_length == 6


def test_query_head_cache_matches_independent_h2o_transition_oracle():
    heavy_size, recent_size = 3, 4
    capacity = heavy_size + recent_size
    cache = H2OCache(
        _config(),
        heavy_size=heavy_size,
        recent_size=recent_size,
        head_mode="query_head",
    )
    generator = torch.Generator().manual_seed(123)
    oracle_ids = torch.empty((4, 0))
    oracle_scores = torch.empty((4, 0))
    next_position = 0

    for query_length in (3, 2, 4, 1, 3, 2):
        keys, values = _kv(next_position, query_length)
        cached_keys, _ = cache.update(keys, values, 0)
        incoming = torch.arange(next_position, next_position + query_length, dtype=torch.float32)
        oracle_ids = torch.cat((oracle_ids, incoming.view(1, -1).expand(4, -1)), dim=-1)
        assert torch.equal(cached_keys[0, :, :, 0], oracle_ids)

        raw = torch.rand((1, 4, query_length, oracle_ids.shape[-1]), generator=generator)
        probs = raw / raw.sum(dim=-1, keepdim=True)
        received = probs.sum(dim=2)[0]
        received[:, :oracle_scores.shape[-1]] += oracle_scores
        oracle_scores = received
        if oracle_ids.shape[-1] > capacity:
            eligible_end = oracle_ids.shape[-1] - recent_size
            heavy_idx = oracle_scores[:, :eligible_end].topk(heavy_size, dim=-1).indices.sort(dim=-1).values
            recent_idx = torch.arange(eligible_end, oracle_ids.shape[-1]).view(1, -1).expand(4, -1)
            keep_idx = torch.cat((heavy_idx, recent_idx), dim=-1)
            oracle_ids = oracle_ids.gather(1, keep_idx)
            oracle_scores = oracle_scores.gather(1, keep_idx)
        cache.record_attention(0, probs)
        assert torch.equal(cache.layers[0].keys[0, :, :, 0], oracle_ids)
        assert torch.allclose(cache.scores[0][0], oracle_scores)
        assert cache.get_seq_length() <= capacity
        next_position += query_length


def test_kv_head_mode_averages_scores_across_gqa_group():
    cache = H2OCache(_config(), heavy_size=2, recent_size=2, head_mode="kv_head")
    keys, values = _kv(0, 2)
    cache.update(keys, values, 0)
    probs = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.8, 0.2]],
            [[0.6, 0.4]],
        ]]
    )
    cache.record_attention(0, probs)
    expected = torch.tensor([[[0.5, 0.5], [0.7, 0.3]]])
    assert torch.allclose(cache.scores[0], expected)


def test_h2o_matches_full_cache_before_any_eviction():
    torch.manual_seed(7)
    base = LlamaForCausalLM(_config(layers=2)).eval()
    full = deepcopy(base)
    adapted = deepcopy(base)
    input_ids = torch.arange(12).remainder(128).view(1, -1)

    full_ids = full.generate(input_ids, max_new_tokens=4, do_sample=False)
    expected = full_ids[:, input_ids.shape[1]:]

    engine = H2OLlamaEngine(
        adapted,
        heavy_size=16,
        recent_size=16,
        prefill_chunk_size=4,
        head_mode="query_head",
    )
    actual = engine.generate(input_ids, max_new_tokens=4, eos_token_ids=[])

    assert torch.equal(actual.token_ids, expected)
    assert actual.diagnostics["max_unpruned_length"] < actual.diagnostics["capacity"]


@pytest.mark.parametrize("head_mode", ["query_head", "kv_head"])
def test_tiny_gqa_llama_smoke_evicts_during_prefill_and_decode(head_mode):
    torch.manual_seed(11)
    model = LlamaForCausalLM(_config(layers=2)).eval()
    engine = H2OLlamaEngine(
        model,
        heavy_size=4,
        recent_size=4,
        prefill_chunk_size=4,
        head_mode=head_mode,
    )
    result = engine.generate(torch.arange(24).remainder(128).view(1, -1), max_new_tokens=3, eos_token_ids=[])

    assert result.token_ids.shape == (1, 3)
    assert result.finish_reason == "length"
    assert result.diagnostics["max_unpruned_length"] == 12
    assert result.diagnostics["max_retained_length"] == 8
    assert result.diagnostics["final_min_layer_length"] == 8
    assert result.diagnostics["final_max_layer_length"] == 8
    # The final sampled token is not fed back because no next-token logits are needed.
    assert result.diagnostics["tokens_seen"] == 24 + 3 - 1


def test_position_rolling_accepts_stream_longer_than_model_window():
    torch.manual_seed(13)
    model = LlamaForCausalLM(_config(layers=2)).eval()
    engine = H2OLlamaEngine(
        model,
        heavy_size=4,
        recent_size=4,
        prefill_chunk_size=4,
        head_mode="query_head",
    )
    input_ids = torch.arange(600).remainder(128).view(1, -1)
    result = engine.generate(input_ids, max_new_tokens=2, eos_token_ids=[])

    assert input_ids.shape[1] > model.config.max_position_embeddings
    assert result.token_ids.shape == (1, 2)
    assert result.diagnostics["position_mode"] == "rolling"
    assert result.diagnostics["max_unpruned_length"] == 12
    assert result.diagnostics["final_max_layer_length"] == 8


@pytest.mark.parametrize("head_mode", ["query_head", "kv_head"])
def test_snapshot_fork_exactly_matches_independent_replay(head_mode):
    torch.manual_seed(17)
    base = LlamaForCausalLM(_config(layers=2)).eval()
    direct_engine = H2OLlamaEngine(
        deepcopy(base),
        heavy_size=4,
        recent_size=4,
        prefill_chunk_size=4,
        head_mode=head_mode,
    )
    reused_engine = H2OLlamaEngine(
        deepcopy(base),
        heavy_size=4,
        recent_size=4,
        prefill_chunk_size=4,
        head_mode=head_mode,
    )
    prompt = torch.arange(40).remainder(128).view(1, -1)
    prefix_length = 24

    expected = direct_engine.generate(prompt, max_new_tokens=4, eos_token_ids=[])
    snapshot = reused_engine.prefill(prompt[:, :prefix_length])
    frozen_keys = [layer.keys.clone() for layer in snapshot.cache.layers]
    frozen_scores = [score.clone() for score in snapshot.cache.scores]
    actual = reused_engine.generate_from_snapshot(
        snapshot,
        prompt[:, prefix_length:],
        max_new_tokens=4,
        eos_token_ids=[],
        total_prompt_tokens=prompt.shape[1],
    )

    assert torch.equal(actual.token_ids, expected.token_ids)
    assert actual.diagnostics["shared_prefix_tokens"] == prefix_length
    for layer, keys in zip(snapshot.cache.layers, frozen_keys, strict=True):
        assert torch.equal(layer.keys, keys)
    for score, frozen in zip(snapshot.cache.scores, frozen_scores, strict=True):
        assert torch.equal(score, frozen)
