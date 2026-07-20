"""Infinite-streaming H2O inference for modern Hugging Face Llama models.

This adapts the official FMInference/H2O ``H2OLlamaAttention_streaming``
variant to Llama 3.1 and grouped-query attention (GQA).  The important detail
is position rolling: retained keys are stored *before* RoPE and re-rotated at
compact cache positions on every forward pass.  The effective position range
is therefore bounded by retained KV plus the incoming chunk even when the
logical stream is much longer than the model's pretrained context window.

The implementation supports two GQA policies:

``query_head``
    Repeat K/V into query-head space and make an independent H2O decision per
    query head.  This is closest to the original MHA policy, at 4x KV memory
    for Llama-3.1-8B.
``kv_head``
    Average the attention scores of query heads sharing a KV head and retain a
    common subset.  This preserves native GQA storage but is an explicit
    efficiency adaptation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Llama RoPE to ``[batch, heads, tokens, dim]`` states."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class H2OCache(DynamicCache):
    """Raw-K/V cache with cumulative-attention H2O eviction and cheap forks."""

    def __init__(
        self,
        config,
        *,
        heavy_size: int,
        recent_size: int,
        rotary_emb=None,
        head_mode: str = "query_head",
    ) -> None:
        if heavy_size <= 0 or recent_size <= 0:
            raise ValueError("heavy_size and recent_size must both be positive")
        if head_mode not in {"query_head", "kv_head"}:
            raise ValueError(f"unsupported H2O head_mode: {head_mode!r}")

        super().__init__(config=config)
        text_config = config.get_text_config(decoder=True)
        self.config = config
        self.rotary_emb = rotary_emb
        self.heavy_size = heavy_size
        self.recent_size = recent_size
        self.capacity = heavy_size + recent_size
        self.head_mode = head_mode
        self.num_attention_heads = text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scores: list[torch.Tensor | None] = [None] * len(self.layers)
        self.tokens_seen = [0] * len(self.layers)
        self.max_unpruned_length = 0
        self.max_retained_length = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Append unrotated K/V; pruning happens after the new attention is observed."""
        if key_states.shape[0] != 1:
            raise ValueError("H2O currently requires batch size 1")
        num_new_tokens = key_states.shape[-2]
        if num_new_tokens > self.recent_size:
            raise ValueError(
                f"an H2O update ({num_new_tokens} tokens) cannot exceed recent_size ({self.recent_size})"
            )
        if self.head_mode == "query_head" and self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        self.tokens_seen[layer_idx] += num_new_tokens
        self.max_unpruned_length = max(self.max_unpruned_length, keys.shape[-2])
        return keys, values

    def record_attention(self, layer_idx: int, attention_probs: torch.Tensor) -> None:
        """Accumulate received attention and prune this layer to the fixed budget."""
        if attention_probs.shape[0] != 1:
            raise ValueError("H2O currently requires batch size 1")

        layer = self.layers[layer_idx]
        seq_len = layer.keys.shape[-2]
        if attention_probs.shape[-1] != seq_len:
            raise RuntimeError(
                f"layer {layer_idx}: attention/cache length mismatch "
                f"({attention_probs.shape[-1]} != {seq_len})"
            )

        detached = attention_probs.detach()
        if self.head_mode == "kv_head" and self.num_key_value_groups > 1:
            bsz, _, q_len, kv_len = detached.shape
            grouped = detached.view(
                bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, kv_len
            )
            scores = grouped.sum(dim=(2, 3), dtype=torch.float32) / self.num_key_value_groups
        else:
            scores = detached.sum(dim=2, dtype=torch.float32)

        previous = self.scores[layer_idx]
        if previous is not None:
            old_len = previous.shape[-1]
            if old_len > scores.shape[-1]:
                raise RuntimeError(f"layer {layer_idx}: H2O score history is longer than the cache")
            scores[..., :old_len] += previous
        self.scores[layer_idx] = scores

        if seq_len > self.capacity:
            self._prune_layer(layer_idx)
        self.max_retained_length = max(self.max_retained_length, self.layers[layer_idx].keys.shape[-2])

    def _prune_layer(self, layer_idx: int) -> None:
        layer = self.layers[layer_idx]
        seq_len = layer.keys.shape[-2]
        old_end = seq_len - self.recent_size
        if old_end < self.heavy_size:
            raise RuntimeError(
                f"cannot retain {self.heavy_size} heavy tokens from only {old_end} eligible positions"
            )
        self._select_indices(layer_idx, old_end=old_end, recent_count=self.recent_size)

    def _select_indices(self, layer_idx: int, *, old_end: int, recent_count: int) -> None:
        layer = self.layers[layer_idx]
        scores = self.scores[layer_idx]
        assert scores is not None
        seq_len = layer.keys.shape[-2]
        if old_end < self.heavy_size:
            raise RuntimeError(
                f"cannot retain {self.heavy_size} heavy tokens from only {old_end} eligible positions"
            )

        heavy_idx = scores[..., :old_end].topk(self.heavy_size, dim=-1).indices
        heavy_idx = heavy_idx.sort(dim=-1).values
        recent_start = seq_len - recent_count
        recent_idx = torch.arange(recent_start, seq_len, device=heavy_idx.device)
        recent_idx = recent_idx.view(1, 1, -1).expand(heavy_idx.shape[0], heavy_idx.shape[1], -1)
        keep_idx = torch.cat((heavy_idx, recent_idx), dim=-1)

        gather_idx = keep_idx.unsqueeze(-1).expand(*keep_idx.shape, layer.keys.shape[-1])
        layer.keys = layer.keys.gather(dim=2, index=gather_idx)
        layer.values = layer.values.gather(dim=2, index=gather_idx)
        self.scores[layer_idx] = scores.gather(dim=2, index=keep_idx)

    def fork(self) -> "H2OCache":
        """Clone retained memory and scores so one snapshot can answer many queries."""
        clone = H2OCache(
            self.config,
            heavy_size=self.heavy_size,
            recent_size=self.recent_size,
            rotary_emb=self.rotary_emb,
            head_mode=self.head_mode,
        )
        for source, target in zip(self.layers, clone.layers, strict=True):
            if not source.is_initialized:
                continue
            target.lazy_initialization(source.keys, source.values)
            target.keys = source.keys.clone()
            target.values = source.values.clone()
        clone.scores = [None if score is None else score.clone() for score in self.scores]
        clone.tokens_seen = self.tokens_seen.copy()
        clone.max_unpruned_length = self.max_unpruned_length
        clone.max_retained_length = self.max_retained_length
        return clone

    def diagnostics(self) -> dict[str, int | str]:
        lengths = [layer.get_seq_length() for layer in self.layers]
        return {
            "capacity": self.capacity,
            "heavy_size": self.heavy_size,
            "recent_size": self.recent_size,
            "head_mode": self.head_mode,
            "position_mode": "rolling",
            "max_unpruned_length": self.max_unpruned_length,
            "max_retained_length": self.max_retained_length,
            "final_min_layer_length": min(lengths, default=0),
            "final_max_layer_length": max(lengths, default=0),
            "tokens_seen": max(self.tokens_seen, default=0),
        }


def h2o_streaming_attention_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Llama attention with raw-key storage and official-style position rolling."""
    del position_embeddings, cache_position, kwargs
    if not isinstance(past_key_values, H2OCache):
        raise RuntimeError("streaming H2O attention requires an H2OCache")
    if past_key_values.rotary_emb is None:
        raise RuntimeError("streaming H2O cache has no rotary embedding")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    raw_key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    past_length = past_key_values.layers[module.layer_idx].get_seq_length()
    query_length = query.shape[-2]
    query_positions = torch.arange(
        past_length, past_length + query_length, device=query.device, dtype=torch.long
    ).unsqueeze(0)
    query_cos, query_sin = past_key_values.rotary_emb(value, query_positions)
    query = _apply_rope(query, query_cos, query_sin)

    raw_keys, values = past_key_values.update(raw_key, value, module.layer_idx)
    key_positions = torch.arange(raw_keys.shape[-2], device=query.device, dtype=torch.long).unsqueeze(0)
    key_cos, key_sin = past_key_values.rotary_emb(raw_keys, key_positions)
    keys = _apply_rope(raw_keys, key_cos, key_sin)

    if keys.shape[1] != query.shape[1]:
        keys = keys.repeat_interleave(module.num_key_value_groups, dim=1)
        values_for_attention = values.repeat_interleave(module.num_key_value_groups, dim=1)
    else:
        values_for_attention = values

    logits = torch.matmul(query, keys.transpose(2, 3)) * module.scaling
    if attention_mask is not None:
        logits = logits + attention_mask[..., :query_length, :keys.shape[-2]]
    probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    if module.training and module.attention_dropout:
        probs = F.dropout(probs, p=module.attention_dropout)

    output = torch.matmul(probs, values_for_attention).transpose(1, 2).contiguous()
    past_key_values.record_attention(module.layer_idx, probs)
    output = module.o_proj(output.reshape(*input_shape, -1).contiguous())
    return output, None


def _chunk_causal_mask(
    *,
    past_length: int,
    query_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Mask future tokens inside a chunk while allowing every retained past key."""
    mask = torch.zeros((1, 1, query_length, past_length + query_length), dtype=dtype, device=device)
    current = torch.full(
        (query_length, query_length), torch.finfo(dtype).min, dtype=dtype, device=device
    ).triu(diagonal=1)
    mask[..., past_length:] = current
    return mask


def _as_token_set(token_ids: int | Iterable[int] | None) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}


@dataclass(frozen=True)
class H2OSnapshot:
    """Read-only retained prefix state; ``generate_from_snapshot`` forks it."""

    cache: H2OCache
    prompt_tokens: int
    prefill_seconds: float


@dataclass(frozen=True)
class H2OGeneration:
    token_ids: torch.Tensor
    finish_reason: str
    diagnostics: dict[str, int | float | str]


class H2OLlamaEngine:
    """Greedy Llama inference using H2O's infinite position-rolling cache."""

    def __init__(
        self,
        model,
        *,
        heavy_size: int = 1024,
        recent_size: int = 1024,
        prefill_chunk_size: int = 128,
        head_mode: str = "query_head",
    ) -> None:
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if prefill_chunk_size > recent_size:
            raise ValueError("prefill_chunk_size must not exceed recent_size")
        self.model = model
        self.heavy_size = heavy_size
        self.recent_size = recent_size
        self.prefill_chunk_size = prefill_chunk_size
        self.head_mode = head_mode
        for layer in self.model.model.layers[: self.model.config.num_hidden_layers]:
            layer.self_attn.forward = MethodType(h2o_streaming_attention_forward, layer.self_attn)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _new_cache(self) -> H2OCache:
        return H2OCache(
            self.model.config,
            heavy_size=self.heavy_size,
            recent_size=self.recent_size,
            rotary_emb=self.model.model.rotary_emb,
            head_mode=self.head_mode,
        )

    @staticmethod
    def _validate_ids(input_ids: torch.Tensor, *, label: str) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"{label} expects input_ids with shape [1, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError(f"{label} requires at least one token")

    def _prefill(self, cache: H2OCache, input_ids: torch.Tensor) -> tuple[torch.Tensor, float]:
        input_ids = input_ids.to(self.device)
        started_at = time.perf_counter()
        next_logits = None
        with torch.inference_mode():
            for start in range(0, input_ids.shape[1], self.prefill_chunk_size):
                end = min(start + self.prefill_chunk_size, input_ids.shape[1])
                chunk = input_ids[:, start:end]
                past_length = cache.get_seq_length()
                query_length = end - start
                positions = torch.arange(
                    past_length, past_length + query_length, device=self.device, dtype=torch.long
                )
                output = self.model(
                    input_ids=chunk,
                    attention_mask=_chunk_causal_mask(
                        past_length=past_length,
                        query_length=query_length,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                next_logits = output.logits[:, -1, :]
        assert next_logits is not None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return next_logits, time.perf_counter() - started_at

    def prefill(self, input_ids: torch.Tensor) -> H2OSnapshot:
        """Encode a shared prefix once for later query-specific snapshot forks."""
        self._validate_ids(input_ids, label="H2O prefill")
        cache = self._new_cache()
        _, seconds = self._prefill(cache, input_ids)
        return H2OSnapshot(cache=cache, prompt_tokens=input_ids.shape[1], prefill_seconds=seconds)

    def generate_from_snapshot(
        self,
        snapshot: H2OSnapshot,
        suffix_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: int | Iterable[int] | None = None,
        total_prompt_tokens: int | None = None,
    ) -> H2OGeneration:
        """Fork a prefix snapshot, consume a query suffix, and decode without mutating the snapshot."""
        self._validate_ids(suffix_ids, label="H2O snapshot generation")
        fork_started_at = time.perf_counter()
        cache = snapshot.cache.fork()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        fork_seconds = time.perf_counter() - fork_started_at
        result = self._generate(
            suffix_ids,
            cache=cache,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            prompt_tokens=total_prompt_tokens or snapshot.prompt_tokens + suffix_ids.shape[1],
            shared_prefix_tokens=snapshot.prompt_tokens,
        )
        result.diagnostics["snapshot_fork_seconds"] = fork_seconds
        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: int | Iterable[int] | None = None,
    ) -> H2OGeneration:
        self._validate_ids(input_ids, label="H2O generation")
        return self._generate(
            input_ids,
            cache=self._new_cache(),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            prompt_tokens=input_ids.shape[1],
            shared_prefix_tokens=0,
        )

    def _generate(
        self,
        input_ids: torch.Tensor,
        *,
        cache: H2OCache,
        max_new_tokens: int,
        eos_token_ids: int | Iterable[int] | None,
        prompt_tokens: int,
        shared_prefix_tokens: int,
    ) -> H2OGeneration:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        eos_ids = _as_token_set(eos_token_ids)
        generated: list[torch.Tensor] = []
        finish_reason = "length"

        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        next_logits, prefill_seconds = self._prefill(cache, input_ids)
        decode_started_at = time.perf_counter()

        with torch.inference_mode():
            for step in range(max_new_tokens):
                next_token = next_logits.argmax(dim=-1)
                generated.append(next_token)
                if int(next_token.item()) in eos_ids:
                    finish_reason = "stop"
                    break
                if step + 1 == max_new_tokens:
                    break

                past_length = cache.get_seq_length()
                positions = torch.tensor([past_length], device=self.device)
                output = self.model(
                    input_ids=next_token.view(1, 1),
                    attention_mask=_chunk_causal_mask(
                        past_length=past_length,
                        query_length=1,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                next_logits = output.logits[:, -1, :]

        if cuda:
            torch.cuda.synchronize(self.device)
        decode_seconds = time.perf_counter() - decode_started_at
        diagnostics: dict[str, int | float | str] = cache.diagnostics()
        diagnostics.update({
            "prefill_chunk_size": self.prefill_chunk_size,
            "prompt_tokens": prompt_tokens,
            "shared_prefix_tokens": shared_prefix_tokens,
            "query_suffix_tokens": input_ids.shape[1],
            "generated_tokens": len(generated),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prefill_tokens_per_second": input_ids.shape[1] / prefill_seconds,
            "decode_tokens_per_second": len(generated) / decode_seconds,
        })
        if cuda:
            diagnostics["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
        tokens = torch.stack(generated, dim=1).detach().cpu()
        return H2OGeneration(tokens, finish_reason, diagnostics)


def load_llama_engine(
    model_name: str,
    *,
    heavy_size: int,
    recent_size: int,
    prefill_chunk_size: int,
    head_mode: str,
    device_map: str = "cuda",
    revision: str | None = None,
):
    """Load a BF16 Llama checkpoint and wrap it in infinite-streaming H2O."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device_map,
    )
    model.eval()
    engine = H2OLlamaEngine(
        model,
        heavy_size=heavy_size,
        recent_size=recent_size,
        prefill_chunk_size=prefill_chunk_size,
        head_mode=head_mode,
    )
    return engine, tokenizer
