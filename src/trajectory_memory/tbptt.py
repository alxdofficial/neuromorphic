"""TBPTT scaffolding — chain D forward_window calls with cross-window
gradient flow.

Linear-in-D implementation: every window's autograd graph stays alive
until backward fires at the end of the chunk. With per-block Llama
checkpointing (already on inside the LM forward), memory cost is roughly
1 window of activations + a slim per-window write graph skeleton.

Constant-in-D (custom autograd streaming) is a v2 optimization — see
plan §4.3.

Usage:
    chunks = TBPTTChunker(cfg).split(input_ids)   # [num_chunks, D, T_window]
    prev_states, prev_hiddens, prev_lm_ctx = None, None, None
    for chunk in chunks:
        result = run_chunk(
            integrated_lm, chunk, prev_states, prev_hiddens, prev_lm_ctx,
        )
        loss = result["aggregate_loss"]
        loss.backward()
        prev_states = result["final_states"].detach()              # cut grad
        prev_hiddens = result["final_hiddens"].detach()
        prev_lm_ctx = result["final_lm_context"]                   # already detached
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM


class TBPTTChunker:
    """Splits a long sequence into D-window chunks for TBPTT.

    Given input_ids of shape [BS, total_T], where total_T = num_chunks *
    D * T_window, return iterable of [BS, D, T_window] chunks.

    For training, each chunk's backward fires before the next chunk's
    forward, with `final_states` and `final_hiddens` detached at the
    boundary.
    """

    def __init__(self, cfg: TrajMemConfig):
        self.cfg = cfg

    def split(self, input_ids: Tensor) -> list[Tensor]:
        cfg = self.cfg
        BS, T = input_ids.shape
        chunk_size = cfg.D * cfg.T_window
        if T % chunk_size != 0:
            # Drop the trailing partial chunk; data loader is expected to pad.
            usable = (T // chunk_size) * chunk_size
            input_ids = input_ids[:, :usable]
            T = usable
        num_chunks = T // chunk_size
        windows_per_chunk = cfg.D
        return [
            input_ids[
                :,
                c * chunk_size : (c + 1) * chunk_size,
            ].view(BS, windows_per_chunk, cfg.T_window)
            for c in range(num_chunks)
        ]


def _trim_kv_cache(cache, max_len: int):
    """Sliding-window trim a HF DynamicCache to keep the LAST max_len tokens.

    Mutates `cache` in place. Returns `cache` for chaining.

    HF transformers 5.x exposes per-layer cache as `cache.layers[i].keys`
    and `.values`. Cache.crop() truncates from the END (keeps first N) —
    we need to keep the LAST N (most recent), so we slice manually.

    Trimming is a tensor slice — autograd through earlier window's backward
    still works on the surviving slice. Dropped entries are released from
    the autograd graph.
    """
    if cache is None:
        return None
    cur_len = cache.get_seq_length()
    if cur_len <= max_len:
        return cache
    drop = cur_len - max_len
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        layer.keys = layer.keys[..., drop:, :]
        layer.values = layer.values[..., drop:, :]
    return cache


def _detach_kv_cache(cache):
    """Detach all tensors in cache so a new TBPTT chunk starts with no
    autograd history through cached KVs (mirrors `prev_states.detach()`)."""
    if cache is None:
        return None
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        layer.keys = layer.keys.detach()
        layer.values = layer.values.detach()
    return cache


def run_chunk(
    model: IntegratedLM,
    windows: Tensor,
    prev_states: Tensor,
    prev_window_hiddens: Tensor | None,
    prev_lm_context: Tensor | None,
    *,
    target_mask: Tensor | None = None,
    hard_routing: bool = True,
    use_kv_cache: bool = False,
    past_key_values: object | None = None,
    cache_abs_pos: int = 0,
) -> dict:
    """Run D consecutive windows with autograd kept alive across the chunk.

    Two execution modes:

    - **Rolling-buffer mode** (`use_kv_cache=False`, default): maintains
      a rolling LM-context buffer of up to `cfg.effective_lm_context`
      tokens. At each window, Llama re-encodes the rolling buffer +
      the new window's tokens.

    - **KV-cache mode** (`use_kv_cache=True`): each window's Llama forward
      only encodes the new T_window tokens against `past_key_values`
      (HF DynamicCache). The cache is trimmed to `cfg.effective_lm_context`
      tokens (sliding window). Across chunks the cache is detached but
      can be carried (see `_detach_kv_cache` and the trainer wiring).
      ~30-50% faster on Phase 1 — the rolling-buffer re-encode was ~70%
      of the T1 vs V1.B gap (see docs/profile_analysis.md).

    Args:
        model:               IntegratedLM
        windows:             [BS, D, T_window] token IDs
        prev_states:         [BS, N, D_concept] state at start of chunk
        prev_window_hiddens: [BS, T_window, d_lm] from window before this
                             chunk, or None if chunk starts a new sequence.
        prev_lm_context:     Rolling-buffer mode: [BS, L] tokens from prior
                             chunk to seed the buffer. KV-cache mode: ignored
                             (use `past_key_values` instead).
        target_mask:         [BS, D, T_window] or None — per-window mask
                             over the CURRENT window's tokens for surprise.
        hard_routing:        Gumbel-STE if True
        use_kv_cache:        Switch to KV-cache mode.
        past_key_values:     KV-cache mode: HF DynamicCache from previous
                             chunk (detached) or None. Updated in-place;
                             `final_past_key_values` returned for next chunk.

    Returns:
        dict with keys:
            window_outputs:        list of D forward_window outputs
            aggregate_loss:        scalar loss summed over windows (NTP CE)
            final_states:          [BS, N, D_concept] state after last window
            final_hiddens:         [BS, T_window, d_lm] hiddens of last window
            final_lm_context:      Rolling-buffer mode: [BS, L] for next chunk.
                                   KV-cache mode: empty (not used).
            final_past_key_values: KV-cache mode only — updated cache.
            surprise_history:      [BS, D] surprise per window
    """
    BS, D, T = windows.shape
    assert D == model.cfg.D, f"chunk has D={D}, expected {model.cfg.D}"
    cfg = model.cfg
    cap = cfg.effective_lm_context

    states = prev_states
    cur_prev_hiddens = prev_window_hiddens
    cache = past_key_values if use_kv_cache else None
    # Absolute position counter — tracks position of next token to encode.
    # Required for RoPE correctness: when cache is sliding-window trimmed,
    # `cache.get_seq_length()` no longer reflects absolute position. We pass
    # `cache_position` explicitly to llama.model() based on this counter so
    # new tokens get RoPE rotations matching their TRUE positions, allowing
    # the cached KVs (which baked their original-position rotations in) to
    # combine correctly in attention.
    abs_pos = cache_abs_pos

    # Rolling LM-context buffer (rolling-buffer mode only; ignored in cache mode).
    if use_kv_cache:
        lm_buffer = torch.empty(BS, 0, dtype=windows.dtype, device=windows.device)
    elif prev_lm_context is None:
        lm_buffer = torch.empty(BS, 0, dtype=windows.dtype, device=windows.device)
    else:
        lm_buffer = prev_lm_context.detach()                       # ensure no grad

    outputs: list[dict] = []
    surprises: list[Tensor] = []
    # N6 — token-weighted aggregation: accumulate sum + count across all
    # windows (instead of summing per-window means). Sparse windows
    # contribute proportionally to their real-token count, not equally
    # to full windows.
    chunk_ce_sum: Tensor | None = None
    chunk_ce_count: Tensor | None = None
    # Fallback path for test mode (no real Llama → no surprise_sum/count
    # in `out`): aggregate per-window mean × T_window like before.
    losses_fallback: list[Tensor] = []

    for d in range(D):
        win_input = windows[:, d, :]                               # [BS, T_window]
        win_mask = target_mask[:, d, :] if target_mask is not None else None

        if use_kv_cache:
            # KV-cache mode: pass only the new window's T_window tokens.
            # last_prev_logit_hidden gives surprise CE the predecessor for
            # target 0 — uses prev_window_hiddens last position.
            last_prev_logit_hidden = (
                cur_prev_hiddens[:, -1:, :] if cur_prev_hiddens is not None else None
            )
            out = model.forward_window(
                lm_input_ids=win_input,
                prev_window_hiddens=cur_prev_hiddens,
                prev_states=states,
                target_mask=win_mask,
                hard_routing=hard_routing,
                past_key_values=cache,
                use_kv_cache=True,
                last_prev_logit_hidden=last_prev_logit_hidden,
                cache_abs_pos=abs_pos,
            )
            cache = out.get("new_past_key_values", cache)
            abs_pos = out.get("new_cache_abs_pos", abs_pos + win_input.shape[1])
            # Sliding-window trim so cache size stays bounded.
            # NOTE: abs_pos is NOT reset by trim — new tokens still get the
            # correct absolute position, so RoPE math against the (older,
            # at-original-position) cached KVs is consistent.
            cache = _trim_kv_cache(cache, cap)
        else:
            # Rolling-buffer mode: build rolling LM context.
            lm_input_ids = torch.cat([lm_buffer, win_input], dim=1)
            if lm_input_ids.shape[1] > cap:
                lm_input_ids = lm_input_ids[:, -cap:]
            out = model.forward_window(
                lm_input_ids=lm_input_ids,
                prev_window_hiddens=cur_prev_hiddens,
                prev_states=states,
                target_mask=win_mask,
                hard_routing=hard_routing,
            )
            # Buffer for next window's LM input — at most (cap - T_window) tokens.
            lm_buffer = lm_input_ids[:, -(cap - T):] if cap > T else lm_input_ids[:, :0]

        # Strip heavy tensors before appending — `outputs` is debug-only.
        outputs.append({
            "new_states": out["new_states"],
            "read_visited": out["read_visited"],
            "write_visited": out["write_visited"],
            "surprise": out["surprise"],
        })
        # N6 — accumulate token-weighted sum + count if real Llama
        # (forward_window surfaces them); else fall back to per-window
        # mean (test mode).
        if "surprise_sum" in out and "surprise_count" in out:
            # Note: out["surprise_sum"] is detached (per integrated_lm.py).
            # We need the WITH-GRAD version for backward. Recompute
            # mean*count from the non-detached surprise:
            cnt = out["surprise_count"]
            ce_sum_with_grad = out["surprise"] * cnt   # surprise has grad
            if chunk_ce_sum is None:
                chunk_ce_sum = ce_sum_with_grad.sum()  # sum across BS
                chunk_ce_count = cnt.sum()
            else:
                chunk_ce_sum = chunk_ce_sum + ce_sum_with_grad.sum()
                chunk_ce_count = chunk_ce_count + cnt.sum()
        else:
            losses_fallback.append(out["surprise"].sum())
        surprises.append(out["surprise"])

        # Carry forward into next window.
        states = out["new_states"]
        cur_prev_hiddens = out["current_hiddens"]

    if chunk_ce_sum is not None:
        # Token-weighted aggregation. `chunk_ce_count.clamp_min(1.0)`
        # avoids div-by-zero when all windows are pad-only.
        aggregate_loss = chunk_ce_sum / chunk_ce_count.clamp_min(1.0)
    else:
        aggregate_loss = torch.stack(losses_fallback).sum()
    surprise_history = torch.stack(surprises, dim=1)               # [BS, D]

    result = {
        "window_outputs": outputs,
        "aggregate_loss": aggregate_loss,
        "final_states": states,
        "final_hiddens": cur_prev_hiddens,
        "final_lm_context": lm_buffer,
        "surprise_history": surprise_history,
    }
    if use_kv_cache:
        result["final_past_key_values"] = cache
        result["final_cache_abs_pos"] = abs_pos
    return result
