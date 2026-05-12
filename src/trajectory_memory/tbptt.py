"""TBPTT scaffolding — chain D forward_window calls with cross-window
gradient flow.

Linear-in-D implementation: every window's autograd graph stays alive
until backward fires at the end of the chunk. Memory cost is roughly
D × per-window activations.

KV cache (single-stream lockstep) is on by default — Llama re-encodes
only the new T_window tokens each window. For BS>1, the per-slot
multi-stream complexity was removed in favor of lockstep reset when
any slot crosses a doc boundary (small efficiency cost, large code
simplification).

Usage:
    chunks = TBPTTChunker(cfg).split(input_ids)   # [num_chunks, D, T_window]
    prev_states, prev_hiddens, prev_lm_ctx = None, None, None
    past_kv, cache_abs_pos = None, 0
    for chunk in chunks:
        result = run_chunk(
            integrated_lm, chunk, prev_states, prev_hiddens, prev_lm_ctx,
            past_key_values=past_kv, cache_abs_pos=cache_abs_pos,
        )
        loss = result["aggregate_loss"]
        loss.backward()
        prev_states = result["final_states"].detach()              # cut grad
        prev_hiddens = result["final_hiddens"].detach()
        prev_lm_ctx = result["final_lm_context"]                   # already detached
        past_kv = _detach_kv_cache(result["final_past_key_values"])
        cache_abs_pos = result["final_cache_abs_pos"]
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

    Used by Phase 2 rollout (long generated sequences capped to a sliding
    context). Phase 1 training uses rolling-buffer mode and never sees a
    cache, so this helper is Phase-2-only despite being colocated here for
    historical reasons.
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
    """Detach all cache tensors so a new GRPO step starts with no autograd
    history. Phase-2 rollout helper."""
    if cache is None:
        return None
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        layer.keys = layer.keys.detach()
        layer.values = layer.values.detach()
    return cache


def _crop_kv_cache(cache, max_length: int):
    """Crop each layer's K/V tensors to the last `max_length` positions.

    Enforces `cfg.effective_lm_context` as a sliding-window cap on Llama's
    attention. Without this, our KV-cache-mode training lets Llama see the
    entire doc (cache grows unbounded), so memory has no functional role.
    Cropping makes the cap effective and forces memory to carry info past
    the LM's attention range.

    Idempotent: if cache length is already ≤ max_length, no-op. Called
    after each `forward_window`'s Llama forward in KV-cache mode.
    """
    if cache is None or max_length <= 0:
        return cache
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        seq_len = layer.keys.shape[-2]
        if seq_len > max_length:
            # keys/values shape: [BS, n_heads, seq, head_dim]
            layer.keys = layer.keys[..., -max_length:, :]
            layer.values = layer.values[..., -max_length:, :]
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
    use_kv_cache: bool = True,
    past_key_values: object | None = None,
    cache_abs_pos: int = 0,
    tau: Tensor | float | None = None,
) -> dict:
    """Run D consecutive windows with autograd kept alive across the chunk.

    Two execution modes:

    - **KV-cache mode** (default, recommended): each window's Llama
      forward encodes only the new T_window tokens against
      `past_key_values`. Cache is sliding-window trimmed to
      `cfg.effective_lm_context`. Across chunks the cache is detached
      but threaded (single-stream lockstep — for BS>1 the caller resets
      the cache to None whenever any slot crosses a doc boundary).

    - **Rolling-buffer mode** (`use_kv_cache=False`): maintains a rolling
      LM-context buffer of up to `cfg.effective_lm_context` tokens. At
      each window, Llama re-encodes the buffer + new tokens. ~1.79×
      slower per step on Phase 1 — only used as a no-cache fallback.

    Args:
        model:               IntegratedLM
        windows:             [BS, D, T_window] token IDs
        prev_states:         [BS, N, D_concept] state at start of chunk
        prev_window_hiddens: [BS, T_window, d_lm] from window before this
                             chunk, or None if chunk starts a new sequence.
        prev_lm_context:     Rolling-buffer mode: [BS, L] tokens from
                             prior chunk. KV-cache mode: ignored.
        target_mask:         [BS, D, T_window] or None.
        hard_routing:        Gumbel-STE if True.
        use_kv_cache:        Pick mode (above).
        past_key_values:     KV-cache mode: HF DynamicCache from previous
                             chunk (detached) or None.
        cache_abs_pos:       KV-cache mode: absolute position of next
                             token in the cache (for RoPE correctness
                             across cache trims).

    Returns:
        dict with keys:
            window_outputs:        list of D forward_window outputs
            aggregate_loss:        scalar loss summed over windows (NTP CE)
            final_states:          [BS, N, D_concept] state after last window
            final_hiddens:         [BS, T_window, d_lm] hiddens of last window
            final_lm_context:      Rolling-buffer mode: [BS, L] for next chunk.
                                   KV-cache mode: empty.
            final_past_key_values: KV-cache mode only — updated cache.
            final_cache_abs_pos:   KV-cache mode only.
            surprise_history:      [BS, D] surprise per window
    """
    BS, D, T = windows.shape
    assert D == model.cfg.D, f"chunk has D={D}, expected {model.cfg.D}"
    cfg = model.cfg
    cap = cfg.effective_lm_context

    states = prev_states
    cur_prev_hiddens = prev_window_hiddens
    cache = past_key_values if use_kv_cache else None
    abs_pos = cache_abs_pos

    # Rolling LM-context buffer (rolling-buffer mode only).
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
    # Routing aux-loss accumulators (Switch-style load-balance + ST-MoE
    # z-loss). Surfaced per-window by forward_window; averaged over D
    # windows here so the chunk-level aux loss has a consistent scale
    # regardless of D (TBPTT depth).
    chunk_aux_lb: Tensor | None = None
    chunk_aux_z: Tensor | None = None
    # Fallback path for test mode (no real Llama → no surprise_sum/count
    # in `out`): aggregate per-window mean × T_window like before.
    losses_fallback: list[Tensor] = []

    for d in range(D):
        win_input = windows[:, d, :]                               # [BS, T_window]
        win_mask = target_mask[:, d, :] if target_mask is not None else None

        if use_kv_cache:
            # KV-cache mode: pass only the new window's T_window tokens.
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
                tau=tau,
            )
            n_new = win_input.shape[1]
            cache = out.get("new_past_key_values", cache)
            abs_pos = out.get("new_cache_abs_pos", abs_pos + n_new)
            # Sliding-window trim. abs_pos is NOT reset by trim — new
            # tokens still get correct absolute positions, so RoPE math
            # against (older, at-original-position) cached KVs stays
            # consistent.
            cache = _trim_kv_cache(cache, cap)
        else:
            # Rolling-buffer fallback.
            lm_input_ids = torch.cat([lm_buffer, win_input], dim=1)
            if lm_input_ids.shape[1] > cap:
                lm_input_ids = lm_input_ids[:, -cap:]
            out = model.forward_window(
                lm_input_ids=lm_input_ids,
                prev_window_hiddens=cur_prev_hiddens,
                prev_states=states,
                target_mask=win_mask,
                hard_routing=hard_routing,
                tau=tau,
            )
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
        if "surprise_weighted_sum" in out and "surprise_count" in out:
            # `surprise_weighted_sum` HAS grad and HAS the float-mask
            # weights baked in (e.g. prior_loss_weight=0.1). Use it
            # directly as the with-grad CE sum. The earlier reconstruction
            # `surprise * count` divided out the mask weights — broke
            # prior_loss_weight entirely.
            cnt = out["surprise_count"]
            ce_sum_with_grad = out["surprise_weighted_sum"]
            if chunk_ce_sum is None:
                chunk_ce_sum = ce_sum_with_grad.sum()  # sum across BS
                chunk_ce_count = cnt.sum()
            else:
                chunk_ce_sum = chunk_ce_sum + ce_sum_with_grad.sum()
                chunk_ce_count = chunk_ce_count + cnt.sum()
        else:
            losses_fallback.append(out["surprise"].sum())
        surprises.append(out["surprise"])

        # Accumulate routing aux losses per window. forward_window already
        # combined read+write inside each window; here we just sum across
        # the D windows and divide once at the end.
        _lb = out.get("aux_load_balance")
        _z = out.get("aux_z_loss")
        if _lb is not None:
            chunk_aux_lb = _lb if chunk_aux_lb is None else chunk_aux_lb + _lb
            chunk_aux_z = _z if chunk_aux_z is None else chunk_aux_z + _z

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

    # Average aux losses over the D windows of this chunk so the
    # chunk-level magnitude is independent of D.
    if chunk_aux_lb is not None:
        chunk_aux_lb = chunk_aux_lb / float(D)
        chunk_aux_z = chunk_aux_z / float(D)

    result = {
        "window_outputs": outputs,
        "aggregate_loss": aggregate_loss,
        "final_states": states,
        "final_hiddens": cur_prev_hiddens,
        "final_lm_context": lm_buffer,
        "surprise_history": surprise_history,
        "aux_load_balance": chunk_aux_lb,
        "aux_z_loss": chunk_aux_z,
    }
    # Surface chunk-level weighted CE sum (with grad) + valid token count
    # (detached) so callers spanning multiple chunks (Phase 1 step_wave2)
    # can aggregate token-weighted across chunks rather than chunk-equal.
    # chunk_ce_sum is None in test-mode fallback.
    if chunk_ce_sum is not None:
        result["chunk_ce_sum"] = chunk_ce_sum
        result["chunk_ce_count"] = chunk_ce_count
    if use_kv_cache:
        result["final_past_key_values"] = cache
        result["final_cache_abs_pos"] = abs_pos
    return result
