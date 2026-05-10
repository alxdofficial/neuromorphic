"""Per-slot KV cache lifecycle for multi-stream batched training.

Replaces the "lockstep KV cache reset on any slot's doc-boundary" hack
in `train_wave1.py` with a proper per-slot cache lifecycle. The hack
costs ~24% of steps a full cache wipe (other slots lose their valid
cache for ~8 windows of refill); this module preserves each slot's
cache across the other slots' resets.

The pattern (vLLM/SGLang inflight batching, adapted for training):

  - Single batched cache `[BS, n_heads, T, d_head]` (HF DynamicCache).
    Tensors shared across slots; HF appends new K/V at the end as
    usual.
  - Per-slot bookkeeping outside the cache:
      `valid_len[s]` = number of MOST RECENT cache positions that
                      contain valid (non-stale) K/V for slot s.
      `abs_pos[s]`  = slot s's logical RoPE position counter (increments
                      monotonically; resets to 0 on slot's doc-start).
  - On per-slot doc-start: `valid_len[s] = 0`, `abs_pos[s] = 0`. The
    cache tensor itself isn't touched — slot s's K/V at positions
    [0, current_cache_len) are now "stale," but that's fine because
    the attention mask masks them out.
  - On every forward:
      * Build a 4D `[BS, 1, Q, K_total]` attention mask that gates
        slot s's queries to only attend to:
          - cache positions [L_old - valid_len[s], L_old)  (slot s's
            valid cached K/V — MOST RECENT valid_len[s] positions)
          - new positions [L_old, L_old + q] for query q  (causal
            within slot s's own new K/V)
      * Pass per-slot `position_ids` so RoPE rotates each slot's queries
        at its own logical absolute position.
      * `cache_position` is the same for all slots (= [L_old, L_old+Q))
        because they all write to the same cache tensor positions.
  - After forward: `valid_len[s] += Q` (clamped to T_max),
    `abs_pos[s] += Q`.
  - On cache trim (sliding window): `valid_len[s] = min(valid_len[s],
    new_max_len)`. `abs_pos[s]` unchanged (RoPE positions are absolute,
    not affected by tensor reorganization).

Key insight: this works with HF's DynamicCache as-is. We don't have to
splice K/V manually or hook the model — the 4D attention mask alone
is sufficient to make each slot ignore the others' cache positions.

`torch.compile(dynamic=True)` compatibility: cache TENSOR shapes are
the same as the lockstep path (no extra dynamism); only `valid_len`
and `abs_pos` data values vary per step. Should compile cleanly.

Caveat: SDPA's flash-attention sub-backend silently disables when
given a 4D mask, falling back to memory-efficient SDPA. At T_max=2048
/ BS=4 / Llama-3.2-1B (16 heads, d=64) this is still GPU-saturated;
profile if doubt.

For correctness verification: see `tests/test_per_slot_kv.py` —
asserts that BS=4 multi-stream produces the SAME logits as 4
sequential BS=1 runs of the same per-slot doc sequences (token-for-
token, mod fp tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PerSlotCacheState:
    """Per-slot KV cache state. The actual K/V tensors live in an HF
    DynamicCache; this struct just carries the bookkeeping that turns
    a single batched cache into N independent per-slot caches."""

    # HF DynamicCache (or None if not yet initialized). Tensors are
    # shared across slots; this struct tracks per-slot validity.
    cache: object | None
    # Per-slot count of MOST RECENT cache positions that are valid for
    # this slot. Always in [0, current_cache_length].
    valid_len: Tensor              # [BS] long
    # Per-slot RoPE absolute position counter. Resets to 0 on doc-start.
    # Increments by Q after each forward of Q new tokens.
    abs_pos: Tensor                # [BS] long
    # Cache length cap (sliding-window). Trim happens when cache exceeds
    # this; per-slot valid_len gets clamped accordingly.
    max_cache_len: int

    @classmethod
    def fresh(cls, batch_size: int, max_cache_len: int, device) -> "PerSlotCacheState":
        return cls(
            cache=None,
            valid_len=torch.zeros(batch_size, dtype=torch.int64, device=device),
            abs_pos=torch.zeros(batch_size, dtype=torch.int64, device=device),
            max_cache_len=max_cache_len,
        )

    def reset_slots(self, is_doc_start: Tensor) -> None:
        """Per-slot reset: where is_doc_start[i] is True, set valid_len[i]
        and abs_pos[i] to 0. Cache tensor is NOT touched — slot i's
        positions in the cache become "stale" and are masked out by the
        4D attention mask going forward.

        Args:
            is_doc_start: [BS] bool. True where the slot just started a
                          new document.
        """
        zero = torch.zeros_like(self.valid_len)
        self.valid_len = torch.where(is_doc_start, zero, self.valid_len)
        self.abs_pos = torch.where(is_doc_start, zero, self.abs_pos)

    def advance_after_forward(self, n_new: int) -> None:
        """Bookkeeping update after forwarding `n_new` tokens. Increments
        valid_len and abs_pos for all slots (each slot got n_new new
        valid positions and advances its RoPE counter by n_new)."""
        self.valid_len = torch.minimum(
            self.valid_len + n_new,
            torch.full_like(self.valid_len, self.max_cache_len),
        )
        self.abs_pos = self.abs_pos + n_new

    def trim_after_cache_trim(self, new_cache_len: int) -> None:
        """Bookkeeping update after the cache tensor was trimmed to
        `new_cache_len`. Each slot's valid_len is clamped to the new
        cache length (slots whose valid range extended past the trim
        boundary lose those positions). abs_pos unchanged."""
        cap = torch.full_like(self.valid_len, new_cache_len)
        self.valid_len = torch.minimum(self.valid_len, cap)


def build_per_slot_4d_mask(
    valid_len: Tensor,        # [BS] long — per-slot valid cache count
    q_len: int,               # number of new query tokens (= T_window)
    cache_len_old: int,       # cache length BEFORE this forward
    *,
    device,
    dtype,
) -> Tensor:
    """Build the 4D additive attention mask `[BS, 1, Q, K_total]` that
    gates slot s's queries to only attend to:
      - cache positions [L_old - valid_len[s], L_old)
        (slot s's valid cached K/V — most recent valid_len[s] of them)
      - new positions [L_old, L_old + q] for query q
        (causal within slot s's own new K/V — slot s sees its own past)

    Returns a tensor of shape `[BS, 1, Q, K_total]` with 0.0 where
    attention is allowed and -inf where masked.

    Vectorized — no Python loops over slots. Builds one tensor with
    cat([cache_part, causal_part], dim=-1).
    """
    BS = valid_len.shape[0]
    K_total = cache_len_old + q_len

    # ── Cache part: [BS, 1, Q, cache_len_old] ──
    # Slot s's cache validity for key position k:
    #   valid iff k >= cache_len_old - valid_len[s]
    # Doesn't depend on query (every query in this forward sees the
    # same cache validity for slot s).
    if cache_len_old > 0:
        cache_idx = torch.arange(
            cache_len_old, device=device, dtype=torch.int64,
        )                                              # [cache_len_old]
        slot_lo = cache_len_old - valid_len            # [BS] — first valid k
        cache_valid = cache_idx[None, :] >= slot_lo[:, None]   # [BS, cache_len_old]
        cache_valid = cache_valid[:, None, None, :].expand(BS, 1, q_len, cache_len_old)
    else:
        cache_valid = torch.zeros(
            BS, 1, q_len, 0, dtype=torch.bool, device=device,
        )

    # ── New-token part: [BS, 1, Q, Q] ──
    # Standard causal: query q can attend to new keys [0, q].
    # Same for all slots (all see their own new tokens causally).
    new_q = torch.arange(q_len, device=device, dtype=torch.int64)
    new_k = torch.arange(q_len, device=device, dtype=torch.int64)
    causal_2d = new_q[:, None] >= new_k[None, :]      # [Q, Q]
    causal = causal_2d[None, None, :, :].expand(BS, 1, q_len, q_len)

    # Concatenate cache_part + new_part along key dim.
    mask_bool = torch.cat([cache_valid, causal], dim=-1)  # [BS, 1, Q, K_total]

    # Convert to additive (-inf where False).
    # Use the dtype the model expects (bf16 / fp32) for proper masking.
    mask_additive = torch.zeros(
        (BS, 1, q_len, K_total), dtype=dtype, device=device,
    )
    mask_additive.masked_fill_(~mask_bool, float("-inf"))
    return mask_additive


def build_per_slot_position_ids(
    abs_pos: Tensor,    # [BS] long — per-slot RoPE position counter
    q_len: int,         # new query length
    *,
    device,
) -> Tensor:
    """Build `[BS, q_len]` position_ids where slot s's queries get
    RoPE positions [abs_pos[s], abs_pos[s] + q_len)."""
    offsets = torch.arange(q_len, device=device, dtype=torch.int64)
    # [BS, 1] + [1, Q] = [BS, Q]
    return abs_pos[:, None] + offsets[None, :]
