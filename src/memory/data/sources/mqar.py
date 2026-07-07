"""MQAR source — random-token key→value pairs (un-guessable associative recall).

Procedural keyed source implementing the Zoology MQAR probe: each item is a FRESH random key and
a FRESH random value (no world/parametric knowledge), so the only way to answer a key's query is
to have actually stored that exact pairing in memory — pure associative recall, ungameable by a
language prior. Pairs with the EXISTING ``reconstruction`` task (multi-query ⇒ addressing).

Runtime-procedural: no offline build. Keys/values are short random alnum strings (UUID-like) which
round-trip cleanly through any BPE tokenizer for exact-match scoring. ``key_len`` / ``val_len`` set
the number of random alnum chunks (≈ tokens) per key / value. Draws yield DISTINCT keys (a repeated
key would make the query target ambiguous).

Generator stub: ``scripts/data_build/generate/mqar/README.md`` (nothing to build). See
DATASETS.md / docs/data_arch_plan.md (Layer L1).
"""
from __future__ import annotations

import string
from typing import List

from .base import Source, KeyedItem

_ALNUM = string.ascii_lowercase + string.digits


def rand_alnum(rng, n_chunks: int, chunk_chars: int = 4) -> str:
    """A short random alnum string: ``n_chunks`` dash-joined ``chunk_chars``-char chunks (UUID-like).

    High-entropy + un-guessable (36^(n_chunks·chunk_chars) space) and round-trips through BPE for
    exact-match scoring. ``n_chunks`` is the length knob (≈ tokens after BPE)."""
    return "-".join(
        "".join(rng.choice(_ALNUM) for _ in range(chunk_chars))
        for _ in range(max(1, n_chunks))
    )


def draw_keyed(rng, n: int, key_len: int, val_len: int, label: str) -> List[KeyedItem]:
    """Draw ``n`` random ``KeyedItem``s with DISTINCT keys (shared by mqar + ruler_overwrite).

    Each item = a random key, a random value, ``value_subs=[value]`` (the whole value is the
    load-bearing span), no entity name/given. Retries on a (near-impossible) key collision so a
    draw never emits two indistinguishable keys."""
    out: List[KeyedItem] = []
    seen: set = set()
    guard = 0
    while len(out) < n and guard < 100 * n + 100:
        guard += 1
        k = rand_alnum(rng, key_len)
        if k in seen:
            continue
        seen.add(k)
        v = rand_alnum(rng, val_len)
        out.append(KeyedItem(key_text=k, value_text=v, value_subs=[v], name="", given=""))
    if len(out) < n:
        raise ValueError(
            f"{label}: could not draw {n} distinct keys at key_len={key_len} (drew {len(out)})")
    return out


class MqarSource(Source):
    """Yields random key→value ``KeyedItem``s (Zoology MQAR); distinct keys per draw."""

    kind = "keyed"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0,
                 key_len: int = 2, val_len: int = 2, **kw):
        # Fully random ⇒ train/val streams are naturally disjoint (collision probability ~0); the
        # per-worker RNG the Task passes to ``sample`` supplies the entropy. split/seed are kept for
        # interface parity + reproducibility bookkeeping.
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.key_len = key_len
        self.val_len = val_len

    def sample(self, rng, n: int) -> list:
        return draw_keyed(rng, n, self.key_len, self.val_len, "mqar")
