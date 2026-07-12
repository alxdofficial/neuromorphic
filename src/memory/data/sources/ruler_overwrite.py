"""RULER-overwrite source — random variables for the OVERWRITE / forced-forgetting task.

Procedural keyed source: the survey's forced-forgetting recipe forks RULER ``variable_tracking`` by
REASSIGNING the SAME key a new value, so a correct memory must return the LATEST binding and forget
the stale one. This source only supplies the raw ingredients — DISTINCT random variables (a random
key + a random value each, non-parametric so it can't be language-guessed) — and the dedicated
``overwrite`` task (``tasks/overwrite.py``) composes the reassignment sequence
(``KEY = v1 … distractors … KEY = v2``) and queries the key for the latest value.

Runtime-procedural: no offline build. Random alnum strings (shared helper with ``mqar``) that
round-trip through BPE for exact-match scoring. Generator stub:
``scripts/data_build/generate/ruler_overwrite/README.md``. See DATASETS.md / docs/DATA.md.
"""
from __future__ import annotations

from .base import Source
from .mqar import draw_keyed


class RulerOverwriteSource(Source):
    """Yields DISTINCT random ``KeyedItem`` variables; the ``overwrite`` task builds the chain."""

    kind = "keyed"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0,
                 key_len: int = 2, val_len: int = 2, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.key_len = key_len
        self.val_len = val_len

    def sample(self, rng, n: int) -> list:
        return draw_keyed(rng, n, self.key_len, self.val_len, "ruler_overwrite")
