"""bAbI source — question-agnostic stories (declarative facts) + a distractor sentence pool.

Source half of the old ``data/babi.py``: loads rows from HF ``Muennighoff/babi`` (offline fallback
synthesizes task-1), splits stories into fact sentences, and exposes a noise pool of other stories'
sentences for the qa task to pad with. The distractor-insertion / packing is the qa Task's job.

Data/build: HF ``Muennighoff/babi`` (1k), auto-downloaded; 10k ingest =
``scripts/data_build/ingest/babi_10k.py`` (TODO). See DATASETS.md / docs/data_arch_plan.md.
"""
from __future__ import annotations

import random
import re
from typing import List

from .base import Source, QAItem

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Memory-focused task subset (default). 1/2/3 = one/two/three supporting facts, 7 = counting,
# 8 = lists/sets, 11/12/13 = coreference, 14 = time reasoning.
DEFAULT_TASKS = (1, 2, 3, 7, 8, 11, 12, 13, 14)

_FALLBACK_NAMES = ["Mary", "John", "Daniel", "Sandra", "Fred", "Julie", "Bill", "Jeff"]
_FALLBACK_PLACES = ["bathroom", "hallway", "kitchen", "garden", "bedroom",
                    "office", "school", "park", "cinema", "kitchen"]
_FALLBACK_MOVES = ["moved to", "went to", "journeyed to", "travelled to", "went back to"]


def _split_sents(text: str) -> List[str]:
    """Split a bAbI passage into individual fact SENTENCES (handles both the HF one-line form and
    the newline-separated fallback): flatten newlines → split after sentence-final punctuation."""
    flat = text.replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _caps_names(text: str) -> set:
    """bAbI named entities = capitalized alphabetic tokens (minus sentence-initial 'The'). Used to
    keep distractors DISJOINT from the gold story (bAbI reuses a tiny name pool, so a same-name
    distractor would silently contradict the queried entity's labelled state)."""
    out = set()
    for w in text.replace("\n", " ").split():
        w = w.strip(".,!?;:'\"")
        if w and w[0].isupper() and w.isalpha() and w != "The":
            out.add(w)
    return out


def _load_babi_rows(tasks, split: str):
    """Return list[(story, question, answer, task_int)] for the given tasks/split. Tries HF, then
    a programmatic task-1 fallback if offline. The story field is QUESTION-AGNOSTIC by construction."""
    task_set = set(tasks)
    if split == "train":
        hf_split = "train"
    elif split in ("validation", "val"):
        hf_split = "validation"
    else:
        raise ValueError(
            f"bAbI: unrecognized split {split!r} (expected 'train', 'validation', or 'val')")

    for name in ("Muennighoff/babi",):
        try:
            from datasets import load_dataset
            ds = load_dataset(name, split=hf_split)
            rows = []
            for ex in ds:
                t = int(ex["task"])
                if t not in task_set:
                    continue
                story = (ex["passage"] or "").strip()
                q = (ex["question"] or "").strip()
                a = (ex["answer"] or "").strip()
                if story and q and a:
                    rows.append((story, q, a, t))
            if rows:
                print(f"[babi] loaded {len(rows):,} rows from {name} "
                      f"(split={hf_split}, tasks={sorted(task_set)})", flush=True)
                return rows
        except Exception as e:  # pragma: no cover — network/offline path
            print(f"[babi] {name} unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  f"trying next source", flush=True)

    if 1 not in task_set:
        raise RuntimeError(
            f"bAbI HF source unreachable and requested tasks {sorted(task_set)} do not include "
            f"task 1 — the offline fallback only synthesizes task-1 stories. Restore network "
            f"access (Muennighoff/babi) or include task 1 in the requested tasks.")
    is_val = split in ("validation", "val")
    print(f"[babi] HF bAbI unreachable — generating programmatic task-1 stories "
          f"(offline fallback, split={'val' if is_val else 'train'})", flush=True)
    gen = random.Random(5678 if is_val else 1234)
    rows = []
    for _ in range(4000):
        n_facts = gen.randint(2, 8)
        loc = {}
        lines = []
        for _ in range(n_facts):
            who = gen.choice(_FALLBACK_NAMES)
            where = gen.choice(_FALLBACK_PLACES)
            loc[who] = where
            lines.append(f"{who} {gen.choice(_FALLBACK_MOVES)} the {where}.")
        who_q = gen.choice(list(loc.keys()))
        rows.append(("\n".join(lines) + "\n", f"Where is {who_q}?", loc[who_q], 1))
    return rows


class BabiSource(Source):
    """Yields bAbI stories as QAItems + a flat distractor-sentence pool."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", tasks=DEFAULT_TASKS, seed: int = 0, **kw):
        self.tasks = tuple(tasks)
        self.rows = _load_babi_rows(self.tasks, split)
        if not self.rows:
            raise ValueError(f"bAbI: no rows for tasks={self.tasks} split={split}")
        self._pool: List[str] = []
        for story, _q, _a, _t in self.rows:
            self._pool.extend(_split_sents(story))

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            story, q, a, t = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(facts=_split_sents(story), question=q, answer=a, task_id=t))
        return out

    def distractor_pool(self) -> list:
        return self._pool
