"""BabiLong-train source — bAbI facts scattered in real PG-19 prose (genuine long context, TRAIN split).

Source half of the ``qa`` task: each HF ``RMT-team/babilong-train-5k-samples`` row is a long PG-19
book excerpt with the bAbI supporting facts (and distractors) scattered through it verbatim, plus a
single-word question/answer pair (e.g. "Where is Sandra?" / "hallway") — exact-match scorable. This
extends our bAbI line (``sources/babi.py``, short question-agnostic stories) to GENUINE long context:
rows run to thousands of tokens of real narrative prose rather than a handful of templated sentences,
so the "retrieve the fact amid real distractor-shaped text" pressure is much closer to deployment than
bAbI's tiny stories or RULER's templated filler (``ruler_niah``).

NAMING GOTCHA — do not confuse with the EVAL reader: ``src/memory/data/babilong.py`` (``BABILongDataset``)
reads the DIFFERENT HF repo ``RMT-team/babilong`` (the held-out eval benchmark, tasks qa1-qa20, its own
train/val index-parity split). This source reads ``RMT-team/babilong-train-5k-samples`` — a separate
HF dataset the BabiLong authors ship specifically as non-eval training fuel — so there is no leakage
between the two: different dataset ids, never the same underlying examples.

Schema gotcha: only ``qa1``-``qa10`` splits exist per length config (qa11-qa20 aren't in this dump).
Length ``config`` knob: one of ``"0k","1k","2k","4k","8k","16k","32k"`` (approx. filler length); this
source mixes a few configs by default for length variety (``DEFAULT_CONFIGS``). GOTCHA: the ``"1k"``
config is broken on the HF repo itself (its data files 404 for every split, verified 2026-07-08) —
excluded from the default mix; the others (0k/2k/4k/8k/16k/32k) all resolve fine.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/babilong_train/<split>.jsonl`` (``{"config","task","input","question","answer"}``
     per line) — ingest cache.
  2. Else HF-stream a BOUNDED sample across the configured ``configs`` x ``tasks`` grid.
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/babilong_train/download.py`` (~2000 samples).
See DATASETS.md / docs/history/docs/history/data_arch_plan.md (Layer L1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import QAItem, Source
from ._corpus import local_jsonl

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

HF_NAME = "RMT-team/babilong-train-5k-samples"

# Only qa1-qa10 exist in this dump (unlike the eval `RMT-team/babilong` repo's qa1-qa20).
DEFAULT_TASKS: Tuple[str, ...] = tuple(f"qa{i}" for i in range(1, 11))
# A length mix for genuine-long-context variety, bounded well short of the 32k extreme by default.
# "1k" is excluded — broken on the HF repo itself (see module docstring gotcha).
DEFAULT_CONFIGS: Tuple[str, ...] = ("2k", "4k", "8k")

Row = Tuple[str, str, str, str]  # (config, task, input, question) -- answer kept separately below


def _split_sents(text: str) -> List[str]:
    """Split a long PG-19+bAbI passage into fact SENTENCES (flatten newlines, split on sentence-final
    punctuation) — mirrors ``babi.py``/``squad.py``'s splitter so the ``qa`` task sees a uniform fact
    granularity regardless of source."""
    flat = (text or "").replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("babilong_train", split)


def _task_id(task: str) -> int:
    """``"qa3"`` -> 3 (falls back to 0 on an unexpected format)."""
    m = re.match(r"qa(\d+)", task)
    return int(m.group(1)) if m else 0


def _rows_from_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[data.babilong_train] loaded {len(rows)} rows from {path}", flush=True)
    return rows


def _rows_from_hf(split: str, n_docs: int, configs, tasks) -> List[dict]:
    """Stream a BOUNDED, evenly-spread sample across ``configs`` x ``tasks`` from HF (train firewalled
    from eval by dataset id, not by split — see module docstring). ``split="train"`` takes each
    combo's first slice; any other split takes the NEXT disjoint slice (a held-out slice of this same
    train-only dump, analogous to ``hotpot_train``'s internal "val")."""
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover — missing dep
        raise RuntimeError(f"[babilong_train] `datasets` unavailable ({type(e).__name__}: {e})") from e

    combos = [(c, t) for c in configs for t in tasks]
    if not combos:
        raise ValueError("[babilong_train] empty configs/tasks grid")
    per_combo = max(1, -(-n_docs // len(combos)))   # ceil-divide, spread evenly
    is_val = split != "train"
    rows: List[dict] = []
    for cfg, task in combos:
        try:
            ds = load_dataset(HF_NAME, cfg, split=task, streaming=True)
        except Exception as e:
            print(f"[babilong_train]   {cfg}/{task} unreachable ({type(e).__name__}: "
                  f"{str(e)[:100]}); skipping", flush=True)
            continue
        take = per_combo if not is_val else max(1, per_combo // 4)   # smaller val slice
        skip = 0 if not is_val else per_combo                        # val = the NEXT disjoint slice
        seen = 0
        got = 0
        for ex in ds:
            if seen < skip:
                seen += 1
                continue
            inp = (ex.get("input") or "").strip()
            q = (ex.get("question") or "").strip()
            a = (ex.get("target") or "").strip()
            if inp and q and a:
                rows.append({"config": cfg, "task": task, "input": inp, "question": q, "answer": a})
                got += 1
            if got >= take:
                break
        if len(rows) >= n_docs:
            break
    if not rows:
        raise RuntimeError(
            f"[babilong_train] HF dataset {HF_NAME!r} unreachable across all {len(combos)} "
            f"config/task combos. Run  python scripts/data_build/ingest/babilong_train/download.py  "
            f"to stage a local data/babilong_train/{{train,val}}.jsonl sample, then retry "
            f"(works fully offline once staged).")
    print(f"[data.babilong_train] {split}: streamed {len(rows)} rows from HF:{HF_NAME} "
          f"(configs={list(configs)}, tasks={list(tasks)})", flush=True)
    return rows


class BabilongTrainSource(Source):
    """Yields BabiLong-train rows as ``QAItem``s: the long PG-19+bAbI passage sentence-split into
    ``facts``, the single-word question/answer kept verbatim (exact-match)."""

    kind = "qa"
    pack_n_queries = (1, 2)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 2000,
                 configs=DEFAULT_CONFIGS, tasks=DEFAULT_TASKS, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        local = _local_jsonl(split)
        if local is not None:
            self.rows = _rows_from_jsonl(local)
        else:
            self.rows = _rows_from_hf(split, n_docs, tuple(configs), tuple(tasks))
        if not self.rows:
            raise ValueError(f"[babilong_train] no rows for split={split!r}")

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            row = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(
                facts=_split_sents(row["input"]), question=row["question"], answer=row["answer"],
                task_id=_task_id(row["task"]),
                meta={"dataset": "babilong_train", "babi_task": row["task"], "config": row["config"]},
            ))
        return out
