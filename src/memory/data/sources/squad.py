"""SQuAD 2.0 QA source — a short context paragraph + an extractive (or abstention) answer.

Source half of the ``qa`` task for reading-comprehension: each SQuAD example is a Wikipedia
paragraph, a question, and either an extractive answer span (answerable) or *no* answer (v2's
unanswerable half → the literal ``"unanswerable"`` abstention target — kept on purpose, it is the
retrieve-vs-abstain signal). Contexts are short (~150 tok) so they fit the budget with huge margin;
the ``qa`` task pads them to ``total_len`` with OTHER contexts' sentences (``distractor_pool``),
turning each into a retrieve-among-noise read where the answer lives only in the gold paragraph.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/squad/<split>.jsonl`` (``{"context","question","answer"}`` per line) — ingest cache.
  2. Else HF-stream a BOUNDED sample of ``rajpurkar/squad_v2`` (train / validation).
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/squad/download.py``. See DATASETS.md / docs/history/docs/history/data_arch_plan.md (L1).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

UNANSWERABLE = "unanswerable"
HF_NAME = "rajpurkar/squad_v2"


def _split_sents(text: str) -> List[str]:
    """Split a context paragraph into fact sentences (flatten newlines, split on sentence-final punct)."""
    flat = (text or "").replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("squad", split)


def _answer_of(answers: dict) -> str:
    """SQuAD v2 answer: first extractive span, or the abstention target if empty (unanswerable)."""
    texts = (answers or {}).get("text") or []
    if texts and (texts[0] or "").strip():
        return texts[0].strip()
    return UNANSWERABLE


def _iter_hf_rows(split: str, n_docs: int) -> List[Tuple[str, str, str]]:
    """Stream up to ``n_docs`` (context, question, answer) triples from HF squad_v2."""
    hf_split = "train" if split == "train" else "validation"
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, split=hf_split, streaming=True)
    except Exception as e:  # network / offline / missing dataset
        raise RuntimeError(
            f"[squad] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/squad/download.py  to stage a local "
            f"data/squad/{{train,val}}.jsonl sample, then retry (works fully offline once staged)."
        ) from e
    rows: List[Tuple[str, str, str]] = []
    for ex in ds:
        ctx = (ex.get("context") or "").strip()
        q = (ex.get("question") or "").strip()
        if not ctx or not q:
            continue
        rows.append((ctx, q, _answer_of(ex.get("answers"))))
        if len(rows) >= n_docs:
            break
    return rows


def _load_squad_rows(split: str, n_docs: int) -> List[Tuple[str, str, str]]:
    """(context, question, answer) triples — local staged jsonl first, else bounded HF stream."""
    local = _local_jsonl(split)
    if local is not None:
        rows: List[Tuple[str, str, str]] = []
        with open(local) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ctx = (o.get("context") or "").strip()
                q = (o.get("question") or "").strip()
                a = (o.get("answer") or "").strip() or UNANSWERABLE
                if ctx and q:
                    rows.append((ctx, q, a))
                if len(rows) >= n_docs:
                    break
        origin = f"data/squad/{local.name}"
    else:
        rows = _iter_hf_rows(split, n_docs)
        origin = f"HF:{HF_NAME}"
    if not rows:
        raise ValueError(f"[squad] no rows loaded from {origin} (split={split}).")
    n_uns = sum(1 for _c, _q, a in rows if a == UNANSWERABLE)
    print(f"[data.squad] {split}: {len(rows)} contexts from {origin} "
          f"({n_uns} unanswerable / {len(rows) - n_uns} answerable)", flush=True)
    return rows


class SquadSource(Source):
    """Yields SQuAD paragraphs as QAItems (sentence-split facts) + an other-contexts distractor pool."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 4000,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.rows = _load_squad_rows(split, n_docs)
        # Distractor pool = OTHER contexts' sentences (padding noise). The qa task already rejects
        # distractors sharing a gold capitalized name, so cross-context contamination is filtered.
        pool: List[str] = []
        seen = set()
        for ctx, _q, _a in self.rows:
            for s in _split_sents(ctx):
                if len(s) < 20 or s in seen:
                    continue
                seen.add(s)
                pool.append(s)
        if len(pool) > pool_cap:
            pool = random.Random(seed).sample(pool, pool_cap)
        self._pool = pool

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            ctx, q, a = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(facts=_split_sents(ctx), question=q, answer=a,
                              task_id=0, meta={"dataset": "squad"}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
