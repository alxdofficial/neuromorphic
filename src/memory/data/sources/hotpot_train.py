"""HotpotQA (TRAIN split) source — multi-hop QA over Wikipedia paragraphs, as QAItems.

Source half for the ``qa`` task (same shaper as bAbI): yields ``QAItem``s whose ``facts`` are the
example's paragraphs with the GOLD supporting paragraphs FIRST (guaranteed to survive the qa task's
front-pack + tail-truncate), then distractor paragraphs, capped to a token budget so both gold hops
AND the answer sit inside the compression window (~900 of the default 1024 ctx tokens).

TRAIN-SPLIT FIREWALL: this reads HF ``hotpot_qa``/``distractor`` split **train**, whereas the eval
reader ``src/memory/data/hotpot.py`` reads split **validation** — so nothing this trainer sees can
leak into the held-out eval. The internal "val" slice here is a held-out slice of TRAIN
(``train[n_docs:2·n_docs]``); the real validation split is reserved for eval.

Data/build (BOUNDED, best-effort, never hangs):
  1. Local ``data/hotpot_train/<split>.jsonl`` (normalized rows) staged by the ingest script.
  2. Else HF non-streaming load from cache (``hotpot_qa``/``distractor``/train), sliced to n_docs.
  3. Else (HF unreachable / offline, uncached) a clear "run ingest first" error — no silent hang.
Ingest: ``scripts/data_build/ingest/hotpot_train/download.py``. See DATASETS.md.

NOTE: we load NON-streaming (loading-script datasets like ``hotpot_qa`` do not stream cleanly and
can hang); the arrow cache is memory-mapped, so materializing only the first ``2·n_docs`` rows is cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .base import Source, QAItem
from ._corpus import local_jsonl

DATASET = "hotpot"
HF_NAME = "hotpot_qa"
HF_CONFIG = "distractor"


def _local_jsonl(split: str):
    """Normalized-row jsonl staged by the ingest script for this split, if present."""
    return local_jsonl("hotpot_train", split)


def _normalize_hotpot(ex: dict) -> dict:
    """HF hotpot row → normalized {question, answer, aliases, paragraphs:[{title,text,support}]}.

    A paragraph is supporting (gold) iff its title appears in ``supporting_facts.title``; each
    paragraph's text is its sentences joined (mirrors the eval reader's ``_tokenize_paragraph``).
    """
    sf = ex.get("supporting_facts") or {}
    support_titles = set(sf.get("title", []) or [])
    ctx = ex["context"]
    titles = ctx["title"]
    sents_list = ctx["sentences"]
    paragraphs = [
        {"title": t, "text": " ".join(s.strip() for s in sl), "support": t in support_titles}
        for t, sl in zip(titles, sents_list)
    ]
    return {
        "question": (ex["question"] or "").strip(),
        "answer": (ex["answer"] or "").strip(),
        "aliases": [],
        "paragraphs": paragraphs,
    }


def _rows_from_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_from_hf(split: str, n_docs: int) -> List[dict]:
    """Load normalized rows from the HF **train** split (firewall), sliced train[0:n] / train[n:2n]."""
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, HF_CONFIG, split="train")   # TRAIN firewall (eval reads validation)
    except Exception as e:  # network / offline / missing cache
        raise RuntimeError(
            f"[hotpot_train] HF dataset {HF_NAME!r}/{HF_CONFIG!r} unreachable "
            f"({type(e).__name__}: {str(e)[:120]}). Run  python "
            f"scripts/data_build/ingest/hotpot_train/download.py  to stage a local "
            f"data/hotpot_train/{{train,val}}.jsonl sample, then retry (works offline once staged)."
        ) from e
    lo = 0 if split == "train" else n_docs
    hi = n_docs if split == "train" else 2 * n_docs
    hi = min(hi, len(ds))
    print(f"[hotpot_train] loaded {hi - lo} rows from HF {HF_NAME}/{HF_CONFIG} train[{lo}:{hi}] "
          f"(split={split!r}; eval reserves the real validation split)", flush=True)
    return [_normalize_hotpot(ds[i]) for i in range(lo, hi)]


class HotpotTrainSource(Source):
    """Yields HotpotQA (train-split) examples as ``QAItem``s: gold-first, budget-capped paragraphs.

    ``facts`` = gold supporting paragraphs FIRST (capped to ``max_ctx_tokens`` ≈ 900 so both hops +
    the answer sit inside the front of the 1024 window and never tail-truncate), then as many
    in-example distractor paragraphs as fit. The REMAINING budget (≈900→total_len) is topped up by
    the qa task from ``distractor_pool()`` — a flat pool of the non-supporting (distractor) paragraphs
    across all loaded rows (cross-example distractors, exactly as the eval readers' top-up does).

    NOTE the pool is intentionally NON-empty: the qa task's fill-loop calls ``rng.choice(pool)`` to
    pad short contexts up to ``total_len`` and raises on an empty pool — since our gold-first facts
    are capped well below the budget, an empty pool would crash. The pool tops up the tail with noise;
    gold + answer remain in the protected ~900-token front.
    """

    kind = "qa"
    DATASET = DATASET

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 max_ctx_tokens: int = 900, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.max_ctx_tokens = max_ctx_tokens

        local = _local_jsonl(split)
        if local is not None:
            self.rows = _rows_from_jsonl(local)
            print(f"[hotpot_train] {split}: {len(self.rows)} rows from data/hotpot_train/"
                  f"{split}.jsonl", flush=True)
        else:
            self.rows = _rows_from_hf(split, n_docs)
        if not self.rows:
            raise ValueError(f"[hotpot_train] no rows for split={split!r}")
        # Flat noise pool for the qa task's tail top-up = non-supporting paragraphs (all rows).
        self._pool = [self._para_text(p) for row in self.rows
                      for p in row["paragraphs"] if not p["support"]]

    def _para_text(self, p: dict) -> str:
        return (p["title"].strip() + " " + p["text"].strip()).strip()

    def _flen(self, s: str) -> int:
        # Match the qa task's per-fact tokenization exactly: it emits ``ids(sent + "\n")``.
        return len(self.tok(s + "\n", add_special_tokens=False).input_ids)

    def _paras_to_facts(self, row: dict):
        """Gold paragraphs FIRST (all kept), then distractors until the token budget is hit.

        Returns ``(facts, ok)``; ``ok=False`` when the gold paragraphs ALONE overflow the budget —
        the caller resamples rather than emit an example whose answer would be truncated away.
        """
        gold = [p for p in row["paragraphs"] if p["support"]]
        distr = [p for p in row["paragraphs"] if not p["support"]]
        facts: List[str] = []
        used = 0
        for p in gold:                       # gold first — never dropped
            s = self._para_text(p)
            facts.append(s)
            used += self._flen(s)
        if not gold or used > self.max_ctx_tokens:
            return facts, False              # unsolvable within budget → resample
        for p in distr:                      # fill remaining budget with in-example distractors
            s = self._para_text(p)
            n = self._flen(s)
            if used + n > self.max_ctx_tokens:
                continue                     # skip a too-big distractor, try the next (smaller) one
            facts.append(s)
            used += n
        return facts, True

    def sample(self, rng, n: int) -> list:
        out: List[QAItem] = []
        for _ in range(n):
            facts, row = [], None
            for _ in range(50):              # bounded resample on oversized-gold draws
                row = self.rows[rng.randrange(len(self.rows))]
                facts, ok = self._paras_to_facts(row)
                if ok:
                    break
            out.append(QAItem(
                facts=facts, question=row["question"], answer=row["answer"], task_id=0,
                meta={"dataset": self.DATASET, "aliases": row.get("aliases", [])}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
