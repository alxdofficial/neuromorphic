"""MuSiQue-Ans (TRAIN split) source — 2-4 hop QA over Wikipedia paragraphs, as QAItems.

Source half for the ``qa`` task (same shaper as bAbI): yields ``QAItem``s whose ``facts`` are the
example's paragraphs with the SUPPORTING paragraphs FIRST (guaranteed to survive the qa task's
front-pack + tail-truncate), then non-supporting distractor paragraphs, capped to a token budget so
every hop AND the answer sit inside the compression window (~900 of the default 1024 ctx tokens).

TRAIN-SPLIT FIREWALL: this reads HF ``dgslibisey/MuSiQue`` split **train**, whereas the eval reader
``src/memory/data/musique.py`` reads split **validation** — so nothing this trainer sees can leak
into the held-out eval. The internal "val" slice here is a held-out slice of TRAIN
(``train[n_docs:2·n_docs]``); the real validation split is reserved for eval. Answerable-only
(mirrors the eval reader's filter).

Data/build (BOUNDED, best-effort, never hangs):
  1. Local ``data/musique_train/<split>.jsonl`` (normalized rows) staged by the ingest script.
  2. Else HF non-streaming load from cache (``dgslibisey/MuSiQue``/train, answerable), sliced to n_docs.
  3. Else (HF unreachable / offline, uncached) a clear "run ingest first" error — no silent hang.
Ingest: ``scripts/data_build/ingest/musique_train/download.py``. See DATASETS.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .base import Source, QAItem
from ._corpus import local_jsonl

DATASET = "musique"
HF_NAME = "dgslibisey/MuSiQue"


def _local_jsonl(split: str):
    """Normalized-row jsonl staged by the ingest script for this split, if present."""
    return local_jsonl("musique_train", split)


def _normalize_musique(ex: dict) -> dict:
    """HF MuSiQue row → normalized {question, answer, aliases, paragraphs:[{title,text,support}]}.

    ``is_supporting`` flags the gold hops; each paragraph's text is ``paragraph_text`` (mirrors the
    eval reader's ``_tokenize_paragraph``). ``answer_aliases`` are kept for liberal downstream scoring.
    """
    paragraphs = [
        {"title": p["title"], "text": p["paragraph_text"], "support": bool(p["is_supporting"])}
        for p in ex["paragraphs"]
    ]
    aliases = [a for a in (ex.get("answer_aliases") or []) if a and a != (ex.get("answer") or "")]
    return {
        "question": (ex["question"] or "").strip(),
        "answer": (ex["answer"] or "").strip(),
        "aliases": aliases,
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
    """Load normalized rows from HF **train** (firewall), answerable-only, sliced train[0:n]/[n:2n]."""
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, split="train")              # TRAIN firewall (eval reads validation)
        if "answerable" in ds.column_names:                    # mirror the eval reader's filter
            ds = ds.filter(lambda ex: ex["answerable"])
    except Exception as e:  # network / offline / missing cache
        raise RuntimeError(
            f"[musique_train] HF dataset {HF_NAME!r} unreachable "
            f"({type(e).__name__}: {str(e)[:120]}). Run  python "
            f"scripts/data_build/ingest/musique_train/download.py  to stage a local "
            f"data/musique_train/{{train,val}}.jsonl sample, then retry (works offline once staged)."
        ) from e
    lo = 0 if split == "train" else n_docs
    hi = n_docs if split == "train" else 2 * n_docs
    hi = min(hi, len(ds))
    print(f"[musique_train] loaded {hi - lo} rows from HF {HF_NAME} train[{lo}:{hi}] (answerable; "
          f"split={split!r}; eval reserves the real validation split)", flush=True)
    return [_normalize_musique(ds[i]) for i in range(lo, hi)]


class MusiqueTrainSource(Source):
    """Yields MuSiQue (train-split) examples as ``QAItem``s: supporting-first, budget-capped paragraphs.

    ``facts`` = supporting paragraphs FIRST (capped to ``max_ctx_tokens`` ≈ 900 so every hop + the
    answer sit inside the front of the 1024 window and never tail-truncate), then as many in-example
    non-supporting paragraphs as fit. The REMAINING budget (≈900→total_len) is topped up by the qa
    task from ``distractor_pool()`` — a flat pool of the non-supporting (distractor) paragraphs across
    all loaded rows (cross-example distractors, exactly as the eval readers' top-up does).

    NOTE the pool is intentionally NON-empty: the qa task's fill-loop calls ``rng.choice(pool)`` to
    pad short contexts up to ``total_len`` and raises on an empty pool — since our supporting-first
    facts are capped well below the budget, an empty pool would crash. The pool tops up the tail with
    noise; supporting facts + answer remain in the protected ~900-token front.
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
            print(f"[musique_train] {split}: {len(self.rows)} rows from data/musique_train/"
                  f"{split}.jsonl", flush=True)
        else:
            self.rows = _rows_from_hf(split, n_docs)
        if not self.rows:
            raise ValueError(f"[musique_train] no rows for split={split!r}")
        # Flat noise pool for the qa task's tail top-up = non-supporting paragraphs (all rows).
        self._pool = [self._para_text(p) for row in self.rows
                      for p in row["paragraphs"] if not p["support"]]

    def _para_text(self, p: dict) -> str:
        return (p["title"].strip() + " " + p["text"].strip()).strip()

    def _flen(self, s: str) -> int:
        # Match the qa task's per-fact tokenization exactly: it emits ``ids(sent + "\n")``.
        return len(self.tok(s + "\n", add_special_tokens=False).input_ids)

    def _paras_to_facts(self, row: dict):
        """Supporting paragraphs FIRST (all kept), then distractors until the token budget is hit.

        Returns ``(facts, ok)``; ``ok=False`` when the supporting paragraphs ALONE overflow the
        budget — the caller resamples rather than emit an example whose answer would be truncated.
        """
        gold = [p for p in row["paragraphs"] if p["support"]]
        distr = [p for p in row["paragraphs"] if not p["support"]]
        facts: List[str] = []
        used = 0
        for p in gold:                       # supporting first — never dropped
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
            # rare all-oversized draw (ok stays False): the emitted oversized gold makes the packer
            # return None → TaskDataset resamples the episode; graceful, not a crash. See hotpot_train.
            out.append(QAItem(
                facts=facts, question=row["question"], answer=row["answer"], task_id=0,
                meta={"dataset": self.DATASET, "aliases": row.get("aliases", [])}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
