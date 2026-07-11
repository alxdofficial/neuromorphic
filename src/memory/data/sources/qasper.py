"""Qasper source — NLP-paper QA over the WHOLE paper (non-gist-gameable by construction).

Source half of the ``qa`` task for scientific-paper reading comprehension: each Qasper example is one
NLP paper (``allenai/qasper``) with several questions written by an NLP practitioner who read ONLY the
title + abstract (never the body) — so a genuine answer requires actually reading the paper, unlike
squad/synthetic QA where the question-writer already saw the gold paragraph. ``facts`` = the paper's
section paragraphs flattened in order (the WHOLE paper, not just the answer-bearing section); the
answer is picked per Qasper's own priority (unanswerable > extractive span > yes/no > free-form).

``pack_n_queries = (1, 2)`` — a paper averages ~5-9 questions, so a couple can be asked per pack
without re-drawing the (expensive, long) paper each time.

GOTCHA: ``allenai/qasper`` is a loading-script dataset — unsupported by ``datasets`` >= 4.x. We instead
read HF's auto-converted parquet mirror via ``revision="refs/convert/parquet"`` (same schema, no
script), which streams/loads cleanly.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/qasper/<split>.jsonl`` (``{"context","question","answer"}``, one line per (paper,
     question) pair — ``context`` = ``"\\n\\n".join(paragraphs)``) — ingest cache.
  2. Else HF-stream a BOUNDED sample of ``allenai/qasper`` (train / validation), exploding each paper
     into its (context, question, answer) rows.
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/qasper/download.py``. See DATASETS.md / docs/history/docs/history/data_arch_plan.md (L1).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

_PARA_SPLIT = re.compile(r"\n{2,}")

UNANSWERABLE = "unanswerable"
HF_NAME = "allenai/qasper"

Row = Tuple[str, str, str]   # (context, question, answer) — context = "\n\n".join(paragraphs)


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("qasper", split)


def _hf_split(split: str) -> str:
    if split == "train":
        return "train"
    if split in ("validation", "val"):
        return "validation"
    raise ValueError(f"qasper: unrecognized split {split!r} (expected 'train'/'validation'/'val')")


def _answer_of(ans: dict) -> str:
    """Qasper per-annotator answer dict → a scorable string, in the dataset's own priority order:
    unanswerable > extractive span(s) > yes/no > free-form (abstractive) fallback."""
    if ans.get("unanswerable"):
        return UNANSWERABLE
    spans = [s.strip() for s in (ans.get("extractive_spans") or []) if s and s.strip()]
    if spans:
        return " ".join(spans)
    yn = ans.get("yes_no")
    if yn is True:
        return "yes"
    if yn is False:
        return "no"
    ff = (ans.get("free_form_answer") or "").strip()
    return ff or UNANSWERABLE


def _flatten_paragraphs(full_text: dict) -> List[str]:
    """``full_text`` = {"section_name": [...], "paragraphs": [[str, ...], ...]} → flat paragraph list,
    order preserved, section headers dropped (facts are the paragraphs themselves)."""
    paras: List[str] = []
    for sec_paras in (full_text.get("paragraphs") or []):
        for p in sec_paras:
            p = (p or "").strip()
            if p:
                paras.append(p)
    return paras


def _rows_from_paper(ex: dict) -> List[Row]:
    """One Qasper HF row (a paper) → its exploded (context, question, answer) rows, one per question
    (using the FIRST annotator's answer — Qasper questions can have multiple independent answer sets)."""
    paras = _flatten_paragraphs(ex.get("full_text") or {})
    if not paras:
        return []
    context = "\n\n".join(paras)
    qas = ex.get("qas") or {}
    questions = qas.get("question") or []
    answers_list = qas.get("answers") or []
    out: List[Row] = []
    for q, ans_block in zip(questions, answers_list):
        q = (q or "").strip()
        if not q:
            continue
        worker_answers = (ans_block or {}).get("answer") or []
        if not worker_answers:
            continue
        out.append((context, q, _answer_of(worker_answers[0])))
    return out


def _iter_hf_rows(split: str, n_docs: int) -> List[Row]:
    """Stream up to ``n_docs`` PAPERS from HF qasper (parquet mirror), exploded to per-question rows."""
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, split=_hf_split(split), revision="refs/convert/parquet",
                           streaming=True)
    except Exception as e:  # network / offline / missing dataset
        raise RuntimeError(
            f"[qasper] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/qasper/download.py  to stage a local "
            f"data/qasper/{{train,val}}.jsonl sample, then retry (works fully offline once staged)."
        ) from e
    rows: List[Row] = []
    n_papers = 0
    for ex in ds:
        rows.extend(_rows_from_paper(ex))
        n_papers += 1
        if n_papers >= n_docs:
            break
    print(f"[data.qasper] {split}: {len(rows)} QA pairs from {n_papers} papers (HF:{HF_NAME})",
          flush=True)
    return rows


def _load_qasper_rows(split: str, n_docs: int) -> List[Row]:
    """(context, question, answer) triples — local staged jsonl first (ALL rows, already bounded at
    ingest time), else a bounded HF stream of ``n_docs`` papers."""
    local = _local_jsonl(split)
    if local is not None:
        rows: List[Row] = []
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
        origin = f"data/qasper/{local.name}"
    else:
        rows = _iter_hf_rows(split, n_docs)
        origin = f"HF:{HF_NAME}"
    if not rows:
        raise ValueError(f"[qasper] no rows loaded from {origin} (split={split}).")
    n_uns = sum(1 for _c, _q, a in rows if a == UNANSWERABLE)
    print(f"[data.qasper] {split}: {len(rows)} QA pairs from {origin} ({n_uns} unanswerable)",
          flush=True)
    return rows


class QasperSource(Source):
    """Yields Qasper NLP papers as ``QAItem``s: facts = the WHOLE paper's paragraphs (order preserved)."""

    kind = "qa"
    pack_n_queries = (1, 2)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 2000,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed
        self.rows = _load_qasper_rows(split, n_docs)
        # Distractor pool = OTHER papers' paragraphs (cross-paper padding noise).
        pool: List[str] = []
        seen = set()
        for ctx, _q, _a in self.rows:
            for s in _PARA_SPLIT.split(ctx):
                s = s.strip()
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
            facts = [s.strip() for s in _PARA_SPLIT.split(ctx) if s.strip()]
            out.append(QAItem(facts=facts, question=q, answer=a, task_id=0,
                              meta={"dataset": "qasper"}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
