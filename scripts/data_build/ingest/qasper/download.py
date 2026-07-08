#!/usr/bin/env python
"""Ingest a BOUNDED Qasper sample -> ``data/qasper/{train,val}.jsonl`` (best-effort).

Loads ``allenai/qasper`` (via HF's auto-converted parquet mirror, ``revision="refs/convert/parquet"``
-- the dataset's own loading script is unsupported in ``datasets`` >= 4.x), shuffles, and explodes each
paper into one row per question: ``{"context","question","answer","paper_id"}`` jsonl, where
``context = "\\n\\n".join(paragraphs)`` (the paper's full text, paragraph breaks preserved) so
``QasperSource`` can load it fully offline (local jsonl takes priority over HF streaming).

Qasper questions are written from the paper's title+abstract ONLY (the annotator never read the body),
so this QA is non-gist-gameable by construction: the context each row carries is the WHOLE paper.

Usage:
    python scripts/data_build/ingest/qasper/download.py [--n-train 2000] [--n-val 400]

``allenai/qasper`` train/validation only have 888/281 papers total, so an n above that just loads the
full split. If HF is unreachable this exits with a clear error (no partial file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "allenai/qasper"
UNANSWERABLE = "unanswerable"


def _answer_of(ans: dict) -> str:
    """Qasper per-annotator answer dict -> a scorable string: unanswerable > extractive span(s) >
    yes/no > free-form (abstractive) fallback."""
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


def _flatten_paragraphs(full_text: dict):
    paras = []
    for sec_paras in (full_text.get("paragraphs") or []):
        for p in sec_paras:
            p = (p or "").strip()
            if p:
                paras.append(p)
    return paras


def stream_split(hf_split: str, n_docs: int):
    from datasets import load_dataset
    # Qasper is small (888/281/416 papers) -- load the FULL split and shuffle before taking n_docs,
    # matching the squad ingest convention (avoids any order bias in the raw dump).
    ds = load_dataset(HF_NAME, split=hf_split, revision="refs/convert/parquet").shuffle(seed=42)
    out = []
    n_papers = 0
    for ex in ds:
        paras = _flatten_paragraphs(ex.get("full_text") or {})
        if not paras:
            continue
        context = "\n\n".join(paras)
        qas = ex.get("qas") or {}
        for q, ans_block in zip(qas.get("question") or [], qas.get("answers") or []):
            q = (q or "").strip()
            worker_answers = (ans_block or {}).get("answer") or []
            if not q or not worker_answers:
                continue
            out.append({"context": context, "question": q, "answer": _answer_of(worker_answers[0]),
                        "paper_id": ex.get("id", "")})
        n_papers += 1
        if n_papers >= n_docs:
            break
    return out, n_papers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    args = ap.parse_args()

    out_dir = REPO / "data" / "qasper"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train, n_train_papers = stream_split("train", args.n_train)
        val, n_val_papers = stream_split("validation", args.n_val)
    except Exception as e:
        raise SystemExit(
            f"[qasper] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, rows, n_papers in (("train", train, n_train_papers), ("val", val, n_val_papers)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        n_uns = sum(1 for r in rows if r["answer"] == UNANSWERABLE)
        print(f"[qasper] wrote {len(rows)} QA pairs from {n_papers} papers "
              f"({n_uns} unanswerable) -> {path}")


if __name__ == "__main__":
    main()
