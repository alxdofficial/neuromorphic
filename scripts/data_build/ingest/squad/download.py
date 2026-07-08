#!/usr/bin/env python
"""Ingest a BOUNDED SQuAD 2.0 sample → ``data/squad/{train,val}.jsonl`` (best-effort).

Streams a small N-example sample from HuggingFace (``rajpurkar/squad_v2``) and stages it as
``{"context","question","answer"}`` jsonl so ``SquadSource`` can load it fully offline (local jsonl
takes priority over HF streaming). Unanswerable (v2, empty answers) examples are kept with the literal
``"unanswerable"`` target — the retrieve-vs-abstain signal.

Usage:
    python scripts/data_build/ingest/squad/download.py [--n-train 4000] [--n-val 500]

If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "rajpurkar/squad_v2"
UNANSWERABLE = "unanswerable"


def _answer_of(answers: dict) -> str:
    texts = (answers or {}).get("text") or []
    if texts and (texts[0] or "").strip():
        return texts[0].strip()
    return UNANSWERABLE


def stream_split(hf_split: str, n_docs: int):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split=hf_split, streaming=True)
    out = []
    for ex in ds:
        ctx = (ex.get("context") or "").strip()
        q = (ex.get("question") or "").strip()
        if not ctx or not q:
            continue
        out.append({"context": ctx, "question": q, "answer": _answer_of(ex.get("answers"))})
        if len(out) >= n_docs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    args = ap.parse_args()

    out_dir = REPO / "data" / "squad"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split("train", args.n_train)
        val = stream_split("validation", args.n_val)
    except Exception as e:
        raise SystemExit(
            f"[squad] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        n_uns = sum(1 for r in rows if r["answer"] == UNANSWERABLE)
        print(f"[squad] wrote {len(rows)} examples ({n_uns} unanswerable) → {path}")


if __name__ == "__main__":
    main()
