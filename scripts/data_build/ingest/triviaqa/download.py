#!/usr/bin/env python
"""Ingest a BOUNDED TriviaQA (rc) sample → ``data/triviaqa/{train,val}.jsonl`` (best-effort).

Streams a small N-example sample from HuggingFace (``mandarjoshi/trivia_qa`` config ``rc``) and stages
it as ``{"context","question","answer","aliases"}`` jsonl so ``TriviaQASource`` can load it fully
offline. Only examples whose evidence (wiki / web search context) actually CONTAINS the answer value
are kept, and the staged ``context`` is a char-window AROUND the answer occurrence (so the on-disk
sample stays small); the source re-windows/token-caps it on load. ``aliases`` are staged for EM.

Usage:
    python scripts/data_build/ingest/triviaqa/download.py [--n-train 3000] [--n-val 500]

If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "mandarjoshi/trivia_qa"
HF_CONFIG = "rc"
CHAR_WINDOW = 3200          # chars kept each side of the answer occurrence in the staged context


def _evidence_texts(ex: dict):
    ep = ex.get("entity_pages") or {}
    sr = ex.get("search_results") or {}
    return list(ep.get("wiki_context") or []) + list(sr.get("search_context") or [])


def _staged_row(ex: dict):
    q = (ex.get("question") or "").strip()
    ans = ex.get("answer") or {}
    value = (ans.get("value") or "").strip()
    aliases = [a for a in (ans.get("aliases") or []) if a]
    if not q or not value:
        return None
    vl = value.lower()
    for text in _evidence_texts(ex):
        p = (text or "").lower().find(vl)
        if p >= 0:
            start = max(0, p - CHAR_WINDOW)
            end = min(len(text), p + len(value) + CHAR_WINDOW)
            return {"context": text[start:end].strip(), "question": q,
                    "answer": value, "aliases": aliases}
    return None


def stream_split(hf_split: str, n_docs: int):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, HF_CONFIG, split=hf_split, streaming=True)
    out, scanned = [], 0
    scan_cap = max(50 * n_docs, 5000)
    for ex in ds:
        scanned += 1
        r = _staged_row(ex)
        if r is not None:
            out.append(r)
        if len(out) >= n_docs or scanned >= scan_cap:
            break
    return out, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    args = ap.parse_args()

    out_dir = REPO / "data" / "triviaqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train, n_tr = stream_split("train", args.n_train)
        val, n_va = stream_split("validation", args.n_val)
    except Exception as e:
        raise SystemExit(
            f"[triviaqa] HF dataset {HF_NAME!r}/{HF_CONFIG} unreachable ({type(e).__name__}: "
            f"{str(e)[:160]}). Restore network access and rerun.")

    for split, rows, scanned in (("train", train, n_tr), ("val", val, n_va)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        print(f"[triviaqa] wrote {len(rows)}/{scanned} scanned (answer-in-evidence) → {path}")


if __name__ == "__main__":
    main()
