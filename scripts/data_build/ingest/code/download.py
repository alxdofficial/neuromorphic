#!/usr/bin/env python
"""Ingest a BOUNDED source-code sample → ``data/code/{train,val}.jsonl`` (best-effort).

Streams a small N-file sample from HuggingFace (default ``bigcode/the-stack-smol`` `data/python`
subdir — self-contained, no gating/S3) and writes it as ``{"text": ...}`` jsonl under ``data/code/``
so ``CodeSource`` loads it fully offline. Code = un-guessable exact-recall binding (a def/const bound
early, referenced far later). Kept small — a binding-variety source, not a pretraining corpus.

Usage:
    python scripts/data_build/ingest/code/download.py [--n-train 2000] [--n-val 500]
        [--hf-name bigcode/the-stack-smol] [--data-dir data/python] [--min-chars 400]

If HF is unreachable this exits with a clear error (no partial file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def _text_of(ex: dict) -> str:
    return ex.get("content") or ex.get("text") or ""


def stream_split(hf_name, data_dir, n_docs, skip, min_chars):
    from datasets import load_dataset
    kw = {"split": "train", "streaming": True}
    if data_dir:
        kw["data_dir"] = data_dir
    ds = load_dataset(hf_name, **kw)
    out, seen = [], 0
    for ex in ds:
        t = _text_of(ex)
        if not t or len(t) < min_chars:
            continue
        if seen < skip:
            seen += 1
            continue
        out.append(t)
        if len(out) >= n_docs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-name", default="codeparrot/codeparrot-clean")  # open python parquet (the-stack* are gated)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    out_dir = REPO / "data" / "code"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split(args.hf_name, args.data_dir, args.n_train, 0, args.min_chars)
        val = stream_split(args.hf_name, args.data_dir, args.n_val, args.n_train, args.min_chars)
    except Exception as e:
        raise SystemExit(f"[code ingest] HF {args.hf_name!r} unreachable ({type(e).__name__}: "
                         f"{str(e)[:120]}). Rerun once online.")
    for split, docs in (("train", train), ("val", val)):
        p = out_dir / f"{split}.jsonl"
        with open(p, "w") as fp:
            for t in docs:
                fp.write(json.dumps({"text": t}) + "\n")
        print(f"[code ingest] wrote {len(docs)} docs → {p}")


if __name__ == "__main__":
    main()
