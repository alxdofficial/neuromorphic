#!/usr/bin/env python
"""Ingest a BOUNDED RedPajama sample → ``data/redpajama/{train,val}.jsonl`` (best-effort).

Streams a small N-doc sample of natural text from HuggingFace (default ``DKYoon/SlimPajama-6B`` —
SlimPajama is the deduplicated RedPajama mixture, parquet-native; the classic
``togethercomputer/RedPajama-Data-1T-Sample`` is a loading-script dataset unsupported by
``datasets`` ≥ 4.x) and writes it as ``{"text": ...}`` jsonl under ``data/redpajama/`` so
``RedpajamaSource`` can then load it fully offline (local jsonl takes priority over HF streaming).
Kept intentionally small — bucket-1 text for the natural-text/KL objective, not a full corpus.

Usage:
    python scripts/data_build/ingest/redpajama/download.py [--n-train 4000] [--n-val 500]
        [--hf-name DKYoon/SlimPajama-6B]

If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def _text_of(ex: dict) -> str:
    return ex.get("text") or ex.get("content") or ""


def stream_split(hf_name, hf_config, n_docs, skip, min_chars):
    from datasets import load_dataset
    ds = (load_dataset(hf_name, hf_config, split="train", streaming=True)
          if hf_config else load_dataset(hf_name, split="train", streaming=True))
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
    ap.add_argument("--hf-name", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--min-chars", type=int, default=512)
    args = ap.parse_args()

    out_dir = REPO / "data" / "redpajama"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split(args.hf_name, args.hf_config, args.n_train, 0, args.min_chars)
        val = stream_split(args.hf_name, args.hf_config, args.n_val, args.n_train, args.min_chars)
    except Exception as e:
        raise SystemExit(
            f"[redpajama] HF dataset {args.hf_name!r} unreachable "
            f"({type(e).__name__}: {str(e)[:160]}). Restore network access and rerun.")

    for split, docs in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for t in docs:
                fp.write(json.dumps({"text": t}) + "\n")
        print(f"[redpajama] wrote {len(docs)} docs → {path}")


if __name__ == "__main__":
    main()
