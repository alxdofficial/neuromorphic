#!/usr/bin/env python
"""Ingest a BOUNDED PG19 sample -> ``data/pg19/{train,val}.jsonl`` (best-effort).

Streams a small N-book sample of full-length Project Gutenberg books from the parquet-native
``emozilla/pg19`` mirror (``deepmind/pg19``'s own loading script is unsupported in ``datasets`` >= 4.x --
its loader fetches raw text from a GCS bucket at runtime, so HF can't auto-convert it to parquet either)
and writes ``{"text": ...}`` jsonl under ``data/pg19/`` so ``PG19Source`` can load it fully offline.
Bounded to a SMALL n (default 500 train / 100 val) -- PG19 books are FULL LENGTH (tens to hundreds of
thousands of words each), so even a few hundred is already a sizeable local cache; this is not meant as
a full-corpus mirror.

Train/val here come from the mirror's REAL upstream train/validation splits (genuine held-out books) --
better than ``PG19Source``'s live-HF fallback, which (sharing ``_corpus.py``'s single-split loader with
Pile/RedPajama) can only skip-carve a pseudo-val out of train. Ingesting once therefore upgrades val to
a true held-out split.

Usage:
    python scripts/data_build/ingest/pg19/download.py [--n-train 500] [--n-val 100]
                                                       [--hf-name emozilla/pg19] [--min-chars 20000]

If HF is unreachable this exits with a clear error (no partial file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def stream_split(hf_name: str, hf_split: str, n_docs: int, min_chars: int):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=hf_split, streaming=True)
    out = []
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if not t or len(t) < min_chars:
            continue
        out.append(t)
        if len(out) >= n_docs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-name", default="emozilla/pg19")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--min-chars", type=int, default=20000)   # keep genuinely FULL-length books
    args = ap.parse_args()

    out_dir = REPO / "data" / "pg19"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split(args.hf_name, "train", args.n_train, args.min_chars)
        val = stream_split(args.hf_name, "validation", args.n_val, args.min_chars)
    except Exception as e:
        raise SystemExit(
            f"[pg19] HF dataset {args.hf_name!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, docs in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for t in docs:
                fp.write(json.dumps({"text": t}) + "\n")
        print(f"[pg19] wrote {len(docs)} books -> {path}")


if __name__ == "__main__":
    main()
