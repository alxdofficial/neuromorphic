#!/usr/bin/env python
"""Ingest a BOUNDED QuALITY sample -> ``data/quality/{train,val}.jsonl`` (best-effort).

Streams N QuALITY rows from HuggingFace (``emozilla/quality``) and writes them as slim
``{"article","question","options","answer"}`` jsonl under ``data/quality/`` so ``QualitySource`` can
load them fully offline (local jsonl takes priority over HF streaming). QuALITY = long-document
multiple-choice reading comprehension (LONG-CONTEXT regime; articles are ~4.7k-7.6k tokens).

``answer`` is kept as-is (0-indexed for emozilla/quality). ``emozilla/quality`` exposes real ``train``
and ``validation`` splits, so val is a genuine held-out slice (not a skip-carved one).

Usage:
    python scripts/data_build/ingest/quality/download.py [--n-train 1500] [--n-val 300]
                                                          [--hf-name emozilla/quality]

If HF is unreachable this exits with a clear error (no partial file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def stream_split(hf_name, hf_split, n_docs):
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=hf_split, streaming=True)
    out = []
    for ex in ds:
        art = (ex.get("article") or "").strip()
        q = (ex.get("question") or "").strip()
        opts = list(ex.get("options") or [])
        ans = ex.get("answer")
        if not art or not q or not opts or ans is None:
            continue
        out.append({"article": art, "question": q, "options": opts, "answer": int(ans)})
        if len(out) >= n_docs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-name", default="emozilla/quality")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=300)
    args = ap.parse_args()

    out_dir = REPO / "data" / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = stream_split(args.hf_name, "train", args.n_train)
        val = stream_split(args.hf_name, "validation", args.n_val)
    except Exception as e:
        raise SystemExit(
            f"[quality] HF dataset {args.hf_name!r} unreachable ({type(e).__name__}: {str(e)[:160]}). "
            f"Restore network access and rerun.")

    for split, docs in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in docs:
                fp.write(json.dumps(r) + "\n")
        print(f"[quality] wrote {len(docs)} rows -> {path}")


if __name__ == "__main__":
    main()
