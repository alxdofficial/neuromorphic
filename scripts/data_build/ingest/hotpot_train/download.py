#!/usr/bin/env python
"""Ingest a BOUNDED HotpotQA **train-split** sample → ``data/hotpot_train/{train,val}.jsonl``.

Loads the HF ``hotpot_qa``/``distractor`` **train** split (TRAIN-SPLIT FIREWALL — the eval reader
``src/memory/data/hotpot.py`` reads *validation*), normalizes each row to the compact schema
``HotpotTrainSource`` expects, and stages a small ``2·n`` slice as jsonl so the source loads fully
offline (local jsonl takes priority over HF). train = ``train[0:n_train]``; val (a held-out slice of
TRAIN, NOT the reserved eval validation) = ``train[n_train:n_train+n_val]``.

NON-streaming load: loading-script datasets like ``hotpot_qa`` do not stream cleanly (can hang); the
arrow cache is memory-mapped so materializing only the first rows is cheap.

Usage:
    python scripts/data_build/ingest/hotpot_train/download.py [--n-train 3000] [--n-val 500]

If HF is unreachable (and uncached) this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

HF_NAME = "hotpot_qa"
HF_CONFIG = "distractor"


def _normalize(ex: dict) -> dict:
    sf = ex.get("supporting_facts") or {}
    support_titles = set(sf.get("title", []) or [])
    ctx = ex["context"]
    paragraphs = [
        {"title": t, "text": " ".join(s.strip() for s in sl), "support": t in support_titles}
        for t, sl in zip(ctx["title"], ctx["sentences"])
    ]
    return {
        "question": (ex["question"] or "").strip(),
        "answer": (ex["answer"] or "").strip(),
        "aliases": [],
        "paragraphs": paragraphs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    args = ap.parse_args()

    out_dir = REPO / "data" / "hotpot_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, HF_CONFIG, split="train")   # TRAIN firewall (eval reads validation)
    except Exception as e:
        raise SystemExit(
            f"[hotpot_train] HF dataset {HF_NAME!r}/{HF_CONFIG!r} unreachable "
            f"({type(e).__name__}: {str(e)[:160]}). Restore network access and rerun.")

    n = len(ds)
    train = [_normalize(ds[i]) for i in range(0, min(args.n_train, n))]
    val = [_normalize(ds[i]) for i in range(args.n_train, min(args.n_train + args.n_val, n))]

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        print(f"[hotpot_train] wrote {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
