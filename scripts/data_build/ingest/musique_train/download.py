#!/usr/bin/env python
"""Ingest a BOUNDED MuSiQue-Ans **train-split** sample → ``data/musique_train/{train,val}.jsonl``.

Loads the HF ``dgslibisey/MuSiQue`` **train** split (TRAIN-SPLIT FIREWALL — the eval reader
``src/memory/data/musique.py`` reads *validation*), keeps answerable-only rows (mirrors the eval
reader), normalizes each row to the compact schema ``MusiqueTrainSource`` expects, and stages a small
``2·n`` slice as jsonl so the source loads fully offline (local jsonl takes priority over HF).
train = ``train[0:n_train]``; val (a held-out slice of TRAIN, NOT the reserved eval validation) =
``train[n_train:n_train+n_val]``.

Usage:
    python scripts/data_build/ingest/musique_train/download.py [--n-train 3000] [--n-val 500]

If HF is unreachable (and uncached) this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

HF_NAME = "dgslibisey/MuSiQue"


def _normalize(ex: dict) -> dict:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    args = ap.parse_args()

    out_dir = REPO / "data" / "musique_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_NAME, split="train")              # TRAIN firewall (eval reads validation)
        if "answerable" in ds.column_names:                    # mirror the eval reader's filter
            ds = ds.filter(lambda ex: ex["answerable"])
    except Exception as e:
        raise SystemExit(
            f"[musique_train] HF dataset {HF_NAME!r} unreachable "
            f"({type(e).__name__}: {str(e)[:160]}). Restore network access and rerun.")

    n = len(ds)
    train = [_normalize(ds[i]) for i in range(0, min(args.n_train, n))]
    val = [_normalize(ds[i]) for i in range(args.n_train, min(args.n_train + args.n_val, n))]

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        print(f"[musique_train] wrote {len(rows)} rows → {path}")


if __name__ == "__main__":
    main()
