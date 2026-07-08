#!/usr/bin/env python
"""Ingest a BOUNDED BabiLong-train sample → ``data/babilong_train/{train,val}.jsonl`` (~2000 rows).

Streams a small sample from HuggingFace ``RMT-team/babilong-train-5k-samples`` — bAbI facts scattered
in real PG-19 prose — spread evenly across a ``configs`` (length) x ``tasks`` (qa1-qa10) grid, and
stages it as ``{"config","task","input","question","answer"}`` jsonl so ``BabilongTrainSource`` loads
it fully offline (local jsonl takes priority over HF streaming).

NAMING GOTCHA: this is NOT the eval BabiLong reader's dataset. ``src/memory/data/babilong.py`` reads
HF ``RMT-team/babilong`` (the held-out benchmark); this ingests the separate ``-train-5k-samples``
repo the BabiLong authors ship specifically as non-eval training fuel — different HF dataset ids, so
this cannot leak into eval.

val = the NEXT disjoint slice of each config/task stream after the train slice (a held-out slice of
this same train-only dump — analogous to ``hotpot_train``'s internal "val", not the reserved eval set).

Usage:
    python scripts/data_build/ingest/babilong_train/download.py [--n-train 2000] [--n-val 300] \\
        [--configs 1k 4k 8k] [--tasks qa1 qa2 qa3 qa7 qa8]

If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "RMT-team/babilong-train-5k-samples"
ALL_TASKS = [f"qa{i}" for i in range(1, 11)]        # only qa1-qa10 exist in this dump
# "1k" excluded — broken on the HF repo itself (its data files 404 for every split; verified 2026-07-08).
DEFAULT_CONFIGS = ["2k", "4k", "8k"]


def stream_grid(configs, tasks, n_docs: int, skip_each: int = 0):
    """Stream ``n_docs`` rows total, spread evenly across the configs x tasks grid; ``skip_each`` rows
    are skipped at the start of every combo's stream (so val can take the NEXT disjoint slice)."""
    from datasets import load_dataset

    combos = [(c, t) for c in configs for t in tasks]
    per_combo = max(1, -(-n_docs // len(combos)))     # ceil-divide, spread evenly
    out = []
    for cfg, task in combos:
        ds = load_dataset(HF_NAME, cfg, split=task, streaming=True)
        seen = got = 0
        for ex in ds:
            if seen < skip_each:
                seen += 1
                continue
            inp = (ex.get("input") or "").strip()
            q = (ex.get("question") or "").strip()
            a = (ex.get("target") or "").strip()
            if inp and q and a:
                out.append({"config": cfg, "task": task, "input": inp, "question": q, "answer": a})
                got += 1
            if got >= per_combo:
                break
        if len(out) >= n_docs:
            break
    return out[:n_docs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                     help="BabiLong length configs to mix, e.g. 0k 1k 2k 4k 8k 16k 32k")
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS, help="bAbI task splits, e.g. qa1 qa2 qa3")
    args = ap.parse_args()

    out_dir = REPO / "data" / "babilong_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # val skips the train slice's rows within each combo's stream, so the two sets never overlap
        # even though both come from the same HF split (no reserved val split exists on HF here).
        per_combo_train = max(1, -(-args.n_train // (len(args.configs) * len(args.tasks))))
        train = stream_grid(args.configs, args.tasks, args.n_train, skip_each=0)
        val = stream_grid(args.configs, args.tasks, args.n_val, skip_each=per_combo_train)
    except Exception as e:
        raise SystemExit(
            f"[babilong_train] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: "
            f"{str(e)[:160]}). Restore network access and rerun.")

    for split, rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        by_cfg = {}
        for r in rows:
            by_cfg[r["config"]] = by_cfg.get(r["config"], 0) + 1
        print(f"[babilong_train] wrote {len(rows)} rows → {path}  (by config: {by_cfg})")


if __name__ == "__main__":
    main()
