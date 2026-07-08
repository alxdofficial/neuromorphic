#!/usr/bin/env python
"""Ingest a BOUNDED, LONG-trajectory-prioritized SWE-agent sample →
``data/swe_trajectories/{train,val}.jsonl``.

Streams a bounded scan of HuggingFace ``nebius/SWE-agent-trajectories`` (~80k reward-labeled
SWE-agent runs), keeps the system-prompt-free (thought/action/observation) turns as slim
``"Tag: text"`` lines, and stages the LONGEST trajectories (most steps — the interesting
long-horizon-memory cases) as normalized jsonl so ``SweTrajectoriesSource`` loads fully offline
(local jsonl takes priority over any live HF read).

Selection is STRATIFIED by ``target`` (resolved / unresolved) before ranking by length, because
unresolved trajectories run systematically LONGER (agent struggling) than resolved ones (median 41
vs 27 turns in a 3k-row scan) — pure longest-first would starve the "resolved" class and make the
"was the issue resolved?" probe trivially gameable by always answering "no".

STREAMING NOTE: the underlying HF streaming loader hits a benign interpreter-SHUTDOWN crash
(aiohttp async cleanup racing CPython finalization -> ``Fatal Python error: PyGILState_Release``)
that fires AFTER every row is already read and written to disk — harmless to the output files, but
this script calls ``os._exit(0)`` at the very end to skip that flaky teardown path deterministically
(bypassing atexit/GC; everything needed is already flushed). ``SweTrajectoriesSource`` itself avoids
streaming entirely (uses a non-streaming slice) for exactly this reason — killing the whole training
process is not an acceptable way to dodge it there.

Usage:
    python scripts/data_build/ingest/swe_trajectories/download.py [--n-train 3000] [--n-val 500]
                                                                    [--scan-cap 15000] [--min-turns 9]

If HF is unreachable this exits with a clear error (no partial/half file); rerun once online.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HF_NAME = "nebius/SWE-agent-trajectories"
_MAX_CHARS = 2000     # per-turn text cap (some tool outputs run to tens of KB)


def _turn_lines(trajectory: list) -> list:
    """trajectory turns -> slim ``"Tag: text"`` lines (system prompt dropped, text char-capped)."""
    lines = []
    seen_user = False
    for t in trajectory:
        role = t.get("role")
        text = (t.get("text") or "").strip()
        if role == "system" or not text:
            continue
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + " …[truncated]"
        if role == "user":
            tag = "Observation" if seen_user else "Issue"
            seen_user = True
            lines.append(f"{tag}: {text}")
        elif role == "ai":
            lines.append(f"Action: {text}")
    return lines


def _normalize(ex: dict) -> dict:
    lines = _turn_lines(ex["trajectory"])
    return {
        "instance_id": ex.get("instance_id", ""),
        "target": bool(ex.get("target")),
        "exit_status": ex.get("exit_status", ""),
        "n_turns": len(lines),
        "lines": lines,
    }


def scan(n: int):
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        out.append(_normalize(ex))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--scan-cap", type=int, default=15000, help="rows scanned before picking the longest")
    ap.add_argument("--min-turns", type=int, default=9, help="drop degenerate near-instant trajectories")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = REPO / "data" / "swe_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        rows = scan(args.scan_cap)
    except Exception as e:
        raise SystemExit(
            f"[swe_trajectories] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: "
            f"{str(e)[:160]}). Restore network access and rerun.")

    rows = [r for r in rows if r["n_turns"] >= args.min_turns]
    pool_true = sorted((r for r in rows if r["target"]), key=lambda r: -r["n_turns"])
    pool_false = sorted((r for r in rows if not r["target"]), key=lambda r: -r["n_turns"])

    n_total = args.n_train + args.n_val
    n_true_want = n_total // 2
    n_false_want = n_total - n_true_want
    picked = pool_true[:n_true_want] + pool_false[:n_false_want]
    short = n_total - len(picked)          # one stratum ran short -> backfill from the other's spares
    if short > 0:
        spare = pool_true[n_true_want:] + pool_false[n_false_want:]
        picked += spare[:short]

    random.Random(args.seed).shuffle(picked)
    val, train = picked[:args.n_val], picked[args.n_val:args.n_val + args.n_train]

    print(f"[swe_trajectories] scanned {len(rows)}/{args.scan_cap} usable "
          f"({len(pool_true)} resolved / {len(pool_false)} unresolved available)")

    for split, out_rows in (("train", train), ("val", val)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in out_rows:
                fp.write(json.dumps(r) + "\n")
        n_resolved = sum(1 for r in out_rows if r["target"])
        turns = sorted(r["n_turns"] for r in out_rows)
        median = turns[len(turns) // 2] if turns else 0
        print(f"[swe_trajectories] wrote {len(out_rows)} trajectories "
              f"({n_resolved} resolved / {len(out_rows) - n_resolved} unresolved, "
              f"median {median} turns) → {path}")

    sys.stdout.flush()      # os._exit() skips normal buffer flushing — do it explicitly first
    sys.stderr.flush()
    os._exit(0)              # skip the flaky HF-streaming interpreter-shutdown crash (see module docstring)


if __name__ == "__main__":
    main()
