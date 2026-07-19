#!/usr/bin/env python3
"""Show per-run progress from the resumable result stores: which questions are answered, failed, correct.

Scans outputs/baselines/cache/*.jsonl and prints one row per (dataset, model, mode) with counts + accuracy,
and lists the errored question_ids (which a rerun of run_api_eval.py will automatically retry). Use this to
see coverage, spot failed questions, and confirm a sweep is complete before reporting.

Usage:
  python scripts/baselines/status.py                     # all runs
  python scripts/baselines/status.py --glob 'longmemeval__*' --show-errors
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys_path = str(REPO)
import sys  # noqa: E402
sys.path.insert(0, sys_path)
from src.memory.eval.results import ResultStore  # noqa: E402

CACHE = REPO / "outputs" / "baselines" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--show-errors", action="store_true", help="list errored question_ids per run")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(CACHE / args.glob)))
    if not paths:
        print(f"(no result stores under {CACHE})")
        return
    print(f"{'run':60} {'answ':>6} {'err':>5} {'scored':>7} {'corr':>6} {'acc':>7}")
    print("-" * 96)
    tot = {"n_answered": 0, "n_errored": 0, "n_scored": 0, "n_correct": 0}
    for p in paths:
        s = ResultStore(p); summ = s.summary()
        name = Path(p).stem
        acc = f"{summ['accuracy']:.3f}" if summ["accuracy"] is not None else "—"
        print(f"{name:60} {summ['n_answered']:>6} {summ['n_errored']:>5} "
              f"{summ['n_scored']:>7} {summ['n_correct']:>6} {acc:>7}")
        for k in tot:
            tot[k] += summ[k]
        if args.show_errors and summ["errored_ids"]:
            print(f"    errored ({len(summ['errored_ids'])}): {summ['errored_ids'][:20]}"
                  + (" ..." if len(summ["errored_ids"]) > 20 else ""))
    print("-" * 96)
    acc = f"{tot['n_correct'] / tot['n_scored']:.3f}" if tot["n_scored"] else "—"
    print(f"{'TOTAL':60} {tot['n_answered']:>6} {tot['n_errored']:>5} "
          f"{tot['n_scored']:>7} {tot['n_correct']:>6} {acc:>7}")
    if tot["n_errored"]:
        print(f"\n{tot['n_errored']} errored question(s) — rerun run_api_eval.py with the same args to retry them.")


if __name__ == "__main__":
    main()
