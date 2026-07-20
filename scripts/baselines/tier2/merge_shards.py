#!/usr/bin/env python3
"""Merge sharded M+ runs into ONE combined, deterministically-scored artifact.

Each `run_memoryllm.py --num-shards N --shard-idx K` processes a DISJOINT set of contexts and writes its own
resumable cache (`..._shKofN__*.jsonl`). This concatenates every shard cache for a dataset, dedups by
question_id, scores the union ONCE with the same scorer as a single run, and writes a combined JSON with the
same shape as a normal run (so report.py picks it up). Idempotent; safe to run while shards are still going
(it just scores whatever's present so far → live partial accuracy).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], required=True)
    ap.add_argument("--variant", default="s")
    ap.add_argument("--out-dir", default="outputs/baselines")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--max-examples", type=int, default=None, help="match the run's scoping for coverage pct")
    args = ap.parse_args()

    from src.memory.eval.tier2_common import score_dataset, valid_for_scoring, load_items
    from src.memory.eval.results import ResultStore

    out_dir = REPO / args.out_dir
    pattern = str(out_dir / "cache" / f"{args.dataset}__memoryllm__*_sh*of*__*.jsonl")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        sys.exit(f"[merge] no shard caches matching {pattern}")
    print(f"[merge] {len(shard_files)} shard caches:\n  " + "\n  ".join(Path(f).name for f in shard_files))

    by_qid = {}
    for f in shard_files:
        for r in ResultStore(Path(f)).all_records():
            by_qid[str(r.get("question_id"))] = r          # contexts disjoint → dedup is just belt-and-braces
    records = [r for r in by_qid.values() if valid_for_scoring(r)]
    print(f"[merge] {len(by_qid)} unique records, {len(records)} scoreable")

    agg = score_dataset(args.dataset, records, use_bem=not args.no_bem)
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    n_cutoff = sum(1 for r in by_qid.values() if r.get("finish_reason") == "length")
    n_eos = sum(1 for r in by_qid.values()
                if valid_for_scoring(r) and r.get("finish_reason") != "length")
    meta = {"n": len(by_qid), "n_scored": len(records), "n_shards": len(shard_files),
            "coverage": round(len(records) / len(items), 4) if items else None,
            "n_gen_cutoff": n_cutoff, "n_eos_completed": n_eos,
            "eos_completion_rate": round(n_eos / len(items), 4) if items else None,
            "scoring_policy": "score_length_capped_output",
            "bem_threshold": (0.85 if (not args.no_bem and args.dataset == "longmemeval") else None)}
    payload = {"dataset": args.dataset, "method": "memoryllm", "model": "YuWangX/mplus-8b", "mode": "memoryllm",
               "meta": meta, "aggregate": {k: v for k, v in agg.items() if k != "details"}}
    tag = f"{args.dataset}__memoryllm__mplus-8b__{args.variant}__MERGED"
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    head = f"overall_acc={agg.get('overall_accuracy', float('nan')):.3f}"
    for k in ("task_averaged_accuracy", "n_scored", "n_skipped"):
        if k in agg:
            head += f"  {k}={agg[k]}"
    print(f"[merge] {head}  coverage={meta['coverage']}  → {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
