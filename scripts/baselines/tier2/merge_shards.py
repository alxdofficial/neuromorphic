#!/usr/bin/env python3
"""Merge sharded Tier-2 runs into one combined, deterministically-scored artifact.

Each sharded runner processes a DISJOINT set of contexts and writes its own
resumable cache (`..._shKofN__*.jsonl`). This concatenates every shard cache for a dataset, dedups by
question_id, scores the union ONCE with the same scorer as a single run, and writes a combined JSON with the
same shape as a normal run (so report.py picks it up). Idempotent; safe to run while shards are still going
(it just scores whatever's present so far → live partial accuracy).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["longmemeval", "memoryagentbench"], required=True)
    ap.add_argument("--variant", default="s")
    ap.add_argument("--method", choices=["memoryllm", "kvzip"], default="memoryllm")
    ap.add_argument("--model", default=None,
                    help="model metadata for the merged artifact; defaults to the canonical method model")
    ap.add_argument("--out-dir", default="outputs/baselines")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--max-examples", type=int, default=None, help="match the run's scoping for coverage pct")
    ap.add_argument("--shard-glob", default=None,
                    help="override the shard-cache glob (relative to --out-dir). Use this to merge ONLY one "
                         "partitioning, e.g. 'cache/rescue_14way/*_sh*of14__*.jsonl'. The default picks up "
                         "every partitioning present, which is usually NOT what you want — see below.")
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse to write unless every shard and every selected question is present")
    args = ap.parse_args()

    from src.memory.eval.tier2_common import score_dataset, valid_for_scoring, load_items
    from src.memory.eval.results import ResultStore

    out_dir = REPO / args.out_dir
    if args.shard_glob:
        pattern = str(out_dir / args.shard_glob)
    else:
        # Recursive: shard caches rescued off pods may sit in a subdirectory, and a non-recursive glob drops
        # them SILENTLY (no error, just a smaller merge) — that hid 19 unique records once.
        pattern = str(out_dir / "cache" / "**" / f"{args.dataset}__{args.method}__*_sh*of*__*.jsonl")
    shard_files = glob.glob(pattern, recursive=True)
    if not shard_files:
        sys.exit(f"[merge] no shard caches matching {pattern}")

    # Dedup is NOT belt-and-braces once more than one partitioning is present. Contexts are disjoint WITHIN a
    # partitioning, but the same question_id appears across partitionings, and M+'s answer differs between
    # them (different context grouping -> different injection schedule; measured: only 7/20 of the overlapping
    # ids were byte-identical). So the winner must be chosen deterministically, not by glob order: sort oldest
    # -> newest by mtime so the most recent run wins, and report what got overridden.
    shard_files.sort(key=lambda f: (Path(f).stat().st_mtime, f))
    parts = sorted({p.split("_sh")[-1].split("__")[0] for f in shard_files for p in [Path(f).name]})
    print(f"[merge] {len(shard_files)} shard caches across partitioning(s) {parts}:\n  "
          + "\n  ".join(Path(f).name for f in shard_files))
    if len({p.split("of")[-1] for p in parts}) > 1:
        print(f"[merge] ⚠ MULTIPLE partitionings present {parts} — newest-mtime wins per question_id. "
              "Pass --shard-glob to merge only one if that is not what you want.")
    shard_specs = [re.search(r"_sh(\d+)of(\d+)__", Path(path).name) for path in shard_files]
    if any(match is None for match in shard_specs):
        sys.exit("[merge] could not parse shard identity from one or more cache names")
    shard_pairs = [(int(match.group(1)), int(match.group(2))) for match in shard_specs if match]
    if args.require_complete:
        totals = {total for _, total in shard_pairs}
        if len(totals) != 1:
            sys.exit(f"[merge] incomplete: mixed shard totals {sorted(totals)}")
        total = next(iter(totals))
        present = {idx for idx, _ in shard_pairs}
        expected_shards = set(range(total))
        if present != expected_shards:
            sys.exit(f"[merge] incomplete: shard ids present={sorted(present)}, expected={sorted(expected_shards)}")

    by_qid, overridden = {}, 0
    for f in shard_files:
        for r in ResultStore(Path(f)).all_records():
            qid = str(r.get("question_id"))
            if qid in by_qid:
                overridden += 1
            by_qid[qid] = r
    if overridden:
        print(f"[merge] {overridden} duplicate question_id(s) resolved in favour of the newer cache")
        if args.require_complete:
            sys.exit("[merge] duplicate question ids in a supposedly complete single partitioning")
    records = [r for r in by_qid.values() if valid_for_scoring(r)]
    print(f"[merge] {len(by_qid)} unique records, {len(records)} scoreable")

    agg = score_dataset(args.dataset, records, use_bem=not args.no_bem)
    items = load_items(args.dataset, variant=args.variant, max_examples=args.max_examples)
    if args.require_complete:
        expected_ids = {str(item["question_id"]) for item in items}
        actual_ids = set(by_qid)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            sys.exit(f"[merge] incomplete question coverage: missing={len(missing)} {missing[:5]}, "
                     f"extra={len(extra)} {extra[:5]}")
    n_cutoff = sum(1 for r in by_qid.values() if r.get("finish_reason") == "length")
    n_eos = sum(1 for r in by_qid.values()
                if valid_for_scoring(r) and r.get("finish_reason") != "length")
    meta = {"n": len(by_qid), "n_scored": len(records), "n_shards": len(shard_files),
            "coverage": round(len(records) / len(items), 4) if items else None,
            "n_gen_cutoff": n_cutoff, "n_eos_completed": n_eos,
            "eos_completion_rate": round(n_eos / len(items), 4) if items else None,
            "scoring_policy": "score_length_capped_output",
            "bem_threshold": (0.85 if (not args.no_bem and args.dataset == "longmemeval") else None)}
    model = args.model or ({"memoryllm": "YuWangX/mplus-8b",
                            "kvzip": "Qwen/Qwen2.5-7B-Instruct-1M"}[args.method])
    payload = {"dataset": args.dataset, "method": args.method, "model": model, "mode": args.method,
               "meta": meta, "aggregate": {k: v for k, v in agg.items() if k != "details"}}
    tag = f"{args.dataset}__{args.method}__{model.split('/')[-1]}__{args.variant}__MERGED"
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    head = f"overall_acc={agg.get('overall_accuracy', float('nan')):.3f}"
    for k in ("task_averaged_accuracy", "n_scored", "n_skipped"):
        if k in agg:
            head += f"  {k}={agg[k]}"
    print(f"[merge] {head}  coverage={meta['coverage']}  → {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
