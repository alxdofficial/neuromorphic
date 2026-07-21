#!/usr/bin/env python3
"""Merge completed A-MEM shard artifacts into one scored result with summed token accounting.

Pass the aggregate JSON files written by `run_agentmem.py --num-shards N --shard-idx K`, not their cache
JSONLs. The merger validates that protocol/model/shard metadata agree before combining anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _sum_usage(payloads: list[dict]) -> dict:
    phases: dict[str, dict] = {}
    keys = ("calls", "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reported_cost_usd")
    for payload in payloads:
        for phase, row in (payload.get("meta", {}).get("token_usage") or {}).items():
            if phase == "TOTAL":
                continue
            dst = phases.setdefault(phase, {k: 0 for k in keys})
            for key in keys:
                dst[key] += row.get(key, 0) or 0
    phases["TOTAL"] = {key: sum(row[key] for row in phases.values()) for key in keys}
    return phases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="+", type=Path, help="A-MEM per-shard aggregate JSON files")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="original pre-sharding selection size; omit for the full dataset")
    ap.add_argument("--no-bem", action="store_true")
    ap.add_argument("--out-dir", default="outputs/baselines")
    args = ap.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.artifacts]
    first = payloads[0]
    if first.get("method") != "a-mem":
        ap.error(f"expected A-MEM artifacts, got method={first.get('method')!r}")
    expected = {
        "dataset": first.get("dataset"), "model": first.get("model"),
        "variant": first.get("meta", {}).get("variant"),
        "retrieve_k": first.get("meta", {}).get("retrieve_k"),
        "max_new_tokens": first.get("meta", {}).get("max_new_tokens"),
        "protocol": first.get("meta", {}).get("a_mem_protocol"),
        "upstream_commit": first.get("meta", {}).get("upstream_commit"),
        "num_shards": first.get("meta", {}).get("num_shards"),
    }
    shard_ids = set()
    for path, payload in zip(args.artifacts, payloads):
        meta = payload.get("meta", {})
        got = {
            "dataset": payload.get("dataset"), "model": payload.get("model"),
            "variant": meta.get("variant"), "retrieve_k": meta.get("retrieve_k"),
            "max_new_tokens": meta.get("max_new_tokens"), "protocol": meta.get("a_mem_protocol"),
            "upstream_commit": meta.get("upstream_commit"), "num_shards": meta.get("num_shards"),
        }
        if got != expected:
            ap.error(f"incompatible shard artifact: {path}")
        shard = meta.get("shard_idx")
        if shard in shard_ids:
            ap.error(f"duplicate shard_idx={shard}: {path}")
        shard_ids.add(shard)
    num_shards = expected["num_shards"]
    if not isinstance(num_shards, int) or num_shards < 2 or shard_ids != set(range(num_shards)):
        ap.error(f"need exactly shards 0..{num_shards - 1 if isinstance(num_shards, int) else '?'}; "
                 f"received {sorted(shard_ids)}")

    from src.memory.eval.results import ResultStore
    from src.memory.eval.tier2_common import finalize, load_items

    fingerprint = hashlib.md5("\n".join(sorted(str(p.resolve()) for p in args.artifacts)).encode()).hexdigest()[:8]
    model_slug = expected["model"].split("/")[-1]
    tag = (f"{expected['dataset']}__a-mem__{model_slug}__{expected['variant']}__MERGED"
           f"-sh{num_shards}__{fingerprint}")
    out_dir = REPO / args.out_dir
    store = ResultStore(out_dir / "cache" / f"{tag}.jsonl")
    by_qid = {}
    for payload in payloads:
        for row in ResultStore(Path(payload["store"])).all_records():
            qid = str(row.get("question_id"))
            if qid in by_qid:
                ap.error(f"question {qid} occurs in multiple context shards")
            by_qid[qid] = row
    for row in by_qid.values():
        store.append(row)

    items = load_items(expected["dataset"], variant=expected["variant"], max_examples=args.max_examples)
    finalize(expected["dataset"], "a-mem", expected["model"], items, store, use_bem=not args.no_bem,
             extra_meta={**{k: v for k, v in expected.items() if k not in ("dataset", "model", "protocol")},
                         "a_mem_protocol": expected["protocol"], "merged_shards": sorted(shard_ids),
                         "token_usage": _sum_usage(payloads)},
             out_dir=out_dir, tag=tag, log_prefix="[merge_agentmem]")


if __name__ == "__main__":
    main()
