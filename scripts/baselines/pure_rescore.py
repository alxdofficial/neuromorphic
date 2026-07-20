#!/usr/bin/env python3
"""Regenerate ALL Tier-1 aggregate JSONs from the existing per-question caches with the CURRENT scorer — no
API, no generation (pure rescore). Determines mode from the store filename and the RAG k from the stored
retrieved_idx length, writes aggregates in the report_mode/config_sig naming, then deletes stale pre-fix
aggregates (whose meta lacks config_sig). Coverage denominator = distinct question_ids in the store (== the
full selection once a run is complete)."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.memory.eval import score_longmemeval, score_memoryagentbench       # noqa: E402
from src.memory.eval.tier2_common import valid_for_scoring                   # noqa: E402

OUT = "outputs/baselines"


def _proj(r):
    return {"question": r.get("question"), "answer": r.get("gold"), "hypothesis": r.get("hypothesis"),
            "question_type": r.get("question_type"), "question_id": r.get("question_id"),
            "source": r.get("source"), "metric": r.get("metric"), "competency": r.get("competency")}


def main():
    made = []
    for f in sorted(glob.glob(f"{OUT}/cache/*.jsonl")):
        name = os.path.basename(f)[:-6]
        parts = name.split("__")
        if len(parts) != 4:
            continue
        dataset, modelslug, mode, sig = parts
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if not recs:
            continue
        model = recs[0].get("model", modelslug)
        report_mode = mode
        if mode.startswith("rag"):
            ks = [len(r["retrieved_idx"]) for r in recs if r.get("retrieved_idx")]
            k = max(set(ks), key=ks.count) if ks else 5
            report_mode = f"{mode}_k{k}"
        valid = [_proj(r) for r in recs if valid_for_scoring(r)]
        scorer = score_longmemeval if dataset == "longmemeval" else score_memoryagentbench
        agg = scorer(valid, use_bem=(dataset == "longmemeval"))
        n_items = len({r.get("question_id") for r in recs})
        meta = {"n": len(recs), "n_scored": len(valid),
                "coverage": round(len(valid) / n_items, 4) if n_items else None,
                "bem_threshold": (0.85 if dataset == "longmemeval" else None),
                "config_sig": sig, "rescored": True}
        payload = {"dataset": dataset, "model": model, "mode": report_mode, "meta": meta,
                   "aggregate": {k2: v for k2, v in agg.items() if k2 != "details"}, "store": f}
        open(f"{OUT}/{dataset}__{modelslug}__{report_mode}__{sig}.json", "w").write(json.dumps(payload, indent=1))
        made.append((f"{dataset}__{modelslug}__{report_mode}", agg.get("overall_accuracy"), meta["coverage"]))

    deln = 0
    for f in glob.glob(f"{OUT}/*.json"):
        b = os.path.basename(f)
        if b.endswith(("_api_summary.json", "report.json", "judge_crosscheck.json")):
            continue
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "config_sig" not in (d.get("meta") or {}):
            os.remove(f); deln += 1

    print(f"[pure_rescore] {len(made)} aggregates; deleted {deln} stale")
    for t, a, c in sorted(made):
        print(f"  {a if a is None else round(a, 3)}  cov={c}  {t}")


if __name__ == "__main__":
    main()
