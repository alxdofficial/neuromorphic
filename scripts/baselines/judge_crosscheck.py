#!/usr/bin/env python3
"""One-time LLM-judge cross-check — quantify the deterministic↔judge OFFSET on ~N LongMemEval items.

Samples already-scored records from a deterministic run and re-grades them with a GPT-4o judge (the metric
the LongMemEval paper uses), so we can state how far our deterministic scorer sits BELOW the judge. Cheap:
the judge prompt is only {question, gold, hypothesis} — NOT the 115k history. This is NOT panel scoring —
it's a calibration you run once; the primary numbers stay deterministic.

Run under the CPU cap:  scripts/cpu_capped.sh -- .venv/bin/python scripts/baselines/judge_crosscheck.py ...
Needs OPENROUTER_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.memory.eval.api_client import OpenRouterClient  # noqa: E402

# TASK-SPECIFIC LongMemEval judge prompts (audit #4). The official evaluate_qa.py applies DIFFERENT rubrics
# per question type — a single generic "contains the answer" prompt cannot calibrate temporal /
# knowledge-update / preference / abstention questions. These mirror the repo's per-type rules.
_HEAD = "I will give you a question, a correct answer, and a model's response.\n\n"
_TAIL = "\n\nQuestion: {q}\nCorrect answer: {gold}\nModel response: {hyp}\n\nAnswer yes or no only."
_QA_JUDGE = (_HEAD + "Answer 'yes' if the response contains or conveys the correct answer, 'no' otherwise."
             + _TAIL)
_TEMPORAL_JUDGE = (_HEAD + "This is a TEMPORAL-REASONING question. Answer 'yes' if the response conveys the "
                   "correct answer with the right time/order; a SMALL date discrepancy (e.g. off by one day) "
                   "is still 'yes' (matches the official rubric). A clearly wrong date/order is 'no'." + _TAIL)
_KU_JUDGE = (_HEAD + "This is a KNOWLEDGE-UPDATE question — the correct answer is the MOST RECENT value. "
             "Answer 'yes' only if the response reflects the updated (latest) value, 'no' if it gives a "
             "stale/earlier value." + _TAIL)
_PREF_JUDGE = (_HEAD + "This is a PREFERENCE question — answer 'yes' if the response is CONSISTENT with the "
               "user's stated preference (the correct answer describes that preference), 'no' if it "
               "contradicts or ignores it." + _TAIL)
_ABS_JUDGE = ("I will give you an UNANSWERABLE question and a model's response. Answer 'yes' if the model "
              "correctly identifies it as unanswerable (says it lacks the information / cannot answer), 'no' "
              "if it fabricates an answer.\n\nQuestion: {q}\nModel response: {hyp}\n\nAnswer yes or no only.")
# GPT-4o list price (not in the deterministic PRICING table); $/token in,out.
_GPT4O_PRICE = (2.50e-6, 10.0e-6)


def _judge_prompt(rec: dict) -> str:
    qt = (rec.get("question_type") or "").lower()
    if "abstention" in qt or str(rec.get("question_id", "")).endswith("_abs"):
        return _ABS_JUDGE.format(q=rec.get("question", ""), hyp=rec.get("hypothesis", ""))
    if "temporal" in qt:
        tmpl = _TEMPORAL_JUDGE
    elif "knowledge-update" in qt or "knowledge_update" in qt:
        tmpl = _KU_JUDGE
    elif "preference" in qt:
        tmpl = _PREF_JUDGE
    else:
        tmpl = _QA_JUDGE
    return tmpl.format(q=rec.get("question", ""), gold=rec.get("gold", ""), hyp=rec.get("hypothesis", ""))


def _load_scored(pattern: str) -> list[dict]:
    seen: dict = {}
    for f in glob.glob(str(REPO / "outputs" / "baselines" / "cache" / pattern)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # only rows that WE actually scored (a complete answer with a deterministic verdict)
            if r.get("error") or r.get("finish_reason") in ("error", "content_filter") or r.get("correct") is None:
                continue
            seen[(r.get("model"), r.get("mode"), r.get("question_id"))] = r   # last-wins dedup
    return list(seen.values())


async def _run(args) -> None:
    recs = _load_scored(args.glob)
    random.Random(0).shuffle(recs)
    recs = recs[:args.n]
    if not recs:
        sys.exit(f"[judge] no scored records matched {args.glob}")
    print(f"[judge] {len(recs)} items from '{args.glob}', judge={args.judge_model}")

    async with OpenRouterClient(concurrency=args.concurrency) as client:
        async def one(r):
            res = await client.chat(args.judge_model, [{"role": "user", "content": _judge_prompt(r)}], max_tokens=4)
            verdict = (res.text or "").strip().lower().startswith("y")
            return {"model": r.get("model"), "mode": r.get("mode"), "question_type": r.get("question_type"),
                    "question_id": r.get("question_id"), "deterministic": bool(r.get("correct")),
                    "judge": verdict, "err": res.error, "pin": res.prompt_tokens, "pout": res.completion_tokens}
        out = await asyncio.gather(*[one(r) for r in recs])

    ok = [o for o in out if not o["err"]]
    n = len(ok)
    det_acc = sum(o["deterministic"] for o in ok) / n
    jud_acc = sum(o["judge"] for o in ok) / n
    agree = sum(o["deterministic"] == o["judge"] for o in ok) / n
    fn = sum(1 for o in ok if not o["deterministic"] and o["judge"])   # we said WRONG, judge said RIGHT
    fp = sum(1 for o in ok if o["deterministic"] and not o["judge"])   # we said RIGHT, judge said WRONG
    cost = sum(o["pin"] for o in ok) * _GPT4O_PRICE[0] + sum(o["pout"] for o in ok) * _GPT4O_PRICE[1]

    print(f"\n=== JUDGE CROSS-CHECK (n={n}, judge={args.judge_model}) ===")
    print(f"  deterministic accuracy : {det_acc:.3f}")
    print(f"  judge accuracy         : {jud_acc:.3f}")
    print(f"  OFFSET (judge − deterministic) : {jud_acc - det_acc:+.3f}   <- add this to read our numbers on the judge scale")
    print(f"  agreement              : {agree:.1%}")
    print(f"  deterministic FALSE NEGATIVES (we=wrong, judge=right): {fn}/{n}  ({fn/n:.1%})  <- what strict matching misses")
    print(f"  deterministic FALSE POSITIVES (we=right, judge=wrong): {fp}/{n}  ({fp/n:.1%})")
    print(f"  judge cost: ${cost:.4f}")
    n_err = sum(1 for o in out if o["err"])
    if n_err:
        print(f"  ({n_err} judge calls errored, excluded)")

    (REPO / "outputs" / "baselines" / "judge_crosscheck.json").write_text(json.dumps({
        "n": n, "judge_model": args.judge_model, "glob": args.glob,
        "deterministic_acc": det_acc, "judge_acc": jud_acc, "offset": jud_acc - det_acc,
        "agreement": agree, "false_neg": fn, "false_pos": fp, "cost_usd": round(cost, 4),
        "details": out}, indent=1))
    print(f"[judge] wrote outputs/baselines/judge_crosscheck.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="longmemeval__*__full_context__*.jsonl",
                    help="store-file glob under outputs/baselines/cache/")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judge-model", default="openai/gpt-4o")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
