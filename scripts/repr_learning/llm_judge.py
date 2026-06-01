#!/usr/bin/env python3
"""LLM judge (OpenRouter) over rescore predictions — 7-category rubric.

Headline correctness that doesn't punish paraphrase or verbosity. Each prediction is
classified into ONE category via STRUCTURED OUTPUT (json-schema enum → no parse
ambiguity); a numeric score is attached post-hoc (re-weightable). temperature=0 +
seed=0 for determinism; robust retries that FAIL LOUDLY rather than silently scoring 0.

Reports the category distribution, the derived score, cost, wall-time, and divergence
vs the lexical containment metric. Key: $OPENROUTER_API_KEY or .openrouter_key (git-ignored).
"""
import sys, os, json, time, argparse, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import requests
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
import scripts.repr_learning.eval_per_family as EPF

CATS = ["correct", "correct_verbose", "incomplete", "partially_wrong", "wrong_value", "irrelevant", "no_answer"]
SCORE = {"correct": 1.0, "correct_verbose": 1.0, "incomplete": 0.5, "partially_wrong": 0.25,
         "wrong_value": 0.0, "irrelevant": 0.0, "no_answer": 0.0}
PRICES = {  # $/M (input, output)
    "google/gemini-3-flash-preview": (0.50, 3.0), "google/gemini-2.0-flash-001": (0.10, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60), "deepseek/deepseek-chat": (0.229, 0.914),
}
SYS = (
    "You grade a question-answering benchmark. You are given a QUESTION, the REFERENCE "
    "answer(s), and a MODEL answer. Choose the single best category.\n"
    "Judge ONLY whether the MODEL answer matches the REFERENCE. Do NOT use outside or world "
    "knowledge: some questions are about fictional settings, so an answer is correct only if it "
    "matches the reference, never because it merely sounds plausible. Paraphrases, aliases, "
    "abbreviations and other valid surface forms of the reference are CORRECT; a more specific "
    "answer that still contains the reference is correct; extra surrounding words are fine; a "
    "different fact or value is WRONG.\n"
    "Categories:\n"
    "- correct: conveys the reference answer, no incorrect claims.\n"
    "- correct_verbose: conveys the reference answer but wrapped in extra unnecessary words.\n"
    "- incomplete: gives part of the reference answer but omits a required part (says nothing wrong).\n"
    "- partially_wrong: contains the correct answer but ALSO states something incorrect.\n"
    "- wrong_value: right type of answer (a date/name/place as asked) but the specific value is wrong.\n"
    "- irrelevant: off-topic, or the wrong type of answer entirely.\n"
    "- no_answer: a refusal, 'I don't know', empty, or a non-answer."
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "grade", "strict": True, "schema": {
        "type": "object", "properties": {"category": {"type": "string", "enum": CATS}},
        "required": ["category"], "additionalProperties": False}},
}
URL = "https://openrouter.ai/api/v1/chat/completions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "outputs/repr_learning/eval_per_family/rescore_newmetric_per_sample.jsonl"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="judge")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    # Prefer the explicit .openrouter_key file over any (possibly stale) env var.
    _kf = ROOT / ".openrouter_key"
    key = _kf.read_text().strip() if _kf.exists() else os.getenv("OPENROUTER_API_KEY")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    rows = [json.loads(l) for l in open(args.jsonl)]
    fams = sorted(set(r["family"] for r in rows))
    cfg = ReprConfig(n_flat_codes=128, max_window_size=8192, fixed_window_size=1024)
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    qcache = ROOT / "outputs/repr_learning/eval_per_family/judge_qa_cache.json"
    qa = {}
    if qcache.exists():
        c = json.loads(qcache.read_text())
        if all(f in c for f in fams):
            qa = {f: [(q, r) for q, r in c[f]] for f in fams}
            print(f"[judge] question cache hit ({sum(len(v) for v in qa.values())} q) — skipping data load", flush=True)
    if not qa:
        print(f"[judge] pairing questions for {len(rows)} predictions across {fams} (one-time ~1-2 min) ...", flush=True)
        for fam in fams:
            ss = EPF.collect_samples([fam], 48, tokenizer=tok, cfg=cfg, chunk_size=8192, passages_per_chunk=600)
            qa[fam] = [(tok.decode(s.question_ids.tolist()).strip(), s.answer_refs) for s in ss]
        qcache.write_text(json.dumps({f: [[q, list(r)] for q, r in v] for f, v in qa.items()}))
    by_vf = defaultdict(list)
    for r in rows:
        by_vf[(r["variant"], r["family"])].append(r)
    tasks = []
    for (v, f), rr in by_vf.items():
        for i, r in enumerate(rr):
            q, refs = qa[f][i]
            assert list(refs) == list(r["refs"]), f"pairing mismatch {v}/{f}/{i}: {refs} vs {r['refs']}"
            tasks.append({"v": v, "f": f, "q": q, "refs": r["refs"], "pred": r["pred"], "contain": r["contain"]})
    if args.n:
        tasks = random.Random(0).sample(tasks, min(args.n, len(tasks)))
    print(f"[judge] judging {len(tasks)} with {args.model}, {args.workers}-way, structured-output ...", flush=True)

    def call(task):
        user = f"QUESTION: {task['q']}\nREFERENCE: {' ; '.join(task['refs'])}\nMODEL: {task['pred']}"
        body = {"model": args.model, "temperature": 0, "seed": 0, "max_tokens": 30,
                "response_format": RESPONSE_FORMAT,
                "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}
        last = ""
        for attempt in range(6):
            try:
                resp = requests.post(URL, headers=headers, json=body, timeout=40)
                if resp.status_code != 200:
                    last = f"HTTP{resp.status_code}:{resp.text[:120]}"
                    raise RuntimeError(last)
                j = resp.json()
                txt = j["choices"][0]["message"]["content"]
                cat = json.loads(txt).get("category")
                if cat not in CATS:
                    raise ValueError(f"bad cat {txt!r}")
                return task, cat, j.get("usage", {})
            except Exception as e:
                last = str(e)[:140]
                if attempt < 5:
                    time.sleep(min(8.0, 1.5 ** attempt) + random.uniform(0, 1.0))  # short jittered backoff; cross-pass retry handles windows
        return task, "__FAIL__:" + last, {}

    out = ROOT / f"outputs/repr_learning/eval_per_family/{args.tag}_per_sample.jsonl"

    def save(by_i):
        with open(out, "w") as fh:
            for i in range(len(tasks)):
                if i in by_i:
                    task, cat, usage = by_i[i]
                    fh.write(json.dumps({"variant": task["v"], "family": task["f"], "pred": task["pred"],
                                         "refs": task["refs"], "contain": task["contain"], "gi": i,
                                         "category": cat, "score": SCORE.get(cat)}) + "\n")

    # Resume: reload already-judged (valid category) so prior progress / partial runs persist.
    by_i = {}
    if out.exists():
        prior = {}
        for line in open(out):
            rec = json.loads(line)
            if rec.get("category") in CATS and "gi" in rec:
                prior[rec["gi"]] = rec["category"]
        for i, task in enumerate(tasks):
            if i in prior:
                by_i[i] = (task, prior[i], {})
        if by_i:
            print(f"[judge] resuming: {len(by_i)}/{len(tasks)} already judged", flush=True)

    # Multi-pass: re-judge failures in later passes to cross transient key-propagation 401
    # windows (new OpenRouter keys 401 'User not found' for tens of seconds at a time). Saves
    # incrementally so a timeout/kill never loses progress (re-run resumes).
    t0 = time.time()
    pending = [(i, t) for i, t in enumerate(tasks) if i not in by_i]
    total = len(tasks)
    for pass_i in range(8):
        if not pending:
            break
        p0 = time.time()
        done = 0
        n_pass = len(pending)
        nxt = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(call, t): i for i, t in pending}
            for fut in as_completed(futs):
                i = futs[fut]
                task, cat, usage = fut.result()
                by_i[i] = (task, cat, usage)
                if str(cat).startswith("__FAIL__"):
                    nxt.append((i, task))
                done += 1
                if done % 25 == 0 or done == n_pass:
                    rate = done / max(time.time() - p0, 0.1)
                    eta = (n_pass - done) / max(rate, 0.1)
                    n_done_total = sum(1 for v in by_i.values() if not str(v[1]).startswith("__FAIL__"))
                    print(f"[judge] pass{pass_i} {done}/{n_pass}  |  total {n_done_total}/{total}  "
                          f"|  {rate:.1f} it/s  |  failed={len(nxt)}  |  ETA {eta:.0f}s", flush=True)
        pending = nxt
        save(by_i)
        if pending and pass_i < 7:
            print(f"[judge] pass {pass_i}: {len(pending)} still failing (key window?), retry in 30s ...", flush=True)
            time.sleep(30)
    wall = time.time() - t0
    results = [by_i[i] for i in range(len(tasks))]

    tin = tout = 0
    fails = [(t, c) for t, c, u in results if str(c).startswith("__FAIL__")]
    per_v = defaultdict(Counter)
    per_v_score = defaultdict(list)
    agree = n_ok = 0
    diverge = []
    for task, cat, usage in results:
        tin += usage.get("prompt_tokens", 0); tout += usage.get("completion_tokens", 0)
        if str(cat).startswith("__FAIL__"):
            continue
        per_v[task["v"]][cat] += 1
        per_v_score[task["v"]].append(SCORE[cat])
        n_ok += 1
        jc = SCORE[cat] >= 1.0
        if jc == (task["contain"] > 0):
            agree += 1
        elif len(diverge) < 8:
            diverge.append((task, cat))
    pin, pout = PRICES.get(args.model, (1.0, 5.0))
    cost = tin / 1e6 * pin + tout / 1e6 * pout

    print(f"\n=== COST / TIME ({len(tasks)} judged, {args.model}) ===")
    print(f"  wall={wall:.1f}s  ({wall/max(1,len(tasks))*1000:.0f} ms/judgment)")
    print(f"  tokens: {tin} in + {tout} out -> ${cost:.4f}  (extrapolated to 1200: ${cost/max(1,len(tasks))*1200:.3f})")
    if fails:
        print(f"  !!! {len(fails)} FAILED judgments (NOT scored) — e.g. {fails[0][1][:90]}")
    else:
        print(f"  failures: 0  (all judgments obtained)")
    print(f"\n=== JUDGE-correct vs CONTAINMENT agreement: {100*agree/max(1,n_ok):.0f}% ===")
    print(f"\n{'variant':22s} {'judgeScore':>10s}   category distribution")
    for v in sorted(per_v_score, key=lambda v: -sum(per_v_score[v]) / len(per_v_score[v])):
        sc = 100 * sum(per_v_score[v]) / len(per_v_score[v])
        dist = " ".join(f"{c}={per_v[v][c]}" for c in CATS if per_v[v][c])
        print(f"  {v:22s} {sc:>9.1f}   {dist}")
    print(f"\n=== sample judgments ===")
    for task, cat, usage in results[:args.show]:
        print(f"  [{cat:15s}] Q:{task['q'][:42]} | GOLD:{str(task['refs'])[:26]} | PRED:{task['pred'][:38]!r}")
    if diverge:
        print(f"\n=== divergences (judge 'correct' but containment=0, OR vice-versa) ===")
        for task, cat in diverge:
            print(f"  judge={cat:15s} contain={int(task['contain'])} | GOLD:{str(task['refs'])[:24]} | PRED:{task['pred'][:38]!r}")

    save(by_i)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
