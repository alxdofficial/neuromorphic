#!/usr/bin/env bash
# Audit-fix operational step: after the in-flight API jobs finish, re-run every eval config so it RESUMES
# (skips done rows — the sig is unchanged/gated), RESCORES with the current scorer (BEM 0.85, EventQA-parsed,
# competency macro/lenient), fills the error/cutoff/content_filter rows (now retryable), and writes
# correctly-formatted aggregates (report_mode → distinct k columns; config_sig/bem_threshold in meta).
# Then delete STALE pre-fix aggregate JSONs (those whose meta lacks config_sig) and print the report.
set -uo pipefail
cd /home/alex/code/neuromorphic
# Read the key from the environment — NEVER hardcode a credential in a tracked file.
# Run as: OPENROUTER_API_KEY=sk-... scripts/baselines/rescore_and_report.sh
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY in your shell before running}"
LOG=outputs/baselines/rescore.log; : > "$LOG"
say(){ echo "[rescore] $*" | tee -a "$LOG"; }

say "waiting for in-flight API jobs to finish ..."
while pgrep -f "run_api_eval.py --dataset memoryagentbench" >/dev/null; do sleep 20; done
say "in-flight jobs done. re-running (resume+rescore+gap-fill):"

run(){ say "-> $*"; scripts/cpu_capped.sh -- .venv/bin/python scripts/baselines/run_api_eval.py "$@" >>"$LOG" 2>&1; }

# LongMemEval (500): floor/rag/full_context, both models — resume + rescore @BEM0.85 + fill 28 llama gaps
run --dataset longmemeval --models meta-llama/llama-3.1-8b-instruct deepseek/deepseek-v4-flash \
    --modes floor rag_bm25 full_context --concurrency 32
# MAB floor + rag k5 (both models)
run --dataset memoryagentbench --models meta-llama/llama-3.1-8b-instruct deepseek/deepseek-v4-flash \
    --modes floor rag_bm25 --concurrency 32
# MAB rag k15 (both models)
run --dataset memoryagentbench --models meta-llama/llama-3.1-8b-instruct deepseek/deepseek-v4-flash \
    --modes rag_bm25 --bm25-topk 15 --concurrency 32
# MAB full_context deepseek (pinned+cached) — fills the 85 cutoffs
run --dataset memoryagentbench --models deepseek/deepseek-v4-flash --modes full_context \
    --pin-provider deepseek --concurrency 32

say "deleting STALE pre-fix aggregate JSONs (meta lacks config_sig) ..."
scripts/cpu_capped.sh -- .venv/bin/python - <<'PY' 2>>"$LOG"
import json, glob, os
for f in glob.glob("outputs/baselines/*.json"):
    b = os.path.basename(f)
    if b.endswith(("_api_summary.json", "report.json", "judge_crosscheck.json")):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if "config_sig" not in (d.get("meta") or {}):
        os.remove(f); print(f"[rescore]   deleted stale {b}")
PY

say "final report:"
scripts/cpu_capped.sh -- .venv/bin/python scripts/baselines/report.py --glob '*.json' 2>>"$LOG" | tee -a "$LOG" | tail -60
say "DONE. full log: $LOG"
