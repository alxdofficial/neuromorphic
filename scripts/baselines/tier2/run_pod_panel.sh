#!/usr/bin/env bash
# Parallel Tier-2 pod panel — the ~2-hour run. Launches the three GPU methods CONCURRENTLY, one per GPU, each
# in its OWN env (KVzip / MemoryLLM / LCLM pin mutually-incompatible torch/transformers/flash-attn/python —
# see README.md "Environment isolation"; they cannot share a process). With per-context reuse wired in
# (tier2_common.run_grouped) each method is ~1.5 hr for LongMemEval; 3-way parallel ⇒ ~2 hr wall-clock.
#
# The 2b agent-memory baseline (A-MEM) is API-only and does NOT belong on the pod — run it from any box:
#   OPENROUTER_API_KEY=... python scripts/baselines/run_agentmem.py --method a-mem --dataset <ds>
#
# Usage (on the pod, from the repo root, after README.md's repo clones + per-method envs):
#   scripts/baselines/tier2/run_pod_panel.sh                       # LongMemEval, full (500 Q)
#   DATASET=memoryagentbench scripts/baselines/tier2/run_pod_panel.sh
#   MAXEX="--max-examples 20" scripts/baselines/tier2/run_pod_panel.sh    # smoke on 20 items
#
# Override the env-activation + GPU assignment to match your pod:
#   KVZIP_ENV="source ~/venvs/kvzip/bin/activate"  KVZIP_GPU=0  ... etc.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

DATASET="${DATASET:-longmemeval}"          # longmemeval | memoryagentbench
MAXEX="${MAXEX:-}"                          # e.g. "--max-examples 20"
KV_METHOD="${KV_METHOD:-kvzip}"            # kvzip works on both datasets; snapkv/h2o are LongMemEval-only
LOGDIR="${LOGDIR:-outputs/baselines/pod_logs}"; mkdir -p "$LOGDIR"

# per-method env activation + GPU. Defaults assume conda envs named per method; override as needed.
KVZIP_ENV="${KVZIP_ENV:-conda activate kvzip}";       KVZIP_GPU="${KVZIP_GPU:-0}"
MEMLLM_ENV="${MEMLLM_ENV:-conda activate memoryllm}"; MEMLLM_GPU="${MEMLLM_GPU:-1}"
LCLM_ENV="${LCLM_ENV:-conda activate lclm}";          LCLM_GPU="${LCLM_GPU:-2}"

echo "[pod_panel] dataset=$DATASET kv_method=$KV_METHOD maxex='${MAXEX:-<all>}' logs=$LOGDIR"

# launch one method: <name> <env-activate-cmd> <gpu-id> <python args...>
# audit #5: background in the CURRENT shell (not inside $(...)) and capture $! at the call site — launching
# inside command substitution makes the job a child of the subshell, so the parent's `wait` can't reap it,
# and mixing a status echo with the PID corrupts the captured value. Status goes to stderr; nothing to stdout.
launch() {
  local name="$1" env="$2" gpu="$3"; shift 3
  local log="$LOGDIR/${name}.log"
  echo "[pod_panel] launching $name on GPU $gpu → $log" >&2
  ( eval "$env" && CUDA_VISIBLE_DEVICES="$gpu" python "$@" ) >"$log" 2>&1 &
}

declare -A PID
launch "$KV_METHOD" "$KVZIP_ENV" "$KVZIP_GPU" \
  scripts/baselines/tier2/run_kvcompress.py --method "$KV_METHOD" --dataset "$DATASET" $MAXEX
PID[kv]=$!
launch memoryllm "$MEMLLM_ENV" "$MEMLLM_GPU" \
  scripts/baselines/tier2/run_memoryllm.py --dataset "$DATASET" $MAXEX
PID[memoryllm]=$!
launch lclm "$LCLM_ENV" "$LCLM_GPU" \
  scripts/baselines/tier2/run_lclm.py --dataset "$DATASET" $MAXEX
PID[lclm]=$!

echo "[pod_panel] PIDs: ${PID[*]} — waiting for all three ..."
rc=0
for name in "${!PID[@]}"; do
  if wait "${PID[$name]}"; then echo "[pod_panel] $name ✓"; else echo "[pod_panel] $name ✗ (see $LOGDIR/*.log)"; rc=1; fi
done

echo "[pod_panel] all done (rc=$rc). Aggregating:"
python scripts/baselines/report.py --glob "${DATASET}__*" --no-published 2>/dev/null | tail -30 || true
echo "[pod_panel] pull outputs/baselines/ back over R2 (scripts/pod/pull_results.sh)."
exit $rc
