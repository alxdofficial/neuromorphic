#!/usr/bin/env bash
# Overnight v2.1 7-arm sweep: train each arm -> score each arm -> judge -> table.
# Fully autonomous + robust: every step is failure-isolated (one crash never
# aborts the run) and wrapped in a timeout (a hung step is killed, run continues).
# Sequential on the single GPU (train frees the GPU before its scoring starts).
# Results land incrementally so a partial table is always buildable.
set -u
cd /home/alex/code/neuromorphic
TAG="${1:-v2_1}"
LOGD="outputs/repr_learning/${TAG}_overnight_logs"
EVALD="outputs/repr_learning/eval_per_family"
mkdir -p "$LOGD" "$EVALD"
ML="$LOGD/overnight.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$ML"; }

TRAINABLE=(graph_v6_baseline flat_baseline continuous_baseline recurrent_baseline memorizing_baseline vanilla_llama)
EVALONLY=(vanilla_full_context)
STD_FAMS="biographical hotpot_qa narrative_qa musique babilong ruler_niah"

train_arm(){
  local v=$1
  log "==== TRAIN $v START ===="
  timeout 3h python scripts/repr_learning/train_repr_qa.py --variants "$v" --out-tag "$TAG" \
    > "$LOGD/train_${v}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && log "==== TRAIN $v DONE ====" || log "!!!! TRAIN $v FAILED rc=$rc (see train_${v}.log)"
}

score_arm(){
  local v=$1
  local ckdir="outputs/repr_learning/${TAG}_${v}/ckpts"
  local which=best
  [ -f "$ckdir/${v}.best.pt" ] || which=last
  if [ ! -f "$ckdir/${v}.${which}.pt" ]; then log "!!!! no ckpt for $v -- skip scoring"; return 1; fi
  local pat="outputs/repr_learning/${TAG}_{v}/ckpts/{v}.${which}.pt"
  log "==== SCORE $v (std @8192, ckpt=${which}) ===="
  timeout 90m python scripts/repr_learning/eval_per_family.py --variants "$v" \
    --families $STD_FAMS --ckpt-pattern "$pat" --tag "eval_${TAG}_${v}" \
    --chunk-size 8192 --window-size 1024 --n-per-family 48 --batch-size 4 \
    > "$LOGD/score_${v}.log" 2>&1 || log "!!!! score(std) $v rc=$? (see score_${v}.log)"
  log "==== SCORE $v (locomo @24576) ===="
  timeout 60m python scripts/repr_learning/eval_per_family.py --variants "$v" \
    --families locomo --ckpt-pattern "$pat" --tag "eval_${TAG}_${v}_locomo" \
    --chunk-size 24576 --window-size 1024 --n-per-family 48 --batch-size 2 \
    > "$LOGD/score_${v}_locomo.log" 2>&1 || log "!!!! score(locomo) $v rc=$? (see score_${v}_locomo.log)"
}

build_table(){
  python scripts/repr_learning/build_overnight_table.py --tag "$TAG" >> "$ML" 2>&1 || log "!!!! build_table rc=$?"
}

log "######## OVERNIGHT SWEEP START (tag=$TAG) ########"
for v in "${TRAINABLE[@]}"; do
  train_arm "$v"
  score_arm "$v"
  build_table   # refresh after each arm so a partial result is always current
done
for v in "${EVALONLY[@]}"; do
  train_arm "$v"      # eval-only: runs a val pass + saves a loadable ckpt
  score_arm "$v"
  build_table
done

# ---- LLM judge (headline metric, best-effort) ----
log "==== JUDGE (combine + score) ===="
JUDGEIN="$EVALD/judgein_${TAG}.jsonl"
cat "$EVALD"/eval_${TAG}_*_per_sample.jsonl > "$JUDGEIN" 2>/dev/null
NJ=$(wc -l < "$JUDGEIN" 2>/dev/null || echo 0)
log "judge input: $NJ rows"
timeout 60m python scripts/repr_learning/llm_judge.py --jsonl "$JUDGEIN" --tag "judge_${TAG}" \
  > "$LOGD/judge.log" 2>&1 && log "==== JUDGE DONE ====" \
  || log "!!!! JUDGE failed rc=$? -- table ships EM/containment only (see judge.log)"

# ---- final table (with judge if it succeeded) ----
log "==== BUILD FINAL TABLE ===="
build_table
log "######## OVERNIGHT SWEEP COMPLETE (tag=$TAG) -- table: docs/repr_learning_v2_1_results.md ########"
