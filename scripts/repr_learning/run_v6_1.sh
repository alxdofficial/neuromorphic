#!/usr/bin/env bash
# v6.1 single-arm retrain: train graph_v6 (with F2+F1+F3 read fixes) → score →
# judge. Compares against the SAVED v2_1 floor/ceiling/baselines (no baseline
# retrain needed). Failure-isolated + per-step timeouts.
set -u
cd /home/alex/code/neuromorphic
TAG=v6_1
V=graph_v6_baseline
LOGD="outputs/repr_learning/${TAG}_logs"
EVALD="outputs/repr_learning/eval_per_family"
mkdir -p "$LOGD"
ML="$LOGD/run.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$ML"; }

log "######## v6.1 RUN START ########"
log "==== TRAIN $V (tag $TAG) ===="
timeout 3h python scripts/repr_learning/train_repr_qa.py --variants "$V" --out-tag "$TAG" \
  > "$LOGD/train.log" 2>&1
rc=$?; [ $rc -eq 0 ] && log "==== TRAIN DONE ====" || log "!!!! TRAIN rc=$rc"

ckdir="outputs/repr_learning/${TAG}_${V}/ckpts"; which=best
[ -f "$ckdir/${V}.best.pt" ] || which=last
pat="outputs/repr_learning/${TAG}_{v}/ckpts/{v}.${which}.pt"
log "==== SCORE std @8192 (ckpt=$which) ===="
timeout 90m python scripts/repr_learning/eval_per_family.py --variants "$V" \
  --families biographical hotpot_qa narrative_qa musique babilong ruler_niah \
  --ckpt-pattern "$pat" --tag "eval_${TAG}_${V}" --chunk-size 8192 --window-size 1024 \
  --n-per-family 48 --batch-size 4 > "$LOGD/score_std.log" 2>&1 || log "!!!! score std rc=$?"
log "==== SCORE locomo @24576 ===="
timeout 60m python scripts/repr_learning/eval_per_family.py --variants "$V" \
  --families locomo --ckpt-pattern "$pat" --tag "eval_${TAG}_${V}_locomo" \
  --chunk-size 24576 --window-size 1024 --n-per-family 48 --batch-size 2 \
  > "$LOGD/score_locomo.log" 2>&1 || log "!!!! score locomo rc=$?"

log "==== JUDGE ===="
JIN="$EVALD/judgein_${TAG}.jsonl"
cat "$EVALD"/eval_${TAG}_*_per_sample.jsonl > "$JIN" 2>/dev/null
timeout 30m python scripts/repr_learning/llm_judge.py --jsonl "$JIN" --tag "judge_${TAG}" \
  > "$LOGD/judge.log" 2>&1 && log "==== JUDGE DONE ====" || log "!!!! judge rc=$?"
log "######## v6.1 RUN COMPLETE ########"
