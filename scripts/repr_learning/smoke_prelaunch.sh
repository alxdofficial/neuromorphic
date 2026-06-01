#!/usr/bin/env bash
# Pre-launch smoke for the v2.1 7-arm sweep. graph_v6 gets 500 steps (the arm we
# care about — watch grad norms, health telemetry, state magnitudes, loss curve);
# every other trainable arm gets 50 steps (just confirm runs / loss not NaN /
# trending down). Single GPU → strictly sequential. Throwaway out-tag so it never
# touches the real run's outputs. Each arm is failure-isolated (one crash != abort).
set -u
cd /home/alex/code/neuromorphic
TAG=smoke_pf
LOGD="outputs/repr_learning/${TAG}_logs"
mkdir -p "$LOGD"
ML="$LOGD/smoke.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$ML"; }

run(){
  local v=$1 steps=$2 le=$3 ve=$4
  log "=== SMOKE $v ($steps steps) START ==="
  python scripts/repr_learning/train_repr_qa.py \
    --variants "$v" --out-tag "$TAG" --steps "$steps" \
    --log-every "$le" --val-every "$ve" --val-batches 8 \
    > "$LOGD/train_$v.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "!!! SMOKE $v FAILED rc=$rc (see $LOGD/train_$v.log)"; \
  else log "=== SMOKE $v DONE ==="; fi
}

log "######## PRE-LAUNCH SMOKE START ########"
run graph_v6_baseline   500 25 100   # the one we care about
run flat_baseline        50 10  25
run continuous_baseline  50 10  25
run recurrent_baseline   50 10  25
run memorizing_baseline  50 10  25
run vanilla_llama        50 10  25
run vanilla_full_context 50 10  25   # eval-only; just confirms its val pass runs
log "######## PRE-LAUNCH SMOKE COMPLETE ########"
