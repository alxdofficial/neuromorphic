#!/usr/bin/env bash
# v2.2 baseline retrain — the 4 prepend/projection arms, now with the _NormMatch
# magnitude fix (terminal LayerNorm's 49×-OOD memory tokens -> ~0.9 Llama scale).
# graph_v6.1 (v6_1) and the two vanillas (v2_1) are UNAFFECTED (graph injects
# per-token; vanillas have no memory projection), so they're reused, not retrained.
# Sequential on one GPU; each step failure-isolated + timeout-wrapped.
set -u
cd /home/alex/code/neuromorphic
TAG="${1:-v2_2}"
LOGD="outputs/repr_learning/${TAG}_logs"
mkdir -p "$LOGD"
ML="$LOGD/run.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$ML"; }

ARMS=(flat_baseline continuous_baseline recurrent_baseline memorizing_baseline)
STD_FAMS="biographical hotpot_qa narrative_qa musique babilong ruler_niah"

log "######## v2.2 BASELINE RETRAIN (magnitude fix) tag=$TAG ########"
for v in "${ARMS[@]}"; do
  log "==== TRAIN $v START ===="
  timeout 3h python scripts/repr_learning/train_repr_qa.py --variants "$v" --out-tag "$TAG" \
    > "$LOGD/train_${v}.log" 2>&1 && log "==== TRAIN $v DONE ====" || log "!!!! TRAIN $v FAILED rc=$? ===="
  ck="outputs/repr_learning/${TAG}_${v}/ckpts"
  which=best; [ -f "$ck/${v}.best.pt" ] || which=last
  if [ -f "$ck/${v}.${which}.pt" ]; then
    pat="outputs/repr_learning/${TAG}_{v}/ckpts/{v}.${which}.pt"
    log "==== SCORE $v (std @8192, ckpt=${which}) ===="
    timeout 90m python scripts/repr_learning/eval_per_family.py --variants "$v" \
      --families $STD_FAMS --ckpt-pattern "$pat" --tag "eval_${TAG}_${v}" \
      --chunk-size 8192 --window-size 1024 --n-per-family 48 --batch-size 4 \
      > "$LOGD/score_${v}.log" 2>&1 || log "!!!! score $v rc=$? ===="
  else
    log "!!!! no ckpt for $v -- skip scoring"
  fi
done
log "######## v2.2 RETRAIN DONE (judge + combined table built separately) ########"
