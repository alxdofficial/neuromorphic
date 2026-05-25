#!/usr/bin/env bash
# v1h_t4k_v3 — overnight retrain after Phase 1+2+4 fixes + graph audit fixes.
#
# What changed since tranche 1 v2:
#   - graph_baseline:
#       * u now derived from pick_count popularity (not update_alpha selectivity)
#       * state update gathers from routed proposal (not slot's own)
#       * recycle admission >= for cold-start
#       * all-pad row protection re-applied after recycle (was bypassed)
#       * graph_overwrite_rate renamed → graph_overwrites_per_row_per_window
#   - eval/training pipeline:
#       * eval_best.py defaults match trainer
#       * fixed-val protocol via materialize_val_set in trainer + eval_zero_mem
#       * resume preserves prior best.pt (best_val_recon stashed in last.pt)
#       * extrap_val_curve skips final/best.pt re-eval rows
#
# Protocol matches tranche-1-v2 for direct comparison:
#   chunk=4096, window=1024, narrative + mix 0.5/0.25/0.25, 8K steps
#
# Variants:
#   trained:    graph_baseline + 4 standard baselines (flat/continuous/memorizing/recurrent)
#   eval-only:  vanilla_llama (floor), vanilla_full_context (ceiling)
#
# Splat and plastic skipped (designs not locked for splat; both deferred per user).

set -euo pipefail

cd /home/alex/code/neuromorphic

OUT_TAG="v1h_t4k_v3"
STEPS=8000
CHUNK=4096
WINDOW=1024
LOG_DIR="outputs/repr_learning/${OUT_TAG}_launch_log"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --steps "$STEPS"
  --chunk-size "$CHUNK"
  --window-size "$WINDOW"
  --out-tag "$OUT_TAG"
  --narrative
  --mix-weights 0.5 0.25 0.25
  --val-batches 32
)

TRAIN_VARIANTS=(
  graph_baseline
  flat_baseline
  continuous_baseline
  memorizing_baseline
  recurrent_baseline
)

EVAL_ONLY_VARIANTS=(
  vanilla_llama
  vanilla_full_context
)

echo "==========================================" | tee "$LOG_DIR/launch.log"
echo "v1h_t4k_v3 overnight launch — $(date -u)" | tee -a "$LOG_DIR/launch.log"
echo "  steps=$STEPS  chunk=$CHUNK  window=$WINDOW" | tee -a "$LOG_DIR/launch.log"
echo "  train: ${TRAIN_VARIANTS[*]}" | tee -a "$LOG_DIR/launch.log"
echo "  eval:  ${EVAL_ONLY_VARIANTS[*]}" | tee -a "$LOG_DIR/launch.log"
echo "==========================================" | tee -a "$LOG_DIR/launch.log"

# Train each variant sequentially. Each gets its own log so a single failure
# doesn't poison the others.
for v in "${TRAIN_VARIANTS[@]}"; do
  echo ""  | tee -a "$LOG_DIR/launch.log"
  echo "=== $(date -u) training $v ===" | tee -a "$LOG_DIR/launch.log"
  python -m scripts.repr_learning.train_repr_qa \
    --variants "$v" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${v}.log" \
    || echo "FAILED $v (continuing with next)" | tee -a "$LOG_DIR/launch.log"
done

# Vanilla variants — eval-only (trainer takes is_vanilla path with 0 steps).
for v in "${EVAL_ONLY_VARIANTS[@]}"; do
  echo "" | tee -a "$LOG_DIR/launch.log"
  echo "=== $(date -u) eval-only $v ===" | tee -a "$LOG_DIR/launch.log"
  python -m scripts.repr_learning.train_repr_qa \
    --variants "$v" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${v}.log" \
    || echo "FAILED $v" | tee -a "$LOG_DIR/launch.log"
done

echo "" | tee -a "$LOG_DIR/launch.log"
echo "=== $(date -u) ALL DONE ===" | tee -a "$LOG_DIR/launch.log"
echo "Outputs at outputs/repr_learning/${OUT_TAG}_<variant>/" | tee -a "$LOG_DIR/launch.log"
