#!/usr/bin/env bash
# v1h_t4k_v3 — overnight retrain after Phase 1+2+4 fixes + graph audit fixes.
# UPDATED: max_steps 8000 → 20000 so models actually plateau.
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
#   chunk=4096, window=1024, narrative + mix 0.5/0.25/0.25
#   STEPS=20000 (was 8000 — too short; graph still improving at 8K with no
#   patience trigger). Patience=5×500 = 2500-step plateau detection still
#   stops anything that has actually converged.
#
# Resume:
#   graph_baseline, flat_baseline, continuous_baseline have v3 ckpts from
#   the earlier 8K run — resume from those (last.pt now carries best.pt
#   tracking thanks to today's resume fix).
#   memorizing_baseline, recurrent_baseline start fresh.

set -euo pipefail

cd /home/alex/code/neuromorphic

OUT_TAG="v1h_t4k_v3"
STEPS=20000
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

# Variants that have existing v3 ckpts → resume
RESUME_VARIANTS=(
  graph_baseline           # done 8K, resume from 8K → 20K
  flat_baseline            # done 8K, resume from 8K → 20K
  continuous_baseline      # got to ~4500, last save at 4000
)

# Fresh starts (no usable ckpt or none yet)
FRESH_VARIANTS=(
  memorizing_baseline
  recurrent_baseline
)

EVAL_ONLY_VARIANTS=(
  vanilla_llama
  vanilla_full_context
)

echo "==========================================" | tee "$LOG_DIR/launch.log"
echo "v1h_t4k_v3 overnight — STEPS=$STEPS — $(date -u)" | tee -a "$LOG_DIR/launch.log"
echo "  resume: ${RESUME_VARIANTS[*]}" | tee -a "$LOG_DIR/launch.log"
echo "  fresh:  ${FRESH_VARIANTS[*]}" | tee -a "$LOG_DIR/launch.log"
echo "  eval:   ${EVAL_ONLY_VARIANTS[*]}" | tee -a "$LOG_DIR/launch.log"
echo "==========================================" | tee -a "$LOG_DIR/launch.log"

# Resumed variants
for v in "${RESUME_VARIANTS[@]}"; do
  echo "" | tee -a "$LOG_DIR/launch.log"
  echo "=== $(date -u) RESUME $v ===" | tee -a "$LOG_DIR/launch.log"
  python -m scripts.repr_learning.train_repr_qa \
    --variants "$v" \
    --resume \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${v}.log" \
    || echo "FAILED $v (continuing with next)" | tee -a "$LOG_DIR/launch.log"
done

# Fresh variants
for v in "${FRESH_VARIANTS[@]}"; do
  echo "" | tee -a "$LOG_DIR/launch.log"
  echo "=== $(date -u) FRESH $v ===" | tee -a "$LOG_DIR/launch.log"
  python -m scripts.repr_learning.train_repr_qa \
    --variants "$v" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${v}.log" \
    || echo "FAILED $v (continuing with next)" | tee -a "$LOG_DIR/launch.log"
done

# Vanillas — eval-only
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
