#!/usr/bin/env bash
# Clean 2x2 attribution: retrain the slotgraph single-component controls from
# scratch (same config, LoRA rank 82, 3 seeds), then build the attribution table.
# full slotgraph + icae already exist in the cohort. Resilient: a failed run is
# logged and skipped.
cd /home/alex/code/neuromorphic || exit 1
LOG=outputs/memory/sg_attribution.log
: > "$LOG"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

COMMON="--task mixed --steps 4000 --batch-size 8 --backbone HuggingFaceTB/SmolLM2-135M --src-tokenizer meta-llama/Llama-3.2-1B --mixed-ctx 1024 --mixed-M 32 --mixed-gate-batches 8 --variants slotgraph_baseline"

# prefix:flags
SPECS=(
  "valrun_sg_structonly:--slotgraph-no-id"
  "valrun_sg_idonly:--slotgraph-no-structure"
  "valrun_sg_neither:--slotgraph-no-structure --slotgraph-no-id"
)

# wait for any in-flight trainer to free the GPU
for i in $(seq 1 60); do pgrep -f "scripts/train/train.py" >/dev/null 2>&1 || break; sleep 30; done

for SEED in 42 1 2; do
  for spec in "${SPECS[@]}"; do
    prefix="${spec%%:*}"; flags="${spec#*:}"
    tag="${prefix}"; [ "$SEED" != 42 ] && tag="${prefix}_s${SEED}"
    log "RUN seed=$SEED $tag  [$flags]"
    rm -rf "outputs/memory/${tag}_slotgraph_baseline" 2>/dev/null
    .venv/bin/python scripts/train/train.py $COMMON $flags --seed "$SEED" --out-tag "$tag" \
        > "outputs/memory/${tag}.log" 2>&1
    log "  exit=$? gradskips=$(grep -ac 'non-finite' outputs/memory/${tag}.log 2>/dev/null)"
  done
done

log "all attribution runs done; building table..."
.venv/bin/python scripts/diagnostics/slotgraph_attribution.py >> "$LOG" 2>&1
log "SG ATTRIBUTION COMPLETE"
