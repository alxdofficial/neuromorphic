#!/usr/bin/env bash
# Sequentially train V1.1 → V1.2 trajectory → V1.2 flat-bank → V1.4 → V1.5.
# Each saves to outputs/v1.X/ (main repo) and is paired with an eval pass.
#
# Run: bash scripts/eval/v1_chain.sh 2>&1 | tee outputs/v1_chain.log
#
# Stop early if obvious failure (NaN loss, OOM). Each entry is ~4h at BS=8.

set -uo pipefail   # NOT -e — we want to continue past a single failure

NUMSTEPS=10000
BS=8
LR=3e-4
LR_ADAPTER=3e-4
DATA_TRAIN=/home/alex/code/neuromorphic/data/wave1/composite_v1/train
DATA_VAL=/home/alex/code/neuromorphic/data/wave1/composite_v1/val
OUT_BASE=/home/alex/code/neuromorphic/outputs

train_one() {
    local LABEL=$1
    local WORKTREE=$2
    local EXTRA=$3
    local OUT=$OUT_BASE/$LABEL
    mkdir -p $OUT

    echo ""
    echo "======================================================================"
    echo "TRAINING $LABEL  (worktree: $WORKTREE)  $(date)"
    echo "  extra args: $EXTRA"
    echo "  output: $OUT/"
    echo "======================================================================"

    cd $WORKTREE
    python scripts/training/train_wave1_retrieval.py \
        --composite-dir $DATA_TRAIN \
        --composite-val-dir $DATA_VAL \
        --num-steps $NUMSTEPS \
        --batch-size $BS \
        --lr $LR \
        --lr-adapter $LR_ADAPTER \
        --val-every 500 \
        --val-batches 20 \
        --log-every 50 \
        --save-every 1000 \
        --checkpoint-out $OUT/ckpt.pt \
        $EXTRA \
        > $OUT/training.log 2>&1
    local EXIT=$?

    echo "$LABEL exit=$EXIT  $(date)"

    # Preserve final ckpt under unique name
    if [ -f "$OUT/ckpt.pt" ]; then
        cp $OUT/ckpt.pt $OUT/ckpt.final.pt
        echo "  saved $OUT/ckpt.final.pt"
    fi

    cd /home/alex/code/neuromorphic
}

START=$(date +%s)
echo "Chain start: $(date)"

train_one v1.1            /home/alex/code/neuromorphic-v1.1  ""
train_one v1.2_trajectory /home/alex/code/neuromorphic-v1.2  ""
train_one v1.2_flatbank   /home/alex/code/neuromorphic-v1.2  "--flat-bank"
train_one v1.4            /home/alex/code/neuromorphic-v1.4  ""
train_one v1.5            /home/alex/code/neuromorphic-v1.5  "--per-hop-contrast-coef 0.05"

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "======================================================================"
echo "ALL DONE  $(date)  elapsed=$((ELAPSED/3600))h$(((ELAPSED%3600)/60))m"
echo "======================================================================"

# Quick summary
echo ""
echo "Per-run final ckpts:"
ls -la $OUT_BASE/v1.*/ckpt.final.pt 2>/dev/null
echo ""
echo "Final loss per run (last line of training.log):"
for d in $OUT_BASE/v1.1 $OUT_BASE/v1.2_trajectory $OUT_BASE/v1.2_flatbank $OUT_BASE/v1.4 $OUT_BASE/v1.5; do
    if [ -f $d/training.log ]; then
        last_step=$(grep -oE "step\s+[0-9]+\s+loss=[0-9.]+" $d/training.log | tail -1)
        echo "  $(basename $d): $last_step"
    fi
done
