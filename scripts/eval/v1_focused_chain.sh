#!/usr/bin/env bash
# Focused v1 retrain: V1.5 (per-hop contrastive — best representation of v1 lineage)
# + V1.2 flat-bank baseline. Total ~8h instead of full chain's 22h.
#
# Run: bash scripts/eval/v1_focused_chain.sh

set -uo pipefail

NUMSTEPS=10000
BS=8
LR=3e-4
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
    echo "======================================================================"

    cd $WORKTREE
    python scripts/training/train_wave1_retrieval.py \
        --composite-dir $DATA_TRAIN \
        --composite-val-dir $DATA_VAL \
        --num-steps $NUMSTEPS \
        --batch-size $BS \
        --lr $LR \
        --lr-adapter $LR \
        --val-every 500 \
        --val-batches 20 \
        --log-every 50 \
        --save-every 1000 \
        --checkpoint-out $OUT/ckpt.pt \
        $EXTRA \
        > $OUT/training.log 2>&1
    local EXIT=$?
    echo "$LABEL exit=$EXIT  $(date)"
    if [ -f "$OUT/ckpt.pt" ]; then
        cp $OUT/ckpt.pt $OUT/ckpt.final.pt
    fi
    cd /home/alex/code/neuromorphic
}

START=$(date +%s)
echo "Focused chain start: $(date)"

# V1.5 first — the headline result
train_one v1.5  /home/alex/code/neuromorphic-v1.5  "--per-hop-contrast-coef 0.05"

# Flat-bank baseline second
train_one v1.2_flatbank  /home/alex/code/neuromorphic-v1.2  "--flat-bank"

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "DONE  $(date)  elapsed=$((ELAPSED/3600))h$(((ELAPSED%3600)/60))m"
echo ""
echo "Final ckpts:"
ls -la $OUT_BASE/{v1.5,v1.2_flatbank}/ckpt.final.pt 2>/dev/null
