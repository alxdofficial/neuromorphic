#!/bin/bash
# Replication experiment: compare VQ across seeds, continuous across sigmas.
# All runs reuse outputs/v13_tiny/cycle_00/{phase1_end.pt, codebook.pt}
# so starting state is bit-identical across conditions.
# Stages 1-3 only (skip stage 4 to avoid OOM in continuous mode).

set -e
cd /home/alex/code/neuromorphic

CKPT=outputs/v13_tiny/cycle_00/phase1_end.pt
CB=outputs/v13_tiny/cycle_00/codebook.pt
OUTROOT=outputs/v13_rep
PY=.venv/bin/python

run_vq() {
    local NAME=$1 SEED=$2
    local OUT=$OUTROOT/$NAME
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] Launching $NAME (VQ, seed=$SEED)"
    echo "=========================================="
    $PY -u -m src.train_phase2 \
      --checkpoint $CKPT --codebook $CB \
      --out $OUT/phase2.pt \
      --bs 8 --group-size 8 \
      --seed $SEED --eval-seed 0 \
      --stage1-tokens 2000000 --stage2-tokens 2000000 --stage3-tokens 2000000 \
      --stage4-tokens 0 \
      2>&1 | tee $OUT/run.log
}

run_cont() {
    local NAME=$1 SEED=$2 SIGMA=$3
    local OUT=$OUTROOT/$NAME
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] Launching $NAME (continuous σ=$SIGMA, seed=$SEED)"
    echo "=========================================="
    $PY -u -m src.train_phase2 \
      --checkpoint $CKPT --codebook $CB \
      --out $OUT/phase2.pt \
      --bs 8 --group-size 8 \
      --seed $SEED --eval-seed 0 \
      --continuous-sigma $SIGMA \
      --stage1-tokens 2000000 --stage2-tokens 2000000 --stage3-tokens 2000000 \
      --stage4-tokens 0 \
      2>&1 | tee $OUT/run.log
}

mkdir -p $OUTROOT/A1 $OUTROOT/A2 $OUTROOT/A3 $OUTROOT/B1 $OUTROOT/B2 $OUTROOT/B3

# A: VQ replication, 3 seeds
run_vq A1 42    # baseline seed (matches existing tiny-boot)
run_vq A2 100
run_vq A3 200

# B: Continuous sigma sweep
run_cont B1 42 0.3    # tight
run_cont B2 42 3.0    # middle
run_cont B3 42 10.0   # wide

echo "[$(date +%H:%M:%S)] All runs complete"
