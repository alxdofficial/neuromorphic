#!/usr/bin/env bash
# post_wave1_chain.sh — runs after Wave 1 v2 completes:
#   1. Wait for Wave 1 training process to exit.
#   2. Run needle EM bench (per-distance) on the final ckpt.
#   3. Run needle EM bench on the best ckpt (val-selected).
#   4. Launch Wave 2 SFT on wildchat_long from the final ckpt.
#
# Launched with: nohup bash scripts/diagnostics/post_wave1_chain.sh &
#
# Outputs everything to outputs/chain.log so the user can re-tail it
# after returning.

set -u
cd /home/alex/code/neuromorphic
exec >> outputs/chain.log 2>&1

WAVE1_PID="${1:-3016612}"
TS() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

echo
echo "=================================================================="
echo "[$(TS)] post-Wave1 chain starting"
echo "  watching pid: $WAVE1_PID"
echo "=================================================================="

# ── 1. Wait for Wave 1 to finish ───────────────────────────────────────
while kill -0 "$WAVE1_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(TS)] Wave 1 process $WAVE1_PID exited."

# Give the trainer 30s to flush + save the final ckpt.
sleep 30

if [ ! -f outputs/wave1_v2/ckpt.pt ]; then
  echo "[$(TS)] FATAL: outputs/wave1_v2/ckpt.pt missing. Aborting chain."
  exit 1
fi
echo "[$(TS)] Wave 1 ckpt confirmed."

# ── 2. EM bench on final ckpt (per-distance) ───────────────────────────
echo "[$(TS)] === EM bench on final ckpt (per-distance) ==="
PYTHONPATH=. python -u scripts/diagnostics/bench_em_accuracy.py \
  --model-type ours \
  --ckpt outputs/wave1_v2/ckpt.pt \
  --val-parquet data/wave1/needle.val.parquet \
  --max-docs 100 \
  --output outputs/em_v2_final.json \
  || echo "[$(TS)] WARN: EM bench (final ckpt) failed"

# ── 3. EM bench on best ckpt (val-selected) ────────────────────────────
if [ -f outputs/wave1_v2/ckpt.best.pt ]; then
  echo "[$(TS)] === EM bench on best ckpt ==="
  PYTHONPATH=. python -u scripts/diagnostics/bench_em_accuracy.py \
    --model-type ours \
    --ckpt outputs/wave1_v2/ckpt.best.pt \
    --val-parquet data/wave1/needle.val.parquet \
    --max-docs 100 \
    --output outputs/em_v2_best.json \
    || echo "[$(TS)] WARN: EM bench (best ckpt) failed"
fi

# ── 4. Launch Wave 2 on wildchat_long, warm-started from Wave 1 final ──
echo "[$(TS)] === Launching Wave 2 SFT (wildchat_long, 4000 steps) ==="
mkdir -p outputs/wave2_v2
PYTHONPATH=. python -u scripts/training/train_wave2.py \
  --data-paths data/wave2/wildchat_long.train.parquet \
  --val-data-paths data/wave2/wildchat_long.val.parquet \
  --checkpoint-in outputs/wave1_v2/ckpt.pt \
  --warm-start \
  --num-steps 4000 \
  --warmup-steps 100 \
  --prior-loss-weight 0.1 \
  --load-balance-coef 1e-3 \
  --z-loss-coef 0 \
  --checkpoint-out outputs/wave2_v2/ckpt.pt \
  --save-every 500 \
  --log-every 50 \
  --plot-path outputs/wave2_v2/training.png \
  > outputs/wave2_v2/training.log 2>&1

echo "[$(TS)] Wave 2 exited with code $?"
echo "[$(TS)] chain done."
