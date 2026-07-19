#!/usr/bin/env bash
# Run a command capped to a fraction of this machine's CPU, so it shares nicely with other projects.
# Combines: taskset (hard core pinning — a HARD ceiling), BLAS/torch thread limits (so numpy/torch/HF
# don't oversubscribe beyond the pinned cores), and nice (yields to other work under contention).
#
# Usage:   scripts/cpu_capped.sh [PERCENT] -- <command...>
#   PERCENT defaults to 40. Example:
#   scripts/cpu_capped.sh 40 -- .venv/bin/python scripts/baselines/run_api_eval.py --dataset longmemeval
#   scripts/cpu_capped.sh -- .venv/bin/python ...        # 40% default
set -euo pipefail

PCT=40
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then PCT="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift

NCPU="$(nproc)"
CAP=$(( NCPU * PCT / 100 )); (( CAP < 1 )) && CAP=1
THREADS=$(( CAP > 1 ? CAP - 1 : 1 ))     # leave a core of headroom for the event loop / IO

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"

echo "[cpu_capped] pinning to cores 0-$((CAP-1)) of $NCPU (~${PCT}%), BLAS/torch threads=$THREADS, nice 10" >&2
exec taskset -c "0-$((CAP-1))" nice -n 10 "$@"
