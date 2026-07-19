#!/usr/bin/env bash
# Pin a command to THIS project's CPU partition so it shares the box with the HALO project without
# either starving the other. Shared box: AMD Ryzen 9 7900X — 12 physical cores / 24 logical.
# Physical core N = logical CPUs N and N+12 (e.g. core 5 = CPUs 5 and 17).
#
#   HALO         (do NOT use): physical 0-3   -> logical 0-3,12-15
#   Neuromorphic (OURS):       physical 4-11  -> logical 4-11,16-23    (8 physical / 16 logical)
#
# MUST pin at LAUNCH: children + worker threads inherit the affinity, but `taskset -p` on an
# already-running PID does NOT hold (this job spawns fresh PIDs every few minutes, each defaulting
# back to all 24 cores). Launching under taskset fixes that via inheritance.
#
# Usage: scripts/cpu_capped.sh -- <command...>
#        scripts/cpu_capped.sh "4-11,16-23" -- <command...>   # override the core list if HALO re-carves
set -euo pipefail

CORES="4-11,16-23"          # Neuromorphic partition — stay OFF HALO's 0-3,12-15
if [[ "${1:-}" =~ ^[0-9][0-9,-]*$ ]]; then CORES="$1"; shift; fi   # starts with a digit (so "--" isn't matched)
[[ "${1:-}" == "--" ]] && shift

THREADS=8                   # = our 8 PHYSICAL cores; optimal for BLAS/torch, avoids hyperthread oversubscription
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS" VECLIB_MAXIMUM_THREADS="$THREADS"

echo "[cpu_capped] taskset -c $CORES (Neuromorphic partition; OFF HALO 0-3,12-15), BLAS/torch threads=$THREADS" >&2
exec taskset -c "$CORES" nice -n 5 "$@"
