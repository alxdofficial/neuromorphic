#!/usr/bin/env bash
# PULL — download a run's results from R2 into outputs/memory/ so the local band/gate
# eval and diagnostics see them exactly as a local run would.
#
# R2 layout: results/<run_id>/<arm>/{run.jsonl, <arm>.last.pt, summary.json, bootstrap.log, _STATUS}
# Local layout it writes (matches the trainer's own):
#   outputs/memory/<run_id>_<arm>/jsonl/<arm>.jsonl
#   outputs/memory/<run_id>_<arm>/ckpts/<arm>.last.pt
#   outputs/memory/<run_id>_<arm>/bootstrap.log
#   outputs/memory/<run_id>_summary.json   (last arm's summary; per-arm kept too)
#
# Usage:
#   scripts/pod/pull_results.sh <run_id>            # all arms
#   scripts/pod/pull_results.sh <run_id> slotgraph_baseline icae_baseline
#   scripts/pod/pull_results.sh <run_id> --status   # just print each arm's _STATUS
set -euo pipefail
cd "$(dirname "$0")/../.."
R2="scripts/pod/r2.sh"
RUN_ID="${1:?usage: pull_results.sh <run_id> [arms...] | <run_id> --status}"; shift || true

DEFAULT_ARMS=(icae_baseline autocompressor_baseline titans_baseline
              gisting_baseline memoryllm_baseline slotgraph_baseline)

if [ "${1:-}" = "--status" ]; then
  echo "[pull] statuses for $RUN_ID:"
  for a in "${DEFAULT_ARMS[@]}"; do
    s="$($R2 raw cp "s3://$(source ~/.config/r2/credentials; echo "$R2_BUCKET/${R2_PREFIX:-neuromorphic}")/results/$RUN_ID/$a/_STATUS" - 2>/dev/null || echo '—')"
    printf '  %-26s %s\n' "$a" "$s"
  done
  exit 0
fi

ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=("${DEFAULT_ARMS[@]}")
for a in "${ARMS[@]}"; do
  base="results/$RUN_ID/$a"
  dst="outputs/memory/${RUN_ID}_${a}"
  mkdir -p "$dst/jsonl" "$dst/ckpts"
  echo "[pull] $a → $dst"
  $R2 down "$base/run.jsonl"      "$dst/jsonl/$a.jsonl"   2>/dev/null || echo "   (no run.jsonl yet)"
  $R2 down "$base/${a}.last.pt"   "$dst/ckpts/$a.last.pt" 2>/dev/null || echo "   (no last checkpoint yet)"
  $R2 down "$base/${a}.best.pt"   "$dst/ckpts/$a.best.pt" 2>/dev/null || true
  $R2 down "$base/summary.json"   "outputs/memory/${RUN_ID}_summary.json" 2>/dev/null || true
  $R2 down "$base/bootstrap.log"  "$dst/bootstrap.log"    2>/dev/null || true
done
echo "[pull] done. Evaluate with:"
echo "  python scripts/diagnostics/mixed/mixed_band_gate_eval.py --out-tag $RUN_ID"
