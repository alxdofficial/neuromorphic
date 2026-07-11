#!/usr/bin/env bash
# PULL — download a run's results from R2 into outputs/memory/ so the local band/gate
# eval and diagnostics see them exactly as a local run would.
#
# R2 layout MIRRORS the local run dir: results/<run_id>/<arm>/{jsonl/<arm>.jsonl,
#   ckpts/<arm>.{last,best,step<N>}.pt, bootstrap.log, _STATUS, summary.json}
# This down-SYNCs each arm's prefix into outputs/memory/<run_id>_<arm>/ so the local band/gate eval +
# diagnostics see it exactly as a local run would — ckpts/ + jsonl/ subdirs, including EVERY milestone.
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
set -a; source ~/.config/r2/credentials; set +a
S3="s3://$R2_BUCKET/${R2_PREFIX:-neuromorphic}"
for a in "${ARMS[@]}"; do
  dst="outputs/memory/${RUN_ID}_${a}"
  mkdir -p "$dst"
  echo "[pull] $a → $dst"
  # mirror the arm's R2 prefix: jsonl/, ckpts/ (last/best + every .step<N>.pt), bootstrap.log, _STATUS
  $R2 raw sync "$S3/results/$RUN_ID/$a/" "$dst/" --no-progress 2>/dev/null || echo "   (nothing on R2 yet)"
  [ -f "$dst/summary.json" ] && cp "$dst/summary.json" "outputs/memory/${RUN_ID}_summary.json" || true
done
echo "[pull] done. Evaluate with:"
echo "  python scripts/diagnostics/mixed/mixed_band_gate_eval.py --out-tag $RUN_ID"
