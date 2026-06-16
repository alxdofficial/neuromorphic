#!/usr/bin/env bash
# Re-run ONLY graph_baseline on the 3 objectives with the new PREPEND read
# (competition OFF — the clean read-fix test). Baselines unchanged on disk; compare
# graph_*_prepend vs the original inject-read graph_* and the baselines.
set -uo pipefail
cd /home/alex/code/neuromorphic
COMMON="--backbone HuggingFaceTB/SmolLM2-135M --src-tokenizer meta-llama/Llama-3.2-1B --variants graph_baseline"
declare -a SPECS=(
  "masked_reconstruction:32:4000:mae4k_prepend"
  "conditioned_reconstruction:8:3000:condrecon3k_prepend"
  "continuation:8:3000:cont3k_prepend"
)
for spec in "${SPECS[@]}"; do
  IFS=: read -r T BS STEPS TAG <<< "$spec"
  echo "================ GRAPH-PREPEND $T  steps=$STEPS bs=$BS tag=$TAG  $(date -u) ================"
  python scripts/train/train.py --task "$T" --batch-size "$BS" --steps "$STEPS" \
      --out-tag "$TAG" $COMMON
  echo "================ EXIT $T = $?  $(date -u) ================"
done
echo "ALL GRAPH-PREPEND RUNS DONE $(date -u)"
