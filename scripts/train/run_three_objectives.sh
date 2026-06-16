#!/usr/bin/env bash
# Train the active cohort (graph + icae/ccm/autocompressor/beacon + vanilla floor/ceiling)
# on 3 objectives SEQUENTIALLY, frozen SmolLM2-135M. Per-objective steps/batch from the
# measured ~1hr-cohort budget (scripts/diagnostics/smoke_all_objectives.py / project memory).
#   masked_reconstruction   4000 @ BS32   (8:1, ctx<=128)
#   conditioned_reconstruction 3000 @ BS8 (30:1, ctx auto->1024)
#   continuation            3000 @ BS8    (30:1, compress 1024 / predict 512)
# QA intentionally excluded. Outputs: outputs/memory/<tag>_<variant>/.
set -uo pipefail
cd /home/alex/code/neuromorphic

COMMON="--backbone HuggingFaceTB/SmolLM2-135M --src-tokenizer meta-llama/Llama-3.2-1B"

declare -a SPECS=(
  "masked_reconstruction:32:4000:mae4k"
  "conditioned_reconstruction:8:3000:condrecon3k"
  "continuation:8:3000:cont3k"
)

for spec in "${SPECS[@]}"; do
  IFS=: read -r T BS STEPS TAG <<< "$spec"
  echo "================ TRAIN $T  steps=$STEPS bs=$BS tag=$TAG  $(date -u) ================"
  python scripts/train/train.py --task "$T" --batch-size "$BS" --steps "$STEPS" \
      --out-tag "$TAG" $COMMON
  echo "================ EXIT $T = $?  $(date -u) ================"
done
echo "ALL RUNS DONE $(date -u)"
