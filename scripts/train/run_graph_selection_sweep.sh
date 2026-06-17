#!/usr/bin/env bash
# Graph-only MAE attribution sweep for the node-selection fix. Compares against the
# existing mae4k_prepend (d_graph=256, softmax, ~6M). Cells:
#   C  d_graph=256, entmax-1.5  (~6M, PARAM-MATCHED → clean commitment effect)
#   A  d_graph=576, softmax     (~26M, max-width reference; param-CONFOUNDED)
#   B  d_graph=384, entmax-1.5  (~14M, moderate width + commitment — the candidate config)
# Read back: top-1 mass, nodes_used, edge_effrank, mem_effrank, zero-endpoints Δ, val loss.
set -uo pipefail
cd /home/alex/code/neuromorphic
COMMON="--task masked_reconstruction --backbone HuggingFaceTB/SmolLM2-135M --src-tokenizer meta-llama/Llama-3.2-1B --variants graph_baseline --steps 4000 --batch-size 32"
declare -a SPECS=(
  "256:1.5:mae_dg256_e15"
  "576:1.0:mae_dg576_sm"
  "384:1.5:mae_dg384_e15"
)
for spec in "${SPECS[@]}"; do
  IFS=: read -r DG ALPHA TAG <<< "$spec"
  echo "================ SWEEP d_graph=$DG entmax=$ALPHA tag=$TAG  $(date -u) ================"
  python scripts/train/train.py $COMMON --graph-d-graph "$DG" --graph-entmax-alpha "$ALPHA" --out-tag "$TAG"
  echo "================ EXIT $TAG = $?  $(date -u) ================"
done
echo "ALL SELECTION-SWEEP RUNS DONE $(date -u)"
