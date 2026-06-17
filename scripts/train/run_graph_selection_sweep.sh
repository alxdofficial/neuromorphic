#!/usr/bin/env bash
# Graph-only MAE attribution sweep for the node-selection fix. Compares against the
# existing mae4k_prepend (d_graph=256, N=1024, softmax, ~6M). Cells (d_graph:N:alpha):
#   C  256:1024:1.5  (~6M, PARAM-MATCHED → clean commitment effect)
#   A  576:1024:1.0  (~26M, full-width softmax reference; param-CONFOUNDED)
#   B  576: 512:1.5  (~26M, full width + tighter vocab + commitment — the candidate config)
# (d_graph stays 576 for B per design — full attention width, 2-3 layers; baselines get
#  matched UP to this tier for the fair verdict, only if the graph improves here.)
# Read back: top-1 mass, nodes_used, edge_effrank, mem_effrank, zero-endpoints Δ, val loss.
set -uo pipefail
cd /home/alex/code/neuromorphic
COMMON="--task masked_reconstruction --backbone HuggingFaceTB/SmolLM2-135M --src-tokenizer meta-llama/Llama-3.2-1B --variants graph_baseline --steps 4000 --batch-size 32"
declare -a SPECS=(
  "256:1024:1.5:mae_dg256_e15"
  "576:1024:1.0:mae_dg576_sm"
  "576:512:1.5:mae_dg576_n512_e15"
)
for spec in "${SPECS[@]}"; do
  IFS=: read -r DG NN ALPHA TAG <<< "$spec"
  echo "================ SWEEP d_graph=$DG N=$NN entmax=$ALPHA tag=$TAG  $(date -u) ================"
  python scripts/train/train.py $COMMON --graph-d-graph "$DG" --graph-n-nodes "$NN" --graph-entmax-alpha "$ALPHA" --out-tag "$TAG"
  echo "================ EXIT $TAG = $?  $(date -u) ================"
done
echo "ALL SELECTION-SWEEP RUNS DONE $(date -u)"
