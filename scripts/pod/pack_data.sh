#!/usr/bin/env bash
# Package the TRAINING-data subset into data.tar.gz and upload to R2. Run ONCE locally.
# Excludes phase-2 / unused sources (eval/, swe_trajectories/, quality/, babilong_train/, perltqa/,
# wikibigedit/, longcite/, govreport/, pg19/, msc/, qasper/, lmsys_chat/, wildchat/, ruler_*/) so the
# tarball is ~lean. babi is fetched from HF at runtime; bio is procedurally generated (no on-disk data).
#
# The 5-task training mix sources (mixes.DEFAULT_TRAIN_MIX):
#   reconstruct  -> fineweb_edu
#   continuation -> multicorpus = fineweb_edu + pile + redpajama + code
#   doc_qa       -> qa_multi     = squad + triviaqa + hotpot_train + musique_train + multiwoz
#   babi         -> HF Muennighoff/babi (runtime; NOT tarred)
#   fact_recall  -> bio (procedural; NOT tarred)
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root

TRAIN_SOURCES=(fineweb_edu pile redpajama code squad triviaqa hotpot_train musique_train multiwoz)
OUT=/tmp/neuromorphic_data.tar.gz

echo "[pack] sources: ${TRAIN_SOURCES[*]}"
present=(); for s in "${TRAIN_SOURCES[@]}"; do
  if [ -d "data/$s" ]; then present+=("data/$s"); else echo "[pack] WARN missing data/$s"; fi
done
echo "[pack] taring $(du -sch "${present[@]}" | tail -1 | cut -f1) → $OUT"
tar czf "$OUT" "${present[@]}"
echo "[pack] tarball size: $(du -h "$OUT" | cut -f1)"

echo "[pack] uploading to R2 (neuromorphic/data.tar.gz)…"
scripts/pod/r2.sh up "$OUT" data.tar.gz
echo "[pack] done. Verify:  scripts/pod/r2.sh ls"
