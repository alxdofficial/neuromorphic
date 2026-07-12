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
MANIFEST=data/MANIFEST.txt

# FATAL (not WARN) on a missing source: a partial tarball would silently train a degraded/reweighted mix
# (audit #6). Also write a per-source manifest (file count, jsonl row count, bytes) INTO the tarball so
# the pod can assert the untarred content matches — catching present-but-truncated dirs that a bare
# `[ -d ]` check misses.
echo "[pack] sources: ${TRAIN_SOURCES[*]}"
: > "$MANIFEST"
present=()
for s in "${TRAIN_SOURCES[@]}"; do
  [ -d "data/$s" ] || { echo "[pack] FATAL missing data/$s — refusing to build a partial tarball"; exit 1; }
  fc=$(find "data/$s" -type f | wc -l)
  rc=$(find "data/$s" -type f -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
  by=$(du -sb "data/$s" | cut -f1)
  printf '%s\t%s\t%s\t%s\n' "$s" "$fc" "$rc" "$by" >> "$MANIFEST"
  present+=("data/$s")
done
present+=("$MANIFEST")
echo "[pack] manifest:"; cat "$MANIFEST"
echo "[pack] taring $(du -sch "${present[@]}" | tail -1 | cut -f1) → $OUT"
# --exclude dead weight from the tarball (audit): fw_cache_bak/ = a stale 2GB cache backup (no code refs);
# *.tmp = partial cache writes. The ACTIVE fineweb caches (cache/cache/*.tokids.npz + the *.meta-llama*.jsonl
# decode files the mtime-check needs) are KEPT. Parquets are kept as a regenerate-on-invalidation fallback.
tar czf "$OUT" --exclude='*fw_cache_bak*' --exclude='*.tmp' "${present[@]}"
echo "[pack] tarball size: $(du -h "$OUT" | cut -f1)"

echo "[pack] uploading to R2 (neuromorphic/data.tar.gz)…"
scripts/pod/r2.sh up "$OUT" data.tar.gz
echo "[pack] done. Verify:  scripts/pod/r2.sh ls"
