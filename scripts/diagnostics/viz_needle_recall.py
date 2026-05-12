#!/usr/bin/env python3
"""viz_needle_recall.py — Tier 3: needle write→read overlap as a memory probe.

**STUB — not implemented yet.** This is the cleanest direct test we have
that the memory manifold is actually doing useful work (vs Llama just
attending within its own context window).

## Goal

For each needle val document:
  - At the **needle position** (planted fact, ~2-32K tokens into the doc),
    record which concepts the **write** module deposits into.
  - At the **answer position** (query+answer, end of doc), record which
    concepts the **read** module retrieves from.
  - Compute their **Jaccard overlap**.
  - Stratify by needle→answer distance bin: 3K / 8K / 16K / 32K.

If memory is doing real work:
  - At distance=3K, Llama can attend to the needle directly — overlap
    should be incidental (depends on routing, not on memory actually
    carrying info).
  - At distance=16K/32K, Llama physically can't see the needle — high
    overlap is the only way the answer can be predicted. Sharp signal.

If Δ(overlap) between near and far distances is large + correlates with
answer-loss, that's a clean memory-contribution proof.

## What it should produce

  A. **overlap-vs-distance scatter** (`outputs/wave{N}/viz_needle_overlap.png`)
     X: needle→answer token distance (3K / 8K / 16K / 32K bins)
     Y: Jaccard overlap = |write_concepts ∩ read_concepts| / |union|
     One dot per doc; faint trend line.

  B. **overlap-vs-answer-loss scatter**
     X: per-doc Jaccard overlap
     Y: per-doc answer-span CE
     Expected: negative correlation. Higher overlap → lower loss.

  C. **per-concept "needle carrier" frequency**
     Which specific concepts most often appear in BOTH write@needle and
     read@answer for the same doc? These are the "memory carrier" concepts.
     Cross-reference with the concept→token table from Tier 2 to see if
     they correspond to "noun-y" / "identifier-y" tokens.

## Implementation outline

```
1. Load needle val parquet (need answer-span metadata + needle_pos_chars
   + query_pos_chars; this is already produced by synthesize_needle.py).
2. Build IntegratedLM, load ckpt.
3. For each needle doc:
     a. Find the window index that contains needle_pos_chars (call it W_n).
     b. Find the window index that contains query_pos_chars (W_q).
     c. Forward pass through the doc (eval mode, hard_routing=True).
     d. Capture write_visited_ids[W_n] (set A) and read_visited_ids[W_q]
        (set B). Each is [BS=1, J=4, K=8] → flatten to {concept_ids}.
     e. Jaccard = |A ∩ B| / |A ∪ B|.
     f. Also capture answer-span CE loss for the doc.
     g. Compute needle→answer distance in tokens.
4. Bucket by distance, plot scatters.
```

## Cost and safety

- **CPU-only** for analysis, **GPU required** for the forward pass.
- **NOT SAFE during active training** — run after a wave finishes or
  pause training.
- Inference cost: ~10 min for the ~256-doc needle val set.
- The full needle val parquet has all the per-doc metadata we need
  (see `src.trajectory_memory.data.needle_haystack.NeedleDoc` fields:
  `needle_pos_chars`, `query_pos_chars`, `target_distance`, `answer`).

## Confounds to address

- Routing is **stochastic** even at hard=True if Gumbel temp > 0 in eval.
  Force argmax (tau→0) for this analysis to remove that variance.
- Overlap might be high simply because some concepts are universal /
  always-on. Worth showing a **null distribution**: jaccard between
  random concept sets sampled from `usage_ema` weights. The real
  overlap should beat that null by a clear margin.
- Concept identity drifts over training. Use the **same ckpt** for both
  the write and read computations.

See `scripts/diagnostics/README.md` for the full Tier 1-4 roadmap.
"""

raise SystemExit(
    "viz_needle_recall.py is a Tier 3 stub — not yet implemented. "
    "See scripts/diagnostics/README.md for the roadmap."
)
