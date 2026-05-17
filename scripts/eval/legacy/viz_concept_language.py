#!/usr/bin/env python3
"""viz_concept_language.py — Tier 2: concept ↔ language correspondence.

**STUB — not implemented yet.** This script is the design for the next
inspection step after Tier 1 (`viz_manifold.py`).

## Goal

Answer: are the 4096 concepts *semantic* (each one fires for a coherent
set of tokens / sentences), or just routing slots that mean nothing?

The Tier 1 plot tells us about manifold structure in vector space and
usage statistics. Tier 2 ties those concepts back to *language* — what
do they actually represent.

## What it should produce

Three deliverables from one inference pass:

  A. **concept → top-K tokens** table (`outputs/wave{N}/concept_tokens.txt`)
     For each of the 4096 concepts, the 10 tokens that most often co-occur
     in windows that route through that concept (read or write visit).
     Example expected output if concepts have learned anything:
         concept #042 (usage 0.0087): "France", "Paris", "European",
                                       "capital", "Macron", ...
         concept #1337 (usage 0.0012): "function", "return", "def", "()", ...

  B. **per-source usage barplot** (`outputs/wave{N}/viz_per_source_usage.png`)
     Same 4096 concepts on x-axis, four overlaid histograms — one per
     data source (needle / fineweb_edu / wikipedia_en / slimpajama_6b).
     Reveals whether different domains light up different concepts (= good,
     semantic specialization) or all sources pile onto the same 50 dominant
     concepts (= bad, generic routing).

  C. **trajectory traces** (`outputs/wave{N}/viz_trajectories.png`)
     For 3-5 hand-picked documents, plot all J read trajectories across
     all windows on the ring topology. Color by window index.
     Reveals whether trajectories evolve coherently over a doc or thrash.

## Implementation outline

```
1. Build IntegratedLM from ckpt + load manifold state.
2. Iterate ~100 docs from each source (need: ~400 docs total).
3. For each window in each doc:
     - Run model forward (eval mode, hard_routing=True for cleanest signal)
     - Capture: read_visited_ids [BS, J, K_read]
                write_visited_ids [BS, J, K_write]
                window's token_ids
                doc source label
4. Build co-occurrence matrix [N=4096, V=128K_vocab]:
     - For each (window, concept) pair, increment counts for all tokens
       in that window. Use sparse matrix (scipy or torch sparse) since
       most entries are zero.
5. Per concept, find top-K tokens (highest column count); decode via
   tokenizer.
6. Per source, accumulate per-concept usage; produce barplot.
7. For trajectory traces, just save the read_visited_ids of the picked
   docs and render on ring topology (see panel_ring_topology in
   viz_manifold.py for layout).
```

## Cost and safety

- **CPU-only** for analysis, but **GPU required** for the forward pass.
- **NOT SAFE to run during active Wave 1/2 training** — it will compete
  for VRAM. Run after training pauses or on a different GPU.
- Inference cost: ~20-30 min for 400 docs × 8 windows × forward.
- Output disk: ~30 MB for the [N, V] sparse matrix dump.

## Open questions

- Use the latest `ckpt.pt` or `ckpt.best.pt`? Probably best.
- Should we compute the matrix on **read** visits, **write** visits, or
  both? Write captures "what got encoded"; read captures "what got
  retrieved." Two different stories — probably want both, with separate
  output files.
- Tokens are wordpieces in Llama-3.2's BPE — top tokens for a concept
  might be sub-word like "##ation". Need a strategy: merge BPE pieces
  within a window before counting? Or just report wordpieces as-is?

See `scripts/diagnostics/README.md` for the full Tier 1-4 roadmap.
"""

raise SystemExit(
    "viz_concept_language.py is a Tier 2 stub — not yet implemented. "
    "See scripts/diagnostics/README.md for the roadmap."
)
