# Moved

This thesis now lives at [`docs/design/kvgraph/THESIS.md`](kvgraph/THESIS.md), alongside the build spec
[`docs/design/kvgraph/BUILD.md`](kvgraph/BUILD.md) and the code in `src/memory/models/kvgraph/`.

Renamed because the design is no longer slot-based: nodes are linguistic particulars (entities and reified
events) grouped from the KV cache, not learned slots. Keeping the old name invited conflation with the
retired slotgraph line, which the thesis explicitly diagnoses as a *different* (L2/semantic) substrate
aimed at the wrong job.
