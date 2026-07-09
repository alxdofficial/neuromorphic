# Slotgraph memory — weekly progress update

**Goal of the work:** build a graph-structured compressed memory that *binds* relational facts (stores
*what is connected to what*), read by a frozen SmolLM2-135M. Benchmark: 4 tasks — `mae` (reconstruction),
`babi` (relational QA), `continuation`, `condrecon_bio` — at 1024→32 compression, 4000 steps.

**Defining "binding" (the central metric).** We probe the memory with three losses on the *same* example:
- **REAL** — the model's own memory for this example.
- **OFF** — memory removed. **`OFF − REAL > 0` ⇒ the memory is *used*** (it contributes to the answer).
- **SHUF** — *another* example's memory swapped in. **`SHUF − REAL > 0` ⇒ the memory is *example-specific***.

**Binding = both gaps are clearly positive:** the memory is genuinely *used* **and** it has to be *this*
example's memory. The failure mode we keep hitting is **"used but not bound"** — `OFF−REAL` large (memory
essential) but `SHUF−REAL ≈ 0` (any example's memory works just as well) — i.e. the model leans on a
*generic prior*, not on example-specific structure. That is pooling / "membership only." (`mae`/
`continuation` are the reliable binding tasks; on `babi` the SHUF gap is partly diluted because
consecutive questions often share answers.)

![Binding gate per task — current model](slotgraph_binding_gate.png)
***[Current model — graph redesign].*** Binding needs both gaps. `mae`/`continuation` (green) show both →
they bind. `babi`/`condrecon` (red) show a large OFF gap but ~0 SHUF gap → "used but not bound" (the
memory is essential, but any example's memory works equally well).

**How the structural metrics are measured** (used in the diagnostics later):
- **Effective rank** (participation ratio): for a set of vectors, `(Σλ)² / Σλ²` over the eigenvalues `λ`
  of their covariance — the *effective number of distinct directions* the set spans. ≈1 means the vectors
  are essentially one repeated vector (collapsed); high means they are spread / distinct. *Why it matters:*
  a memory the LM reads with effective rank ≈1 is functionally a single vector — it cannot carry per-item
  distinctions, which is the hallmark of pooling.
- **Routing diversity**: for each edge, the entropy (normalized by `ln N`) of *which node it selects across
  different inputs*, averaged over edges. High ⇒ an edge connects to different nodes for different inputs
  (**input-dependent** routing); ≈0 ⇒ it wires to the same node regardless of input. *Why it matters:*
  relational binding requires the connections to depend on *which entities are in this example*, so
  input-dependent routing is the prerequisite for binding.

This document walks the model's evolution **one change at a time** — each step says *what we added*,
*why*, and *what we observed* — so the reasoning chain from the June 24 model to today is clear.

---

## Step 0 — The starting point (June 24): ICAE + graph-inspired identity

**What it was.** At its core, **ICAE**: learnable memory slots appended to the passage, compressed by the
frozen LM's own layers, and read by *prepending* them. On top: a fixed node/edge labelling of the slots,
content-addressed routing, and fixed orthonormal **identity embeddings** per slot.

**What we observed.** It **bound well** — the strongest binder in our cohort (`mae` SHUF−REAL ≈ **0.46**,
`continuation` ≈ **0.49**, `babi` exact-match ≈ **30%**) — and the identity embeddings gave a small, free
gain over plain ICAE. **But** turning the graph machinery *off* matched turning it *on*: the topology was
**inert**. The model was binding as a **flat bank of slots**; the "graph" was decorative.

**The question this raised.** Can we make the graph **topology itself** carry the memory — so that the
*relations* between slots, not just the flat slot contents, do the binding?

---

## Step 1 — Make the topology load-bearing (learned graph encoder + hard endpoint selection)

**Added.** Replaced the inert read with a real graph: a learned encoder where each **edge hard-selects two
endpoint nodes** (Gumbel straight-through: discrete forward, soft gradient), and the read gathers content
*through* those endpoints. We also **decoupled the graph dimension** (`d=64`) from the LM's (`d=576`) to
force composition (no single slot can hold a full answer).

**Why.** So that *which node an edge connects to* actually changes the output — i.e., the topology is part
of the computation, not decoration.

**Observed.** A new failure surfaced — the **"edge-state bypass"**: the read rode entirely on a free
per-edge channel and *ignored* the selected endpoints. Selection still wasn't materially affecting the
loss, so the topology still wasn't doing work.

---

## Step 2 — Relation-operator edge readout (close the bypass)

**Added.** Turn each edge into a vector by a **multiplicative bind** of its two endpoints and its relation,
`out = W_o[(W_s·src) ⊙ (W_r·rel) ⊙ (W_d·dst)]`, with no bias term — so zeroing the endpoints zeros the
output. (This is the "operation to turn edges into vectors.")

**Why.** Structurally force the read to depend on the selected endpoints — there is no longer a free
channel to bypass them.

**Observed.** The bypass closed (verified numerically ≈ 0), and `mae` binding rose to ≈ **0.20**. **But** a
new failure on `babi`: **hub collapse** — all edges selected the *same ~6 nodes*, so every edge read
near-identical content and the memory the LM saw collapsed to ~rank-1 (pooling, by a different route).

![Hub collapse — UMAP of one example's slots](slotgraph_hub_collapse_umap.png)
***[June 24-era ICAE+identity architecture].*** One example's slots (UMAP). Every edge funnels into a tiny
cluster of nodes (0–3); the other slots are unused. Diagnostically this shows up as node-target usage
spiking on a single slot and the read-token effective rank sitting at ~3 (out of 32). The graph wires
itself into a hub instead of a distributed structure — i.e. the topology is degenerate, which is *why* it
was inert and the binding fell back on the flat slots. (The current model also collapses the read-rank —
see the depth×time figure — so this failure persists across both architectures.)

---

## Step 3 — Competition (slot-attention) writes

**Added.** Made the slots **compete** for input content (softmax over slots, not over inputs), so each slot
specializes to a different aspect.

**Why.** We suspected the slots were redundant — all absorbing the same content — which would explain the
hub.

**Observed.** **No effect** on `babi` (exact-match unchanged at ≈ 0.24). This was informative: the slot
*contents* were already distinct. The collapse was in the **selection** (*which* nodes get connected), not
in the content. So we'd been fixing the wrong axis.

---

## Step 4 — Balanced selection (Gumbel-Sinkhorn)

**Added.** A balancing step on the edge→node selection (node-side normalization) so that no single node can
be claimed by all the edges.

**Why.** Attack the hub directly — cap how many edges may connect to each node, forcing them to spread.

**Technique note (a real lesson).** While doing this we found that a **learnable scalar temperature cannot
sharpen** the selection — under Adam a lone scalar barely moves (it drifted 8.00 → 8.02 over a full 4000
steps). We moved the sharpness into the projection weights (scaled dot-product) instead.

**Observed.** The hub broke cleanly — nodes used went from ~6 to **~121** (out of 144). **But still no
binding:** `babi` SHUF−REAL stayed ≈ 0 and exact-match stayed ≈ 0.24. The memory spread across many nodes
but remained generic — *"spread, but doesn't bind."*

---

## Step 5 — Per-layer, per-step diagnostics (find the root cause)

**Added.** Instrumentation that reports metrics for **each write layer** at **each training checkpoint**,
to locate exactly *where* and *when* binding fails.

**Why.** Three different fixes had each solved their local symptom without improving binding — we needed to
see the underlying mechanism, not chase symptoms.

**Observed (the key finding).** On `babi`, the **routing never becomes input-dependent** — `routing_diversity`
sits at ~0.02 from step 1 and never rises. The relational binding signal *never forms*, at any layer, at any
point in training. Meanwhile the read-rank collapses both early in training and layer-by-layer, while the
slot *contents* stay distinct. The decisive contrast: on `mae` (which **does** bind) routing_diversity rises
to ~0.10 and SHUF−REAL rises to ~0.19 — **binding appears exactly when routing becomes input-dependent**,
and on the relational task it never does.

![Depth × time diagnostics](slotgraph_depthtime.png)
***[Current model — graph redesign].*** Left: read-token effective rank by write layer over training
(collapses across depth and early in training). Middle: routing diversity — `babi` (red) stays flat ~0.02,
`mae` (green) rises. Right: binding (SHUF−REAL) — `babi` flat at 0, `mae` rises to ~0.19. Binding tracks
routing diversity. *(This diagnostic is current-model only: it traces the GT write layers, which the June
24 model does not have.)*

Supporting tables (relational task, `babi`):

**Read-token effective rank — collapses across depth AND early in training** (≈1 = fully pooled):

| layer ↓ / step → | 500 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|
| L0 | 22.2 | 13.5 | 10.0 | 9.9 | 9.1 |
| L1 | 24.2 | 11.4 | 7.1 | 6.5 | 5.9 |
| L2 | 20.0 | 9.6 | 5.4 | 5.2 | 4.5 |
| **L3 (what the LM reads)** | 16.6 | 6.9 | 3.4 | 3.3 | **3.0** |

**Routing diversity — flat ~0.02, never lifts** (this is the binding signal that should grow):

| layer ↓ / step → | 500 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|
| L0 | 0.016 | 0.012 | 0.011 | 0.011 | 0.007 |
| L3 | 0.026 | 0.030 | 0.027 | 0.036 | 0.032 |

**Slot-content effective rank — stays high** (so content is NOT the problem):

| step → | 500 | 2000 | 4000 |
|---|---|---|---|
| node content (L0) | 41.8 | 41.0 | 40.9 |
| node content (L3) | 37.5 | 34.3 | 33.1 |

---

## Step 6 — Controls + stepping back (what it all means)

**Added.** A structure-off control, and a re-read of the original June 24 cohort numbers.

**Observed.**
- **`babi` is the textbook "used but not bound."** Removing the memory is catastrophic there
  (`OFF−REAL` ≈ **+4 to +8** — the task is impossible without it), yet swapping in another example's memory
  costs almost nothing (`SHUF−REAL` ≈ **0**). Both gaps are *not* present — only the "used" one — so by our
  definition it is **not binding**: the model leans on the memory as a *generic* resource, not as this
  story's specific facts.
- **Effective rank is orthogonal to binding** — a high-capacity control had read-rank ~36 (high) yet
  `babi` SHUF−REAL ≈ 0 (no binding). Chasing rank does not produce binding.
- **It is not a gradient bug** — selection gradient flows normally (content-vs-selection magnitude ratio
  ≈ 1×, vs ≈30,000× in an earlier broken version). Routing fails to specialize *despite* healthy gradient,
  because **pooling already minimizes the loss** — nothing pushes routing to become input-dependent.
- **The June 24 model was the cohort's best binder**, and our whole redesign had *regressed* binding. The
  learned graph topology was inert at best and harmful at worst; **the binding was always the flat ICAE
  memory + the identity-embedding trick.**

---

## Where this leaves us

| binding signal | June 24 (ICAE + identity) | Latest (full graph redesign) |
|---|---|---|
| `mae` SHUF−REAL | **0.46** | 0.20 |
| `continuation` SHUF−REAL | **0.49** | 0.11 |
| `babi` example-specific binding | present | **≈ 0** |
| `babi` exact-match | ~30% | ~24% |

![Binding gate — June 24 vs current](slotgraph_binding_gate_compare.png)
***[Side-by-side: June 24 (3-seed) vs current (1-seed)].*** Same metric, both models. Right panel is the
headline: the redesign **regressed binding** on the reliable tasks (`mae` 0.46→0.20, `continuation`
0.49→0.10); `babi`/`condrecon` were "used but not bound" in *both*. (June 24 bars show 3-seed mean ± std;
current is a single seed — the *direction* is robust, the exact current values are not yet seed-averaged.)

**Summary of the story.** Each engineered step *succeeded at its narrow target* — the bypass closed, the hub
broke, the slots differentiated — yet binding never improved, and the more we forced genuine graph structure
the *worse* the relational task got. The diagnostics explain why: the routing never becomes input-dependent,
so the relational binding signal never forms — and it never has to, because a generic (pooled) memory is
already enough to minimize the loss. The architecture *can* bind when the task rewards it (`mae`/
`continuation`); it cannot form example-specific *relational* structure on `babi`.

**Next steps under consideration.**
- **Disentangle read vs. write**: a prepend, full-dimension read with the structure kept (param-matched),
  to test whether the regression was the read mechanism or the learned graph *write*.
- **Re-confirm the ICAE + identity model** as the standing baseline.
- **Reconsider where structure can help**: add structure to the *read* over ICAE-written slots, or move to a
  regime flat banks can't handle (streaming write/update over time, or binding installed in the write).
