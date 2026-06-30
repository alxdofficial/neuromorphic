# FurlGraph — design memo (working name)

*A compressed graph memory that **folds the input token-chain into a graph** instead of inventing
topology in fixed slots. Captures the full design discussion; build-later.*

---

## 0. TL;DR

Every prior graph-memory design failed the same way: it had to **invent** example-specific topology in
random/fixed slots, and since pooling already minimized the loss, the routing never became
input-dependent and the memory pooled ("used but not bound"). FurlGraph removes the invention problem:
**the input sequence is already a graph — a chain** — so we *initialize the memory as the input chain and
learn to compress it by merging nodes*, keeping a fixed budget (e.g. 32). Because the memory is derived
from the input, it is example-specific **by construction**, and over-merging is **punished by the
compression loss** (pooling is no longer free). One learned primitive: *each node chooses a node to merge
onto*. No edge states (the historical escape hatch); edges are pure identity pointers. Read = prepend, at
the LM's own dimension (576).

---

## 1. Why a new design — what we learned from slotgraph

Diagnostics on the slotgraph line (operator read → competition writes → Gumbel-Sinkhorn) established:

- **The membership wall.** Binding requires *both* gaps: `OFF−REAL > 0` (memory is used) **and**
  `SHUF−REAL > 0` (memory is example-specific). The relational task (babi) was always **"used but not
  bound"**: `OFF−REAL` huge, `SHUF−REAL ≈ 0`.
- **The root cause is not gradient, content, or eff-rank.** Per-layer × per-step tracing showed
  `routing_diversity` (input-dependence of selection) sits at ~0.02 from step 1 and **never rises** — the
  relational binding signal *never forms*. Gradient flows fine (sel-gap ≈ 1×); node content stays distinct
  (eff-rank ~37); and a structure-off control had eff-rank ~36 with `SHUF−REAL ≈ 0` (so **eff-rank ⊥
  binding**). The model never *needs* input-dependent routing because **pooling already minimizes the
  loss**.
- **Forcing structure regressed binding.** The June 24 model (≈ ICAE + identity, inert graph) was the
  cohort's best binder (`mae` SHUF−REAL 0.46); the graph redesign *halved* it (0.20) — the learned graph
  topology was inert at best, harmful at worst.
- **Two things that demonstrably help and we keep:** graph dimension = LM dimension (**d = 576**), and a
  **prepend** read (the easiest-to-train read mechanism).

**Conclusion that motivates FurlGraph:** *inventing and learning topology from scratch is too hard, and
unnecessary, because the input already carries a graph.* Switch the problem from "invent example-specific
structure" (never happened) to "**compress a given example-specific structure**" (learnable, and the loss
rewards preserving distinctions).

---

## 2. Core idea — the input is already a graph

A token window is a **linear chain**: 256 token-nodes connected consecutively. Initialize the memory as
that chain, then **iteratively fold it onto itself** — merging recurring/related tokens onto shared nodes,
re-wiring edges — until it hits the budget (e.g. 32 nodes). Entities and relations are *both* nodes (in
"Mary loves John", `loves` is a node wired to `Mary` and `John`); edges are just adjacency.

Two reasons this should clear the wall:

1. **Binding is free by construction.** The memory is a compression of *this* passage's chain, so another
   example's memory won't fit → `SHUF−REAL` is high without the model having to learn input-dependence.
   The hard question shifts from *"can it bind at all?"* (which beat us) to *"how lossy is the
   compression?"*.
2. **The loss now regularizes against pooling.** The model is doing explicit lossy compression, so
   over-merging two things that needed to stay distinct **directly costs reconstruction/answer loss.** In
   slotgraph, pooling was free; here it is penalized. Corollary: **concentration is good** here
   (many `Mary` mentions → one node = coreference), so — unlike slotgraph — we **do NOT add Sinkhorn /
   anti-hub balancing**; the compression loss alone decides how far to merge.

---

## 3. Architecture

### 3.1 Representation
- **Nodes** = folded token content, at **d = 576**. This is the *only* place information lives.
- **Edges** = **ordered identity pointers**, `edge_vec = id_src ⊕ id_dst`. **No stored edge state.**
  Edge states were the historical "escape hatch" (the model dumped answer-content into the free edge
  vector and ignored topology — measured directly as the *edge_state bypass*). Removing them leaves
  nowhere to hide information except the (input-grounded) nodes. Relation *semantics* live in
  relation-token-nodes; the edge only conveys *which nodes connect* (and direction, via the ordering).
- **Identity / role embeddings**: fixed orthonormal per-slot `id` (the validated free win) + role tags,
  carried through merges. The `id`s are frozen buffers — gradient flows through *which* survivor an edge
  points at (the assignment), not through the id values.

### 3.2 The single learned primitive — merge
Per node, per round, the model emits **two continuous outputs**:
1. a **keep-score** `s_i` — "should I survive this round?"
2. an **assignment distribution** `a_i` over the *survivors* — "if not, who absorbs me?"

A merge = a **gated additive delta** of the merged node's content into its receiver, plus the receiver
**inheriting the merged node's edges** (both driven by `a_i`, see §4).

### 3.3 Per-round loop (within one 256-token window)
1. **Move** — a GT / TokenGT message-passing step updates every node embedding over the current graph
   (so a node "sees" its neighbors before deciding). Uses adjacency; no edge states.
2. **Elect survivors first** — `survivors = topK(s)`. This set is **frozen for the round.**
3. **Assign over survivors only** — every non-survivor assigns across `{survivors}`; survivors
   self-assign. (Restricting the candidate set to survivors is what removes the chain confusion — §4.)
4. **Merge** — gated additive content fold + assignment-driven edge contraction.

Run **gradually** (e.g. 256 → 128 → 64 → 32 over a few rounds) with a **recurrent / shared-weight**
operator — it is one "furl" operation applied repeatedly. (Recurrence also makes **d = 576 affordable**:
one reused block, not the 4 stacked d=576 layers that blew the param budget in slotgraph.)

### 3.4 Read
**Prepend** the final 32 nodes (content) plus the edge id-pointer tokens (structure) to the frozen LM, at
d = 576. Norm-match the prepended tokens to the LM embedding distribution (as ICAE/June-24 did). No
cross-attention; no per-layer injection.

### 3.5 Streaming / persistent state
At an arbitrary point we hold a persistent 32-node graph. A new 256-token window arrives as a fresh chain
in the same embedding space, initially disconnected. Take the **union** (32 + 256 = 288 nodes) and apply
the **same furl operator** back down to 32 — learning to fold the new chain onto existing survivors
(coreference / integration) or evict old ones by folding them together to make room.

**Why 256-token windows (not 1024 at once):** it forces the model to *maintain a persistent state*.
Because folding is non-destructive and the readout can query old content, the **gradient forces the
operator to keep the persistent info** — it cannot fold the old state away without paying loss on old
queries. That's stability–plasticity pressure built in by construction, not by a regularizer.

---

## 4. Gradient design (the crux of the discussion)

The guiding principle: **the model never makes a brittle hard choice that gets silently overridden.** It
emits continuous quantities (keep-scores, assignment logits); the discrete events (top-K survival, hard
assignment, edge contraction) are a **deterministic function** of those with **straight-through**
gradients. So every discrete outcome is a differentiable consequence of knobs the model controls.

### 4.1 Non-destructive merge ⇒ gradient credit is complete
Merging *adds* the merged node's (gated) content into its receiver — it never removes a node from the
computation graph. So a "discarded" node's content persists inside a survivor, giving a literal path
`merged → receiver → memory → loss`. Therefore **an early merge of a node that turns out important is
never lost**: its content is still in the memory, and gradient reaches it. With **full BPTT** across all
rounds (gradient-checkpoint to afford it — preferred over truncation), credit reaches the first round.

### 4.2 The chain confusion (A→B, but B didn't survive) — resolved by survivors-first
The confusion only exists if a node may point at a *non-survivor*. **Forbid it**: elect survivors first,
then let non-survivors assign **only** over the frozen survivor set. Then A's candidate set is
`{survivors}`; B (a non-survivor) is not on it, so "A picks B and B vanishes" *cannot occur within a
round*.

- *"But A's best match B isn't a survivor"* → handled by **representative election**: the model learns to
  give one member of each tight cluster a high keep-score so the rest fold onto it. The loss coordinates
  this globally (a split/smeared cluster costs loss → gradient raises the representative's keep-score and
  steers the others' assignments). The keep-score *is* the election; whether it learns good
  representatives is the thing to watch.
- *"A should end up where B went"* → realized **temporally**: round 1 A→B (B survives), round 2 B→C.
  Full BPTT composes the chain `loss → C → (B) → (A)`. You get the transitive "follow to final survivor"
  behavior across rounds without within-round chains.

**Unifying view (the math):** the general object is the absorption matrix of an absorbing Markov chain,
`B = (I − Q_TT)⁻¹ R_TS` (transient = non-survivors, absorbing = survivors). Survivors-first is the
special case `Q_TT = 0` (non-survivors can't point at non-survivors), so `B = R_TS` — **one hop, no
matrix solve, no chains.** The alternative (allow pick-any, solve the inverse) is differentiable but
heavier and unstable; gradual rounds give the same hierarchy without it. **Recommendation: survivors-first
+ gradual rounds.**

### 4.3 Edge-identity transfer is differentiable, on the same assignment
Edge inheritance must not be opaque bookkeeping. Drive it with the **same `a_i`** as the content merge so
one decision moves both:
- **Content:** `survivor_s += gate · Σ_i a_i[s] · content_i`.
- **Edge endpoint:** an edge `(A, X)` becomes `(surv(A), surv(X))`, where the contracted endpoint id is
  the **assignment-weighted survivor id**, `id_src' = Σ_s a_A[s] · id_s` (hard argmax forward, soft
  backward). Gradient path: `loss → edge token (id of surv(A)) → a_A → A`. So the model learns whether
  re-pointing A's edge onto B helped — through the same `a_A` that folded A's content. Content fold and
  edge fold are *one* differentiated decision.
- **Duplicate edges after contraction:** prefer **binary edges (dedupe to a set)** for purity (no
  per-edge scalar to abuse → bypass stays closed); a **bounded count** ("relation strength") is a mild
  optional alternative.

---

## 5. Why this should clear the membership wall (summary)
1. **Input-grounded** → example-specific memory by construction → `SHUF−REAL` high without learning
   input-dependence.
2. **Compression loss penalizes over-merging** → pooling is no longer free; the loss itself is the
   anti-collapse regularizer.
3. **Concentration is the goal, not a pathology** → coreference; no Sinkhorn/anti-hub needed (this retires
   the entire hub-collapse problem, which was only a pathology because slots were fixed and arbitrary).
4. **No escape hatch** → no edge states; the only storage is input-grounded node content + bypass-proof
   id-pointer edges.

---

## 6. What carries over vs. what's dropped
**Keep:** GT / TokenGT message-passing + graph tokenization; fixed orthonormal identity (+ role)
embeddings; gated delta-rule update; prepend read; **d = 576**; full BPTT + checkpointing.

**Drop:** edge **states** (escape hatch); the **operator/multiplicative edge readout** (nothing to bind —
edges are pointers now); **Sinkhorn / anti-hub balancing** (concentration is desired here); **fixed random
slots** (replaced by the input chain); **cross-attention read** and **d = 64** (replaced by prepend at
d = 576).

---

## 7. Open design choices (decide at build time)
- **Merge granularity:** pairwise/ToMe-style (gentle, stable, easy to train) vs cluster-to-K/DiffPool-style
  (aggressive, less stable). Start pairwise/gradual.
- **Survivor selection:** hard top-K (straight-through) vs **soft top-K** (a couple of Sinkhorn /
  successive-softmax steps) — switch to soft top-K only if the keep-score trains sluggishly near the budget
  boundary; the additive-assignment path already gives every node gradient.
- **Aggregation:** plain additive vs **gated** additive (the safe "soft discard" — down-weight a member
  without losing its gradient path). Use gated.
- **Edges:** binary set (recommended, dedupe) vs bounded count; directed via ordered id-pointer.
- **Rounds & schedule:** how many, and the coarsening ratio per round (e.g. halving).
- **Weights:** recurrent/shared across rounds (recommended — needed for d=576 budget and matches
  streaming) vs per-round.
- **Window / budget:** 256-token windows, 32-node budget (per the 1024→32, 4-window setup).

---

## 8. Evaluation plan
- **Sanity (should pass by construction):** `SHUF−REAL` clearly > 0 on the reliable tasks (mae /
  continuation). If it *isn't*, the coarsening is over-merging → tighten merges.
- **Headline:** reconstruction / QA **fidelity at the 32-token budget vs ICAE at the same budget** — the
  cleanest test of the structure thesis: *same compression task, graph-coarsening compressor vs
  attention-pooling compressor.*
- **Streaming:** write/update quality across windows (does old content survive new windows? — the
  stability–plasticity axis flat banks struggle with).
- Reuse the existing per-layer × per-step tracer (effective rank, routing/usage) on the furl rounds; watch
  the 32 survivors' effective rank and per-round merge statistics.

---

## 9. Prior art to borrow from
- **Token Merging (ToMe)** — compress a sequence by merging similar tokens via bipartite matching;
  "chain wraps around and overlaps."
- **DiffPool / Graph U-Nets (gPool)** — learnable graph coarsening / top-k node pooling.
- **TokenGT** — pure-transformer over graph-as-tokens (nodes + edges as tokens).
- **Recurrent Memory Transformer / Compressive Transformer** — fixed-budget recurrent compression with
  BPTT (the streaming structure).
- **Absorbing Markov chains** — the formal model for "follow merges to a final survivor" (§4.2).

---

## 10. Risks / failure modes
- **Over-merge blur** — survivors absorb too many members and node content averages out. Mitigate: sparse
  merges, gated aggregation, edges + ids carry binding; watch survivor eff-rank.
- **Representative-election failure** — keep-scores don't elect clean cluster reps (split/smeared
  clusters). Watch per-round merge maps; consider soft top-K.
- **Streaming integration** — learning *where* a new chain attaches to the persistent graph (coreference
  across windows) is the genuinely hard, valuable part; may need an explicit "new-vs-persistent" matching
  step.
- **Differentiable top-K instability** near the budget boundary — soft top-K as the fallback.

---

*Status: design only. Supersedes the slotgraph line for the binding goal; reuses its tooling
(TokenGT, identity embeddings, the per-layer tracer). Build when ready.*
