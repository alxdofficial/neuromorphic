# FurlGraph — design memo (working name)

*A compressed graph memory that **folds the input token-chain into a graph** instead of inventing
topology in fixed slots. Captures the full design discussion; the operator + tokenization are now
specified (2026-07-04 refinement); build-later.*

---

## 0. TL;DR

Every prior graph-memory design failed the same way: it had to **invent** example-specific topology in
random/fixed slots, and since pooling already minimized the loss, the routing never became
input-dependent and the memory pooled ("used but not bound"). FurlGraph removes the invention problem:
**the input sequence is already a graph — a chain** — so we *initialize the memory as the input chain and
learn to compress it by merging nodes*, keeping a fixed budget (e.g. 32). Because the memory is derived
from the input, it is example-specific **by construction**, and over-merging is **punished by the
compression loss** (pooling is no longer free). One learned primitive: *each node chooses a node to merge
onto*. **No edge states** (the historical escape hatch); edges are pure identity pointers, and their
endpoints are **inherited from the merge**, never selected by a routing head. Read = prepend (nodes +
edge-pointers), raw, at the LM's own dimension (576).

**Decisions locked this session (2026-07-04):** the per-round **Move is plain self-attention** over the
union `[persistent graph ‖ new input chain ‖ edge-pointer tokens]` (not adjacency-gated message-passing);
tokenization is **concat-then-project** with a **content-free edge token**; edge pruning is the
**source-survives** rule (§4.3); the read relies on **id-incidence tokenization + bidirectional memory
attention** to be structurally legible.

---

## 1. Why a new design — what we learned from slotgraph

Diagnostics on the slotgraph line (operator read → competition writes → Gumbel-Sinkhorn) established:

- **The membership wall.** Binding requires *both* gaps: `OFF−REAL > 0` (memory is used) **and**
  `SHUF−REAL > 0` (memory is example-specific). The relational task (babi) was always **"used but not
  bound"**: `OFF−REAL` huge, `SHUF−REAL ≈ 0`.
- **The root cause is not gradient, content, or eff-rank.** Per-layer × per-step tracing showed
  `routing_diversity` (input-dependence of selection) sits at ~0.02 from step 1 and **never rises** — the
  relational binding signal *never forms*. The model never *needs* input-dependent routing because
  **pooling already minimizes the loss**.
- **Forcing structure regressed binding.** The June 24 model (≈ ICAE + identity, inert graph) was the
  cohort's best binder (`mae` SHUF−REAL 0.46); the graph redesign *halved* it (0.20) — learned topology
  inert at best, harmful at worst.
- **The 2026-07-04 slotgraph3 "simple version" audit confirmed the diagnosis** (see
  `docs/slotgraph3_simple_version_audit.md`). It identified **two independent walls**: **Wall A** =
  over-smoothing / *pooling is free* (loss-neutrality); **Wall B** = *pool-then-address* — binding must be
  installed by the write as a key-indexed structure, a **theorem** (Set-Transformer/Perceiver PMA), not a
  tuning issue. Slot designs fight both; the "simple version" fix (re-inject a fixed id label each layer)
  only re-asserts **address**, not **content**, and Goodharts the collapse metrics.
- **Two things that demonstrably help and we keep:** graph dimension = LM dimension (**d = 576**), and a
  **prepend** read (the easiest-to-train read mechanism).

**Conclusion that motivates FurlGraph:** *inventing and learning topology from scratch is too hard, and
unnecessary, because the input already carries a graph.* Switch the problem from "invent example-specific
structure" (never happened) to "**compress a given example-specific structure**" (learnable, and the loss
rewards preserving distinctions). This is the design the audit's two-walls verdict points toward: it
makes `SHUF−REAL` free by construction (Wall B: nothing to invent), and it **inverts the default** from
*pool-unless-the-loss-demands-binding* (which it never did) to *preserve-unless-the-loss-rewards-merging*
(Wall A: binding is the starting condition, not an emergent target).

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

*(Honest residual — see §5 and §10: the "over-merging costs loss" claim is only as strong as the loss
charging for the destroyed distinction, and the diagnosis showed binding is loss-neutral. FurlGraph's
`SHUF−REAL` win is unconditional; its **relational**-binding win still has a residual objective dependency,
softer than slotgraph's because the input-distinct prior + a generous budget put the default on the right
side, but not gone.)*

---

## 3. Architecture

### 3.1 Representation and tokenization
- **Nodes** = folded token content, at **d = 576**. This is the *only* place information lives.
- **Edges** = **ordered identity pointers**, **content-free**. **No stored edge state.** Edge states were
  the historical "escape hatch" (the model dumped answer-content into the free edge vector and ignored
  topology — measured directly as the *edge_state bypass*). Removing them leaves nowhere to hide
  information except the (input-grounded) nodes. Relation *semantics* live in relation-token-nodes; the
  edge only conveys *which nodes connect* and *direction* (via the ordering of the two ids).
- **Identity / role embeddings**: fixed orthonormal per-**survivor-slot** `id` (32 of them) + role tags,
  carried through merges. The `id`s are **frozen buffers that never train** — gradient flows through
  *which* survivor an edge points at (the assignment), not through the id values. The frozen orthonormal
  basis is the coordinate system; all learning is in *which content lands on which coordinate* (topology).

- **Tokenization = concat-then-project (TokenGT `w_in`), ONE shared projection** for node and edge tokens
  so the id subspaces land in the same output directions (incidence — see §3.4):
  ```
  tok_proj( [ content(d) ‖ id_A(h) ‖ id_B(h) ‖ type(h) ] ) → d          (h = d/2)

  NODE token:  content = folded node content,   id_A = id_B = id_self,     type = node
  EDGE token:  content = 0  (EMPTY)          ,   id_A = id_src, id_B = id_dst, type = edge
  ```
  The **edge's content block is zero** — that *is* the "no edge state" property, made concrete. A node
  repeats its own id in both slots; an edge places src in slot A, dst in slot B, so `i→j` and `j→i` are
  distinct tokens (direction preserved — unlike an additive `id_src+id_dst` sum, which loses it).

### 3.2 The single learned primitive — merge
Per node, per round, the model emits **two continuous outputs**:
1. a **keep-score** `s_i` — "should I survive this round?"
2. an **assignment distribution** `a_i` over the *survivors* — "if not, who absorbs me?"

A merge = a **gated additive delta** of the merged node's content into its receiver, plus the receiver
**inheriting the merged node's (incoming) edges** (both driven by `a_i`, see §4).

### 3.3 Per-round loop (within one 256-token window)
1. **Move = self-attention** — a single self-attention block over the union
   `[persistent graph nodes ‖ new input-chain nodes ‖ current edge-pointer tokens]`, all in the same
   d=576 space, so every node contextualizes against every other (and against the input chain) before it
   decides. This is **NOT adjacency-gated message-passing** — the frozen-LM-style attention is the mixer.
   Structure is *available* to the mixing as **edge-pointer tokens** (a node reaches its neighbors' ids by
   attending to the edge tokens that reference its id — "soft structure"), but it is **not enforced as an
   attention mask**. The adjacency's *hard* roles are the read (§3.4) and the contraction (§4).
2. **Elect survivors first** — `survivors = topK(s)`. This set is **frozen for the round.**
3. **Assign over survivors only** — every non-survivor assigns across `{survivors}`; survivors
   self-assign. (Restricting the candidate set to survivors removes the chain confusion — §4.2.)
4. **Merge** — gated additive content fold + assignment-driven **source-survives** edge contraction (§4.3).

Run **gradually** (e.g. 256 → 128 → 64 → 32 over a few rounds) with a **recurrent / shared-weight**
operator — it is one "furl" operation applied repeatedly. (Recurrence also makes **d = 576 affordable**:
one reused block, not the 4 stacked d=576 layers that blew the param budget in slotgraph.)

### 3.4 Read — simple prepend, made legible by two quiet ingredients
**Prepend** the final 32 node tokens (content) plus the edge id-pointer tokens (structure) to the frozen
LM at d = 576, norm-matched to the LM embedding distribution. No cross-attention; no per-layer injection
(the jun24 lesson — the easiest-to-train read, and the one that actually bound). "The LM sees the tokens
raw and observes the structure" — which works **because** of:
1. **id-incidence tokenization** (§3.1): because node and edge tokens share the projection, an edge's
   `id_src` lands in the *same directions* as the node token carrying that id, so the LM's attention
   matches endpoints to nodes by inner product — that is how it reconstructs "this edge connects these two
   nodes" from raw tokens. Without the shared id subspace the pointer is noise to a frozen LM.
2. **Bidirectional attention within the memory block** (Set-LLM): the nodes+edges are an unordered *set*;
   an edge token must see *both* endpoint node tokens regardless of emission order. Text stays causal.

The sophistication is all in the *tokenization*, not the read path.

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
matrix solve, no chains.** **Decision: survivors-first + gradual rounds.**

### 4.3 Edge contraction — the *source-survives* rule (DECIDED)
Content fold (matrix form): with the assignment matrix `S` (`S[i,s] = a_i[s]`, non-survivors → survivors,
survivors self-assign), `survivor_s += gate · Σ_i S[i,s] · content_i`.

Edge contraction is **NOT** the symmetric `SᵀAS`. We **keep an edge iff its source survives**, and
redirect the destination to that destination's survivor (hard argmax forward / soft backward via the same
`a`). Equivalently, when node N is absorbed: **N's outgoing edges die; N's incoming edges (from surviving
sources) redirect to `s(N)`.** So the surviving edge set is exactly `{ (i, s(j)) : (i,j) ∈ E and i ∈ survivors }`, deduped to a **binary set**.

Why this rule (three reasons):
1. **Bounds edges to ~#survivors, structurally.** Each survivor keeps only *its own* out-degree (~1 for a
   chain start), so the graph stays sparse no matter how much folding happens — **no spaghetti, ever**,
   without a per-edge weight or a read-time budget.
2. **Keeps the graph pure — no per-edge scalar.** The alternative (symmetric contraction + a bounded
   relation-**count**, prepend top-E by count) preserves multi-relations but the count *is* a per-edge
   scalar = a (bounded) reintroduced escape hatch — exactly the `edge_state bypass` we deleted. The
   source-survives rule prunes **structurally**, preserving binary-edge purity.
3. **Principled contraction semantics.** "You speak for your cluster with your *own* pointers, so you
   don't inherit the outgoing pointers of what you absorb; but anything that *referred to* what you
   absorbed now refers to you."

What is / isn't lost (be precise): N's role as a **target** is preserved (incoming from surviving sources
redirects; incoming from non-surviving sources is dropped with those sources). N's role as a **source**
(its outgoing edges) is dropped as edges. It is **not fully lost** for two reasons — *not* because
incoming-inheritance covers it (that is different information):
- **Non-destructive content fold**: N's content (incl. the context of what it pointed at) persists in `s(N)`.
- **The keep-score is self-correcting**: the model made N a non-survivor because it judged N's pointer-role
  redundant. If dropping N's outgoing edge costs answer loss, full BPTT raises N's keep-score → N survives
  → the edge is kept. **The election *is* the pruning criterion, and the gradient polices it.**

**Bonus (state-tracking):** if representative-election favors the **latest** mention as survivor, dropping
absorbed mentions' outgoing edges = **forgetting stale state** — the T2 forced-forgetting behavior, for free.

**Caveat to instrument:** the keep-score is `topK`-constrained. On a **dense** passage (more
important-outgoing nodes than the budget), `topK` forces dropping some that matter → real, budget-imposed
loss. Generous budget vs entity count (babi ~5–8 entities, 32 nodes) → never bites. So instrument per-round
edge count + degree distribution + whether dropped-outgoing correlates with a loss bump. **Build symmetric
+ dedup first, measure density, adopt source-survives only if spaghetti actually manifests** (it's a
one-line mask on the contraction).

---

## 5. Why this should clear the membership wall (summary)
1. **Input-grounded** → example-specific memory by construction → `SHUF−REAL` high without learning
   input-dependence (answers **Wall B** — nothing to invent).
2. **Compression loss penalizes over-merging + the default is inverted** (preserve-by-default, not
   pool-by-default) → the loss itself is the anti-collapse regularizer (answers **Wall A**).
3. **Concentration is the goal, not a pathology** → coreference; no Sinkhorn/anti-hub needed (retires the
   whole hub-collapse problem, which was only a pathology because slots were fixed and arbitrary).
4. **No escape hatch** → no edge states; the only storage is input-grounded node content + bypass-proof
   content-free id-pointer edges.

**The one thing to stay honest about:** points 1 (SHUF−REAL) is unconditional, but point 2 has a
**residual objective dependency** — over-merging two *distinct* entities is nearly-free if only the gist
matters (the loss-neutrality the diagnosis found). The input-distinct prior + generous budget + the
keep-score election put the default on the right side, but the *relational* win is not guaranteed by
construction the way the `SHUF−REAL` win is. The honest test is two-pronged (§8).

---

## 6. What carries over vs. what's dropped
**Keep:** self-attention mixer + graph tokenization (TokenGT concat-project); fixed orthonormal identity
(+ role) embeddings; gated delta content fold; prepend read; bidirectional memory attention; **d = 576**;
full BPTT + checkpointing.

**Drop:** edge **states** (escape hatch); the **routing head / endpoint selection** (endpoints are
inherited from the merge now); the **operator/multiplicative edge readout** (nothing to bind — edges are
pointers); **Sinkhorn / anti-hub balancing** (concentration is desired); **fixed random slots** (replaced
by the input chain); **cross-attention read** and **d = 64** (replaced by prepend at d = 576); **hard
adjacency-gated message-passing** (replaced by plain self-attention + edge tokens as soft structure).

---

## 7. Open design choices (decide at build time)
**Decided this session:** Move = plain self-attention (not adjacency-gated). Edge contraction = the
source-survives rule (with symmetric+dedup as the measure-first fallback, §4.3). Tokenization =
shared concat-project, content-free edge token. Edges = **binary set** (dedupe; no count). Weights =
recurrent/shared across rounds. Read = prepend + bidir memory attention.

**Still open:**
- **Merge granularity:** pairwise/ToMe-style (gentle, stable) vs cluster-to-K/DiffPool-style (aggressive).
  Start pairwise/gradual.
- **Survivor selection:** hard top-K (straight-through) vs **soft top-K** (a couple of successive-softmax
  steps) — switch only if the keep-score trains sluggishly near the budget boundary.
- **Representative-election criterion:** what makes "the latest / highest-degree mention" fall out of the
  keep-score (matters for the state-tracking / forced-forgetting bonus in §4.3).
- **Rounds & schedule:** how many, and the coarsening ratio per round (e.g. halving).
- **Window / budget:** 256-token windows, 32-node budget (per the 1024→32, 4-window setup).

---

## 8. Evaluation plan
- **Sanity (should pass by construction):** `SHUF−REAL` clearly > 0 on the reliable tasks (mae /
  continuation). If it *isn't*, the coarsening is over-merging → tighten merges.
- **The real (two-pronged) test:** (1) `SHUF−REAL > 0` — example-specific memory (free by construction);
  (2) **does it preserve the relations** — bAbI EM above the ~0.20 floor, *not* over-merged where the loss
  doesn't charge (the §5 residual). Prong 2 is the one that decides whether FurlGraph beats the wall or
  just re-earns membership.
- **Headline:** reconstruction / QA **fidelity at the 32-token budget vs ICAE at the same budget** — the
  cleanest test of the structure thesis: *same compression task, graph-coarsening vs attention-pooling.*
- **Streaming:** write/update quality across windows (does old content survive new windows? — the
  stability–plasticity axis).
- **Instrumentation:** reuse the per-layer × per-step tracer (effective rank, usage) on the furl rounds;
  **watch per-round edge count + degree distribution** (the spaghetti / budget-vs-density check, §4.3),
  survivor eff-rank, and per-round merge maps (representative-election health).

---

## 9. Prior art to borrow from
- **Token Merging (ToMe)** — compress a sequence by merging similar tokens via bipartite matching.
- **DiffPool / Graph U-Nets (gPool)** — learnable graph coarsening / top-k node pooling (`SᵀAS`).
- **TokenGT** — pure-transformer over graph-as-tokens (nodes + edges as tokens; the incidence read).
- **Set-LLM** — bidirectional memory-block attention for an unordered prepended set.
- **Recurrent Memory Transformer / Compressive Transformer** — fixed-budget recurrent compression with BPTT.
- **Absorbing Markov chains** — the formal model for "follow merges to a final survivor" (§4.2).

---

## 10. Risks / failure modes
- **Over-merge blur** — survivors absorb too many members and node content averages out (the same
  value-path collapse, relocated to the merge). The compression loss is supposed to penalize it, but only
  charges to the extent the task charges for the destroyed distinction (the §5 residual objective
  dependency). Mitigate: sparse/gated merges, generous budget vs entity count, watch survivor eff-rank +
  the two-pronged eval.
- **Representative-election failure** — keep-scores don't elect clean cluster reps (split/smeared
  clusters). Watch per-round merge maps; consider soft top-K.
- **Budget-vs-density edge loss** — on dense passages, `topK` forces dropping important-outgoing nodes
  under the source-survives rule (§4.3). Instrument; loosen the budget or fall back to symmetric+count.
- **Streaming integration** — learning *where* a new chain attaches to the persistent graph (coreference
  across windows) is the genuinely hard, valuable part; may need an explicit "new-vs-persistent" matching
  step.
- **Differentiable top-K instability** near the budget boundary — soft top-K as the fallback.

---

*Status: design specified (operator + tokenization + edge rule), build-later. Supersedes the slotgraph
line for the binding goal (see `docs/slotgraph3_simple_version_audit.md` for why); reuses its tooling
(TokenGT concat-project, identity embeddings, bidir memory attention, the per-layer tracer).*
