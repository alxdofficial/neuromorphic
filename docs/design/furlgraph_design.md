# FurlGraph — design memo (working name)

*A compressed graph memory that **folds the input token-chain into a graph** instead of inventing
topology in fixed slots. Operator + tokenization + gradient design specified; hardened by a 5-agent
research sweep (2026-07-05). Build-later, but this is buildable as written.*

---

## 0. TL;DR

Every prior graph-memory design failed the same way: it had to **invent** example-specific topology in
random/fixed slots, and since pooling already minimized the loss, the routing never became
input-dependent and the memory pooled ("used but not bound"). FurlGraph removes the invention problem:
**the input sequence is already a graph — a chain** — so we *initialize the memory as the input chain and
learn to compress it by merging nodes*, keeping a fixed budget (e.g. 32). Because the memory is derived
from the input, it is example-specific **by construction**, and over-merging is punished by the
compression loss (pooling is no longer free). One learned primitive: *each node chooses a node to merge
onto*. **No edge states** (the historical escape hatch); edges are content-free identity pointers, and
their endpoints are **inherited from the merge**, never selected by a routing head. Read = prepend
(nodes + edge-pointers), raw, at the LM's own dimension (576).

**The 6 fixes from the 2026-07-05 sweep (all folded in below):**
1. **Per-block normalize the concat blocks before `tok_proj`** — the id was being drowned ~77% content /
   15% id, worsening every streaming round (§3.1).
2. **Soft-survival gate `p_i` + SIMPLE top-K** — the keep-score was a gradient dead-end; make survival a
   differentiable gate multiplied into content + edges (hard forward, soft backward) (§4.4).
3. **Error-correcting (gated-interpolation) fold, not additive `+=`** — the additive fold both over-smooths
   and grows unbounded across windows; write the *residual* toward a bounded fixed point (§3.2, §4.3).
4. **PairNorm / initial-residual between rounds** — the recurrent fold is a smoothing operator; hold total
   pairwise survivor distance constant (architectural, not a loss) (§3.3).
5. **Soft source-survives edge prune (gated by `p_src`)** — a hard prune is a gradient dead-end and can
   disconnect the graph; fade edges by their source's survival prob (§4.3).
6. **Move = a full transformer layer** (attn + skip + MLP + multi-head), not naked attention, or it
   rank-collapses on its own (§3.3).

Verdict of the sweep: **right frame, and now a repaired operator.** `SHUF−REAL` (example-specificity) is
free by construction (~0.80); the *relational* prong is ~0.35 as originally drawn, ~0.55 with fixes 2–4.
The residual ceiling is loss-neutrality (§5), which input-grounding softens but does not remove.

---

## 1. Why a new design — what we learned from slotgraph

Diagnostics on the slotgraph line established:

- **The membership wall.** Binding needs *both* `OFF−REAL > 0` (memory used) **and** `SHUF−REAL > 0`
  (example-specific). babi was always "used but not bound": `SHUF−REAL ≈ 0`.
- **Root cause = pooling is free.** `routing_diversity` sat at ~0.02 from step 1 and never rose — the
  relational binding signal never formed, because the loss never demanded input-dependent structure.
- **Forcing structure regressed binding.** jun24 (≈ ICAE + identity, inert graph) was the best binder
  (mae SHUF−REAL 0.46); the graph redesign halved it — learned topology inert at best, harmful at worst.
- **The 2026-07-04 slotgraph3 audit** named **two walls**:
  **Wall A** = over-smoothing / pooling-is-free (loss-neutrality); **Wall B** = pool-then-address (binding
  must be installed as key-indexed structure — a PMA theorem, not tuning).
- **Keep:** graph dimension = LM dimension (**d = 576**); a **prepend** read (easiest to train).

**Conclusion:** inventing topology from scratch is too hard and unnecessary — the input already carries a
graph. Switch from "invent example-specific structure" (never happened) to "**compress a given
example-specific structure**." This makes `SHUF−REAL` free (Wall B: nothing to invent) and **inverts the
default** from pool-by-default to preserve-by-default (Wall A: binding is the starting condition).

**The 2026-07-05 sweep** (5 parallel agents: coarsening prior-art, discrete-step differentiability,
stable persistent write, tokenization/id-magnitude, collapse-mode recurrence) confirmed the frame and
found the *operator as first drawn* re-imported the project's residual killers (additive-fold
over-smoothing + a gradient-dead-end topology pathway). §§3–4 below are the repaired version.

---

## 2. Core idea — the input is already a graph

A 256-token window is a **linear chain**: token-nodes connected consecutively. Initialize the memory as
that chain, then **iteratively fold it onto itself** — merging recurring/related tokens onto shared nodes,
re-wiring edges — down to the budget (32). Entities and relations are *both* nodes ("loves" is a node
wired to `Mary` and `John`); edges are just adjacency + direction.

Two reasons this should clear the wall:
1. **Binding is free by construction.** The memory is a compression of *this* passage's chain, so another
   example's memory won't fit → `SHUF−REAL` high without learning input-dependence.
2. **The compression loss regularizes against pooling** — over-merging two things that needed to stay
   distinct costs reconstruction/answer loss. Corollary: **concentration is good** (coreference), so **no
   Sinkhorn / anti-hub balancing** — the compression loss decides how far to merge.

*(Honest residual — §5, §10: "over-merging costs loss" is only as strong as the loss charging for the
destroyed distinction, and binding was measured loss-neutral. The `SHUF−REAL` win is unconditional; the
**relational** win keeps a softened objective dependency.)*

---

## 3. Architecture

### 3.1 Representation and tokenization
- **Nodes** = folded token content at **d = 576** — the *only* place information lives.
- **Edges** = **content-free ordered identity pointers**. No stored edge state (the escape hatch). They
  convey only *which nodes connect* + direction (via id ordering). Relation *semantics* live in
  relation-token-nodes.
- **Ids** = fixed orthonormal per-survivor-slot, **frozen** buffers that never train — gradient flows
  through *which* id an edge inherits (topology), not the id values. The frozen orthonormal basis *is* the
  anti-collapse mechanism (a learned bank + sharp-softmax was audited and **rejected**: it reinstates a
  learned selection head, codebook collapse, and id=f(content) — the exact things the frozen design
  removes; §9 refs).

- **Tokenization = ONE shared concat-project (TokenGT `w_in`):**
  ```
  tok_proj( [ content(d) ‖ id_A(h) ‖ id_B(h) ‖ type(h) ] ) → d          (h = d/2)
  NODE:  content = folded content,  id_A = id_B = id_self,     type = node
  EDGE:  content = 0  (EMPTY)     ,  id_A = id_src, id_B = id_dst, type = edge
  ```
  Content-free edge (`content=0`) = "no edge state" made concrete. Direction via slot order (id_src in A,
  id_dst in B) — `i→j` ≠ `j→i`. Shared projection ⇒ edge ids land in the same directions as node ids ⇒
  the frozen LM matches endpoints by inner product (incidence).

- **[FIX 1] Per-block normalize the four concat blocks to equal energy BEFORE `tok_proj`.** `content`
  lives in LM-hidden space (norm ≈ 3.2 at init, *growing* across streaming rounds); the ids/type are
  unit-norm — so the raw concat is ~77% content / ~15% id, and `tok_proj` passes that imbalance through →
  the id-incidence match is out-voted by content-content similarity (fatal on similar entities like
  Mary/John), and it **worsens monotonically** as `content` norm grows. Fix: RMSNorm/L2-normalize each of
  `{content, id_A, id_B, type}` to unit and multiply by a **fixed (or learnable-init-equal) per-block
  gain** so all four enter at parity. ~3 lines. Do **not** rely on a learned scalar under a gist-loss (it
  won't move — the frozen-`sel_scale` lesson). Secondary hardening: reserve a small orthogonal output
  subspace for the id/type columns so content can't interfere with the incidence directions.

### 3.2 The single learned primitive — merge
Per node, per round, the model emits two continuous outputs: a **keep-score** `s_i` ("should I survive?")
and an **assignment** `a_i` over the survivors ("if not, who absorbs me?"). A merge = **an error-correcting
(gated-interpolation) write** of the absorbed content into the receiver + the receiver **inheriting the
absorbed node's incoming edges** — both driven by `a_i` and the soft-survival gate `p_i` (§4).

**[FIX 3] The content write is an error-correcting fold, NOT additive `+=`.** The original
`survivor += gate·Σ a_i content_i` both over-smooths (it *is* DiffPool `SᵀX` cluster-mean aggregation) and
grows **unbounded** across streaming windows (no decay/erase — the additive-saturation the repo already
diagnosed and fixed for edges). Replace with a bounded, error-correcting write toward the absorbed content:
```
survivor_s ← survivor_s + β · ( v_s − survivor_s ) ,   v_s = Σ_i a_i[s]·(1−p_i)·h_i
```
(gated interpolation; `h_i` = the Move's contextualized rep of node i). Repeated same-content writes
**converge** to a v-bounded fixed point instead of accumulating — the DeltaNet/EntNet lesson, an *update
rule* not a loss (so it respects the no-aux-loss rule and stays fully differentiable: gradient still
reaches every overwritten thing, plus the correction term). This is *the persistent content write* — it is
NOT the Move (§3.3); there is exactly one content write per round, and it is this one.

### 3.3 Per-round loop (within one 256-token window; N → K survivors)
1. **[FIX 6] Move = a full transformer layer** (self-attention **+ residual skip + MLP + multi-head**) over
   `[persistent graph ‖ new chain ‖ content-free edge tokens]`, all in d=576. It *contextualizes* every
   node (produces `h_i`) so keep-scores / assignments / the write-candidate see neighbors first. Naked
   attention rank-collapses doubly-exponentially (Dong 2021) — the skip+MLP+multi-head are non-negotiable
   counteractants. Structure is *soft* here (edge tokens are attendable), not a hard adjacency mask; the
   hard roles of adjacency are the read + the contraction.
2. **Elect survivors** — hard `topK(s)` in the forward (crisp 32-set), but every node carries a **soft
   survival prob `p_i`** (§4.4) that is multiplied into content + edges so the keep-score gets gradient.
3. **Assign over survivors only** — each non-survivor's `a_i` is a softmax over the *frozen* survivor set
   (fully soft ⇒ every candidate gets gradient; survivors-first kills chain confusion, §4.2).
4. **Merge** — the error-correcting fold (§3.2), gated by `a_i·(1−p_i)`, + the soft source-survives edge
   contraction (§4.3).
5. **[FIX 4] PairNorm / initial-residual between rounds** — the recurrent fold is a smoothing operator;
   center+rescale the survivors to hold total pairwise distance constant (Zhao & Akoglu 2020), and/or
   re-inject each survivor's round-0 seed (GCNII initial residual). Architectural, not a loss.

Run **gradually** (256 → 128 → 64 → 32) with a **recurrent / shared-weight** operator (makes d=576
affordable). Full BPTT + gradient-checkpointing (never truncate).

### 3.4 Read — simple prepend, made legible by two ingredients
**Prepend** the final 32 node tokens (content) + edge id-pointer tokens (structure), norm-matched, to the
frozen LM. No cross-attention, no per-layer injection. It works because of **(a) id-incidence
tokenization** (edge ids match node ids by inner product) and **(b) bidirectional attention within the
memory block** (Set-LLM — an edge sees both endpoints; text stays causal). The soft edge prune (§4.3) rides
in as a `log p_src` bias on this same bidirectional mask, so a dying edge fades and a dead one masks out —
no new read path.

### 3.5 Streaming / persistent state
Persistent 32-node graph + a new 256-chain → **union** (288) → apply the same furl operator back to 32,
folding the new chain onto existing survivors (coreference) or evicting old ones.

**[FIX 5-adjacent] Anchor cross-window matching with a stable key.** Folding a new mention onto the right
persistent survivor is the hard write; matching against the survivor's *drifting content* is fragile (and
compounds with any residual fold-drift). Anchor the assignment on a **stable key** — the survivor's frozen
orthonormal id as a persistent handle (TGN/DNC-style keyed addressing), or a slowly-decayed key updated on
its own schedule — so a new "Mary" can match the persistent "Mary" without chasing a moving target. Flagged
as the genuinely hard part; needs the streaming instrumentation (§8) before trusting "union + same
operator" alone. **The error-correcting write (FIX 3) is a prerequisite** — additive drift is what erodes
the very signal new mentions must match against.

**Why 256-token windows:** forces a persistent state — non-destructive-ish folds + old queries mean the
gradient can't fold old state away without paying loss on old queries (stability–plasticity by
construction). *Caveat (§10): this argument is unverified in-repo (the Week-0 BPTT canary) and BPTT over
many recurrent segments is known to degrade past a handful — instrument before relying on it.*

---

## 4. Gradient design (the crux)

Guiding principle: **the model never makes a brittle hard choice that gets silently overridden.** It emits
continuous quantities; the discrete events (top-K survival, assignment, edge contraction) are hard in the
forward but carry gradient in the backward via smooth surrogates that are **multiplied into things that
reach the loss** (the recurring failure point — see the sparsemax scar in §10).

### 4.1 Non-destructive-ish merge ⇒ complete content credit
The error-correcting fold (§3.2) writes the residual toward the absorbed content; it does not delete a
node from the graph, so a "discarded" node keeps a live path `merged → receiver → memory → loss`. Full
BPTT (checkpointed) reaches round 1. (Bounded ≠ destructive: gradient still reaches everything the write
touches, plus the correction term.)

### 4.2 Survivors-first removes chain confusion
Elect survivors first, then non-survivors assign **only** over the frozen survivor set → "A picks B and B
vanishes" cannot occur within a round. Formally the absorbing-Markov special case `Q_TT=0 ⇒ B = R_TS` —
one hop, no matrix solve. Transitive "follow to final survivor" is realized **temporally** across rounds
(round-1 A→B survives, round-2 B→C), composed by full BPTT.

### 4.3 Edge contraction — soft source-survives (DECIDED, now differentiable)
Keep an edge iff its source survives; redirect the destination to its survivor. **[FIX 5]** Make it
**soft**: an edge `(i,j)`'s contribution to the read is scaled by `p_i` (source survival prob) — a
`log p_i` bias in the bidirectional memory mask. Rationale unchanged (bounds edges to ~#survivors, keeps
binary-edge purity — no per-edge scalar), but now the prune is **differentiable**: a pruned edge fades
continuously, so `∂L/∂p_i → ∂L/∂s_i` is live and "keeping this edge would have helped → raise the source's
keep-score" is a real gradient (the hard version was a dead-end — the §4.4 defect). Not-fully-lost via
content-fold + the (now working) self-correcting keep-score. **Caveat:** aggressive source-survives can
disconnect the graph (gPool needed an A² two-hop fix); **build symmetric `SᵀAS`+dedup first, measure
per-round edge-count/degree/connectivity, adopt soft-source-survives only if spaghetti manifests** — the
prior should not be source-survives by default.

### 4.4 [FIX 2] Soft survivor selection — the differentiability repair (was a DEAD-END)
**The defect (as first drawn):** the keep-score `s_i` had exactly one consumer — `topK(s)` — and was
multiplied into nothing, so `∂L/∂s_i = 0`. A threshold has zero Jacobian; BPTT composes gradients that
exist, it cannot create one. The §4.3 "the gradient polices the prune" claim was **false** — the same
sparsemax dead-gradient ratchet the project already hit. The *content/assignment* fold was already sound
(soft over survivors, every candidate gets gradient); only the *topology* pathway (keep-score → top-K →
prune) was starved.

**The fix — hard forward, soft gate (the MoE-router / gPool trick):** keep the hard top-K for the 32-budget
(crisp read), but define a **soft survival probability `p_i ∈ [0,1]`** (smooth in `s_i`) and **multiply it
into everything that reaches the loss**:
- survivor keeps its own content gated by `p_s`;
- each loser folds in weighted by `a_i[s]·(1−p_i)` (§3.2 write);
- each edge enters the read scaled by `p_src` (§4.3).

Now `L → (content magnitude + folded weight + edge presence) → p_i → s_i` is live — raising a keep-score
continuously shifts a node from "folded away" toward "retained as survivor" and strengthens its edges. A
boundary node has `p≈0.5` (half-retained, half-folded); it **sharpens** as training proceeds (`p→0/1` away
from the boundary).

**Estimator for `p_i` (they ship together — a gate over a bare set-index is still a dead-end):**
- **Primary — SIMPLE (Ahmed et al. 2023):** hard k-subset forward, **exact inclusion marginals** `μ_i`
  backward; `Σ μ_i = K` (budget-consistent), no temperature, lower bias/variance than ST-Gumbel. `p_i = μ_i`.
- **Cheap fallback — gPool gate:** `p_i = σ((s_i − τ)/T)` (Gao & Ji 2019). Simple; one knob.
- **Avoid:** sparsemax / hard-top-k-from-scratch (the documented dead-gradient ratchet).

This is the **pathwise-surrogate** family (STE / Gumbel / gate-multiply / perturbed / exact-marginal);
the project's **trajectory/GRPO** objective is the other family (score-function/REINFORCE). We use pathwise
here because survival→gate is a clean smooth surrogate (lower variance than sampling).

### 4.5 Multi-round composition
Content path: a product of soft error-correcting folds — differentiable, credit reaches round 1 under full
BPTT. Topology path: now differentiable via §4.4. **Differentiable ≠ non-collapsing** — a product of
folds is still a diffusion operator whose fixed point is the mean, so FIX 4 (PairNorm/initial-residual)
and a sharpening `p` (or a hard top-1 fold) are what keep it from over-smoothing; that is orthogonal to the
gradient question.

---

## 5. Why this should clear the membership wall (summary)
1. **Input-grounded** → `SHUF−REAL` high by construction (**Wall B** — nothing to invent).
2. **Compression loss + inverted default** (preserve-by-default) → the loss is the anti-collapse
   regularizer (**Wall A**).
3. **Concentration is the goal** → coreference; no Sinkhorn/anti-hub.
4. **No escape hatch** → no edge states; only input-grounded node content + content-free id-pointer edges.

**The honest residual:** point 1 is unconditional; point 2 has a **residual objective dependency** —
over-merging two *distinct* entities is nearly-free if only the gist is charged (the measured
loss-neutrality). Input-distinct prior + generous budget + the keep-score election put the default on the
right side, but the relational win is not guaranteed the way the `SHUF−REAL` win is. Test both prongs (§8).

---

## 6. What carries over vs. what's dropped
**Keep:** full-transformer-layer Move; TokenGT concat-project tokenization; fixed orthonormal frozen ids;
prepend read + bidirectional memory attention; **d = 576**; full BPTT + checkpointing; recurrent
shared-weight furl.

**Drop:** edge **states** (escape hatch); the **routing head / endpoint selection** (endpoints inherited
from merges); the **multiplicative edge readout** (edges are pointers); **Sinkhorn / anti-hub**
(concentration desired); **fixed random slots** (replaced by the input chain); **cross-attention read** and
**d = 64**; **hard adjacency-gated message-passing** (replaced by full-layer self-attention + soft edge
tokens); **the additive `+=` fold** (replaced by the error-correcting write, FIX 3); **hard top-K without a
gate / hard source-survives prune** (replaced by the soft-survival gate, FIX 2/5); a **learned id bank +
sharp-softmax** (rejected — keep frozen orthonormal).

---

## 7. Design choices — decided vs open
**Decided (this session + the sweep):** Move = full transformer layer. Content write = error-correcting
gated-interpolation fold (bounded), not additive. Survivor selection = hard-top-K forward + soft `p_i` gate,
`p_i =` SIMPLE marginals (gPool-sigmoid fallback). Edge prune = soft source-survives (`log p_src` mask
bias), with symmetric+dedup as the measure-first fallback. PairNorm/initial-residual between rounds.
Per-block-normalized tokenization. Frozen orthonormal ids (no bank). Recurrent shared weights. Prepend +
bidirectional memory attention.

**Still open:**
- **Merge granularity fork:** keep the election + soft-gate (above) **vs** ToMe-style bipartite 2:1
  similarity merge (Bolya 2023), which **eliminates the keep-score entirely** (no election gradient
  problem, gentler over-smoothing) at the cost of no explicit importance score. Decide early — it changes
  the operator shape.
- **Fold form:** gated-interpolation delta (chosen) **vs** hard top-1 gather (rank-preserving, even less
  smoothing) — the gather is the stronger anti-collapse but coarser.
- **Representative-election criterion:** what makes "latest / highest-degree mention survives" fall out of
  the keep-score (the state-tracking / forced-forgetting bonus).
- **Rounds & schedule; coarsening ratio; window/budget** (256-token windows, 32-node budget).
- **Stable cross-window key** form (frozen id handle vs slowly-decayed key).

---

## 8. Evaluation plan
- **Sanity (should pass by construction):** `SHUF−REAL > 0` on mae/continuation. If not → over-merging.
- **The real two-pronged test:** (1) `SHUF−REAL > 0`; (2) **relations preserved** — babi EM above the
  ~0.20 floor, not over-merged where the loss doesn't charge. Prong 2 decides beat-the-wall vs
  re-earn-membership.
- **Headline:** fidelity at the 32-token budget **vs ICAE at the same budget** (graph-coarsening vs
  attention-pooling, same task).
- **Measure-first canaries (before building recurrence — everyone in the sweep prescribed this):**
  - per-round **survivor within-example cosine + effective-rank** (fold over-smoothing?);
  - **assignment entropy / max-column-mass** (block-sparse safe vs uniform-spread = the smoothing driver);
  - **`gate=0` ablation** (discriminates fold-smooths-the-content from edges-are-inert: eff-rank jumps but
    EM flat ⇒ edges decorative; eff-rank stays low ⇒ the fold is the smoother);
  - **`‖∂L/∂s‖` keep-score gradient-norm canary** (is FIX 2's soft gate actually delivering gradient?);
  - **one-round vs multi-round** on babi (localizes "operator blurs" vs "recurrence blurs" in one pass).
- **Streaming:** survivor content-norm + cosine-drift of a fixed persistent node over N windows; the T2
  resolution-split (EXACT vs DEGRADED retrieval) for fold-induced blur; land the Week-0 BPTT canary first.

---

## 9. Prior art
- **ASAP** (Ranjan et al. 2020) — top-k medoid election + soft assignment + aggregate: the closest system
  (FurlGraph ≈ ASAP + persistent recurrence + content-free edges).
- **Graph U-Nets / gPool** (Gao & Ji 2019) — top-k node pooling + **sigmoid-gate-multiply** for
  differentiability (the FIX-2 template); its A² two-hop fix flags the disconnection risk (§4.3).
- **DiffPool / MinCutPool** (Ying 2018; Bianchi 2020) — soft-assignment `SᵀX` coarsening; **required
  auxiliary orthogonality/entropy losses** to avoid degenerate/over-smoothed clusters — the losses we ban,
  hence FIX 4 instead.
- **ToMe** (Bolya et al. 2023) — bipartite gentle 2:1 merge, no importance score, no aux loss (the §7 fork).
- **EdgePool** (Diehl 2019) — edge-contraction pooling made differentiable by gating the merged feature
  with the edge score (the source-survives-soft template).
- **SIMPLE** (Ahmed et al. 2023) / **Perturbed optimizers** (Berthet et al. 2020) / **SOFT top-k** (Xie et
  al. 2020) — differentiable top-K estimators (FIX 2).
- **DeltaNet / Gated DeltaNet** (Yang et al. 2024) / **Titans** (Behrouz 2025) / **MemoryLLM** (Wang 2024)
  — error-correcting + decaying persistent writes (FIX 3); every comparable streaming memory bakes
  decay/erase into the mechanism.
- **PairNorm** (Zhao & Akoglu 2020) / **Dong et al. 2021** / **Wu et al. NeurIPS 2023** — over-smoothing /
  attention rank collapse; input-dependent attention does **not** prevent it (FIX 4, FIX 6).
- **TokenGT** (Kim et al. 2022) — concat orthonormal node identifiers + type tokens (our tokenization);
  note it *trains* the reader — a frozen LM does not, so the id must be handed to it pre-balanced (FIX 1).
- **VQ-VAE** (van den Oord 2017) / **Gumbel-softmax** (Jang 2017) — why a learned id-bank + sharp-softmax
  is rejected (codebook collapse; τ→0 starves unselected codes).
- **RMT / Compressive Transformer** (Bulatov 2022; Rae 2019) — recurrent fixed-budget BPTT compression
  (the streaming structure; and the ~5-segment BPTT-degradation caveat).
- **Absorbing Markov chains** — "follow merges to a final survivor" (§4.2).

---

## 10. Risks / failure modes
- **Over-merge blur (dominant).** The fold is a smoothing operator and recurrence compounds it (the
  measured `node_wcos 0.34→0.94`). FIX 3 (error-correcting write) + FIX 4 (PairNorm) + a sharpening `p` /
  hard top-1 fold are the counters; **still the #1 risk** — instrument per-round eff-rank from day one.
- **Loss-neutrality ceiling.** Even with the fold fixed, over-merging distinct entities is near-free if
  only the gist is charged (the §5 residual). The compliant escape is ICAE-style full-fidelity
  reconstruction pressure (endogenous high rank), not a rank aux loss. FurlGraph *bets* input-grounding
  makes this unnecessary; nothing in the operator guarantees it.
- **Edges decorative / topology inert.** The Move folds structure into node content, so the separate
  edge-pointer tokens can go unread (jun24 "MP-read inert," relocated to edges) → babi re-earns membership
  ~0.20 via node content. The read geometry *can* read them (id-incidence + bidir); legibility ≠ demand —
  needs a relational objective or read-side pressure. The `gate=0` canary (§8) detects this.
- **Uniform-spread assignment** (max-entropy `a`) is the catastrophic Mode-1 driver ("no anti-hub" is
  defensible against a common *hub*, silent about uniform spread). Watch assignment entropy.
- **Persistent-write instability / capacity blur** across many windows — mitigated by FIX 3 + post-fold
  renorm; validate with the T2 resolution-split.
- **Cross-window coreference** (where a new chain attaches) — the genuinely hard part; may need explicit
  keyed matching (§3.5), not just "same operator on the union."
- **BPTT over many segments** degrades past ~a handful (RMT) — the "gradient forces retention" argument is
  unverified in-repo (Week-0 canary) before it can be relied on.
- **Graph disconnection** from aggressive source-survives (§4.3) — measure connectivity; symmetric fallback.

---

*Status: operator + tokenization + gradient design specified and sweep-hardened; build-later but buildable.
Supersedes the slotgraph line for the binding goal. Reuses its
tooling (TokenGT concat-project, identity embeddings, bidir memory attention, the per-layer tracer, the T3
delta-write pattern).*
