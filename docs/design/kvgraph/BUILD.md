# KV-graph — build spec

_Written 2026-07-25 from the working session that settled the construction. `THESIS.md` is the **why**
(what a graph is, why it should carry binding, how it gets falsified). This is the **how**: the pipeline,
its representation, its costs, its failure modes, and what becomes learnable when. Code lives in
`src/memory/models/kvgraph/`._

---

## 0. The shape of it

```
tokens (one window)
   │
   ├─► frozen LM forward  ──►  hidden states  +  KV cache        (paid anyway)
   │
   └─► parser  ──►  mentions, predicates, dep arcs               (v0: off-the-shelf, no training)
                        │
              [0] align     spans → token indices
              [1] parse
              [2] build     nodes; coref clusters collapse; node count is DATA-DEPENDENT
              [3] edges     dep arc → canonical relation; events become hubs
              [4] ground    pool PRE-RoPE K/V over each node's token positions
              [5] merge     link into the persistent graph; contract to budget
              [6] mix       TokenGT over node AND edge tokens
              [7] inject    NODES only → norm-match → RoPE at compact rank → KV
                        │
                        └─► frozen LM decodes from graph-KV only
```

Trainable in v0: **the mixer, the pooling, and the injection projection.** Nothing else. The frozen LM
stays frozen and the parser is preprocessing, so the gradient path is short — one module, ~13M params.

---

## 1. Node formation is data-dependent

We never choose `K`. The parser finds however many particulars the text contains: one entity node per
within-window coref cluster, one event node per predicate. For 512 tokens of prose expect **~50–100
nodes**, heavily text-dependent (dialogue compresses far better than dense prose, because entities repeat).

Fixed-`K` only reappears in phase 2's differentiable assignment, where an **∅ slot** absorbs the slack:
over-provision `K`, let the null slot take non-content tokens, and variable node count still falls out
differentiably.

**Node = identity key; facts live on edges.** Never pool an entity's whole meaning into its node vector —
merging then averages toward a centroid matching no context and the node becomes membership-only (the
pool-then-address trap). This is safe here *because of reification*: an entity node's V being a generic
"Maria" is correct, since the specific propositions live in the event nodes attached to it.

---

## 2. Edges: the deterministic relation rule (v0)

Full tables in `src/memory/models/kvgraph/edges.py`. The construction is neo-Davidsonian: every predicate
becomes an **event hub**, arguments hang off it by relation-labelled spokes, direction is uniform
(hub → participant) and carries no meaning — the **relation** carries the argument structure.

| arc | relation | why it matters |
|---|---|---|
| `nsubj` | agent (experiencer for mental predicates) | |
| `nsubjpass` / `agent` | **theme / agent — inverted** | voice handling is mandatory, not optional |
| `dobj`, `dative` | theme, recipient | |
| `ccomp` / `xcomp` / `acl` | **content** | *the* load-bearing arc: `Rumour --content--> Sale`. Without it the graph asserts the sale as world-fact — a truth-conditional error, not lossy compression |
| `advcl` + `mark` | because / although / then | discourse relations, expressible **only** because both endpoints are hubs |
| `prep` + `pobj` | by preposition; in/at/on split TIME vs LOCATION by the object | |
| `neg`, `advmod` | *attributes*, not edges | a bare adverb ("secretly") is not a particular; nothing to point at |
| `amod`/`compound` | edge **only if the modifier is independently referable** | else a string attribute; a node per adjective would blow the budget |
| `appos` | *merge*, not an edge | apposition asserts identity |

Two known systematic gaps, stated up front: **control/raising** loses arguments ("Maria wanted to call him"
has no `nsubj` on *call*), and **coordination** mangles predictably. Both are recovered by SRL/AMR, which is
the concrete reason AMR is the upgrade path rather than a nicety.

Edges come only from licensed arcs, so `E ≈ 1.4·N` measured on a worked example — the sparse typed topology
is handed to us, not induced.

**Relations are borrowed, not invented** — of the 24 in `schema.py`, 22 are standard thematic roles from
PropBank / FrameNet / VerbNet / AMR; the other two (`SUPERSEDES`, `GRAVESTONE_POINTER`) are **system
relations** written by our own machinery, never by the parser. And correctness matters less than it looks: the relation is an index
into an operator bank, so what the design needs is **consistency**, not linguistic truth. A label that is
slightly wrong but *consistently* wrong still lets its operator learn the right transformation.

---

## 3. Edge representation: relation + edge vector

A bare relation label is lossy — `theme` says the farm was acted on, but drops the "merchandise-ness" that
the hand-drawn graph captured. So each edge carries both, exactly mirroring the node's anchor+residual and
the injection's `K_pooled + W·mixed`:

```
edge = relation           # discrete, 1-of-24, shared and stable. What the operator bank indexes.
     + edge_vector        # continuous. Everything the relation throws away.
```

**Where the edge vector comes from — free in v0.** Each edge records its `licensing_tokens` (the arc's own
tokens plus both endpoint heads). Attention-pool their hidden states. "Merchandise-ness" then comes from
the actual tokens *sold* / *the farm* in context, with zero trained parameters. Upgrades to learned
attention pooling later.

**How the operator uses it.** The relation picks a base operator; the edge vector supplies a **bounded
low-rank correction**:

```
R_ij = R_{relation(ij)} + U(e_ij) V(e_ij)ᵀ        rank r ≪ d,   U,V zero-initialised
```

At init `R_ij ≡ R_relation` exactly — pure discrete relation, nothing else. Training adds deviation only
where it reduces loss, and the low rank caps how far any edge can drift. That is "properly initialised with
limited learnability" made concrete: **zero-init gives the initialisation, low rank gives the limit.** Add
weight decay on `U,V` so "behave like your relation" stays the default.

**Why the bound is not optional.** If the edge vector is free to carry everything, information routes around
the discrete relation, the relation goes decorative, and we have rebuilt the loss-neutrality wall in a new
idiom — with a *good reconstruction number* hiding it. Hence the ablation is **two-sided**: collapse the
relation typing (one shared untyped operator) *and separately* zero the edge vector. If only the vector
matters, the graph typing was decorative and we learn that cleanly instead of being fooled.

**Three levels of edge learnability, in order:**

| level | what changes | risk |
|---|---|---|
| 1. edge vector | refines within its relation | low — relation stays load-bearing by construction |
| 2. retyping | Gumbel head overrides the parser's relation | medium — parser label as prior, annealed |
| 3. learned inventory | discover the K relations instead of borrowing | high — slot-style collapse lives here |

---

## 4. Grounding: from annotation to KV

A node owns a set of token positions; its content is the pooled KV over them, **per layer, one assignment
shared across all layers** (membership is a property of positions, not of depth). Out comes `[L, kv_heads,
dim]` per node — *exactly the shape of one token's entry*. A node is a **drop-in cache entry**, which is
what makes it readable by a frozen decoder and directly comparable to KVzip/H2O.

**K is the address, V is the content.** K decides when a node is retrieved; V is what flows into the
residual stream once it is.

### RoPE

Rotation is applied per-token *inside* attention, so cached keys are already spun by their own position.
Pooling positions 4 and 87 averages vectors whose high-frequency dims differ by ~83 radians — effectively a
random relative angle, so **they cancel**. The damage scales with mention distance, i.e. it is worst exactly
for the long-range coref merges the design exists to exploit.

- Pool from **pre-RoPE** K (hook `k_proj`). V is never rotated, so it pools straight from the cache.
- Re-apply RoPE at the node's **rank** in first-mention order — compact positions `0..M-1`, *never* the
  original token index, which would put every node in the deep-decay regime.
- There is no "off": an unrotated key behaves exactly as a key at position 0, because the query is rotated
  regardless. Many entries sharing a position is off-distribution for a model that only ever saw strictly
  increasing positions; compact ranks look like an ordinary short prompt, which is in-distribution.

**Free bonus, and its limit.** Rank ordering gives narrative order at no cost, and because each
fact-assertion is its own event node, versions order correctly (`Sale` at rank 14, `SaleFellThrough` at
rank 61) while the entity node stays anchored at its introduction. But positions are reassigned after
eviction/recall and compact ranks discard real time gaps — so carry supersession **explicitly** (a
`SUPERSEDES` edge plus a recency attribute) and treat position as a weak prior only.

### Norm matching

Mean-pooling shrinks vectors by roughly `1/√n`, so the most-mentioned nodes end up quietest in attention.
Rescale pooled K/V to the layer's real-token statistics. sg3's norm-match component is reusable.

---

## 5. The mixer: TokenGT over node *and* edge tokens

Graph structure reaches the model through **token identifiers**, not through a positional encoding —
following TokenGT (Kim et al. 2022, *Pure Transformers are Powerful Graph Learners*):

```
node token   v      = [ node_vec_v , P_v , P_v ]  + type_embedding_NODE
edge token (u,v)    = [ edge_vec_uv, P_u , P_v ]  + type_embedding_EDGE
```

then a plain transformer over the token set, alternating edge-update and node-update (the Graph Networks
block: `e'_ij = φ_e(e_ij, h_i, h_j)`, then `h'_i = h_i + Σ_j R(relation_ij, e'_ij)·h_j`).

**Why identifiers beat a structural PE for us.** Graph PE must be invariant to a graph that keeps changing —
Laplacian eigen-coordinates shift the instant anything is evicted. TokenGT sidesteps this: an edge token
literally *contains* its endpoints' identifiers, so topology is reconstructible from the token set alone.
Evict half the graph and every surviving token's identifiers are unchanged. No spectral embedding, no
recomputation, no drift.

- Identifiers are **random orthonormal, resampled per forward pass**. Orthonormality lets attention match a
  node token to the edge tokens incident to it; resampling stops the model memorising particular IDs.
- **This is not the sg3 "id-tags are free" failure.** There, a learned per-slot tag distinguished slots and
  nothing else — address without content, generic, loss-neutral. Here the identifier carries no information
  by itself; the information is in the **pattern of reuse across tokens**. `P_u` appearing in one node token
  and three edge tokens *is* the statement that those edges are incident to that node.
- Sum (injective) aggregation, residual form, 3–4 layers. Over-smoothing is the graph-native restatement of
  our collapse; over-squashing, of our multi-hop wall. Monitor `node_cos` / `effrank` — the sg3 diagnostics
  already emit both.
- Operators `R_r` are diagonal-plus-low-rank, **initialised near identity** (diagonal ≈ 1, low-rank ≈ 0).

**Inject nodes only.** Edges participate in mixing and stay behind. Injecting edge tokens would double the
memory budget (breaking the matched comparison against KVzip) and slide toward the serialise-the-graph-as-
tokens path the thesis rules out. Edges act *through* the mixed node vectors, which also keeps the ablation
surgical.

At window scale (~70 nodes + ~100 edges = 170 tokens) the quadratic attention is trivial. At full-graph
scale it is not — so **the mixer runs over the retrieved subgraph, never the whole graph**, which is
consistent with §6 anyway.

---

## 6. Staging and the training loop

**Eviction is a precondition of the objective, not a scaling feature.** The gate is *reconstruction-after-
eviction*; without eviction running during training, this is KVzip-with-a-graph and will be loss-neutral
exactly as predicted.

| | setup | reads | purpose |
|---|---|---|---|
| **smoke test** | one paragraph, **no training** | all nodes | plumbing. **M1** |
| **Stage 1 — streaming** | ctx 2048 = 8 windows, budget **annealed** to 96/32, dropout on throughout | retrieved subgraph | the real objective. **M2 / the gate** |
| **Stage 2 — scale** | long documents, disk offload, recall | retrieved subgraph | scaling, not new science |

### There is no warmup training stage — and that is deliberate

_Revised 2026-07-26._ An earlier draft had a **Stage 0** that trained on read-all reconstruction before
turning eviction on, reasoning that "warmup teaches representation, streaming teaches selection." That was
wrong, and actively harmful: **a mixer trained on read-all is free to smear one fact across several
co-dependent nodes.** That is a perfectly good solution when everything is always present, and catastrophic
the moment a subset is retrieved — so a read-all warmup would spend its entire duration installing the exact
habit random node dropout exists to prevent. Worse than no warmup.

Stage 0 was doing two separable jobs. Split them:

- **Correctness verification → a smoke test, not a training stage.** Pool a single-token node and assert its
  injected KV is bit-identical to the original entry (catches the alignment bug *and* the RoPE bug at once);
  dump a graph and read it; reconstruct one paragraph with eviction off. An afternoon, zero training. That is
  what M1 actually is.
- **Curriculum → a budget anneal inside Stage 1.** Start with no eviction pressure and full reads, anneal to
  `storage 96 / read 32`. Same easier-first ordering, no regime switch, no second config or checkpoint to
  drift, and **dropout is on from step 0** so the smearing habit is never learned. The warmup becomes the
  first few hundred steps of Stage 1 rather than a separate stage.

Anneal a **single pressure parameter** driving both budgets, not two independent knobs — otherwise there is
an awkward early regime where storage is loose but `B=32`, so the model reads 11% of a graph it was never
taught to index.

**The KVzip comparison survives.** An earlier draft claimed read-all training was what made the compression
comparison matched. Sloppy: evaluation need not match training. Train with dropout under an annealed budget;
**evaluate** at full read for the headline compression number. Nothing is lost.

**What to watch instead:** with eviction annealing in, the mixer sees a **non-stationary** node distribution —
the graph keeps changing shape underneath it. That is inherent to the design rather than something staging
would have fixed, but log node-count and eviction-rate curves alongside loss, because if training
destabilises that is where it shows first.

### The Stage-1 loop

The awkward question in training-time retrieval is *what is the query?* — you cannot seed retrieval with the
answer. Streaming supplies a non-circular one for free:

```
pressure p: 0 → 1 over training      # ONE knob; drives both budgets together
    storage(p) = anneal(∞ → 96)      # no eviction pressure at p=0
    B(p)       = anneal(all → 32)

for window w in 0..7:
    ingest w          → parse → nodes → coref-merge into the persistent graph
    contract to budget → LRU + sinks + recent window; coldest-first (§8)
    seed  = tail of window w                     ← genuinely available, not circular
    read  = PPR(seed) → top-B(p) nodes → random node dropout → mix → inject
    loss  = predict window w+1   (and/or reconstruct w from a hinted seed)
```

The anneal is what replaces the deleted warmup stage: at `p=0` this *is* the old Stage 0 (full reads, no
eviction) except that **dropout is already on**, so the smearing habit is never learned in the first place.

This is **streaming continuation from memory**, which the harness already supports (`window_size < ctx` is
wired; `continuation` is already in the task mix). No new training machinery — only the new encoder. Every
window is simultaneously a write test and a read test, which is the stability-plasticity axis our eval
protocol says MAE-style objectives never touch.

**Terminal budget = 96 nodes**, matching `M=96` in the existing sweep regime, so Stage 1 is directly comparable to the
prior slotgraph runs *at the same persistent-state size* — the budget at which the flat bank tied. At ctx
2048 the accumulated graph wants ~280 nodes, so 96 is ~3× over-subscription: genuine forgetting pressure.

**Random node dropout runs from step 0, not just at the end of the anneal.** Otherwise the mixer may smear one fact across several co-dependent
nodes — fine under read-all, catastrophic once a subset is retrieved.

**Benchmark asymmetry:** MAB shares ~85 questions per context, so access genuinely accumulates.
LongMemEval gives each question its own haystack — one query per memory, no accumulation, frequency
mechanism degenerate. Design experiments accordingly.

---

## 7. Retrieval

**Retrieval is the decoder's own attention, not a separate pre-pass.** Following Memorizing Transformers
(Wu et al. 2022) and Landmark Attention: layers `0..ℓ` run on context alone; **layer ℓ's queries select from
memory**; layers `ℓ+1..L` attend over `[retrieved ∪ context]`. Single forward pass. One retrieval layer is
enough — doing it at every layer is expensive and unstable.

Our twist on the Memorizing-Transformer pattern: the retrieved unit is not a flat KV pair but a **node plus
its neighbourhood** — a small typed subgraph.

- **Put ℓ late.** With many layers of self-attention *before* the retrieval layer, each token's query is
  already contextualised by every other token's, so the top-k union comes out **naturally diversified**
  instead of 512 tokens asking the same thing. The trade-off is fewer layers left to use what came back;
  optimum probably ~⅔ depth. Cheap to sweep.
- **Storage bonus:** if retrieval happens at layer 16 of 32, node KV is only ever needed for layers 16–32 —
  a further ~2× storage cut on top of the ~7× node compression.
- **Per-token top-k → union → cap at read budget `B` → inject.** No per-token masking needed; attention is
  already a soft selection, so it only needs the right candidate set present.

**Two budgets, and the gap between them is the point:**

| | caps | v0 |
|---|---|---|
| **storage budget** | nodes resident in memory | 96 |
| **read budget `B`** | nodes injected per query | ~32 |

If read = storage, every node is accessed every time, the frequency signal is uniform, and eviction has
nothing to rank on. The gap is what *creates* the access signal.

**Learned probe queries** — initialise `B` query vectors in the context, self-attend, use them to interrogate
memory — are a real alternative with a real advantage (trained to ask good questions rather than being
whatever next-token prediction produced). Deferred to phase 1.5 because learned queries under a
reconstruction objective can collapse to fetching a fixed generic set, with token-driven retrieval as the
control.

---

## 8. Eviction

Runs only when the graph approaches the storage budget. **Policy** (§8.1) decides *which* node dies;
**mechanism** (§8.2) decides what happens to it. They are independent choices and each is ablated separately.

### 8.1 Policy — LRU + protected sinks + protected recent window

Three parts, three genuinely different failure modes:

| part | protects against |
|---|---|
| **protected sinks** | breaking the model's attention mechanism |
| **protected recent window** | evicting content before it has had a chance to be used |
| **LRU** | keeping genuinely stale content |

**LRU.** Evict the node whose last real use is oldest, where "used" = *was in the retrieved top-B and
received non-trivial attention*. No scoring function, no decay constant, no tuning.

**Protected sinks — non-optional, and a silent-failure bug if missed.** Transformers dump a large share of
attention onto the first few tokens regardless of content: softmax must sum to 1, and initial tokens are the
only positions visible to every query under causal masking, so they become the default no-op target.
StreamingLLM (2309.17453) showed that keeping **4** initial tokens plus a rolling window gives stable
generation over 4M tokens, while a pure rolling window collapses. **Our injected prefix is entirely nodes —
position 0 is a pooled entity node — so the sink is missing and the frozen decoder was trained expecting it.**
Keep the literal first ~4 token KV entries, unpooled, unscored, never evicted, at the head of the prefix.
Four slots of 96.

**Protected recent window.** New nodes are exempt from eviction for W windows regardless of score. This is
the fix for the fact that *any* usage-based policy is blind to content nobody has queried yet — a fact
written thirty seconds ago has zero usage by construction, so unprotected LRU/LFU evicts it before it can
prove itself. Note our window means *recently-created nodes*, not raw tokens: the current window's tokens are
already in the decoder's context, so memory only ever covers what has scrolled off.

**Why not something cleverer.** The attention-EMA rule earlier drafts specified is **LRFU** (Lee et al., IEEE
TC 50(12) 2001) — `score ← γ·score + usage` is exactly its CRF recurrence with `γ = 2^(−λ)`. Three lines of
evidence say it does not earn its complexity: Compressive Transformer (1911.05507) ran "most-used by average
attention" as an ablation arm and it **lost** (0.980 BPC vs 0.973 for learned conv); SIEVE (NSDI'24) and
S3-FIFO (SOSP'23) match or beat ARC/LIRS/W-TinyLFU across 6,594 traces with near-trivial FIFO variants
(SIEVE's mean gain over ARC: **1.5%**); and attention-based scoring is *structurally* blind to the un-queried.
**Sinks + recent window with no scoring at all is StreamingLLM, and it is a strong baseline** — "keep first 4
+ last N" is one of the baselines a learned importance scorer failed to beat. Attention-EMA ships only if it
beats **both LRU and random** at 3 seeds with CIs.

### 8.2 Mechanism — contraction into a surviving neighbour

**No supernodes.** The evicted node is contracted into an existing neighbour, which inherits its edges; its
content goes to disk. This is standard **edge contraction**, the primitive behind multilevel graph coarsening
(METIS, algebraic multigrid) — more standard than supernode summarization, not less.

```
contract B into A  (A = B's hottest surviving neighbour):
  A inherits B's edges to nodes A is not already connected to,
      typed GRAVESTONE_POINTER, original relation carried in the edge vector
  B's content joins A's disk record (one record per survivor)
  B is removed
```

Three exhaustive cases, applied **coldest-first**:

```
≥1 surviving neighbour   →  contract into the hottest one
neighbours all evicted    →  append to their record (follow the pointer chain)
no neighbours             →  delete outright (orphan; free, no pointer needed)
```

There is no fourth case and no refuse-to-evict rule. The terminal node of a fully-evicted component is
subject to the same policy: evict it, and its now-unreachable record dies with it. Holding a resident slot as
a handle for a region **nothing has ever queried** is exactly the cache pollution LRU exists to prevent. This
is rare — a record follows the contraction chain, so it is orphaned only when an entire connected component
goes cold, which in our graph means a one-off entity plus its facts, mentioned once and never again.

**Edge accounting, honestly.** Contraction grows edge count by `|N(B) \ N(A)| + [A has no record yet]`.
Bounded by B's boundary, and usually far smaller because A and B are neighbours so most of B's neighbours are
already A's — triangle overlap does most of the work. Transferring the edges is not optional: without it,
paths through B break and we are back to severing multi-hop chains as memory fills, which is literally the M+
failure we measured (0.423 → 0.286 as context grew).

### 8.3 Gravestone pointer edges

The `GRAVESTONE_POINTER` edge carries three things, all on the **edge**, none in the survivor's node vector:

1. **A disk offset** to the archived record.
2. **A gist** in the edge vector — so a query needing only the general shape is answerable without paying for
   recall, and so the gist doubles as the recall trigger (`query · gist > threshold`).
3. **The original relation** of the edge it replaced, also in the edge vector, so `Sale --agent--> Brother`
   does not degrade to an untyped "something was here."

**The survivor's own vector is never touched.** That is the property that makes this work: routing happens by
matching the survivor's real key, which never gets crowded no matter how much contracts into it. Only the
*gist* is a bundle, and a degraded gist costs an unnecessary recall — a budget cost, not a wrong answer.

### 8.4 Caps

- **`k` entries per record**, set from *measured* ρ̄ (§8.6), since the gist is a superposition. Overflow
  creates an additional pointer edge.
- **`m` pointer edges per survivor**; beyond that, spill the oldest into a deeper record.

**Pointer chains are lossless.** Following three hops of exact references loses nothing, unlike three levels
of superposition — which is why depth is free here and was not for supernodes.

### 8.5 Recall

```
survivor in top-B  ∧  query · gist > threshold  →  page in, expand in place, evict to stay in budget
```

A cache-line fill. Bound recalls per query or this degenerates into RAG with extra steps.

### 8.6 Why not supernodes (the abandoned design, and why it was abandoned)

The earlier design evicted a *group* into a new "gravestone" node whose key was the normalised sum of its
members' keys. Recorded here because the reasoning is expensive to reconstruct:

- **The `−1` was the whole problem.** Creating a node cost one node, so single-node eviction was never
  profitable, so groups were needed, so clustering was needed, so a `gain()` function with a `λ` was needed.
  Contraction removes the `−1` and the entire chain collapses.
- **Bundled keys cap out at ~9 members, not ~30.** The familiar `~√n` crosstalk bound holds only for
  *uncorrelated* vectors. Plate (1994, thesis App. B.2) solved the correlated case: with mean pairwise cosine
  `ρ`, discriminability decays as **1/k not 1/√k**, and capacity drops from `Θ(d)` to `Θ(√d)`. Ethayarajh
  (1909.00512) measures ~0.6 mean cosine between *uniformly random* words in GPT-2's middle layers — two
  orders of magnitude past the `ρ* ≈ 0.018` crossover. At `d=1024, ρ=0.3` the limit is **9**.
- **The grouping criterion contradicted the mechanism.** Plate's explicit advice is *"avoid the situation
  where vectors of high similarity are superimposed"* — but co-retrieval grouping selects for exactly that.
- **A size-rewarding objective builds a degenerate attractor.** Parga & Virasoro (1986) named the object (the
  "ancestor" — a category average) and proved **its basin of attraction grows with category size**; Ramsauer
  et al. (2008.02217) rediscovered it as metastable states. Unbounded, every query lands on the mega-node.
- **Nesting does not work.** Bundle-of-bundles reliability decays as **1/2^depth** (Clarkson et al.
  2301.10352, Lemma 17), so "beacons over beacons gives hierarchy for free" was wrong.

If supernodes are ever revisited, the fixes are: **center before bundling** (subtract the group mean, probe
with residuals — independently recommended by Plate, Tsodyks–Feigel'man, Parga–Virasoro, Mu & Viswanath's
All-but-the-Top, and IVF-ADC), a hard cap as a *constraint* rather than an objective term, and SPANN's
(2111.08566) finding that the **actual member nearest the centroid** beats the mean as a proxy.

Grouping machinery also went with them. Co-retrieval clustering is published twice already, GraphKV
(2509.00388) documents its failure mode (*"evicting all nodes connected to the source node unintentionally
removes too many important tokens"*), and the free substitute — **temporal contiguity**, as in EM-LLM's
contiguity buffer — was never beaten by it.

### 8.7 What contraction gives for free

**One graph, no merge operation.** Each window contributes nodes into the same persistent pool; coreference
links them where they denote the same particular; budget contracts the cold parts. There is no step that
combines graph A with graph B, because per-window graphs were never objects — just batches of admissions.

Keep **coref merge** and **contraction** distinct, though — both are "merge" and they mean different things:

| | coref merge | contraction |
|---|---|---|
| why | these are the *same particular* | out of room |
| content | unified into one node | archived to disk |
| loss | none — it was always one thing | lossy but recoverable |
| between | any two nodes, however distant | neighbours only |

Contraction must never make identity claims: that is the over-merge failure, which is worse than
under-merging because it produces wrong answers rather than missing ones.

**Hierarchy emerges instead of being trained in.** Over a long document the graph stratifies on its own —
recurring entities stay resident, their facts contract in as they cool, older facts sit deeper along pointer
chains. That is Stage 3's multi-resolution memory arriving as a *property of the eviction rule*, and unlike
the nested-supernode version it is sound, because pointer chains are exact.

**Partial repair for under-merged coref.** If `Maria(¶1)` and `Maria(¶5)` fail to link but both connect to
`Brother`, the colder duplicate eventually contracts into `Brother`, consolidating its facts under a shared
anchor. Not equivalent to having merged them, but not nothing. There is **no** corresponding repair for
over-merging.

---

## 9. Bio-inspired mechanisms: what is already in, what is queued

The mechanisms that are load-bearing are **already in the design, under engineering names**:

| our name | what it is |
|---|---|
| EMA of attention mass (§8.1) | a **Hebbian trace** — co-active with a query → potentiate, unused → decay |
| sg3's delta-rule write | the **Widrow–Hoff / error-corrected Hebbian rule** (what Titans and DeltaNet are) |
| competitive assignment, sum aggregation | **lateral inhibition** — the cell-assembly anti-collapse principle |
| homeostatic downscaling (§8.1) | **synaptic downscaling** (SHY) — stops potentiation saturating |

Note sg3's write is the *delta rule*, not STDP: it has no temporal asymmetry, which is STDP's entire content.

**Queued, deliberately not built:** Hebbian **edge creation** between co-retrieved nodes. It would add
associative links the parser structurally cannot supply (two facts three paragraphs apart are connected only
through their shared entity). Deferred for two reasons: densification risk (everything co-retrieved →
everything linked → we have reinvented attention), and — decisively — **it would confound the gate.** A
mechanism added before the ablation makes a *positive* result uninterpretable, which is worse than making a
negative one uninterpretable.

**The right home for the ambitious version is the L1→L2 boundary.** Systems consolidation (hippocampus →
neocortex: replay recent episodic traces into a slower abstract store) is the principled answer to *where do
evicted memories eventually go* — for content that recurs, eviction from L1 should mean **consolidation into
L2**, not just disk. That also gives L2 the training signal it currently lacks.

**The bar:** our own biomem cohort found **LIF neurons inert**. Biological plausibility has never been a
reason for anything in this project to work; bio-inspired mechanisms earn their place by ablation like
everything else.

---

## 10. Costs

Llama-3.1-8B, 512-token window, bf16:

| | |
|---|---|
| full KV | 512 × 65,536 elems = **67 MB** |
| graph KV (~70 nodes) | **9 MB** — ~7× compression, *before* eviction |
| LM forward | ≈ **8.2 TFLOPs** |
| mixer (≈170 tokens, 4 layers, diag+rank-64 operators) | ≈ **0.3 GFLOPs — under 0.01% of the forward** |
| **parser (v0)** | **0.1–0.5 s/window on CPU — 2–10× the LM forward** |

The graph machinery is free; the parser is the only real cost, and it is the part phase 2 deletes. ~7×
compression is also the regime KVzip wins in, so Stage 1 starts at a sane operating point.

**Honest accounting: the 7× is on the PERSISTENT footprint, not peak.** We build the full window KV before
pooling it into nodes, so peak memory during ingestion is the full cache. "Cache Me If You Can"
(2506.17121) makes exactly this criticism of post-fill eviction methods (H2O/SnapKV/PyramidKV), whose "%
reduction" headlines misstate real peak footprint. For a memory layer the persistent figure is the one that
matters across a long document — but it must be stated, not left to read as a peak-memory claim.

Iterate against SmolLM2-135M (what the harness already uses); validate at 8B.

---

## 11. Instability risks, ranked

1. **Injected-KV distribution shift** *(highest, silent)*. Frozen decoder expects particular per-layer
   statistics. Mitigate with norm-matching **and** initialising the mixer near identity so at init the
   injection ≈ raw pooled KV — already a sane compressed cache. Training then only has to improve on a
   working starting point.
2. **RoPE on pooled keys** — §4. Silent degradation if missed.
3. **Edge vector swamping the relation** — §3. Loss-neutrality relocated, and invisible in the
   reconstruction number. Bounded rank + zero init + two-sided ablation.
4. **Over-smoothing in the mixer** — sum aggregation, residual form, shallow depth; watch `node_cos`.
5. **Operator-bank explosion/vanishing** — `R_r` applied repeatedly behaves like an RNN. Identity init; the
   residual form keeps signal alive even if operators → 0.
6. **Loss-neutrality of the whole graph** under plain reconstruction — expected, which is why the gate runs
   under reconstruction-after-eviction and never under plain reconstruction.
7. Minor: variable node count needs padded/masked batching (harness does this already); attention pooling
   would reintroduce the frozen-scalar-temperature problem, so put sharpness in the projections.

Note what is *absent*: no write-gate to collapse, and no gradient through the parser. The parser-first
design removes most of the ways this could go unstable.

---

## 12. Milestones — do not conflate them

- **M1 (engineering — a smoke test, NOT a training stage).** The graph forms, injects, and the decoder
  reconstructs one paragraph sanely with eviction off. Zero training. Proves nothing scientific; it is
  plumbing with clear failure signals, and it exists so that a Stage-1 failure is attributable.
- **M2 (science, Stage 1).** Under streaming with the budget annealed to 96/32 and live eviction, ablating
  **either** the relation typing **or** the edge vector measurably hurts multi-hop on
  `factconsolidation_mh`, surviving the SHUF−REAL co-gate.

**Correctness gate for the plumbing:** pool a **single-token** node and assert its injected KV is
bit-identical to that token's original entry. That one test catches both the alignment bug and the rotation
bug.

**Early cheap experiment:** three injection variants — (a) compact re-rotated positions, (b) no rotation,
(c) original absolute positions — same everything else, compare reconstruction. Build the position scheme as
a flag so all three are one config change.

**Open question, still unanswered:** do `factconsolidation_mh`'s hop chains run through relations a
dependency parse *keeps* (predicate-argument, coref) or through the temporal/discourse residue it *drops*?
If the latter, §2's mapping needs to change before any of this is worth writing.

---

## 13. Literature verdict (2026-07-25) — the minimal design

Four parallel literature sweeps. Of six components, **two are load-bearing and well-evidenced, four have
simpler published equivalents, and the graph — our most distinctive piece — is the least evidenced of all.**

### 13.1 The finding that changes the training setup

**Compressive Transformer (1911.05507) already ran our loss-neutrality experiment, and the coupled objective
lost.** Their Table 5, enwik8, lower is better:

| compressor | loss | BPC |
|---|---|---|
| conv | **BPTT through the main LM loss** | **0.996** ← worst of seven |
| most-used (attention mass) | — | 0.980 |
| conv | **separate local attention-reconstruction loss** | **0.973** ← best |

The winner is a **separate, local, lossy objective** — `‖attn(h, old_mem) − attn(h, new_mem)‖₂` — with
gradients **explicitly stopped** from entering the main network. Training the compressor through the main
loss was the *worst* of seven configurations.

Two independent corroborations: HuggingFace's failed Infini-attention reproduction diagnosed *"the
compressive memory is not learnable"* — no signal for which information deserves high-fidelity storage; and
**Titans Revisited (2510.09551)** found *"memory updates alone proved insufficient for meaningful test-time
learning when the backbone is frozen"*, hypothesised as a mismatch between the frozen backbone's input
projections into KV space and how memory evolves. **We have a frozen backbone. That is a direct warning
about our exact setup.**

→ **Train the mixer with a local attention-reconstruction objective, gradients stopped at the LM boundary —
not by backprop through the LM loss.** This was not in the plan and it is the single most actionable finding
of the sweep.

### 13.2 Keep

- **In-forward retrieval at a middle layer.** Memorizing Transformers (2203.08913) ablated layers 3/6/9/12 →
  ppl 2.40 / **2.36** / **2.37** / 2.43; multiple kNN layers gave "no further benefits". Memory Layers at
  Scale independently ships the middle FFN. One middle site.
- **Recall-from-disk.** Our strongest component and the best-evidenced idea in the area. ArkVale's motivating
  sentence is verbatim ours (*"tokens initially evicted might regain importance"*); M+ proves it in latent
  space. The central criticism of the entire eviction literature is that it is irreversible, and
  *"nobody has demonstrated bounded capacity + importance-driven eviction + reversible recall in one
  system."* **Frame the paper around this, not around the graph.**

### 13.3 Cut or demote

| component | verdict |
|---|---|
| attention-EMA scoring | → LRU + sinks + recent window (§8.1) |
| co-retrieval clustering | → temporal contiguity first (§8.2) |
| boundary-edge rewriting | → subsumed by contraction: the survivor inherits the evicted node's edges, deduped, retyped `GRAVESTONE_POINTER`. Transferring them is **not** optional — without it, paths break and multi-hop severs as memory fills. |
| supernode gravestones | → **replaced by contraction into a surviving neighbour** (§8.2). No new node, so no clustering and no `gain()`; nothing superposed on the routing path. **Measure page-back precision as a first-class metric.** M+'s page-back recall is only ~30%. |

### 13.4 The graph is on probation

This is our least-evidenced component. Three controlled studies find write-time structure is where the money
*is not*: a 3×3 write × retrieval study where retrieval spans 20 points and write strategy only 3–8,
concluding *"raw chunked storage, which requires zero LLM calls, matches or outperforms expensive lossy
alternatives"*; swapping only the embedding model flipping Mem0-beats-RAG into RAG-beats-Mem0; GraphRAG
13.4% *worse* than vanilla RAG on NQ at 2.3× latency.

Our parse differs (no LLM extraction, latent KV-pooled nodes), so the extraction-noise failure mode does not
transfer — **but the burden of proof does.**

### 13.5 Benchmarks and baselines

- **LoCoMo is unusable**: ~6.4% of its answer key is wrong (ceiling ~93.6%) and the standard gpt-4o-mini
  judge accepts **62.8% of deliberately wrong on-topic answers**. Several published scores exceed the honest
  ceiling. We never used it; do not start.
- **BABILong is a filter benchmark, not a memory benchmark** — its own authors admit the distributional tell
  twice in print, and IBM (2503.07903) showed renaming the answer vocabulary collapses RMT from ~100% to
  **44.7 (1-hop) / 0.6 (2-hop)**. Downgrade every BABILong citation in THESIS.md accordingly.
- **BEAM is the cliff**: Mem0 reports LoCoMo 91.6 → **BEAM-10M 48.6**, temporal reasoning 16.3.
- **The real baseline is LRU + verbatim-chunk RAG at matched budget.** On MemoryAgentBench simple retrieval
  beats the memory systems: BM25 61.0, embedding-RAG 65.0, HippoRAG-2 71.0 vs MemGPT 30.6, Mem0 ~25. No
  vendor charts this comparison.
- **The white space is where we already aimed:** every system scores **≤7% on multi-hop conflict
  resolution** — which MemoryAgentBench v1 called *"Selective Forgetting"*. That is `factconsolidation_mh`.

### 13.6 The minimal design, and the experiment order

```
parse → node KV → inject at a middle layer
      → LRU + sinks + recent window
      → tombstone (no edge rewriting)
      → recall from disk                    ← the novelty
mixer trained by a LOCAL attention-reconstruction loss, gradients stopped at the LM boundary
```

**Experiments, in this order — the first two can kill the design cheaply:**

1. **Measure ρ̄** (mean pairwise cosine) inside real candidate groups. Substitutes for `d` in every capacity
   theorem in §8.5. Falsifiable prediction: at ρ̄ ≈ 0.3 with |S| = 32, membership becomes statistically
   indistinguishable from topical similarity.
2. **Graph off vs graph on**, flat KV blocks at matched budget. Per §13.4 this is now the *first* ablation,
   not the last.
3. **Noise-ablation of the retrieved memory** — replace retrieved node KV with padding/zeros/noise. M+'s
   ICML reviewer asked for exactly this and it was never run; the paper was marked down for it. Cheap,
   essential, and the only way to show the memory does anything at all.
4. **Page-back precision.**
5. Only then: the relation-typing and edge-vector ablations (§3, THESIS §5).

---

## 14. Why parser-first is not just convenience

Stage 1 already carries **discrete input-dependent bits** — the parser's grouping decisions, plus the
eviction decisions. Different text yields a different node set, topology and resident set, deterministically
and meaningfully.

That matters because the structure-vs-flat proof says structure cannot beat flat unless the topology carries
such bits, and sg3 never had them: its topology was soft superposition, static and generic
(`edge_inputdep` 2.79→2.90 while collapse was fixed and slots were distinct). So the parser-given graph
satisfies the proof's precondition on day one, without the model needing to *discover* structure under an
objective that does not reward discovering it.

Learned induction later has to match that bar — but it starts from a working system instead of trying to
bootstrap structure out of a loss-neutral objective, which is exactly where the previous three runs died.
