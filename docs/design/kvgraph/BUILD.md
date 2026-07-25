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
              [5] merge     link into the persistent graph; evict to budget
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

**Relations are borrowed, not invented** — the 23 in `schema.py` are the standard thematic roles of
PropBank / FrameNet / VerbNet / AMR. And correctness matters less than it looks: the relation is an index
into an operator bank, so what the design needs is **consistency**, not linguistic truth. A label that is
slightly wrong but *consistently* wrong still lets its operator learn the right transformation.

---

## 3. Edge representation: relation + edge vector

A bare relation label is lossy — `theme` says the farm was acted on, but drops the "merchandise-ness" that
the hand-drawn graph captured. So each edge carries both, exactly mirroring the node's anchor+residual and
the injection's `K_pooled + W·mixed`:

```
edge = relation           # discrete, 1-of-23, shared and stable. What the operator bank indexes.
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
eviction*; without eviction running during training, Stage 1 is KVzip-with-a-graph and will be loss-neutral
exactly as predicted. So read-everything applies only to a short warmup.

| | setup | reads | purpose |
|---|---|---|---|
| **Stage 0 — warmup** | one paragraph (~256–512 tok) | **all** nodes | plumbing: does the decoder reconstruct from graph-KV at all? **M1** |
| **Stage 1 — streaming** | ctx 2048 = 8 windows, node budget **96**, eviction + retrieval live | retrieved subgraph | the real objective. **M2 / the gate** |
| **Stage 2 — scale** | long documents, disk offload, recall of evicted subgraphs | retrieved subgraph | scaling, not new science |

Warmup teaches **representation**; streaming teaches **selection**. That is a curriculum, not two theories.

### The Stage-1 loop

The awkward question in training-time retrieval is *what is the query?* — you cannot seed retrieval with the
answer. Streaming supplies a non-circular one for free:

```
for window w in 0..7:
    ingest w         → parse → nodes → merge into the persistent graph
    evict to budget  → graph never exceeds 96 nodes
    seed  = tail of window w                     ← genuinely available, not circular
    read  = PPR(seed) → top-B nodes → mix → inject
    loss  = predict window w+1   (and/or reconstruct w from a hinted seed)
```

This is **streaming continuation from memory**, which the harness already supports (`window_size < ctx` is
wired; `continuation` is already in the task mix). No new training machinery — only the new encoder. Every
window is simultaneously a write test and a read test, which is the stability-plasticity axis our eval
protocol says MAE-style objectives never touch.

**Budget = 96 nodes**, matching `M=96` in the existing sweep regime, so Stage 1 is directly comparable to the
prior slotgraph runs *at the same persistent-state size* — the budget at which the flat bank tied. At ctx
2048 the accumulated graph wants ~280 nodes, so 96 is ~3× over-subscription: genuine forgetting pressure.

**Train with random node dropout too.** Otherwise the mixer may smear one fact across several co-dependent
nodes — fine under read-all, catastrophic once a subset is retrieved.

### Reading at Stage 1+

Two-stage: **PPR** seeded at query-similar nodes gives cheap structural recall (top-B candidates); the
decoder's **attention mass** over the injected candidates gives the fine-grained, differentiable access
signal — H2O's heavy-hitter criterion applied to graph nodes.

**Eviction during ingestion has no query.** The memory fills while reading, long before a question arrives,
so ingestion-time eviction can only use structural centrality (unseeded PageRank), recency and mention
count — precisely H2O's blind-eviction problem, and why recoverability is what makes it survivable.
Query-access refines the resident set afterwards. Note this policy is hand-written in v0 and *still*
supplies the discrete input-dependent bits the proof needs, because *which* nodes are central depends on the
input; it becomes learnable (straight-through) in phase 2, with the hand-written version as its control.

**Benchmark asymmetry:** MAB shares ~85 questions per context, so access genuinely accumulates.
LongMemEval gives each question its own haystack — one query per memory, no accumulation, frequency
mechanism degenerate. Design experiments accordingly.

---

## 7. Costs

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

Iterate against SmolLM2-135M (what the harness already uses); validate at 8B.

---

## 8. Instability risks, ranked

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

## 9. Milestones — do not conflate them

- **M1 (engineering, Stage 0).** The graph forms, injects, and the decoder reconstructs the paragraph
  sanely. Success ≈ reconstruction in the neighbourhood of KVzip at matched compression. Proves nothing
  scientific; it is plumbing with clear failure signals.
- **M2 (science, Stage 1).** Under streaming with a 96-node budget and live eviction, ablating **either**
  the relation typing **or** the edge vector measurably hurts multi-hop on `factconsolidation_mh`, surviving
  the SHUF−REAL co-gate.

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

## 10. Why parser-first is not just convenience

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
