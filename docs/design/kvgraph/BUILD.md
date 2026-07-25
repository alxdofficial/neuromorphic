# KV-graph — build spec

_Written 2026-07-25 from the working session that settled the construction. `THESIS.md` is the **why**
(what a graph is, why it should carry binding, how it gets falsified). This is the **how**: the pipeline,
its costs, its failure modes, and what becomes learnable when. Code lives in
`src/memory/models/kvgraph/`._

---

## 0. The shape of it

```
tokens (one 512-tok window)
   │
   ├─► frozen LM forward  ──►  hidden states  +  KV cache        (paid anyway)
   │
   └─► parser  ──►  mentions, predicates, dep arcs               (v0: off-the-shelf, no training)
                        │
              [0] align spans → token indices
              [1] parse
              [2] build nodes        coref clusters collapse; node count is DATA-DEPENDENT
              [3] ground             pool PRE-RoPE K/V over each node's token positions
              [4] edges              dep arc → canonical role; events become hubs
              [5] merge              link into the persistent graph
              [6] mix + inject       typed-operator MP → norm-match → RoPE at compact rank → KV
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

## 2. Edges are a deterministic rule (v0)

Full tables in `src/memory/models/kvgraph/edges.py`. The construction is neo-Davidsonian: every predicate
becomes an **event hub**, arguments hang off it by role-labelled spokes, direction is uniform
(hub → participant) and carries no meaning — the **role** carries the argument structure.

Highlights:

| arc | role | why it matters |
|---|---|---|
| `nsubj` | agent (experiencer for mental predicates) | |
| `nsubjpass` / `agent` | **theme / agent — inverted** | voice handling is mandatory, not optional |
| `dobj`, `dative` | theme, recipient | |
| `ccomp` / `xcomp` / `acl` | **content** | *the* load-bearing arc: `Rumour --content--> Sale`. Without it the graph asserts the sale as world-fact — a truth-conditional error, not lossy compression |
| `advcl` + `mark` | because / although / then | discourse relations, expressible **only** because both endpoints are hubs |
| `prep` + `pobj` | by preposition; in/at/on split TIME vs LOCATION by the object | |
| `neg` | *attribute*, not an edge | ground facts need no scope region |
| `appos` | *merge*, not an edge | apposition asserts identity |

Two known systematic gaps, stated up front: **control/raising** loses arguments ("Maria wanted to call him"
has no `nsubj` on *call*), and **coordination** mangles predictably. Both are recovered by SRL/AMR, which is
the concrete reason AMR is the upgrade path rather than a nicety.

Edges come only from licensed arcs, so `E ≈ O(N)` — the sparse typed topology is handed to us, not induced.

---

## 3. Grounding: from annotation to KV

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
- Graph-structural PE is a *separate system*, consumed by the **mixer**, and it reaches the decoder through
  the content path (it is an input feature to the mixer, whose output is projected into K/V). The key's
  geometry is already spoken for by RoPE; do not put graph structure there.

**Free bonus, and its limit.** Rank ordering gives narrative order at no cost, and because each
fact-assertion is its own event node, versions order correctly (`Sale` at rank 14, `SaleFellThrough` at
rank 61) while the entity node stays anchored at its introduction. But positions are reassigned after
eviction/recall and compact ranks discard real time gaps — so carry supersession **explicitly** (a
`SUPERSEDES` edge plus a recency attribute) and treat position as a weak prior only.

### Norm matching

Mean-pooling shrinks vectors by roughly `1/√n`, so the most-mentioned nodes end up quietest in attention.
Rescale pooled K/V to the layer's real-token statistics. sg3's norm-match component is reusable.

---

## 4. Reading

| | Stage 1 — compression | Stage 2 — memory |
|---|---|---|
| graph size | one window, ~70 nodes | accumulated, thousands |
| read | **all nodes** | query-selected subset |
| why | matched-budget comparison against KVzip/H2O | reading everything makes access frequency uniform and destroys the eviction signal |

Stage 2's read is two-stage: **PPR** seeded at query-similar nodes gives cheap structural recall (top-B
candidates); the decoder's **attention mass** over the injected candidates gives the fine-grained,
differentiable access signal — which is H2O's heavy-hitter criterion applied to graph nodes.

**Train Stage 1 with random node dropout** even though the headline number is measured at full read.
Otherwise the mixer is free to smear one fact across several co-dependent nodes — fine under read-all,
catastrophic once a subset is retrieved. It is also the cheapest form of reconstruction-after-eviction, so
the two stages stay continuous instead of switching regime.

**Eviction during ingestion has no query.** The memory fills while reading a 600k-token document, long
before a question arrives, so ingestion-time eviction can only use structural centrality, recency and
mention count — which is precisely H2O's blind-eviction problem, and why recoverability is what makes it
survivable. Query-access refines the resident set afterwards.

**Benchmark asymmetry:** MAB shares ~85 questions per context, so access genuinely accumulates.
LongMemEval gives each question its own haystack — one query per memory, no accumulation, frequency
mechanism degenerate. Design experiments accordingly.

---

## 5. Costs

Llama-3.1-8B, 512-token window, bf16:

| | |
|---|---|
| full KV | 512 × 65,536 elems = **67 MB** |
| graph KV (~70 nodes) | **9 MB** — ~7× compression, *before* eviction |
| LM forward | ≈ **8.2 TFLOPs** |
| mixer (≈150 edges, 4 MP layers, diag+rank-64 operators) | ≈ **0.3 GFLOPs — under 0.01% of the forward** |
| **parser (v0)** | **0.1–0.5 s/window on CPU — 2–10× the LM forward** |

The graph machinery is free; the parser is the only real cost, and it is the part phase 2 deletes. ~7×
compression is also the regime KVzip wins in, so Stage 1 starts at a sane operating point.

Iterate against SmolLM2-135M (what the harness already uses); validate at 8B.

---

## 6. Instability risks, ranked

1. **Injected-KV distribution shift** *(highest, silent)*. Frozen decoder expects particular per-layer
   statistics. Mitigate with norm-matching **and** initialising the mixer near identity so at init the
   injection ≈ raw pooled KV — already a sane compressed cache. Training then only has to improve on a
   working starting point.
2. **RoPE on pooled keys** — §3. Silent degradation if missed.
3. **Over-smoothing in the mixer** — sum (injective) aggregation, residual form `h + Σ(...)`, shallow depth
   (3–4 MP layers, enough for 2–3-hop chains). Monitor `node_cos` / `effrank`; the sg3 diagnostics already
   emit both.
4. **Operator-bank explosion/vanishing** — `R_r` applied repeatedly behaves like an RNN. Initialise near
   identity (diagonal ≈ 1, low-rank ≈ 0); the residual form keeps signal alive even if operators → 0.
5. **Loss-neutrality** *(the known wall)*. Under plain reconstruction, edge typing will very likely be
   neutral. Expected, not surprising — which is why the scientific gate runs under
   **reconstruction-after-eviction**, never plain reconstruction.
6. Minor: variable node count needs padded/masked batching (harness does this already); attention pooling
   would reintroduce the frozen-scalar-temperature problem, so put sharpness in the projections.

Note what is *absent*: no write-gate to collapse, and no gradient through the parser. The parser-first
design removes most of the ways this could go unstable.

---

## 7. Milestones — do not conflate them

- **M1 (engineering).** The graph forms, injects, and the decoder reconstructs the window sanely. Success ≈
  reconstruction in the neighbourhood of KVzip at matched compression. Proves nothing scientific; it is
  plumbing with clear failure signals.
- **M2 (science).** Under reconstruction-after-eviction, ablating edge types (collapse `R_r` to one shared
  untyped operator) measurably hurts multi-hop on `factconsolidation_mh`, surviving the SHUF−REAL co-gate.

**Early cheap experiment, worth doing before anything else is tuned:** three injection variants —
(a) compact re-rotated positions, (b) no rotation, (c) original absolute positions — same everything else,
compare reconstruction. Build the position scheme as a flag so all three are one config change.

**Open question, still unanswered:** do `factconsolidation_mh`'s hop chains run through relations a
dependency parse *keeps* (predicate-argument, coref) or through the temporal/discourse residue it *drops*?
If the latter, §2's label mapping needs to change before any of this is worth writing.

---

## 8. Why parser-first is not just convenience

Stage 1 already carries **discrete input-dependent bits** — the parser's grouping decisions. Different text
yields a different node set and topology, deterministically and meaningfully.

That matters because the structure-vs-flat proof says structure cannot beat flat unless the topology carries
such bits, and sg3 never had them: its topology was soft superposition, static and generic
(`edge_inputdep` 2.79→2.90 while collapse was fixed and slots were distinct). So the parser-given graph
satisfies the proof's precondition on day one, without eviction having to supply the bits and without the
model needing to *discover* structure under an objective that does not reward discovering it.

Learned induction later has to match that bar — but it starts from a working system instead of trying to
bootstrap structure out of a loss-neutral objective, which is exactly where the previous three runs died.
