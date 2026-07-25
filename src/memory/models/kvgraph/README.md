# `kvgraph` — an entity/event graph over the KV cache

The L1 episodic memory from `docs/design/kvgraph/THESIS.md`. A window of tokens goes through the frozen
LM once; a parser groups those tokens into **particulars** (entities and reified events); each group's KV
entries are pooled into one node; typed role edges connect them; the mixed node vectors are injected back
as KV entries the frozen decoder reads.

**This is not the retired slotgraph line.** That code lives in `../slotgraph/` and is kept as the record of
a diagnosed failure — implicit soft-superposed edges are formally flat, and it asked a parametric
(L2/semantic) substrate to store particulars (an L1/episodic job). See THESIS.md §1 and §5. What carries
over from it is machinery, not design: the delta-rule write primitive, the KBLaM boundary-token read
surface, and Gumbel routing.

## Pipeline (one module per stage)

| module | stage | status |
|---|---|---|
| `schema.py` | shared types: `Mention`, `Node`, `Edge`, `Graph`, the `Relation` inventory | **written** |
| `edges.py` | dependency arc -> canonical relation; the deterministic v0 edge rule | **written** |
| `align.py` | char span <-> token index. The pipeline's most dangerous silent failure | planned |
| `parse.py` | spaCy (deps + noun chunks) + coref -> `[Mention]`, predicates, arcs | planned |
| `build.py` | mentions + predicates -> `[Node]`; coref clusters collapse here | planned |
| `ground.py` | pool **pre-RoPE** K/V over each node's token positions | planned |
| `merge.py` | link a window's nodes into the persistent graph (coref); LRU+sinks policy; contraction into a surviving neighbour | planned |
| `mixer.py` | TokenGT over node+edge tokens; relation-indexed operators + bounded edge-vector correction | planned |
| `inject.py` | **node** vectors -> KV entries: norm-match, RoPE at **compact rank** | planned |
| `encoder.py` | the `nn.Module` tying it together; the entry in `src/memory/model.py` | planned |

Each edge carries a **discrete relation** (1-of-24, what the operator bank is indexed by) plus a
**continuous edge vector** (what `theme` throws away when the real relation was "merchandise"). The edge
vector is bounded — low-rank, zero-init, decay — because an unbounded one makes the relation decorative and
rebuilds the loss-neutrality wall behind a good reconstruction number. Hence the ablation is two-sided.

## The three things most likely to be silently wrong

1. **Span/token alignment.** The parser and the LM must see byte-identical text. Normalise whitespace for
   one and not the other and every offset shifts — no exception is raised, the nodes are just meaningless.
2. **RoPE.** Pool from **pre-rotation** keys (hook `k_proj`), then re-rotate at the node's *rank* in
   first-mention order — never at the original token index, which would put nodes in the deep-decay regime.
   Pooling post-RoPE keys destroys them by destructive interference, and worst for the far-apart mentions
   that coref merging exists to exploit. V is never rotated.
3. **Norm matching.** Mean-pooling shrinks vectors by ~1/sqrt(n), so the most-mentioned nodes end up the
   quietest in attention. Rescale to the layer's real-token statistics.

The correctness gate for all three is one test: pool a **single-token** node and assert its injected KV is
bit-identical to that token's original entry.

## What is learned, and what is its control

Every learned component has a fixed version that doubles as its ablation control — which is what the
anti-Goodhart discipline requires.

| component | v0 | becomes learned |
|---|---|---|
| mixer (typed operators) | trainable from day 1 | — |
| pooling | mean / head-weighted | learned attention pooling |
| edge typing | `edges.py` tables | Gumbel retype/add/prune |
| node assignment | parser spans | competitive soft assignment, straight-through (phase 2) |
| eviction & promotion | none | straight-through, budget-driven (phase 2) |

## Reading

Stage 0 (warmup, one paragraph) reads **all** nodes — the matched comparison against KVzip/H2O. Stage 1
(streaming, 96-node budget) retrieves a subset with live eviction, because the gate is
reconstruction-*after*-eviction and a read-everything stage cannot fire it. Train with **random node dropout**
too, or the mixer smears a fact across co-dependent nodes and subset retrieval breaks it.

**Eviction** = LRU + protected attention sinks + protected recent window (policy), then **contraction** into
the hottest surviving neighbour (mechanism). No supernodes: the survivor's vector is untouched, so nothing is
superposed on the routing path. See `docs/design/kvgraph/BUILD.md` §8.
