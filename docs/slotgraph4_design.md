# slotgraph4 — dense edge-state slot graph (design memo)

*The free-invent graph memory done right: fixed node slots + a DENSE edge-state tensor, updated by
attention with NO materialization and NO routing head. Differentiable end-to-end; buildable + trainable
now. Read alongside `docs/graph_thesis.md` (why topology collapses) and `docs/OBJECTIVES.md` (the ladder).*

## 0. TL;DR

slotgraph3's worst pathology was the **routing head** — a hard selection (sparsemax/top-k) that
*materializes* edges into slots, which is a documented dead-gradient ratchet AND gets absorbed into the
attention's own co-adaptation (Routing-Absorption arXiv:2603.02227, the mechanistic `routing_diversity≈0.02`).
slotgraph4 **deletes it**. The graph is a fixed **N node slots + a dense `N×N×d_e` edge-state tensor**; the
position `(i,j)` *is* the endpoint identity, so there is nothing to *choose* — every pair is a slot whose
*state* smoothly encodes whether/how i relates to j. Topology becomes a dense, differentiable quantity
(every pair gets gradient every step). This is exactly Relational Attention's "edge vectors read+written
each layer" (Diao & Loynd, ICLR 2023) — the literature-preferred expressive edge form.

**What it fixes vs slotgraph3:** routing collapse (no routing head), routing-absorption (graph is a
separate channel, not a content-bias), dead-gradient materialization (none). **What it does NOT fix:**
loss-neutrality — dense edges still decay to zero unless the *objective* charges for them (§5).

## 1. State (fixed footprint)
- **Nodes** `X ∈ [N, d]` at the LM dimension (d=576). Invented free latents (per-forward init noise for
  symmetry-break, like slotgraph3). N small (16–32).
- **Edges** `E ∈ [N, N, d_e]` — a DENSE edge-state tensor; `d_e` small (e.g. 32–64). `E[i,j]` is the state
  of the directed edge i→j (asymmetric: `E[i,j] ≠ E[j,i]`, which a scalar bilinear provably cannot do —
  DistMult symmetry). Persistent across streaming windows. Footprint = `N·d + N²·d_e` (≈ 9.2K + 262K floats
  at N=16, d_e=64 — fixed, node-bounded).

## 2. Tokenization — fully static (no learned selection)
Because endpoints are fixed by position, the id/role structure is **known and constant during the forward**:
- node token = `node_proj([ X_i ‖ id_i ‖ role=node ])`
- edge token = `edge_proj([ E[i,j] ‖ id_i ‖ id_j ‖ role=edge ])`

`id_*` = frozen orthonormal buffers (never train — the anti-collapse basis, EntNet-style reusable keys);
`role` = a fixed embedding. No `tok_proj` routing weight to learn *which* endpoints — the tensor position
carries them. (Per-block-normalize the concat blocks before projection — FurlGraph FIX 1 — so the id isn't
drowned by content.)

## 3. Write — three attention stages, free update, no materialization
Per 256-token window, over `[window tokens ‖ node tokens ‖ edge tokens]`:
1. **Token self-attention** — contextualize the input chain (causal).
2. **Token↔graph cross-attention** — nodes/edges *read from* the window (this is the WRITE: the graph
   absorbs the observation). Bidirectional on the graph side.
3. **Graph self-attention** — nodes and edges attend to each other (internal relational computation:
   an edge sees its endpoints, a node sees its incident edges).

Each stage = a **full transformer layer** (attn + residual skip + MLP + multi-head) — non-negotiable, or it
rank-collapses (Dong 2021). Node and edge states are then updated by a **bounded error-correcting delta**
`s ← s + g·(cand − s)` (gated interpolation, DeltaNet-family — reuse slotgraph3's `comp_*`/`gate_head`), so
repeated writes converge instead of saturating. **No routing, no top-k, no materialization** in the write.
Optional: PairNorm / GCNII initial-residual between stages to hold pairwise distance (anti-over-smoothing).

## 4. Read — node-centric + top-k explicit edges
The N² edges can't all fit the M budget, so the read compresses:
- **Node-centric tokens:** each node → 1 (or few) vectors = attention-pool of `[X_i ‖ its edge row E[i,:] ‖
  its edge col E[:,i]]` — the node's relations fold onto it.
- **Top-k explicit edge tokens:** spend a few budget tokens on the k strongest edges (by ‖E[i,j]‖) as
  *explicit* edge tokens (id_i, id_j pointers), so the sharpest relations survive the read intact rather
  than being blurred into node vectors. (This is the *only* soft-selection, read-side, soft-gated — never
  sparsemax.)
- Present via the shared path: **prepend + bidirectional memory attention** (id-incidence lets the LM match
  an edge's endpoint ids to node tokens), OR per-layer KV (`decoder.build_prefix_cache`) like the baselines.

## 5. The load-bearing part is the OBJECTIVE (architecture ≠ binding)
Deleting the routing head removes the *gradient* trap; it does not remove *loss-neutrality* — dense edges
will decay to zero if nothing charges for them (`docs/graph_thesis.md`). So slotgraph4 must ship with:
- **behavioral-KL** (USE) + **MAE-CE** (high-rank anti-collapse) — the shared backbone.
- **The graph as the EXCLUSIVE read channel** (NRI): the decoder reads *only* the memory tokens, no parallel
  content path to absorb the structure; make its loss-effect large (multi-hop / multi-horizon).
- **provenance-InfoNCE** (`docs/OBJECTIVES.md` Rung 2): positives = node/edge tokens written from the target
  span — the direct reward for addressing (free labels in synthetic data).
- **bypass-gap** (Larimar): `λ·relu(CE_memory − sg(CE_no_memory))` so the memory must beat the no-memory floor.

## 6. The decisive experiment (why this arm is worth building)
With the routing head **gone**, slotgraph4 is the clean test of the project's thesis: run the honest
canaries — **ID-subtracted content-only edge-state effective-rank** (does structure stay high-rank?) and
**`SHUF−REAL`** (does it bind?). If binding still fails with the gradient trap removed, the verdict is
**objective-bound, not architectural** — which is the single most valuable thing we can learn. If it binds,
we have the free-invent graph memory that slotgraph3 was trying to be.

## 7. Anti-pattern checklist
| pathology | slotgraph4 |
|---|---|
| routing collapse | **GONE** — no routing head; topology is dense edge state |
| routing absorption (2603.02227) | **AVOIDED** — graph is a separate channel, not a content-softmax bias |
| dead-gradient (sparsemax/hard-topk) | **GONE** in write; read top-k is soft-gated only |
| over-smoothing / rank collapse | guard: full-transformer stages + PairNorm; watch edge-effrank canary |
| loss-neutral collapse | **NOT architectural** — needs §5 objectives (the real work) |
| per-edge relation semantics | **carried** by edge vectors (Relational Attention) — the fix for the scalar ceiling |
| symmetry-breaking | node slots need per-forward init noise (Slot Attention) |
| read-side edge blur | mitigated by top-k explicit edge tokens |

## 8. Prior art
Relational Attention (Diao & Loynd, ICLR 2023, 2210.05062 — edge vectors r/w each layer); Perceiver IO
(Jaegle 2021) / Set Transformer (Lee 2019 — latent slots + cross-attention); Graphormer w/ edge features
(Ying 2021, 2106.05234); EntNet (Henaff 2017, 1612.03969 — keyed competitive slots, solved bAbI); Slot
Attention (Locatello 2020, 2006.15055 — competition-over-slots binding); DeltaNet (Yang 2024 — bounded
error-correcting write); PairNorm/GCNII (over-smoothing).

## 9. Status
Buildable + trainable now (fully differentiable, no RL). Relation to the line: **slotgraph4 = the
free-invent (fixed-slot, invented-node) arm done right**, the differentiable substrate; **FurlGraph** = the
input-grounded (chain-merge) arm; the graph-generative memory (`docs/graph_generative_memory.md`) is the
score-function/RL upgrade that would sit *on top of* slotgraph4 once the memory is competent.
