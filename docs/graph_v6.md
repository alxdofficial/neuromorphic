# graph_v6 — soft-pointer graph memory with a relation-gated, per-token read

Design doc. Supersedes graph_v5. Status: **designed, not yet built.**

## Motivation

The 5-model head-to-head (`docs/repr_learning_head_to_head.md`) concluded graph_v5 ≈
baselines, with its relational `edge_state` **read-side dead** (ablating it changed answers
by ~0). Root cause: the edge payload was a *concatenated input* the readout learned to zero,
and the node-centric readout let the frozen decoder do hops itself so the graph never had to.

v6 is a ground-up redesign of the **write and read** around one governing principle:

> **No-op-free:** every persistent quantity must be *consumed by the read on the critical
> path to the answer*. If a component can be deleted without changing the output, the
> optimizer will zero it.

Fixed-footprint invariant is unchanged: the memory **state size is O(1) in context length**
(constant floats whether context is 8k or 1M); access pattern (how often/much we read) is
free and not part of the budget.

### Relationship to TokenGT (honest positioning)
We borrow TokenGT's *tokenization primitive* (represent nodes **and** edges as typed
tokens, run a transformer). We are **not** TokenGT: it is a one-shot encoder of a *given*
graph; ours is a **streaming, gated, persistent memory** that *constructs* the graph from
text, with **soft-pointer edges** (queries that re-resolve to nodes dynamically, not fixed
(u,v)), a **fixed soft node basis** (entities = distributions over the basis → forced reuse →
relational joins), and a **relation-gated per-token read**. The novelty is the system; cite
TokenGT for the scaffold.

---

## Persistent state (carried across windows by slot index)

| symbol | shape | meaning |
|---|---|---|
| `N` | `[K_node, d_node]` | shared node bank (the soft basis) |
| `q_src` | `[K_edge, d_node]` | per-edge source soft-pointer query |
| `q_dst` | `[K_edge, d_node]` | per-edge destination soft-pointer query |
| `state` | `[K_edge, d_state]` | per-edge relation |

Chunk-fresh random (μ,σ) init; **no trained per-slot params** (slot identity ephemeral).
Sizes inherited from v5 / TBD — substrate-float matching is **deferred** (it will change when
the read changes; don't re-balance until the architecture is frozen).

---

## WRITE (per window)

A unified typed-token transformer (TokenGT-style), then a per-token FFN readout + gate.

**Tokenization:**
- `K_node` node tokens.
- **3 tokens per edge:** `q_src`, `q_dst`, `state` (separate → role-specific attention; the
  relation is a first-class token).
- pins (projected input text) as **cross-attention KV only** (not refined, not read out).

**Embeddings (added to tokens):**
- **Type** — learnable, ×5: `NODE / EDGE_SRC / EDGE_DST / EDGE_STATE / PIN`.
- **Instance tag** — **learnable**, one per edge slot, added to that edge's 3 sub-tokens so
  they bind (so `q_src_i` can find *its* `q_dst_i`/`state_i`). Factorize the position
  embedding as `type + instance`.
- **Nodes get no id** — content-addressed only (so similar entities cluster and joins work);
  symmetry from random init.

**Updater:** L layers of graph-token **self-attention** (all-to-all over nodes+edges, carrying
type+instance) + **cross-attention to pins**, standard softmax attention + RMSNorm. **No
proposal pool / routing. No competitive write head.**

**Write (output):** per-token **FFN → new field value** (node token→new `N` row; `q_src`
token→new `q_src`; etc.). **Anchor-biased gate** per slot: `field_new = field_old + g·(FFN_out
− field_old)`, `g∈[0,1]` low by default → persistence. Keep cross-position whitening.

**Soft-pointer:** an edge's endpoint = its src/dst token attending to node tokens *by content*,
`softmax(q·N)` — a distribution over nodes (materialized implicitly inside the self-attention).

**Anti-collapse plan:** rely on whitening + gate + distinct carried states + type/instance
embeddings + pull cross-attention. **Instrument the smoke test** for (a) cross-slot cosine
(collapse) and (b) dedup (do repeated mentions of one entity converge to one node?). Only if
those fail, add slot-competition (softmax-over-slots) **for nodes only** as a targeted fix.

---

## READ

Two stages: build edge **fact-tokens** (where `state` is made load-bearing), then a
**per-decode-token soft-top-k retrieval** over them.

### Stage A — build fact-tokens (directional, FiLM-by-`state`)

For each edge, materialize endpoints `src = softmax(q_src·N)·N`, `dst = softmax(q_dst·N)·N`,
then:
```
h        = combine(W_src · src,  W_dst · dst)     # directional: W_src ≠ W_dst, order matters
γ, β     = MLP_film(state)                        # per-channel scale + shift from the relation
fact_e   = γ ⊙ h + β                              # FiLM: state MODULATES, can't be zeroed-around
```
- **Directionality:** distinct `W_src`/`W_dst` ⇒ `(Marie, married-to, Bob) ≠ (Bob, married-to, Marie)`.
- **`state` load-bearing by construction:** it is the *gain* on the content (multiplier), not a
  concatenated addend the MLP can ignore. (Grounded in FiLM, Perez 2018; R-GCN relation-typed
  transforms; TransE/RotatE asymmetry.) Escalation if `state` still slack on the ablation probe:
  replace FiLM with an R-GCN-style basis — `fact_e = (Σ_k a_k(state)·W_k)·h`, relation *selects*
  the operation.
- **Query is NOT in this FiLM** — folding the question in here risks the query explaining the
  edge away and re-killing `state`. The query enters only at Stage B.

### Stage B — per-decode-token soft-top-k retrieval

Per the fixed-footprint principle, the memory is re-read at **every decode position** (like
Memorizing Transformers; unlike a once-at-start read, which leaves information on the table):
1. self-attention over (question + decoded-so-far) → contextualized hidden per position.
2. project hidden → a **query**.
3. **soft top-k** retrieval over the fact-tokens (`query · fact_key` → top-k weights). **Soft,
   not hard** — hard selection starves unselected edges of gradient (dead paths), a failure
   mode we've hit before.
4. **query-conditioned readout** over the selected fact-tokens (the query shapes *extraction*
   here — separate from, and after, the `state`-FiLM that built the tokens).
5. fuse the readout back into the decode hidden state.
6. **iterations = 1** (single retrieve per token).

**Multi-hop** emerges from the **autoregressive decode** (hop 1 surfaces Bob as the model
decodes; the next position re-queries from Bob → doctor) — **no internal traversal / message
passing**, so we sidestep the convergence problems that plague iterative MP. Caveat: a terse
answer needing 2 hops in a *single* token can't lean on a prior token; mitigate by letting the
model decode brief intermediate reasoning, or (only if a probe shows it failing) allow a small
**bounded** within-token iteration (≤2) — knowing that is a soft return of internal multi-hop.

### Why the read is no-op-free
- zero `state` → `γ`/`β` change → every fact-token changes (Stage A);
- zero an edge's connectivity (`q_src`/`q_dst`) → endpoints don't materialize → fact-token is
  garbage;
- the relation gates *which* edge the query retrieves (Stage B selection).

---

## Decoder coupling
The decoder (Llama) is **LoRA-tuned** (whole-LM, rank-16 q/v) and jointly trained with the
memory. Relevant literature finding: pooling/cross-attention graph reads train *unstably* as a
frozen-decoder soft prompt, but **LoRA stabilizes them** (*"Is One Token All It Takes?"*, 2026)
— so the joint-training pivot directly de-risks this read.

## Literature grounding
- Tokenization: TokenGT (Kim 2022).
- `state` modulation: FiLM (Perez 2018), R-GCN (Schlichtkrull 2018), TransE/RotatE.
- Read = query-conditioned attention pooling: Set Transformer / PMA (Lee 2018), Perceiver IO
  (Jaegle 2022); VNPool ≡ Perceiver-IO encoder and SemPool's edge-level attention pooling
  (both MP-free graph→LLM-token interfaces).
- Per-token retrieval: Memorizing Transformers, kNN-LM.

## Open / deferred
- Substrate-float matching across variants — deferred until the architecture is frozen.
- `state` as free continuous vector (default) vs typed relation vocabulary — revisit if FiLM
  alone underuses it.
- KV source for the pointer writes (pins vs node bank) — build-time detail.
- Node slot-competition (anti-collapse) and bounded within-token iteration — both fallbacks,
  added only if smoke probes demand them.

## Build plan
1. `src/repr_learning/graph_substrate_v6.py` — state, write updater, FiLM fact-token builder,
   per-token soft-top-k read.
2. `GraphV6BaselineEncoder` in `encoder.py` + config + factory wiring.
3. Smoke test: forward/backward, shape + grad checks; **probes wired from day 1** —
   cross-slot cosine (collapse), entity dedup, and a **`state`-ablation** check (zero `state`
   → answer must change, i.e. the no-op is dead).
4. Slot into the fair joint-trained sweep as a 6th arm (vs flat/Mamba/continuous/MT + v5.6).
