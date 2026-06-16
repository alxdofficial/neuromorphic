# The `graph` model — relational-parser graph memory over a learnable node bank

**Status:** current design, locked 2026-06-16. Supersedes (a) the VQ-codebook graph
(commits b55f5ff…2360ff7) and (b) the abandoned `hierarchical_learned_vocab` (graph_v9)
and `soft_pointer_graph` (graph_v6). This file is the **source of truth** — code must
conform to it; if a design question arises, it is answered here, not re-litigated.

---

## 1. Thesis

A learned **vocabulary** (nodes) + **grammar** (edges/relations) = a compositional
"language." The model **parses** an observation into a small graph (a set of
`(src, relation, dst)` triples over the vocabulary) and **regenerates** the passage
from that graph. Reconstruction is **holistic** — there is **no** per-token / per-node
construction loss (translating a sentence ≠ transcoding word-by-word; that pressure
killed HLV).

- **Write = a learned relational parser** (not synaptic plasticity, not "STDP").
- **Read = bind each edge into a vector and inject** into the frozen LLM.

### Why this is a graph, not a transformer-in-disguise (the load-bearing constraints)
1. **Endpoints are constrained to the discrete vocabulary** — the parser *selects* a
   node by **pointing** into the bank; it never regresses a free endpoint vector.
2. **Reuse** — two edges pointing at the same bank node *share* that node (coreference /
   dedup); the dynamic content lives in the edges, the identities are fixed.
3. **Per-edge bound read** — binding is installed in `op(src,dst,edge)` *before*
   attention; the read is not pooling over raw node/edge tokens.
4. **Specialized edge slots** — each edge-query carries a unique instance tag, so slots
   specialize (DETR-style) instead of collapsing onto one pair.

Litmus test the design must pass: **rewire the edges (keep the same token multiset) and
the read output must change.** A transformer-over-a-bag fails this; this design passes
because the read routes through `op(src,dst,edge)`.

---

## 2. Components

### 2.1 Node bank — the vocabulary (replaces the VQ-VAE)
`N` **fixed learnable** node vectors `node_bank ∈ ℝ^{N×d_graph}`. Learned by gradient —
**no** encode→snap, **no** EMA codebook, **no** commitment loss, so **none** of the VQ
collapse dynamics. The bank is **static within a forward**, so its match-keys cache for
free. A node is a stable *identity*; **reuse = multiple edges pointing at the same node**.
Single vector per node (it is both the match-key source and the gathered value; a learned
`bank_key` projection produces the match-key, the gathered value is `node_bank` itself).
Separate key/value per node is a deferred upgrade, not v1.

### 2.2 Graph state — the persistent memory
A budget of `E` **edges**. Each edge = `(src, dst, edge_state)`:
- `src`, `dst` — **selected** nodes (pointer into the bank → gathered bank value).
- `edge_state` — a continuous `d_graph` relation vector (regressed).
- a fixed per-edge **instance tag** (the slot signature).

There is **no separate node budget / node-activation step** — the active nodes are
exactly the endpoints the `E` edges point to (≤ `2E` distinct, fewer with reuse). The
full bank is the *selection space*; the `E` edges are the *state*.

### 2.3 Write — the relational parser (`GraphParser`)
Input: the frozen LLM's hidden states of the passage at one mid tap (the *observation*),
projected to `d_graph` (`obs_proj`). The parser is `E` edge-query slots, each **3 tokens**
(`src` / `dst` / `edge`) = `init_tok + role + instance_tag`, so `3E` tokens total. Per
layer, ×`write_layers`:
1. **cross-attend** the edge tokens → the observation,
2. **self-attend** the edge tokens over each other (slots coordinate / specialize),
3. FFN. (pre-LN residual; QK-RMSNorm + learnable temp on every attention.)

After the stack, split into `src`/`dst`/`edge` tokens and produce, **per edge**:
1. **src pointer** → select a bank node → gather `src_value`,
2. **dst pointer** → select a bank node → gather `dst_value`,
3. **edge_state** = `edge_head(edge_token)` (regressed).

This is **set/graph prediction** (DETR-style), not per-token self-prediction.

#### Pointer selection (how endpoints are chosen — never regressed)
For each `src`/`dst` query token: **QK-RMSNorm + learnable-temperature softmax** over the
`N` bank match-keys → gather `ptr @ node_bank`. Same mechanism as every other attention
in the model (uniform). It **starts moderately sharp and learns to sharpen** toward
near-hard — *not* a fixed harsh temperature (which would saturate → no gradient → the
cold-start failure). No Gumbel, no STE. Caveats to monitor: a softmax still technically
blends (watch `ptr_entropy`); over a large bank the tail can blend (cheap fallback:
top-k mask before the softmax, only if the entropy canary shows it). Selection precedes
filling: there is no chicken-and-egg, because the query *points*; src/dst are **outputs**,
never pre-filled inputs, and the cost is `O(E·N)`, never `N²`.

### 2.4 Read — per-edge bound vector → cross-attention inject (`GraphReader`)
Per edge, bind into **one vector** (binding *before* attention):
```
sd        = w_sd( concat(src_value, dst_value) )      # bind the two endpoints
edge_vec  = w_gamma(edge_state) ⊙ sd + w_beta(edge_state)   # relation FiLM-modulates the pair
```
At a mid-late LLM layer (`inject_layer`), the per-position residual hidden *is* the query
(taken via a forward hook). Project `d_llama→d_graph` (`q_in`); per layer, ×`read_layers`:
1. **cross-attend** decode positions → the `E` edge vectors,
2. **self-attend** decode positions — **CAUSAL** (frozen causal LLM),
3. FFN.
Then project `d_graph→d_llama` **once** (after pooling — never per-edge), **RMS-match** to
the residual stream, **learnable dim-scaled gate** (`tanh(gate)`, init `1/√d_llama`), and
**add** into the residual at `inject_layer`. `M=0` prepend (inject, not prepend).

---

## 3. Non-negotiables (or the read collapses like every prior version)
1. **Pointer-select, never regress** an endpoint (else: v6 free-endpoint collapse + no reuse).
2. **QK-RMSNorm + learnable temperature** on the pointer and every attention (read cold-start).
3. **RMS-match the inject to the stream + nonzero gate** (v8c: inject ~2e4× too quiet → SHUF=REAL).
4. **Bind in `op(src,dst,edge)` before the read attention** (never pool raw tokens = membership wall).
5. **Per-edge instance tags** so slots specialize (anti-collapse HLV/the VQ-graph lacked).
6. **No aux losses** — anti-collapse is architectural (pointer-select + tags + fixed bank). Holistic
   recon only; no per-token/per-node construction loss.

---

## 4. Trained vs frozen
- **Frozen:** the Llama backbone (observation source at `obs_tap_layer` + the decoder injected into).
- **Trained:** the node bank, the parser (write), the reader (read incl. the bind `op` + gate),
  and a shared decoder LoRA (q/v, rank 16 — identical across all arms, the learnable read protocol).
- The graph is **state**, not parameters.

---

## 5. Persistence (BUILT) + objective

### Persistent graph state — slot-carry, windowed (in the code)
The observation is processed in **`graph_window`-token windows**, and the graph **persists
and is UPDATED across them**: each window the parser ingests the **current graph state** as
its input edge tokens (`src_value/dst_value/edge_state + role + tag`) and re-points endpoints
+ re-states edges. The **per-edge instance tag is the persistent slot identity** — edge slot
`e` stays slot `e` across windows, so the parser *refines* it (and can copy-default to keep
stable edges stable) instead of regenerating from scratch and scrambling identities. The `E`
slots are fixed, so add/remove/evict happen *implicitly* by a slot re-pointing — no separate
`fresh_edge_init` / pair-identity bookkeeping (this supersedes the earlier content-addressed-
by-`(src,dst)` framing; slot-carry is the model that fits fixed-`E` instance-tagged slots).
- **Window 1:** `state=None` → fresh `init_tok` slots.
- **Window t+1:** `state =` the previous window's graph.
- Implemented in `GraphParser.forward(obs, mask, state)` + the window loop in
  `GraphEncoder.finalize_memory`. Full BPTT through the window chain (checkpoint per window if
  long inputs OOM). *(Streaming TODO: per-example all-padding windows currently no-op via the
  `_Attn` all-masked guard rather than truly skipping that example's update.)*

### Objective
- **v1 (current): single-passage `masked_reconstruction`** — same MAE harness, data, masking,
  REAL/OFF/SHUF as HLV/baselines. Every MAE sentence is **≤ one window**, so the persistence
  machinery runs exactly once (`state=None` → one parse) — i.e. MAE de-risks parse→read but
  **does not exercise persistence**. The carry-forward is in the code, correct, and reduces to
  the single-parse behavior here.
- **next: a streaming / multi-window task** — the only thing that actually *tests* persistence
  (reuse / editing / stability over a flat bank). Not yet added.

---

## 6. Capacity (MAE-matched to the baseline cohort)
Param cost is dominated by the transformer blocks (`≈12·d²` each); `N` and `E` are cheap.
At `d_graph=512` a real parser is ~8M (3× over budget); at `d_graph=256` it fits.

**v1 MAE config (matched to the cohort ~4.6M memory params, ~8k read-surface floats):**
`d_graph=256, n_nodes=1024, n_edges=16, write_layers=2, read_layers=2, heads=4, ffn_mult=2,
obs_tap_layer=6, inject_layer=18`. Exact count verified by `scripts/diagnostics/param_count.py`;
tune `write_layers`/`read_layers` to land on the anchor.

**"Full" / streaming config (later, needs its own larger baseline tier):**
`n_nodes=1024, d_graph=512, n_edges=128` — the user's design target; ~8M, not MAE-matched.

Read-surface fairness: decoder cross-attends `E×d_graph` edge vectors ≈ `16×256 ≈ 4k`
(vs baselines `k≤16 × 576 ≈ 9k`) — comparable order; primary match is on **params**.

---

## 7. Anti-collapse + monitoring (live, every step, cheap — PR via covariance, no SVD)
Canaries at every stage:
- **selection:** `ptr_entropy` (sharp→near-one-hot good; high→blending/hedging bad), `nodes_used`
  (distinct nodes pointed to → reuse / vocabulary coverage).
- **vocabulary:** `bank_effrank` (node-bank effective rank — vocabulary collapse).
- **relation:** `edge_effrank` (edge-state effective rank across edges).
- **read:** `read_effrank` (injected signal across decode positions — the rank prior models
  collapsed to ~1), `read_gate`.
- **gradient flow (per mechanism):** parser `obs_proj / blocks / pointer(q_src,q_dst,bank_key) /
  edge_head / bank / slots`; reader `op(w_sd,w_gamma,w_beta) / q_in / blocks / out / gate` — shows
  which path carries the signal / is starved.
- **binding gate:** REAL / OFF / SHUF (SHUF = graph rolled along batch). Want REAL ≪ SHUF ≲ OFF.
- **topology sensitivity** (post-hoc): rewire edges → Δ output (≈0 ⇒ still a transformer).

---

## 8. Decisions locked in this design pass (rationale, to prevent re-litigation)
- **Drop the VQ-VAE → learnable node bank.** The bank *is* the learned vocabulary; pointer
  selection replaces encode-snap-EMA and removes the collapse dynamics. Still the thesis.
- **Drop "STDP" / global co-activation as the selector.** It's a misnomer (STDP tunes existing
  synapses in fixed topology; ours creates topology over a vocabulary; global co-occurrence ≠
  relation). Selection is the **parser's pointer** (learned), with an optional locality bias.
- **No separate node budget.** Active nodes = the `E` edges' endpoints.
- **Fully-connected `K²` rejected** (unsustainable as N grows; edges must stay expressive
  vectors, not scalar attention biases). Fixed **edge budget E** instead.
- **Pointer = sharp *learnable-temp* softmax**, not Gumbel/STE and not a fixed harsh temp.
- **Read = per-edge bound vector + cross-attn**, not 3 separate tokens pooled.
- **Nodes fixed/stateless** in v1 (reuse via shared identities; content in edges); stateful nodes deferred.

---

## 9. Open knobs (defaults chosen; revisit via sweep, not redesign)
- `op(src,dst,edge)`: **FiLM** (chosen) vs concat-MLP vs binding product.
- Selection: **pure pointer** (chosen) vs shortlist + keep-gate; locality bias **off** (chosen) to start.
- Read endpoints: **single vector per node** (chosen) vs separate key/value.
- `inject_layer=18`, `obs_tap_layer=6` (chosen, mid / mid-late).

---

## 10. Gates
- `mae_smoke` (constructs / trains / grad-flows; all trainable tensors receive gradient).
- `param_count` (lands on the cohort anchor).
- eff-rank + REAL/OFF/SHUF (the binding gate — the bar prior models failed) + `ptr_entropy`
  (selection actually sharpens) + topology-sensitivity (it's actually a graph).
