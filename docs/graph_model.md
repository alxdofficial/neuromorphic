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
- **Read = bind each edge into a vector and PREPEND** the E memory tokens; the frozen
  LLM reads them via its own attention (was a custom cross-attn *inject* — that collapsed
  the read to ≈rank-1, the same additive nudge at every position; see
  `scripts/diagnostics/why_graph_collapses_mae.py`). Binding stays in the write (the FiLM
  `op(src,dst,edge)` forms each token); only the read mechanism changed.

### Why this is a graph, not a transformer-in-disguise (the load-bearing constraints)
1. **Endpoints are constrained to the discrete vocabulary** — the parser *selects* a
   node by **pointing** into the bank; it never regresses a free endpoint vector.
2. **Reuse** — two edges pointing at the same bank node *share* that node (coreference /
   dedup); the dynamic content lives in the edges, the identities are fixed.
3. **Per-edge bound memory token** — binding is installed in `op(src,dst,edge)` when each
   memory token is *formed*; the prepended token is a bound edge, not a raw node/edge pooled.
4. **Specialized edge slots** — each edge-query carries a unique instance tag, so slots
   specialize (DETR-style) instead of collapsing onto one pair.

Litmus test the design must pass: **rewire the edges (keep the same token multiset) and
the memory must change.** A transformer-over-a-bag fails this; this design passes because
each memory token is `op(src,dst,edge)` of a *specific* edge.

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
Inputs: the frozen LLM's hidden states of the passage at one mid tap (the *observation*),
projected to `d_graph` (`obs_proj`); and the node bank.

**Working set (self-attended) = `6E` tokens, two parts** (each token = role + instance tag):
- **Part 1 — current graph, WITH values** (`3E` tokens): per edge `e`,
  `[src_val + role_src + tag_e , edge_state + role_edge + tag_e , dst_val + role_dst + tag_e]`
  + a "current" part-marker. **Window 1** (every MAE sentence): a **learnable initial graph**
  (`init_graph`); **window t+1**: the **carried** previous graph (persistence).
- **Part 2 — prediction slots, NO values** (`3E` tokens): per edge `e`,
  `[role_src + tag_e , role_edge + tag_e , role_dst + tag_e]` + a "prediction" part-marker.

**Cross-attention targets:** the **available nodes** (all `N`, `node_bank + role_available`)
and the **observation** (passage hiddens). The bank's K/V are fixed → cacheable; cross-
attending the bank (not self-attending it) gives node-availability awareness at `O(6E·N)`,
not `O(N²)`.

**Per layer, ×`write_layers` (≥3):** self-attend the `6E` working set → **cross-attend the
available nodes** → **cross-attend the observation** → FFN. (pre-LN; QK-RMSNorm + learnable
temp on every attention.)

**Predict the new graph off Part 2 only** (the value-less slots — reading off fresh slots,
not in-place on Part 1, teaches *active re-arrangement* not copy-the-current-graph). A head
per slot:
1. **src slot → pointer → SNAP to a bank node** → `src_value`,
2. **dst slot → pointer → SNAP to a bank node** → `dst_value`,
3. **edge slot → `edge_head` → `edge_state`** (from itself, regressed).

This is **set/graph prediction** (DETR-style): Part 2 attends Part 1 (current state) + the
bank + the observation, and *chooses* the new graph.

#### Pointer selection (how endpoints are chosen — never regressed)
For each `src`/`dst` query token: **QK-RMSNorm + learnable-temperature softmax** over the
`N` bank match-keys → gather `ptr @ node_bank`. Same mechanism as every other attention
in the model (uniform). **Init:** the pointer temp inits at `graph_ptr_logit_temp_init`
(default `0` ⇒ temp=1, consistent with every attention block). Because the logits are
QK-RMSNorm'd cosine-scale *and untrained*, this default starts **soft / near-uniform**
(entropy ≈ ln N) — the **gradient-rich** regime (the cold-start failure the design avoids
is the *too-sharp* one, where the softmax saturates → no sharpening gradient; soft is the
safe side). The **learnable temp then sharpens** toward near-hard as selection lowers the
loss — **watch `graph_ptr_entropy`**: if it does *not* fall, the pointer is stuck hedging
(de-facto mean-blend = the collapse attractor) and the **prime sweep knob** is a negative
`graph_ptr_logit_temp_init` (e.g. `-1` ⇒ temp≈0.37) to bias toward selection from step 0.
No Gumbel, no STE. Other caveat: over a large bank the tail can blend (cheap fallback:
top-k mask before the softmax, only if the entropy canary shows it). Selection precedes
filling: no chicken-and-egg — the query *points*; src/dst are **outputs**, never pre-filled
inputs, and the cost is `O(E·N)`, never `N²`.

### 2.4 Read — per-edge bound token → PREPEND (`GraphReader`)
Per edge, bind into **one vector** (binding when the token is formed):
```
sd        = w_sd( concat(src_value, dst_value) )      # bind the two endpoints
edge_vec  = w_gamma(edge_state) ⊙ sd + w_beta(edge_state)   # relation FiLM-modulates the pair
```
The `E` edge vectors **self-attend among themselves** (`×read_layers`, no causal mask — a
set, no decoder cross-attn), then `LayerNorm` + project `d_graph→d_llama`. The result is
`memory ∈ ℝ^{B×E×d_llama}`, which the loss path **PREPENDS** (`M=E`) exactly like the
baseline compressors — the frozen decoder reads it through its **own** attention,
**per-position**. No forward hook, no gate, no RMS-match.

**Why prepend, not inject:** the old cross-attn inject produced one additive vector per
position that was ≈identical across positions (`read_effrank≈1`) — a constant nudge that
can reconstruct *one* addressed value (conditioned_reconstruction) but not the many
distinct per-position tokens MAE/continuation need. Prepending hands the read to the
decoder's native attention, which reads a different mixture per position (rank ≤ E). This
also makes the comparison clean: graph vs baselines now differ **only in the write**
(structured parser vs flat compressor), same read. (Diagnostic:
`scripts/diagnostics/why_graph_collapses_mae.py`.)

---

## 3. Non-negotiables (or the read collapses like every prior version)
1. **Pointer-select, never regress** an endpoint (else: v6 free-endpoint collapse + no reuse).
2. **QK-RMSNorm + learnable temperature** on the pointer and every attention (read cold-start).
3. **Read = PREPEND the bound edge tokens** (the decoder reads them per-position via its own
   attention). The custom cross-attn *inject* collapsed the read to ≈rank-1 — do not bring it back.
4. **Bind in `op(src,dst,edge)` when forming each memory token** (never pool raw tokens = membership wall).
5. **Per-edge instance tags** so slots specialize (anti-collapse HLV/the VQ-graph lacked).
6. **No aux losses** — anti-collapse is architectural (pointer-select + tags + fixed bank). Holistic
   recon only; no per-token/per-node construction loss.

---

## 4. Trained vs frozen
- **Frozen:** the Llama backbone (observation source at `obs_tap_layer` + the decoder that
  reads the prepended memory).
- **Trained:** the node bank, the parser (write), the reader (the bind `op` + self-attn
  memory-former + `d_graph→d_llama` projection), and a shared decoder LoRA (q/v, rank 16 —
  identical across all arms, the learnable read protocol).
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

## 6. Capacity (matched to the baseline cohort on BOTH axes)
**Param axis (the mechanism cost):** matched to **~5.98M memory params**. Cohort:
`graph 5.98M · icae r104 6.00M · ccm r52 6.00M · autocompressor r52 6.01M · beacon 11
wrap-layers 6.08M` (all within 1.7%). Verified by `scripts/diagnostics/param_count.py`.

**v1 MAE config:** `d_graph=256, n_nodes=1024, n_edges=16 (E_max), write_layers=3,
read_layers=2, heads=4, ffn_mult=2, obs_tap_layer=6` (prepend read — no inject layer).

**Read-surface axis (capacity-relative — same bins as the baselines):** the parser predicts
`E_max=16` edges, formed into `E` memory tokens, but the prepend **slices to the first
`k = ceil(L/8) ∈ [3,16]` tokens** (the same Matryoshka-prefix bins the baselines slice to).
So the graph obeys the cohort's **8:1 compression ratio and k-bins**, and — because the
memory tokens are projected to `d_llama` and prepended — the read surface is now
**`k×d_llama = k×576`, identical to the baselines** (the old inject read at `d_graph` is
gone). On non-MAE tasks (no `k_slots`) the full `E_max` budget is prepended. *(All `E_max`
edge slots still receive gradient even when sliced — the 6E working-set self-attention
couples kept and dropped slots.)*

**"Full" / longer-task config (later — conditioned-reconstruction etc., own larger tier):**
bump `n_edges`/`d_graph` (design target `n_nodes=1024, d_graph=512, n_edges=128`) and drop the
`k`-slice (use the full budget); re-match a larger baseline tier then.

---

## 7. Anti-collapse + monitoring (live, every step, cheap — PR via covariance, no SVD)
Canaries at every stage:
- **selection:** `ptr_entropy` (sharp→near-one-hot good; high→blending/hedging bad), `nodes_used`
  (distinct nodes pointed to → reuse / vocabulary coverage).
- **vocabulary:** `bank_effrank` (node-bank effective rank — vocabulary collapse).
- **relation:** `edge_effrank` (edge-state effective rank across edges).
- **read:** `mem_effrank` (effective rank of the PREPENDED memory tokens across edges×batch —
  the content rank the decoder can read; the old inject read collapsed this to ~1).
- **gradient flow (per mechanism):** parser `obs_proj / blocks / pointer(q_src,q_dst,bank_key) /
  edge_head / bank / slots`; reader `op(w_sd,w_gamma,w_beta) / blocks(memory-former) / out` — shows
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
- **Read = per-edge bound token, PREPENDED** (decoder reads via its own attention), NOT a custom
  cross-attn inject. The inject collapsed to ≈rank-1 (one additive nudge for all positions) — fine
  for one addressed value, fatal for the many distinct per-position tokens MAE/continuation need.
  Prepend also isolates the experiment to the WRITE (same read as the baselines). *(Superseded the
  v1 inject design 2026-06-16; see `scripts/diagnostics/why_graph_collapses_mae.py`.)*
- **Nodes fixed/stateless** in v1 (reuse via shared identities; content in edges); stateful nodes deferred.

---

## 9. Open knobs (defaults chosen; revisit via sweep, not redesign)
- `op(src,dst,edge)`: **FiLM** (chosen) vs concat-MLP vs binding product.
- Selection: **pure pointer** (chosen) vs shortlist + keep-gate; locality bias **off** (chosen) to start.
- Read endpoints: **single vector per node** (chosen) vs separate key/value.
- `obs_tap_layer=6` (chosen, mid). Read is prepend — no inject layer.
- **Node competition** (`graph_node_competition`, default **off**): slot-attention edge
  competition in `_point` (softmax over edges per node → renormalize per edge) so edges
  spread instead of hub-collapsing (6/1024 nodes observed). Default off so prepend-alone vs
  prepend+competition is a clean A/B; flip on if `nodes_used`/`edge_effrank` stay low.

---

## 10. Gates
- `mae_smoke` (constructs / trains / grad-flows; all trainable tensors receive gradient).
- `param_count` (lands on the cohort anchor).
- eff-rank + REAL/OFF/SHUF (the binding gate — the bar prior models failed) + `ptr_entropy`
  (selection actually sharpens) + topology-sensitivity (it's actually a graph).
