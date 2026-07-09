# slotgraph — v2 redesign (graph-as-language) — design + implementation plan

This is the **next version of `slotgraph`** — NOT a separate model. Same variant name
(`slotgraph_baseline`); we evolve it in place and the current slotgraph becomes the prior version
(re-run for the comparison). The reframe: the memory is **a graph-structured utterance in an
invented vocabulary** — many small composable vectors (the "words"), wired into a graph (the
"grammar"), that re-describes the observation efficiently. Three coupled changes vs slotgraph:
1. **Many small nodes/edges** instead of 32×576 slots.
2. **Encoder→decoder** split: per-layer endpoint *materialization* in the write; **gated
   cross-attention** read (no mean-aggregation MP, no prepend).
3. **Node-dropout anti-bypass curriculum** so the edges are forced to pull their weight.

Status: PLAN (no code yet). Build only after this is agreed + the smoke checklist passes.

---

## 0. What's reused / new / removed

- **Reused:** frozen SmolLM2-135M (d=576) + encoder/decoder LoRA; `_NormMatch`; the
  content-addressed endpoint selection (query·key → log-space Sinkhorn → straight-through argmax,
  no self-loops); the mixed 4-task objective + trainer; the `slotgraph_metrics.py` panel.
- **New:** small-dim graph latents (d_node=64); an **encoder graph-transformer** (cross-attends to
  frozen-LM passage features + per-layer structure prediction + endpoint-derived edge ids);
  **per-layer bottleneck gated cross-attention** decoder read; the **node-dropout curriculum**;
  near-orthonormal ids (N>d_node).
- **Removed:** the mean-aggregation MP read (it over-smooths) and the prepend read (can't host many
  small units). Fixed-positional-only edge id (edges now get an **endpoint-derived** id).

---

## 1. Memory layout + scaling config

Small composable units, **float-matched** to the baselines' memory (32×576 = 18,432 floats), NOT
token-matched. Report compression as floats (1024×576 → 18,432 ≈ 32:1 by floats); flag that the
*unit count* is no longer 32.

- `d_node = 64` — small enough that no single node can carry the answer (forces composition), big
  enough for a primitive (~head_dim scale).
- Budget: `(N + E) · d_node ≈ 18,432` → **N + E ≈ 288**. Default split **144 nodes / 144 edges**
  (avg node degree ≈ 2). Configurable; first sweep axis = {d_node, N:E split}.
- **role**: fixed by index — units `0..N-1` are nodes, `N..N+E-1` are edges (a buffer, 0 params).
- **id (LEARNED):** a learned `[N, d_node]` embedding table (learned positional codes) — **not**
  fixed-orthonormal, **not** sinusoidal. Sinusoidal encodes 1-D sequence *order*, but the nodes are a
  SET (no inherent order), so it would impose a false inductive bias; learned codes are order-free and
  more expressive. The fixed-orthonormal version was a *crutch* for the old read where addressing
  didn't matter — here addressing IS load-bearing (edges must hit the right nodes), so the task
  pressures the ids to stay distinct on their own. **Critical magnitude control:** give the id a
  *learnable, modest-init* scale (NOT √d). In the old slotgraph the id was scaled to √d and ended up
  supplying ~53% of the routing magnitude, anchoring the argmax → input-blind. We want the id to make
  nodes *addressable* without *dominating* the content-driven choice; the panel's `router_id_frac`
  metric watches this. Node id = its row; **edge id = `combine(id_src, id_dst)`** (endpoint-derived,
  from the learned node ids; see §2). Cost ≈ N·64 ≈ 9k params (negligible).
- `slot_init`: learned seeds `[N+E, d_node]` (mean-embed-scaled init, like icae).

Config fields (in `ReprConfig`):
```
sg2_d_node:int=64  sg2_n_nodes:int=144  sg2_n_edges:int=144
sg2_enc_layers:int=4  sg2_d_key:int=32  sg2_temp_init:float=1.0
sg2_xattn_every:int=1            # decoder cross-attn cadence (1 = every layer)
sg2_xattn_heads:int=4
sg2_lora_rank:int=…  sg2_lora_alpha:int=…   # set in the mixed capacity block
sg2_node_drop_max:float=0.5  sg2_node_drop_anneal_frac:float=0.5  sg2_node_drop_adaptive:bool=False
```

---

## 2. Encoder (the write) — build the graph-utterance

Two stages: **perceive with the frozen LM**, then **compose the graph** with a small transformer.

**Stage A — perception (reuse the LM's strength):**
- Passage embeds `[B,T,576]` → frozen LM (encoder-LoRA) → contextual features `H_ctx[B,T,576]`.
  (One frozen-LM forward, icae-style; no slots in the sequence.)

**Stage B — graph-transformer in `d_node`=64 (the new trainable speaker):**
- Init graph latents `G = slot_init + role + id` → `[B, N+E, 64]`.
- Repeat for `sg2_enc_layers` (=4):
  1. **cross-attn to the passage:** `G` (query, 64) attends to `H_ctx` with K/V down-projected
     `576→64`. → each unit gathers observation content. (This is the slot-attention/Perceiver write.)
  2. **self-attn among graph units** (64-d) — units mix (entities ↔ relations).
  3. **structure head** (only edges predict): edge→src-query, edge→dst-query, node→key, all from
     `[norm(G) ; id]`; `q·k` → log-Sinkhorn (competition) → straight-through argmax → hard `src,dst`
     `[B,E,N]` (no self-loops). **Soft** assignment kept for the gradient; **hard** for downstream.
  4. **re-inject** the materialized structure: `edge_latent += combine(id[src], id[dst])` where
     `combine = Linear([id_src ; id_dst] → 64)` — the endpoint-derived edge id (TokenGT-style); also
     re-add fixed role + node id. This is the feedback loop: next layer's self/cross-attn sees the
     current graph.
  5. FFN (64-d), residuals + norm.
- Output: final `G[B,N+E,64]` (node + edge latents, structure baked into edge ids) + final `src,dst`.

All graph-graph mixing happens here (Stage B self-attn). At read the graph is **frozen**.

Params (rough): cross-attn + self-attn + FFN + structure heads, all in 64-d × 4 layers ≈ **~1–1.5M**.

---

## 3. Decoder (the read) — frozen LM + per-layer gated cross-attention

The decoder reconstructs/answers; it reads the frozen graph via **bottleneck gated cross-attention**
at every layer (skip-VAE: injecting at many layers maximizes output↔memory mutual information).

- Target embeds → frozen LM (decoder-LoRA). After each decoder layer's self-attn, insert:
  **gated cross-attn (bottleneck, attend in 64-d):**
  - `q = W_dq(h_dec)` : `576→64` (down-project the decoder hidden to the graph space)
  - `k = W_k(G)`, `v = W_v(G)` : `64→64` (G is already 64)
  - `attn = softmax(q kᵀ/√dk + struct_bias?) v` ; `out = W_o(attn)` : `64→576` (up-project)
  - `h_dec += tanh(gate) · out` ; **gate is a scalar param init 0** (cold-start = pure LM/icae).
  - multi-head (`sg2_xattn_heads`=4) within the 64-d attention.
- Per-layer modules `{W_dq,W_k,W_v,W_o,gate}` ≈ **~82k params/layer × 30 ≈ 2.5M** (cheap because the
  attention is in 64-d, not 576). `sg2_xattn_every` lets us thin to every-k if needed; shared-across-
  layers is the param-light fallback.
- **Edges are KV tokens** in `G` (their latents, with endpoint-derived ids) — *content is a vector,
  never a scalar*. Connectivity rides in the edge id. (Optional later: an additive scalar
  `struct_bias` from the adjacency, Graphormer-style, as a routing aid — **off in v1**.)

Total new trainable ≈ encoder (~1.5M) + decoder cross-attn (~2.5M) + LoRA + heads ≈ **~5–7M** —
roughly icae-scale, so we can likely stay ~param-matched even with per-layer cross-attn.

---

## 4. Anti-bypass: stochastic node-dropout curriculum

Forces the *edges* to carry weight (kills "use nodes, ignore edges"). **One model, one forward per
step, dropout-style — no second model, no alternation, no new loss term.**

- In the decoder cross-attn, each forward, **drop a random fraction `p` of node KV entries** (mask
  them out of every layer's cross-attn) — **keep all edges** (a dropped node's incident edges remain,
  so the edge is the only trace of it → the read must use edges to recover).
- **Anneal `p → 0`** over `sg2_node_drop_anneal_frac` of training (cosine). Final phase: `p=0` →
  both paths full strength, **inference == training** (no train/test mismatch). The crippling is a
  curriculum that *shapes* the edges, then removed.
- **Stability:** it's dropout (never fully starved — decoder keeps its own context + surviving nodes
  + edges); the cross-attn gate (init 0) means crippling has ~no effect during the fragile cold-start
  and only bites once the read is in use; floor `p ≤ p_max=0.5`; balance vs the passage-mask so the
  task stays solvable.
- **Hyperparameter principle (avoid magic numbers):**
  - *simple:* cosine-decay `p` from `p_max→0`; pick `p_max` so node-only reading *fails* (verify: the
    structure-ablation gap at `p_max` should be large).
  - *principled (preferred, `sg2_node_drop_adaptive`):* a **panel-driven curriculum** — raise `p`
    while edges are unused (low cross-input diversity / low edge-contribution from the panel), lower
    it as edges contribute, → `p=0` when edges are load-bearing. Self-tuning, no fixed constant; uses
    the instrumentation we built as the controller. Start simple, switch on adaptive once it runs.

- **THE open risk — does annealing to 0 *revert* the model?** (still discussing.) When `p→0` the nodes
  return at full strength; if the edges only ever *substituted* for dropped nodes, the model can
  revert to reading nodes and ignore edges again — undoing the whole point. The anneal works only if
  the edges learn to carry **non-redundant** info. Three things make that hold, and we should lean on
  all of them: (1) **small nodes** — node content alone is insufficient, so edges stay useful even at
  full read; (2) **write-side feedback** — edges are materialized + re-injected during the *write*, so
  they shape the node latents regardless of read crippling (load-bearing through the representation,
  not just the read); (3) **adaptive schedule as a controller** — don't blindly decay to 0; keep
  *edges-load-bearing* as the control target and re-raise `p` if the edges-off gap shrinks.
  **Resolved:** **per-batch random** drop (one mask per batch — gentler than per-example, and lets
  whole batches see full, correct nodes), **cosine-decay `p` from `p_max → 0`** (no residual).
  **Fallbacks held in reserve** if the post-anneal edges-off gap snaps back to ~0 (= reverted): switch
  to *targeted* drop (drop the most-used nodes) and/or hold a small residual `p_min`. **The edges-off
  ablation gate, measured *after* annealing, is the acceptance test.**

---

## 5. Objective + training

- Same **mixed 4-task** (mae / babi / continuation / condrecon_bio), masked-reconstruction loss.
  Two orthogonal anti-bypass holes, same loss: **passage masking** (existing → "don't ignore the
  memory") + **node-dropout** (new → "don't ignore the edges").
- Encoder forward (perceive+compose) → `G`; decoder forward (read `G`) → loss. Two frozen-LM passes,
  like icae.
- Capacity: float-match the *memory* (18,432); aim to ~param-match too (the bottleneck cross-attn
  makes it affordable). If a config exceeds budget, label runs "capability test vs icae" and trim
  later — per the agreed priority (a capable success > a param-matched failure).
- Seeds {42,1,2}, same trainer/harness as the cohort.

---

## 6. Metrics / validation (reuse + extend the panel)

The `slotgraph_metrics.py` panel already covers WRITE/SELECT/MP-READ/OUTPUT. Adapt for slotgraph2:
- **SELECT** metrics unchanged (cross-input diversity, per-node variance, coverage, id-vs-content,
  key input-dependence) — now measured on the *encoder's per-layer* structure.
- **READ**: replace MP rank-by-hop with **cross-attn diagnostics** — gate magnitude per layer,
  cross-attn entropy (does the decoder attend sharply or smear), and **node-vs-edge attention mass**
  (how much read mass lands on edges — the load-bearing signal).
- **Anti-bypass gate (decisive):** structure-ablation at eval — REAL vs **edges-off** vs
  **nodes-off** vs **OFF** vs **SHUF**. Edges-off hurting ⇒ edges load-bearing (what we want). This is
  the metric the literature says to trust (not "the gate opened").

**Smoke checklist (before any train):**
1. Both forwards finite; loss finite; no NaN in bf16.
2. Cross-attn **gates init 0** → step-0 read ≈ pure LM (sanity: matches a no-memory floor).
3. Structure heads + cross-attn + encoder get real gradient (no starvation).
4. Node-dropout: drops nodes, keeps edges, p-schedule moves; finite with p=p_max.
5. Endpoint selection valid (edges→nodes, no self-loops); near-orthonormal ids built.
6. Panel runs end-to-end on a smoke checkpoint.

---

## 7. Implementation steps (files + order)

1. **`src/memory/config.py`** — add the `sg2_*` fields (§1).
2. **`src/memory/models/slotgraph/encoder.py`** — REWRITE in place: Stage-A perception + Stage-B
   graph-transformer (§2) + `finalize_memory` returning `G` + structure + panel canaries. (Old
   slotgraph encoder → git history / historical.)
3. **Decoder cross-attn** — a `GatedGraphXAttn` module (§3) inserted per decoder layer. Wire via the
   existing decoder-LoRA forward path (forward hooks or a thin custom layer loop). Node-dropout (§4)
   lives in this module's KV construction.
4. **`src/memory/model.py`** — keep the `slotgraph_baseline` variant (now the v2 architecture); thread
   `G` from encoder to the decoder cross-attn; expose canaries; honor the node-drop schedule.
5. **`scripts/train/train.py`** — update the slotgraph capacity block (new counts/rank); pass the
   global step/total to the node-drop schedule; flags `--sg-node-drop-max`, `--sg-adaptive-drop`,
   `--slotgraph-no-structure` (ablation, reused).
6. **`scripts/diagnostics/`** — update `smoke_slotgraph.py` (the §6 checklist) + extend
   `slotgraph_metrics.py` for the cross-attn read metrics + the edges-off ablation gate.
7. **Smoke → single-seed 4k → panel read → 3-seed** only if the smoke + single-seed look healthy.

---

## 8. Open knobs / sweeps / risks

- **Sweeps:** {d_node, N:E split}; cross-attn cadence (every-1 vs every-k); `p_max` + adaptive on/off;
  encoder depth `sg2_enc_layers`.
- **Risks:** (a) frozen LM may underuse the cross-attn even when trainable — mitigated by the gate +
  per-layer injection; the smoke's gate-grad check is the early warning. (b) **Bypass via surviving
  nodes** (node-dropout pressures but doesn't guarantee edge use) — the edges-off ablation gate is the
  judge; if it fails, escalate (keep dropped-node edges only / structural bias / harder drop).
  (c) Routing over 144 nodes is a bigger Sinkhorn — watch coverage/hub metrics.
  (d) Engineering: inserting per-layer cross-attn into the frozen HF decoder is the main lift.

---

## 9. The decisive question this whole design answers
Does making the structure **load-bearing in the write** (per-layer materialization), **read without
pooling** (gated cross-attn over many small units), and **forced** (node-dropout) finally produce an
**input-dependent topology** (cross-input diversity ↑, per-node variance ↑, edges-off gap ↑) that
**beats icae on babi** (relational binding), not just reconstruction? The panel answers it directly.
