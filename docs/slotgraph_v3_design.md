# slotgraph v3.1 — graph memory with load-bearing binding

**Status:** building (2026-06-27). Supersedes slotgraph v2 (`87e1c99`).
**v3.2** = v3.1 (4-facet literature validation, §0) + the **node-centric "vocabulary" refinement**:
nodes are a standing vocabulary whose *meanings* we tweak (soft, all nodes), NOT slots that input
tokens get routed into; hardness is **edges-only**. This dropped the token-routing node-write, the
per-token write gate, the τ-anneal, and the soft warmup.
**One-line:** turn the pooling memory (weakly input-dependent, membership-only) into a
graph memory (strongly input-dependent, binding) by making the discrete endpoint selection
**load-bearing** in the read — *and* pairing it with the anti-collapse machinery the
literature says load-bearing-ness alone does not provide.

---

## 0. Research validation (4 adversarial facets, 2026-06-26)

All four returned **YELLOW** → **conditional GREEN: build the corrected v3.1.** The plan's
*shape and core bets are validated*; the corrections below are non-negotiable.

| Facet | Verdict | Bottom line |
|---|---|---|
| 1. Load-bearing hard selection trains? | YELLOW | **Necessary but NOT sufficient.** Restoring gradient is step 1 of 2; discrete selection over ~144 options *still* collapses without anti-collapse machinery (MoE/VQ/PKM all confirm). Our prior query eff_rank≈1.4 was a *query-collapse*, orthogonal to gradient existence. |
| 2. Competitive hard write + delta? | YELLOW | Sound IF anti-collapse added. Three-stage collapse chain (geometric accident → value pooling → dead-node lock-in). **Deepest risk: value-pooling is architectural** — many tokens→one slot = pool-in-the-write. |
| 3. Frozen-LM reads bound triples? | YELLOW | Core fear refuted (frozen LMs *do* bind across separate tokens via additive id-tags). **Single-hop workable; multi-hop needs iteration.** Within-token additive SUM is the highest-risk choice → use concat-into-blocks. |
| 4. 4-layer refinement stable? | YELLOW | Fixed-address/writable-value split *is* better than GCNII. **Skip+FFN per layer is a hard requirement** (else rank-1 collapse, Dong 2021). β learnable init 0.3; post-delta LayerNorm. |

**Net:** the core bets — load-bearing selection, fixed-address/writable-value, edges-as-
separate-tokens into a frozen LM — are validated. Build with the §3.4 anti-collapse stack and
the §3.3 concat-binding correction from day one. Aux losses stay in the instrumented fallback
tier (respects no-aux-loss).

---

## 1. Why v2 failed (the diagnosis we are designing against)

v2 hit the **membership wall**: it could answer "is X present?" but not "what is the object of
relation r with subject X?". Concretely:

- **The selection was not load-bearing.** The read pooled node *content* directly (soft
  cross-attention = PMA = convex combination). Whichever node an edge "selected" had ~zero
  effect on the loss → selection heads got ~zero gradient (`sel_gradparam` 8.3e-4→1e-5 by step
  250; `src_entropy` glued to ln(144)=4.969). Definitive negative result (`87e1c99`).
- **Query-collapse, separately:** query eff_rank≈1.4 across inputs — queries *ignored the
  input*. This is distinct from the gradient problem and is NOT fixed by load-bearing-ness.
- **Structure bypassed twice** (edges cross-attended the passage directly; read pooled nodes).
- **Identity drowned the signal** (GCNII anchor re-injected input-*independent* identity into
  the full state each layer).

The objective is **not** the problem. The fix is **architectural**.

---

## 2. Core principles

1. **Separate address from content.** Every unit splits into a *fixed, input-independent* part
   (key / id / role) and a *writable, input-dependent* part (value / edge state). The fixed
   part is re-injected every layer **without drowning the signal** (it never touches the value
   channel). Validated as anti-smoothing (TokenGT fixed ids; NTM/DNC key-value separation; RIMs).
2. **The read flows through the selection — but that is necessary, not sufficient.** Edge tokens
   gather their selected endpoints' *values*, so the selection affects the loss → real gradient.
   This removes the *zero-gradient* failure but NOT the *collapse* failure → must be paired with
   the §3.4 anti-collapse stack.
3. **Binding, not pooling.** A relation is not permutation-invariant; pooling (PMA) provably
   destroys it. Preserve bindings by (a) **hard** endpoint selection and (b) **structured**
   (concat-into-disjoint-blocks) within-token binding so `A rel B ≠ B rel A` — never additive
   superposition of role-filler pairs into one vector.
4. **Compositionality = capacity.** An entity is a *subgraph* of generic primitives, not one fat
   slot. Reusable generic keys + specific values + which-edges-exist give combinatorial capacity
   from a small substrate, with *local* updates (add an edge / nudge one value, never clobber).
   The project's discrete-vocab-plus-grammar thesis, instantiated. Nodes are a *standing
   vocabulary* with distinct learnable base meanings, nudged by bounded deltas (§3.2) — not slots
   overwritten by pooled input, so the value-pooling failure mode does not arise here.
5. **Continuity (3 legs).** (a) **Stable addresses** — keys/ids/roles fixed, re-injected, never
   written; (b) **incremental content** — delta-rule writes accumulate; (c) **state-aware
   updates** — both writers attend holistically to the *current* full graph (so layer-L routing
   already sees what's been written, a built-in recurrent awareness across layers).

---

## 3. Architecture

### 3.1 Units

- **Nodes** (N=144): `[key_n (fixed, learnable, input-independent)]` + `[value_n = learnable base
  meaning + bounded contextual delta]`, plus fixed `id_n` (orthogonal) and `role_node`. The node is
  a *vocabulary word*: stable address + standing meaning that gets nudged. `d_node = 64`, `d_key = 64`.
- **Edges** (E=144): `[state_e (written)]` + binding `(src_onehot, dst_onehot)` over nodes +
  fixed `role_edge`. No fixed key of its own; its identity is its endpoints.

Binding handle = **slot id** (stable within a pass), not key content. Keys are generic; they only
need to be *distinguishable addresses*. Competitive write makes a slot specialize; the id is the
handle the edge re-selects.

### 3.2 Write — two sub-steps per layer (axis-1: within-passage refinement)

> **UPDATE 2026-06-27 (post operator-read run, `sgv3op`) — the escalation below is now ACTIVE.**
> The relation-operator read (§3.3, replaced concat) *structurally closed* the edge_state bypass
> (zero endpoints → zero bound, verified 0.00), and lifted gist binding (mae SHUF−REAL 0.07→0.20).
> But **babi hub-collapsed**: 144 edges selected ~6 nodes (`nodes_used 6.4`, `mem_effrank 1.3` =
> rank-1 read = pooling-by-hub) — the membership wall relocated from edge_state-bypass to
> node-usage-collapse. Root cause: the writes used **ordinary cross-attention (softmax over the
> *source*)**, so every slot independently pools the same content → redundant slots, no pressure to
> differentiate. **Fix (built): both writers now use COMPETITION (slot attention — softmax over the
> *slot* axis)** so slots compete for content and each specializes to a distinct aspect (Step A:
> nodes compete over the input; Step B: edges compete over input ∪ updated-nodes). Within-layer flow
> is now input→nodes→edges (no node↔edge back-messaging; competition does the coordinating).
> **Durable lesson — frozen scalar temperature:** sharpness must come from the **q/k projections**
> (scaled dot-product, 1/√d), NOT a learnable scalar temp on cosine logits. A lone scalar's gradient
> is tiny and Adam moves it only ~lr·steps — confirmed: `sel_scale` went 8.003→8.018 over a full 4k
> run, which is *why* selection (and would-be competition) never sharpened. Endpoint selection stays
> HARD (argmax forward is temp-invariant; the frozen soft-temp only shapes the Gumbel backward).

> **UPDATE 2026-06-28 (`sgv3comp` was NEGATIVE → Sinkhorn selection, `sgv3sink`).** Competition on the
> WRITE did nothing for babi (EM 0.242, identical to `sgv3op`; hub if anything worse) — because the
> collapse is in the discrete endpoint SELECTION, not the slot CONTENT (node_effrank was always ~37).
> Plain per-edge softmax normalizes rows (edges) only ⇒ a node can be picked by all E edges ⇒
> rich-get-richer hub. **Fix (built): BALANCED selection via Gumbel-Sinkhorn (Mena 2018)** — alternate
> col(node)/row(edge) log-normalization (`_sinkhorn`, 3 iters, `slotgraph_sinkhorn_iters`); the col step
> caps node over-subscription. Hard one-hot forward (argmax per edge), straight-through the balanced soft
> P. Selection STAYS hard (soft endpoints = a blended val_src = pooling = the wall by construction; the
> cure for an unbalanced hard assignment is to BALANCE it, not soften it). Precedent: SwAV uses Sinkhorn
> to stop all-samples→one-cluster collapse — isomorphic to all-edges→one-node. Probe: synthetic total hub
> (all edges→node0) → plain nodes_used 1/144, Sinkhorn 143/144 (col-mass cap 1.01 vs 142); real-path
> nodes_used 130 / avg_degree 2.2 (vs collapsed 4–6 / 45–72), bypass still 0.00, smoke green, 7.155M.
> **CAVEAT (unchanged): Sinkhorn guarantees SPREAD, not BINDING** — the 4k run tests whether balanced
> selection is meaningful (babi SHUF−REAL > 0) or just balanced-pooling (the 4th hatch relocating again).

Each GT layer: **node write, then edge write** ("choose the words, then draw the sentences").
Both writers tokenize and attend over all three at once (the "state-aware" leg of continuity):

- **Input tokens:** `frozen_LM_hidden + role_input`
- **Nodes:** `value_n + id_n + role_node`  *(value, NOT key — key is for selection only)*
- **Edges:** `state_e + role_edge + id_{src} + id_{dst}`

**Step A — node value update (soft, all nodes, NO selection).**
Nodes are a *vocabulary*; we tweak the contextual meaning of every word — we do **not** route input
tokens into slots. Build each node token = `value_n + id_n + role_node` (id/role re-injected, never
written). Attention mixes information — node tokens attend over input ∪ nodes ∪ edges (gather the
observation + see the current graph). A head reads the post-attention node representation → proposes
a **target meaning**, and we **delta-write toward it**: `value_n ← value_n + β · (target_n − value_n)`,
β learnable, sigmoid-bounded, *small* (bounded drift = continuity). `value_n` starts from a
**learnable base meaning** (distinct per node). No hard selection — every word's meaning is nudged.
Anti-collapse here is free: the distinct base + distinct keys/ids (each node reads through its own
lens) + the load-bearing read (selected nodes get gradient to stay distinct) keep the vocabulary
spread. Slot-competition (inverted attention) is the escalation *only if* node-value eff_rank collapses.

**Step B — edge write (delta) + endpoint selection.**
From the post-step-A state, per-edge heads emit: `Δstate_e` (delta-written), and `q_src`,
`q_dst` matched against the **fixed node keys** → `src_onehot`, `dst_onehot` via Gumbel-ST
(`dst` masked vs `src`). Selection is on **fixed keys** (stable), never on drifting values.

### 3.3 Read — edges only, materialized as STRUCTURED bound triples

We do **not** present nodes as tokens. Each edge → one memory token carrying relational content
*and* its endpoints' content, bound by **disjoint coordinate blocks** (structured / TPR-lite —
NOT additive sum):

```
edge_tok = W_e( concat[ state_e , gather(src_onehot, value_n) , gather(dst_onehot, value_n) ] )
```

- **Concat into disjoint blocks** = the role binding: src and dst occupy different coordinates ⇒
  `A rel B ≠ B rel A` for free, with **no within-token superposition** (this is the v3.0→v3.1
  fix; additive-sum of role-filler pairs rebuilds the pooled-additive failure *inside* the
  token — Smolensky cross-talk, TP-Transformer hierarchical ambiguity).
- `gather(·) = onehot @ value_n` is differentiable wrt **values** and (via the Gumbel-ST onehot)
  wrt **the query** ⇒ selection load-bearing on both content and structure.
- **Cross-token** id/role tags (so the LM/selection can tell edges apart) stay **additive** and
  **mutually orthogonal** — this is native to frozen LMs (Feng & Steinhardt 2024) and was our
  prior id-tag win. *Additive across tokens, structured within tokens — opposite answers, both right.*
- **Staleness fix:** store only the binding (onehots) + accumulated `state_e` across layers;
  gather **final-layer** node values at read time. Edges hold the stable skeleton; content read
  fresh; edges never go stale.

Injected into frozen SmolLM2 via per-layer **gated cross-attention** + projection (graph dim ≠
LM dim). The LM's layers do the relational reasoning over crisp bound triples.

**Multi-hop note (scope):** single-hop ("object-of(r, X)") is in-class for a frozen reader;
*true* two-hop ("object-of(r2, object-of(r1, X))") provably needs super-poly width in one layer
(Sanford 2023) and a 135M backbone is small. **Build single-hop-correct first.** Planned
multi-hop extension = **read-iteration** (re-query memory with the hop-1 result) or **2-hop
pre-materialization** — NOT a binding-format change. Do not expect 30 frozen layers to chain.

### 3.4 Anti-collapse stack (non-negotiable — build in from day one)

Load-bearing-ness restores gradient; these stop the selection/bank from collapsing anyway. All
architectural (no aux loss) except the explicitly-flagged fallback tier.

**Edge selection (HARD — escape the K=144 cold-start dilution + collapse attractor):**
- **Fixed moderate Gumbel τ (~1.0), NOT annealed** — the hard-forward argmax is τ-invariant, so τ
  only shapes the *backward* softmax. Keep it moderate so non-selected nodes get undiluted gradient;
  annealing τ→0 re-concentrates gradient on the winner (the dead zone), so we don't.
- **LayerNorm on the queries** + **L2-normalize keys AND queries** before the dot-product (PKM's
  query-spread fix, aimed at our measured query eff_rank≈1.4; DeltaNet magnitude stability).
  Selection logits in **fp32**. **Default (Kaiming) init, bias-free** selection projections — exact
  zero-init is unusable here (L2-normalizing a zero query is singular → dead/huge gradient), and a
  shared bias would tilt every edge's query toward the same keys; cosine of random directions is
  already small so the start is near-uniform + Gumbel-exploratory. *(No soft warmup — the soft backward
  already supplies the gentle early gradient, without a pooling phase that would entrench membership.)*
- **Dead-node revival** (track per-node selection EMA; reset unused keys) is a **canary-gated
  fallback, OFF by default** — enabled only if usage-entropy collapses past the above.

**Node anti-collapse needs NONE of the hard-selection machinery** — it is handled architecturally by
the distinct learnable base + bounded delta + distinct keys (§3.2).

**Encoder stability (4-layer GT):**
- **Skip connection + FFN in every layer** — HARD requirement; pure attention collapses to
  rank-1 doubly-exponentially regardless of depth (Dong 2021).
- **β learnable, sigmoid-bounded, init ≈0.3** (not 1.0) — avoids period-2 oscillation across the
  coupled node/edge delta updates; satisfies no-magic-numbers (learnable + principled init).
- **LayerNorm on values & edge-states after each delta write**, before next-layer attention.
- *(Optional)* preheat node values with a pooled-input linear projection before layer 1 (cold-
  start; SmoothSA) — cheap, parameter-light.

**Fallback tier (instrument first; add ONLY if collapse persists past the above):**
- Entropy / load-balance term on the selection marginal (Switch coeff ~0.01) and/or wav2vec2-
  style diversity. Treated as scaffolding, not permanent.

---

## 4. Explicit non-choices

- **No additive-SUM within-token binding.** Use concat-into-blocks (§3.3). *(v3.1 correction.)*
- **No read-side self-attention over graph tokens.** It smears (over-smoothing). Crisp tokens →
  LM attention does the mixing (our cohort: id-tagged crisp tokens beat message-passing).
- **No aux losses for anti-collapse** (except the instrumented fallback tier, §3.4).
- **No N ≤ d capacity shrink.** *(Corrected.)* In the slot/token regime capacity scales with N,
  not d; the d-bound only bites on per-node temporal saturation + key-selection crowding — both
  instrumented. Keep nodes/edges plentiful and small (compositionality wants this).
- **No streaming merge yet (axis-2).** Prove within-passage binding first; delta machinery
  transfers cleanly to persistent streaming + bounded eviction later.
- **No selection on values.** Always on the fixed keys.
- **No expectation of frozen multi-hop chaining in one pass** (see §3.3 multi-hop note).

---

## 5. Config (first decisive run)

| knob | value | note |
|---|---|---|
| N nodes / E edges | 144 / 144 | many-small for compositionality |
| d_node / d_key | 64 / 64 | selection in full node dim |
| layers | 4 | each: skip + FFN (mandatory), node-write→edge-write |
| edge binding | concat[state, src_val, dst_val] → W_e | structured/TPR-lite (NOT sum) |
| roles / ids | fixed, **mutually orthogonal**; cross-token additive | native to frozen LM |
| node write | **soft** per-node meaning-tweak: head→target, `value += β(target−value)` | β learnable, small init; distinct learnable base; NO selection, NO gate |
| edge write | delta on state; Gumbel-ST src/dst on fixed keys | dst masked vs src |
| Gumbel τ | **fixed ~1.0, NOT annealed** | forward argmax is τ-invariant; τ shapes backward only |
| selection init/norm | default bias-free init; LayerNorm(q) + L2-norm(q,k); fp32 logits | PKM + DeltaNet; **no soft warmup** |
| dead-node revival | canary-gated fallback, **OFF by default** | edge selection only |
| read | edges-only structured triples, fresh final values | gated per-layer cross-attn, projected |
| scope | within-passage (axis-1), single-hop first | streaming + multi-hop deferred |

Frozen SmolLM2-135M, ctx 1024 → M=32, mixed 4-task objective, ~4k steps (~14 min), capacity-
matched cohort. Keep the standing grad-norm canary.

---

## 6. Instrumentation (success + watch-points)

**Decisive success test:**
- **bAbI EM**, split **Match2 (single-hop) vs Match3 (two-hop)** — disentangles memory-binding
  failure from frozen-reader-chaining failure. (Don't read a low aggregate EM as a memory failure.)
- **SHUF−REAL** — should now be clearly > 0 (v2 ≈ +0.035 = pools). The membership-wall discriminator.
- **OFF−REAL** stays large (memory essential — already true in v2).

**Live canaries (collapse early-warning; check by step ~200–1000):**
- selection grad/param (`sel_gradparam`); **node-usage entropy / src-dst eff_rank** (the
  collapse canary — if rank drops < 5 in first 200 steps, suspect missing query-norm / decoupled-τ).
- `mem_effrank` (must NOT decline like v2's 5→3); per-node value eff_rank (saturation).
- key eff_rank / selection confusion (key-crowding among 144).
- **read-side attention sharpness** (max weight / entropy over the 144 tokens) = binding-vs-
  pooling canary on the read.
- **node/edge norms** (`slotgraph_node_norm`/`edge_norm` ≈ √dn=8 after post-delta LayerNorm) — drift canary.
- **selection grad is reported for the LAST layer only** — it alone sets the read topology (the read
  gathers the final-layer onehots); an all-layer aggregate would mask last-layer starvation.

**THE #1 risk to settle empirically — edge_state bypass (the recurring wall).** `edge_state` is a
full-width (dn) free channel concatenated into the read, so it *could* carry the whole answer and make
the hard endpoints vestigial — exactly the v2 `edge_state`-bypass ([[project_graph_edge_state_bypass]]).
v3.1 differs (the selection is load-bearing — `node_key.grad` confirmed nonzero, sel-gap 8–12×), so it
*may* escape; can't be settled statically. **Test:** the `read_ablate` flag — `zero_values` (zero the
endpoint meanings; if loss ≈ REAL ⇒ bypass, edge_state carries it; if ≈ OFF ⇒ endpoints load-bearing)
and `zero_state` (zero edge_state; if ≈ REAL ⇒ endpoints carry it). Run post-hoc on the 4k checkpoint.
**Ready fix if it bypasses:** bottleneck/discretize `edge_state` (relations are low-dim *types*, not a
content dump) to force entity content through the endpoints — the v2-confirmed remedy.

---

## 7. Literature anchors

**Closest prior art / the gap** — EntNet (Henaff 2017, fixed-key writable-value, solved bAbI),
RelNet (Bansal 2017, EntNet+edges — *nearly this design, but soft gates ⇒ pooled ⇒ our wall*).
Novel unbuilt gap: fixed-key slots + **hard** competitive assignment + **structured** sparse
edges + **frozen-LM** reader + bounded eviction.

**Validation anchors (by facet):**
- *Selection:* Jang 2017 (Gumbel-ST), Shah 2024 (decoupled fwd/bwd τ), Lample 2019 (PKM query-
  norm), Berthet 2020 (perturbed optimizers — plain ST structurally insufficient), Fedus 2021 /
  van den Oord 2017 (load-bearing routing/codebooks still collapse).
- *Write:* Locatello 2020 / Zhang 2023 (slot competition + collapse modes), Zheng & Vedaldi 2023
  (dead-code revival), Yang 2024 (DeltaNet key-norm), Henaff 2017 (gated delta + content×location).
- *Read:* Feng & Steinhardt 2024 (frozen LMs bind via additive id-tags — refutes re-pooling
  fear), Sanford 2023 (single-hop in-class, two-hop super-poly), Schlag 2019 (TP-Transformer —
  additive ambiguity, multiplicative/structured fix), Sukhbaatar 2015 (MemN2N lost chaining when
  soft), Alayrac 2022 (Flamingo frozen-LM + gated cross-attn reads, binding ceiling), Févry 2020
  (EaE hard 61.8 vs soft 46.9).
- *Encoder:* Dong 2021 (rank collapse — skip+FFN mandatory), Chen 2020 (GCNII), Kim 2022
  (TokenGT fixed ids), Schlag 2021 (delta stability), Ramsauer 2021 (Hopfield binding basin).

**Theory backbone:** pooling = membership-only (Set Transformer / PMA, Lee 2019); TPR/VSA bound
triples recovered by contraction (Smolensky; Plate; Kanerva); capacity ≈ d per superposed state.

---

## 8. Build order

1. **This memo** ✅ (v3.1, validated) — review before code.
2. **`_GTLayer` rewrite** (`encoder.py`): split key/value; step-A competitive gated-delta
   node-write; step-B delta edge-write + Gumbel-ST (decoupled τ) on fixed keys; query LayerNorm
   + L2-norm; re-inject fixed key/id/role each layer; **skip+FFN+post-delta LayerNorm per layer**;
   learnable β; dead-node revival hook.
3. **Read materialization** (`model.py` read hooks): edges-only **concat-block** triples, gather
   final-layer endpoint values, orthogonal fixed roles, projected per-layer cross-attn.
4. **Config + canaries** (`config.py`, `train.py`): roles, write-gate, τ_f/τ_b schedule, soft
   warmup, dead-node revival, watch-point instrumentation (incl. Match2/Match3 split, read
   sharpness).
5. **Smoke** (`smoke_slotgraph.py`): finite both paths; params ~capacity; selection gets
   gradient; **node-usage entropy + eff_rank healthy at init**; directionality canary nonzero —
   *before* any 4k run.
6. **Single-seed 4k run** → SHUF−REAL + Match2/Match3 EM + node-value eff_rank + read sharpness.

---

## 9. Open decisions (locked unless instrumentation says otherwise)

1. **Node write:** **soft per-node base+delta meaning-tweak** (chosen) — no token routing, no
   per-token gate. Slot-competition (inverted attention) / EntNet gate are escalations *only if*
   node-value eff_rank collapses.
2. **Within-token edge binding:** concat-into-disjoint-blocks → W_e (chosen). Full multiplicative
   TPR `role⊗filler` is the escalation if concat-block proves insufficient.
3. **Scope:** within-passage axis-1, **single-hop first** (chosen). Streaming delta-merge AND
   read-iteration for multi-hop both deferred (read-iteration is the planned multi-hop path).
4. **N / E / layers:** 144 / 144 / 4 (chosen; don't confound the decisive run with a count change).
