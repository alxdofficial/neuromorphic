# Compression-by-Vocabulary — graph_v9 model design (current, 2026-06-14)

**Single source of truth for the active model.** Supersedes the retired operator/
Householder graph_v9 (`docs/archive/graph_v9_operator_householder_design.md`) and the
earlier v2.1 / graph_v5–v8 lineages (all in `docs/archive/`). Implemented in
`src/repr_learning/graph_substrate_v9.py` (+ `encoder.py` `GraphV9PyramidEncoder`).

## 0. Thesis & objective
- **Goal:** compress a passage into its OWN learned vocabulary — a directed graph of
  concept-nodes over a learned node bank — and reconstruct the text from it. Like
  language: shared atoms (words/concepts), novel arrangements (the per-input graph).
- **Objective: true MAE.** Mask ~85% of the span; predict the masked tokens in ONE
  forward from the code. Masked neighbours can't be guessed, so the code MUST carry
  content (this is what made the baselines show real SHUF−REAL, unlike EMAT).
- **Backbone:** frozen **SmolLM2-135M** (weak prior → wider floor→ceiling band →
  more pressure on the code).
- **Data:** FineWeb-EDU sentence PAIRS, 24–128 tokens, bucketed by code size.
- **Budget:** code = M = ceil(L / ratio) tokens (ratio 8), each d_llama wide. Capacity
  matched slot-for-slot to the baselines. **Target: beat AutoCompressor 4.64 recon =
  38% of the floor(6.96)→ceiling(0.81) band** (4000-step result, `compression_objective.md`).

## 1. Architecture overview (3 phases)
A frozen base contextualizes the span (one mid-stack tap, layer 6). The substrate then:
1. **Signal processing** — route each token against a per-layer learned node bank;
   re-express the token purely as its activated nodes' value-blend and pass it up
   (the sequence stays length L; layers re-describe, they don't shorten).
2. **Graph code** — multi-scale STDP co-activation over the token axis builds a directed
   graph over the active nodes; the informative edges become the code.
3. **Selection + render + read** — a transformer scores all candidate edges in context;
   E edge-query slots softly (sharp-softmax) pick one edge each; each edge renders as
   two stateless TokenGT node-tokens; a causal graph reader emits M memory tokens.

## 2. Phase 1 — signal processing (`_phase1`)
- 3 layers, nodes **(512, 256, 128)** low→high (multi-resolution), code width **d_code=256**.
- Each layer: cosine routing (unit query · unit node-key) / a calibrated, learnable
  temperature (no √d double-scale) → softmax activations `A_l [B,L,N_l]`. Keys are
  small-init so the cosine-unit direction actually learns.
- **Perturbation (no residual):** `x_{l+1} = norm( Σ_{top-k} A·node_value_l )`. The input
  token is **fully destroyed** — only the node re-description survives. This forces each
  layer into a more-abstract space and makes the slow node params data-load-bearing
  (a token in layer l+1 is *what layer-l's nodes said about it*).
- Per-node corpus marginal: bias-corrected EMA (Adam-style), the anti-hub denominator.

## 3. Phase 2 — multi-scale STDP graph code (`_forward_v2`)
- Prefilter to the top-**P=32** active nodes per layer (edges form among active concepts).
- **Multi-scale STDP**, K=3 learnable time-scales τ_k per source:
  `C_k[i→j] = Σ_t Σ_{Δ>0} exp(−Δ/τ_k)·A[t,i]·A[t+Δ,j]  =  A_iᵀ W_k A_j` (causal kernel).
  Sources = within-layer (relational) + adjacent inter-layer (compositional, low→high).
  τ is a learnable per-layer ladder × K log-spaced scales (small at atoms → large at top).
- Edge feature = **lift** (above-chance co-activation) `log( C / (marg_i·marg_j) )`, one per
  scale; within-layer self-loops (i=i) masked.
- Cheap prefilter: top-**Cand=48** candidate edges by max-over-scale lift (a recall step).

## 4. Phase 3 — selection + render + read
- **Candidate-edge transformer (whole-list view):** each candidate → token
  `[unit(src_value), unit(dst_value), K lifts, is_inter]` → `edge_in` → a 2-layer
  bidirectional transformer. Every candidate is scored *with attention to all the others*
  → it can learn the grammar + non-redundant coverage (not independent per-edge scores).
- **Soft selection (no hard top-k):** E = M//2 edge-query slots; each slot takes a
  **sharp softmax** (learnable temperature) over the candidates → near-one-hot ⇒ ≈ one
  edge, but fully differentiable and stable. Slot output = soft-blended `src_e`, `dst_e`,
  and contextualized `ctx_e`.
- **Stateless TokenGT render:** each selected edge → two node-tokens, src-then-dst:
  `token = unit(value) + role_emb[src|dst] + tag_emb[edge] + sel_to_tok(ctx)`.
  An edge IS the pair sharing one instance-tag; direction = role; relation/spelling =
  emission order. The `sel_to_tok(ctx)` term makes the selector **load-bearing** (else it
  only gates magnitude, which the output norm washes out — see §7).
- **Causal graph reader:** read_in → 2 transformer layers (causal mask) → read_out →
  `_NormMatch` (target = backbone embed norm, ~3.18). Causal ⇒ the first-k memory tokens
  form a valid **prefix code**, so the harness slicing to k is leak-free.
- Decode: `[memory ; masked_span]` → frozen Llama (+ LoRA) → MAE-CE on masked positions.

## 5. v1 ablation (`use_graph=False`)
Nodes-only: anti-hub activation-weighted node **centroids** + presence selection,
prepended directly (no edges, no reader — identical decode path to the baselines).
Used to answer "does the graph structure buy compression over flat node-slots?"

## 6. Config, params, capacity
- Selector d_sel=192 (2 layers, 4 heads); reader d_read=192 (2 layers).
- **~4.45M trainable** (3.52M substrate + 0.92M shared decoder LoRA). The dedicated reader
  + selector raise this above the 2M baseline anchor; the anchor was raised to ~5M (still
  tiny) — **baselines to be re-matched up to ~5M at launch** for a fair comparison.
- Capacity: emits M tokens, sliced to k=ceil(L/8) by the harness (prefix-valid).

## 7. Key design decisions (current — no contradictions)
- **Nodes = key+value vectors, slow backprop** (NOT Householder operators — operators
  bought state-tracking, which compression doesn't need).
- **No-residual perturbation** — destroying the input forces hierarchy + data-dependence;
  a fading residual left layers too similar so higher nodes never specialised.
- **Multi-scale STDP** — one τ can't serve both adjacent and long-range relations; K
  learnable scales + the selector picks which matters.
- **Soft, learnable selection** — hard top-k is unstable and non-differentiable, and a
  hand-coded NPMI+gate score optimises *surprise*, not reconstruction. A transformer
  selector (whole-list view) + sharp-softmax edge-queries is soft, stable, learnable, and
  redundancy-aware. The fixed budget (M) stays hard; *which* edges fill it is soft.
- **Edges carry a learned contextual summary** (`sel_to_tok(ctx)`) — a *mild* relaxation of
  "stateless edges". A score-only selector is gradient-starved (it only gates magnitude,
  which the output norm washes out → τ/keys grad/param ~5e-5). The contextual summary is
  load-bearing content, NOT the content-free pointer token the design originally rejected.
- **TokenGT node-tokens** `[role, value, instance-tag, edge-ctx]` — keeps the discrete
  directed graph representable; M generic Perceiver slots were rejected (dissolve the graph).
- **STDP statistics are the only input-specific signal into the graph** (content-free):
  routing activations in, learned selection out.

## 8. Ablations that decide whether the thesis mattered
- v2 (full graph) vs v1 (nodes-only) — does graph structure help?
- v1 vs a flat ICAE/Perceiver bottleneck at matched k + params — does vocabulary+selection
  beat a flat learned bottleneck?
- Compositional-generalization split — hold out sentence STRUCTURES (templates), not just
  sentences; a vocabulary basis should win if it's real.

## 9. Design history (superseded — see docs/archive/)
- Operator/Householder graph_v9 (retired; failed the binding gate): `graph_v9_operator_*`.
- Hard top-8 + STE selection, the plasticity-bilinear grammar, the residual perturbation,
  the M-generic-slot Perceiver read — all considered and replaced by §4/§7 above.
- v2.1 / graph_v5–v8 / EMAT / QA lineages: `docs/archive/`.
