# Compression-by-Vocabulary — model design (converged 2026-06-13)

Successor design to the graph_v9 memory line, for the **compression objective**
(docs/compression_objective.md). The graph_v9 operator/Householder line is retired
here — see "Decisions" for why. This doc is the build spec.

## 0. Thesis & objective
- **Goal:** compress a passage into its OWN learned vocabulary — a directed graph
  of concept-nodes — and reconstruct the text from it. Compression, NOT lifelong
  persistence (persistence is the separate, deferred concern).
- **Objective: true MAE.** Mask ~85% of the span; predict the masked tokens in
  ONE forward from the code. Masked neighbors can't be guessed, so the code MUST
  carry content (this is what made the baselines show real SHUF−REAL, unlike EMAT).
- **Backbone:** SmolLM2 (135M), FROZEN — a weak prior to stress the memory
  (composes with MAE: removes both the local and the global guess-cheat).
- **Data:** sentence PAIRS from FineWeb-EDU, 24–128 tokens, bucketed by code size.
- **Budget:** code = M = ceil(L / 8) tokens, each d_llama-wide; capacity- AND
  param-matched (~2M trainable) to the baselines. **Target to beat: AutoCompressor,
  4.64 recon = 38% of the floor(6.96)→ceiling(0.81) band** (4000-step result).

## 1. Slow parameters (learned by backprop)
Per layer: node KEYS + node VALUES (the vocabulary); routing projection;
perturbation projection + gate; ID projection (key → shared d_id); layer
embedding; learnable STDP decay τ (small at atom layers → large at the top).
Global: plasticity-bias MLP (the learned co-activation grammar); role embeddings
(source / destination / standalone); instance-tag bank (learnable grouping keys);
per-layer emit projections (→ d_llama); (v2) the dedicated graph-reader; Llama
LoRA; mask_embed.

## 2. Phase 1 — signal processing (encoder forward; input passes ONCE)
1. Tokens → frozen LM → hidden states [L × d_model] (one mid-stack tap).
2. Layer 0: score hiddens vs layer-0 keys → activation A0 [L × N0]; PERTURB each
   token = norm( proj(token) + Σ_{top-k activated} A0·value0 )  — residual +
   sparse + norm (re-describe for routing; no smear). Pass up.
3. Layers 1..K: identical, each on the perturbed codes from below (own
   keys/values/dims; projection between layers). Emit Aℓ [L × Nℓ] per layer.
   SEQUENCE STAYS LENGTH L — layers re-describe, they do not shorten.

## 3. Phase 2 — graph-code production (read off the activation traces)
4. STDP (one unified metric over the token axis), for node pairs:
   C[i→j] = Σ_t Σ_{Δ>0} exp(−Δ/τ) · A[t,i] · A[t+Δ, j]
   - WITHIN-layer (same ℓ) = relational edges; INTER-layer (cross ℓ) =
     compositional edges with a low→high direction prior. The decay kernel IS the
     soft window; τ is the per-layer ladder; shaped by the plasticity bias.
5. Score edges: NPMI (anti-hub Bayesian surprise = above-chance co-activation) +
   a reconstruction-trained selection score.
6. Select within budget M: edge-centric — most informative directed edges + their
   endpoint nodes; MULTI-RESOLUTION (low layers for literal/spelling, high for
   abstract); preserve firing-order.
7. Build TOKENS (all node-tokens; NO edge tokens — edges are stateless):
   each token = [ role(src/dst/standalone) , value , instance-tag ].
   - An EDGE = a pair of node-tokens sharing one instance-tag (one src, one dst),
     emitted in firing order. Direction = role; relation TYPE & spelling order =
     emission order. Standalone selected node = role `standalone`, no tag.
   - High-degree nodes are DUPLICATED (a 3-edge predicate appears 3×); token count
     ≈ 2·(#edges); accepted (selection caps it at M). All projected to d_llama.

## 4. Phase 3 — decode (reconstruction)
8. (v2) DEDICATED graph-reader: a small FULLY-trainable graph-transformer over the
   tokens (groups by instance-tag, orients by role) → M soft memory tokens in
   Llama space. (v1: skip — prepend node-tokens directly.)
9. Decoder input: [soft memory tokens] + [masked span] (mask_embed at ~85% of
   real positions).
10. Frozen Llama (+ LoRA): predict the true token at each masked position (MAE CE).
    Loss backprops → updates everything in §1. That IS phase 1.

## 5. Three timescales
- Backprop (slowest): vocabulary content + reader + LoRA — WHAT each concept is.
- Plasticity bias (slow): corpus-level co-activation grammar — WHICH concepts relate.
- Per-input STDP (fast, per forward): the instantaneous directed graph — the CODE.

## 6. Decisions (with the one-line reason)
- Nodes = key+value VECTORS, slow backprop (NOT Householder operators — operators
  bought state-tracking, which compression doesn't need).
- Perturbation = residual + top-k + norm (NOT mean/attention pool, which smears
  over depth). It only re-describes for routing; STRUCTURE is carried by the STDP
  graph, so the perturbation need not be order-sensitive.
- Edges are STATELESS directed links — no edge vector, hence NO edge tokens. Roles
  & spelling come from direction + emission order; meaning lives in the vocabulary.
- Tokens are all NODE tokens with [role, value, instance-tag]; instance-tags group,
  roles orient. (Deduped-ID alternative rejected: it reintroduces content-free
  pointer tokens.)
- One unified STDP metric for within- AND inter-layer edges (cross-layer
  co-activation = composition, free from the same tracking; low→high prior).
- Edge state input-specificity (if ever re-added) would flow ONLY through STDP
  statistics (content-free), never raw hiddens — but v1/v2 use stateless edges.
- Selection = NPMI prior + reconstruction-trained (anti-hub; coverage from MAE).
- Variable per-layer dims are FREE (projections between layers); code-tokens unify
  to d_llama. (Per-slot pointer-dims for entities = future refinement.)
- Decode via a dedicated trainable reader (v2): the graph/instance-tag format is
  far OOD for a frozen decoder; LoRA-only is too low-rank to parse it.
- Edge type is recoverable from direction + emission order + node identity; an
  explicit typed-edge head is the v2+ add-back ONLY if reconstruction shows
  role-confusion.

## 7. Staging
- **v1 — nodes-only:** no edges, no reader. Prepend M node-tokens → LoRA Llama
  (IDENTICAL decode path to the baselines). Proves "does our vocabulary +
  selection beat AutoCompressor 4.64" on a clean apples-to-apples comparison.
- **v2 — full graph:** add directed STDP edges (within + inter-layer), the
  stateless instance-tag node-tokens, and the dedicated graph-reader. Tests
  whether the GRAPH STRUCTURE buys compression over flat node-slots.

## 8. Ablations that decide whether the thesis mattered
- v1 vs a FLAT ICAE/Perceiver bottleneck at matched k and params (does our
  vocabulary+selection beat a flat learned bottleneck?).
- v2 vs v1 (does the graph structure help?).
- Compositional-generalization split: hold out sentence STRUCTURES (templates),
  not just sentences — where a vocabulary basis should win if it's real.

## 9. Relation to prior lines
- Chunking/segmentation lineage = H-Net dynamic chunking (we read the code off the
  STDP graph instead of physically merging, but the adjacency-similarity idea is
  shared). Learned vocabulary = VQ-VAE codebook lineage. Hebbian/STDP concept
  formation = cell-assembly literature (Garagnani 2009 — thresholded/competitive
  plasticity keeps assemblies DISTINCT, the anti-collapse fix). Graph-as-tokens
  decode = TokenGT. The novel combination: emergent STDP graph over a learned
  vocabulary, stateless ordered edges, as a compression memory.

## v2 IMPLEMENTED + design evolution (2026-06-14)
Full v2 built (use_graph=True default). Two design changes from §3-4, both forced by
debug sweeps + the "make it learnable" principle:

1. **Edge selection = a learnable TRANSFORMER selector, not the hard-coded NPMI+gate.**
   Cheap STDP-lift prefilter → top-48 candidate edges → a small bidirectional
   transformer over the candidates (the "whole-list view": each edge scored WITH
   attention to all others → learns grammar + non-redundant coverage) → keep-logit
   → STE-top-8 (discrete graph forward, soft gradient back). STDP lift/C are kept as
   INPUT FEATURES (thesis intact); the plasticity bilinear is subsumed by the
   selector. Reason: independent per-edge scoring (even a learned MLP) is myopic to
   redundancy; PMI/lift optimizes surprise, not reconstruction-relevance.

2. **Edges carry a learned contextual summary (mild 'stateless' relaxation).**
   A score-only selector is gradient-starved: keep-logit only gates token magnitude,
   which token_norm washes out (selector+τ+keys at grad/param 5e-5). FIX: the
   selector's contextualized edge rep feeds token content (sel_to_tok) → it's
   load-bearing → node_keys 1.9e-4→0.059, τ 5e-5→0.02, selector 3.7e-3→0.67. A
   learned summary is NOT the content-free pointer §6 rejected.

Params 4.59M (3.67M substrate + 0.92M LoRA); anchor raised to ~5M (still tiny) —
baselines to be re-matched up at launch for a fair comparison. Known-weak: sel_head
(which-edge decision) is weakly trained — inherent to hard top-k; Gumbel-top-k is
the upgrade IF the run shows selection is the bottleneck. Diagnostics: scripts/
_v9v2_diag.py (gradient/selection health), _v9_why.py (eff_rank / per-slot value).
