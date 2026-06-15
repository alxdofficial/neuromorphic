# The `graph` model — VQ-codebook graph memory + TokenGT controller

Design from the 2026-06-15 design dialogue. Supersedes the abandoned
`hierarchical_learned_vocab` (graph_v9) and `soft_pointer_graph` (graph_v6), both of
which hit the rank-1 read/membership wall (see memory: project_mae_4k_collapse_result).

## Thesis context (the three parts)
1. **Learned vocabulary = a stable base to express information** → a VQ-VAE codebook.
2. **Graph + an inference-active controller = a persistent, editable memory state** → this model.
3. **Hebbian/plasticity** → *dropped*: a learned controller that stays active at inference
   subsumes a hand-coded Hebbian rule. The TokenGT update *is* the plasticity.

This model is parts 1+2. The single fix that makes it different from every prior version:
**the graph's endpoints are discrete VQ codes → distinct addresses**, the thing the
soft-pointer (v6, continuous bank) and delta-write (v8, convex-average keys) graphs never had.

## Components

### VQ codebook — *purely a quantizer* (not a passage codec)
A learned codebook of `n_codes` vectors in `d_graph`. There is **no** "encode the passage
into codes / decode codes into the passage." The codebook only **snaps** the continuous
endpoint representations the TokenGT predicts to the nearest code (straight-through). The
codebook is the shared discrete vocabulary the graph endpoints live in. Use a standard VQ
(EMA codebook + commitment loss + STE; `vector-quantize-pytorch` if available, else inline).

### Graph state — the persistent memory
A fixed budget of `K` edges. Each edge stores:
- `src_code`, `dst_code` — **code indices** (discrete; the endpoints).
- `edge_state` — a continuous `d_graph` vector (the relation).
- `instance_tag` — a fixed, distinct per-edge id (learned), for separability.

**Storage vs tokenization:** store compact code *indices* + edge-states + tags; when the
TokenGT or reader operates, materialize each code's `d_graph` **codebook vector**. Store
indices, embed on use. v1: graph resets per observation; later: persists across a stream.

### TokenGT write — the controller (modifies nodes & edges)
A cross+self transformer over the graph tokens. Per layer, ×N:
1. **cross-attend** graph tokens → the LLM observation (Llama hiddens of the passage),
2. **self-attend** graph tokens over each other (nodes + edges; holistic global rewire),
3. FFN.

Graph tokens (TokenGT-style): per edge → a source token (`codebook[src] + role_src +
instance_tag`), a destination token (`codebook[dst] + role_dst + instance_tag`), and an
edge-state token (`edge_state + edge_emb + instance_tag`). After the stack, the updated
**node-endpoint** reps are **VQ-snapped → new code indices**; edge-states stay continuous.
The controller stays active at inference (it ingests each observation and edits the graph).

### Custom reader — bridges `d_llama ↔ d_graph`, injects into the frozen LLM
The reader keeps the graph in its own `d_graph` (decoupled from `d_llama`). At one mid-late
LLM layer, the per-position residual hidden *is* the query (a forward hook taps it — no
special query emission). A cross+self transformer over the decode positions, per layer ×M:
1. **cross-attend** decode positions → the graph snapshot (K/V from the edge tokens),
2. **self-attend** decode positions over each other — **CAUSAL** (frozen causal LLM),
3. FFN.

Then project `d_graph→d_llama`, RMS-match to the residual stream, gated add into the LLM
residual at the inject layer.

## Non-negotiables (or the read collapses like every prior version)
1. **QK-RMSNorm + learnable temperature on the read cross-attn.** Without it, 1/√d init →
   near-uniform softmax → no sharpening gradient → the read averages everything (v8c cold-start).
2. **RMS-match the injected output to the residual stream.** v8c found the inject ~2e4× under
   the stream → differential signal ≈ 0 → SHUF=REAL. The `Wback` output must be RMS-normed
   to the stream, then gated.
3. **Distinct keys are what make it work now.** K/V come from distinct VQ codes + per-edge
   instance tags → the cross-attn finally has separable addresses. Every prior collapse was
   this exact read over smeared/membership keys.

## What's trained vs frozen
- Frozen: the Llama backbone (observation source + the decoder being injected into).
- Trained: the VQ codebook, the TokenGT write stack, the custom reader. (Decoder LoRA optional,
  shared, as in the baselines.)
- The graph is *state*, not parameters.

## Objective
- **v1: single-passage MAE** (one passage → write graph → read → reconstruct masked tokens) —
  de-risks the VQ→graph→read pipeline on the existing harness.
- **then: streaming/continual** (observe a stream → graph accumulates via the TokenGT → query
  later, SHUF-gated) — where the persistent graph + inference-active controller earn their keep.

## Defaults (capacity-matched ~4–5M to the baselines)
`K≈8` edges, `n_codes≈1024`, `d_graph=256`, write stack `N=3`, read stack `M=2`, 4 heads,
TokenGT cross-attends a mid Llama tap, 1 inject layer (mid-late).

## Gates
- mae_smoke (constructs/trains/grad-flows on masked_reconstruction).
- eff_rank of the read-injected signal + REAL/OFF/SHUF (the binding gate — the bar prior
  models failed). Distinct VQ keys + the two read fixes are the bet for clearing it.
