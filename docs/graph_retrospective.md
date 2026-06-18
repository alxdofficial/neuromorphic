# Graph Memory Compressor — Retrospective & Decision to Move On

**Date:** 2026-06-17
**Verdict:** The learned-graph memory compressor underperforms simple flat-bank baselines on the
one clean discriminative task (masked reconstruction), at *equal or greater* parameter count, and the
failure is **architectural and provably fundamental** for this task — not a bug, not under-training, not
an encoder handicap, not under-sizing. We are moving the primary line to a **fast-weights / Hebbian**
memory. This document is the evidence.

---

## 1. How the current graph model works

A **frozen** SmolLM2-135M contextualizes the input span; one hidden layer is tapped as the
*observation*. A learned **relational parser** turns that observation into a small graph, and a
**reader** turns the graph into memory tokens that are **prepended** to the frozen decoder.

**Write — `GraphParser` (TokenGT-style).** The working set is two copies of E edges: Part 1 = the current
graph with values (`[src, edge, dst]` + role + per-edge instance tag), Part 2 = value-less *prediction
slots*. Per layer (×`write_layers`): self-attend the working set → cross-attend the node bank → cross-
attend the observation → FFN. The new graph is read off Part 2:
- **Discrete endpoints (vocabulary mode):** each src/dst slot emits a query that **points** (QK-RMSNorm +
  temperature softmax, or entmax) into a learnable **node bank** of N vectors; the endpoint *snaps* to a
  bank vector. This is the "compression-by-vocabulary" thesis — endpoints are reused identities.
- **Free endpoints mode:** the bank/pointer is dropped; src/dst are regressed directly as free vectors
  (no reuse, no topology).
- Edge state is regressed by a head in both modes.

**Read — `GraphReader`.** Per edge, a FiLM bind `op(src, dst, edge)` → one vector; the E edge tokens
self-attend, then project to `d_llama`. These E tokens are **prepended** as memory; the frozen decoder
reads them via its own attention (decoder-LoRA, rank 16, shared by all variants).

Architecture source of truth: `src/memory/models/graph/substrate.py`, `.../graph/encoder.py`.

---

## 2. Experimental circumstances

- **Backbone:** SmolLM2-135M (30 layers, d=576), **frozen**. All compression happens around it.
- **Data:** FineWeb-EDU; objective-specific framings (sentence pairs for MAE).
- **Decoder adaptation:** rank-16 LoRA on the frozen decoder, **identical for every variant** (≈0.92M).
- **Baselines** (icae, ccm, autocompressor, beacon): frozen Llama + encoder-LoRA + a handful of learnable
  query/slot tokens; the compression is done **by Llama's own attention**, not a custom module. Bespoke
  params ≈ 0.01M; total ≈ **6.9M trainable**.
- **Band:** every objective is bracketed by `vanilla_llama` (no memory = **floor**) and
  `vanilla_full_context` (the full text in context = **ceiling**). "% of band" = fraction of
  floor→ceiling recovered. Best baseline `best_step` 3500, 4000 steps, BS 32 on a 4090.
- **Fairness work done for the graph:** we (a) switched the read from a collapsing cross-attn inject to a
  prepend; (b) added encoder-LoRA + read-final so the graph adapts its encoder *like the baselines do*
  (it had been reading a frozen tap — a real handicap); (c) verified per-parameter gradient flow.

---

## 3. Results

### 3a. Masked reconstruction (MAE) — the clean discriminator
Band: floor **6.978** → ceiling **0.300** (width 6.68). Higher %band = better.

| Variant | Trainable | val_recon | **% band** | binding (off−real) |
|---|---|---|---|---|
| vanilla_full_context (ceiling) | — | 0.300 | 100% | 10.72 |
| **autocompressor** (best baseline) | ~6.9M | 4.188 | **41.8%** | 5.91 |
| beacon | ~7.0M | 4.282 | 40.4% | 5.84 |
| icae | ~6.9M | 4.713 | 33.9% | 5.89 |
| ccm (worst baseline) | ~6.9M | 5.177 | **27.0%** | 4.57 |
| graph — even-encoder (+enc-LoRA) | **13.1M** | 5.756 | 18.3% | 3.24 |
| graph — free endpoints, no enc-LoRA | ~6.0M | 6.174 | 12.0% | 1.63 |
| graph — prepend (discrete) | ~6M | 6.218 | 11.4% | 2.34 |
| graph — original | ~4–6M | 6.512 | 7.0% | 1.01 |
| vanilla_llama (floor) | — | 6.978 | 0% | 0.00 |

**The best graph variant (18.3%) sits below the *worst* baseline (ccm, 27%) — and it got there only with
a 2× parameter advantage (13.1M vs 6.9M).** At fair parameter count (~6M, free endpoints) it recovers
**12%**, roughly a quarter of the best baseline. Binding (off−real gap) is also weak everywhere
(≤3.2 vs baselines' 4.6–5.9): the graph barely uses its own memory.

### 3b. Conditioned reconstruction — a degenerate band (graph's *best* showing)
Band is **inverted** (floor 2.730 *better* than ceiling 6.095): putting the full text in context *hurts*,
so the task is a poor discriminator (the decoder effectively sees the target twice). Here the graph is
**competitive**: `graph_baseline` 2.522 (binds, off−real 1.79) sits among icae 2.462 / autocompressor
2.499 / ccm 2.542. So the graph is **not categorically broken** — on a task where structure can bind, it
keeps up. But this is the *degenerate* task, not the discriminator.

### 3c. Continuation — a near-zero band (uninformative)
Band width is only **0.19** (memory barely helps *anyone*: ceiling 2.720 vs floor 2.911). Baselines
recover 10–26% of this tiny band; the graph **fails to bind at all** (`graph_baseline` off−real 0.000, at
the floor). When the band is this narrow the result carries little signal — but the graph is the one model
that doesn't bind even slightly.

**Cross-line corroboration:** a parallel architecture (the `hlvocab` / `soft_pointer_graph` arms on the
v9 branch) independently hit the same wall on a matched 4k MAE run — pinned at ~5% of band with the
emitted memory collapsing to ≈rank-1 (see memory `project_mae_4k_collapse_result.md`).

---

## 4. Why it failed — ruling out the easy explanations

We systematically eliminated every "it's just X" explanation:

1. **Not a bug / gradient starvation.** Per-parameter audit on the real training path: **109/109 trainable
   tensors receive nonzero gradient** (`src_head`/`dst_head` ~72–76, `obs_proj` ~184, FiLM/`w_sd` active,
   decoder-LoRA active). Nothing is starved.
2. **Not an encoder handicap.** We evened the footing — encoder-LoRA + read-final so the graph adapts its
   encoder exactly like the baselines. It improved to 18.3% and **still lost to the worst baseline**.
3. **Not under-sizing / under-capacity.** Diagnostics on the even-encoder run: vocabulary **N=1024 but only
   ~55 nodes ever used (~5%)**; `bank_effrank` 204/256 (the bank is *diverse*, just unused); `d_graph` not
   saturated; `edge_effrank` 13/16 (edges are fine). The model is over-provisioned, not under. And it loses
   at **2× the baselines' parameters**.
4. **The actual mechanism that breaks: selection.** The endpoint pointer **never commits** — pointer
   entropy 4.89 nats (≈130 effective nodes per choice). It's a soft blend, not a discrete selection. The
   "graph" never becomes a graph.

---

## 5. Why it's *fundamental* on this task (the theory)

At a fixed read interface (degrees of freedom = number of stored reals), a structured memory is a
**reparameterization** of a flat one: its reachable states are the *image* of a map, always ⊆ the flat
codomain. **Structure cannot exceed a flat bank of equal DOF — except in one way: if the topology
(which node connects to which) varies with the input, the wiring itself stores extra bits** (≈`E·2·log₂N`
per input) that a flat bank has nowhere to put.

Realizing those bits requires **discrete, committed** selection. A soft blend (entropy 4.9, 55/1024 nodes)
carries almost none — so the graph collapses into a *worse-conditioned flat memory*. That is exactly what
every number above shows.

And masked reconstruction is the **worst possible task** for the thesis: it is high-entropy *verbatim*
recall, where the optimal code is flat (maximum-entropy), and structure has nothing to add. Flat banks are
provably optimal here; the graph is paying for structure it can't use. (Full argument:
memory `project_structure_vs_flat_proof.md`.)

---

## 6. What is salvageable

- The graph **binds and stays competitive on conditioned reconstruction** — structure isn't worthless, it's
  *mismatched* to the bandwidth tasks.
- The relational insight (linking related memories) is the right tool for **contradiction / dedup** — keep
  it as a *light* add-on later, not the load-bearing mechanism.
- The diagnostic harness (band framing, binding gap, effrank/entropy canaries, per-param grad audit) carries
  forward unchanged to the next architecture.

---

## 7. Decision

**Move the primary line to a fast-weights / Hebbian memory** (the `bio-memory` branch). Rationale:
1. The graph's wall is fundamental *and* recurring (graph_v5→v9, the membership-only failures,
   write-grad collapse).
2. The real application — an **always-on implicit memory cache** between context and RAG for small/edge
   agents (gist + presence-pointer + procedural bias; no tool, always on) — **does not require** learned
   discrete graph construction.
3. The decisive axis for that application is **write/update quality** (stability–plasticity: integrate the
   new without clobbering the old, forget only what's safe). Fast-weights owns this natively (local
   non-clobbering writes + a forget gate; cf. Titans/DNC); the graph has no intrinsic update mechanism.

**Caveat on the evidence:** the clean discriminator (MAE) is *also the task the thesis is worst-suited to*.
The right next step is not "fix the graph" but "**measure the right axis**": a streaming write/update proxy
(retention-under-interference + selective in-place update + graceful forgetting), scored alongside
compression, comparing graph vs. fast-weights vs. a flat baseline. Prediction: the flat bank that *wins*
MAE *loses* there, and fast-weights separates from both.

See: `project_memory_layer_application.md`, `project_arch_pivot_recommendation.md`, `project_eval_protocol.md`,
`project_structure_vs_flat_proof.md`.
