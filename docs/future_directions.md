# Future Directions: Neuromorphic AI Brain-Core

> **Note (2026-03-02):** This document was written for the **v2 architecture** (scan-based recurrence + WM/PM/EM). The current architecture is **v5.1** (dense scan + grouped PCM with causal write buffers). Key differences from v2:
> - WM (working memory) is removed — scan recurrence replaces it
> - PM is now a simple bias vector with gain modulation (not holographic slots)
> - EM uses trail-based composition (not top-k retrieval)
> - PCM uses grouped vector surprise (not cross-pass scalar surprise)
> - Scan layers are dense (nn.Linear); PCM/W_seed_w remain grouped (GroupedLinear)
> - No separate FFN — scan projections serve this role
> - No R-loop — single three-stage cycle per segment
>
> See `architecture_v4_iterative_memory_scan.md` for the current v5.1 design. The high-level principles (bounded state, online adaptation, selective plasticity) carry forward. Architecture-specific details (WM, tier sizes, GLA) are outdated.

This document summarizes the architectural conclusions, insights, and forward-looking directions. It is intended as a **strategic roadmap**, not an implementation spec.

---

## 1. High-Level Positioning

### What This Architecture *Is*

The model is best understood as a **stateful, adaptive agent core** rather than a drop-in replacement for transformer LLMs.

Architecturally, it emphasizes:

* **Bounded internal state** (fixed memory footprint over time)
* **Online adaptation** (experience changes computation, not just retrieval)
* **Persistent memory** that lives inside the model’s dynamics
* **Selective plasticity** (learn when to learn)

This aligns more closely with a *brain-like system* than with a prompt-centric language model.

### What This Architecture Is *Not* (Remaining Gaps)

* It does not yet include intrinsic **consolidation** of experience into slow weights.
* **Sequence-parallel pretraining** via scan is available and optimized with `torch.compile` (~24K tok/s on RTX 4090 for Tier A Wide), but further optimization is possible (FLA library kernels, custom CUDA kernels, multi-GPU).

### What v2 Addresses (Previously Missing)

* ✅ **Declarative / episodic memory** — added via per-block EM (vector store with top-k retrieval)
* ✅ **Scan-friendly recurrence** — GRU replaced with affine form (h_t = a_t ⊙ h_{t-1} + b_t)
* ✅ **Working memory** — Gated Linear Attention (GLA) recurrence
* ✅ **Lifelong learning** — Phase C soft reset: PM/EM persist across doc boundaries with natural decay + budget enforcement

---

## 2. Memory Taxonomy: What v2 Provides

### Working Memory (WM) — v2 ✅

Gated Linear Attention (GLA) with recurrent state matrix:

* Short-term precision: copying, binding, recent tokens
* No post-training evolution (recurrent cache with learned decay gates)
* One shared WM across the model

### Procedural / Implicit Memory (PM) — Strong

Holographic modulation slots (K_pm/V_pm/a_pm) per layer per block (B×L instances):

* Persistent, abstract memory as input-dependent transformations
* Holographic read: input flows through stored modulation patterns (quadratic in x)
* Hebbian eligibility: stores input-output interaction (x * h), not just output
* Each instance controlled by its own `PMNeuromodulator`

Psychologically, this maps to:

* Procedural memory / skills
* Habits / priors
* "Subconscious knowing"

### Declarative / Episodic Memory (EM) — v2 ✅

Per-block vector store (B instances), each with:

* Itemized facts or events (M=256 items per block)
* Top-k retrieval (k_ret=4 latent tokens) + cross-attention aggregation
* Each instance controlled by its own `EMNeuromodulator`

This addresses the gap from v1.7: the model now has both *implicit* and *declarative* memory.

---

## 3. Why Transformers Beat RNNs — and What v2 Addresses

### Historical Reasons Transformers Won

1. Training-time parallelism over tokens
2. Stable scaling with data and compute
3. Token-level retrieval via attention
4. Strong in-context learning

### What v2 Successfully Fixes

* ✅ Long-lived context without growing state (PM + EM)
* ✅ Online learning without finetuning (PM + EM writes)
* ✅ Memory as computation, not text retrieval
* ✅ Stability via gated plasticity and budgets
* ✅ **Sequential training bottleneck** — scan-friendly affine recurrence + block-level torch.compile (~24K tok/s on 4090 for Tier A Wide)
* ✅ **Exact copying and pointing** — WM Gated Linear Attention
* ✅ **Episodic recall** — EM with top-k retrieval
* ✅ **Cross-document persistence** — Phase C lifelong mode with soft reset

### Remaining Gaps

* Consolidation (PM/EM → slow weights)
* FLA library GLA kernels or custom CUDA kernels (potential further 1.5-2× over torch.compile)
* Multi-GPU data parallelism

---

## 4. Scan-Friendly Recurrence — ✅ Implemented in v2

### Key Insight (from v1.7 roadmap)

The v1.7 limitations were **not caused by fast memory** — they were caused by the **GRU-style recurrent core**.

### v2 Solution

v2 replaces GRU with an affine recurrence per block:

```
h_t = a_t ⊙ h_{t-1} + b_t
```

where:

* `a_t`, `b_t` depend only on input features (not on `h_{t-1}`)
* the update is affine in `h_{t-1}`
* per-step transforms can be composed associatively

This enables:

* Parallel training via prefix scans within plasticity spans
* Reasonable throughput (~24K tok/s on 4090 with `--compile`; slower than Mamba-130M ~52K tok/s and Pythia-160M ~116K tok/s due to memory system overhead)
* RNN-like inference behavior

### Plasticity Boundaries for Scan-Friendliness

v2 adds plasticity span boundaries (every P=32 tokens):

* PM/EM are read-only within spans → core loop is scan-friendly
* PM/EM writes happen only at span boundaries
* Eligibility updates remain differentiable within TBPTT
* K+V eligibility scans are fused into a single double-width scan for efficiency

**Status:** Implemented and optimized. Block-level `torch.compile(mode="default")` fuses scan loops, gate computations, and elementwise ops into optimized CUDA kernels. PM eligibility updates are compiled separately with `fullgraph=True`. PM, EM, and controllers remain intact around the new scan-friendly core. `max-autotune` was benchmarked but is 3.5% slower at current scale — the default mode's kernels are already near-optimal.

---

## 5. Declarative Memory — ✅ Implemented in v2

### v2 Episodic Memory (EM)

v2 implements the recommended episodic memory:

* Item-based vector store (K_e, V_e, S_e)
* Supports top-k retrieval (k_ret=4 latent tokens)
* Capacity M=256 items per block (B blocks, e.g. B=2 → 512 total items)
* Per-block `EMNeuromodulator` for write decisions

This enables:

* ✅ Explicit fact recall
* ✅ Multi-entity reasoning
* ✅ Variable binding (via cross-attention aggregation)

### Clean Memory Stack (v2)

| Memory | Type | Capacity | Controller |
|--------|------|----------|------------|
| Working Memory (WM) | Gated Linear Attention (GLA) | O(1) state matrix per head | None |
| Procedural Memory (PM) | Holographic modulation slots | r=8 × B×L | `PMNeuromodulator` per instance |
| Episodic Memory (EM) | Vector store | M=256 × B | `EMNeuromodulator` per instance |

---

## 6. Attention Is Not the Enemy — ✅ Addressed in v2

The goal was not to avoid attention entirely, but to avoid **unbounded token replay**.

### v2 Implements Bounded Attention

* ✅ **Gated Linear Attention (WM)**: recurrent state for short-term precision
* ✅ **Episodic cross-attention (EM)**: k_ret=4 latent tokens aggregated via 1-query cross-attention

These provide transformer advantages (copying, binding, retrieval) without unbounded state.

---

## 7. Consolidation: Turning Experience Into Knowledge

### Current State

* The model can learn online
* That learning lives in fast state (K/V/a)
* It does not automatically become durable skill

### Needed for “Subconscious Knowledge”

Some form of **consolidation**, e.g.:

* Periodic distillation (“sleep”) from plastic model into slow weights
* Replay from episodic memory
* Extremely conservative online slow-weight updates

Without consolidation, lifelong learning becomes adaptation without long-term improvement.

---

## 8. Stability at Lifelong Timescales

You already have strong foundations:

* Budgets
* Decay
* Weakness bias
* Commit gating

Further improvements may include:

* Multi-timescale commit budgets
* Write-ahead or shadow memory before activation
* Explicit plasticity-off regression tests during training

---

## 9. v2 Implementation Status

The v1.7 roadmap proposed these upgrades. v2 status:

1. ✅ **Scan-friendly recurrent core** — affine recurrence (h_t = a_t ⊙ h_{t-1} + b_t)
2. ✅ **Procedural Memory (PM)** — holographic modulation slots, B×L instances with `PMNeuromodulator`
3. ✅ **Episodic Memory (EM)** — vector store, B instances with `EMNeuromodulator`
4. ✅ **Working Memory (WM)** — Gated Linear Attention (GLA) recurrence
5. ✅ **Lifelong learning (Phase C)** — soft reset at doc boundaries, PM/EM persist, eval framework
6. ✅ **Training speed optimization** — block-level torch.compile + GLA WM + fused scans (~24K tok/s on 4090)
7. ⏳ **Consolidation mechanism** — not yet implemented (future work)

v2 remains fundamentally different from transformers:

* Memory is internal, bounded, and adaptive
* Attention is local and auxiliary, not the core

---

## 10. Final Takeaway

v2 implements the key architectural recommendations from the v1.7 roadmap:

* ✅ Scan-friendly recurrence for parallel training
* ✅ Episodic memory for declarative recall
* ✅ Working memory for short-term precision
* ✅ Lifelong learning with persistent cross-document memory (Phase C)
* ✅ Training speed optimization (block-level torch.compile, GLA WM, ~24K tok/s)
* ⏳ Consolidation remains for future work

The architecture remains fundamentally different from transformers:

> A scalable, stable, adaptive agent core where memory lives inside computation rather than inside prompts.

The next major milestones are:
- Demonstrating that PM/EM provide measurable advantages on memory benchmarks and online adaptation tasks
- Phase C evaluation: domain adaptation speed, drift monitoring, cross-document recall
- Consolidation: transferring fast memory (PM/EM) to slow weights for permanent knowledge acquisition
