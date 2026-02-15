# Future Directions: Neuromorphic AI Brain-Core

> **Note:** Many recommendations from the original v1.7 roadmap have been implemented in **v2.0**. This document has been updated to reflect what v2 achieves and what remains for future work.

This document summarizes the architectural conclusions, insights, and forward-looking directions. It is intended as a **strategic roadmap**, not an implementation spec. The goal is to clarify what the model already achieves (v2), what it fundamentally cannot do yet, and how to evolve it further.

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
* **Sequence-parallel pretraining** via scan is available and optimized with `torch.compile` (~10K tok/s on RTX 4090), but further optimization is possible (custom CUDA kernels, multi-GPU).

### What v2 Addresses (Previously Missing)

* ✅ **Declarative / episodic memory** — added via per-block EM (vector store with top-k retrieval)
* ✅ **Scan-friendly recurrence** — GRU replaced with affine form (h_t = a_t ⊙ h_{t-1} + b_t)
* ✅ **Working memory** — sliding-window attention (W=256 tokens)
* ✅ **Lifelong learning** — Phase C soft reset: PM/EM persist across doc boundaries with natural decay + budget enforcement

---

## 2. Memory Taxonomy: What v2 Provides

### Working Memory (WM) — v2 ✅

Sliding-window attention over last W=256 tokens:

* Short-term precision: copying, binding, recent tokens
* No post-training evolution (standard attention)
* One shared WM across the model

### Procedural / Implicit Memory (PM) — Strong

Low-rank adapters (K_pm/V_pm/a_pm) per layer per block (B×L=32 instances):

* Persistent, abstract memory in representation space
* Associative recall by similarity
* Behavioral biasing rather than explicit recall
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
* ✅ **Sequential training bottleneck** — scan-friendly affine recurrence + torch.compile (~10K tok/s on 4090)
* ✅ **Exact copying and pointing** — WM sliding-window attention
* ✅ **Episodic recall** — EM with top-k retrieval
* ✅ **Cross-document persistence** — Phase C lifelong mode with soft reset

### Remaining Gaps

* Consolidation (PM/EM → slow weights)
* Custom CUDA scan kernels (potential further 1.5-2× over torch.compile)
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
* ~40-50% of Mamba's throughput at equivalent parameter count (~10K tok/s on 4090 with `--compile`)
* RNN-like inference behavior

### Plasticity Boundaries for Scan-Friendliness

v2 adds plasticity span boundaries (every P=64 tokens):

* PM/EM are read-only within spans → core loop is scan-friendly
* PM/EM writes happen only at span boundaries
* Eligibility updates remain differentiable within TBPTT
* K+V eligibility scans are fused into a single double-width scan for efficiency

**Status:** Implemented and optimized. `torch.compile(fullgraph=True)` fuses scan loops, gate computations, and elementwise ops into optimized CUDA kernels. PM, EM, and controllers remain intact around the new scan-friendly core.

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
| Working Memory (WM) | Sliding-window attention | W=256 tokens | None |
| Procedural Memory (PM) | Low-rank adapters | r=8 × B×L | `PMNeuromodulator` per instance |
| Episodic Memory (EM) | Vector store | M=256 × B | `EMNeuromodulator` per instance |

---

## 6. Attention Is Not the Enemy — ✅ Addressed in v2

The goal was not to avoid attention entirely, but to avoid **unbounded token replay**.

### v2 Implements Bounded Attention

* ✅ **Sliding-window attention (WM)**: W=256 tokens for short-term precision
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
2. ✅ **Procedural Memory (PM)** — low-rank adapters, B×L instances with `PMNeuromodulator`
3. ✅ **Episodic Memory (EM)** — vector store, B instances with `EMNeuromodulator`
4. ✅ **Working Memory (WM)** — sliding-window attention, W=256 tokens
5. ✅ **Lifelong learning (Phase C)** — soft reset at doc boundaries, PM/EM persist, eval framework
6. ✅ **Training speed optimization** — torch.compile + fused scans + vectorized ops (~10K tok/s on 4090)
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
* ✅ Training speed optimization (torch.compile, fused scans, ~10K tok/s)
* ⏳ Consolidation remains for future work

The architecture remains fundamentally different from transformers:

> A scalable, stable, adaptive agent core where memory lives inside computation rather than inside prompts.

The next major milestones are:
- Demonstrating that PM/EM provide measurable advantages on memory benchmarks and online adaptation tasks
- Phase C evaluation: domain adaptation speed, drift monitoring, cross-document recall
- Consolidation: transferring fast memory (PM/EM) to slow weights for permanent knowledge acquisition
