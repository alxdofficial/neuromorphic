# Design Motivations: Neuromorphic Language Model

**Date:** 2026-02-16

## Core Philosophy

This is a **biologically-inspired language model** that decomposes memory into four distinct systems (genetic/slow weights, working memory, procedural memory, episodic memory), each with different persistence mechanisms and update rules. The fundamental innovation is **separating WHAT to remember from HOW LONG to remember it**, enabling lifelong learning without catastrophic forgetting.

---

## Memory System Decomposition

| Memory System | Brain Analogy | Persistence | Update Mechanism | Capacity |
|---------------|---------------|-------------|------------------|----------|
| **Genetic** (slow weights) | DNA / evolutionary | Permanent after training | Backprop | ~85M params |
| **Working Memory** | Prefrontal cortex | Recurrent state | Gated Linear Attention (GLA) | O(1) state matrix per head |
| **Procedural Memory** | Cerebellum / basal ganglia | Across documents (lifelong) | Eligibility traces + neuromodulated commits | r=8 slots/layer, B*L instances |
| **Episodic Memory** | Hippocampus | Across documents (lifelong) | Novelty-based writes + neuromodulation | M=256 slots/block, B instances |

---

## Working Memory (WM)

### Brain Analogy
Prefrontal cortex — mental "scratchpad" for active maintenance of recent information.

### Core Mechanism
Gated Linear Attention (GLA). Each token updates a recurrent state matrix `S` via learned decay gates and outer-product writes. The state `S: [H, K, V]` per head compresses recent context into a fixed-size matrix. Decay gates (`logsigmoid(W_g(x)) / 16`) provide learned recency bias — replacing ALiBi's fixed slopes with data-dependent forgetting.

### Sacred Properties
- **Fixed-size state**: O(1) memory per stream (H × head_dim × head_dim matrix)
- **Non-plastic nature**: WM is pure ephemeral cache, not a learning system
- **Gradient flow within TBPTT chunks**: Recurrence preserves gradients through state updates
- **Doc-boundary reset**: Zero the state matrix for masked streams

### Negotiable
- State dimensions (D_wm, n_heads_wm, gate_low_rank)
- **Implementation**: Softmax ring-buffer attention available as fallback (`wm_type="softmax"`)
- Could switch to chunk-parallel GLA (fla library) for longer spans

---

## Procedural Memory (PM)

### Brain Analogy
Cerebellum / basal ganglia — motor skills, cognitive procedures, learned associations. "Muscle memory" of cognition.

### Core Mechanism: Neo-Hebbian Three-Factor Learning

**Read (every token):** Linear key-value lookup weighted by slot strengths.
```
y_pm = (pm_a * (pm_K @ x_q)) @ pm_V
```

**Eligibility traces (every token, differentiable):**
```
k_cand = normalize(W_k_pre(x))          # pre-synaptic
v_cand = W_v_post(h)                    # post-synaptic
gate = (surprise / 5.0).clamp(0, 1)     # third factor
route_w = softmax(pm_K @ k_cand / tau)  # slot-specific routing
elig_K = rho * elig_K + gate * route_w * k_cand
```

**Commit (span boundary, every P=64 tokens):**
Neuromodulator MLP produces (p_commit, lambda, g, slot_logits, tau). Continuous softmax slot selection + EMA update.

### Sacred Properties
- **Differentiable eligibility traces**: W_k_pre, W_v_post are nn.Linear layers trained by backprop
- **Surprise gating**: Prevents trace saturation
- **Neuromodulator trained by main loss**: No RL, no counterfactual rollouts
- **Slot-specific routing**: Weighted by affinity, not uniform broadcast
- **Budget enforcement + unit normalization**: Prevents unbounded growth/drift
- **Lifelong persistence (Phase C)**: Committed state persists; only eligibility resets

### Negotiable
- Number of slots r, eligibility decay rho, base decay rate
- Surprise scaling, routing temperature
- EMA vs hard overwrite for commits
- Readout FFN (optional)

---

## Episodic Memory (EM)

### Brain Analogy
Hippocampus — stores specific autobiographical events, surprising experiences, contextual episodes.

### Core Mechanism

**Retrieval (every token):** Query from input + WM, score against all M slots, top-k selection, cross-attention aggregation over retrieved K_top/V_top.

**Candidate proposal (every token, buffered):** Novelty = learned blend of surprise + dissimilarity from existing keys.

**Write (span boundary):** Neuromodulator produces (g_em, tau, ww, decay). Sequential EMA update for C_em=8 candidates. No binary gate — g_em near-zero = soft "don't write".

### Sacred Properties
- **Novelty-based writes**: Must blend surprise + dissimilarity
- **Learned novelty weighting (Phase B+)**: W_nov trained by backprop
- **Top-k retrieval**: Can't attend to all M slots (cost)
- **Query from input-side features**: x_proj + y_wm, NOT recurrent state h
- **Candidate validity masking**: Filter pre-reset candidates
- **Per-stream learned decay**: Each stream adapts its forgetting timescale
- **Neuromodulator trained by main loss**
- **Full persistence (Phase C)**: All EM state persists; decay + budget handle staleness

### Negotiable
- Capacity M, dimension D_em, top-k count k_ret
- Cross-attention vs weighted average
- Number of candidates per span
- Readout FFN (optional)

---

## Persistent Streams

BS=32 is **architectural, not just a memory constraint**. Each batch element is a continuous stream of documents carrying state across TBPTT chunks:

```
Stream 0: [doc1...] <eot> [doc2...] <eot> [doc3...]
Stream 1: [docA...] <eot> [docB...] <eot> ...
```

All state tensors are per-stream: Layer.h, pm_K/pm_V/pm_a, elig_K/elig_V, em_K/em_V/em_S, gla_state (or wm_K/wm_V if softmax WM). No cross-stream mixing. In Phase C, PM/EM accumulate knowledge across documents within the same stream.

**Sacred**: The persistent stream abstraction is fundamental to lifelong learning.

---

## Lifelong Learning: Memory Distillation

### Phase A-B: Document Isolation (Train Controllers)
- All state reset at doc boundaries
- Neuromodulators learn when to be selective

### Phase C: Lifelong Mode (Soft Reset)
- Reset: h (recurrent), eligibility traces (stale)
- Persist: pm_K/pm_V/pm_a (consolidated), em_K/em_V/em_S (episodes)

### Distillation Process
1. New domain encountered -> surprise spikes
2. PM eligibility traces accumulate domain patterns (gated by surprise)
3. PM neuromodulator commits strong patterns
4. EM writes surprising events
5. Over time, slow weights learn domain patterns -> surprise drops
6. Slow weights = general knowledge; PM/EM = domain-specific skills/episodes

**Sacred**: The interplay between fast (PM/EM) and slow (parameters) learning is fundamental.

---

## What Makes This Fundamentally Different

### vs Transformers
- Transformers ask "how much can we cache?" — Neuromorphic asks "what should we remember and for how long?"
- Fixed memory footprint vs O(n) KV cache growth
- Functional memory separation vs single KV cache
- Online PM/EM updates vs retraining required

### vs SSMs (Mamba, RWKV)
- Same core recurrence (affine scan)
- SSMs compress everything into h; neuromorphic separates memory by function and timescale
- No explicit lifelong learning mechanism in SSMs
- No neuromodulation in SSMs

---

## Non-Negotiable (Sacred) — Core Design Philosophy

1. Four memory systems with distinct timescales
2. Persistent streams with state continuity
3. Differentiable eligibility traces (nn.Linear projections)
4. Neuromodulators trained by main loss backprop (no RL)
5. Surprise gating on eligibility
6. Novelty-based EM writes
7. Budget enforcement + unit normalization
8. Lifelong persistence with soft resets
9. Scan-friendly recurrence (gates independent of previous h)
10. Fixed memory capacity O(1) per token
11. Per-stream state isolation
12. Single optimizer, single loss

## Negotiable — Implementation Details

- Architecture dimensions (D, L, B, r, M, W, D_wm, D_em)
- WM attention mechanism (softmax, linear attention, GLA, DeltaNet)
- Scan implementation (sequential loop, parallel prefix, accelerated-scan)
- Span length P (64, 128, 256)
- Hyperparameters (decay rates, scaling factors, budgets, temperatures)
- Optional components (readout FFNs, spatial decoder)
- Training schedules, phase curriculum
