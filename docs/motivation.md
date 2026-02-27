# Motivation & Design Philosophy — Neuromorphic Language Model

## Why This Exists

Modern LLMs are powerful but structurally limited:

1. **LLMs are static** — once deployed, weights don't change. Improving requires
   finetuning, retraining, or bolting on retrieval (RAG). This leaves a gap between
   short-term learning, long-term personalization, and robust agent memory.

2. **RAG is not learning** — retrieval can fetch text but doesn't integrate knowledge
   into the model's computations, adapt behavior continuously, or compress and
   generalize experience.

3. **Transformers are expensive for long context** — O(N²) attention makes lifelong
   learning impractical. If we want AI embedded everywhere and eventually in silicon,
   we need O(1) memory that scales as state, not as quadratic attention.

4. **Agents need to improve by doing** — a practical agent should remember what it has
   done, refine tool-use patterns, adapt to workflows, and get better over time without
   cloud retraining.

## What We're Building

A **brain-inspired sequence model** that decomposes memory into three distinct systems,
each with different persistence, update rules, and biological analogues. The fundamental
innovation: **separating WHAT to remember from HOW LONG to remember it**, enabling
lifelong learning without catastrophic forgetting.

The model maintains constant-memory state that evolves during use — no growing KV
cache, no retraining. It processes all tokens in a segment simultaneously via
parallel cortical columns, with memory updated between segments
(see `architecture_v4_iterative_memory_scan.md` for the full v4 design).

---

## Core Philosophy

### Three Memory Systems + Predictive Coding

| Memory System | Brain Analogy | Persistence | Update Mechanism | In v4 |
|---------------|---------------|-------------|------------------|-------|
| **Genetic** (slow weights) | DNA / evolutionary | Permanent after training | Backprop | Column FFN weights |
| **Procedural Memory (PM)** | Basal ganglia | Across documents (lifelong) | Hebbian eligibility + neuromodulated commits | Between segments |
| **Episodic Memory (EM)** | Hippocampus | Across documents (lifelong) | Novelty-based writes + neuromodulation | Between segments |

Plus **Predictive Coding (PCM)** — per-column cross-pass prediction + surprise signal
(inspired by cortical prediction error). Drives PM eligibility gating and EM novelty.
PCM measures how much each token's representation changes between refinement passes.

### Parallel Processing with Memory Between Segments

The key architectural insight (v4): **decouple token processing from memory
accumulation.** All tokens in a segment are processed simultaneously by cortical
columns (embarrassingly parallel). PM/EM update between segments using accumulated
outputs. No within-segment sequential scan — cross-segment context comes entirely
from PM/EM.

---

## Memory Systems in Detail

### Procedural Memory (PM) — Basal Ganglia

"Muscle memory" of cognition — learned behavioral patterns, skills, associations.

- **Read**: Holographic modulation: `y_d = x_d * [W @ x]_d` (quadratic in x). Each
  slot applies an input-dependent transformation, not fixed vector retrieval.
- **Eligibility traces**: Per-token accumulation gated by surprise (third factor).
  Pre-synaptic (k_cand) and Hebbian post-synaptic (v_cand = pre × post) signals,
  with slot-specific routing weights.
- **Commit**: Neuromodulator MLP at periodic boundaries produces (p_commit, lambda,
  g, slot_logits, tau). Continuous softmax slot selection + soft EMA update.
- **Sacred**: Differentiable eligibility traces (nn.Linear), surprise gating,
  neuromodulator trained by main loss, slot-specific routing, budget enforcement +
  unit normalization, lifelong persistence.
- **Negotiable**: Number of slots r, eligibility decay rho, routing temperature,
  EMA vs hard overwrite.
- **v4**: Slow-changing partition (updated at periodic boundaries). Column accumulates
  eligibility per-token; neuromodulator decides commits at boundaries. The commit
  itself is affine (EMA blend).

### Episodic Memory (EM) — Hippocampus

Specific events, surprising experiences, contextual episodes.

- **Retrieval**: Query from input features, score against M slots, top-k
  selection (k=4), cross-attention aggregation.
- **Candidate proposal**: Novelty = learned blend of surprise + dissimilarity from
  existing keys. Top-C highest-novelty candidates buffered.
- **Write**: Neuromodulator at boundaries produces (g_em, tau, write_weights, decay).
  Sequential EMA update. g_em near-zero = soft "don't write."
- **Sacred**: Novelty-based writes, learned novelty weighting (Phase B+), top-k
  retrieval (can't attend all M slots), query from input-side features, candidate
  validity masking, per-stream learned decay, neuromodulator trained by main loss.
- **Negotiable**: Capacity M, dimension D_em, cross-attention vs weighted average.
- **v4**: Slow-changing partition. Column does novelty scoring per-token; actual
  writes at periodic boundaries. Budget enforcement + decay prevent unbounded growth.
  Write is affine (EMA into selected slot).

### Predictive Coding Module (PCM) — Cortical Prediction Error

Per-column local prediction and surprise computation.

- **z_hat**: EMA prediction of next token's representation.
- **Surprise**: RMS-normalized prediction error `||delta||/sqrt(D_pc)` (~1.0 scale).
- **FFN gain**: `1 + 0.1 * tanh(W_gain(delta))` — bounded [0.9, 1.1] modulation.
- **v4**: Cross-pass prediction (not in scan carry). Each column predicts what the
  token's encoding will look like next pass. Surprise gates PM/EM updates.

---

## How Learning During Use Works

### Eligibility Traces: "What Could Be Learned"

Every token produces a candidate learning signal based on local activity (pre-synaptic
input, post-synaptic hidden state). This signal accumulates in eligibility traces —
a buffer of "potential updates" that does NOT automatically change memory. The third
factor (surprise) gates whether activity is eligible at all.

### Neuromodulators: "Should We Actually Store It?"

Neuromodulator MLPs output low-dimensional control signals that decide:
- **when** to commit (write event timing)
- **how strongly** to write (gating strength)
- **how much** to decay old memory
- **where** to store (slot selection via softmax)

Crucially, neuromodulators are **trained by the main CE loss** — gradient flows through:
commit → memory state → future reads → logits → loss. No RL, no counterfactual
rollouts, single optimizer. The model learns to write only when it improves future
predictions.

### Lifelong Learning: Memory Distillation

**Phase A**: All systems active, state resets at doc boundaries. Controllers learn
when to be selective.

**Phase B**: Lifelong mode — PM/EM persist across documents. Only eligibility traces
and recurrent state reset at boundaries. The distillation cycle:

1. New domain → surprise spikes
2. PM eligibility accumulates patterns (gated by surprise)
3. PM neuromodulator commits strong patterns to slots
4. EM writes surprising episodes
5. Over time, slow weights learn domain patterns → surprise drops
6. Result: slow weights = general knowledge; PM/EM = domain-specific skills/episodes

---

## What Makes This Fundamentally Different

### vs Transformers
- Transformers ask "how much can we cache?" — we ask "what should we remember?"
- Fixed memory footprint vs O(N) KV cache growth
- Functional memory separation vs single KV cache
- Online PM/EM updates vs retraining required

### vs SSMs (Mamba, RWKV)
- SSMs use sequential scan with flat state; we use parallel processing + structured memory
- SSMs compress everything into flat state; we separate memory by function + timescale
- No explicit lifelong learning mechanism in SSMs
- No neuromodulation, no eligibility traces, no surprise gating in SSMs

### vs RAG
- Memory is inside the model, not external
- Updates are automatic (neuromodulated), not manually triggered
- Knowledge integrates into computation, not just retrieved text
- Compression and generalization, not verbatim storage

---

## Non-Negotiable (Sacred) Design Principles

1. Three memory systems with distinct timescales (genetic/PM/EM) + PCM
2. Differentiable eligibility traces (nn.Linear projections trained by backprop)
3. Neuromodulators trained by main loss (no RL, single optimizer, single loss)
4. Surprise gating on eligibility (third factor)
5. Novelty-based EM writes (surprise + dissimilarity)
6. Budget enforcement + unit normalization (prevents drift)
7. Lifelong persistence with soft resets
8. Fixed memory capacity O(1) per token at inference
9. Per-stream state isolation (no cross-stream mixing)
10. Parallel token processing (columns) decoupled from memory accumulation (between segments)

## Negotiable — Implementation Details

- Architecture dimensions (D, B, r, M, D_em, D_h)
- Number of iterative passes R
- Segment length N (= PM/EM boundary interval)
- Hyperparameters (decay rates, temperatures, budgets)
- Shared vs per-pass column weights
- Token subsampling schedule across passes

---

## What Success Looks Like

### Core: The model learns language normally
Validation perplexity improves during training. Baseline competence established.

### Adaptation: The model gets better as you use it
- PM: repeated patterns → fewer mistakes within-session
- EM: facts seen once → retrieved accurately later
- Controllers: learned write selectivity outperforms heuristics

### Stability: Core competence is preserved
- Long adaptation runs don't destabilize base performance
- Commit/write rates stay sparse and bounded

### Hardware story: Constant memory, parallel compute
- O(1) inference memory (no growing KV cache)
- O(R) serial depth (fully parallel within each pass, no scan)
- Slot-based memory → hardware-friendly dot products + weighted sums
