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

The model processes tokens causally through dense scan layers (nn.Linear projections, full
feature mixing) with structured memory operations between segments (slow hippocampal
stream). Predictive coding and memory projections remain grouped (per-feature-group
surprise via free `.view()` from dense tensors). See
`architecture_v4_iterative_memory_scan.md` for the full v5.1 design.

---

## Core Philosophy

### Three Memory Systems + Predictive Coding

| Memory System | Brain Analogy | Persistence | Update Mechanism | In v5.1 |
|---------------|---------------|-------------|------------------|---------|
| **Genetic** (slow weights) | DNA / evolutionary | Permanent after training | Backprop | Scan params, projection weights |
| **Procedural Memory (PM)** | Basal ganglia | Across documents (lifelong) | Surprise-gated bias shift | Between segments |
| **Episodic Memory (EM)** | Hippocampus | Across documents (lifelong) | Novelty-based primitive decomposition + neuromodulation | Between segments |

Plus **Predictive Coding (PCM)** — grouped prediction of next token's encoding.
Vector surprise (per-feature-group prediction error) drives PM eligibility gating and EM
novelty scoring. Each feature group predicts what the next token's features will look
like; the per-dimension error tells the model which features it failed to anticipate.
PCM operates on a grouped view [BS,N,C,D_col] of the dense scan output (free `.view()`).

### Two-Stream Processing: Fast Scan + Slow Memory

The key architectural insight (v5.1): **separate fast causal computation from slow
memory operations.**

- **Fast stream (Stages 1 & 3)**: Dense affine scan recurrence — causal, parallelizable.
  nn.Linear projections with full feature mixing process tokens. Grouped PCM computes
  predictions, grouped W_seed_w prepares trail seeds and write candidates. All linear
  operations that compose into a scan.
- **Slow stream (Stage 2)**: Non-affine memory operations — batched across all
  positions. Write-before-read with causal write buffers: per-token deltas are
  prefix-summed, giving each position causal access to within-segment writes.
  PM gain modulation with causal bias, EM trail from frozen primitives, segment-end
  commit. The bio-inspired logic that can't be linearized runs here, once per segment.

This mirrors the brain: neocortex processes quickly and feedforward; hippocampal
consolidation is slower, periodic, and handles memory formation.

---

## Memory Systems in Detail

### Procedural Memory (PM) — Basal Ganglia

"Muscle memory" of cognition — automatic response biases, habitual processing.

- **Read**: Element-wise delta modulation: `delta = H ⊙ (pm_bias + cum_pm)`. Coarse,
  fast, no matmuls — each feature has a frozen bias plus causal write buffer.
  Baseline H is added once in Stage 3 integration (not per-bank).
- **Update**: Two paths. Fast: per-token delta `δ_pm = lr_pm · surprise`, prefix-summed
  to `cum_pm` for within-segment causal feedback. Slow: `pm_bias += Σ_n δ_pm` at
  segment end. Only adapts on unpredicted features (vector surprise per-feature).
- **Column-local**: Column c only addresses dims c*D_col:(c+1)*D_col of its bank.
- **State**: pm_bias [BS, B, D] — one vector per bank. Not parameters — evolves
  at inference for lifelong adaptation.
- **Sacred**: Surprise-gated updates, lifelong persistence, per-feature bias.
- **Negotiable**: Learning rate, decay rate toward neutral.

### Episodic Memory (EM) — Hippocampus

Dictionary of primitive patterns — atomic building blocks of concepts. Complex
ideas emerge from composing multiple primitives via trail-based navigation.

- **Read (trail)**: Seed from scan navigates primitive space through iterative
  refinement (2 steps). At each step: score ALL primitives via softmax (no top-k),
  compose weighted sum, gate, move seed. Different seed → different trail →
  different composition. M primitives + continuous seeds = infinite readouts.
- **Novelty**: Reconstruction error — can existing primitives explain this input?
  Blended with surprise magnitude.
- **Write (two paths)**: Fast: `δ_em = novelty · w_cand`, prefix-summed to `cum_em`
  for within-segment feedback (additive signal to Stage 3). Slow: decompose across
  primitives via soft routing at segment end, neuromodulator gates EMA write strength.
  Fully differentiable — no hard selection anywhere.
- **Column-local**: Column c only addresses its D_col slice of the bank's EM.
- **State**: em_K, em_V [BS, B, M, D] — primitives are state, not parameters.
  Evolve at inference for lifelong learning.
- **Sacred**: Trail-based composition, soft activation (no top-k), novelty-based
  writes, neuromodulator trained by main loss, primitive decomposition.
- **Negotiable**: Number of primitives M, trail steps, temperature τ, noise σ.

### Predictive Coding Module (PCM) — Cortical Prediction Error

Per-column within-scan prediction and surprise computation.

- **z_hat**: Prediction of next token's encoding (linear projection of scan state).
- **Vector surprise**: Elementwise prediction error `delta = z_hat_{t-1} - z_t`,
  D_col dimensions. Each feature dimension carries independent surprise.
- **PCM gain**: `1 + 0.1 * tanh(W_gain(delta))` — bounded [0.9, 1.1] modulation.
- **v5.1**: Grouped prediction (token t predicts t+1) on `.view(BS,N,C,D_col)`, not
  cross-pass. Surprise gates memory updates per-feature-group.

---

## How Learning During Use Works

### PM Update: "Adapt My Biases"

Surprise from PCM drives per-feature bias shifts via **two write paths**:
- **Fast (within segment)**: Each token computes `δ_pm = lr_pm · surprise`.
  Prefix sum gives `cum_pm` — causal accumulation. PM read uses
  `H ⊙ (pm_bias + cum_pm)`, so later tokens benefit from earlier surprise.
- **Slow (segment end)**: `pm_bias += Σ_n δ_pm`. Bias persists to next segment.
  Automatic, habitual adaptation with no explicit routing.

### EM Write: "Two Paths — Fast Buffer + Slow Decomposition"

**Fast path**: Write candidates from the scan are gated by novelty and
prefix-summed: `cum_em = cumsum(novelty · w_cand)`. This raw, unstructured
signal goes to Stage 3 alongside the trail read, providing within-segment
feedback and cross-column mixing. Same-segment gradient flows through.

**Slow path**: At segment end, write candidates are decomposed across existing
primitives via soft routing. The neuromodulator MLP controls write strength
(`g_em`). Crucially, neuromodulators are **trained by the main CE loss** —
gradient flows through: write → primitive update → future trail reads → logits
→ loss. No RL, no counterfactual rollouts, single optimizer.

### Lifelong Learning: Memory Distillation

**Phase A**: All systems active, state resets at doc boundaries. Controllers learn
when to be selective.

**Phase B**: Lifelong mode — PM/EM persist across documents. Only scan hidden states
reset at boundaries. The distillation cycle:

1. New domain → surprise spikes (PCM prediction errors)
2. PM bias shifts to compensate for persistent surprises
3. EM primitives update: new concepts decomposed and stored
4. Over time, slow weights learn domain patterns → surprise drops
5. Result: slow weights = general knowledge; PM bias = domain-specific habits;
   EM primitives = domain-specific concepts

---

## What Makes This Fundamentally Different

### vs Transformers
- Transformers ask "how much can we cache?" — we ask "what should we remember?"
- Fixed memory footprint vs O(N) KV cache growth
- Functional memory separation vs single KV cache
- Online PM/EM updates vs retraining required

### vs SSMs (Mamba, RWKV)
- SSMs use flat state; we use structured memory with distinct systems
- SSMs compress everything into hidden state; we separate memory by function + timescale
- No explicit lifelong learning mechanism in SSMs
- No neuromodulation, no eligibility traces, no surprise gating in SSMs
- We use simple linear recurrence (simpler than SSM) — bio-inspired memory is the key addition

### vs RAG
- Memory is inside the model, not external
- Updates are automatic (neuromodulated), not manually triggered
- Knowledge integrates into computation, not just retrieved text
- Compression and generalization, not verbatim storage

---

## Non-Negotiable (Sacred) Design Principles

1. Three memory systems with distinct timescales (genetic/PM/EM) + PCM
2. EM as primitive dictionary with trail-based compositional read
3. PM as coarse bias vector with surprise-gated updates
4. Neuromodulators trained by main loss (no RL, single optimizer, single loss)
5. Vector surprise (per-feature prediction error) drives memory updates
6. Novelty-based EM writes (surprise + reconstruction error)
7. Budget enforcement + unit normalization (prevents drift)
8. Lifelong persistence — memory is state, not parameters
9. Fixed memory capacity O(1) per token at inference
10. Causal token processing (scan recurrence) with memory between segments
11. Causal write buffers (prefix sums) for within-segment memory feedback

## Negotiable — Implementation Details

- Architecture dimensions (D, B, C, M, D_embed)
- Segment length N (= memory update interval)
- Number of scan layers per stage
- Hyperparameters (decay rates, temperatures, budgets)

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

### Hardware story: Constant memory, efficient compute
- O(1) inference memory (no growing KV cache)
- O(log N) serial depth (parallel-scan, no quadratic attention)
- Scan kernels for GPU throughput + batched memory ops
