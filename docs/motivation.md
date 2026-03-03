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
`architecture_v4_iterative_memory_scan.md` for the full v6 design.

---

## Core Philosophy

### Three Memory Systems + Predictive Coding

| Memory System | Brain Analogy | Persistence | Update Mechanism | In v6 |
|---------------|---------------|-------------|------------------|-------|
| **Genetic** (slow weights) | DNA / evolutionary | Permanent after training | Backprop | Scan params, projection weights |
| **Procedural Memory (PM)** | Basal ganglia | Across documents (lifelong) | Hebbian fast-weight learning | Between segments |
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

"Muscle memory" of cognition — habitual associative transformations, automatic pattern recall.

- **Read**: `pre = proj_in(H)` projects to D_pm-dim pre-synaptic space [BS,N,D_pm].
  Apply bank-summed fast-weight: `post = pre @ W_sum.T` where `W_sum = Σ_b W_b [BS,D_pm,D_pm]`.
  Project back via `proj_out` to [BS,N,D]. A full learned linear transform — not element-wise bias.
- **Write (Hebbian)**: At segment end, compute eligibility G = (1/N)·Σ_t σ(‖surp_t‖)·pre_t⊗pre_tᵀ
  (surprise-gated pre-synaptic autocorrelation). Commit: `W_b ← W_b @ (decay·I + β_b·G)`.
  One batched right-multiply per bank. High-surprise tokens drive larger weight changes.
- **Bank plasticity**: Each bank has a learned scalar β_b (via softplus). Banks specialize:
  fast β = recent-context habits; slow β = deep long-term associations.
- **State**: W_pm [BS, B, D_pm, D_pm] — fast-weight matrices per bank. Not parameters —
  evolves at inference for lifelong adaptation. ~512KB total (BS=16, B=4, D_pm=64).
- **Sacred**: Surprise-gated writes, lifelong persistence, Hebbian rule.
- **Negotiable**: D_pm, bank count, decay rate, Frobenius budget.

### Episodic Memory (EM) — Hippocampus

Dictionary of primitive patterns — atomic building blocks of concepts. Complex
ideas emerge from composing multiple primitives via trail-based navigation.

- **Read (trail)**: Seed from scan navigates primitive space through iterative
  refinement (3 steps). At each step: score ALL primitives via softmax (no top-k),
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

### PM Write: "Reinforce Associations"

Surprise from PCM drives Hebbian weight updates at segment boundaries (single path):
- **Eligibility accumulation**: Each token contributes surprise magnitude σ(‖surp_t‖)
  and pre-synaptic activity `pre_t = proj_in(H_t)` to an eligibility matrix
  G = (1/N)·Σ_t σ(‖surp_t‖)·pre_t⊗pre_tᵀ across the segment.
- **Commit (segment end)**: `W_b ← W_b @ (decay·I + β_b·G)` — one right-multiply
  per bank. Features that co-activated under surprise have their associations reinforced.
  Frobenius norm budget prevents unbounded growth across lifelong use.
- **No within-segment fast path**: Unlike EM, PM writes take effect from the next segment.
  This matches the slower timescale of procedural learning — habits form across many
  segments, not within a single one.

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

## Genuine Novelties — What No Other Architecture Does

These are the ideas that, to our knowledge, do not exist together in any published
model. Each is individually motivated by neuroscience; their combination is the
architecture's core contribution.

### 1. Trail-Based Compositional Memory Read
EM stores M atomic primitives. Reading is **not** top-k retrieval — it's an iterative
navigation: a seed vector takes multiple steps through primitive space, composing a
weighted mixture at each step via softmax attention + gated movement. Different seeds
produce different compositions from the same primitives. This gives combinatorial
capacity (M primitives → infinite readouts) from finite memory.

**Why novel**: Standard memory-augmented networks (NTM, DNC, MemoryNet) use single-step
attention or top-k retrieval. Trail-based composition is closer to how hippocampal
replay chains memories together.

### 2. Vector Surprise as the Universal Learning Signal
PCM predicts the next token's representation per-feature-group. The prediction error
is a **vector** (D_col dimensions), not a scalar. This per-feature surprise drives:
- PM bias updates (adapt features that were surprising)
- EM novelty scoring (blended with reconstruction error)
- PCM gain modulation (amplify/dampen features before memory ops)

**Why novel**: Most prediction-error-driven systems (curiosity, surprise-based gating)
use scalar surprise. Vector surprise lets the model know *which features* were
unexpected, enabling selective, per-feature plasticity.

### 3. Causal Write Buffers via Prefix Sums
Within a segment, memory writes are prefix-summed so token t sees the accumulated
effect of writes from tokens 0..t. This preserves strict NTP causality while giving
within-segment memory feedback — the model can "learn and use" within a single segment.

**Why novel**: Most memory systems either batch writes at sequence boundaries (no
within-sequence feedback) or violate causality. Prefix-sum write buffers solve both.

### 4. Neuromodulator-Gated Memory Writes Trained by Main Loss
EM write gates (`g_em`) are produced by a small MLP that takes novelty and usage as
input. Crucially, the neuromodulator is trained end-to-end by the **main CE loss** —
gradient flows: write → primitive update → future trail reads → logits → loss.
No RL reward, no auxiliary objective, single optimizer.

**Why novel**: Most gated-write systems use heuristics or auxiliary losses. End-to-end
gradient through the write gate teaches the model *when writing helps prediction*.

### 5. Dual-Path EM + Hebbian PM (Fast Buffer + Slow Decomposition)
EM uses two write paths; PM uses a single Hebbian path:
- **EM fast path**: `cum_em = cumsum(novelty · w_cand)` — per-token write candidates
  prefix-summed for causal within-segment feedback. Token t sees the cumulative effect
  of writes from positions 0..t. Same-segment gradients flow through.
- **EM slow path**: At segment end, write candidates decomposed across existing
  primitives via soft routing with neuromodulated EMA write strength.
- **PM Hebbian path**: Single commit at segment end — eligibility G accumulates across
  the segment, then `W_b ← W_b @ (decay·I + β_b·G)`. Matches the slower timescale of
  procedural learning (habits need reinforcement across many segments, not immediate feedback).

The EM fast path provides within-segment gradient flow and utility; the slow paths
build durable, structured memory representations.

**Why novel**: No other model separates within-segment approximation from structured
cross-segment writes for episodic memory, while separately using Hebbian fast-weight
plasticity for procedural memory. This decouples gradient flow from memory organization.

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

## Sacred Design Principles (Non-Negotiable)

These define the architecture's identity. Any change that violates these is a different
model, not an optimization. Grouped by category:

### Memory Architecture
1. **Three memory timescales**: Genetic (slow weights, permanent after training),
   Procedural (PM, bias that persists across documents), Episodic (EM, primitives
   that persist across documents). Each has distinct function and update rules.
2. **Memory is state, not parameters**: PM fast-weight matrices (W_pm) and EM primitives
   are runtime tensors that evolve at inference. They are not learned weights — they are
   what the model remembers from experience.
3. **O(1) memory per token at inference**: Fixed capacity (B banks × M primitives for
   EM, B × D_pm² fast-weight matrices for PM). No unbounded growth. Bounded by budgets.

### Memory Operations
4. **Trail-based compositional EM read**: Iterative seed navigation through primitive
   space. No top-k, no hard selection — full softmax over all primitives at each step.
   Combinatorial capacity from finite storage.
5. **Surprise-gated PM updates**: PM adapts only on unpredicted features. Vector
   surprise (per-feature prediction error from PCM) determines which features shift.
6. **Novelty-based EM writes**: EM stores what is both surprising AND not already
   representable by existing primitives (surprise + reconstruction error blend).
7. **Neuromodulators trained by main loss**: Write gates are MLPs trained end-to-end
   by cross-entropy loss. No RL, no auxiliary objective. Single optimizer, single loss.

### Causality and Compute
8. **Causal token processing**: Scan recurrence ensures token t cannot see token t+1.
   Autoregressive (NTP), not bidirectional.
9. **Causal write buffers**: Prefix sums of per-token memory deltas. Token t's memory
   read includes writes from 0..t (inclusive). Strict NTP causality within segments.
10. **Write-before-read ordering**: Within a segment, PM/EM writes are computed before
    reads — the causal buffer ensures reads reflect all prior writes.

### Plasticity Control
11. **Budget enforcement**: PM norm budgets and EM strength budgets prevent unbounded
    drift. Memory stays bounded regardless of deployment duration.
12. **Vector surprise**: Per-feature prediction error (D_col dimensions), not scalar.
    Enables selective, per-feature plasticity — only surprising features trigger updates.

## Negotiable — Implementation Details

- Architecture dimensions (D, B, C, M, D_embed, d_inner)
- Segment length N (= memory update interval)
- Number of scan layers per stage, scan depth asymmetry
- Trail steps count, number of banks
- Hyperparameters (decay rates, temperatures, budgets, learning rates)
- Norm type (RMSNorm vs LayerNorm), activation functions
- Dense vs grouped scan (currently dense for GPU efficiency)
- Specific kernel implementations (HGRN, Triton, etc.)

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
