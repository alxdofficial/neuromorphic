# Motivation & Design Philosophy — Neuromorphic Language Model

> **NOTE (2026-03-27):** This doc was written for v8 (RL/GRPO neuromodulator).
> Current code (v9-ES) replaces RL with Evolution Strategies, replaces port
> neurons with broadcast inject/readout, uses 5 scan layers (not 7), and
> has per-neuron modulator MLP + dendritic FC trained by ES.
> The core philosophy (memory as signal flow, not database) still applies.

## Why This Exists

Modern LLMs are powerful but structurally limited:

1. **LLMs are static** — once deployed, weights don't change. Improving requires
   finetuning, retraining, or bolting on retrieval (RAG). This leaves a gap between
   short-term learning, long-term personalization, and robust agent memory.

2. **RAG is not learning** — retrieval can fetch text but doesn't integrate knowledge
   into the model's computations, adapt behavior continuously, or compress and
   generalize experience.

3. **Transformers are expensive for long context** — O(N^2) attention makes lifelong
   learning impractical. If we want AI embedded everywhere and eventually in silicon,
   we need O(1) memory that scales as state, not as quadratic attention.

4. **Agents need to improve by doing** — a practical agent should remember what it has
   done, refine tool-use patterns, adapt to workflows, and get better over time without
   cloud retraining.

## What We're Building

A **brain-inspired sequence model** with three components:

- **Cortical Columns (CCs)**: The language model. Dense scan layers with per-column
  predictive coding. Process tokens causally, produce surprise signals. Trained by
  backprop. See `docs/architecture_v8_neural_memory_graph.md`.

- **Neural Memory Graph**: A persistent network of 1024 neurons outside the autograd
  graph. At every token, neurons receive presynaptic messages, integrate them with
  internal state, and broadcast outgoing messages modulated by their primitives.
  Memory IS the pattern of activation and connectivity flowing through the graph.

- **Neuromodulator**: An RL-trained policy (GRPO trajectory scoring, no value function)
  that gates Hebbian plasticity and controls decay. Three-factor learning: eligibility
  traces accumulate what neurons encode/receive, the neuromod outputs a gate (consolidate
  vs reverse) and decay target per neuron. Scores trajectories across all 4 collected
  chunks for long-range credit assignment. Only K=96 neurons get actions per RL step;
  best trajectory's state persists. Substitutes for the billions of years of evolution
  that shaped the brain's neuromodulatory systems.

---

## Core Philosophy

### The Brain's Memory Is Not a Database

The human brain doesn't do "memory read" and "memory write" operations. There is no
cosine similarity lookup, no discrete read/write schedule. Instead:

- **Neurons constantly fire and receive signals.** Memory IS the pattern of activity
  that emerges from connection weights. Recall is pattern completion — a partial input
  activates connected neurons, which activate others, until the full pattern emerges.

- **Plasticity is continuous and local.** When two neurons co-activate, their connection
  strengthens (Hebbian learning). Neuromodulators (dopamine, acetylcholine) gate HOW MUCH
  connections change — they signal "this is important, learn more."

- **The cortical column is universal.** The same 6-layer circuit handles vision, language,
  motor control — everywhere in the neocortex. Specialization comes from connectivity
  and input, not different hardware.

### What We Take from the Brain

| Principle | Brain | Our Model |
|-----------|-------|-----------|
| Universal compute unit | Cortical column (same everywhere) | CC: scan + PCM (shared scan weights) |
| Memory as activation | Neurons firing through weighted connections | Neural memory graph: h + messages through A |
| Per-token dynamics | Neurons fire and exchange signals continuously | Sequential loop: receive -> integrate -> message |
| Neuromodulated plasticity | Dopamine/ACh gate learning rate | RL-trained gate controls Hebbian trace consolidation |
| Stable signal propagation | Balanced excitation/inhibition | Key-based sigmoid routing (independent per-connection gating [0, 1]) |
| Predictive coding | Cortical prediction error signals | PCM: vector surprise per feature group |

### What We Don't Take from the Brain

- Spiking dynamics -> continuous activations (D=128 vector per neuron)
- Exact cortical layers -> scan recurrence + PCM
- Axonal delays, glial cells -> ignored
- Specific neurotransmitters -> single learned neuromodulator

### Temporal Patterns Compressed into Vectors

The brain's expressiveness comes from temporal patterns — spikes, oscillations, burst
timing. Simulating this is computationally impractical. We collapse it: a D_mem=128
float vector at one timestep represents what the brain would express as a low-dimensional
signal evolving over many milliseconds. The capacity is sufficient to encode all meaningful
temporal patterns in a neural population's short time window.

---

## How It Works (v8)

### Split-Scan + Per-Token Memory Graph

```
Lower scan: layers 0-3 over T=2048 tokens (parallel)
PCM: surprise computed at split point (side input to upper scan)

Memory: Per-token neuron dynamics (sequential, every token):
  For each token:
    1. RECEIVE (dendritic tree): 3-level nonlinear gather
       (8 branches of 12 -> tanh, 2 groups of 4 -> tanh, soma avg)
       Port neurons also get CC signal from H_mid (unchanged)
    2. INTEGRATE: h = decay * h + (1-decay) * received
    3. MESSAGE: message = tanh(h * primitives)
  Port neuron messages -> mem_signals

Inject: H_enriched = H_mid + gate * mem_signals (mid-scan)
Upper scan: layers 4-6 over H_enriched (parallel)
Output: logits = output_head(H_final)
```

The scan is split so upper layers see memory-enriched representations. Memory
runs a sequential per-token loop with one bmm per token. Signals propagate
hop-by-hop — K tokens = K hops of inter-neuron communication.

### Neuromodulator as RL Agent

The neuromodulator observes each neuron's state (primitive, key, mean input,
mean output, msg_magnitude, decay, trace norms) and outputs a gate (tanh'd to
[-1,1]) and decay target per neuron. Gate > 0 consolidates Hebbian eligibility
traces; gate < 0 reverses them (exploration); gate ~ 0 maintains. Trained via
GRPO trajectory scoring: choose K=96 neurons, sample 8 trajectories across ALL
4 collected chunks, rank by total CE, z-score normalize, encourage above-average
trajectories. Non-K neurons get gate=0 (no plasticity) for clean credit
assignment. Best trajectory's state persists. No value function or critic.
Neuromod LR decays alongside the LM LR.

This replaces the brain's neuromodulatory system, which was shaped by billions
of years of evolution. We compress this into an RL training loop.

### Lifelong Learning

Memory is fully persistent across documents from the start. The memory graph
learns what document transitions look like through CC signal changes (scan
carries reset at doc boundaries, causing abrupt H_mid shifts). The neuromodulator
learns to maintain useful memories and forget stale ones. The scan (slow weights)
encodes general language knowledge; memory adapts to specific context.

---

## Genuine Novelties

### 1. Memory as Continuous Signal Flow
Memory is not a database with read/write operations. It is a persistent neuron
graph where recall IS activation pattern completion. No cosine similarity, no
explicit queries — just signal propagation through weighted connections at every
token step.

### 2. RL-Trained Plasticity (Not Backprop Through Memory)
Memory learning uses three-factor learning: Hebbian eligibility traces accumulate
local statistics, and an RL-trained neuromodulator (GRPO) gates whether to consolidate
or reverse those traces. Memory is an environment, the neuromodulator is the agent.
This decouples memory from the autograd graph, solving the throughput problem of
differentiable memory systems.

### 3. Per-Token Neuron Dynamics
Each neuron receives, integrates, and messages at every token. Signals propagate
through the graph hop-by-hop. This is fundamentally different from attention-based
memory (discrete read/write) or SSM memory (flat hidden state). The graph structure
gives memory spatial organization — neurons develop specialization through
neuromodulator-driven plasticity.

### 4. Vector Surprise as Universal Learning Signal
PCM predicts next token's representation per-feature-group. The vector prediction
error (D_cc=128 dims per column) feeds into memory as part of the CC signal,
telling the memory graph which features were unexpected.

### 5. Stable Signal Propagation via Graph Structure
Key-based sigmoid routing (each connection independently gated [0, 1]) controls signal
propagation without normalization — strong connections don't suppress weak ones, matching
biological synaptic independence. Structural plasticity (co-activation-based prune + regrow)
reshapes the graph over time. Anti-correlated connections are pruned; new connections form
toward neurons with high temporal co-firing (phi coefficient).

---

## What Makes This Fundamentally Different

### vs Transformers
- Fixed memory footprint vs O(N) KV cache growth
- Memory adapts at inference vs frozen weights
- O(N) compute vs O(N^2) attention

### vs SSMs (Mamba, RWKV)
- Structured persistent memory vs flat hidden state
- RL-trained plasticity vs no explicit learning mechanism
- Per-neuron, per-token dynamics vs single state vector

### vs RAG
- Memory inside the model, not external
- Continuous automatic updates, not manually triggered
- Compressed/generalized knowledge, not verbatim storage

### vs Differentiable Memory (NTM, DNC)
- Memory outside autograd — no gradient-through-memory bottleneck
- RL-trained write policy — learns from outcome, not gradient signal
- Continuous signal flow — not discrete read/write operations

---

## Success Criteria

### Core: Language modeling works
Loss decreases during training. CCs produce coherent predictions without memory.

### Memory helps: Lower loss with memory than without
The memory-enabled model outperforms the no-memory baseline on the same data.

### Neuromodulator learns: Non-random plasticity policy
RL converges. Actions correlate with context. Policy loss decreases.

### Lifelong: Memory accumulates useful context
In Phase B, long-document perplexity improves as memory accumulates context.
Memory neurons develop specialization (diverse primitives, structured routing).

### Hardware: Efficient
Phase 1 (memory, no neuromod): ~64K tok/s. Phase 2 (memory + neuromod, frozen
LM): ~87K tok/s. No memory baseline: ~161K tok/s.
