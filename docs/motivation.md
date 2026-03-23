# Motivation & Design Philosophy — Neuromorphic Language Model

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

- **Neuromodulator**: An RL-trained policy (REINFORCE with learned value baseline)
  that modifies neuron primitives, connection weights, and decay. Collects across
  multiple chunks for longer reward horizon. Substitutes for the billions of years
  of evolution that shaped the brain's neuromodulatory systems.

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
| Neuromodulated plasticity | Dopamine/ACh gate learning rate | RL-trained policy modifies neuron state |
| Stable signal propagation | Balanced excitation/inhibition | L1-normalized connection weights (energy conservation) |
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
PCM: surprise + learnable gain modulation (at split point)

Memory: Per-token neuron dynamics (sequential, every token):
  For each token:
    1. RECEIVE: each neuron gets weighted sum of presynaptic messages
       Port neurons also get CC signal from the LM (gain-modulated)
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

The neuromodulator observes each neuron's state (primitive, mean activity,
firing rate, decay, routing entropy) and outputs modifications to primitives,
connection weight distribution, and decay. Trained via REINFORCE with a learned
value function baseline (V(global_state) → scalar). Collects across 2 chunks
(16 segments) before updating for longer reward horizon. Value bootstrap at
collection boundary gives credit for structural changes that pay off beyond
the current window. Neuromod LR decays alongside the LM LR.

This replaces the brain's neuromodulatory system, which was shaped by billions
of years of evolution. We compress this into an RL training loop.

### Lifelong Learning

**Phase A**: Memory resets at document boundaries. CCs and neuromodulator learn
basic language ability and write selectivity.

**Phase B**: Memory persists across documents. The neuromodulator learns to
maintain useful memories and forget stale ones. The scan (slow weights) encodes
general language knowledge; memory adapts to specific context.

---

## Genuine Novelties

### 1. Memory as Continuous Signal Flow
Memory is not a database with read/write operations. It is a persistent neuron
graph where recall IS activation pattern completion. No cosine similarity, no
explicit queries — just signal propagation through weighted connections at every
token step.

### 2. RL-Trained Plasticity (Not Backprop Through Memory)
Memory learning is controlled by a REINFORCE-trained neuromodulator, not by
backprop through memory state. Memory is an environment, the neuromodulator is the
agent. This decouples memory from the autograd graph, solving the throughput problem
of differentiable memory systems.

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
L1-normalized connection weights (energy conservation) bound signal propagation.
Structural plasticity (co-activation-based prune + regrow) reshapes the graph
over time. Anti-correlated connections are pruned; new connections form toward
neurons with high temporal co-firing (phi coefficient).

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
Memory graph roughly halves throughput vs no-memory baseline. ~44K tok/s
with memory vs ~85K without (RTX 4090, BS=12).
