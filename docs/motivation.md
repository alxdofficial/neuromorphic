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

A **brain-inspired sequence model** with three components:

- **Cortical Columns (CCs)**: The language model. Dense scan layers with per-column
  predictive coding. Process tokens causally, produce surprise signals. Trained by
  backprop. See `docs/architecture_v8_neural_memory_graph.md`.

- **Neural Memory Graph**: A persistent network of neurons outside the autograd graph.
  Neurons have stored information (primitives) and energy-conserving routing. Signal
  flows continuously — every token. Memory IS the pattern of activation and connectivity,
  not a database that gets queried.

- **Neuromodulator**: An RL-trained policy (PPO) that modifies neuron primitives and
  routing thresholds. Substitutes for the billions of years of evolution that shaped
  the brain's neuromodulatory systems.

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
| Memory as activation | Neurons firing through weighted connections | Neural memory graph: primitives + routing |
| Neuromodulated plasticity | Dopamine/ACh gate learning rate | PPO-trained policy modifies neuron state |
| Energy conservation | Finite metabolic budget per neuron | Output magnitude divided among connections |
| Predictive coding | Cortical prediction error signals | PCM: vector surprise per feature group |

### What We Don't Take from the Brain

- Spiking dynamics → continuous activations
- Exact cortical layers → scan recurrence + PCM
- Axonal delays, glial cells → ignored
- Specific neurotransmitters → single learned neuromodulator

### Temporal Patterns Compressed into Vectors

The brain's expressiveness comes from temporal patterns — spikes, oscillations, burst
timing. Simulating this is computationally impractical. We collapse it: a D_mem=256
float vector at one timestep represents what the brain would express as a low-dimensional
signal evolving over many milliseconds. The capacity is sufficient to encode all meaningful
temporal patterns in a neural population's short time window.

---

## How It Works (v8)

### Two-Pass Scan with Memory Loop

```
Pass 1: Pre-memory scan layers (parallel over T=2048 tokens)
         → Build representation H, compute per-CC surprise

Memory loop: For each token (sequential, cheap, no_grad):
  CC → memory: inject (H_slice + surprise) into memory graph
  Memory graph step: all neurons receive → modulate → route
  Memory → CC: read signals from block port neurons

Pass 2: Post-memory scan layers (parallel over T=2048 tokens)
         → Integrate memory signals → logits
```

Scans run at full GPU efficiency over all T tokens. The memory loop is cheap
(no autograd, SIMD across neurons). One backward pass, one PPO update per chunk.

### Neuromodulator as RL Agent

The neuromodulator observes each neuron's state (primitive, recent activity,
routing entropy, CC surprise) and outputs modifications to primitives and
thresholds. Trained via PPO with per-token language modeling loss as reward.

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
explicit queries — just signal propagation through weighted connections.

### 2. RL-Trained Plasticity (Not Backprop Through Memory)
Memory learning is controlled by a PPO-trained neuromodulator, not by backprop
through memory state. Memory is an environment, the neuromodulator is the agent.
This decouples memory from the autograd graph, solving the throughput problem of
differentiable memory systems.

### 3. Vector Surprise as Universal Learning Signal
PCM predicts next token's representation per-feature-group. The vector prediction
error (D_cc=128 dims per column) feeds into memory as part of the CC signal,
telling the memory graph which features were unexpected.

### 4. Energy-Conserving Routing
Neuron output magnitude is divided among connections proportionally to learned
thresholds. Natural sparsity emerges without top-k or budget mechanisms. The
neuromodulator learns to adjust thresholds to route information effectively.

---

## What Makes This Fundamentally Different

### vs Transformers
- Fixed memory footprint vs O(N) KV cache growth
- Memory adapts at inference vs frozen weights
- O(N) compute vs O(N²) attention

### vs SSMs (Mamba, RWKV)
- Structured persistent memory vs flat hidden state
- RL-trained plasticity vs no explicit learning mechanism
- Per-CC memory channels vs single state vector

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
PPO converges. Actions correlate with context. KL divergence stays bounded.

### Lifelong: Memory accumulates useful context
In Phase B, long-document perplexity improves as memory accumulates context.
Memory neurons develop specialization (diverse primitives, structured routing).

### Hardware: Efficient
Memory graph adds <10% overhead vs no-memory baseline. Throughput competitive
with Pythia-160M / Mamba-130M at similar param count.
