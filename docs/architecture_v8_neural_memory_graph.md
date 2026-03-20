# Architecture v8: Neural Memory Graph + Cortical Columns

> **Status**: Design document. Not yet implemented.
> **Previous versions**: v7 (single scan stack) on `v7-single-scan-stack` branch.
> v6 (3-stage scan-memory-scan), v4 on `v4-iterative-backup`, v1 on `v1-legacy`.

## Design Philosophy

Start simple. The transformer succeeded with one core mechanism (attention).
Mamba succeeded with one core mechanism (selective state spaces). We need one
core mechanism, grounded in neuroscience, that we can later scale.

The human brain's memory is not a database with read/write operations. It is a
**sea of neurons constantly emitting and receiving signals**, where the pattern of
activity IS memory, and the connection weights determine what patterns emerge.
Recall is pattern completion through signal propagation. Learning is plasticity
modulated by neuromodulators. All of this happens without backpropagation.

We aim to replicate the principles that matter for language modeling, not the
biological implementation details that exist only because brains are made of
carbon.

## Neuroscience Grounding

### What we take from the brain

1. **The cortical column is the universal compute unit.** Same 6-layer circuit
   everywhere in the neocortex. Specialization comes from connectivity and input,
   not different hardware. Our cortical columns (CCs) are the language model —
   they process tokens, predict next tokens, and generate surprise signals.

2. **Memory is activation flowing through weighted connections.** There is no
   separate memory store that gets queried. Neurons fire, signals propagate
   through synaptic weights, and the network settles into patterns. What we call
   "recall" is a partial input activating a stored pattern. What we call "storage"
   is weight changes that make patterns reproducible.

3. **Plasticity is local and neuromodulated.** Synapses strengthen when
   pre-synaptic and post-synaptic neurons co-activate (Hebbian). But how MUCH
   they change is gated by neuromodulators (dopamine, acetylcholine, etc.) that
   signal "this is important, learn more." The neuromodulator doesn't know the
   optimal weight — it just controls the learning rate. Evolution shaped the
   neuromodulatory system over billions of years. We substitute RL training.

4. **Energy conservation constrains signal routing.** A neuron's output energy
   is finite. It must be divided among downstream connections. This creates
   natural sparsity and forces selective routing without explicit top-k or
   budget mechanisms.

5. **Temporal patterns compressed into vectors.** The brain uses spike timing,
   oscillations, and burst patterns across time for expressiveness. We collapse
   this into high-dimensional vectors at each timestep — a 512-dim float vector
   can encode all meaningful temporal patterns in a neural population's short
   time window.

### What we don't take from the brain

- Spiking dynamics (we use continuous activations)
- Exact cortical layer structure (we approximate with scan + PCM)
- Axonal delays, dendritic computation, glial cells
- The specific neurotransmitter system (we use a learned neuromodulator)

---

## Architecture Overview

Three components, each with a clear role:

```
┌─────────────────────────────────────────────────────────┐
│                    CORTICAL COLUMNS                      │
│  (The language model — scans + PCM + output head)        │
│  Process tokens causally, predict next token, produce    │
│  surprise signals. Each CC handles a slice of D.         │
│                                                          │
│  Connected to memory graph as if CCs are just neurons.   │
│  Receive memory signals as input. Send CC output as      │
│  signal back into memory.                                │
└────────────────────┬────────────────────┬────────────────┘
                     │ signals in         │ signals out
                     ▼                    │
┌─────────────────────────────────────────────────────────┐
│                 NEURAL MEMORY GRAPH                       │
│  (The environment — persistent, runs every token)         │
│                                                          │
│  N neurons, each with:                                   │
│    - primitive_value [D_mem]: stored information          │
│    - energy_thresholds [connections]: routing weights     │
│                                                          │
│  Each timestep, each neuron:                             │
│    1. Receives combined signal from pre-synaptic neurons │
│    2. Modulates signal with primitive_value → output     │
│    3. Routes output proportional to energy thresholds    │
│       (conservation: sum of outbound = output magnitude) │
│       (sparsity: below-threshold connections get nothing) │
│                                                          │
│  Organized in blocks (local neighborhoods, fully         │
│  connected within block, sparse between blocks).         │
│  CCs attach as special neurons at block boundaries.      │
│                                                          │
│  NOT in the autograd graph. Runs as CUDA kernel (SIMD).  │
└────────────────────┬────────────────────────────────────┘
                     │ neuron states, activations
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   NEUROMODULATOR                          │
│  (The learning rule — trained by RL)                     │
│                                                          │
│  Observes: neuron state, recent inputs, recent outputs,  │
│            routing patterns, CC surprise signals          │
│  Produces: modifications to primitive_values and          │
│            energy_thresholds for each neuron              │
│                                                          │
│  Shared MLP across all neurons (or per-block).           │
│  Trained via PPO using CC surprise/loss as reward.        │
│  Collects experience during segment, trains at segment   │
│  end.                                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Component 1: Neural Memory Graph

### Neuron State

Each neuron i has:
- `primitive[i]`: vector `[D_mem]` — the neuron's stored information
- `thresholds[i]`: vector `[max_connections]` — energy cost to send to each
  connected neuron. Higher threshold = harder to activate that connection.
- `activation[i]`: vector `[D_mem]` — current activation (signal buffer)

### Signal Flow (per timestep, per neuron)

```
1. RECEIVE: input_i = sum of all incoming signals from pre-synaptic neurons

2. MODULATE: output_i = f(input_i, primitive_i)
   where f is a small nonlinear function:
     output_i = activation_fn(W_mod @ [input_i; primitive_i])
   W_mod is a small shared projection (not per-neuron — too expensive).
   This produces the neuron's response given its stored info + current input.

3. ROUTE: divide output_i among post-synaptic neurons
   - Compute routing weights: w_j = softmax(-thresholds_j / temperature)
     over connected neurons j (lower threshold = easier to send = higher weight)
   - Apply sparsity: zero out bottom half of routing weights
   - Renormalize so sum = 1
   - Send: signal_to_j = w_j * output_i   (energy conserved)
```

### Block Structure

Neurons are organized into B blocks of M neurons each.
- **Within block**: fully connected (every neuron can route to every other)
- **Between blocks**: sparse random connections (k connections per neuron to
  other blocks, where k << M)
- **CC attachment**: each block has one or more CC "ports" — special connection
  points where CC signals enter and exit the memory graph

Total neurons: B * M (e.g., 8 blocks × 64 neurons = 512 neurons)

### Why This Is SIMD-Parallel

Each neuron's timestep is: read inputs → modulate → route outputs.
All neurons can execute simultaneously because:
- Inputs are computed from the PREVIOUS timestep's outputs (double-buffered)
- Modulation is independent per neuron
- Routing writes to output buffers (scatter, no conflicts with proper indexing)

This maps to a CUDA kernel where each thread handles one neuron.

---

## Component 2: Cortical Columns

CCs are the language model. They process the token stream causally and
produce next-token predictions. They connect to the memory graph as
special neurons that both receive memory signals and emit signals back.

### CC Internal Structure

Each CC processes a slice of the token embedding (D_cc = D / num_CCs).
Internal computation per token:

```
1. INPUT FUSION:
   cc_input = concat(token_slice, memory_signal_from_graph)

2. SCAN RECURRENCE (causal, parallelizable):
   h_t = sigmoid(a_t) * h_{t-1} + silu(b_t)
   where [a, b] = proj_in(RMSNorm(cc_input))
   With SwiGLU output: out = silu(gate) * up + residual

3. PCM (Predictive Coding):
   z_hat_t = W_pcm(h_t)           — predict next token's encoding
   z_{t+1} = W_enc(x_{t+1})       — actual next encoding
   surprise = z_hat_t - z_{t+1}   — vector surprise (D_cc dims)
   Surprise feeds back into memory graph as a signal.

4. OUTPUT:
   cc_output → memory graph (as signal into connected neurons)
   cc_output → fusion layer → logits (for language modeling)
```

### CC-to-Memory Interface

From the memory graph's perspective, each CC is just another neuron:
- It receives signals from its connected memory neurons (memory read)
- It sends its output signal back into the graph (memory write)
- No special read/write schedule — happens every token

### Cross-CC Fusion

CCs each process a slice of D. To produce logits, CC outputs must be
combined. Options (in order of simplicity):
1. **Concatenation + linear**: simplest, just concat CC outputs and project
2. **Small transformer block**: 1-2 layers of self-attention over CC outputs
3. **Scan across CCs**: treat CCs as a sequence, scan to integrate

Start with option 1.

### Scan Still Works

The scan recurrence within each CC operates across the token sequence
(t=1..N). This is the same parallel scan we already use. Memory signals
are just additional input to the scan at each timestep — they don't break
the recurrence structure.

---

## Component 3: Neuromodulator (RL-Trained)

The neuromodulator controls plasticity in the memory graph. It doesn't
know the optimal neuron states — it learns a policy for WHEN and HOW MUCH
to modify neurons, based on context.

### Architecture

Shared MLP (or per-block MLP) that processes each neuron's context:

```
Input features per neuron:
  - primitive_value [D_mem]
  - recent_mean_input [D_mem]     — running average of inputs
  - recent_mean_output [D_mem]    — running average of outputs
  - routing_entropy [1]           — how spread out is the routing?
  - connected_cc_surprise [D_cc]  — surprise from nearest CC
  - usage_frequency [1]           — how often activated recently

Action per neuron:
  - delta_primitive [D_mem]       — additive modification to primitive
  - delta_thresholds [max_conn]   — additive modification to thresholds
```

### Training via PPO

**Reward signal** (per block, dense):
- Primary: reduction in CC surprise/prediction error after neuromodulator
  action. Dense per-token signal, measured at block level.
- Secondary: per-token language modeling loss improvement (sparser but
  more directly aligned with the objective).

**Experience collection:**
During a segment of N tokens, the neuromodulator acts at each timestep
(or every k timesteps). Actions and rewards are stored in a replay buffer.

**Policy update:**
At segment end (or every K segments), run PPO update:
1. Compute advantages using GAE (Generalized Advantage Estimation)
2. Clipped surrogate objective (prevent destructive updates)
3. Value function baseline per block (reduce variance)
4. Entropy bonus (encourage exploration, prevent mode collapse)

**Why PPO over alternatives:**
- REINFORCE: too high variance for continuous actions in high-dim spaces
- GRPO: designed for discrete actions (token selection), not continuous control
- PPO: standard for continuous control, well-understood stability properties
- SAC: also viable (entropy-regularized, off-policy), but PPO is simpler to start

### Bootstrap Solution

To avoid the cold-start problem where CCs learn to ignore useless memory:

1. **Initialize primitives from embeddings**: seed neuron primitives with
   token embedding vectors or random projections thereof. Memory starts
   with meaningful content.
2. **Initialize thresholds uniformly**: all connections start with equal
   energy thresholds, so routing is initially uniform (no dead connections).
3. **Entropy bonus in PPO**: strongly encourage exploration early in training.
   The neuromodulator should try diverse modifications before settling.
4. **CC memory gate**: CCs have a learnable gate on memory input that starts
   near 0.5 (not zero). This ensures CCs always attend to memory signals
   somewhat, giving the neuromodulator something to optimize against.
5. **Warm-start CCs**: optionally pre-train CCs without memory for a few
   thousand steps, then attach memory. CCs have basic language ability,
   memory has something useful to augment.

---

## Putting It Together: Training Loop

```
For each TBPTT chunk of T tokens:

  1. Memory graph is running (CUDA kernel, persistent state)

  2. For each token t = 1..T:
     a. Memory graph step: all neurons receive, modulate, route (SIMD)
     b. CCs receive memory signals + token embedding
     c. CCs process token (scan recurrence + PCM)
     d. CCs send output signal back into memory graph
     e. Neuromodulator observes neuron states, stores experience
     f. Neuromodulator acts: modifies primitives and thresholds

  3. Language modeling loss: standard NTP cross-entropy on CC outputs
     → Backprop through CCs only (memory graph not in autograd)

  4. Neuromodulator update (PPO):
     → Use collected experience + CC surprise/loss as reward
     → Update neuromodulator policy
     → This is the only way memory graph learns

  5. Detach CC scan carries at TBPTT boundary
     Memory graph state persists (no detach — it's not in autograd)
```

### Computational Flow

```
Token t arrives
    │
    ├──► Embedding lookup ──► CC input (token slice + memory signal)
    │                              │
    │                              ▼
    │                         CC scan layer(s) ──► CC output
    │                              │                    │
    │                              ▼                    │
    │                         PCM surprise              │
    │                              │                    │
    │    ┌─────────────────────────┘                    │
    │    │                                              │
    │    ▼                                              ▼
    │  Neuromodulator                          Signal into memory graph
    │  (observe + act)                                  │
    │    │                                              ▼
    │    ▼                                    Memory graph step (CUDA)
    │  Modify neuron                          All neurons: receive →
    │  primitives +                           modulate → route
    │  thresholds                                       │
    │                                                   ▼
    │                                         Memory signals for t+1
    │                                              │
    └──────────────────────────────────────────────┘
                    (next token)
```

---

## Prototype Plan

### Phase 0: Minimal viable memory graph

**Goal**: Prove that the RL-trained neuromodulator can learn useful memory
modifications, and that CCs can learn to use memory signals.

**Scale**:
- 256 memory neurons, 4 blocks of 64
- D_mem = 32 (small primitive vectors)
- 4 CCs (one per block), D_cc = 64
- Tiny LM: D=256, vocab=32000, ~5M params in CCs
- Neuromodulator: shared 2-layer MLP, ~50K params

**Memory graph**: pure PyTorch first (not custom CUDA). Optimize later.

**Training**:
- Pre-train CCs alone for 1K steps (basic language ability)
- Attach memory graph, train jointly for 10K steps
- Monitor: does CC surprise decrease? Does the neuromodulator learn
  non-trivial policies? Do memory neurons develop specialization?

**Success criteria**:
- Neuromodulator policy is not random (actions correlate with context)
- At least some memory neurons develop distinct primitives
- CC loss with memory < CC loss without memory (even slightly)

### Phase 1: Scale up

If Phase 0 succeeds:
- 1024-4096 neurons, 8-16 blocks
- D_mem = 128-256
- Full-scale CCs matching current Tier A
- Custom CUDA kernel for memory graph
- Benchmark throughput against v7

### Phase 2: Long-context and lifelong

- Test on documents requiring cross-segment recall
- Lifelong mode: memory persists across documents
- Compare to transformer + RAG baselines

---

## Open Questions

1. **Memory graph timestep granularity**: every token? Every 4 tokens? Every
   CC scan layer? The brain's hippocampus runs slower than neocortex. Coarser
   memory steps reduce CUDA kernel launches.

2. **Neuromodulator action frequency**: every token? Every N tokens? At segment
   boundaries only? More frequent = more experience but more compute.

3. **Block-to-block connectivity**: random fixed? Learned? Should the
   neuromodulator also modify inter-block connections?

4. **W_mod sharing**: the modulation function f(input, primitive) needs
   parameters. Shared across all neurons? Per-block? Trained by backprop
   (since it's inside CCs' forward path if memory is in autograd) or by RL?
   If memory is not in autograd, W_mod must be trained by RL too.

5. **Memory signal dimensionality**: should memory signals into CCs be the
   full D_mem, or projected down? CCs process D_cc slices — memory signals
   should probably match.

6. **Value function architecture**: for PPO, we need V(s) per block. What
   state representation? Mean neuron activation? CC hidden state?

7. **Off-policy vs on-policy**: PPO is on-policy. If memory graph runs as
   environment, we could use off-policy methods (SAC) and reuse experience
   more efficiently. Worth exploring after PPO baseline.

---

## Comparison: v8 vs v7

| Aspect | v7 (single scan stack) | v8 (neural memory graph) |
|--------|----------------------|--------------------------|
| Memory model | PM fast-weight matrix + EM dictionary | Continuous neuron graph |
| Memory access | Segment boundaries (N=128) | Every token |
| Memory in autograd | Yes (limits throughput) | No (CUDA environment) |
| Learning rule | Backprop through commits | RL (PPO) on neuromodulator |
| Biological accuracy | Moderate (Hebbian PM, trail EM) | Higher (continuous signal flow) |
| Complexity | Accumulated engineering | Principled but more components |
| Throughput bottleneck | K sequential segments | Memory CUDA kernel latency |
| Gradient for memory | Cross-segment TBPTT | N/A (RL, no gradients needed) |

## Comparison: v8 vs Transformer

| Aspect | Transformer | v8 |
|--------|------------|-----|
| Context mechanism | KV cache (O(N) memory) | Persistent neuron graph (O(1)) |
| Memory at inference | Grows with context | Fixed size, adapts content |
| Learning at inference | None | Continuous (neuromodulator) |
| Core innovation | Attention | Neural memory graph + RL plasticity |
