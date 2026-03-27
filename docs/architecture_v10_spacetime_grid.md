# v10 Design: Spacetime Neuron Grid

**Status**: Design phase (brainstorming, not yet implemented)
**Date**: 2026-03-27

## Motivation

v8 (RL/GRPO) and v9 (ES + per-neuron MLPs) both failed to train the memory
graph effectively. Root causes:

- **Sequential bottleneck**: Per-token neuron dynamics loop (128 steps × 16
  segments = 2048 sequential Python iterations per chunk). With per-neuron
  MLPs, throughput dropped from 47K to 2K tok/s.
- **Non-differentiable memory**: RL and ES both failed to provide useful
  training signal. GRPO: neuron homogeneity + credit assignment noise. ES:
  8 trajectories can't optimize 31M params, internal neurons dead.
- **Dead internal neurons**: Port neuron shortcuts — CC input goes directly
  to LM output in one hop, bypassing 992 internal neurons.
- **VRAM blowup**: Differentiable memory through 128 steps = 19.3GB at BS=2.

## Core Insight: Spacetime Nodes, Not Evolving State

Instead of thinking "1024 neurons that evolve over 128 time steps" (sequential),
reframe as "131,072 spacetime nodes that all exist simultaneously" (parallel).

Neuron n at time t is a **separate entity** (n, t). The "state evolution" of a
neuron is just **message passing along temporal edges** — the same mechanism as
spatial message passing between different neurons. Time is just another spatial
dimension.

```
Old: 1024 neurons × 128 sequential steps  → Python loop over t
New: 131,072 spacetime nodes × L layers   → fully parallel
```

## Architecture

### The Spacetime Grid

Neurons arranged on a 2D grid (e.g., 32×32 = 1024 neurons). Time is the 3rd
dimension. For a segment of T_seg=128 tokens:

```
Grid: [BS, 32, 32, 128, D]
       ↑    ↑   ↑   ↑    ↑
      batch  H   W   T   features
```

Each position (h, w, t) is a spacetime node. All nodes are processed
simultaneously by each layer. No sequential loop over T.

### Connectivity

Each spacetime node (h, w, t) has neighbors defined by the grid:

- **Spatial**: (h±1, w±1, t) — neurons at the same time step
- **Temporal**: (h, w, t-1), (h, w, t-2), ... — the neuron's own past (causal)
- **Cross-spacetime**: (h±1, w±1, t-1) — neighbors in the past

Causal constraint: temporal edges only go backward. (n, t) can see (n, t-k)
but never (n, t+k).

A 3×3 spatial × 3 temporal causal kernel gives ~13 spacetime neighbors per node.

### Unified Routing Mechanism

**No attention.** Routing comes from the neuron's own features, not from
query-key content matching.

Each neuron's feature vector has two parts:

```
features = [content (D_c), routing (K_neighbors)]

content:  what the neuron "knows" — changes every token
routing:  how much to listen to each neighbor — changes slowly (plasticity)
```

The routing weights are sigmoid-gated scalars per neighbor, predicted from the
neuron's own slow-changing features. The neuron decides how much to listen to
each direction based on its internal state, not based on what neighbors contain.

### Per-Layer Computation

```python
# 1. Extract routing weights from neuron's own features
routing = sigmoid(features[..., D_c:])         # [BS, H, W, T, K_neighbors]

# 2. Gather neighbors' content (fixed grid topology)
neighbors = gather_3d_neighborhood(content)     # [BS, H, W, T, K_neighbors, D_c]

# 3. Weighted receive (neuron's own routing decision)
received = (routing.unsqueeze(-1) * neighbors).sum(dim=-2)

# 4. Transform and update
output = MLP(cat(content, received)) + residual
```

Three levels of the design:
- **Topology** (which neighbors): fixed by grid position
- **Routing weights** (how much from each): per-neuron, from slow features, evolves at inference
- **Transform** (how to process): shared weights, learned during training

### Multi-Timescale Features

```
D = D_fast + D_slow

D_fast: changes freely every token (working memory, current content)
D_slow: strong temporal self-connection, changes gradually (plasticity state)
```

Slow features modulate routing — they act as learned routing biases that
evolve over many segments. This IS synaptic plasticity: the connection
strength changes over long timescales based on accumulated experience.

At inference: slow features continue evolving through the forward pass.
No gradients needed. The neuron's routing naturally adapts.

### Sparse I/O (Write/Read Ports)

- C write port neurons (random grid positions): receive token embeddings
- C read port neurons (different random positions): produce output for decoder
- The remaining ~1000 neurons are internal computation and memory
- Information must propagate spatially from write ports through the grid to
  read ports — no shortcuts

### Encoder/Decoder

```
Input tokens [BS, T_seg]
  → Encoder: embed + project → split across C write ports on the grid
  → Spacetime grid: [BS, 32, 32, T_seg, D]

  → L layers of:
      gather from grid neighbors (fixed topology)
      weight by neuron's own routing features (dynamic, per-neuron)
      transform (shared MLP)
      residual

  → Gather from read ports: [BS, T_seg, C × D_c]
  → Decoder: scan layers or transformer → logits [BS, T_seg, vocab]
```

The memory grid is the core compute. The encoder/decoder are thin wrappers.

### Cross-Segment Persistence

Between segments, carry forward the last few temporal slices as causal
padding for the next segment. The "persistent memory" is the temporal
context that flows from one segment to the next through the slow features.

With dilated temporal kernels [1, 2, 4, 8, 16, 32, 64], 7 layers covers
the full 128-token temporal range per segment.

## Key Properties

| Property | How it's achieved |
|----------|-------------------|
| Fully differentiable | Standard gather + MLP, full autograd |
| Parallel over T | All spacetime nodes processed simultaneously per layer |
| Inference-time adaptation | Slow features evolve through forward pass, modulate routing |
| Bounded VRAM | [BS, 32, 32, 128, D] per layer, checkpointable |
| Signal traverses graph | Write/read port separation, no shortcuts |
| Per-neuron specialization | Positional embedding on grid + per-neuron routing features |
| Persistent memory | Temporal context carries across segments via slow features |
| Dynamic routing | Routing from neuron's own features, not content-based attention |
| No sequential loop | Time is a parallel dimension, not a loop variable |

## What This Is NOT

- **Not a transformer**: No attention, sparse local connectivity, most neurons
  have no direct I/O
- **Not an SSM/Mamba**: 2D spatial structure + inter-neuron communication,
  not 1D state space
- **Not a standard CNN**: Per-neuron dynamic routing (not fixed shared kernels),
  no spatial dimension reduction
- **Not a GNN**: Explicit temporal dimension with causal structure, grid topology

The closest analogy: a **spatially-structured persistent memory processor**
with sparse I/O — a neural cellular automaton with learned dynamic routing
and multi-timescale features.

## What's Genuinely Novel

1. **Sparse I/O into large internal grid**: Only C positions receive input.
   ~1000 neurons are pure internal computation/memory. Unlike transformers/SSMs
   where every position gets a token.

2. **Unified spatial-temporal routing**: Same mechanism for inter-neuron
   communication and temporal persistence. "Memory" = strong self-routing.
   "Communication" = strong neighbor-routing. Same sigmoid weights.

3. **Plasticity through slow features**: No separate modulator, no eligibility
   traces, no special mechanism. Just features at different timescales within
   the same forward pass.

## Open Design Questions

1. **Grid size vs feature dim**: 32×32 grid with D=64 vs 16×16 grid with D=256?
   Tradeoff: more neurons (spatial capacity) vs richer per-neuron features.

2. **Temporal kernel dilation pattern**: Fixed dilation schedule [1,2,4,...] vs
   learned? How many layers needed for full temporal coverage?

3. **Encoder/decoder complexity**: Thin projection vs full scan layers?
   The grid should do most of the work, but the decoder may need some
   sequence modeling capacity.

4. **Long-range spatial connections**: Grid locality limits direct reach.
   Do we need learnable long-range connections (bilinear-interpolated grid
   positions as slow features)?

5. **Training stability**: Gradient flow through L layers × (gather + MLP).
   Need to verify gradients don't vanish/explode.

6. **Parameter budget allocation**: How much goes to grid layers vs
   encoder/decoder? Target ~100M total.

7. **Throughput target**: Can we hit ≥30K tok/s on RTX 4090 at BS=8?

## Comparison to Previous Versions

| | v8 (RL) | v9-ES | v10 (this) |
|---|---|---|---|
| Memory training | GRPO (failed) | ES (failed) | Backprop (unified) |
| Parallelism | Sequential loop | Sequential loop | Fully parallel |
| Memory role | Add-on to LM scan | Add-on to LM scan | IS the model |
| Internal neurons | Dead (msg_mag 0.03) | Dead (msg_mag 0.004) | TBD |
| Throughput | 47-64K tok/s | 2K tok/s | Target ≥30K |
| VRAM | Bounded (no grad) | Bounded (no grad) | Standard autograd |
| Inference adaptation | Modulator (untrained) | Modulator (untrained) | Slow features |
