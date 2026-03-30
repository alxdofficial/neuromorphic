# v10-gnn: Shared-Weight GNN Memory Graph

> **Branch:** `v10-gnn` (from `v9-backprop`)
> **Date:** 2026-03-30
> **Status:** Design phase

## Motivation

v9-backprop demonstrated that the memory graph survives training (no explosions,
gradients flow, neurons differentiate) but provides negligible benefit over an
LM-only baseline with the same parameter count:

| Model | Params | Val Loss | Val PPL |
|-------|--------|----------|---------|
| LM + Memory (v9) | 110M | 5.40 | 221 |
| LM only (52M) | 52M | 5.42 | 227 |
| LM only (110M) | 110M | 5.18 | 178 |

Root cause: 58M params spent on 512 per-neuron MLPs (hidden=24) that are
individually too small to be expressive, collectively too expensive in params,
and computationally inefficient (batched tiny GEMMs instead of one large matmul).

Solution: GNN-style shared weights. One set of MLPs for all neurons, conditioned
on a learnable identity embedding. Neurons differentiate through their inputs
(connectivity, inject signal, state history) and identity — not through
independent weight matrices.

---

## Architecture Overview

```
Tokens → Embedding → Lower Scan (sensory cortex, small, fast)
                          |
                    H_mid [BS, T, D_scan]
                          |
                    split_mlp(H_mid, RMSNorm(surprise))   ← PCM transition prediction
                          |
                    Inject: slice into D_neuron=32 chunks
                    64 slices × 64 neurons/slice = 4096 neurons
                          |
                    Memory Graph (sequential, 1 step/token)
                    N=4096 neurons, D=32, K sparse connections
                    Shared MLPs, rolling window of activations
                          |
                    Group neurons into "words":
                    64 neurons × 32 dims = 2048-dim word
                    4096/64 = 64 words
                          |
                    Perceiver: compress W=16 timesteps → 1 vec/word
                    → 64 memory words [64, D_scan]
                          |
                    Upper Decoder (frontal cortex)
                    Causal cross-attention to 64 memory words
                    Few layers with FFN
                    → logits [BS, 1, vocab]   (one token at a time)
```

The memory graph is the ONLY path from input to output. No H_mid skip connection
to the decoder. This forces the memory to carry all information — short-term
context comes from the rolling window of recent activations, not from a bypass.

---

## Per-Token Neuron Step (Sequential)

Each token t triggers one simulation step. All neurons execute simultaneously
(parallel matrix ops via shared weights). Messages from the CURRENT step's
neighbors are used (true sequential, not frozen multi-pass).

### Step 1: MODULATE (shared modulator MLP)

Runs first so its effects on connection weights and identity are visible
to the loss through the subsequent gather/update/message ops.

```
Input:  cat(hebbian[n], h[n], identity[n])     — [N, K + D + D_id]
Output: w_conn[n, K], decay[n, 1], identity_delta[n, D_id]  — [N, K + 1 + D_id]

identity[n] += identity_delta[n]
```

The modulator adjusts:
- **w_conn**: scalar connection strengths (sigmoid → [0,1] for gather weighting)
- **decay**: per-neuron persistence rate (sigmoid → [0,1] for leaky integration)
- **identity**: the neuron's learned embedding (allows identity to evolve during a segment)

The modulator does NOT write to h — state only changes through the receive→update path.

### Step 2: GATHER (sparse, weighted)

```
neighbor_msgs = messages[conn_indices[n]]       — [N, K, D]
received[n] = sum_k(sigmoid(w_conn[n,k]) * neighbor_msgs[k])  — [N, D]
```

Sparse index lookup + weighted sum. O(N×K×D). Same as v9.

### Step 3: UPDATE STATE (shared state MLP + structural decay)

```
Input:  cat(received[n], inject[n,t], h[n], identity[n])  — [N, 2D + D + D_id]
Output: update[n]                                          — [N, D]

update = tanh(shared_state_MLP(input))
h[n] = decay[n] * h[n] + (1 - decay[n]) * update
```

Structural decay (leaky integration) bounds h naturally. The modulator
controls persistence per-neuron through decay. tanh on the update ensures
the new contribution is bounded.

### Step 4: PRODUCE MESSAGE (shared message MLP)

```
Input:  cat(h[n], identity[n])    — [N, D + D_id]
Output: msg[n]                     — [N, D]

msg[n] = tanh(shared_msg_MLP(input))
```

### Step 5: STORE (detached, circular buffer)

```
rolling_window[n, ptr % W] = h[n].detach()
ptr += 1
```

The window persists across segments. Never reset. W=16 (stores last 16
token steps). Detached so the decoder's gradient doesn't flow backward
through the entire window history — only the current step's computation
is on the autograd graph.

### Step 6: READOUT + STORE for decoder

The current step's readout (on the autograd graph, not detached) is computed
and stored for the decoder to consume:

```
readout[t] = current neuron states grouped into words
```

---

## Rolling Window and Perceiver

### Window Structure

```
rolling_window: [N, W, D]   — circular buffer, W=16
```

At each token t, the window contains the last W neuron state snapshots
(detached). This is the "short-term memory" — recent activation history
that the decoder cross-attends to.

### Grouping into Words

Neurons are grouped into fixed "words" (determined at initialization, never
changed — like physical neuron locations in the brain):

```
N=4096 neurons, D=32, D_scan=2048
Neurons per word: D_scan / D = 64
Number of words: N / 64 = 64

word[w, t] = concat(h[w*64 : (w+1)*64, t])   — [2048]
```

Each word is a 2048-dim vector formed by concatenating 64 neurons' 32-dim states.

### Perceiver Compression

For each word, compress its W=16 timestep history into one vector:

```
word_history[w]: [W, D_scan]     — 16 timesteps of this word
learned_query:   [1, D_scan]     — one learnable query per word (or shared)
output[w] = cross_attention(query=learned_query, kv=word_history[w])  — [D_scan]
```

This produces 64 memory words, each D_scan-dim, summarizing the recent
temporal trajectory of that neuron group.

---

## Upper Decoder (Frontal Cortex)

Causal, autoregressive. Predicts one token at a time.

### Input

At token position t, the decoder receives:
- 64 memory words from the perceiver (each [D_scan])
- Its own previous hidden state / token embedding (for autoregressive context)

### Architecture

```
For each layer (L_decoder layers):
  self_attention(causal)       — attend to previous decoder states
  cross_attention              — attend to 64 memory words
  FFN                          — feed-forward processing
```

### Output

Final layer → linear projection → logits [vocab_size]

No separate lm_head — the decoder IS the language model head. Its job is
to read from memory and predict the next token.

---

## PCM (Predictive Coding Module)

Same as v9-backprop:
- Predicts transitions: delta_hat = MLP(H_mid[t]), surprise = delta_hat - actual_delta
- RMSNorm on surprise before combining with H_mid via split_mlp
- Combined signal becomes the inject to the memory graph
- PCM learns from its own aux_loss (prediction error on transitions)

---

## Inject / I/O

### Inject (LM → Memory)

```
H_combined = H_mid + split_mlp(cat(H_mid, RMSNorm(surprise)))
slices = H_combined.view(BS, T, 64, 32)       — 64 slices of 32 dims
inject[n] = slices[slice_of_neuron_n]          — replicated within each group
```

Same replication scheme as v9. Each group of 64 neurons sees the same
32-dim inject signal.

### No explicit readout to H_mid

The memory graph does NOT feed back into a residual stream. Instead,
the decoder directly cross-attends to the memory words. This eliminates
the readout scaling / gradient attenuation issues from v9.

---

## Structural Plasticity

Same as v9-backprop:
- Phi correlation: Pearson coefficient from binary firing patterns (75th percentile threshold)
- EMA-smoothed co-activation matrix [N, N]
- Global percentile prune/grow: bottom 2% pruned, top 2% created
- 20% random exploration
- Non-differentiable, runs at chunk boundaries
- Hebbian traces: per-connection |msg| × sigmoid(w_conn), fed to modulator

---

## Gradient Flow Analysis

### Why this design has better gradient flow than v9:

**v9 bottleneck chain:**
```
loss → upper_scan → mem_scale → readout(avg) → msgs → MLPs → modulator
                                    ↑
                              19.6× MLP attenuation (removed)
                              0.125× readout scaling
                              high-dim dot product (gather backward)
```

**v10 gradient path:**
```
loss → decoder → cross_attention → memory words → perceiver → neuron states
                                                                    ↑
                                              direct gradient to h[n] at current step
                                              → state_MLP (shared, large)
                                              → received → gather → w_conn → modulator
```

Key improvements:
1. **No readout averaging** — decoder attends to individual words, preserving spatial patterns
2. **No mem_scale/mem_mlp bottleneck** — cross-attention provides direct gradient
3. **Shared MLPs are larger** — bigger hidden dim = better gradient flow per step
4. **Sequential (not frozen 2-pass)** — true message propagation, simpler gradient graph

### Potential gradient concerns:

1. **128 sequential steps** with shared weights — gradient to early steps flows
   through 128 MLP applications. With shared weights, each step's gradient
   contribution to the weights is independent (doesn't need to flow through
   the chain). Same as any RNN with shared weights — the per-step gradient
   sums over all timesteps.

2. **Decoder cross-attention to memory words** — the gradient from the decoder
   reaches the current step's neuron states directly through the perceiver.
   It does NOT flow backward through the rolling window (detached). So the
   gradient signal is fresh each step, not attenuated by history.

3. **Modulator gradient** — still goes through the gather backward
   (high-dimensional dot product issue from v9). But with larger shared MLPs
   and more neurons, the signal is stronger. And the modulator has more
   leverage (4096 neurons to differentiate vs 512).

4. **tanh saturation** — state and message MLPs output tanh. With shared
   weights and proper initialization (Xavier with gain for tanh), saturation
   should be minimal. Monitor during training.

---

## Normalization and Magnitude Balancing

### Inject magnitude

H_mid ≈ 0.46/elem (from v9 measurements). After slicing to D=32, each
inject signal is a 32-dim vector with ≈0.46/elem. The received signal
from neighbors will be smaller (messages are tanh-bounded, ≈0.1-0.3/elem
weighted by sigmoid w_conn ≈ 0.5). So inject dominates initially — same
as v9. This is acceptable: it means neurons are primarily driven by the
LM signal early in training, with inter-neuron communication growing
as the graph learns.

### Neuron state magnitude

tanh bounds each element to [-1, 1]. With D=32, the max norm is sqrt(32)
≈ 5.66. No explosion possible.

### Word magnitude

Each word = concat of 64 neuron states. If each neuron has RMS ≈ 0.3/elem,
the word has RMS ≈ 0.3/elem and norm ≈ 0.3 × sqrt(2048) ≈ 13.6.

### Perceiver output magnitude

The perceiver cross-attention should produce outputs at similar scale to
the word inputs. Standard attention normalization (1/sqrt(d_k)) handles this.

### Decoder normalization

Each decoder layer should use pre-norm (RMSNorm before attention/FFN),
same as standard transformer practice. This keeps activations stable
across layers.

### Identity vector initialization

Xavier init, same scale as neuron_id in v9: randn × 1/sqrt(D_id).
D_id=32 → ≈0.18/elem. Small enough not to dominate, large enough to
differentiate neurons.

---

## Parameter Budget (Target: ~110M)

```
Lower Scan:                                    ~25M
  Embedding (32K × 768):                       24.6M
  proj_up (768 → D_scan):                      TBD (depends on D_scan)
  2 scan layers:                                TBD

Memory Graph:                                  ~45M
  Shared state MLP:                             TBD
  Shared message MLP:                           TBD
  Shared modulator MLP:                         TBD
  Neuron identity [4096, 32]:                   0.13M
  Perceiver (per-word compression):             TBD
  PCM:                                          ~1M

Upper Decoder:                                 ~40M
  L layers of:
    Self-attention (causal):                    TBD
    Cross-attention (to 64 memory words):       TBD
    FFN:                                        TBD
  Final projection → vocab:                     TBD

Total:                                         ~110M
```

Exact param allocation TBD after prototyping. The memory graph shared MLPs
will be small (~2-5M). The bulk of memory-related params go to the perceiver
and decoder cross-attention.

---

## Training Strategy

### Phase 1: Backprop (current)

Standard next-token prediction. Backprop through:
- Decoder → perceiver → current step neuron states → shared MLPs → modulator
- PCM aux_loss (transition prediction)

Rolling window is detached — no gradient through history.
TBPTT at segment boundaries (detach h, messages, identity).

### Phase 2: GRPO (future)

1. Forward pass with no_grad — sample trajectories
2. Score trajectories (reward = -loss or similar)
3. Retrace forward with grad
4. GRPO policy gradient update on modulator / plasticity decisions

This enables the modulator to learn strategic decisions (when to consolidate,
when to explore, what to remember) that can't be captured by per-token
backprop alone.

---

## Key Design Principles

1. **Memory is patterns, not states** — information lives in which neurons
   co-activate and how activity evolves. The rolling window + perceiver
   captures temporal trajectories, not just snapshots.

2. **No skip connection around memory** — forces the model to use memory.
   Short-term context comes from the rolling window, not an H_mid bypass.

3. **Shared weights, many neurons** — GNN-style parameter efficiency.
   Scale by adding neurons, not per-neuron params.

4. **Sequential simulation** — true message propagation, one step per token.
   Information travels one hop per step, 128 hops per segment.

5. **Two-timescale learning** — backprop trains shared weights (fast),
   phi-correlation plasticity rewires topology (slow).

6. **Sensory → Memory → Frontal** — biologically-inspired hierarchy.
   Lower scan (sensory processing), memory graph (hippocampal-like),
   decoder (frontal cortex / decision making).

---

## Implementation: PyTorch Geometric

Using PyG (torch-geometric 2.7.0) for message passing. This gives us
optimized sparse scatter/gather kernels and a clean `MessagePassing` API
that scales to large N without custom Triton kernels.

### NeuronLayer (extends MessagePassing)

```python
class NeuronLayer(MessagePassing):
    def __init__(self, D, D_id, K, H_state, H_msg, H_mod):
        super().__init__(aggr='add')  # sum aggregation

        # Shared MLPs (one set for ALL neurons)
        self.state_mlp = MLP(3*D + D_id, H_state, D)  # received+inject+h+id → update
        self.msg_mlp = MLP(D + D_id, H_msg, D)         # h+id → msg
        self.mod_mlp = MLP(K + D + D_id, H_mod, K + 1 + D_id)  # hebb+h+id → w_conn+decay+id_delta

    def forward(self, h, msgs, inject, identity, edge_index, w_conn, hebbian):
        # 1. Modulate
        w_conn, decay, id_delta = self.modulate(h, identity, hebbian, w_conn)
        identity = identity + id_delta

        # 2. Gather (PyG message passing with edge weights)
        received = self.propagate(edge_index, x=msgs, edge_weight=sigmoid(w_conn))

        # 3. Update state (structural decay)
        update = tanh(self.state_mlp(cat(received, inject, h, identity)))
        h = decay * h + (1 - decay) * update

        # 4. Message
        msgs = tanh(self.msg_mlp(cat(h, identity)))

        return h, msgs, w_conn, decay, identity
```

PyG's `propagate()` handles the sparse gather + weighted sum + scatter
efficiently using CUDA-optimized kernels. No custom Triton needed.

### Edge Index Format

PyG uses COO format: `edge_index [2, num_edges]` where `edge_index[0]`
is source, `edge_index[1]` is target. For our graph:

```python
# Convert conn_indices [N, K] → edge_index [2, N*K]
src = conn_indices.reshape(-1)                                    # [N*K]
tgt = torch.arange(N).unsqueeze(1).expand(-1, K).reshape(-1)     # [N*K]
edge_index = torch.stack([src, tgt])                              # [2, N*K]
```

### Plasticity Rewiring

When `conn_indices` changes (plasticity), we rebuild `edge_index`.
This is O(N*K) and happens once per chunk — negligible cost.

---

## Scratchpad Format

All design scratchpads, notes, and debugging logs MUST use:
- Point-form (bullet points)
- Markdown formatting
- No prose paragraphs in scratchpads
