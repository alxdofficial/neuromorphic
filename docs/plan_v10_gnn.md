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
                    Shared MLPs conditioned on neuron identity
                          |
                    Collect word_states at every step
                    word_states: [T, 64 words, D_scan]
                          |
                    Upper Decoder (frontal cortex)
                    Runs ONCE after full segment simulation
                    All T tokens in parallel (standard causal training)
                    Self-attention (causal mask)
                    Cross-attention to word_states (causal sliding window, W=16)
                    → logits [BS, T, vocab]
```

The memory graph is the ONLY path from input to output. No H_mid skip connection
to the decoder. Short-term context comes from the sliding window of word_states
(last W=16 timesteps visible per token).

---

## Per-Token Neuron Step (Sequential)

Each token t triggers one simulation step. All neurons execute simultaneously
(parallel matrix ops via shared weights). Messages from the CURRENT step's
neighbors are used (true sequential, no frozen multi-pass).

### Step 1: MODULATE (shared modulator MLP)

Runs first so its effects on connection weights, decay, and identity are
visible to the loss through the subsequent gather/update/message ops.

```
Input:  cat(hebbian[n], h[n], identity[n])     — [N, K + D + D_id]
Output: w_conn[n, K], decay[n, 1], identity_delta[n, D_id]  — [N, K + 1 + D_id]

identity[n] += identity_delta[n]
```

The modulator adjusts:
- **w_conn**: scalar connection strengths (sigmoid → [0,1] for gather weighting)
- **decay**: per-neuron persistence rate (sigmoid → [0,1] for leaky integration)
- **identity**: the neuron's learned embedding (evolves during a segment)

The modulator does NOT write to h — state only changes through the receive→update path.

### Step 2: GATHER (sparse, weighted)

```
neighbor_msgs = messages[conn_indices[n]]       — [N, K, D]
received[n] = sum_k(sigmoid(w_conn[n,k]) * neighbor_msgs[k])  — [N, D]
```

Sparse index lookup + weighted sum via PyG's `propagate()`. O(N×K×D).

### Step 3: UPDATE STATE (shared state MLP + structural decay)

```
Input:  cat(received[n], inject[n,t], h[n], identity[n])  — [N, 2D + D + D_id]
Output: update[n]                                          — [N, D]

update = tanh(shared_state_MLP(input))
h[n] = decay[n] * h[n] + (1 - decay[n]) * update
```

Structural decay (leaky integration) bounds h naturally. The modulator
controls persistence per-neuron through decay.

### Step 4: PRODUCE MESSAGE (shared message MLP)

```
Input:  cat(h[n], identity[n])    — [N, D + D_id]
Output: msg[n]                     — [N, D]

msg[n] = tanh(shared_msg_MLP(input))
```

### Step 5: COLLECT WORD STATE (on autograd graph)

Group neuron states into words and store for the decoder:

```
word_states[t] = h.view(num_words, neurons_per_word * D)  — [64, 2048]
```

All T word_states are collected on the autograd graph. The decoder receives
`word_states: [T, 64, D_scan]` and cross-attends with causal sliding window.

---

## Upper Decoder (Frontal Cortex)

Runs ONCE after the memory graph simulates all T steps. Processes all T
tokens in parallel using standard causal training.

### Input

- **Queries**: token position embeddings or a learned sequence [T, D_dec]
- **KV for cross-attention**: word_states [T, 64, D_scan] with causal sliding window mask

### Causal Sliding Window Cross-Attention

Token t can only attend to word_states from steps `[max(0, t-W+1), ..., t]`.
This is implemented as a mask on the cross-attention:

```
mask[t, s] = True   if max(0, t-W+1) <= s <= t
             False  otherwise
```

W=16 means each token sees the last 16 timesteps of memory word evolution.
This preserves causality (no future information leaks) and captures temporal
patterns in neuron activity.

### Architecture

```
For each layer (L_decoder layers):
  RMSNorm → self_attention(causal)          — attend to previous decoder states
  RMSNorm → cross_attention(sliding_window) — attend to word_states
  RMSNorm → FFN                             — feed-forward processing
```

### Output

Final layer → RMSNorm → linear projection → logits [BS, T, vocab_size]

No separate lm_head — the decoder IS the language model head.

### Cross-Chunk Context

At the start of a new chunk, the decoder has no self-attention history
(carries are not stored for the decoder). However, the memory graph's h
and messages persist across chunks, and the first W=16 word_states in
the new chunk reflect memory from the previous chunk's final states.

Optionally: carry the last W=16 word_states from the previous chunk
as prefix KV for the decoder's cross-attention. This gives the decoder
immediate access to previous-chunk memory at token 0. Cost: [W, 64, D_scan]
≈ 4 MB, fixed.

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
the decoder directly cross-attends to word_states with a sliding window.
This eliminates readout scaling / gradient attenuation issues from v9.

---

## Structural Plasticity

Same as v9-backprop:
- Phi correlation: Pearson coefficient from binary firing patterns (75th percentile)
- EMA-smoothed co-activation matrix [N, N]
- Global percentile prune/grow: bottom 2% pruned, top 2% created
- 20% random exploration
- Non-differentiable, runs at chunk boundaries
- Hebbian traces: per-connection |msg| × sigmoid(w_conn), fed to modulator

---

## Memory Budget

### Per-segment tensors (recomputed each segment, not stored across chunks)

```
word_states: [T, 64, D_scan] = [128, 64, 2048] × bf16    ≈ 32 MB
decoder KV cache: [L_dec, T, D_dec] × bf16                ≈ small
```

### Persistent state (carried across chunks)

```
h:            [BS, N, D]     = [48, 4096, 32] × bf16      ≈ 12 MB
messages:     [BS, N, D]     = [48, 4096, 32] × bf16      ≈ 12 MB
identity:     [N, D_id]      = [4096, 32] × f32           ≈ 0.5 MB
w_conn:       [BS, N, K]     = [48, 4096, 32] × bf16      ≈ 12 MB
hebbian:      [BS, N, K]     = [48, 4096, 32] × bf16      ≈ 12 MB
co_act_ema:   [N, N]         = [4096, 4096] × f32         ≈ 64 MB
cross-chunk word carry: [W, 64, D_scan] × bf16            ≈ 4 MB
```

Total persistent: ~117 MB (fixed, does not grow with training)

---

## Gradient Flow Analysis

### Gradient path (improved vs v9)

```
loss → decoder logits
  → decoder cross_attention → word_states[t] (on graph)
  → h[n] at step t (grouped into words)
  → state_MLP(received, inject, h_prev, identity)   — shared, large matmul
  → received → gather(msgs, w_conn_sig)              — sparse scatter backward
  → w_conn → modulator(hebbian, h, identity)         — shared modulator
```

### Key improvements over v9:
- **No readout averaging** — cross-attention attends to word patterns directly
- **No mem_scale/mem_mlp bottleneck** — cross-attention provides direct gradient to h
- **Shared MLPs = large matmuls** — GPU-efficient, better gradient flow
- **Every timestep gets gradient** — decoder attends to all T word_states, each step's h gets gradient through its word_state contribution
- **Sequential (not frozen 2-pass)** — simpler gradient graph

### Gradient concerns:
- **Gather backward dot product** — same high-dim orthogonality issue as v9 for w_conn gradient. Mitigated by larger shared MLPs.
- **128 sequential steps** — shared weights get summed gradient from all T steps (standard RNN behavior). No vanishing through the chain since each step's word_state provides independent gradient.
- **tanh saturation** — monitor during training. Xavier init with tanh gain.

---

## Normalization and Magnitude Balancing

### Inject magnitude
- H_mid ≈ 0.46/elem → sliced to D=32 chunks, same scale per element
- Received from neighbors: tanh-bounded msgs (≈0.1-0.3) × sigmoid w_conn (≈0.5) → smaller
- Inject dominates initially — neurons driven by LM signal, inter-neuron communication grows

### Neuron state magnitude
- tanh bounds each element to [-1, 1]
- With structural decay: h stays within [-1, 1] (convex combination)
- D=32 → max norm = sqrt(32) ≈ 5.66

### Word magnitude
- Each word = concat of 64 neurons × 32 dims = 2048-dim vector
- RMS ≈ 0.3/elem, norm ≈ 0.3 × sqrt(2048) ≈ 13.6
- Decoder's RMSNorm before cross-attention handles scale

### Decoder normalization
- Pre-norm (RMSNorm before each attention/FFN), standard transformer practice
- Cross-attention uses 1/sqrt(d_k) scaling

### Identity initialization
- randn × 1/sqrt(D_id) → ≈0.18/elem
- Small enough not to dominate, large enough to differentiate neurons

---

## Parameter Budget (Target: ~110M)

```
Lower Scan:                                    ~27M
  Embedding (32K × 768):                       24.6M
  proj_up (768 → D_scan):                      ~1.6M
  2 scan layers (d_inner TBD):                  TBD

Memory Graph:                                  ~40M
  Shared state MLP (large H):                   TBD
  Shared message MLP:                           TBD
  Shared modulator MLP:                         TBD
  Neuron identity [4096, 32]:                   0.13M
  PCM:                                          ~1M

Upper Decoder:                                 ~43M
  L layers of:
    Self-attention (causal):                    TBD
    Cross-attention (sliding window):           TBD
    FFN:                                        TBD
  Final projection → vocab (32K):               TBD

Total:                                         ~110M
```

Memory graph is ~40M — roughly equal to lower scan + decoder combined.
Shared MLPs can be sized flexibly. Decoder should be relatively small
(2-4 layers) since the memory graph does the heavy processing.

---

## Training Strategy

### Phase 1: Backprop (current)

Standard next-token prediction. Backprop through:
- Decoder → word_states → current step neuron states → shared MLPs → modulator
- PCM aux_loss (transition prediction)

word_states are on the autograd graph (gradient flows to each step's h).
TBPTT at segment boundaries (detach h, messages, identity).

### Phase 2: GRPO (future)

1. Forward pass with no_grad — sample trajectories
2. Score trajectories (reward = -loss or similar)
3. Retrace forward with grad
4. GRPO policy gradient update on modulator / plasticity decisions

---

## Key Design Principles

1. **Memory is patterns, not states** — information lives in which neurons
   co-activate and how activity evolves. The sliding window cross-attention
   captures temporal trajectories across word_states.

2. **No skip connection around memory** — forces the model to use memory.
   Short-term context comes from the sliding window, not an H_mid bypass.

3. **Shared weights, many neurons** — GNN-style parameter efficiency.
   Scale by adding neurons (N) or making shared MLPs bigger (H).

4. **Sequential simulation** — true message propagation, one step per token.
   Information travels one hop per step, 128 hops per segment.

5. **Decoder runs ONCE** — all T tokens in parallel with causal masking.
   Cross-attention to word_states with sliding window preserves causality.
   No per-token decoder calls, no perceiver.

6. **Two-timescale learning** — backprop trains shared weights (fast),
   phi-correlation plasticity rewires topology (slow).

7. **Sensory → Memory → Frontal** — biologically-inspired hierarchy.
   Lower scan (sensory processing), memory graph (hippocampal-like),
   decoder (frontal cortex / decision making).

---

## Implementation: PyTorch Geometric

Using PyG (torch-geometric 2.7.0) for message passing. Optimized sparse
scatter/gather kernels that scale to large N without custom Triton.

### NeuronLayer (extends MessagePassing)

```python
class NeuronLayer(MessagePassing):
    def __init__(self, D, D_id, K, H_state, H_msg, H_mod):
        super().__init__(aggr='add')
        self.state_mlp = MLP(3*D + D_id, H_state, D)
        self.msg_mlp = MLP(D + D_id, H_msg, D)
        self.mod_mlp = MLP(K + D + D_id, H_mod, K + 1 + D_id)

    def forward(self, h, msgs, inject, identity, edge_index, w_conn, hebbian):
        # 1. Modulate
        w_conn, decay, id_delta = self.modulate(h, identity, hebbian, w_conn)
        identity = identity + id_delta
        # 2. Gather
        received = self.propagate(edge_index, x=msgs, edge_weight=sigmoid(w_conn))
        # 3. Update state
        update = tanh(self.state_mlp(cat(received, inject, h, identity)))
        h = decay * h + (1 - decay) * update
        # 4. Message
        msgs = tanh(self.msg_mlp(cat(h, identity)))
        return h, msgs, w_conn, decay, identity
```

### Edge Index Format

```python
# conn_indices [N, K] → edge_index [2, N*K] (COO format)
src = conn_indices.reshape(-1)
tgt = torch.arange(N).unsqueeze(1).expand(-1, K).reshape(-1)
edge_index = torch.stack([src, tgt])
```

### Plasticity Rewiring

When conn_indices changes, rebuild edge_index. O(N*K), once per chunk.

---

## Scratchpad Format

All design scratchpads, notes, and debugging logs MUST use:
- Point-form (bullet points)
- Markdown formatting
- No prose paragraphs in scratchpads
