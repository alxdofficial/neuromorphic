# Neuromorphic Memory Graph — Design Document

## Goals

1. **Long-context seq2seq performance**: augment a recurrent LM with a structured memory that captures long-range dependencies better than the recurrence alone.
2. **Inherent information distillation**: the memory graph compresses and organizes information spatially across neurons, rather than storing raw token representations.
3. **Bounded memory and compute**: the graph has a fixed number of neurons and connections. Memory footprint and per-token compute are constant regardless of context length.
4. **Context rot mitigation**: structural plasticity and neuromodulation continuously reorganize memory, preventing stale information from degrading performance over long sequences.
5. **Lifelong learning**: the neuromodulator is designed as a policy that adapts the graph's connectivity, decay, and identity vectors online. Currently trained by backprop; future work will train it via GRPO RL for continual adaptation beyond a single training run.

## Architecture Overview

```
tokens → embedding → lower scan → split_mlp(H_mid, PCM_surprise) → H_aug
                                                                      │
                                              ┌───────────────────────┤
                                              ▼                       ▼
                                        memory graph            upper scan
                                              │                       │
                                              └──► combine ──────────►│
                                                                      ▼
                                                                  LM head
```

The LM is an affine recurrent model (element-wise linear scan with gating: `h_t = sigmoid(a_t) * h_{t-1} + b_t`). The scan stack is split at a midpoint into lower and upper layers. Between them, a Predictive Coding Module (PCM) computes surprise, which is fused with H_mid via a small MLP to produce H_aug. H_aug feeds both the memory graph and the upper scan.

## LM Scan

- **Type**: causal linear recurrence with element-wise gating
- **Layers**: L total, split at layer S (layers 0..S-1 = lower, S..L-1 = upper)
- **Per layer**: pre-norm (RMSNorm) → linear projections → scan → SwiGLU output → residual
- **Carry**: hidden state h passed across segments (TBPTT with detach at segment boundaries)

## Predictive Coding Module (PCM)

Predicts state transitions (deltas) rather than raw states:
- `delta_hat[t] = pred_MLP(norm(H[t]))` — predicted change
- `delta_actual[t] = H[t+1] - H[t]` — actual change
- `surprise[t] = delta_hat[t-1] - delta_actual[t]` — transition prediction error

The surprise signal is combined with H_mid through a split-point MLP:
- `H_aug = split_mlp(H_mid, surprise)` — produces augmented hidden state
- H_aug is passed to both the memory graph (as inject signal) and the upper scan
- Surprise does not enter the upper scan as a separate side input

## Memory Graph

### Dimensions

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Neuron hidden dim | D_neuron | 32 |
| Total neurons | N | 8096 |
| Connections per neuron | K | 64 |
| Port multiplier | alpha | 4 |
| LM hidden dim | D | 2048 |

### Port Neurons

Port neurons are the interface between the LM and the memory graph.

- **Input port count**: D * alpha / D_neuron = 2048 * 4 / 32 = 256
- **Output port count**: 256 (separate neurons from input ports)
- **Total port neurons**: 512
- **Internal neurons**: 8096 - 512 = 7584

**Injection** (LM → graph): H_aug [BS, D] is reshaped to [BS, C_mem, D_neuron] where C_mem = D / D_neuron = 64. Each slice is replicated alpha=4 times across the input port neurons in that slice group (n_per_slice = alpha = 4 neurons per slice). Parameter-free — just reshape and expand. The inject signal is added to the received presynaptic messages for input port neurons.

**Readout** (graph → LM): output port neurons emit D_neuron-dim messages. Within each slice group, messages are summed over the alpha replicas and scaled by `1/sqrt(n_per_slice)`: `readout = sum(msgs, dim=replica) * n_per_slice^(-0.5)`. The slice outputs are concatenated back to [BS, D].

**Combination**: `H_enriched = H_aug + mem_scale * readout`, where `mem_scale` is a learnable per-dim scale [D], initialized to `sqrt(n_per_slice)` to cancel the `1/sqrt` in readout (starting near unit magnitude so gradients flow freely from the start).

### Neuron State

Each neuron maintains:
- **h**: hidden state vector [D_neuron] — the neuron's internal representation
- **w_conn**: connection weights [K] — scalar weights for each presynaptic neighbor
- **identity**: learnable embedding [D_neuron] — distinguishes this neuron, updated by modulator
- **decay**: learnable scalar — controls state persistence (`h = decay * h_old + (1-decay) * update`)

### Connectivity

- Each neuron has K=64 presynaptic connections (neurons it receives messages from)
- Connections are directed: A listing B as presynaptic does not imply B lists A
- Initialized randomly, adapted by structural plasticity

### Per-Token Neuron Step

One neuron step executes per token, fully sequential (no multi-pass approximation):

1. **Modulate** (step 0): per-neuron neuromodulator MLP predicts updated w_conn, decay, and identity from current (identity, hebbian_trace, w_conn, decay). Happens first so its effects are visible to backprop gradients.

2. **Receive**: gather messages from K presynaptic neighbors, scale each by its w_conn scalar, combine (sum or weighted sum) into a received vector [D_neuron].

3. **Inject**: input port neurons additionally receive their slice of the LM signal. The inject signal is added to the received message.

4. **State update**: shared state MLP takes (received + injected, h_current) → candidate. New state: `h = decay * h_old + (1 - decay) * tanh(candidate)`.

5. **Emit message**: shared message MLP takes (h_new, identity) → outgoing message [D_neuron]. This message is what presynaptic-neighbor lookups will gather on the next step.

6. **Readout**: output port neurons' messages are collected and reassembled into a D-dim vector for the LM.

7. **Update hebbian traces**: running correlation between received messages from each presynaptic neighbor and this neuron's own outgoing messages.

### MLPs

- **State MLP**: shared weights across all neurons. Identity embedding provides per-neuron differentiation.
- **Message MLP**: shared weights across all neurons. Same rationale.
- **Neuromodulator MLP**: **per-neuron weights** (8096 separate small MLPs). This is where model capacity lives. Hidden dim = D_neuron = 32. Input: (identity [32], hebbian_trace [K], w_conn [K], decay [1]) → output: (new_w_conn [K], new_decay [1], new_identity [32]).

### Structural Plasticity

Runs once every 1024 tokens (8 segments). Uses hebbian traces as the signal:

**Pruning** (existing connections): hebbian traces [BS, N, K] are averaged
across the batch dimension to get a per-connection score. The bottom
`plasticity_pct` (2%) of all connections globally are pruned.

**Regrowth**: for each pruned slot, a new target is selected:
- ~80% exploit: the candidate non-connected neuron with the highest estimated
  affinity. Affinity for a non-connected pair (i, j) is estimated from the
  average outgoing message magnitudes of both neurons (a lightweight proxy
  that avoids materializing an [N, N] correlation matrix).
- ~20% explore: random non-connected, non-self neuron.

After rewiring, the `conn_idx` buffer is updated, and hebbian traces and
w_conn for rewired connections are reset to zero. A dedup + re-sort pass
ensures the K-distinct-neighbors invariant is maintained.

### Hebbian Traces

Per-neuron running statistics tracking how messages from each of K presynaptic
neighbors correlate with this neuron's own outgoing messages. Updated every
token step (EMA with decay 0.995). Used by:
- The neuromodulator (as input — sees fresh traces each token step)
- Structural plasticity (batch-averaged traces used for pruning decisions)

## Segments and Timing

- **Segment**: 128 tokens. One segment = one forward_segment call. TBPTT detaches
  at segment boundaries.
- **Neuron step**: one per token (128 per segment). All N neurons processed in
  parallel via vectorized ops.
- **Neuromodulator**: runs every token step. Sees the most recent hebbian traces
  (updated at the end of each token step).
- **Structural plasticity**: runs once every 1024 tokens (8 segments). Rewires
  connections based on hebbian trace statistics.

## Gradient Path (Design Decision)

The memory graph receives `H_aug.detach()` — the LM hidden state with surprise
mixed in, but with the gradient path cut. This is a deliberate choice:

- **Lower scan** learns purely from CE loss through the upper scan path.
- **Memory graph** learns to produce useful readout given whatever H_aug it
  receives, trained through `mem_scale * mem_out` flowing back from CE loss.
- **Rationale**: in v8/v9, coupling memory gradients into the lower scan caused
  optimization instability (memory gradients fighting scan gradients). Decoupling
  gives each system a cleaner learning signal.

The memory graph DOES receive gradient signal — it flows through:
```
CE loss → upper scan → H_enriched = H_aug + mem_scale * mem_out
                                                   │
                                          mem_scale and mem_out carry grad
                                          to all memory parameters
```

## Training

- Single optimizer for all parameters
- Memory graph parameters at reduced LR (e.g., 0.3x base LR)
- Memory parameters kept in f32 (small gradients round to zero in bf16)
- TBPTT within segments, detach at segment boundaries
- Recurrent carries (scan hidden states + neuron hidden states) passed across segments
- Gradient checkpointing over the 128-step token loop (required for BS > ~8)

## Open Questions

- GRPO RL training for neuromodulator (future work — architecture supports it)
