# Design Notes: Energy Conservation + Autonomous Plasticity

## Changes Summary

### 1. Connection Weight Routing (Historical -> Key-Based Sigmoid)

**Historical**: conn_weights were L1-normalized per neuron after every `apply_actions`,
giving each neuron a fixed routing budget of 1.0. The neuromod controlled the
*distribution* of routing, not the total magnitude.

**Current (replaced)**: conn_weights have been replaced by **key-based sigmoid routing**.
Each neuron has a `key` vector (128 dims). Routing weights are computed
as `sigmoid(key . neighbor_messages)` (raw dot product, not cosine similarity) over K=96
neighbors, once per segment. Each connection is independently gated [0, 1] — strong
connections don't suppress weak ones (unlike softmax which diluted signal across all 96
connections). Keys are updated via gated Hebbian plasticity: eligibility traces accumulate
mean_input, the neuromod gate controls consolidation direction, and keys are L2-normalized
after each update. This is content-based gating over presynaptic neighbors rather than
fixed weight distribution.

### 2. Binary Firing + Percentile Threshold

**Problem**: usage_count used magic `norm > 0.01` threshold.

**Previous fix**: Per-neuron adaptive threshold using activation_ema + activation_std_ema.

**Current**: Percentile-based firing threshold (75th percentile within each segment).

```
activation = prev_messages.norm(dim=-1)                    # scalar per neuron per token
threshold = activation.quantile(0.75, dim=time_dim)        # 75th percentile
fired = activation > threshold                              # binary
```

- No per-neuron EMA tracking needed — threshold computed from the segment's own data
- "Firing" = activation in the top 25% for that neuron within the segment
- Roughly 25% firing rate by construction
- Simpler than the EMA approach — no activation_ema or activation_std_ema state

### 3. Co-Activation-Based Structural Plasticity

**Problem**: Pruning based on absolute weight threshold (0.01). Never fires because
weights grow large. Random regrowth with no information.

**Fix**: Autonomous plasticity driven by temporal co-activation patterns.

**Measurement**: During per-token loop, record binary firing trace [BS, T_seg, N].
At segment end, compute phi coefficient (binary Pearson correlation) between all
neuron pairs:

```
phi[i,j] = (p(i AND j fire) - p(i fires)*p(j fires)) /
           sqrt(p(i)*(1-p(i)) * p(j)*(1-p(j)))
```

One bmm per segment: [BS, N, T_seg] @ [BS, T_seg, N] → [BS, N, N]. ~0.5 GFLOP.
Maintain EMA: `co_activation_ema = decay * co_activation_ema + (1-decay) * phi`

**Pruning**: Per neuron, prune existing connections where phi < 0 (anti-correlated —
neurons activate at opposite times, connection is counterproductive). Natural
threshold at 0, not a magic number.

**Growth**: Per neuron, form connection to non-connected neuron with highest phi > 0.
80% correlation-guided, 20% random (exploration).

**Regrown connections**: Topology-only rewiring — new connections have no persistent edge
weights. Routing is entirely determined by key-based sigmoid gating (no explicit edge weights).

### 4. Other Magic Number Fixes

| Old | New | Rationale |
|-----|-----|-----------|
| `usage_count: norm > 0.01` | Firing rate from binary threshold | Relative to neuron's own stats |
| `max_action_magnitude = 0.3` | Removed (gate is tanh'd, decay is blended) | Neuromod outputs 2-dim action, not 257-dim |
| `regrown weight = rand * 0.2` | Topology-only (no edge weights) | Routing via key-based sigmoid |
| `prune_threshold = 0.01` | `phi < 0` | Relative, biologically meaningful |
| `mean_norm in adjacency` | Removed (sigmoid handles it) | No double normalization |

### 5. Decay — Left Alone For Now

Decay collapsed to 0.012 (neurons nearly stateless). This might be correct —
the LM scan already provides temporal memory, and the graph's value may be in
spatial routing rather than temporal storage. If other fixes change the picture,
we can revisit.

### 6. Separation of Concerns

**Neuromod controls**: gate (scalar, tanh'd to [-1,1] — controls Hebbian trace
consolidation direction) and decay_target (blended into decay_logit). Primitives
and keys are updated indirectly via gated Hebbian plasticity: eligibility traces
accumulate what neurons encode (trace_prim) and receive (trace_key), and the gate
controls whether to consolidate or reverse. These are fast, per-segment adjustments.

**Plasticity controls**: which connections exist (topology). This is slow,
autonomous, driven by observed co-activation patterns. The neuromod doesn't
directly control topology — topology rewiring is autonomous based on co-activation
statistics. Anti-correlated connections get pruned automatically.

### 7. State Tensors

- `co_activation_ema` [N, N] — batch-averaged phi coefficient matrix (NOT per-batch,
  topology is shared across batch)
- `msg_magnitude` [BS, N] — EMA of mean message norm per neuron (replaces firing_rate in obs)
- `mean_input` [BS, N, D_mem] — per-segment mean of received signals
- `mean_output` [BS, N, D_mem] — per-segment mean of outgoing messages

Note: `activation_ema` and `activation_std_ema` have been removed — the percentile-based
firing threshold no longer needs per-neuron EMA tracking.

### 8. obs_dim and act_dim Impact

obs_dim is now D_mem*4 + 4 = 516:
- primitive[128] + key[128] + mean_input[128] + mean_output[128] + msg_magnitude[1] + decay[1] + trace_prim_norm[1] + trace_key_norm[1]

act_dim is now 2:
- gate[1] (tanh'd to [-1,1]) + decay_target[1]

Previous obs_dim was 514 (before trace norms were added).
Previous act_dim was 257 (delta_primitive[128] + delta_key[128] + delta_decay[1]),
replaced by 2-dim action with gated Hebbian plasticity.
Neuromod backbone reduced from 3-layer/2048 hidden to 2-layer/512 hidden.
