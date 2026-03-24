# Design Notes: Energy Conservation + Autonomous Plasticity

## Changes Summary

### 1. Connection Weight Routing (Historical -> Key-Based Softmax)

**Historical**: conn_weights were L1-normalized per neuron after every `apply_actions`,
giving each neuron a fixed routing budget of 1.0. The neuromod controlled the
*distribution* of routing, not the total magnitude.

**Current (replaced)**: conn_weights have been replaced by **key-based softmax routing**.
Each neuron has a `key` vector (L2-normalized, 128 dims). Routing weights are computed
as `softmax(cosine_sim(key, neighbor_messages))` over K=96 neighbors, once per segment.
Weights sum to 1 by construction (softmax). The neuromod controls routing selectivity
via `delta_key` — adjusting what each neuron "listens for" in its neighbors' messages.
This is content-based attention over presynaptic neighbors rather than fixed weight
distribution.

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

**Regrown weight**: Initialize to median of that neuron's current |weight| distribution
(matches existing scale, no magic number).

### 4. Other Magic Number Fixes

| Old | New | Rationale |
|-----|-----|-----------|
| `usage_count: norm > 0.01` | Firing rate from binary threshold | Relative to neuron's own stats |
| `max_action_magnitude = 0.3` | `1.0` | RMS/L2 normalization bounds the effect |
| `regrown weight = rand * 0.2` | `median(neuron_abs_weights)` | Matches current scale |
| `prune_threshold = 0.01` | `phi < 0` | Relative, biologically meaningful |
| `mean_norm in adjacency` | Removed (softmax handles it) | No double normalization |

### 5. Decay — Left Alone For Now

Decay collapsed to 0.012 (neurons nearly stateless). This might be correct —
the LM scan already provides temporal memory, and the graph's value may be in
spatial routing rather than temporal storage. If other fixes change the picture,
we can revisit.

### 6. Separation of Concerns

**Neuromod controls**: primitives (what neurons say), routing keys (what neurons
listen for), decay (temporal persistence). These are fast, per-segment adjustments.

**Plasticity controls**: which connections exist (topology). This is slow,
autonomous, driven by observed co-activation patterns. The neuromod doesn't
directly control topology — it controls routing strength, and weak/anti-correlated
connections get pruned automatically.

### 7. State Tensors

- `co_activation_ema` [N, N] — batch-averaged phi coefficient matrix (NOT per-batch,
  topology is shared across batch)
- `firing_rate` [BS, N] — EMA of per-neuron firing rate (replaces usage_count in obs)
- `mean_input` [BS, N, D_mem] — per-segment mean of received signals
- `mean_output` [BS, N, D_mem] — per-segment mean of outgoing messages

Note: `activation_ema` and `activation_std_ema` have been removed — the percentile-based
firing threshold no longer needs per-neuron EMA tracking.

### 8. obs_dim Impact

obs_dim is now D_mem*4 + 2 = 514:
- primitive[128] + key[128] + mean_input[128] + mean_output[128] + firing_rate[1] + decay[1]

Previous value was D_mem*3 + 3 = 387 (before key was added and routing_entropy removed).
Neuromod input layer resized accordingly.
