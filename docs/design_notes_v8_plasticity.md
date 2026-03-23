# Design Notes: Energy Conservation + Autonomous Plasticity

## Changes Summary

### 1. Energy-Conserving Connection Weights

**Problem**: conn_weights grew unbounded (±16), no normalization on absolute values.

**Fix**: L1-normalize conn_weights per neuron after every `apply_actions`. Each neuron
has a fixed routing budget of 1.0:

```
sum_k |conn_weights[n, k]| = 1.0  for every neuron n
```

The neuromod controls the *distribution* of routing, not the total magnitude.
Strengthening one connection weakens others — like biological synaptic competition.

**Consequence**: Remove mean-normalization from `_build_adjacency` (no longer needed).
The L1 normalization IS the normalization. Adjacency scatters raw L1-normalized weights.

With all-positive equal weights: each = 1/96, received = average of neighbors.
With non-uniform weights: received emphasizes some neighbors, total still bounded.

### 2. Binary Firing + Adaptive Threshold

**Problem**: usage_count used magic `norm > 0.01` threshold.

**Fix**: Per-neuron adaptive firing threshold based on the neuron's own statistics.

```
activation = prev_messages.norm(dim=-1)           # scalar per neuron per token
threshold = activation_ema + activation_std_ema    # one std above mean
fired = activation > threshold                     # binary
```

- `activation_ema`: EMA of activation magnitude (tracks neuron's baseline)
- `activation_std_ema`: EMA of activation std (tracks neuron's variability)
- "Firing" = significantly above that neuron's own average
- No magic numbers — everything relative to neuron's own statistics
- Roughly 15-20% firing rate per neuron (one std above mean)

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
| `max_action_magnitude = 0.3` | `1.0` | L1 normalization bounds the effect |
| `regrown weight = rand * 0.2` | `median(neuron_abs_weights)` | Matches current scale |
| `prune_threshold = 0.01` | `phi < 0` | Relative, biologically meaningful |
| `mean_norm in adjacency` | Removed (L1 handles it) | No double normalization |

### 5. Decay — Left Alone For Now

Decay collapsed to 0.012 (neurons nearly stateless). This might be correct —
the LM scan already provides temporal memory, and the graph's value may be in
spatial routing rather than temporal storage. If other fixes change the picture,
we can revisit.

### 6. Separation of Concerns

**Neuromod controls**: primitives (what neurons say), conn_weights distribution
(routing allocation), decay (temporal persistence). These are fast, per-segment
adjustments.

**Plasticity controls**: which connections exist (topology). This is slow,
autonomous, driven by observed co-activation patterns. The neuromod doesn't
directly control topology — it controls routing strength, and weak/anti-correlated
connections get pruned automatically.

### 7. New State Tensors

- `activation_ema` [BS, N] — running mean of activation magnitude
- `activation_std_ema` [BS, N] — running std of activation magnitude
- `co_activation_ema` [N, N] — batch-averaged phi coefficient matrix (NOT per-batch,
  topology is shared across batch)
- `firing_rate` [BS, N] — EMA of per-neuron firing rate (replaces usage_count in obs)

### 8. obs_dim Impact

Replace `usage_count` with `firing_rate` in obs — same shape [BS, N, 1]. obs_dim
stays at 391. No neuromod architecture change.
