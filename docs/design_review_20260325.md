# Design Review — 2026-03-25

Deep sweep of the v8 codebase. Line-by-line read of every source file,
followed by discussion of design intent vs actual implementation.

## Issues Found

---

### 1. PCM (Predictive Coding Module) needs rework

**Current implementation** (`src/v8/pcm.py`):
- `z = W_enc(x)` — encodes raw input token embedding only
- `z_hat = W_pcm(H)` — predicts from scan hidden state only
- `surprise = z_hat[t-1] - z[t]`

**Problem**: `z = W_enc(x)` doesn't capture what the model has learned from
context — it's just the raw embedding. The scan hidden state H contains
the model's accumulated understanding. The encoding should reflect
"what the model knows at this position," not just the token identity.

**Intended design**:
- `z_t = W_enc(H_t, x_t)` — encoding is a function of both scan state
  AND input token (what the model "sees" at position t)
- `z_hat_t = W_pcm(H_t, x_t)` — prediction of z_{t+1}, also conditioned
  on both the current hidden state and token
- `surprise_t = z_hat_{t-1} - z_t` — how unexpected this position's
  full representation is, given the previous position's prediction

This is better because the encoding captures the model's contextual
understanding at each position, not just the token identity. The
prediction then anticipates how the model's understanding will change
at the next position. Surprise measures the gap between expectation
and reality in representation space.

**Status**: To be implemented.

---

### 2. Triton path mean_input/mean_output mismatch

**Current issue** (`src/v8/memory_graph.py`):

The Python path correctly accumulates per-neuron received signals and
outgoing messages across all tokens in the segment:
```python
received_accum += received      # what each neuron received from neighbors
msg_accum += prev_msg            # each neuron's outgoing messages
self.mean_input = received_accum / T_seg
self.mean_output = msg_accum / T_seg
```

The Triton path approximates incorrectly:
```python
self.mean_input = (h_start + self.h) * 0.5    # average of h, NOT received
self.mean_output = (msg_start + self.prev_messages) * 0.5
```

`h` (integrated state) and `received` (what the neuron received from
neighbors) are different quantities. The Triton path should track
received_accum and msg_accum properly, or at minimum use a more
accurate approximation.

**Confirmed**: mean_input and mean_output are per-neuron `[BS, N, D_mem]`,
which is correct. The bug is Triton using the wrong quantities.

**Status**: To be fixed.

---

### 3. Routing weight scope — confirmed correct

Each neuron's softmax routing is over its K=96 connected neighbors only:
```python
neighbor_msgs = self.prev_messages[:, self.conn_indices]  # [BS, N, K=96, D]
sim = (key.unsqueeze(2) * neighbor_msgs).sum(dim=-1)      # [BS, N, K]
routing_weights = softmax(sim, dim=-1)                      # [BS, N, K]
```

A neuron never attends to neurons it's not connected to. The weights
sum to 1 across the K connections only. This is the intended design:
each neuron has a fixed set of presynaptic partners, and the routing
weights determine how much it listens to each one within that fixed set.

**Status**: Correct, no change needed.

---

### 4. PCM surprise integration into memory graph — confirmed correct

The flow is:
1. Lower scan -> H_mid
2. PCM applies gain modulation: `H_mid *= sigmoid(W_gain(surprise)) * gain_scale`
3. CC signals = `H_mid.detach()` sliced per CC -> memory graph input

Surprising tokens produce amplified CC signals into the memory graph.
The memory graph never sees the surprise vector directly — it sees its
effect through stronger/weaker CC input amplitude. This is the intended
design: PCM modulates the *amplitude* of the signal entering the
memory graph, not the signal content.

**Status**: Correct, no change needed. (Will improve further with PCM
rework in issue #1 — better surprise means better gain modulation.)

---

### 5. Structural plasticity parameters

**Current settings**:
- `structural_plasticity_every = 4` segments (~twice per chunk of 16 segments)
- `plasticity_exploration_frac = 0.2` (80% correlation-guided, 20% random)
- Prune only connections with phi < 0 (anti-correlated)
- At most 1 connection pruned per neuron per step

**Analysis**:

*Frequency (every 4 segments)*: Possibly too aggressive early in training.
The co-activation EMA has decay=0.995, so it takes ~200 updates to get
reliable phi estimates. With plasticity every 4 segments, that's ~800
segments of rewiring on noisy correlations. More conservative: every 8
or 16 segments (once per chunk), or add a warmup period before plasticity
begins.

*80/20 split*: Reasonable. 20% random exploration prevents getting stuck
in local optima. Could consider annealing: more random early, more
guided later. But 80/20 fixed is fine as a starting point.

*1 connection per neuron per step*: Conservative and good for stability.
With 96 connections, full rewiring takes at least 96 plasticity steps.

*phi < 0 threshold*: Natural boundary (anti-correlation is a clear signal
to prune). No magic numbers. Early on, phi values are near zero
everywhere, so few connections get pruned — which is the right behavior
(don't rewire until you have evidence).

**Decision**: Consider bumping `structural_plasticity_every` from 4 to 8
for stability. Keep 80/20 and phi < 0 threshold. Optionally add a warmup
(no plasticity for first N segments until phi estimates stabilize).

**Status**: Minor tuning, not urgent.

---

### 6. Memory graph reset at doc boundaries — confirmed correct

Reset is **per-batch-element only**:
```python
def reset_streams(self, mask: Tensor):
    # mask: [BS] — True for batch elements hitting a doc boundary
    self.h = self.h * keep2          # only masked batch elements zeroed
    self.prev_messages = self.prev_messages * keep2
    self.firing_rate = self.firing_rate * keep1
```

If batch element 0 hits a doc boundary but element 1 doesn't, only
element 0's dynamic state (h, prev_messages, firing_rate) gets zeroed.
Element 1 continues uninterrupted. Structural state (primitives, key,
decay, connectivity, co_activation) is never reset for any element.

**Status**: Correct, no change needed.

---

### 7. Document boundary frequency in training data

The Pile is heterogeneous — everything from short web snippets (tens of
tokens) to full books (hundreds of thousands of tokens). With T=2048
chunks and BS=8 independent streams reading from different shard
positions, a typical chunk sees 0-3 batch elements hitting a doc
boundary, depending on the local document mix.

Resets are not happening every token — more like every few chunks on
average per stream. Long documents span many chunks without any reset,
which is the intended use case for persistent memory.

The potential concern: documents that are just slightly longer than T
build up useful state that gets lost immediately at the next chunk
boundary. But this is inherent to the data distribution, not a design
flaw.

**Status**: Informational, no action needed.

---

### 8. RL trajectory design — significant mismatch

**Current implementation** (`src/v8/model.py: score_trajectories`):
- K=96 neurons chosen randomly
- Scoring on the **last chunk only** (chunk 4 of 4)
- Non-K neurons get trajectory 0's baseline actions
- 4 chunks of RL data collected, but only the last is scored

**Intended design**:
- K=96 neurons chosen once, **fixed for all 4 chunks**
- All 8 trajectories scored **across all 4 chunks** (total score =
  cumulative CE over the full 4-chunk horizon)
- Non-K neurons get **no actions at all** (zero delta, not baseline)
- Same K neurons throughout the RL step

**Why this is better**:

1. *Longer evaluation horizon*: 4 chunks x 16 segments = 64 neuromod
   actions per trajectory. Memory modifications compound over time.
   Scoring over 4 chunks captures long-range effects that 1 chunk misses.

2. *Cleaner credit assignment*: Zero delta for non-K neurons means any
   loss difference between trajectories is 100% attributable to the K
   neurons' actions. No noise from other neurons' baseline actions.

3. *Consistent K neurons*: Same K neurons throughout means you're
   evaluating a coherent strategy for those neurons, not a patchwork.

**Implementation implications**:
- Choose K neurons at RL time (once per RL update)
- During the 4 collection chunks, run the real forward normally
  (all neurons act as usual)
- At scoring time: replay all 4 chunks x 8 trajectories. In each
  trajectory, only K neurons get stochastic actions, non-K get zero delta
- Score = total CE across all 4 chunks for each trajectory
- More expensive (8 trajectories x 4 chunks instead of 8 x 1), but
  4x more compute for much better signal quality is worth it

**Status**: To be reimplemented.

---

### 9. GRPO mechanism — why replaying above-average trajectories works

For reference, here is why the replay step produces useful gradients:

Say trajectory 3 took action `a = [0.3, -0.1, 0.5, ...]` at segment 7
for neuron 42, and this trajectory got the best CE loss (highest
advantage after z-score normalization).

When we replay:
```python
_, log_prob, entropy, _ = neuromod.get_action_and_value(obs, action=a)
loss = -(advantage * log_prob.mean())
loss.backward()
```

`log_prob(a | obs)` asks: "how likely is the current policy to output
exactly action `a` given observation `obs`?"

The gradient of `-advantage * log_prob` does two things:
1. **Shifts the mean**: The policy's mean output moves toward `a`.
   If `a` was above the current mean, the mean shifts up.
2. **Adjusts the std**: If `a` was far from the mean but got high
   advantage, std may increase (explore that region more). If `a` was
   close to the mean, std may decrease (exploit this good region).

The **relative** aspect is key: by z-scoring across trajectories, we
say "trajectory 3 was better than average, so make its actions more
likely." We don't need absolute quality estimates — just relative
ranking. This removes the need for a learned value function.

Over many RL updates with different K neurons, the policy learns:
"when a neuron's observation looks like X, take action Y, because that
tends to reduce CE loss."

**Status**: Informational, no action needed.

---

### 10. Best trajectory state should persist after RL

**Current implementation**: After GRPO scoring, memory graph is restored
to the real forward pass's state (the state from the actual forward
that produced logits for LM loss).

**Intended design**: After RL, set the memory graph state to the **best
trajectory's final state**. Also keep that trajectory's upper scan
carries. This way the next chunk starts from the best-performing
memory configuration.

**Why this is better**:
- The best trajectory, by definition, produced the lowest CE loss.
  Its memory graph state led to the most useful memory signals.
- Starting the next chunk from that state gives the model a better
  foundation — it's best-of-N selection at the state level.
- Even before the neuromod learns anything useful, the best-of-8
  random action sets get applied as "real" state changes. This is
  a free performance boost from selection pressure alone.
- The neuromod's actions from the best trajectory become the actual
  state progression, creating a natural curriculum.

**Implementation note**: Need to save both memory graph state AND upper
scan carries from the best trajectory during scoring, then apply them
at the end instead of restoring the original state.

**Status**: To be implemented.

---

## Summary of Action Items

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | PCM rework (encode from H+x, predict from H+x) | Design change | DONE |
| 2 | Triton mean_input/mean_output uses wrong quantities | Bug | DONE |
| 3 | Routing weight scope | — | Correct |
| 4 | PCM surprise -> memory graph integration | — | Correct |
| 5 | Structural plasticity frequency | Minor | Consider tuning |
| 6 | Memory graph reset scope | — | Correct |
| 7 | Document boundary frequency | — | Informational |
| 8 | RL: score all 4 chunks, zero non-K neurons, fixed K | Design change | DONE |
| 9 | GRPO mechanism explanation | — | Informational |
| 10 | Best trajectory state persists after RL | Design change | DONE |
| 11 | Remove doc boundary memory graph resets | Design change | DONE |
