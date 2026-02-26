# Architecture v3: Cortical Columns with Affine Memory Scan

## Problem Statement

We have brain-inspired memory mechanisms (PM, EM, WM, PCM) that are non-affine
(involve thresholds, argmax, conditional slot replacement). We want:

1. Per-token memory interaction (brain-like)
2. Parallel computation across tokens and columns
3. Constant memory footprint (training and inference)
4. Clean gradient flow / credit assignment across the full segment

The fundamental tension: non-affine updates create sequential dependencies.
The key insight: **separate decisions from updates.** Decisions are non-affine
but can be precomputed. Updates, given decisions, are affine in the memory state.

## Architecture Overview

### Cortical Columns

B independent processing units (B=16), each containing:
- **FFN** — local feature transformation (column-specific weights)
- **PCM** — local predictive coding (column-specific predictions + surprise)

Columns do NOT contain RNN or attention. Each column is a simple per-token
processor. No recurrence, no sequential dependency between tokens within a column.

### Shared Memory Systems

Shared across all columns (biologically accurate — hippocampus, basal ganglia,
and prefrontal cortex are centralized systems that cortical columns read/write to):
- **PM** — procedural memory (content-addressed slots)
- **EM** — episodic memory (hippocampal-like episode store)
- **WM** — working memory (short-term buffer, global workspace)

### Two-Phase Forward Pass

```
Phase 1: Column Processing (fully parallel across B columns x N tokens)
  - All tokens read from FROZEN memory snapshot (start of segment)
  - Each column: embed → memory reads → FFN → PCM surprise → logits
  - Produces write candidates for PM/EM/WM, grouped by span
  - Serial depth: O(1) — no dependencies between tokens

Phase 2: Memory State Scan (parallel scan across K spans)
  - Each span's write candidates → affine transform (A_k, b_k) on memory state
  - Parallel scan composes all K transforms in O(log K) steps
  - Yields all intermediate memory states S_0, S_1, ..., S_K
  - Fully differentiable — gradients flow through the scan
```

Total serial depth: **O(log K)** instead of O(K * P) in current architecture.

## Phase 1: Column Processing

With memory frozen, every token is independent. No scan, no recurrence — just
a massive batched matmul across B columns and N tokens.

```python
def phase1_forward(tokens, frozen_memory):
    """Fully parallel: B columns x N tokens.

    tokens:         [BS, N]
    frozen_memory:  snapshot of PM/EM/WM state from start of segment

    Returns logits and per-span write candidates.
    """
    x = embed(tokens)                              # [BS, N, D]

    # Memory reads — all use frozen snapshot, all parallel
    # These are content-addressed lookups (dot products), no state dependency
    x = x + WM.read(x, frozen_memory.wm)          # [BS, N, D]
    x = x + PM.read(x, frozen_memory.pm)          # [BS, N, D]
    x = x + EM.read(x, frozen_memory.em)          # [BS, N, D]

    # Per-column processing (B columns fold into batch dim)
    surprise = PCM.surprise(x)                     # [BS, B, N, D_h]
    x = FFN(x, surprise)                           # per-column weights
    logits = output_head(x)                        # [BS, N, vocab]

    # Compute write candidates, grouped by span
    # Decisions (which slot, whether to write) made here using frozen state
    candidates = compute_write_candidates(x, frozen_memory, spans)

    return logits, candidates
```

### What columns are NOT

Columns are NOT deep stacks. They're lightweight parallel processors:
- No multi-layer computation (L=1)
- No recurrence (no RNN state)
- No attention (no KV cache)
- Just: input projection → FFN → output projection, conditioned on memory reads + PCM surprise

The "depth" in this architecture comes from memory interaction, not layer stacking.

## Phase 2: Affine Memory Scan

### Why Memory Updates Are Affine

Each memory module's state update, given precomputed decisions, is affine:

**PM** — "replace slot i if eligible":
```
slot_i_new = (1 - mask_i) * slot_i_old + mask_i * new_value_i

A = diag(1 - mask)     # mask ∈ {0,1} per slot, precomputed
b = mask ⊙ new_values   # new_values precomputed from column outputs
```

**EM** — "write episode to slot j":
```
S_em_new = (I - e_j e_j^T) S_em_old + e_j @ episode^T

A = I - e_j e_j^T      # zeros out slot j
b = e_j @ episode^T    # writes new episode into slot j
```

**WM** — scatter update:
```
slot_new = (1 - gate) * slot_old + gate * new_value

A = diag(1 - gate)     # per-slot gate
b = gate ⊙ new_values
```

**PCM** — exponential moving average:
```
z_hat_new = alpha * z_hat_old + (1 - alpha) * z_mean_span

A = alpha (scalar)
b = (1 - alpha) * z_mean_span
```

All four are: **S_new = A * S_old + b** with A diagonal (or scalar). Affine in S.

### Parallel Scan

Affine composition is associative:
```
(A2, b2) ∘ (A1, b1) = (A2 * A1, A2 * b1 + b2)
```

Given K spans with updates (A_0, b_0), ..., (A_{K-1}, b_{K-1}):
- Parallel scan yields all prefix compositions in O(log K) steps
- Each intermediate state: S_k = (A_k * ... * A_0) * S_init + composed_b_k
- Since A's are diagonal, composition is element-wise (cheap)

```python
def phase2_memory_scan(span_updates, initial_state):
    """Parallel scan over K span memory updates.

    span_updates: [(A_k, b_k) for k in range(K)]
        A_k: [num_slots] diagonal gate (element-wise)
        b_k: [num_slots, D] additive update
    initial_state: memory state at segment start

    Returns: list of K intermediate memory states (all in computation graph)
    """
    # Combine all memory modules into single state vector for scan
    # A_k and b_k encompass PM + EM + WM + PCM updates for span k

    states = parallel_associative_scan(
        combine_fn=affine_compose,
        elements=(As, bs),
    )

    # Apply to initial state
    memory_states = [A_prefix @ initial_state + b_prefix
                     for A_prefix, b_prefix in states]

    return memory_states  # all differentiable
```

## Gradient Flow and Credit Assignment

**Phase 1 backprop:** Standard backprop through FFN + memory reads. All tokens
independent → no BPTT within Phase 1. Embarrassingly parallel.

**Phase 2 backprop:** Backprop through parallel scan is another parallel scan
(reverse direction). O(log K) depth. Differentiable w.r.t. both A_k and b_k.

**Credit assignment chain:**
```
loss @ token t (in span k)
  → Phase 1: logits ← FFN ← memory_read(frozen_state) ← embed
  → Phase 2: frozen_state = S_0 (initial) or connected via scan to S_{k-1}
  → scan connects: S_{k-1} ← S_{k-2} ← ... ← S_0
  → each S_j ← (A_j, b_j) ← write candidates from Phase 1 span j
  → gradient reaches all earlier spans' column outputs
```

Credit assignment horizon = **full segment** (all K spans), computed in O(log K).

### Important nuance

Tokens within the same span all read from the same frozen state. So the gradient
from a token in span 3 flows through the Phase 2 scan to reach the write
candidates from spans 0, 1, 2 — but NOT directly to other tokens' column
processing. Inter-token gradient flow is mediated entirely through the memory
state chain.

This is by design: tokens influence each other ONLY through memory, which is
the brain-inspired inductive bias.

## Staleness Analysis

All tokens in a segment read from start-of-segment memory. The maximum staleness
is N tokens (one segment length).

**Why this is acceptable:**
- Memory changes incrementally (soft gating, EMA updates)
- Over N=256 tokens: PM updates ~1-2 slots, EM adds ~2 episodes, WM shifts buffer
- The model learns to produce useful outputs given stale memory — it's a consistent
  training signal, not a source of noise
- Biologically plausible: neurons operate on stale synaptic state (axonal delays,
  synaptic consolidation timescales)

**Staleness is bounded and constant.** It doesn't grow with sequence length.
At segment boundaries, memory is fully up-to-date.

## Serial Depth Comparison

```
Current architecture:
  Within span:  O(P) = O(32)    sequential RNN steps
  Across spans: O(K) = O(8)     sequential boundary ops
  Total:        O(K*P) = O(256) sequential steps

Proposed (v3):
  Phase 1:      O(1)            all tokens independent (parallel)
  Phase 2:      O(log K) = O(3) parallel scan across spans
  Total:        O(log K) = O(3) sequential steps

  (If PCM scans within span: O(log P + log K) = O(10))
```

## Scaling Configuration

```
B  = 16 columns          (parallel, fold into batch dim)
L  = 1 per column        (single FFN — simplest possible)
D  = 768 total           (B * D_h)
D_h = 48 per column
P  = 128 tokens/span
K  = 2 spans/segment     (N = 256 tokens)

Shared memory:
  PM:  r slots x D       (r = 16 → 12,288 floats)
  EM:  M slots x D_em    (M = 1024, D_em = 128 → 131,072 floats)
  WM:  W slots x D       (W = 64 → 49,152 floats)
  PCM: B x D_h           (768 floats, per-column)

Total memory state: ~750 KB per stream, constant forever.
```

## Open Questions

1. **Local coherence**: Without attention/RNN, how do adjacent tokens influence
   each other? Options: (a) causal conv before memory reads, (b) WM provides
   recent-token context, (c) it's fine — memory provides sufficient context.

2. **Column dimensionality**: Is D_h=48 enough for meaningful PM/EM reads?
   Could use D for shared memory reads and D_h only for column-internal processing.

3. **Lateral inhibition**: Should columns compete (winner-take-all on writes)?
   Biologically motivated but adds complexity.

4. **Document boundaries**: PCM z_hat reset at doc boundaries. In affine scan,
   this is A_k=0 for the span containing a doc boundary (wipes state).

5. **torch.compile compatibility**: Phase 1 is pure matmuls (compiles perfectly).
   Phase 2 parallel scan — need to verify associative_scan works under compile.

6. **Train vs inference parity**: During inference, tokens arrive one at a time.
   Phase 1 processes single tokens. Phase 2 becomes a simple sequential update
   (apply one affine transform per span). Constant memory, O(1) per token.
