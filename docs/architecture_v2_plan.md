# Architecture v2: Parallelization & Scaling Plan

**Date:** 2026-02-26
**Status:** Proposed — not yet implemented
**Goal:** 8-20x training speedup while preserving all bio-inspired mechanisms

---

## What We're Keeping (Sacred)

These are the biological motivations. Any architecture change must preserve all of them:

1. **Procedural Memory (PM)** — Hebbian eligibility traces with neuromodulated commits
2. **Episodic Memory (EM)** — Content-addressed retrieval with surprise-gated writes
3. **Predictive Coding (PCM)** — Evidence vs hypothesis, surprise signals
4. **Constant memory** — State size doesn't grow with context length
5. **Lifelong learning** — Memory persists across document boundaries
6. **Three-memory-system architecture** — WM/PM/EM, validated by Titans, Larimar, SlotSSMs

---

## Change 1: Parallelize the B Blocks

### The Problem

Our model has B=4 independent "cortical columns" (blocks). Each block processes
its own D_h-dimensional slice of the input. They share no state — block 0's
computation has zero dependency on block 1, 2, or 3.

Despite this independence, we process them in a sequential Python `for` loop:

```python
# model.py — current code
for b, block in enumerate(self.blocks):
    result = block.forward_span(...)   # block 1 waits for block 0 to finish
    h_blocks.append(h_b)
h_final = torch.cat(h_blocks, dim=-1)
```

This means the GPU does B=4 sequential forward passes where it could do 1.

### The Solution

Stack all block state tensors along a new `B` dimension and process them as a
single batched computation. Instead of 4 separate [BS, P, D_h] tensors processed
one at a time, we have one [B, BS, P, D_h] tensor processed all at once.

Concretely:
- Current: `h` is [BS, D_h] per layer, 40 separate tensors (4 blocks × 10 layers)
- Proposed: `h` is [B, BS, D_h], 10 tensors (one per layer depth)

The block loop becomes a single batched operation. Every matmul, scan, and
activation that currently processes [BS, P, D_h] now processes [B*BS, P, D_h]
(or equivalently [B, BS, P, D_h] with the B dimension absorbed into batch).

Similarly for PM, EM, PCM — their state tensors get a leading B dimension.

### Intuitive Explanation

Think of it like this: if you have 4 independent math problems to solve on a
calculator, you could solve them one after another (current approach), or you
could use 4 calculators simultaneously (proposed approach). GPUs are massively
parallel — they have thousands of "calculators" sitting idle while we feed them
one block at a time.

### Expected Speedup: ~2-3x

Not a full 4x because the GPU isn't completely idle during single-block
processing — there's still parallelism across the batch and sequence dimensions.
But at Tier B (BS=24, P=32, D_h=512), each block's tensors are large enough
that stacking them gives the GPU significantly more work per kernel launch.

### Effort: Medium (1 week)

Requires restructuring how state tensors are stored and accessed. PM, EM, PCM
modules need to accept the B dimension. The hardest part is boundary operations
(commits, writes) which currently loop over blocks in Python.

---

## Change 2: Replace Sequential Scan with Chunkwise Parallel Training

### The Problem

Each of our 10 layers per block runs a recurrence:

```
h_0 = a_0 * h_init + b_0
h_1 = a_1 * h_0 + b_1
h_2 = a_2 * h_1 + b_2
...
h_31 = a_31 * h_30 + b_31
```

This is 32 sequential steps. Each step depends on the previous one. During
training, we execute this as a Python `for` loop that `torch.compile` fuses
into a single CUDA kernel — but it's still 32 serial operations inside that
kernel. The GPU's thousands of cores mostly wait while one core computes each
step.

We have a parallel algorithm (`torch.associative_scan`) that reduces 32
sequential steps to ~5 (log2(32) = 5), but it's disabled during training
because PyTorch's autograd (the system that computes gradients for learning)
doesn't fully support it yet.

### The Solution: Chunkwise Parallel Computation

There's a mathematical trick that avoids the scan entirely. It's based on the
insight that a linear recurrence can be rewritten as matrix multiplication.

**The key idea:**

Our recurrence `h_t = a_t * h_{t-1} + b_t` can be "unrolled" into:

```
h_t = (a_t * a_{t-1} * ... * a_0) * h_init + sum of (products of a's) * b's
```

This is a weighted sum — and weighted sums are matrix multiplications. If we
organize the a's and b's into matrices, we can compute ALL 32 outputs in one
matrix multiply.

**Concretely:**

Within a chunk of P=32 tokens, build a lower-triangular matrix L where:
```
L[i][j] = product(a_k for k in j+1..i)   if i >= j
L[i][j] = 0                                if i < j
```

Then all outputs at once: `H = L @ B + L_col0 * h_init`

This single matmul replaces the 32-step loop. Matrix multiplication is what
GPUs are built for — it maps perfectly to tensor cores (the dedicated fast
matrix multiply hardware on modern GPUs).

**Why this wasn't obvious:**

For a simple scalar recurrence (like our current `h` which is a vector), the
matrices would be diagonal and the matmul wouldn't be faster than the loop.
But when the state is a matrix (see Change 3), the matmul form is dramatically
faster because it uses tensor cores instead of scalar CUDA cores.

### What About Autograd (Backpropagation)?

The `fla` (Flash Linear Attention) library has hand-written Triton kernels that
implement both the forward AND backward pass of this chunkwise computation.
We already depend on `fla` and use its GLA kernel for working memory. The
kernels handle autograd correctly — this is production-ready code used by
Mamba-2, RWKV-7, and GLA at scale.

### Expected Speedup: ~3-5x

The sequential scan currently dominates training time. Replacing it with matmul-
based chunkwise computation maps everything to tensor cores. This is the single
biggest speedup opportunity.

### Effort: Medium (1-2 weeks)

Requires changing the layer recurrence to use `fla.ops.gated_delta_rule` (or
similar) instead of our custom `_sequential_scan`. The gate parameterization
needs slight adjustment (log-space gates instead of sigmoid). We already went
through a similar exercise when adding FLA support for GLA WM.

---

## Change 3: Upgrade Layer State from Vector to Matrix (Gated DeltaNet)

### The Problem

Each layer's hidden state `h` is a vector of shape [D_h] (e.g., 512 numbers for
Tier B). This is the layer's entire "memory" of what it has seen. One vector.

This limits capacity — a 512-dim vector can store maybe a few dozen "facts" about
the sequence. By contrast, a transformer's KV cache stores a separate key-value
pair for every token, giving it near-perfect recall (but at the cost of O(T)
memory growth, which violates our constant-memory requirement).

We want more capacity without growing memory with context length.

### The Solution: Matrix-Valued State

Instead of a vector `h` of shape [D_h], use a matrix `S` of shape [d_k × d_v].
With multi-head structure (H heads, each with head_dim d), this becomes
[H, d, d] — e.g., [8, 64, 64] = 32,768 numbers per layer instead of 512.
That's 64x more capacity at constant size.

The matrix state stores **key-value associations**. Each column of S represents
a stored "value" associated with a particular "key direction." When we query
with key k, the output is `S^T @ k` — the stored value most aligned with that
key. This is exactly how a dictionary/hash table works, but in continuous
vector space.

### The Update Rule (Gated DeltaNet)

The matrix state updates via:

```
S_t = β_t * S_{t-1} + k_t * (v_t - β_t * S_{t-1}^T @ k_t)^T
```

Breaking this down:

1. **`β_t * S_{t-1}`** — Decay the old state. β is a learned gate (0 to 1)
   that controls how much to forget. This is like our current `a_t * h_{t-1}`.

2. **`S_{t-1}^T @ k_t`** — Query the current state: "what value does S
   predict for key k_t?" This is the state's **prediction**.

3. **`v_t - β_t * S_{t-1}^T @ k_t`** — The **prediction error**: the
   difference between what actually arrived (v_t) and what S expected.
   This is a surprise signal — sound familiar?

4. **`k_t * (error)^T`** — The outer product stores the error as a new
   key-value association. This is a **Hebbian learning rule**: strengthen
   the connection between what you saw (k) and what you didn't expect (error).

The delta rule means: "only store what's new." If S already perfectly predicts
v_t from k_t, the error is zero and nothing is written. If there's a mismatch,
the error gets stored. This prevents interference between memories — old
associations aren't overwritten by redundant information.

### Why This Is Bio-Inspired

This update rule is literally how neuroscience models synaptic plasticity:

- **Hebbian learning**: "neurons that fire together wire together" — the outer
  product `k * v^T` strengthens associations between co-occurring signals
- **Anti-Hebbian correction**: the `- S^T @ k` term prevents runaway
  strengthening — it's the anti-Hebbian component that keeps the system stable
- **Gated forgetting**: β controls memory decay, like neuromodulator-driven
  synaptic depression

This is more biologically plausible than our current design, where the layer
recurrence is a simple gated average with no Hebbian structure.

### Connection to PCM

Our Predictive Coding Module computes `surprise = evidence - hypothesis` at the
block level. The Gated DeltaNet computes `error = v_t - S^T @ k_t` at every
layer at every token. These are structurally identical — both are prediction
errors.

With a DeltaNet backbone, every layer already produces a per-token surprise
signal for free. PCM could tap into these layer-level deltas to get richer
surprise information, or PCM could be simplified since the backbone already
"does predictive coding" natively.

### Connection to PM

Our PM stores r=8 key-value slots with eligibility traces and neuromodulated
writes. The DeltaNet matrix S stores ~d key-value associations with gated
decay and error-driven writes. They're doing similar things at different
granularities:

- **S (backbone)**: Fast, implicit, per-token updates. High-bandwidth but
  limited control. Like short-term synaptic plasticity.
- **PM (overlay)**: Slow, explicit, boundary-gated updates. Lower-bandwidth
  but neuromodulator-controlled. Like long-term potentiation.

PM stays as a modular overlay — it augments the backbone's matrix state with
explicitly controlled persistent associations. The two systems are complementary,
not redundant.

### Expected Speedup: Indirect

The matrix state itself doesn't speed things up — it adds computation.
But it ENABLES Change 2 (chunkwise parallel training). The chunkwise algorithm
is based on matrix multiplies, and it needs a matrix state to work efficiently.
With a vector state, the "matrices" would be diagonal and you'd get no benefit
from tensor cores. With a [H, d, d] matrix state, every operation maps to
dense matmul on tensor cores.

Also: higher capacity per layer means we might need fewer layers (L=10 → L=6?)
for the same model quality, which directly reduces sequential depth.

### Effort: High (2 weeks)

This is the biggest change. The layer module needs rewriting. Gate
parameterization changes. State initialization, detach, reset, save/load all
need updating. Tests need rewriting. But the `fla` library provides the core
kernel — we don't have to implement the chunkwise algorithm ourselves.

---

## Change 4: Decide What Happens to Working Memory (WM)

### The Problem

We currently have a separate Working Memory module — sliding-window softmax
attention over the most recent W=128 tokens. This is shared across all blocks
and provides high-fidelity local recall.

With a Gated DeltaNet backbone, each layer already has a matrix state that
functions as a form of "working memory." Do we still need a separate WM?

### The Options

**Option A: Keep WM as sliding-window attention (hybrid approach)**

This is the "Griffin" / "BASED" design: pair efficient recurrence (DeltaNet)
with a small window of exact attention. The recurrence handles long-range
context; the attention handles precise local recall.

Pros:
- Best recall accuracy for recent tokens (exact attention is unbeatable)
- Validated by multiple papers (Griffin, BASED, Zamba, Jamba)
- Minimal change to current code

Cons:
- Two separate systems is more complex
- The attention has O(P × W) cost per span (currently ~5-10% of compute)

**Option B: Replace WM with GLA (already implemented)**

We already have `GLAWorkingMemory` using `fla.ops.gla`. With a DeltaNet
backbone, we could either:
- Keep GLA WM as a separate "fast recurrence" at a different timescale
- Remove WM entirely and let the backbone handle everything

Pros:
- Simpler architecture
- Less code to maintain
- GLA + DeltaNet might be redundant

Cons:
- Lose exact local recall — everything goes through lossy compression
- Might hurt performance on tasks requiring precise recent-token lookup

**Option C: Absorb WM into the backbone (simplest)**

Remove the separate WM module entirely. The backbone layers' matrix states
serve as working memory. Use 1-2 dedicated "local attention" layers
(standard sliding-window self-attention) interleaved with DeltaNet layers
for high-fidelity local recall.

Pros:
- Simplest possible architecture
- Follows the hybrid pattern proven at scale (Griffin, Jamba)

Cons:
- Need to tune which layers get attention vs recurrence
- Changes the information flow (currently WM output feeds ALL blocks)

### Recommendation: Start with Option A, then experiment

Keep the current WM as-is for the initial migration. Once the backbone is
working, we can ablate: does removing WM hurt? If not, simplify to Option C.

### Effort: Low (part of integration)

No immediate work needed. This is a decision for after the backbone change.

---

## Change 5: Compile Boundary Operations

### The Problem

After each span of P=32 tokens, we run "boundary operations" — PM commits,
EM writes, PCM hypothesis updates. These are implemented as Python loops
over B blocks and L layers:

```python
# span_ops.py — simplified
for block in model.blocks:
    for layer in block.layers:
        layer.pm.commit(...)           # 40 separate Python calls
    block.em.write_at_boundary(...)    # 4 separate Python calls
    block.pcm.boundary_update(...)     # 4 separate Python calls
```

Each call dispatches a small GPU kernel. With 48 calls per boundary and 8
boundaries per training step, that's ~384 small kernel launches where the
GPU starts, does a tiny computation, and stops.

`torch.compile` currently only covers `block.forward_span`. The boundary
operations are uncompiled.

### The Solution

Extend `torch.compile` to boundary operations by:

1. Batching PM commits: instead of 40 separate calls, stack all PM state
   along a [B, L] dimension and commit in one batched operation
2. Batching EM writes: 4 separate calls → 1 batched call over B dimension
3. Wrapping the entire boundary function in `torch.compile`

This is synergistic with Change 1 (parallelize blocks): once state is batched
along B, boundary operations naturally become single batched calls.

### Expected Speedup: ~1.2-1.5x

Boundary operations are currently ~10-15% of step time. Eliminating the
Python loop overhead and kernel launch latency should cut that roughly in half.

### Effort: Low-Medium (days)

Mostly mechanical once Change 1 is done. The tricky part is ensuring
`torch.compile` doesn't break on the conditional logic in commit/write
(neuromodulator gating, slot selection).

---

## Combined Impact

| Change | Speedup | Effort | Dependencies |
|--------|---------|--------|-------------|
| 1. Parallelize B blocks | ~2-3x | 1 week | None |
| 2. Chunkwise parallel scan | ~3-5x | 1-2 weeks | Change 3 |
| 3. Matrix state (Gated DeltaNet) | Enables #2 | 2 weeks | None |
| 4. WM decision | Neutral | After #3 | Change 3 |
| 5. Compile boundaries | ~1.2-1.5x | Days | Change 1 |

**Conservative combined estimate: ~8-15x**
**Optimistic estimate: ~15-25x**

For Tier B at current 9.3K tok/s:
- 8x → ~74K tok/s → 10B tokens in ~37 hours
- 15x → ~140K tok/s → 10B tokens in ~20 hours
- 20x → ~186K tok/s → 10B tokens in ~15 hours

---

## Implementation Order

**Phase 1 — Quick wins (1 week):**
1. Parallelize B blocks (Change 1)
2. Compile boundary operations (Change 5)
3. Benchmark: expect ~3-4x speedup

**Phase 2 — Backbone upgrade (2-3 weeks):**
1. Implement Gated DeltaNet layer using `fla.ops.gated_delta_rule` (Change 3)
2. Wire chunkwise parallel training (Change 2)
3. Keep PM/EM/PCM as overlays — they don't change
4. Extensive testing: gradient flow, state management, lifelong mode
5. Benchmark: expect ~8-15x total speedup over current

**Phase 3 — Polish (1 week):**
1. WM ablation (Change 4) — does removing it hurt?
2. Tune: number of layers, head dimensions, chunk size
3. Re-benchmark all tiers
4. Update training configs and token budgets

---

## Risk Assessment

**Low risk:**
- Change 1 (parallelize blocks): Pure execution optimization, no architecture change
- Change 5 (compile boundaries): Same computation, better execution

**Medium risk:**
- Change 3 + 2 (DeltaNet backbone): Significant code change, but the math
  is well-understood and the `fla` kernels are battle-tested. Our tests will
  catch gradient issues. The main risk is integration complexity.

**Architectural risk:**
- Change 4 (WM removal): Might hurt recall. We'll ablate carefully.

**What we're NOT risking:**
- PM, EM, PCM are untouched as modular overlays
- The three-memory-system design is preserved
- Constant memory and lifelong learning are preserved
- Bio-inspired mechanisms (Hebbian learning, surprise signals, neuromodulation)
  are preserved — and arguably enhanced by DeltaNet's native delta rule

---

## References

1. Dao & Gu, "Transformers are SSMs" (Mamba-2/SSD), arXiv:2405.21060
2. Yang et al., "Gated Linear Attention Transformers" (GLA), ICML 2024
3. Yang, Kautz, Hatamizadeh, "Gated Delta Networks" (Gated DeltaNet), ICLR 2025, arXiv:2412.06464
4. Peng et al., "RWKV-7 Goose," arXiv:2503.14456
5. De et al., "Griffin: Gated Linear Recurrences + Local Attention," arXiv:2402.19427
6. Arora et al., "BASED: Simple Linear Attention" (recall-throughput tradeoff), ICML 2024
7. Behrouz et al., "Titans: Learning to Memorize at Test Time," Google Research, NeurIPS 2025
8. Beck et al., "xLSTM: Extended LSTM," arXiv:2405.04517
