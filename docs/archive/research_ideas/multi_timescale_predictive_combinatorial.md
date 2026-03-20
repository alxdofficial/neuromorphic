# Research Ideas: Multi-Timescale, Predictive Coding, Combinatorial Memory

Status: **#2 Predictive Coding is IMPLEMENTED** (see `src/model/predictive_coding.py`). #1 Multi-Timescale was implemented then **removed** (single shared weight vector too few degrees of freedom, LayerNorm washes out the effect, recurrence already handles temporal dynamics). #3 Combinatorial Memory remains brainstorm.
Date: 2025-02-19 (ideas), 2026-02-25 (implementation of #2, removal of #1)

**Implementation notes (2026-02-25):**
- **PCM surprise** is RMS-normalized: `‖δ‖/√D_pc` (keeps scale near ~1 instead of raw L2 ~√D_pc ≈ 11.3).
- **FFN gain** is bounded: `1 + 0.1 * tanh(W_gain(δ))` → [0.9, 1.1].
- **z_hat_valid** flag prevents garbage surprise before first boundary_update.
- **Surprise warmup blend**: `(1-α)*CE + α*PCM` with α ramping 0→1 over pcm_warmup_steps.
- **Mid-span doc boundary reset** clears PCM hypothesis when new doc starts mid-span.

---

## 1. Multi-Timescale Blocks

### Core insight

The brain has no global clock — neurons operate on heterogeneous timescales.
Currently` our B blocks all receive the same token stream simultaneously
(`x_proj.view(BS, B, D_h)` — a pure spatial split of the same timestep).
The blocks differ in *what dimensions* they see, but not in *when* they see
them.

### Proposal

Instead of splitting D spatially, give each block a **time-shifted or
time-smoothed view** of the token stream:

- **Block 0 (fast):** sees raw tokens at time t (current behavior)
- **Block 1 (medium):** sees a causal conv1d smoothed version (kernel=3-5),
  capturing short-range context
- **Block 2 (slow):** sees a dilated causal conv (dilation=4-8), capturing
  coarser long-range patterns
- **Block k:** sees an exponential moving average with learned decay,
  creating a continuous multi-scale hierarchy

Each block still processes its stream through L layers of recurrence + PM
as usual — it doesn't know its timescale.  The spatial decoder then fuses
all timescales in the thalamic integrator, which already handles B block
summaries + type embeddings.

### Concrete mechanisms (candidates)

**Option A — Causal dilated convolutions:**
```
For block b with dilation d_b:
    x_b[t] = Conv1d(x[t-d_b*(k-1) : t+1], kernel_size=k, dilation=d_b)
```
Dilation schedule: `d_b = 2^b` (exponential) or `d_b = fibonacci(b)`.
Each conv is a small `nn.Conv1d(D_h, D_h, kernel_size, dilation, causal_padding)`.

Pro: well-understood, efficient on GPU, fixed receptive fields.
Con: fixed timescales (not learned), adds parameters.

**Option B — Learned EMA per block:**
```
x_b[t] = alpha_b * x[t] + (1 - alpha_b) * x_b[t-1]
where alpha_b = sigmoid(learnable_scalar_per_block)
```
Pro: minimal params (one scalar per block), fully differentiable, each
block learns its own timescale.
Con: exponential smoothing may be too simple; single scalar doesn't capture
multi-frequency content.

**Option C — Strided / offset views:**
```
Block b processes every s_b-th token, offset by o_b:
    stream_b = x[o_b::s_b]  (subsampled)
```
Pro: truly different temporal resolutions, natural multi-scale.
Con: variable sequence lengths per block complicate batching; information
loss from subsampling; needs interpolation for decoder fusion.

**Option D — Frequency-domain decomposition:**
```
Apply learned bandpass filters per block in the token embedding space.
Block 0: high-frequency (local syntax)
Block 1: mid-frequency (phrase structure)
Block 2: low-frequency (discourse/topic)
```
Could implement as depthwise conv1d with different filter banks per block.

### How it addresses the plasticity gap

Currently PM/EM operate at span boundaries (every P=32 tokens) while WM
operates at every token. This creates a mismatch: WM has fine-grained
temporal resolution but PM/EM can only write coarse summaries.

With multi-timescale blocks:
- Fast blocks (small dilation) capture token-level patterns well-suited
  for WM interaction
- Slow blocks (large dilation) produce smoother, more abstract
  representations that are **naturally aligned** with the timescale at
  which PM/EM operate (span-level)
- The decoder fuses both, so the final prediction gets both fast reactive
  and slow contextual signals

### Integration points

1. **Input projection:** After `W_in` projects to D, apply per-block
   temporal transform before the `.view(BS, B, D_h)` split.  During span
   processing (`process_span`), we have the full `[BS, T, D]` tensor so
   causal convs are natural.

2. **Decoder fusion:** The thalamic integrator already processes B cortical
   tokens + memory tokens with type/position embeddings.  Could add a
   "timescale embedding" analogous to the existing type embedding so the
   decoder knows which block is fast vs slow.

3. **PM/EM interaction:** Slow blocks' outputs fed to PM/EM may produce
   more stable eligibility traces and better EM candidates (less noisy
   than token-level signals).

### Risks and open questions

- Does temporal smoothing destroy token-level information needed for
  next-token prediction?  Mitigation: keep at least one block on raw
  tokens.
- Causal convolutions need access to previous tokens — works naturally in
  `process_span` but needs a small buffer for `step` (autoregressive).
  The recurrence state already carries forward; we'd need to add a small
  conv buffer per block.
- Compile impact: adding per-block convs may increase compile graph count.
  Likely minor since it's just one conv per block before the existing L
  layers.
- How to choose timescale hyperparameters?  Option B (learned EMA) avoids
  this but may underfit.  Could start with fixed exponential schedule and
  see if it helps before learning.

---

## 2. Predictive Coding / Free Energy Principle via Autoencoding

### Core insight

The brain isn't a passive perceiver — it's a prediction machine.  Karl
Friston's free energy principle says the brain maintains a generative model
and minimizes *prediction error* (free energy).  What's perceived as
"surprise" is really the discrepancy between predicted and actual sensory
input.

Our current surprise signal is just `token_surprise = -log p(target)` from
the language model head — a measure of how poorly the model predicted the
next token.  This is a scalar per token, derived from the *output* of the
model.  It doesn't measure surprise at the *memory state* level.

### Proposal

Add a small **state-prediction autoencoder** that:

1. **Encodes** current memory state + recent input into a compressed
   latent: `z = Encoder(memory_state, recent_tokens)`
2. **Decodes/predicts** what the next memory state will look like:
   `predicted_next = Decoder(z)`
3. **Waits** for the actual next memory state to materialize (after
   processing the next span)
4. **Compresses** the actual next state: `z_actual = Encoder(actual_next_state)`
5. **Computes prediction error:** `delta = z_actual - z_predicted`
   (or a learned distance in latent space)
6. Uses `delta` as a **better surprise signal** for PM eligibility and EM
   write gating

### Why this is better than token-level surprise

- **Token surprise** measures output prediction error (vocabulary space).
  A rare word can be "surprising" even if the model understood the context
  perfectly.
- **State prediction error** measures whether the model's *internal
  representation* changed unexpectedly.  This captures genuine novelty at
  the representation level — exactly what PM/EM should care about.
- Example: seeing "the cat sat on the ___" → "mat" is low token surprise
  but also low state surprise (predictable context change).  Seeing an
  unexpected topic shift mid-paragraph → low token surprise for common
  words but HIGH state surprise (the memory state is changing in ways the
  model didn't predict).

### Architecture sketch

```
State = concat(pm_summary, em_summary, wm_summary, h_final)  # [BS, D_state]

# Small bottleneck autoencoder
Encoder: D_state → D_bottleneck (e.g. 64-128 dims)
    Linear(D_state, D_bottleneck) + LayerNorm + GELU

Predictor: D_bottleneck → D_bottleneck
    Linear(D_bottleneck, D_bottleneck)  # predict next latent from current

Decoder: D_bottleneck → D_state  (only needed for auxiliary loss, not inference)
    Linear(D_bottleneck, D_state)
```

**Per-span flow:**
```
# After processing span t:
z_t = Encoder(state_t)
z_pred_t1 = Predictor(z_t)      # predict next span's latent

# After processing span t+1:
z_t1 = Encoder(state_t1)        # encode actual next state
delta = ||z_t1 - z_pred_t1||    # prediction error

# Use delta as surprise signal for PM/EM gating
# Optionally add auxiliary reconstruction loss:
#   L_recon = ||Decoder(z_t) - state_t||^2  (keeps encoder meaningful)
```

**Compute cost:**
- Encoder: 2 matrix multiplies per span (tiny vs main model FLOPS)
- Predictor: 1 matrix multiply
- No decoder needed at inference (only for training auxiliary loss)
- Total: ~3 small matmuls per span × B blocks ≈ negligible

### Candidate variations

**Option A — Latent-space L2 prediction error (above):**
Simplest.  `delta = ||z_actual - z_predicted||^2` as scalar surprise.
Risk: L2 in latent space may not capture structured differences well.

**Option B — Contrastive prediction error:**
Instead of L2, use an InfoNCE-style loss:
```
score(z_pred, z_actual) = cos_sim(z_pred, z_actual) / tau
surprise = -log(softmax(score) over negative samples)
```
Negatives = z from other batch elements or other spans.
Pro: captures structural differences better than L2.
Con: needs negative sampling, more compute.

**Option C — Distribution-matching (VAE-style):**
Encoder outputs `(mu, log_var)` instead of point estimate.
Prediction error = KL divergence between predicted and actual posterior.
Pro: principled probabilistic formulation, KL is a proper surprise measure.
Con: VAE training dynamics can be finicky; KL collapse risk.

**Option D — Multi-scale prediction:**
Predict next state at multiple horizons (1 span, 2 spans, 4 spans ahead).
Different horizons gate different memory systems:
- 1-span error → PM (fast, local)
- 4-span error → EM (slow, episodic)
Pro: naturally separates what each memory system should care about.
Con: needs delayed state buffers, more complex training.

### Recommendation

Start with Option A (L2 latent prediction error) for simplicity.  Keep the
autoencoder tiny (D_bottleneck=64).  Add it as an auxiliary signal that
**blends** with existing token-surprise rather than replacing it:

```
effective_surprise = alpha * token_surprise + (1 - alpha) * state_surprise
```

where alpha is a learned parameter initialized at 0.9 (mostly token
surprise at start, gradually shifts as autoencoder trains up).

### Integration points

1. **State collection:** At each span boundary, collect
   `(pm_summary, em_summary, wm_output, h_final)` — most of these are
   already computed.  PM summary = mean of pm_K weighted by pm_a.
   EM summary = mean of em_K weighted by em_S (or top-k retrieval output).
   WM = y_wm output.

2. **Delayed comparison:** Need to buffer z_predicted from span t and
   compare when span t+1 finishes.  Simple: store one `z_prev_predicted`
   tensor in the model state, shift each span.

3. **Gating replacement:** Feed `state_surprise` into PM neuromodulator
   and EM neuromodulator alongside or instead of `span_surprise_mean`.
   Both neuromods already accept surprise as input feature.

4. **Auxiliary loss:** Add `L_prediction = ||z_pred - z_actual.detach()||^2`
   to total loss with small weight (0.01-0.1).  Detach z_actual so
   prediction doesn't affect main model gradients.

### Risks

- Auxiliary loss could interfere with main LM training if weight is too
  high.  Start small (0.01).
- State prediction may be trivially easy if memory states change slowly
  (most spans are within-document).  Mitigation: normalize delta by running
  mean to get relative surprise.
- The encoder needs to learn quickly or the surprise signal is garbage
  early in training.  Mitigation: pretrain-freeze the encoder briefly, or
  use warmup schedule on the blend alpha.

---

## 3. Combinatorial Memory Composition

### Core insight

Simple aggregation (sum, mean, concat) has capacity linear in the number of
components.  But *combinations* grow exponentially: 10 digits encode 10^10
numbers.  If memory slots could be *composed* rather than aggregated, the
effective capacity would be exponential in the number of slots, not linear.

Currently, PM readout is a linear lookup: `h_pm = pm_V @ weights` (weighted
sum of value vectors).  EM readout is cross-attention: weighted sum of
retrieved values.  Both are fundamentally *additive* — the capacity scales
linearly with the number of slots and their dimensionality.

### Proposal

Replace additive memory readout with a **compositional chain** where each
memory slot configures a differentiable operation, and the chain of
operations has combinatorial capacity:

```
# Instead of: output = sum(w_i * V_i)
# Do:         output = Op_n(Op_{n-1}(...Op_1(query, V_1), V_2)..., V_n)
```

Each slot's value vector V_i parameterizes an operation Op_i that
transforms the running state.  The *sequence* of operations encodes
information combinatorially.

### Candidate mechanisms

**Option A — Slot-conditioned affine transforms:**
```
state_0 = query  # [BS, D_h]
for i in selected_slots:
    # V_i parameterizes a transform
    scale_i = Linear_scale(V_i)  # [D_h] — per-dim scaling
    shift_i = Linear_shift(V_i)  # [D_h] — per-dim shifting
    state_{i+1} = scale_i * state_i + shift_i
output = state_final
```
This is a chain of learned affine transforms.  With r=8 slots, each having
D_h-dimensional scale+shift, the space of possible outputs is the
composition of 8 affine maps — much richer than a single weighted sum.

Pro: simple, differentiable, each slot's contribution depends on all
previous slots (true composition).
Con: deep chain of affines can be unstable (exploding/vanishing); may need
residual connections or normalization between steps.

**Option B — Slot-conditioned gated projections (hypernetwork-lite):**
```
state_0 = query
for i in selected_slots:
    # V_i generates a small weight matrix via low-rank factorization
    W_i = V_i[:rank].unsqueeze(-1) @ V_i[rank:2*rank].unsqueeze(0)  # [rank, rank]
    gate_i = sigmoid(Linear_gate(V_i))
    state_{i+1} = gate_i * (state_i + state_i @ W_i) + (1-gate_i) * state_i
output = state_final
```
Each slot applies a gated low-rank projection to the running state.
The gate prevents instability (can choose to be identity).

Pro: richer than affine, gating prevents collapse/explosion.
Con: low-rank W_i may limit expressiveness; overhead of forming W_i.

**Option C — Product-key composition (hash-based):**

Inspired by product quantization in retrieval systems:

```
# Split query into sub-vectors (sub-keys)
q_parts = query.view(n_parts, D_h // n_parts)

# Each memory slot stores n_parts sub-codebook entries
# V_i = [v_i^1, v_i^2, ..., v_i^{n_parts}]

# Compositional readout: select one slot per part independently
for part_j in range(n_parts):
    scores_j = q_parts[j] @ V[:, j, :].T  # [r]
    weights_j = softmax(scores_j)
    output_j = weights_j @ V[:, j, :]

# Concatenate: output = [output_0 || output_1 || ... || output_{n_parts-1}]
```

With r slots and n_parts parts, there are r^n_parts possible composed
outputs (exponential in n_parts).

Pro: proven in retrieval literature, clean exponential scaling, naturally
parallelizable.
Con: loses cross-part interaction (each part is independent); the
"combinatorial" aspect is really about *selection*, not *transformation*.

**Option D — Compositional function chaining via MLPs:**
```
state_0 = query
for i in selected_slots:
    # Small MLP conditioned on slot value
    state_{i+1} = MLP_shared(concat(state_i, V_i))
output = state_final
```
A single shared MLP that takes (current_state, slot_value) and outputs
next_state.  The same MLP applied with different slot values creates a
rich compositional function.

Pro: maximum expressiveness (universal function approximation per step),
weight-efficient (shared MLP).
Con: sequential chain limits parallelism; deeper chain = harder to train.

### Analysis: where does capacity actually come from?

The key distinction is between:

1. **Additive capacity** (current): `sum(w_i * V_i)` — the output lives
   in the convex hull of the value vectors.  With r slots of dimension d,
   the capacity is O(r * d).

2. **Compositional capacity**: the output is a *function* of the ordered
   sequence of slots.  With r slots, there are r! orderings; with k
   selected from r, there are P(r,k) possible compositions.  Even with
   simple affine transforms, the space of reachable outputs is the
   *product* of r affine groups, which is vastly larger than their sum.

The practical question: **does the model need this capacity?**  PM has r=8
slots of D_h=384 dimensions.  Additive capacity = 3072 effective dims.
For language modeling at 32K vocab, this is probably sufficient for local
pattern storage.  Combinatorial composition would matter more for:
- Very small D_h (few dimensions per slot)
- Many slots used simultaneously
- Tasks requiring relational/structural reasoning (not just pattern matching)

### Recommendation

Option A (slot-conditioned affine chain) is the simplest starting point.
Add residual connections and LayerNorm between steps for stability:

```
state = query
for i in top_k_slots:
    scale = 1 + tanh(W_scale @ V_i)  # centered at identity
    shift = W_shift @ V_i
    state = LayerNorm(state * scale + shift) + state  # residual
output = state
```

The `1 + tanh(...)` initialization ensures the chain starts as near-
identity and learns to compose.  The residual connection prevents
vanishing gradients through the chain.

This is also fully compatible with the existing slot-selection mechanism
in PM (softmax routing) — just changes what happens after slots are
selected.

### Integration points

1. **PM readout** (procedural_memory.py `read()` / `read_batch()`):
   Currently `h_pm = pm_V @ weights`.  Replace with compositional chain
   over the top-k selected slots.

2. **EM readout** (episodic_memory.py `read()` / `read_batch()`):
   Currently cross-attention weighted sum.  Could apply same composition
   to retrieved values.

3. **Ordering:** The chain needs a canonical ordering of slots.  Options:
   - By weight (highest-weighted slot applied first)
   - By slot index (fixed)
   - Learned ordering via a small sorting network

### Risks

- Gradient flow through r sequential operations may cause
  vanishing/exploding gradients.  With r=8 and residual connections this
  should be manageable (similar depth to an 8-layer transformer).
- Increased compute: r sequential affine transforms vs 1 matrix multiply.
  But r=8 and dims are small (D_h=384), so the absolute cost is tiny.
- May not help if the current additive readout isn't the bottleneck.
  Would need ablation to confirm.
- torch.compile may struggle with the sequential loop — could unroll
  manually for r=8.

---

## Interactions Between These Ideas

These three ideas have natural synergies:

1. **Multi-timescale + Predictive coding:** Slow blocks produce slowly-
   varying representations that are easier to predict.  The state-
   prediction autoencoder could operate per-timescale, giving timescale-
   specific surprise signals.  Fast-timescale surprise gates PM (local
   patterns), slow-timescale surprise gates EM (episodic events).

2. **Predictive coding + Combinatorial memory:** If the memory has higher
   effective capacity (via composition), the predicted-vs-actual state
   delta becomes a more meaningful signal.  With additive memory, state
   changes are small and continuous; with compositional memory, state
   changes can be more discrete/structural, making prediction errors more
   informative.

3. **Multi-timescale + Combinatorial memory:** Different timescale blocks
   could use different composition depths.  Fast blocks: shallow
   composition (1-2 steps, react quickly).  Slow blocks: deep composition
   (4-8 steps, build complex representations).

---

## Priority and Sequencing

If implementing:

1. **Multi-timescale blocks** — lowest risk, clearest implementation path,
   most likely to show immediate throughput/quality improvements.  Start
   with Option B (learned EMA per block).

2. **Predictive coding surprise** — medium risk, needs auxiliary loss
   tuning but the architecture change is small (tiny autoencoder).  Would
   replace the weakest part of the current system (raw CE as surprise).

3. **Combinatorial memory** — highest risk/reward.  Most speculative,
   hardest to tune, but could unlock qualitatively different capabilities.
   Save for after the other two are validated.
