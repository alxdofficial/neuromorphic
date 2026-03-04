# Neuromorphic LM: How It Works

This document walks through the model's architecture, traces how a token flows
through the system, and explains each memory system and training procedure.

## The Big Picture

The neuromorphic LM is a recurrent language model with three biologically-inspired
memory systems. Unlike a transformer (which re-reads all prior tokens each step)
or a plain RNN (which compresses everything into one state vector), this model
maintains separate memory structures that persist indefinitely across the input
stream.

```
               ┌──────────────────────────────────────────────┐
Tokens ──► Embed ──► Stage 1 Scan ──► PCM ──► Memory Ops ──► Stage 3 Scan ──► Logits
               │       (encode)      (surprise)  (PM+EM read)  (integrate)      │
               └──────────────────────────────────────────────┘
                                                  │
                                        at segment end:
                                        PM commit (Hebbian)
                                        EM commit (neuromod EMA)
```

The model processes text in fixed segments of N=512 tokens. Within each segment,
everything is causal (token t never sees token t+1). At the end of each segment,
memory systems update their persistent state based on what was seen. This state
then influences the next segment's processing.

### Three memory systems, three purposes

| System | What it does | Biological analogy |
|--------|-------------|-------------------|
| **PM** (Procedural Memory) | Learns repeated input→output transforms via Hebbian weight updates | Basal ganglia — habitual skills |
| **EM** (Episodic Memory) | Stores novel events as key-value primitives, retrieved via "trail" navigation | Hippocampus — episodic recall |
| **PCM** (Predictive Coding) | Predicts the next token's encoding and computes surprise when wrong | Cortical prediction error |

---

## Token Flow: Step by Step

### 1. Embedding

Each token is looked up in a learned embedding table and combined with a
positional bias:

```
token_ids [BS, N]  →  embedding lookup  →  + positional  →  x [BS, N, D]
                       [BS, N, D_embed]      bias             (after proj_up if D_embed ≠ D)
```

For Tier A: `D_embed=384` (compact embeddings), projected up to `D=2048` (wide
internal representation). The same embedding table is reused as the output layer
(tied embeddings).

### 2. Stage 1: Causal Scan (Encoding)

Six scan layers process the sequence. Each layer applies a causal linear
recurrence — a lightweight alternative to attention:

```
h_t = sigmoid(a_t) · h_{t-1} + silu(b_t)
```

where `a` is a decay gate (how much to remember) and `b` is the input signal.
Both are projected from the input via a single linear layer. The recurrence runs
left-to-right, so each position's output depends only on preceding tokens.

Each layer follows a pre-norm residual pattern:

```
H_out = H_in + proj_out(scan(RMSNorm(H_in)))
```

After six layers, `H [BS, N, D]` is a contextual encoding where each position
has accumulated information from all preceding tokens in the segment.

On GPU, the scan dispatches to a fused Triton kernel (from the FLA library)
that computes the entire recurrence in a single pass.

### 3. PCM: Predictive Coding (Surprise Computation)

Before entering memory operations, the model computes *surprise* — how much
each token deviates from what the scan state predicted.

The D-dimensional hidden state is viewed as C=16 independent "cortical columns"
of D_col=128 dimensions each (a free reshape, no computation). Within each
column:

1. **Encode** the input token: `z_t = W_enc(x_col_t)`
2. **Predict** the next token's encoding from the scan state: `z_hat_t = W_pcm(H_col_t)`
3. **Surprise** at position t = what was predicted minus what actually appeared:
   `surprise_t = z_hat_{t-1} - z_t`

This is a *vector* surprise — not a single number but D_col=128 dimensions of
prediction error per column. This rich error signal drives both PM plasticity
gating and EM novelty detection.

The surprise also modulates the scan output through a learned gain:

```
gain_t = 1.0 + 0.1 · tanh(W_gain(surprise_t))    # ∈ [0.9, 1.1]
H_modulated = H · gain
```

Well-predicted tokens are slightly suppressed; surprising tokens are amplified.
`W_gain` starts at zero, so this effect is learned, not imposed.

PCM also contributes an auxiliary loss (MSE between prediction and reality) that
trains the scan layers to be better predictors.

### 4. Stage 2: Memory Operations

Three signals are computed from the encoded tokens and fed to Stage 3:

#### PM Read (Procedural Memory)

PM maintains a fast-weight matrix `W_pm [BS, B, D_pm, D_pm]` per memory bank.
Reading it is a projection sandwich:

```
pre     = proj_in(H)           # D → D_pm (64 dims)
W_sum   = sum over B banks     # [BS, D_pm, D_pm]
post    = pre @ W_sum^T        # matmul through the fast weights
pm_read = proj_out(post)       # D_pm → D
```

The bank sum means all B=4 banks contribute to every read via a single matrix
multiply. Banks are invisible at read time — their specialization comes from
having different plasticity rates (how fast they update).

`proj_out` is zero-initialized, so PM starts completely silent and gradually
learns to contribute.

#### EM Trail Read (Episodic Memory)

EM stores M=384 key-value primitives per bank. Reading uses "trail navigation" —
an iterative process that composes a response from multiple primitives:

```
Start at seed (projected from H)
For each trail step (2-3 iterations):
    1. Attend to all primitives using current position as query
    2. Retrieve weighted mix of values
    3. Compute gate based on alignment with retrieval
    4. Move: position += gate · retrieved_content
Return net displacement from seed
```

This is more expressive than a single attention lookup — the trail can compose
information from multiple primitives by visiting them in sequence, building up
a composite retrieval.

#### EM Write Buffer (Causal Cumulative Mean)

The model also computes a running average of what's been novel so far in the
segment:

```
cum_em_t = (1/t) · Σ_{s=1}^{t} (gate_s · novelty_s · w_cand_s)
```

This gives the integration scan real-time awareness of novel content, without
waiting for the segment-end commit. The division by position keeps it bounded.

**Novelty** combines two signals: PCM surprise magnitude (how unexpected was
this token?) and EM reconstruction error (how poorly can existing primitives
reproduce this content?). A learned blend weight `W_nov` balances these two
components.

A **neuromodulator** MLP takes (novelty, memory usage) and outputs a per-token
write gate `g_em ∈ [0.001, 0.95]`. This solves the resource allocation problem:
when memory is nearly full or content isn't novel, suppress writes. When
something genuinely new appears and capacity is available, amplify writes.

### 5. Stage 3: Integration Scan

Four signals are combined additively at each token position:

```
integrated = H + pm_read + em_trail_read + cum_em_write_buffer
```

Then six more scan layers process this combined signal. Stage 3's job is to
integrate long-term knowledge (EM), habitual patterns (PM), within-segment
novelty (write buffer), and the baseline encoding (H) into a coherent
representation for prediction.

### 6. Output

```
out    = proj_down(H_prime)     # D=2048 → D_embed=384
out    = LayerNorm(out)
logits = lm_head(out) · D_embed^{-0.5}
```

The final linear layer shares weights with the embedding table (tied embeddings).
The `1/√D_embed` scaling ensures logits have standard deviation ~1 at
initialization, so the initial cross-entropy loss equals the random baseline
`ln(vocab) ≈ 10.37` rather than being inflated by peaked but wrong predictions.

---

## Memory Systems In Depth

### Procedural Memory (PM)

**What it is.** A per-bank D_pm × D_pm weight matrix that acts as a learned
linear transform in a low-dimensional space. Think of it as a tiny neural
network layer whose weights update via Hebbian learning instead of
backpropagation.

**How it updates (segment-end commit).**

1. Compute surprise gate: `s_t = sigmoid(‖surprise_t‖)` — how surprised
   was the model at each token? Values near 1 mean high surprise.

2. Build gated autocorrelation matrix:
   `G = (1/N) · Σ_t s_t · pre_t ⊗ pre_t^T`

   This captures which input dimensions co-activate during surprising events.
   Implemented efficiently as a single batched outer product:
   `G = (√s · pre)^T @ (√s · pre) / N`.

3. Update via right-multiply:
   `W_b = W_b @ (decay · I + β_b · G)`

   Each bank has its own plasticity rate β_b (learned, always positive via
   softplus). Fast banks adapt quickly to recent patterns; slow banks encode
   deeper habits. The decay (0.999) ensures old associations gradually fade.

4. Budget enforcement: clip Frobenius norm of each bank's matrix to prevent
   unbounded growth.

**Why banks matter.** All banks are summed at read time, so they jointly produce
one output. But because they update at different rates, they capture patterns at
different timescales. Bank 1 might rapidly track the current paragraph's style
while Bank 4 slowly accumulates document-level patterns.

### Episodic Memory (EM)

**What it is.** A dictionary of M=384 key-value primitives per bank.
Keys `em_K [BS, B, M, D]` encode *what* was stored (unit-normalized direction
vectors). Values `em_V [BS, B, M, D]` encode the *content*. Strengths
`em_S [BS, B, M]` track how established each primitive is.

**How it reads.** Trail navigation (described above) — an iterative attention
process that composes multiple primitives into a single retrieval. The trail
approach lets the model build complex responses from simple stored parts.

**How it writes (segment-end commit).**

1. Soft-route the segment's write candidates to primitives via attention.
2. Aggregate routes weighted by per-token novelty.
3. Update via neuromodulated EMA:
   ```
   α = clamp(g_em · route_aggregate)
   em_K = (1 - α) · em_K + α · normalize(new_key)
   em_V = (1 - α) · em_V + α · new_value
   em_S = clamp(em_S + α, max=S_max)
   ```
4. Between commits: `em_S *= 0.999` (base decay). Primitives that are never
   refreshed gradually decay and become available for reuse.

**Key design choice:** EM reads use the state from *before* the current segment
(frozen). Writes happen after. This ensures causal ordering: you retrieve based
on what you knew before seeing the current segment, then update memory with what
you just learned.

### Predictive Coding Module (PCM)

**What it is.** A per-column next-token predictor that operates within the scan.
It doesn't store anything — it generates a real-time surprise signal that drives
the other two memory systems.

**The prediction loop:**
```
At time t, the scan state H_t reflects tokens 0..t.
PCM predicts: "given what I've seen through t, token t+1's encoding should be z_hat_t"
At time t+1: actual encoding is z_{t+1}
Surprise = z_hat_t - z_{t+1}   (vector, 128 dims per column)
```

**What surprise does:**
- Modulates scan output via learned gain (amplify surprising tokens)
- Gates PM Hebbian plasticity (surprise magnitude → sigmoid → write strength)
- Contributes to EM novelty (blended with reconstruction error)
- Provides auxiliary training loss (MSE) that improves predictive ability

PCM operates on grouped "cortical columns" — 16 independent groups of 128
dimensions. This means surprise is computed per feature group, not globally.
Different columns can be surprised by different aspects of the input
simultaneously.

---

## Training

### TBPTT (Truncated Backpropagation Through Time)

The model trains on persistent streams — continuous concatenations of documents
processed by BS=16 parallel reading heads. Each training step processes a TBPTT
chunk of K=4 segments × N=512 tokens = 2048 tokens per stream.

```
Chunk [2048 tokens]
├── Segment 1 [512 tok]: forward → loss₁, PM/EM commit
├── Segment 2 [512 tok]: forward → loss₂, PM/EM commit
├── Segment 3 [512 tok]: forward → loss₃, PM/EM commit
└── Segment 4 [512 tok]: forward → loss₄, PM/EM commit
                                     ↓
                              sum losses / valid_count
                              + regularizer × 0.1
                              backward, clip grad, optimizer step
                              detach memory states ← TBPTT boundary
```

Within a chunk, gradients flow through memory states across all K segments.
At the TBPTT boundary, memory values are preserved but their gradient history is
severed (`detach_()`). This limits the gradient horizon to K segments while
allowing memory state to accumulate indefinitely.

### Loss Function

```
total_loss = CE_avg + pcm_pred_weight · PCM_MSE + reg_weight · regularizers

where:
  CE_avg          = sum of per-token cross-entropy / valid token count
  PCM_MSE         = mean squared prediction error (pcm_pred_weight = 0.1)
  regularizers    = soft penalties for PM/EM approaching capacity limits
  reg_weight      = 0.1
```

EOT tokens are excluded from the CE loss to prevent the model from learning
to predict across document boundaries.

### Document Boundaries

When an `<|endoftext|>` token is detected at the start of a segment, PM and
EM states are reset for that stream (in Phase A). This means each document gets
fresh memory. In Phase B (lifelong mode), no reset occurs — memory accumulates
across documents.

### What Gets Optimized

| Component | Parameters | Updated by |
|-----------|-----------|------------|
| Embedding + pos_embed | Token vectors, position bias | Backprop (every step) |
| Scan layers (×12) | proj_in, proj_out, RMSNorm | Backprop (every step) |
| PCM | W_enc, W_pcm, W_gain | Backprop (every step) |
| PM projections | proj_in, proj_out | Backprop (every step) |
| PM plasticity rates | raw_beta (per bank) | Backprop (multi-segment chain) |
| EM trail params | gate_alpha, gate_bias, tau, sigma, tau_w | Backprop (multi-segment chain) |
| Neuromodulator | MLP weights | Backprop (every step) |
| W_seed_w, W_nov | Grouped projections, novelty blend | Backprop (every step) |
| **PM fast weights** | W_pm [BS, B, D_pm, D_pm] | **Hebbian commit** (not backprop) |
| **EM primitives** | em_K, em_V, em_S | **Neuromodulated EMA commit** (not backprop) |

The crucial distinction: PM and EM *states* (W_pm, em_K/V/S) are updated by
their own learning rules (Hebbian correlation, EMA), not by the optimizer.
Backprop trains the *parameters* that control these learning rules (plasticity
rates, trail parameters, neuromodulator weights). The memory systems learn how
to learn.

---

## Scale and Efficiency

**Tier A (dev tier):** 92M parameters, 2048 internal width, 12 scan layers.

On an RTX 4090: ~65K tok/s training throughput at BS=16 with `torch.compile`.
The scan recurrence is O(N) per layer (vs O(N²) for attention), making it
efficient on long segments.

PM and EM are computationally negligible — PM read is one 64×64 matmul
(~84M ops, 0.24% of a single scan layer). EM state is O(1) per token regardless
of sequence length.
