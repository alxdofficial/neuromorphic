# Architecture v5: Scan-Memory-Scan with Cortical Columns

> **Previous versions**: v4 (iterative R-loop refinement) preserved on `v4-iterative-backup` branch.
> v1 preserved on `v1-legacy` branch.

## The Core Insight

The brain processes sensory input through two interleaved systems:

1. **Fast neocortical stream** — feedforward, massively parallel, processes each
   input near-instantly. Cortical columns extract features, predict upcoming input,
   and prepare eligibility signals for memory.

2. **Slow memory stream** — hippocampal/thalamic pattern completion, slower,
   asynchronous. Reads from and writes to structured memory stores based on what
   the fast stream found surprising or novel.

We implement this as a **three-stage cycle** per segment of N tokens:

```
Stage 1: FAST SCAN (affine, parallel via parallel-scan)
  → Process N tokens causally through C narrow cortical columns
  → Each column maintains a recurrent state via linear recurrence
  → Produces: EM trail seeds, write candidates, surprise vectors, column states

Stage 2: MEMORY OPS (non-affine, batched across all N positions)
  → Proposed writes computed first (per-token, parallel, from frozen memory)
  → Prefix sums of write deltas (causal accumulation, O(N))
  → PM read with causal bias (frozen + prefix-summed surprise)
  → EM trail read from frozen primitives (compositional, long-term)
  → End of segment: actual structured commit to memory state

Stage 3: INTEGRATION SCAN (affine, parallel via parallel-scan)
  → Second causal scan integrates PM read + EM trail + EM write buffer
  → Write buffer (prefix sum of write candidates) provides within-segment
    new information and cross-column mixing
  → Produces final predictions (NTP logits)
```

**Key properties:**
- **Causal within segments** (scans are causal recurrences)
- **Causal across segments** (memory state flows forward)
- **Causal write buffers** (prefix-summed writes visible within segment)
- **Parallel within segments** (parallel-scan algorithm, memory ops batched)
- **No R-loop** — single cycle replaces R iterative passes
- **NTP training** — causal scans naturally support next-token prediction

## Why This Architecture Exists

We face a fundamental tension:

1. **Softmax attention**: KV cache grows linearly with context length, and
   O(N²) compute per layer during prefill — too expensive for lifelong learning
2. **Linear recurrence / affine scans**: our bio-inspired memory updates (Hebbian PM,
   episodic EM) involve non-affine decisions (thresholds, similarity matching,
   slot selection) — can't naively parallelize within a scan
3. **Pure sequential processing**: O(N) serial depth — underutilizes GPU

Our resolution: **separate affine computation from non-affine memory operations.**

- **Stages 1 & 3** (affine scans): causal, parallelizable via chunked parallel-scan.
  All column processing, PCM prediction, eligibility computation —
  everything that can be expressed as `h_t = a_t ⊙ h_{t-1} + b_t` goes here.
- **Stage 2** (memory ops): non-affine but batched. PM gain modulation, EM trail
  composition, novelty scoring, neuromodulated primitive writes — all the bio-inspired
  logic runs here. Not scan-friendly, but runs once per segment across all N
  positions in parallel.

**Biological motivation:** The neocortex processes quickly and mostly feedforward.
The hippocampal loop is slower, asynchronous, and handles consolidation. Our
three-stage cycle mirrors this: fast processing → memory interaction → integration.

## Architecture

### Components

**Cortical columns** (narrow, many, process independently):
- C columns, each operating on a D_col = D/C feature slice of every token
- At each timestep, all C columns process the same token's embedding, each
  seeing its own D_col-wide feature slice
- Each column maintains a recurrent state h_c,t via element-wise linear recurrence
  (`h_t = a_t ⊙ h_{t-1} + b_t`) — columns are **independent** within the scan
- **PCM** — predictive coding within the scan (predict next token's encoding,
  compute vector surprise)
- No separate FFN — the scan's input/output projections serve this role
- Columns produce trail seeds and write candidates as linear projections of
  their scan state

**Shared memory systems** (B independent banks, addressed by column slices):
- **PM** — procedural memory: learned bias vector per bank (basal ganglia).
  Element-wise gain modulation — fast, automatic, coarse.
- **EM** — episodic memory: dictionary of M primitive patterns (hippocampus).
  Trail-based compositional read — seed navigates primitive space.
- B memory banks. Each bank is addressed by the columns assigned to it.
- Column c in block b addresses only dims `c*D_col:(c+1)*D_col` of bank b's
  memory — pure dimension slicing, no routing, no gather/scatter.
- PM state shape: `[BS, B, D]` (one bias vector per bank)
- EM state shape: `[BS, B, M, D]` where D = C × D_col (M primitives per bank)

**No within-scan lateral mixing**:
- Columns are **independent** within each scan — element-wise A, no cross-column
  coupling. This mirrors cortical biology where columns process independently
  and mix through downstream convergence zones.
- Cross-column integration happens through **memory** (Stage 2): all columns
  read from and write to shared memory banks, which serve as the binding
  mechanism — analogous to thalamic/hippocampal hubs in the brain.

### How It Works (High Level)

```
For each segment of N tokens:

  STAGE 1 — Fast Scan (causal, parallel via parallel-scan):
    For t = 1..N (parallelized):
      h_t = a_t ⊙ h_{t-1} + b_t               # element-wise recurrence per column
      seed_t = W_seed · H_t                     # EM trail seed
      w_cand_t = W_w · H_t                      # write candidate
      z_hat_t = W_pcm · H_t                     # PCM prediction
      surprise_t = z_hat_{t-1} - encode(X_t)    # vector surprise (D_col)

  STAGE 2 — Memory Operations (parallel across N positions):
    # 1. Proposed writes (per-token, parallel, from frozen memory):
    For each bank b = 1..B:
      delta_pm_t = lr_pm · surprise_t            # per-token PM bias delta
      delta_em_t = novelty_t · w_cand_t          # per-token EM write delta

    # 2. Prefix sums (causal accumulation, O(N)):
      cum_pm_t = Σ_{i≤t} delta_pm_i             # inclusive, causal for NTP
      cum_em_t = Σ_{i≤t} delta_em_i

    # 3. Reads (from frozen memory + causal write buffers):
      pm_read = H[:, b] ⊙ (1 + pm_bias_b + cum_pm)  # PM: causal gain mod
      em_read = EM_b.trail_read(seed[:, b])           # EM: trail from frozen

    # 4. End of segment: actual structured commit
      PM_b.commit(delta_pm)                      # bias update
      EM_b.commit(w_cand, novelty)               # decompose across primitives

  STAGE 3 — Integration Scan (causal, parallel via parallel-scan):
    For t = 1..N (parallelized):
      input_t = pm_read_t + em_read_t + cum_em_t # frozen reads + write buffer
      h'_t = a'_t ⊙ h'_{t-1} + b'_t             # scan on input_t
      logits_t = LM_head(H'_t)

  Carry memory state to next segment.
```

Stage 1 and Stage 3 are embarrassingly parallel via parallel-scan algorithms.
Stage 2 is a fixed-cost batch operation (all N positions independent).

### Stage 1: Fast Scan (The Neocortical Stream)

All C columns process every token, each seeing its D_col = D/C feature slice.
The scan is causal — token t's state depends on tokens 1..t-1.

```python
def fast_scan(self, input_ids):
    """Stage 1: Causal affine scan through all N tokens.

    Produces column states, trail seeds, write candidates, and surprise.
    Parallelized via chunked parallel-scan (à la Mamba).

    Returns:
        H:        [BS, N, C, D_col]    — column hidden states
        seed:     [BS, N, B, D]        — EM trail seeds (per bank)
        w_cand:   [BS, N, B, D]        — write candidates (per bank)
        surprise: [BS, N, C, D_col]    — vector surprise signals
    """
    x = embed(input_ids)             # [BS, N, D_embed]
    x = proj_up(x)                   # [BS, N, D]
    x = x + pos_embed                # [BS, N, D]

    # Each column gets its D_col slice: column c sees x[..., c*D_col:(c+1)*D_col]
    # Element-wise linear recurrence: h_t = a_t ⊙ h_{t-1} + b_t
    # a_t, b_t are input-dependent linear projections
    # Columns are independent — no cross-column mixing in scan

    H = parallel_scan(x)             # [BS, N, C, D_col]

    # Linear projections from column states
    seed = W_seed(H)                 # [BS, N, B, D] — EM trail starting points
    w_cand = W_w(H)                  # [BS, N, B, D] — write candidates per bank
    z_hat = W_pcm(H)                 # [BS, N, C, D_col] — PCM prediction

    # Vector surprise: elementwise prediction error
    z = pcm_encode(x)                # [BS, N, C, D_col] — current encoding
    surprise = z_hat_prev - z        # [BS, N, C, D_col] — VECTOR, not scalar
    # z_hat_prev is the prediction from token t-1 (carried in scan state)

    return H, seed, w_cand, surprise
```

**What's in the scan state:** Each column c maintains a hidden state h_c ∈ R^D_col.
The full scan state is H ∈ R^{C × D_col} = R^D. Because A is element-wise (no
cross-column mixing), this is C independent scans — each column evolves its own D_col
state. Cross-column integration happens through shared memory in Stage 2.

**No separate FFN:** The scan's input-dependent projections (computing a_t, b_t from
the input) serve the same role as FFN layers — nonlinear feature transformation. Adding
a separate FFN would be redundant.

**PCM in the scan:** The prediction z_hat_t and encoding z_t are linear functions of H_t,
so they can be folded into the scan as additional output projections. Surprise is the
elementwise difference between consecutive predictions and encodings — also linear.

### Stage 2: Memory Operations (The Hippocampal Loop)

Memory reads use the **frozen** segment-start state. Proposed writes are computed
per-token in parallel, then **prefix-summed** to provide causal within-segment
write signals. The actual structured commit happens at segment end.

```python
def memory_ops(self, H, seed, w_cand, surprise):
    """Stage 2: Write-then-read with causal write buffers.

    1. Compute proposed writes per-token (parallel, from frozen memory)
    2. Prefix sum write deltas (causal accumulation)
    3. PM read uses causal bias (frozen + prefix-summed surprise)
    4. EM trail reads from frozen primitives (long-term knowledge)
    5. End of segment: actual structured commit to memory

    Returns:
        pm_reads:  [BS, N, B, D]  — PM gain reads (with causal bias)
        em_reads:  [BS, N, B, D]  — EM trail reads (from frozen)
        cum_em:    [BS, N, B, D]  — EM write buffer (within-segment new info)
    """
    pm_reads, em_reads, cum_ems = [], [], []

    for b in range(B):
        # --- 1. PROPOSED WRITES (per-token, parallel, from frozen memory) ---
        # PM write delta: surprise-driven bias shift
        delta_pm = lr_pm * surprise[:, :, b]       # [BS, N, D] — per-token

        # EM write delta: novelty-weighted write candidate
        novelty = compute_novelty(w_cand[:, :, b], surprise, em_K[b], em_V[b])
        delta_em = novelty * w_cand[:, :, b]       # [BS, N, D] — per-token

        # --- 2. PREFIX SUMS (causal accumulation, O(N)) ---
        cum_pm = cumsum(delta_pm, dim=seq)         # [BS, N, D] — inclusive
        cum_em = cumsum(delta_em, dim=seq)         # [BS, N, D] — inclusive
        # Token t sees writes from tokens 0..t (inclusive — causal for NTP)

        # --- 3. PM READ with causal bias (write-before-read) ---
        pm_read = H[:, :, b] * (1 + pm_bias[b] + cum_pm)  # [BS, N, D]
        # pm_bias[b] is frozen from previous segment
        # cum_pm adds per-token causal bias adaptation

        # --- 4. EM TRAIL READ from frozen primitives ---
        s = seed[:, :, b]                          # [BS, N, D] — trail start
        if training:
            s = s + σ[b] * randn_like(s)           # noise for exploration

        y = s
        for step in range(n_steps):                # n_steps = 2
            attn = softmax(y @ em_K[b].T / τ[b])   # [BS, N, M] — all primitives
            delta = attn @ em_V[b]                  # [BS, N, D] — composition
            gate = sigmoid(W_gate(cat(y, delta)))   # [BS, N, D] — learned gate
            y = y + gate * delta                    # seed moves

        em_read = y - s                             # [BS, N, D] — net contribution

        pm_reads.append(pm_read)
        em_reads.append(em_read)
        cum_ems.append(cum_em)                      # write buffer for Stage 3

        # --- 5. SEGMENT-END COMMIT (actual structured write) ---
        # PM: apply total bias shift (sum over ALL N tokens)
        pm_bias[b] += delta_pm.sum(dim=seq)         # [BS, D] — aggregate

        # EM: full decomposition write (routing, neuromodulator, EMA)
        route = softmax(normalize(w_cand[:, :, b]) @ em_K[b].T / τ_w)
        update_K = sum_n(novelty * route * w_cand[:, :, b])
        update_V = sum_n(novelty * route * w_cand[:, :, b])
        g_em = em_neuromodulator(novelty_mean, usage)
        α = g_em * route_aggregated
        em_K[b] = (1 - α) * em_K[b] + α * normalize(update_K)
        em_V[b] = (1 - α) * em_V[b] + α * update_V

    return stack(pm_reads), stack(em_reads), stack(cum_ems)
```

**Column-local memory addressing:** PM bias bank b has shape `[BS, D]`.
EM bank b has shape `[BS, M, D]`. Column c reads/writes only its slice
`[..., c*D_col:(c+1)*D_col]`. This means:
- No routing decisions — pure dimension indexing
- GPU-friendly — contiguous memory access patterns
- Surprise, seeds, and candidates are all dimension-specific
- Each column independently contributes to its slice of the bank's memory

**Two write paths — fast and slow:**
- **Fast path (within-segment):** Per-token write deltas are prefix-summed to
  produce causal write buffers. PM's `cum_pm` modifies the gain formula directly
  (write-before-read). EM's `cum_em` flows into Stage 3 as an additive signal.
  These provide immediate within-segment feedback — token t sees what tokens
  0..t wrote. Gradient flows from THIS segment's loss.
- **Slow path (cross-segment):** The actual structured commit (PM bias aggregate,
  EM routing + neuromodulator + EMA) happens at segment end and updates the
  frozen state for the next segment. This is the long-term memory path.

**Why frozen reads + prefix sums (not per-token memory updates):** Updating the
actual EM primitives per-token would require storing per-position memory states
([N, M, D] — too expensive) and sequential trail reads. Instead, the trail reads
from frozen primitives (batchable), and the prefix-summed write buffer carries
the within-segment new information as a simple D-dimensional signal. This is
equivalent in effect: token t gets frozen long-term knowledge (trail) + causal
recent writes (prefix sum).

**State vs parameters:** The primitives (em_K, em_V) and bias (pm_bias) are
**state**, not parameters. Parameters (W_seed, W_gate, neuromodulator, τ, σ)
are frozen after training. State evolves at inference — this is lifelong learning.

### Stage 3: Integration Scan (Context Fusion)

A second causal scan integrates memory reads with column states.

```python
def integration_scan(self, H, pm_reads, em_reads, cum_em):
    """Stage 3: Second causal scan with memory context + write buffers.

    Three signals contribute:
    - pm_reads:  PM gain modulation (includes causal bias via cum_pm)
    - em_reads:  EM trail from frozen primitives (long-term knowledge)
    - cum_em:    EM write buffer (within-segment new information)

    Returns:
        logits: [BS, N, vocab]
    """
    # Additive integration: PM read + EM trail + EM write buffer
    # Each column c takes its D_col slice of each signal
    integrated = pm_reads_per_col + em_reads_per_col + cum_em_per_col

    # Second scan: h'_t = a'_t ⊙ h'_{t-1} + b'_t (element-wise, independent columns)
    H_prime = parallel_scan_2(integrated)        # [BS, N, C, D_col]

    # Project to model dim and decode
    out = proj_down(H_prime.reshape(BS, N, D))   # [BS, N, D_embed]
    logits = lm_head(out)                         # [BS, N, vocab]

    return logits
```

**Why two scans?** The first scan processes raw input and produces
seeds/candidates before seeing any memory context. The second scan has
access to both memory reads AND causal write buffers — it integrates
long-term knowledge (EM trail), habitual bias (PM gain), and recent
within-segment writes (write buffer) with the causally ordered token stream.

**Write buffer as cross-column mixer:** `cum_em_t` is a D-dimensional signal
(from `w_cand = W_w(H)`, which projects across all columns). When added to
Stage 3's input, it provides within-segment cross-column information — token t
at column c sees what columns 0..C-1 wrote at positions 0..t. This partially
fills the cross-column mixing role alongside the cross-segment memory reads.

## Predictive Coding: Vector Surprise

### PCM in the Scan

PCM operates within the first scan. Each column predicts what the **next
token's** encoding will look like. Surprise is the vector difference between
the prediction and the actual encoding.

```
Token t-1: column state H_{t-1} → z_hat_{t-1} = W_pcm · H_{t-1}  (prediction)
Token t:   column input X_t → z_t = W_enc · X_t                    (actual encoding)
           surprise_t = z_hat_{t-1} - z_t                           (VECTOR, D_col dims)
```

**Vector surprise** (not scalar): Each dimension of D_col carries its own
surprise signal. The model was wrong about feature 3 but correct about feature
7 — memory writes can be modulated per-feature. This is richer than a scalar
`||z_hat - z||` because:
- Memory writes are dimension-specific (column c writes to its D_col slice)
- Surprise naturally has the same dimensionality as the write candidates
- PM bias shifts and EM writes operate per-dimension: "I was wrong about THESE
  features, adapt bias and store corrections for THOSE features"

### What Surprise Means in v5

"Based on the causal context so far, I predicted the next token's features
would look like z_hat. The actual features were z. The per-feature error tells
me which aspects of this token I failed to anticipate — those aspects are
informationally dense and worth writing to memory."

This is pure within-scan prediction (token t predicts token t+1), not
cross-pass prediction like v4. The surprise signal is available immediately
as part of the scan computation.

### PCM Optimization

1. **Auxiliary prediction loss**: `L_pred = MSE(z_hat_t, z_{t+1}.detach())`
   Gradient flows to the prediction network but not the encoding target.
2. **Downstream (within-segment)**: surprise → delta_pm (prefix-summed into PM
   gain) + novelty → delta_em (prefix-summed into write buffer) → Stage 3 → loss.
   PCM encoder gets same-segment gradient through both write buffer paths.
3. **Downstream (cross-segment)**: surprise → novelty → structured EM commit →
   next segment's trail reads → loss. PCM encoder learns what's worth predicting.

## Memory System Details

### PM: Procedural Bias (Gain Modulation)

PM is a **single bias vector per bank** — coarse, automatic, fast. It modulates
the signal by scaling features up or down based on learned habits.

```
# State: one bias vector per bank
pm_bias [BS, B, D]                           # D = C × D_col

# Per-token write delta (Stage 2, parallel):
delta_pm_t = lr_pm * surprise_t              # [BS, N, D] — per-token, per-feature

# Prefix sum (causal accumulation):
cum_pm_t = cumsum(delta_pm, dim=seq)         # [BS, N, D] — inclusive

# Read with causal bias (write-before-read):
y_pm = H[:, :, b] * (1 + pm_bias[b] + cum_pm)  # [BS, N, D]
# Token t's gain reflects surprise from tokens 0..t — no lag

# Segment-end commit:
pm_bias[b] += delta_pm.sum(dim=seq)          # [BS, D] — total shift
```

**Why this is coarse:** One vector, not slots. No routing, no selection. Every
feature dimension has a single learned bias. This captures automatic response
tendencies (basal ganglia habits) — "I always up-weight feature 7 and
down-weight feature 12." Surprise-gating ensures it only adapts on
unpredicted features.

**Causal within-segment adaptation:** Unlike the cross-segment-only design,
the prefix-summed write delta gives PM immediate effect. If feature 7 is
consistently surprising in the first 100 tokens, token 101 already sees an
amplified bias for that feature — within the same segment.

**Biologically:** Basal ganglia procedural memory is fast, automatic, habitual.
Not compositional, not explicit — just learned biases in processing.

### EM: Primitive Dictionary with Trail-Based Composition

EM stores M **primitive patterns** — atomic building blocks of concepts. Reading
is done via a **trail**: a seed vector navigates primitive space through
iterative refinement. The output is a composition of activated primitives.

**State:**
```
em_K   [BS, B, M, D]     — primitive keys (unit-norm): "what triggers this"
em_V   [BS, B, M, D]     — primitive values: "what this contributes"
em_S   [BS, B, M]        — strengths (0 = inactive)
em_age [BS, B, M]        — age tracking for decay
```

**Read (trail-based pattern completion):**
```
# Seed from Stage 1 scan (the starting point for the trail)
seed = W_seed(H)                             # [BS, N, B, D]
if training:
    seed = seed + σ * randn_like(seed)       # noise for exploration (σ learned)

# Trail: iterative refinement through primitive space (n_steps = 2)
y = seed
for step in range(n_steps):
    scores = y @ em_K[b].T                   # [BS, N, M] — score ALL primitives
    scores[em_S == 0] = -inf                 # mask inactive
    attn = softmax(scores / τ)               # sparse activation (τ learned)
    delta = attn @ em_V[b]                   # [BS, N, D] — primitive composition
    gate = sigmoid(W_gate(cat(y, delta)))    # [BS, N, D] — learned step gate
    y = y + gate * delta                     # seed moves through space

y_em = y - seed                              # [BS, N, D] — net memory contribution
```

**What this does:** Step 1 activates primitives near the seed (e.g., "animal",
"outdoor"). The push shifts the query, so step 2 activates different primitives
(e.g., "running", "park"). The final position encodes "dog running in park" —
something no single primitive stores. The primitives are a fixed coordinate
system; the seed is a program that navigates it.

**Combinatorial capacity:** M primitives with continuous seeds → infinite
possible readouts. The memory is O(M × D) but the output space is continuous.

**Novelty scoring:**
```
# Can existing primitives explain this input?
attn = softmax(normalize(w_cand) @ em_K[b].T / τ)
reconstruction = attn @ em_V[b]
recon_error = ||w_cand - reconstruction||    # [BS, N, B] — unexplained signal

# Novelty = surprise magnitude + reconstruction error
w_nov = sigmoid(W_nov(H))                   # [BS, N, B] — learned blend
novelty = w_nov * ||surprise|| + (1 - w_nov) * recon_error
```

**Write (decompose across primitives):**
```
# Soft routing: which primitives should absorb this signal?
route = softmax(normalize(w_cand) @ em_K[b].T / τ_w)  # [BS, N, M]

# Aggregate across N tokens, weighted by novelty
update_K = sum_n(novelty * route * k_cand)   # [BS, B, M, D]
update_V = sum_n(novelty * route * v_cand)   # [BS, B, M, D]

# Neuromodulated commit (fully differentiable, no hard selection)
g_em = em_neuromodulator(novelty_mean, usage)  # scalar gate
α = g_em * route_aggregated                    # [BS, B, M] per-primitive
em_K[b] = (1 - α) * em_K[b] + α * normalize(update_K)
em_V[b] = (1 - α) * em_V[b] + α * update_V
em_S[b] = clamp(em_S[b] + α.sum(token_dim), 0, s_max)
```

**Writes decompose, not store:** Each write signal is spread across multiple
primitives based on soft routing. Existing primitives that partially match
absorb their share. Over time, primitives self-organize: frequently useful
ones strengthen, redundant ones decay, and the dictionary covers the input
distribution efficiently.

**Write buffer (within-segment path):**
```
# Per-token write delta (Stage 2, parallel):
delta_em_t = novelty_t * w_cand_t              # [BS, N, D] — per-token

# Prefix sum (causal accumulation):
cum_em_t = cumsum(delta_em, dim=seq)           # [BS, N, D] — inclusive

# Added to Stage 3 input alongside trail read:
stage3_input_t = pm_read_t + em_trail_read_t + cum_em_t
```

**Two write paths:** The prefix-summed `cum_em` provides immediate within-segment
feedback — token t sees what tokens 0..t wrote (fast path, raw D-vector, same-segment
gradient). The structured decomposition commit updates the actual primitives for the
next segment's trail reads (slow path, M-primitive routing, cross-segment TBPTT).
This mirrors the brain: hippocampal fast encoding (immediate trace) vs. slow
consolidation (structured long-term storage).

### Memory Decay and Aging

`pm_bias` decays toward zero (slow drift back to neutral). `em.age_tick(N)`
increments ages and applies strength decay — both run **once per segment**
after writes. Memory state then carries to the next segment.

### Budget Enforcement

EM primitives compete for limited budget. When total strength exceeds the
budget, the weakest (lowest em_S) primitives are soft-pruned. This prevents
unbounded growth and forces the dictionary to keep only useful primitives.

## Column-Local Memory Addressing

A key simplification in v5: columns address memory by dimension slicing.

```
Memory bank b:
  pm_bias [BS, D]        where D = C × D_col
  em_K    [BS, M, D]
  em_V    [BS, M, D]

Column c of block b addresses:
  pm_bias[:, c*D_col : (c+1)*D_col]
  em_K[:, :, c*D_col : (c+1)*D_col]
  em_V[:, :, c*D_col : (c+1)*D_col]
```

This means:
- **No routing logic** — column c always addresses the same slice
- **Surprise is dimension-specific** — column c's surprise only affects its slice
- **Reads are independent** across columns — no coordination needed
- **GPU-friendly** — contiguous memory access within each column's slice
- **Biologically motivated** — cortical columns in the brain don't have bandwidth
  to address the entire memory cortex at once; they connect to local regions

All columns within a block contribute to the same bank's memory, but each
column only modifies its own feature slice. The full bank stores the
concatenation of all column contributions.

## Gradient Flow

### Within-Segment (direct — no TBPTT needed)

```
loss → logits → Stage 3 scan →
  → cum_em (write buffer):
      → delta_em = novelty · w_cand → novelty params, W_w, Stage 1 scan
      (gradient through prefix sum is a suffix sum — linear, stable)
  → pm_read (causal gain):
      → cum_pm → delta_pm = lr_pm · surprise → lr_pm, PCM params
      → pm_bias (frozen) gets gradient through gain modulation
  → em_trail (frozen primitives):
      → through trail steps → softmax → em_K, em_V, W_gate, W_seed, τ, σ
  → Stage 1 scan → column projection params, PCM params
```

**Major improvement over pre-write-buffer design:** Write parameters (W_w,
novelty scoring, lr_pm) now get gradient from THIS segment's loss through the
prefix-summed write buffers. Previously, writes only got gradient via
cross-segment TBPTT.

### Cross-Segment (requires TBPTT)

The **structured commit** (EM decomposition write with neuromodulator) still
only affects the next segment's frozen memory reads:

```
TBPTT chunk = K segments (e.g., K=2-4)
Segment 1: 3 stages → memory committed → carry to segment 2
Segment 2: 3 stages → loss flows back through trail reads → segment 1's commit
```

K=2 is likely sufficient (one segment of lookahead).

### What Gets Gradient From Where

| Component | Gradient source | Path |
|-----------|----------------|------|
| Scan params (a, b projections) | Same segment's NTP loss | Direct through both scans |
| PCM encode/predict | L_pred (auxiliary) + downstream | Same segment |
| PM bias (pm_bias) | Same segment's loss through gain modulation | Direct, smooth |
| PM lr_pm | Same segment's loss through cum_pm → gain | Through prefix sum (linear) |
| EM trail params (W_seed, W_gate, τ, σ) | Same segment's loss through trail | Through softmax, differentiable |
| EM primitives (em_K, em_V) | Same segment's loss through trail read | Through softmax attention |
| Write candidate proj (W_w) | Same segment's loss through cum_em (write buffer) | Through prefix sum (linear) |
| Novelty scoring | Same segment's loss through cum_em (write buffer) | Through prefix sum (linear) |
| EM neuromodulator | **Next segment's** loss through committed primitives | Cross-segment TBPTT |
| EM commit routing | **Next segment's** loss through committed em_K/em_V | Cross-segment TBPTT |

## Correctness Analysis

### 1. Within-Segment: Causal (NTP)

Both scans are causal recurrences — token t only sees tokens 1..t-1 through the
hidden state. This naturally supports next-token prediction (NTP).

Memory reads in Stage 2 are from the **segment-start** state (frozen). Write
buffers use **inclusive prefix sums** (`cum_t = Σ_{i≤t} delta_i`). Since
`logits_t` predicts `token_{t+1}` and is allowed to depend on tokens 0..t,
the inclusive prefix sum is causal. Both `delta_pm_t` and `delta_em_t` depend
on tokens 0..t (through the causal Stage 1 scan), so `cum_t` at position t
depends on tokens 0..t — matching the NTP causality requirement.

### 2. Cross-Segment: Causal

Memory state at segment k reflects only tokens from segments 0..k-1. Writes from
segment k are committed at the end and only visible to segment k+1.

### 3. Train/Inference Equivalence

**Training**: Process N-token segments through 3 stages, NTP loss on all positions.
**Inference**: Process tokens through 3 stages. Can be done:
- Segment-at-a-time: process N tokens, get N predictions (most efficient)
- Token-at-a-time: update scan state per token, defer memory ops to segment
  boundaries (streaming mode)

Inference memory: scan states (C × D_col per scan × 2 scans) + PM + EM.
**Constant**, does not grow with sequence length.

## Practical Design Decisions

### Column Dimensions

```
# Tier A example (target ~105M params)
D = 2048          # internal model width
D_embed = 384     # embedding / LM head width
B = 6             # memory banks
C = 16            # columns per bank
D_col = 128       # = D / C — column width
N = 512           # segment length
```

Tier presets need recalculation for scan-based architecture.
Scan parameters (layer count, state expansion) affect param count
differently than the R-loop FFN stacks. Anchor on Mamba-130M for comparison.

### Segment Length

Segment length N determines:
- How many tokens contribute to each round of memory updates
- Parallel-scan granularity (longer = more parallelism)
- Memory update frequency (shorter = more frequent writes)

### Lateral Mixing

**Resolved:** No within-scan lateral mixing. Element-wise A means each column
evolves independently. Cross-column integration happens through:
1. **Cross-segment:** Shared memory bank reads (EM trail, PM gain) — all columns
   in a block read/write the same bank.
2. **Within-segment:** The EM write buffer (`cum_em`) is a D-wide projection of
   column states (`w_cand = W_w(H)`), mixing across all columns. When added to
   Stage 3's input, it provides within-segment cross-column information.
3. **Input/output:** `proj_up` and `proj_down` mix across all D dimensions.

This keeps the scan simple (standard element-wise prefix scan) while the
memory systems and write buffers handle the binding role.

### Scaling: Memory Banks

```
Bank b: C columns share PM_b (D) + EM_b (M × D)
```

Total banks = B. Each bank's memory can specialize — analogous to different
brain regions maintaining different types of memories.

## Complexity

### Serial Depth

Per segment: O(N) for each scan (parallelized to O(log N) with parallel-scan)
             + O(1) for memory ops (fixed cost, not N-dependent)
Total: O(log N) work depth (assuming parallel-scan).

Compare:
- Transformer (L=6):  O(L) = **6 steps** (but O(N²) compute per step)
- Mamba (L=24):       O(L × log N) with parallel-scan
- **v5 (2 scans)**:   O(2 × log N) + O(1) memory ops

### Compute

Per segment:
- Stage 1: O(N × D²) for scan (same as one Mamba layer)
- Stage 2: O(N × B × (D + M×D)) for PM/EM reads + writes + O(N×D) prefix sums
- Stage 3: O(N × D²) for second scan
- Total: ~2 Mamba layers + memory ops per segment

### Memory

Training: O(N × D) for scan states + O(N × D) for activations.
Inference: O(D) scan states + O(B × (r + M) × D) memory — **constant**.

## Training

### Training Objective: NTP (Next-Token Prediction)

Causal scans naturally produce autoregressive predictions. Standard cross-entropy
loss on all positions:

    L = (1/N) Σ_t CE(logits_t, token_{t+1}) + α · L_pred

where L_pred is the PCM auxiliary prediction loss.

### TBPTT Chunks

Process K segments per chunk (K=2-4). Memory state detached at chunk boundaries.

### Phase A: Per-Document Learning

PM/EM reset at document boundaries. Neuromodulators learn when to commit.

### Phase B: Lifelong Adaptation

PM/EM persist across documents. Only scan hidden states reset at doc boundaries.
The distillation cycle:

1. New domain → surprise spikes (PCM prediction errors)
2. PM bias shifts to compensate (driven by surprise mean)
3. EM primitives update via neuromodulated decomposition
4. New concepts decomposed and stored across primitives
5. Over time, slow weights learn domain → surprise drops
6. Slow weights = general; PM/EM = domain-specific

## Inference

### Streaming Generation

Process tokens in N-token segments. Each segment:
1. Run 3-stage cycle
2. Emit N token predictions
3. Carry scan states + memory to next segment

Throughput: N tokens per forward pass (no R multiplier).

### Token-by-Token Mode

For interactive use: update scan states per token, run memory ops at segment
boundaries. Latency per token = one scan step (no quadratic attention).

## Design Decisions (Resolved)

1. **Scan formulation** → Simple linear recurrence (`h_t = a_t ⊙ h_{t-1} + b_t`).
   No SSM. Memory systems handle the role SSM's expanded state was designed for.

2. **Lateral mixing** → Element-wise A, independent columns. No within-scan mixing.
   Memory is the cross-column integration mechanism (biologically: thalamic hub).

3. **FFN** → No separate FFN. The scan's input-dependent projections (computing
   a_t, b_t) serve the same nonlinear transformation role.

4. **Scan layers** → Equal depth per stage. Target 12 layers Stage 1 + 12 layers
   Stage 3 = 24 total for ~130M param model (anchored to Mamba-130M).

5. **Memory integration** → Additive. `Stage3_input = pm_read + em_trail + cum_em`.
   Parameter-free, gradient-friendly, degrades gracefully when memory is empty.

6. **CrossBlockMixer** → Dropped. v4 artifact. In v5, memory reads already provide
   cross-bank integration, and scan provides cross-position context.

7. **PM** → Simplified to bias vector with element-wise gain modulation.
   Coarse, fast, automatic — distinct from EM's rich compositional read.

8. **EM** → Trail-based primitive composition. Dictionary of M primitives,
   seed from scan navigates via iterative refinement (2 steps). Soft writes
   decompose across primitives. Clean differentiable credit assignment.

9. **PositionAttention** → No. Scan handles cross-position causally. No attention
   mechanisms — keeps compute O(N) not O(N²).

10. **Implementation** → Custom prefix scan. No external dependency (no mamba-ssm).
    Standard element-wise parallel-scan is simple to implement.

11. **Causal write buffers** → Prefix-summed write deltas provide within-segment
    feedback. PM cum_pm goes inside gain formula (write-before-read). EM cum_em
    is additive in Stage 3. Inclusive prefix sum (i≤t) — causal for NTP. Write
    params get same-segment gradient. Structured commit still happens at segment
    end for long-term memory (cross-segment TBPTT for neuromodulator).

### Implementation Clarifications

These resolve ambiguities in the pseudocode above for actual PyTorch implementation.

**Bank-column mapping (H ↔ memory):**
H is `[BS, N, C, D_col]` (column-indexed). Memory is `[BS, B, ...]` (bank-indexed).
For bank b's operations: `H_bank = H.reshape(BS, N, D)`. All B banks share the
same column states — different banks hold different memory but modulate the same H.
Similarly, `surprise.reshape(BS, N, D)` for bank-level delta computation.
Stage 2 returns `[BS, N, B, D]` signals; Stage 3 sums over B and reshapes to
`[BS, N, C, D_col]` for the integration scan.

**Pseudocode indexing notation:**
`em_K[b]` means `em_K[:, b]` (index the B dimension, keep batch). `pm_bias[b]`
means `pm_bias[:, b]`, shape `[BS, D]`. `H[:, :, b]` means `H.reshape(BS, N, D)`
(same for all banks — bank index selects memory, not column states).

**W_gate is element-wise (not a matrix):**
`gate = sigmoid(w1 * y + w2 * delta + bias)` where w1, w2 are learned `[D]`
vectors. No matmul — O(D) not O(D²). Per-bank or shared across banks.

**w_cand serves as both routing key and write content:**
No separate k_cand/v_cand projections. In the EM write: `route` uses normalized
w_cand as the routing key, and w_cand is also the value being decomposed across
primitives. Simplifies Stage 1 (one projection instead of three).

**lr_pm is a scalar nn.Parameter per bank:**
Shape: `[B]`, parameterized as `softplus(raw_lr_pm)` to stay positive. Gets
gradient from this segment's loss through `cum_pm → gain modulation → logits`.

**em_K/em_V and pm_bias: state in computation graph:**
These are plain tensors, NOT nn.Parameter. But they are NOT detached within a
TBPTT chunk — gradient flows through them via the EMA commit chain from the
previous segment. "Frozen within segment" means their values don't change during
the forward pass, but they participate in the computation graph. At TBPTT chunk
boundaries, `detach_states()` breaks the gradient chain.

**Scan layer structure:**
Each scan layer: fused input projection `[D_col → E]` producing (a, b) where
E = expansion factor × D_col. For E = 2 × D_col: `proj(x) → [a_raw, b_raw]`,
then `a = sigmoid(a_raw)` (decay gate ∈ [0,1]), `b = silu(b_raw)` (input gate).
Each layer has: proj_in `[D_col, E]` + proj_out `[E, D_col]` + LayerNorm.
Param count per layer: ~2 × D_col × E ≈ 4 × D_col². With D_col=128, that's
~66K params/layer × 24 layers = ~1.6M (scan params only). Remaining params
in proj_up/down, W_seed, W_w, W_pcm, W_enc, lm_head, and memory params.

**Stage 3 a'_t derivation:**
Same structure as Stage 1. `a'_t` and `b'_t` are derived from the integrated
signal (pm_read + em_trail + cum_em) via learned projections (separate from
Stage 1's projections).

**z_hat carry (PCM surprise):**
z_hat is computed post-scan as `W_pcm(H)`, not carried in scan state. Surprise
is a shifted comparison: `surprise[:, 1:] = z_hat[:, :-1] - z[:, 1:]`. Position
0 has zero surprise (no prior prediction). This avoids enlarging the scan state.

**route_aggregated in EM commit:**
`route_aggregated = route.mean(dim=seq_dim)` — average routing weight per
primitive across all N tokens. Shape: `[BS, B, M]`. Used to scale the EMA
learning rate per primitive.

**PCM gain modulation:**
Applied to H before Stage 2: `H = H * (1 + 0.1 * tanh(W_gain(surprise)))`.
Bounded [0.9, 1.1]. W_gain is `[D_col, D_col]` per column. Slight
amplification of surprising features, suppression of predicted features.

### Remaining Open

- **Tier presets**: Need recalculation for scan architecture. Anchor on Mamba-130M
  total param count for fair comparison. Scan params (layer count, state expansion)
  differ from v4's FFN stacks.

## Comparison: v4 → v5

| Aspect | v4 (Iterative) | v5 (Scan-Memory-Scan) |
|--------|---------------|----------------------|
| Token processing | R passes, all positions parallel | 2 causal scans, parallel-scan |
| Cross-token context | Memory-mediated (R hops) | Scan recurrence (causal) + memory |
| Training objective | FITB (masked prediction) | NTP (next-token prediction) |
| Token partitioning | Interleaved (column c gets tokens c, c+C, ...) | All columns see every token (D_col slices) |
| Fan-out/fan-in | Required (D ↔ G×D_col) | Not needed (columns = D_col slices of D) |
| Memory within segment | Updates R times (between passes) | Frozen reads + causal write buffers (prefix sum) |
| Surprise | Scalar (cross-pass: ‖z^r - z_hat^{r-1}‖) | Vector (within-scan: z_hat_{t-1} - z_t) |
| Memory addressing | Block-level D, sliced per column | Column-local D_col slices of bank |
| Serial depth | O(R × 2L) FFN layers | O(L × log N) parallel-scan |
| PM | Holographic slots (r slots, routing) | Bias vector (1 vector, gain mod) |
| EM read | Top-k retrieval (hard selection) | Trail-based composition (soft, iterative) |
| EM write | Store full vector in one slot | Decompose across primitives (soft routing) |
| EM capacity | M independent facts | M primitives → ∞ compositions (continuous seeds) |
| Credit assignment | Through top-k mask (sparse grad) | Through softmax (trail) + prefix sum (writes) |
| Write gradient | Cross-segment only (TBPTT) | Same-segment (write buffer) + cross-segment (commit) |
| Memory state vs params | State (PM slots, EM slots) | State (pm_bias, em_K/V primitives) |
| GPU utilization | ~7.4% (narrow bmm, R-loop) | Expected much higher (scan kernels) |
| Compute per token | ~79M FLOPs (Tier A) | ~2× scan layer + trail + PM gain |
