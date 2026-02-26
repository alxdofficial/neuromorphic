# Architecture v4: Iterative Memory Refinement via Causal Scan

## The Core Insight

Instead of processing tokens sequentially to maintain a perfect running memory
state, we let the model build an **imperfect but causal** view of the sequence
in parallel, then **refine it iteratively** — trading exactness for parallelism
while keeping memory bounded.

The current architecture insists: "memory at token t must reflect a fully
processed, sequential pass through tokens 0..t-1." This forces O(N) serial depth.

The v4 insight: that's unnecessarily strict. The model can work with an
**approximate memory that improves over R passes**. Each pass is parallel, each
pass's state is causal, and R passes is much less serial depth than N sequential
steps. The memory systems don't need a perfect sequential view — they need a
*good enough* view, refined iteratively.

## Why This Architecture Exists

We face a fundamental tension:

1. **Softmax attention**: O(N²) memory — too expensive for lifelong learning
2. **SSMs / affine recurrence**: our bio-inspired memory updates (Hebbian PM,
   episodic EM) involve non-affine decisions (thresholds, similarity matching,
   slot selection) — can't naively parallelize
3. **Sequential processing**: O(N) serial depth — underutilizes GPU

Our resolution: **separate the decision from the update.**

- The **decision** (which PM slot to update, whether to write to EM, how much
  to gate WM) can be arbitrarily complex and non-affine — it's computed by
  the cortical column's FFN, which reads from memory and does Hebbian/episodic
  logic internally
- The **update itself** (the actual state mutation) is always affine:
  `S_new = A * S_old + b`, where A and b are the OUTPUT of the decision

Since the updates are affine, they compose associatively, enabling parallel
prefix scan. Since the decisions are unconstrained, we preserve the full
expressiveness of our bio-inspired memory mechanisms.

## Architecture

### Components

**Cortical columns** (B copies, independent parallel processors):
- **PCM** — local predictive coding (per-column predictions + surprise)
- **FFN** — local feature transformation (per-column weights)
- Columns read from shared memory, compute surprise, do "thinking," and
  decide how to update memory. This is the non-affine, complex part.

**Shared memory systems** (biologically: centralized infrastructure that
columns read from and write to):
- **PM** — procedural memory: Hebbian association slots (basal ganglia)
- **EM** — episodic memory: episode buffer (hippocampus)
- **WM** — working memory: recent-token buffer (prefrontal cortex)

### How It Works (High Level)

```
1. All N tokens are presented simultaneously
2. Each cortical column reads from frozen memory, does its processing,
   and decides how memory should be updated → outputs (A, b) per token
3. These (A, b) updates are composed causally via prefix scan:
   memory state at position t = accumulated updates from tokens 0..t-1
4. Repeat for R passes — each pass, columns read richer causal states
5. After R passes, decode from final causal states → next-token loss
```

The broad parallel sweep (step 2) is embarrassingly parallel.
The causal composition (step 3) is O(log N) via scan.
Repeated R times (step 4) gives depth of contextual understanding.

### Forward Pass: R Iterative Passes

```python
def forward_segment(tokens, S_init):
    """Process N tokens through R refinement passes.

    tokens:  [BS, N]       — full segment of tokens
    S_init:  memory state  — from previous segment (constant memory)

    Returns: logits [BS, N, vocab], final memory state
    """
    # Pass 1: all tokens see S_init (no within-segment context yet)
    # Pass r>1: each token sees its causal state from pass r-1
    S_prev = broadcast(S_init, N)  # [BS, N, state_size]

    for r in range(R):
        # Step 1: PARALLEL — each column reads memory, does processing,
        # decides on memory updates. Arbitrarily non-linear internally.
        # No cross-token operations. Embarrassingly parallel over B × N.
        A, b = column_updates_r(tokens, S_prev)

        # Step 2: SCAN — compose updates causally via prefix scan.
        # S[t] includes updates from tokens 0..t-1 only (exclusive).
        # O(log N) serial depth.
        S_curr = exclusive_affine_scan(A, b, S_init)

        S_prev = S_curr  # next pass reads this pass's causal states

    # Decode from final pass's causal states
    logits = decode(tokens, S_curr)  # [BS, N, vocab]

    # Next segment's S_init = state after all N tokens
    S_final = A[:, -1] * S_curr[:, -1] + b[:, -1]

    return logits, S_final
```

### Per-Token Column Update (The Non-Affine Decision)

```python
def column_updates_r(tokens, S):
    """Each cortical column reads memory, processes, decides on updates.

    The internals are arbitrarily non-linear (FFN, Hebbian rules,
    surprise thresholds, cosine similarity, softmax slot selection).
    The OUTPUT is affine parameters (A, b) for the scan.

    tokens: [BS, N]              — token ids
    S:      [BS, N, state_size]  — per-position causal state (from prev pass)

    Returns: (A, b) — structured affine update descriptors per token
    """
    x = embed(tokens)  # [BS, N, D]

    # Memory reads — content-addressed lookups (attention over slots, not tokens)
    # Each position reads from its OWN causal state (different per position)
    x = x + PM.read(x, S.pm)   # Hebbian association retrieval
    x = x + EM.read(x, S.em)   # episodic memory retrieval
    x = x + WM.read(x, S.wm)  # working memory retrieval

    # Per-column processing
    surprise = PCM.surprise(x, S.pcm)  # local prediction error
    x = FFN(x, surprise)               # per-column transformation

    # --- Decide memory updates (non-affine, complex) ---

    # PM (Hebbian): compute eligibility, select slots, determine values
    pm_eligibility = hebbian_rule(x, S.pm)
    pm_gate = sigmoid((pm_eligibility - threshold) / temp)  # soft threshold
    pm_slot_scores = softmax(slot_similarity(x, S.pm))      # which slot
    pm_value = project_pm(x)                                 # what to write

    # EM (episodic): surprise-based write decision
    em_gate = sigmoid((surprise - em_threshold) / temp)
    em_episode = compress_episode(x)
    em_slot = softmax(write_scores(S.em))                    # where to write

    # WM: recent-token update
    wm_gate = sigmoid(project_wm_gate(x))
    wm_content = project_wm(x)
    wm_slot_scores = softmax(slot_scores(x))

    # PCM: prediction update (EMA)
    pcm_alpha = sigmoid(project_alpha(x))
    pcm_target = project_pcm(x)

    # --- Pack into structured (A, b) descriptors ---
    # Small descriptors, expanded to full state via outer products / broadcasting
    A, b = pack_affine_updates(
        pm_gate, pm_slot_scores, pm_value,
        em_gate, em_slot, em_episode,
        wm_gate, wm_slot_scores, wm_content,
        pcm_alpha, pcm_target,
    )

    return A, b  # affine params, ready for scan
```

**CRITICAL**: No cross-token operations inside this function. No attention over
the token axis, no convolution, no batch norm across tokens. Each position is
fully independent. Only LayerNorm (per-position) is allowed. This guarantees
causality.

**Memory reads ARE attention** — but over memory slots, not over other tokens.
The claim is "no self-attention over the token axis," not "no attention at all."

### Structured Update Descriptors (Factored, Not Dense)

The column does NOT produce a dense state-sized vector. It produces small
structured descriptors that expand into full (A, b) cheaply:

```python
def pack_affine_updates(pm_gate, pm_slots, pm_val, ...):
    """Convert small decision outputs into full affine parameters.

    Example for WM with W=64 slots, D=768:
      Input:  wm_gate (scalar), wm_slot_scores (R^64), wm_content (R^768)
      Expand: A[slot_k, :] = 1 - wm_gate * wm_slot_scores[k]    # broadcast
              b[slot_k, :] = wm_gate * wm_slot_scores[k] * wm_content

    Column projection: D → ~1700 floats (not D → 100K)
    Expansion: ~1700 → full state_size via outer products / broadcasting
    """
    ...
```

| Module | Descriptor size | Full state size | Expansion |
|--------|----------------|-----------------|-----------|
| PM | r + D + 1 = 785 | r × D = 12,288 | gate × slot_scores ⊗ value |
| EM | M_write + D_em + 1 ≈ 200 | M × D_em = 131,072 | gate × slot ⊗ episode |
| WM | W + D + 1 = 833 | W × D = 49,152 | gate × slot_scores ⊗ content |
| PCM | D_h + 1 = 49 | D_h = 48 | alpha blend |
| **Total** | **~1,870** | **~193K** | |

Projection is D=768 → 1,870 (a small linear layer), not D → 386K.

### Causal Affine Scan

```python
def exclusive_affine_scan(A, b, S_init):
    """Compute causal memory state at every position via EXCLUSIVE prefix scan.

    S[t] includes updates from positions 0..t-1, NOT position t itself.
    The recurrence: S_{t+1} = A_t ⊙ S_t + b_t  (element-wise gating)
    Affine composition is associative: (A2,b2)∘(A1,b1) = (A2⊙A1, A2⊙b1+b2)

    Returns: S[BS, N, state_size] — all N causal states in parallel
        S[0] = S_init
        S[1] = A_0 ⊙ S_init + b_0
        S[t] = compose(Δ_0, ..., Δ_{t-1})(S_init)

    Parallel: O(N log N) work, O(log N) depth
    Differentiable: backward pass is also a scan
    """
    ...
```

## Correctness Analysis

### 1. Causality Proof (by induction)

**Claim**: At position t, pass r, memory state S_t^r contains information
ONLY from tokens 0..t-1. Never from t or later.

**Base case (pass 1):** All tokens compute updates from S_init (predates
current segment). Token i's update depends only on token_i + S_init. The
exclusive scan accumulates positions 0..t-1 only, by construction. ✓

**Inductive step:** Assume S_i^r has info from tokens 0..i-1 only (all i).
Token i computes Δ_i^{r+1} from (token_i, S_i^r) — info from tokens 0..i.
Exclusive scan accumulates j=0..t-1 → S_t^{r+1} has tokens 0..t-1 only. ✓

**Critical requirement**: column function has NO cross-token operations.

### 2. Train/Inference Equivalence

**Inference** maintains R running states. Pass 1 always reads S_init (fixed
per segment). Pass r reads running state from pass r-1. Decode BEFORE update
(exclusive semantics).

```python
def inference_step(token, S_init, states):
    """Process one token. states = list of R running states."""
    logits = decode(token, states[R-1])  # decode FIRST (exclusive)

    (A1, b1) = column_1(token, S_init)   # pass 1: reads S_init always
    (A2, b2) = column_2(token, states[0]) # pass 2: reads pass-1 state
    ...
    (AR, bR) = column_R(token, states[R-2])

    states[0] = A1 * states[0] + b1       # advance all states
    states[1] = A2 * states[1] + b2
    ...
    states[R-1] = AR * states[R-1] + bR

    return logits
```

**Verified with concrete R=2 example** (see detailed proof in previous version).
Training scan and streaming inference produce identical outputs at every position.

Inference memory: (R+1) × state_size. Constant, does not grow with sequence length.

### 3. What R=1 Actually Is

R=1 is NOT a unigram model. It's a single-layer gated linear recurrence
(diagonal SSM). The update *parameters* (A_t, b_t) are context-free within the
segment (computed from token + S_init only), but the *state* S_t is
context-dependent — it's the scan accumulation of all previous tokens' updates.
Order matters (gate products affect how earlier writes survive).

- R=1: single scan layer. Tokens compress into state, but update params lack
  within-segment context. Still a real sequence model.
- R=2+: update params become contextual (computed from previous pass's causal
  states). This is where "deep" behavior emerges.

### 4. Gradient Flow

```
loss@t → decode → S_t^R → (pass-R scan) → Δ_j^R for j<t
  → column_R(token_j, S_j^{R-1}) → (pass-(R-1) scan) → ...
  → column_1(token_m, S_init) → token_m embedding
```

Gradient reaches all earlier tokens through R levels of scan chains.
Each scan backward is O(log N). Total backward depth: R × O(log N).

**Numerical concern**: gate products A_t·A_{t-1}·...·A_0 vanish if gates < 1.
Mitigations:
- Parameterize A = exp(-softplus(...)) — stable decay rate interpretation
- Initialize gates near 1 (sigmoid with positive bias, like LSTM forget gate)
- Residual paths across passes (see improvements section)
- Decode from all passes' states (multi-scale)

### 5. Document Boundary Handling

Setting A_d = 0 blocks state flow through the scan, but b_d computed from
S_d^{r-1} could carry pre-boundary info.

**Fix**: At document boundaries, the column must read a RESET state (zeros or
fresh S_init) instead of S_d^{r-1}. This ensures both A_d and b_d are clean.
Alternatively, use a segmented scan with boundary markers.

### 6. Information Propagation

After R passes, token t's state carries R levels of refined context from all
preceding tokens. Analogous to R transformer layers.

Key difference: transformer passes N×D values between layers (full residual
stream). We pass state_size values (memory bottleneck). This forces compression
but limits bandwidth — we may need larger R than a transformer needs L.

This bottleneck IS the inductive bias: all information must flow through
memory, forcing the model to compress and prioritize, just like biological
memory systems.

## Practical Design Decisions

### Scan State Partitioning

Not everything needs per-token scanning. Partition by update frequency:

| In scan (per-token) | Outside scan (periodic boundary) |
|---------------------|----------------------------------|
| PCM z_hat (~768 floats) | PM slots (~12K floats) |
| WM buffer (~50K floats) | EM episodes (~131K floats) |
| Small controller state | |

PM and EM update at periodic boundaries (every 128 tokens). Their update logic
(Hebbian eligibility, surprise-based episode selection) runs at boundaries using
accumulated column outputs from the preceding window. This preserves the
existing PM/EM update mechanisms almost unchanged.

### Residual Across Passes

Instead of each pass recomputing state from S_init, use explicit residual:

    S^r = S^{r-1} + scan(Δ^r, initial=zeros)

This makes "refinement" literal (each pass adds an increment), improves
gradient flow, and stabilizes training. Each pass doesn't fight to reconstruct
everything — it just adds what was missing.

### Token Subsampling

Earlier passes can process random subsets of tokens for efficiency:

```
Pass 1: process 25% of tokens (random), scan over sparse updates
Pass 2: process 25% (different random subset), scan
...
Pass R: process ALL tokens, scan, decode, take loss
```

Each pass fills in more of the memory state. Earlier passes are cheaper.
The stochasticity acts as regularization — the model builds robust memory
representations that don't depend on any single token's contribution.
This echoes adding noise in diffusion models.

Final pass must process all tokens (for loss computation at every position).

### Per-Column vs Shared Memory in the Scan

- **PCM**: per-column (B independent scans of D_h floats — tiny, parallel)
- **WM**: could be partitioned (each column owns W/B slots) or shared
  (columns aggregate write candidates before scan)
- **PM/EM**: outside the scan, updated at boundaries. Shared across columns.
  Columns contribute write candidates; aggregation at boundary resolves conflicts.

## Complexity

### Serial Depth

Per pass: O(1) column + O(log N) scan = O(log N)
Total: R × O(log N)

R=6, N=256: 6 × 8 = **48 steps**

Compare:
- Current (v1):       O(K×P) = 8×32 = **256 steps**
- Transformer (L=6):  O(L) = **6 steps** (but O(N²) compute per step)
- Mamba (L=6):        O(L) = **6 steps** (sequential scan per layer)

### Compute

Per pass: O(N × D²) for columns + O(N log N × scan_state) for scan
Total: R × O(N × D²)

Roughly R × cost of a single transformer FFN layer, without the N² attention.

### Memory

Training: O(R × N × scan_state_size). With gradient checkpointing: O(N × scan_state_size).
Inference: O(R × state_size) — **constant**, does not grow with sequence length.

## Training Modes

### Mode 1: Causal Next-Token Prediction (Pretraining)

Standard teacher-forced CE loss at every position:

    L = (1/N) Σ_t CE(decode(token_t, S_t^R), token_{t+1})

Training memory: O(R × N × state_size) — linear in N.
Inference memory: O(R × state_size) — constant.

### Mode 2: Chunk Prediction (Post-Pretraining)

Fine-tune with iterative chunk generation:
- Process prompt tokens normally (populate memory)
- Predict next chunk of K tokens, starting from noise/zeros
- Iteratively refine predictions (diffusion-like)
- Loss on refined chunk

Uses the same R-pass machinery. Aligns training with generation use case.

## Relationship to Existing Work

This architecture IS a gated linear recurrence (diagonal SSM) parallelized via
prefix scan — squarely in the prefix-scannable model family. The novelty is:

1. **Structured memory state** (PM/EM/WM/PCM, not flat vectors)
2. **R iterative passes** with inter-pass context flow (stacked scans)
3. **Factored updates** (small descriptors expand to full state updates)
4. **Bio-inspired decision logic** (Hebbian, episodic) producing affine outputs
5. **Cortical column architecture** (B parallel processors with local PCM)

| Approach | Scan? | Multi-pass? | Structured memory? | Constant inference? |
|----------|-------|-------------|-------------------|-------------------|
| Transformer | No | Yes (L layers) | No | No (KV cache grows) |
| Mamba/S4 | Yes | No (1/layer) | No (flat state) | Yes |
| RWKV | Yes | No (1/layer) | No (flat state) | Yes |
| Universal Transformer | No | Yes (shared) | No | No |
| **v4 (this)** | **Yes** | **Yes (R passes)** | **Yes (PM/EM/WM/PCM)** | **Yes** |

## Open Questions

1. **Optimal R**: Empirical. Start with R=4-8. Ablate with auxiliary per-pass loss.

2. **Shared vs per-pass column weights**: Shared = fewer params, true iterative
   refinement. Separate = more capacity, like distinct transformer layers. Hybrid
   possible.

3. **Decode from multiple passes**: Use [S_t^1, ..., S_t^R] concatenated for
   richer decoding. No extra inference memory (already maintain R states).

4. **Scan implementation**: For N=256, a fused sequential scan kernel may beat
   parallel prefix tree (less memory movement). torch.scan (sequential, compilable)
   vs torch.associative_scan (parallel, prototype, no autograd yet) vs custom
   Triton kernel.

5. **Convergence**: Does the iterative process converge? Can we prove contraction?
   Does fixed-point theory apply?

6. **Scale of scan state vs R**: R controls depth of reasoning. state_size controls
   breadth of context. For long sequences, grow state_size. For complex reasoning,
   grow R.
