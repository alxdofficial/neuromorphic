# Architecture v4: Iterative Refinement with Cortical Columns

## The Core Insight

We process all tokens in a segment **simultaneously** through R iterative passes.
Each pass, every cortical column reads from shared memory (PM/EM), processes its
token, and produces a richer representation. **After each pass, PM/EM update** —
absorbing what the columns found surprising or novel. The next pass reads the
updated memory, giving every token indirect access to what other tokens contributed.

PCM measures **cross-pass surprise**: how much did this token's representation
change now that memory is richer? High surprise = updated memory significantly
changed understanding of this token = worth further commitment.

R passes = R rounds of both **perception and memory update**, all within one
segment of N tokens. Each pass sees richer PM/EM because previous passes
committed their findings.

There is **no within-segment sequential scan**. All tokens in a pass see the
same PM/EM state. Within-segment order comes from positional encoding.
Within-segment *context* comes from PM/EM updates between passes — each pass's
commits are visible to the next pass's reads. Over R passes, information flows
between tokens through the PM/EM bottleneck.

If prototyping reveals that direct within-segment causal context is necessary,
a small learned recurrent state can be added to a per-token scan carry. But we
start without it.

## Why This Architecture Exists

We face a fundamental tension:

1. **Softmax attention**: KV cache grows linearly with context length, and
   O(N²) compute per layer during prefill — too expensive for lifelong learning
2. **SSMs / affine recurrence**: our bio-inspired memory updates (Hebbian PM,
   episodic EM) involve non-affine decisions (thresholds, similarity matching,
   slot selection) — can't naively parallelize
3. **Sequential processing**: O(N) serial depth — underutilizes GPU

Our resolution: **decouple token processing from memory accumulation.**

- **Token processing** (cortical columns): embarrassingly parallel over B × N.
  Each column reads memory, processes one token, outputs features. No
  cross-token operations. No sequential dependency.
- **Memory accumulation** (PM/EM updates): happens between passes. Hebbian
  eligibility, novelty scoring, neuromodulated commits — all the complex
  bio-inspired logic runs here, unconstrained by parallelism.
- **Cross-pass refinement** (PCM + PM/EM): R passes let each token's
  representation and the shared memory improve iteratively together.

## Architecture

### Components

**Cortical columns** (narrow, many, independent parallel processors):
- **PCM** — local predictive coding (per-column cross-pass prediction + surprise)
- **FFN** — local feature transformation (per-column weights, narrow D_col)
- Columns read from shared memory, compute surprise, do "thinking," and
  accumulate eligibility/novelty signals for memory updates.

Input shape: `[BS, N, D]` → project → `[BS, N, B_blocks, C, D_col]`
Each column operates on D_col dimensions. After processing, project back to D.

**Shared memory systems** (per-block, updated between passes):
- **PM** — procedural memory: Hebbian holographic slots (basal ganglia)
- **EM** — episodic memory: key-value episode buffer (hippocampus)

PM/EM have their own dimension D_mem (decoupled from D_col). Columns project
up to D_mem for reads and write candidates, project back down to D_col.

### How It Works (High Level)

```
For r = 1..R passes:
  1. All B columns process all N tokens in PARALLEL
     - read from PM/EM (same state for all positions this pass)
     - PCM encodes token → compare with previous pass → surprise
     - surprise gates PM eligibility accumulation + EM novelty scoring
     - FFN processes (modulated by surprise)
  2. PM/EM UPDATE (neuromodulators commit eligibility, write novel episodes)
     - next pass reads the UPDATED PM/EM

After R passes: decode from final representations → next-token loss
Carry PM/EM state to next segment.
```

Step 1 is embarrassingly parallel (B × N independent computations).
Step 2 is the only sequential-across-passes part (R times per segment).

### Forward Pass: R Iterative Passes

```python
def forward_segment(self, input_ids, reset_mask=None):
    """Process N tokens through R refinement passes.

    input_ids:  [BS, N]     — segment of token IDs
    reset_mask: [BS] bool   — streams to reset PM/EM (doc boundary)

    Returns: (logits [BS, N, vocab], aux_loss scalar)
    PM/EM state is mutated in-place (instance attributes on each block).
    """
    if reset_mask is not None and reset_mask.any():
        self._reset_memory(reset_mask)  # zero PM/EM for masked streams

    x = embed(input_ids) + pos_embed(positions)  # [BS, N, D]

    # Project to column space: [BS, N, D] → [BS, N, B_blocks, C, D_col]
    x_blocks = fan_out(x).view(BS, N, B_blocks, C, D_col)

    z_hat_prev = [None] * B_blocks  # per-block PCM state
    lam = sigmoid(lambda_logit)     # learnable mixing parameter

    for r in range(R):
        block_outputs = []
        z_hat_new = []

        for b, block in enumerate(self.blocks):
            x_b = x_blocks[:, :, b]  # [BS, N, C, D_col]

            # --- STEP 1: COLUMN FORWARD (all C columns × N tokens) ---
            x_out, z, z_hat, pcm_loss, elig, em_cands = \
                block.forward_pass(x_b, z_hat_prev[b])

            aux_loss += pcm_loss * pcm_pred_weight

            # --- STEP 2: PM/EM UPDATE (between passes) ---
            block.commit_and_write(elig, em_cands)

            block_outputs.append(x_out)
            z_hat_new.append(z_hat)

        x_new = stack(block_outputs, dim=2)  # [BS, N, B, C, D_col]

        # Damped mixing
        if r > 0:
            x_blocks = (1 - lam) * x_blocks + lam * x_new
        else:
            x_blocks = x_new

        z_hat_prev = z_hat_new

    # Project back: [BS, N, B*C*D_col] → [BS, N, D]
    x = fan_in(x_blocks.reshape(BS, N, -1))
    logits = lm_head(ln_final(x))  # [BS, N, vocab]

    return logits, aux_loss
```

### Per-Block Column Forward (The Parallel Core)

In code, this is split across two classes:
- **`CorticalColumnGroup.forward`**: column-level ops (PM/EM reads, PCM, FFN,
  eligibility candidates, novelty candidates). Operates on `[BS, N, C, D_col]`.
- **`ColumnBlock.forward_pass`**: calls `CorticalColumnGroup.forward`, then
  aggregates PM eligibility across N×C positions and selects top EM candidates.

Conceptually, for one block:

```python
def block_forward_pass(x_block, pm_state, em_state, z_hat_prev):
    """One block: all C columns process all N tokens. Embarrassingly parallel.

    x_block:    [BS, N, C, D_col]     — column representations for this block
    pm_state:   PM slots for block     — same for all positions this pass
    em_state:   EM slots for block     — same for all positions this pass
    z_hat_prev: [BS, N, C, D_pcm]     — PCM predictions from prev pass

    Returns: x_out, z, z_hat, pcm_loss, pm_elig, em_cands
    """
    # --- CorticalColumnGroup.forward ---

    # 1. PM holographic read (project up to D_mem, modulate, project back)
    q_pm = W_pm_up(x_block)                             # [BS,N,C,D_mem]
    y_pm = pm_state.read(q_pm)                           # holographic
    x_block = x_block + W_pm_down(y_pm)                  # residual

    # 2. EM top-k read (project up, retrieve, cross-attend, project back)
    q_em = W_em_up(x_block)                              # [BS,N,C,D_mem]
    y_em = em_state.read(q_em)                           # top-k retrieval
    x_block = x_block + W_em_down(y_em)                  # residual

    # 3. PCM: encode, surprise, gain
    z = PCM.encode(x_block)                              # [BS,N,C,D_pcm]
    if z_hat_prev is not None:
        delta = z - z_hat_prev.detach()
        surprise = norm(delta, dim=-1) / sqrt(D_pcm)     # [BS,N,C]
        gain = 1 + 0.1 * tanh(W_gain(delta))             # [BS,N,C,D_col]
    else:
        surprise = zeros; gain = ones                     # pass 1

    # 4. FFN with gain modulation (gain is per-dimension, not scalar)
    h = LayerNorm(x_block) * gain                        # element-wise
    x_out = x_block + ffn_down(gelu(ffn_up(h)))          # residual

    # 5. PCM hypothesis
    z_hat = PCM.predict(z)                               # [BS,N,C,D_pcm]

    # 6. PM eligibility candidates
    k_cand = normalize(W_k_pre(x_out))                  # [BS,N,C,D_mem]
    v_cand = W_v_post(x_out)                             # [BS,N,C,D_mem]
    gate = (surprise / scale).clamp(0, 1)                # [BS,N,C]

    # 7. EM novelty candidates
    q_nov = normalize(W_k_cand(x_out))                  # [BS,N,C,D_mem]
    v_nov = W_v_cand(x_out)                              # [BS,N,C,D_mem]
    w_nov = sigmoid(W_nov(x_out))                        # [BS,N,C]

    # --- ColumnBlock.forward_pass (aggregation) ---

    # 8. PM eligibility aggregation across N×C
    route_w = softmax(k_cand @ pm_K.T / tau)             # [BS,N,C,r]
    gated = gate * route_w                               # [BS,N,C,r]
    elig_K = einsum('bncr, bncd -> brd', gated, k_cand)  # [BS,r,D_mem]
    elig_V = einsum('bncr, bncd -> brd', gated, v_cand)

    # 9. EM candidate selection (top-C_cand across N×C by novelty)
    novelty = w_nov * surprise + (1-w_nov) * (1 - max_sim)
    em_cands = topk(novelty.flatten(N*C), C_cand)

    # 10. PCM prediction loss
    pcm_loss = MSE(z_hat_prev, z.detach()) if z_hat_prev else 0

    return x_out, z, z_hat, pcm_loss, (elig_K, elig_V), em_cands
```

**CRITICAL**: No cross-token operations inside column forward. Each position
is fully independent. This is what makes C × N parallelism possible within
each block. The aggregation step (8-9) sums across positions but doesn't
affect the per-token output.

**Memory reads ARE attention** — but over memory slots, not over other tokens.

**Ordering matters**: PCM surprise must be computed before eligibility and
novelty candidates, since surprise gates both.

### PM Holographic Read (Why "Holographic")

Standard memory: query → retrieve stored vector. Output is independent of input.

PM holographic read: the input **flows through** stored patterns. Output depends
on both input AND stored pattern multiplicatively:

```
q = W_pm_up(x_col)                              # [D_col] → [D_mem]
scores = normalize(q) @ pm_K.T                   # [r_slots]
y_pm = sum_i(pm_a_i * scores_i * q * pm_V_i)    # [D_mem]
y_pm_out = W_pm_down(y_pm)                       # [D_mem] → [D_col]
```

Mathematically: `y_d = x_d * [W @ x]_d` — quadratic in x. The stored pattern
**modulates** the input rather than replacing it. Like a hologram: the stored
pattern is an interference pattern, the input is the reference beam, you need
both to reconstruct the output.

This makes PM a **learned transformation** — each slot stores a "skill" (how to
process inputs), not a "fact" (a fixed vector). Same slot, different input →
completely different output. This is why PM is "procedural memory": it encodes
*how to process*, not *what was seen*.

### EM Read (Key-Value Retrieval)

```
q = W_em_up(x_col)                              # [D_col] → [D_mem]
scores = q @ em_K.T                              # [M_slots]
top_k = topk(scores, k=4)                       # sparse retrieval
y_em = cross_attention(q, em_K[top_k], em_V[top_k])  # [D_mem]
y_em_out = W_em_down(y_em)                       # [D_mem] → [D_col]
```

Standard key-value store with sparse top-k retrieval. EM stores "facts" and
"episodes" — specific things that were seen. The output IS a stored vector
(weighted combination of retrieved values), independent of the query beyond
selection. This is the fundamental difference from PM.

### Cross-Pass PCM (Predictive Coding)

PCM predicts what each token's representation will look like in the **next pass**
(after PM/EM have updated and the column re-processes with richer memory).

```
Pass 1: column encodes token → z^1. Predicts z_hat^1.
         No surprise yet (no previous prediction to compare against).
         PM/EM update with unmodulated eligibility/novelty.

Pass 2: column encodes token → z^2 (now reading UPDATED PM/EM).
         Surprise = ||z^2 - z_hat^1||.
         "Did the PM/EM update change my understanding of this token?"
         Predicts z_hat^2.
         PM/EM update with surprise-modulated eligibility/novelty.
...
Pass R: column encodes token → z^R. Surprise = ||z^R - z_hat^{R-1}||.
         Final PM/EM update. Decode.
```

**What surprise means in v4**: "now that memory has been updated (other tokens
committed their findings), did my understanding of THIS token change?" High
surprise = this token's meaning depends heavily on context that wasn't in
memory before = it's informationally dense.

**PCM optimization**: trained by both:
1. Auxiliary prediction loss: `L_pred = MSE(z_hat^r, z^{r+1}.detach())`
2. Downstream: surprise → PM eligibility gating → PM commits → next pass's
   PM reads → loss. All within the same segment's forward pass.

## Within-Segment Context via PM/EM Updates

Although there is no direct token-to-token communication, tokens gain indirect
access to each other through **PM/EM updates between passes**:

```
Pass 1: all tokens process with PM/EM_0 (from previous segment).
         Token 42 is surprising → high eligibility → committed to PM.
         Token 99 is novel → written to EM.
         → PM/EM_1

Pass 2: all tokens process with PM/EM_1.
         Token 7 now reads PM and retrieves what token 42 committed.
         Token 50 now reads EM and retrieves token 99's episode.
         Their representations change → new surprise → new commits.
         → PM/EM_2

...and so on for R passes.
```

This is **memory-mediated within-segment context**. Information flows between
tokens through the PM/EM bottleneck, not through direct attention. Each pass
adds one hop of indirect communication. After R passes, information has had
R opportunities to propagate between tokens via memory.

The bottleneck is deliberate: it forces the model to compress and prioritize,
like biological memory. Not every token's information survives — only what the
neuromodulators decide is worth committing.

**Remaining limitation**: within a single pass, all positions see the same PM/EM.
Position ordering within a pass comes only from positional encoding. The causal
ordering emerges across passes as PM/EM accumulate.

## Gradient Flow

### Within-Segment (Pass-to-Pass)

```
loss → decode → x^R → column_R(x^{R-1}, PM/EM_{R-1})
  → PM.read → PM/EM_{R-1}
  → pm_commit(PM/EM_{R-2}, elig_{R-1}, neuromod_{R-1})
  → neuromodulator_{R-1} gets gradient!
  → elig_{R-1} → column_{R-1} outputs → ...back to pass 1
```

The neuromodulator at pass r gets gradient from pass r+1's loss contribution,
**within the same segment**. No multi-segment TBPTT needed for neuromodulator
learning. Each pass is a mini perception-and-commit cycle.

### What Gets Gradient From Where

| Component | Gradient source | Path |
|-----------|----------------|------|
| Column FFN | Same pass's contribution to loss | Direct through decode |
| PCM encode/hypothesis | L_pred (auxiliary) + downstream surprise | Same segment |
| PM read projections | Same pass's loss through PM read | Direct |
| EM read projections | Same pass's loss through EM read | Direct |
| PM neuromodulator | **Next pass's** loss through committed PM state | Pass r → r+1 |
| EM neuromodulator | **Next pass's** loss through written EM state | Pass r → r+1 |
| Eligibility projections | **Next pass's** loss through committed eligibility | Pass r → r+1 |

All gradient paths are within the same segment's forward/backward pass. The
computation graph spans R passes of column processing + R-1 PM/EM commits.

### Cross-Segment Gradient

For the neuromodulator at the **final pass** (pass R), gradient comes from the
**next segment** — the committed PM/EM_R is read by the next segment's columns.
This requires TBPTT chunks spanning at least 2 segments:

```
TBPTT chunk = K segments (e.g., K=2-4)
Segment 1: R passes → PM/EM updated R times → carry to segment 2
Segment 2: R passes → loss flows back through PM/EM → segment 1's final commit
```

K=2 is likely sufficient (the final pass's neuromodulator only needs 1 segment
of lookahead). Internal passes (1..R-1) get gradient from within their own
segment.

## Correctness Analysis

### 1. Causality

Causality is maintained at two levels:
- **Across segments**: PM/EM state at segment k reflects only tokens from
  segments 0..k-1 and the current segment's completed passes.
- **Across passes**: PM/EM at pass r reflects all N tokens from passes 1..r-1.
  All tokens within pass r see the same PM/EM_r (no within-pass ordering).

Within a single pass, there is no causal ordering — all positions are equivalent
(modulo positional encoding).

### 2. Train/Inference Equivalence

**Training**: process N tokens simultaneously for R passes, decode all.
**Inference**: process one token at a time for R passes, decode.

Equivalent because no cross-token operations within a pass. At inference, PM/EM
still update after each pass (single token's eligibility/novelty may or may not
trigger commits — the neuromodulator decides).

Inference memory: PM + EM + per-column PCM z_hat + eligibility accumulators.
**Constant**, does not grow with sequence length.

### 3. What R=1 Is

R=1: single pass. No cross-pass surprise (no previous prediction). No PM/EM
update within the segment (only carries forward to next segment). This is a
**feedforward model with memory reads** — similar to retrieval-augmented generation.

R=2+: cross-pass surprise kicks in. PM/EM update between passes. Tokens gain
indirect access to each other through memory. "Depth" emerges.

## Practical Design Decisions

### Column Dimensions

```
D = 768           # embedding / model dimension
B_blocks = 6      # blocks (each with own PM + EM)
C = 4             # columns per block (24 total)
D_col = 128       # column width
D_mem = 256       # PM/EM slot dimension (decoupled from D_col)
D_pcm = 64        # PCM encoding dimension

Fan-out:  D → B_blocks × C × D_col = 768 → 3072  (~4x)
Fan-in:   B_blocks × C × D_col → D = 3072 → 768
```

### Segment Length

Segment length N determines how many tokens contribute to each round of R
PM/EM updates. N=128 means each pass sees 128 tokens of eligibility/novelty
before committing. Shorter N = more frequent cross-segment updates.

### Damped Pass-to-Pass Mixing

    x^r = (1 - λ) * x^{r-1} + λ * column_output^r

λ can be learned or per-pass schedule. Start with λ=0.5.

### Token Subsampling

Earlier passes can process subsets of tokens:

```
Pass 1: process 25% of tokens (strided)
Pass 2: process 50%
...
Pass R: process ALL tokens, decode, take loss
```

Skipped tokens retain their representation unchanged. PM/EM still update
from the processed subset's eligibility/novelty.

**Edge cases:**
- Deterministic schedule at inference (strided). Random only during training.
- Per-pass auxiliary loss to prevent "early passes don't matter" collapse.

### Scaling: Blocks of Columns

```
Block b: C columns share PM_b (r_slots × D_mem) + EM_b (M_slots × D_mem)
```

Total columns = B_blocks × C. Each block's PM/EM can specialize. Analogous
to different brain regions maintaining different types of memories.

## Complexity

### Serial Depth

Per pass: O(1) — all B × N computations are parallel.
PM/EM commit between passes: O(1) (small, not N-dependent).
Total: O(R).

Compare:
- Current (v1):       O(K×P) = 8×32 = **256 steps**
- Transformer (L=6):  O(L) = **6 steps** (but O(N²) compute per step)
- Mamba (L=6):        O(L×N) (sequential scan per layer)
- **v4 (R=6)**:       O(R) = **6 steps** (and O(N×D²) compute per step)

### Compute

Per pass: O(N × B_blocks × C × D_col²) for column FFNs
        + O(N × B_blocks × r × D_mem) for PM reads
        + O(N × B_blocks × k × D_mem) for EM reads
Total: R × above

### Memory

Training: O(R × N × D) for intermediate representations.
With gradient checkpointing: O(N × D) + recompute per pass.
Inference: O(PM + EM + B × D_pcm) — **constant**.

## Training

### Loss

    L = (1/N) Σ_t CE(decode(x^R_t), token_{t+1}) + α * L_pred

where L_pred = Σ_r MSE(z_hat^r, z^{r+1}.detach()) averaged over passes.

### TBPTT Chunks

Process K segments per chunk (K=2-4). Backprop through all K × R passes.
PM/EM state detached at chunk boundaries.

### Phase A: Per-Document Learning

PM/EM reset at document boundaries. Neuromodulators learn when to commit.

### Phase B: Lifelong Adaptation

PM/EM persist across documents within a stream. Only eligibility accumulators
and PCM z_hat reset at doc boundaries. The distillation cycle:

1. New domain → surprise spikes (PCM cross-pass surprise)
2. PM eligibility accumulates (gated by surprise)
3. PM neuromodulator commits at each pass boundary
4. EM writes novel episodes at each pass boundary
5. Over time, slow weights learn domain → surprise drops
6. Slow weights = general; PM/EM = domain-specific

## Memory System Learning Mechanisms

### PM Eligibility Accumulation

Each token, each pass, each column accumulates eligibility:

```
# Project to D_mem for compatibility with PM slots
k_cand = normalize(W_k_pre(x_col))         # → [D_mem]
v_cand = W_v_post(x_col)                   # → [D_mem]
gate = (surprise / 5.0).clamp(0, 1)        # third factor
route_w = softmax(pm_K @ k_cand / tau)     # [r_slots] — which slot?

# Accumulate across all N tokens in this pass
elig_K += gate * route_w ⊗ k_cand          # [r_slots, D_mem]
elig_V += gate * route_w ⊗ v_cand
```

At the end of each pass, the neuromodulator decides how much to commit:

```
(g, slot_logits, tau) = PM_neuromodulator(elig_summary)
slot_weights = softmax(slot_logits / tau)   # which slots to update
pm_K = (1 - g * slot_weights) * pm_K + g * slot_weights * elig_K
pm_V = (1 - g * slot_weights) * pm_V + g * slot_weights * elig_V
```

Eligibility accumulators reset after each commit (each pass gets fresh traces).

### EM Novelty Accumulation

Each token, each pass, each column scores novelty:

```
novelty = w_nov * surprise + (1 - w_nov) * (1 - max_cosine_sim(q, em_K))
```

Top-C candidates (highest novelty) are buffered per pass. At the end of each
pass, the neuromodulator decides writes:

```
(g_em, tau, decay) = EM_neuromodulator(novelty_mean, em_usage, content)
slot_scores = candidates @ em_K.T / tau   # soft slot selection
slot_weights = softmax(slot_scores)
em_K[slot] = (1 - g_em * slot_weights) * em_K[slot] + g_em * slot_weights * candidate_K
em_V[slot] = (1 - g_em * slot_weights) * em_V[slot] + g_em * slot_weights * candidate_V
em_S *= decay                             # strength decay before write
```

### Summary

| Component | Where | When | Parallel? |
|-----------|-------|------|-----------|
| PM holographic read | Column | Every token, every pass | Yes (B×N) |
| EM top-k read | Column | Every token, every pass | Yes (B×N) |
| PCM encode + surprise | Column | Every token, every pass | Yes (B×N) |
| FFN | Column | Every token, every pass | Yes (B×N) |
| PM eligibility accumulate | Column | Every token, every pass | Yes (B×N) |
| EM novelty score | Column | Every token, every pass | Yes (B×N) |
| PM neuromod commit | Between passes | Once per pass | Small |
| EM neuromod write | Between passes | Once per pass | Small |

## Prototyping Plan

### Prototype 0: Columns + PCM only (no PM/EM)

- B columns with FFN + PCM, R passes, no memory reads or updates
- Goal: can R-pass refinement with cross-pass surprise learn language at all?

### Prototype 1: Add PM/EM reads (frozen)

- PM/EM initialized, frozen (no updates between passes)
- Goal: does reading from structured memory improve perplexity?

### Prototype 2: Full system

- PM/EM updated between passes via neuromodulators
- Full Hebbian eligibility + novelty writes
- Goal: demonstrate the iterative perception + memory update loop

### Prototype 2.5: Optional within-segment scan

- Add small recurrent scan state if within-pass token context proves necessary
- Goal: handle local syntactic tasks that pure PM/EM can't cover

### Measure

1. tokens/sec at fixed batch and N
2. peak training memory (GB)
3. validation loss vs baseline (RWKV-small or tiny transformer)

## Open Questions

1. **Optimal R**: Start R=4-6. Per-pass auxiliary loss for ablation.

2. **Shared vs per-pass column weights**: Shared = true iterative refinement.
   Separate = more capacity. Hybrid possible.

3. **Eligibility reset between passes**: Currently reset after each commit. Could
   carry eligibility across passes (exponential decay) for multi-pass patterns.

4. **Within-segment context**: Is memory-mediated context (R hops through PM/EM)
   sufficient? Or do we need direct token interaction? Empirical.

5. **Block/column scaling**: How many blocks × columns per block?

6. **PCM prediction target**: Predict next pass's encoding, final pass's encoding,
   or contrastive loss?

7. **Pass 1 (no surprise)**: What should gate eligibility/novelty in pass 1? Options:
   a) No gating (accumulate everything), b) Use a fixed prior, c) Skip accumulation.
