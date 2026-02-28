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
- **FFN** — two stacks of L residual FFN layers each (ffn_pre + ffn_post,
  2L total per pass). ffn_pre runs after lateral mixing with PCM gain modulation;
  ffn_post runs after cross-block mixing without gain.
- Columns read from shared memory, compute surprise, do "thinking," and
  accumulate eligibility/novelty signals for memory updates.

Input shape: `[BS, N, D]` → fan_out → `[BS, N, G, D_col]` where G = B_blocks × C.
Each column group operates on D_col dimensions. After processing, fan_in back to D.

**Shared memory systems** (single instance each, batched across B_blocks):
- **PM** — procedural memory: Hebbian holographic slots (basal ganglia)
- **EM** — episodic memory: key-value episode buffer (hippocampus)

PM/EM state is batched: `[BS, B, r, D]` / `[BS, B, M, D]` (block level in D
space, C columns concatenated). Columns reshape to/from 5D for memory reads.

### How It Works (High Level)

```
For r = 1..R passes:
  1. All G columns process all N tokens in PARALLEL (single forward call):
     a. PCM encode → z, surprise, gain (compare with previous pass's prediction)
     b. PM holographic read (block-level, residual)
     c. EM top-k retrieval (block-level, residual)
     d. Lateral mixer [C,C] — local mixing across columns within each block
     d2. PositionAttention — content-adaptive attention across N_C positions (zero-init residual)
     e. ffn_pre: L FFN layers (gain-modulated on first layer)
     f. Cross-block mixer [B,B] — global mixing across blocks
     g. ffn_post: L FFN layers (no gain)
     h. PCM predict → z_hat (hypothesis for next pass)
     i. W_post → eligibility info + novelty info
  2. PM/EM UPDATE (neuromodulators commit eligibility, write novel episodes)
     - next pass reads the UPDATED PM/EM

After R passes: decode from final representations → loss (FITB or NTP)
Carry PM/EM state to next segment.
```

Step 1 is embarrassingly parallel (G × N independent computations).
Step 2 is the only sequential-across-passes part (R times per segment).

**Processing order rationale**: PCM runs first so surprise/gain are available for
the rest of the pass. Local mixing (lateral) runs before FFN to establish
within-block context. PositionAttention then distributes lateral-mixed context
across all N_C positions (content-adaptive, O(N_C² × D_col) per group — negligible
vs FFN). ffn_pre transforms with gain modulation. Cross-block mixing then shares
information across blocks. ffn_post refines without gain. Total FFN layers per
pass = 2L (where L = `ffn_depth`).

### Forward Pass: R Iterative Passes

```python
def forward_segment(self, input_ids, reset_mask=None, fitb_mask=None):
    """Process N tokens through R refinement passes.

    input_ids:  [BS, N]     — token IDs (may have <FITB> at masked positions)
    reset_mask: [BS] bool   — streams to reset PM/EM (doc boundary)
    fitb_mask:  [BS, N] bool — FITB masked positions (None = NTP path)

    Returns:
        NTP:  (logits [BS, N, vocab], aux_loss scalar)
        FITB: (per_pass_logits list[R × [BS,N,vocab]], aux_loss scalar)
    PM/EM state is mutated in-place.
    """
    if reset_mask is not None:
        self._reset_memory(reset_mask)

    is_fitb = fitb_mask is not None
    sequence = input_ids          # FITB: has <FITB> at masked positions
    original_ids = input_ids

    x = embed(sequence) + pos_embed(positions)       # [BS, N, D]
    x_cols = fan_out(x).view(BS, N, G, D_col)        # G = B*C groups

    z_hat_prev = None
    lam = sigmoid(lambda_logit)

    for r in range(R):
        # FITB: re-embed from updated sequence (r > 0)
        if is_fitb and r > 0:
            x_fresh = embed(sequence) + pos_embed(positions)
            x_cols_fresh = fan_out(x_fresh).view(BS, N, G, D_col)
            x_cols = (1 - lam) * x_out + lam * x_cols_fresh

        # Single batched call — all G groups, all N tokens
        x_out, z, z_hat, surprise, elig_info, nov_info = \
            columns.forward(x_cols, pm, em, z_hat_prev)

        # PCM prediction loss (cross-pass)
        if pcm_enabled and z_hat_prev is not None:
            aux_loss += columns.pcm.prediction_loss(z_hat_prev, z) * pcm_pred_weight

        # PM eligibility routing + commit, EM novelty + write
        _process_and_commit(elig_info, nov_info, x_out)

        if is_fitb:
            # Per-pass logits (fan_in + ln_final + lm_head run R times)
            logits_r = lm_head(ln_final(fan_in(x_out.reshape(BS, N, -1))))
            per_pass_logits.append(logits_r)
            # Re-embed: argmax predictions replace <FITB> for next pass
            if r < R - 1:
                with no_grad():
                    sequence = where(fitb_mask, logits_r.argmax(-1), original_ids)
        else:
            # NTP: damped mixing  x^r = (1-λ)x^{r-1} + λ·column_output^r
            x_cols = x_out if r == 0 else (1 - lam) * x_cols + lam * x_out

        z_hat_prev = z_hat

    if is_fitb:
        return per_pass_logits, aux_loss

    # NTP: single decode from final mixed representation
    logits = lm_head(ln_final(fan_in(x_cols.reshape(BS, N, -1))))
    return logits, aux_loss
```

### Per-Pass Column Forward (The Parallel Core)

In code, this is a single `CorticalColumnGroup.forward` call per R pass.
All column-level ops (PCM, PM/EM reads, mixers, two FFN stacks, fused
post-processing) run in one call. Operates on `[BS, N, G, D_col]` where
G = B*C. PM eligibility routing and EM candidate selection happen in
`model._process_and_commit`.

Conceptually, for one pass:

```python
def column_forward_pass(x_col, pm_state, em_state, z_hat_prev):
    """One R-pass: all G columns process all N tokens. Embarrassingly parallel.

    x_col:      [BS, N, G, D_col]     — column representations
    pm_state:   PM slots              — same for all positions this pass
    em_state:   EM slots              — same for all positions this pass
    z_hat_prev: [BS, N, G, D_pcm]    — PCM predictions from prev pass

    Returns: x_out, z, z_hat, surprise, elig_info, novelty_info
    """
    # 1. PCM: encode, surprise, gain (FIRST — surprise/gain needed by everything else)
    z = PCM.encode(x_col)                                # [BS,N,G,D_pcm]
    if z_hat_prev is not None:
        delta = z - z_hat_prev.detach()
        surprise = norm(delta, dim=-1) / sqrt(D_pcm)     # [BS,N,G]
        gain = 1 + 0.1 * tanh(W_gain(delta))             # [BS,N,G,D_col]
    else:
        surprise = zeros; gain = ones                     # pass 1

    # 2. PM holographic read (block-level in D space, residual)
    q_pm = view_5d(x_col).reshape(BS, N, B, D)           # concat C cols
    y_pm = pm_state.read(q_pm)                            # holographic
    x_col = x_col + reshape_g(y_pm.view(BS, N, B, C, D_col))

    # 3. EM top-k read (block-level in D space, residual)
    q_em = view_5d(x_col).reshape(BS, N, B, D)
    y_em = em_state.read(q_em)                            # top-k retrieval
    x_col = x_col + reshape_g(y_em.view(BS, N, B, C, D_col))

    # 4. Lateral mixer: [C,C] mixing within each block (local context first)
    x_col = reshape_g(lateral(view_5d(x_col)))

    # 4b. Position attention: content-adaptive attention across N_C positions
    #     O(N_C² × D_col) per group — negligible vs FFN. Zero-init → identity at start.
    if pos_attn is not None:
        x_col = pos_attn(x_col)            # [BS, N_C, G, D_col]

    # 5. ffn_pre: L residual FFN layers with gain on first layer
    x_col = ffn_pre(x_col, gain=gain)

    # 6. Cross-block mixer: [B,B] mixing across blocks (global after local FFN)
    x_col = reshape_g(cross_block(view_5d(x_col)))

    # 7. ffn_post: L residual FFN layers (no gain)
    x_out = ffn_post(x_col)

    # 8. PCM hypothesis (predict next pass's encoding)
    z_hat = PCM.predict(z)                                # [BS,N,G,D_pcm]

    # 9. Fused post-processing (W_post_fused: D_col → 4*D_col + 1)
    proj = W_post_fused(x_out)                            # [BS,N,G,4*D_col+1]
    k_pre, v_post, k_cand_raw, v_cand, nov_raw = split(proj, [D_col]*4 + [1])
    k_cand = normalize(k_pre)                             # [BS,N,B,C,D_col]
    gate = (surprise / scale).clamp(0, 1)                 # [BS,N,B,C]
    q_nov = normalize(k_cand_raw)
    w_nov = sigmoid(nov_raw)

    elig_info = (k_cand, v_post, gate)
    novelty_info = (q_nov, v_cand, w_nov, surprise)

    return x_out, z, z_hat, surprise, elig_info, novelty_info
```

**Then in `model._process_and_commit`** (once per R pass):

```python
# PM eligibility routing: concat columns → block level, route to slots
route_w = softmax(k_norm @ pm_K.T / tau_route)           # [BS,N,B,r]
gated = gate_block * route_w                              # surprise-gated
elig_K = bmm(gated, k_block)                              # [BS,B,r,D]
elig_V = bmm(gated, v_block)

# EM candidate selection: top-C_cand by novelty across N positions
novelty = w_nov * surprise + (1-w_nov) * (1 - max_sim)
em_cands = topk(novelty, C_cand)

# Neuromodulated PM commit + EM write
pm.commit(elig_K, elig_V, g, slot_logits, tau)
em.write(cand_K, cand_V, cand_scores, g_em, tau_em, decay, ww)
```

**CRITICAL**: No cross-token operations inside column forward. Each position
is fully independent. This is what makes G × N parallelism possible. The
aggregation in `_process_and_commit` sums across positions but doesn't affect
the per-token output.

**Memory reads ARE attention** — but over memory slots, not over other tokens.

**Ordering rationale**: PCM runs first so surprise/gain are available for all
subsequent steps. Lateral mixing establishes within-block context. PositionAttention
then distributes this local context across all N_C positions with content-adaptive
attention. FFN pre-processes with gain. Cross-block mixing shares information
globally after local processing. FFN post-processes without gain. Two FFN stacks
(ffn_pre + ffn_post) with L layers each give 2L total FFN layers per pass.

### PM Holographic Read (Why "Holographic")

Standard memory: query → retrieve stored vector. Output is independent of input.

PM holographic read: the input **flows through** stored patterns. Output depends
on both input AND stored pattern multiplicatively:

```
q = concat_cols(x_col)                           # [D] — C columns concatenated to block level
scores = normalize(q) @ pm_K.T                   # [r_slots]
y_pm = sum_i(pm_a_i * scores_i * q * pm_V_i)    # [D]
y_pm_cols = split_to_cols(y_pm)                  # back to [C, D_col]
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
q = concat_cols(x_col)                           # [D] — C columns concatenated to block level
scores = q @ em_K.T                              # [M_slots]
top_k_mask = scores >= topk(scores, k).min       # gather-free top-k
attn = softmax(masked(scores, top_k_mask))       # [M_slots]
y_em = attn @ em_V                               # [D]
y_em_cols = split_to_cols(y_em)                  # back to [C, D_col]
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

**PositionAttention**: In addition to memory-mediated context, PositionAttention
provides direct content-adaptive attention across N_C positions within each pass.
Each of G groups independently attends over N_C positions with O(N_C² × D_col)
compute — negligible vs FFN. This gives tokens direct access to each other's
representations within a pass, complementing the PM/EM bottleneck path.

Within a single pass, all positions still see the same PM/EM state. PositionAttention
adds direct cross-position mixing that PM/EM alone cannot provide.

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
| Column FFN (ffn_pre + ffn_post) | Same pass's contribution to loss | Direct through decode |
| PCM encode/hypothesis | L_pred (auxiliary) + downstream surprise | Same segment |
| PM read (holographic) | Same pass's loss through PM modulation | Direct |
| EM read (top-k) | Same pass's loss through EM retrieval | Direct |
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

### 1. Within-Segment: Bidirectional (No Causal Mask)

Within a single pass, there is no causal ordering — all N positions are equivalent
(modulo positional encoding). All tokens within pass r see the same PM/EM state,
and PM/EM updates between passes let every token's information flow to every other
token through the memory bottleneck.

This is **explicitly bidirectional by design**, which is why we use FITB (masked
token prediction) rather than next-token prediction as the training objective. See
the Training Objective section.

**Across segments**: PM/EM state at segment k reflects only tokens from
segments 0..k-1 and the current segment's completed passes — cross-segment
context is strictly causal.

**Across passes**: PM/EM at pass r reflects all N tokens from passes 0..r-1.
All tokens within pass r see the same PM/EM_r.

### 2. Train/Inference Equivalence

**Training (FITB)**: mask ~30% of N tokens, process all N through R passes, decode
at each pass.
**Training (NTP, backward compat)**: process N tokens through R passes, decode once.
**Inference (autoregressive)**: process one token at a time for R passes, decode.
**Inference (non-autoregressive)**: generate N tokens at once via FITB refinement.

Equivalent because no cross-token operations within a pass. At inference, PM/EM
still update after each pass (single token's eligibility/novelty may or may not
trigger commits — the neuromodulator decides).

Inference memory: PM + EM + per-column PCM z_hat + eligibility accumulators.
**Constant**, does not grow with sequence length.

### Non-Autoregressive Segment Generation

`generate_segments()` exploits the architecture's bidirectional nature for fast inference:

1. **Phase 1 (Prompt)**: Process prompt through NTP segments to warm PM/EM
2. **Phase 2 (Generate)**: Generate N-token chunks, all positions start as `<FITB>`,
   R passes refine, sample from last pass's logits

Throughput: N/R tokens per forward pass vs 1 for autoregressive `generate()`.
For N=512, R=4: **128x** throughput improvement per forward pass.

```
Autoregressive:  [tok] → R passes → 1 token    → O(R×N) per token
Non-autoregressive: [<FITB>×N] → R passes → N tokens → O(R) per N tokens
```

Quality tradeoff: non-autoregressive generation lacks left-to-right causal
ordering within a segment. Quality depends on how well PM/EM carry context
from prompt segments and how effectively R-pass refinement resolves ambiguity.

### 3. What R=1 Is

R=1: single pass. No cross-pass surprise (no previous prediction). No PM/EM
update within the segment (only carries forward to next segment). This is a
**feedforward model with memory reads** — similar to retrieval-augmented generation.

R=2+: cross-pass surprise kicks in. PM/EM update between passes. Tokens gain
indirect access to each other through memory. "Depth" emerges.

## Practical Design Decisions

### Column Dimensions

```
# Tier A example (~96M params)
D = 2048          # internal model width
D_embed = 384     # embedding / LM head width (decoupled from D)
B_blocks = 6      # memory blocks (PM/EM state batched as BS*B)
C = 4             # columns per block → G = B*C = 24 groups total
D_col = 512       # = D // C. Column width (also PM/EM dimension — unified)
D_pcm = 64        # PCM encoding dimension
ffn_depth = 1     # L = 1 → 2 total FFN layers per pass (1 pre + 1 post)
R = 2             # refinement passes

Fan-out:  D → G × D_col = 2048 → 12288  (6x)
Fan-in:   G × D_col → D = 12288 → 2048

# Tier presets (see config.py):
# Tier A (~96M):   D=2048 D_embed=384 B=6  C=4  G=24  Dcol=512  R=2  L=1
# Tier B (~390M):  D=3072 D_embed=512 B=12 C=4  G=48  Dcol=768  R=4
# Tier C (~894M):  D=4096 D_embed=768 B=16 C=4  G=64  Dcol=1024 R=6
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
Block b: C columns share PM_b (r_slots × D_col) + EM_b (M_slots × D_col)
```

Total columns = B_blocks × C. Each block's PM/EM can specialize. Analogous
to different brain regions maintaining different types of memories.

## Complexity

### Serial Depth

Per pass: O(2L) FFN layers — all G × N computations are parallel within each layer.
PM/EM commit between passes: O(1) (small, not N-dependent).
Total: O(R × 2L).

Compare:
- Current (v1):       O(K×P) = 8×32 = **256 steps**
- Transformer (L=6):  O(L) = **6 steps** (but O(N²) compute per step)
- Mamba (L=6):        O(L×N) (sequential scan per layer)
- **v4 (R=2, L=1)**:  O(R×2L) = **4 steps** (and O(N×D²) compute per step)

### Compute

Per pass: O(N × G × 2L × D_col²) for column FFNs (2 stacks × L layers)
        + O(N × B_blocks × r × D) for PM reads
        + O(N × B_blocks × k × D) for EM reads
Total: R × above

### Memory

Training: O(R × N × D) for intermediate representations.
With gradient checkpointing: O(N × D) + recompute per pass.
Inference: O(PM + EM + B × D_pcm) — **constant**.

## Training

### Training Objective: FITB (Fill-In-The-Blank)

The architecture's R-pass iterative refinement is fundamentally bidirectional: all
N tokens share PM/EM state that gets updated between passes, so every token can
see information from every other token through memory after pass r=0. Rather than
forcing causal masking on this naturally bidirectional architecture, we use FITB
(masked token prediction) as the training objective.

**FITB R-pass loop:**
1. Mask ~30% of tokens with `<FITB>` token (random or span masking)
2. For each pass r=0..R-1:
   a. Re-embed from updated sequence (r > 0 only): damped mix of previous
      pass's column output with freshly re-embedded predictions
   b. Column processing (PM read, EM read, FFN, PCM — unchanged)
   c. fan_in + ln_final + lm_head → logits_r for this pass
   d. PM/EM update (including FITB positions — model should know what it predicted)
   e. Re-embed: argmax(logits_r) replaces `<FITB>` positions for next pass (no grad)

**Damped mixing (FITB vs NTP):**
- NTP:  `x_cols = (1-λ)*x_cols_prev + λ*x_out` — old input blended with new column output
- FITB: `x_cols = (1-λ)*x_out_prev + λ*x_cols_fresh` — previous column output blended
  with fresh re-embedding. In both cases, λ weights the "new information."

**Per-pass loss** (each pass independently produces logits and CE loss):

    L = (1/(R*M)) Σ_r Σ_{m∈masked} CE(logits_r_m, original_token_m) + α * L_pred

where M = number of masked positions, targets are original tokens at masked
positions (NOT shifted-by-1), and L_pred = Σ_r MSE(z_hat^r, z^{r+1}.detach()).

**Gradient flow:** Cross-pass gradients flow through `x_out` in the damped mixing
formula (pass r's logits depend on pass r-1's column output). The only blocked
gradient path is the argmax re-embedding (non-differentiable, wrapped in no_grad).
This means later-pass losses can influence earlier-pass parameters.

Uniform loss weighting across passes (later-pass upweighting is a future option).

**Config fields:** `mask_rate` (0.3), `span_mask_prob` (0.5), `span_mask_mean_len` (3),
`fitb_id` and `null_id` (set from tokenizer at runtime).

**Backward compatibility:** `fitb_mask=None` in `forward_segment()` triggers the
original NTP path. `mask_rate=0.0` disables FITB in the trainer.

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
# Fused post-processing produces candidates in D_col space directly
k_cand = normalize(k_pre)                  # [D_col] — from W_post_fused
v_cand = v_post                            # [D_col]
gate = (surprise / 5.0).clamp(0, 1)        # third factor
route_w = softmax(pm_K @ k_cand / tau)     # [r_slots] — which slot?

# Accumulate across all N tokens in this pass
elig_K += gate * route_w ⊗ k_cand          # [r_slots, D_col]
elig_V += gate * route_w ⊗ v_cand
```

At the end of each pass, the neuromodulator decides how much to commit:

```
(g, slot_logits, tau, ww) = PM_neuromodulator(elig_summary)
g *= elig_mag / max(elig_mag)              # scale by eligibility magnitude (pass-0 → g≈0)
slot_logits -= ww * pm_a                   # learned weakness bias (prefer weaker slots)
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
(g_em, tau, decay, ww) = EM_neuromodulator(novelty_mean, em_usage, content)
slot_scores = candidates @ em_K.T + ww * weakness  # learned weakness bias
slot_weights = softmax(slot_scores / tau)
em_K[slot] = (1 - g_em * slot_weights) * em_K[slot] + g_em * slot_weights * candidate_K
em_V[slot] = (1 - g_em * slot_weights) * em_V[slot] + g_em * slot_weights * candidate_V
em_S *= decay                             # strength decay before write
```

**Note on decay/aging timing:** `pm.base_decay()` and `em.age_tick(N)` run
**once per segment** (after all R passes), not per pass. Passes are refinement
steps at the same time index — decay/aging should reflect elapsed segments,
not refinement depth. PM/EM *commits and writes* still happen between passes.

### Summary

| Component | Where | When | Parallel? |
|-----------|-------|------|-----------|
| PCM encode + surprise | Column | Every token, every pass | Yes (G×N) |
| PM holographic read | Column | Every token, every pass | Yes (G×N) |
| EM top-k read | Column | Every token, every pass | Yes (G×N) |
| Lateral mixer [C,C] | Column | Every token, every pass | Yes (G×N) |
| PositionAttention [N_C,N_C] | Column | Every token, every pass | Yes (G groups) |
| ffn_pre (L layers, gain) | Column | Every token, every pass | Yes (G×N) |
| Cross-block mixer [B,B] | Column | Every token, every pass | Yes (G×N) |
| ffn_post (L layers) | Column | Every token, every pass | Yes (G×N) |
| PM eligibility accumulate | Column | Every token, every pass | Yes (G×N) |
| EM novelty score | Column | Every token, every pass | Yes (G×N) |
| PM neuromod commit | Between passes | Once per pass | Small |
| EM neuromod write | Between passes | Once per pass | Small |
| PM base decay | After R loop | Once per segment | Tiny |
| EM age tick | After R loop | Once per segment | Tiny |

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

4. **Within-segment context**: ~~Is memory-mediated context (R hops through PM/EM)
   sufficient?~~ **Addressed**: PositionAttention adds direct content-adaptive
   cross-position mixing (O(N_C² × D_col) per group). Zero-init preserves
   existing training dynamics; empirical evaluation will show the benefit.

5. **Block/column scaling**: How many blocks × columns per block?

6. **PCM prediction target**: Predict next pass's encoding, final pass's encoding,
   or contrastive loss?

7. **Pass 1 (no surprise)**: What should gate eligibility/novelty in pass 1? Options:
   a) No gating (accumulate everything), b) Use a fixed prior, c) Skip accumulation.
