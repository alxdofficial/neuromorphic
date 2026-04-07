# Neuromorphic Memory Graph — Dense-W, Memory-Head Design

## Goals

1. Lifelong personal-assistant memory: a language model that gets better at
   remembering user-specific facts and preferences through use, with no RAG,
   no memory documents, no prompt engineering.
2. Split-scan LM with interleaved memory graph and weight-tied memory-prediction head.
3. One memory interaction per token.
4. GPU-friendly: all hot-path operations are batched matmuls (bmm) or F.linear.
5. Intrinsic surprise signal computable at inference with no external labels.
6. Minimizing surprise ↔ better next-token prediction ↔ useful memory writes.
7. Surprise-driven neuromodulator continuously adjusts connectivity.
8. No document-boundary resets in the memory graph (memory is lifelong).

## Architecture Overview

```
tokens → embedding → lower scan → H_mid[0..T-1]  (parallel, fused_scan)
                                       │
                           ┌───────────┘
                           ▼
                 ┌─── per-token loop (t = 0..T-1) ───────────────┐
                 │                                                 │
                 │  (1) live surprise (no_grad)                   │
                 │        logit = mem_head(prev_readout)[x_t]     │
                 │        s_mem_live = -logit                     │
                 │        update s_mem_ema_fast/slow              │
                 │                                                 │
                 │  (2) if t % 4 == 0: modulate cells             │
                 │        sees fresh s_mem_live + EMAs +          │
                 │              readout_drift + cell state        │
                 │        updates W, decay, context                │
                 │                                                 │
                 │  (3) memory_step(H_mid[t])                     │
                 │        W @ msg (dense bmm)                     │
                 │        + inject                                │
                 │        + state MLP + decay blend               │
                 │        + message MLP + identity                │
                 │        → readout[t]                            │
                 │                                                 │
                 │  (4) update readout_drift and prev_readout     │
                 │                                                 │
                 │  (5) W *= (1 - w_decay_rate)                   │
                 │                                                 │
                 └─────────────────────────────────────────────────┘
                           │
                           ▼
         H_enriched = H_mid + mem_scale * readouts
                           │
                           ▼
                 upper scan → H_upper[0..T-1]
                           │
                           ▼
                        LM head → logits

At segment end:
   mem_pred_loss = chunked CE(mem_head(shifted_readouts), input_ids)
                   — trains memory to carry prediction-useful info
                   — on the autograd graph, backprops through readouts
```

Notes:
- H_enriched uses raw H_mid (no surprise augmentation MLP).
- The in-loop s_mem computation is `torch.no_grad()` — it's a detached
  signal fed to the modulator. The training gradient comes from the
  segment-level mem_pred_loss, which flows through the readouts into
  memory (modulator, state/msg MLPs, inject) and through the tied
  lm_head weight.

## Why the memory-prediction head

The job of "surprise" in this architecture is to gate *when to write to W*.
That question has a clean answer for a personal-assistant use case:

> Write when the memory's own next-token prediction from its readout is
> bad. Relax writes when it's already good.

The memory-prediction head makes that signal measurable. Its architecture:
- Takes **only** `memory_readout[t]` as input — nothing from H_mid, H_upper,
  or the LM scan. This forces the memory to carry prediction-useful
  information all by itself.
- Weight-tied to the main `lm_head` (`lm_head.weight = embedding.weight`),
  through the shared `proj_down → ln_final → lm_head` path. Zero new
  parameters, and the memory's readout is forced into the same semantic
  space as the rest of the LM.
- Trained by self-supervised CE against the actual next token. No external
  labels, no counterfactual forward passes.

This follows the precedent set by Lu et al. (2022) and Yogatama et al.
(2021): the modulator does not need a hand-crafted intrinsic reward —
backprop through the memory head's CE loss teaches it what to remember.

## Surprise Pipeline

All four surprise channels are computed inside the compiled `_run_block`,
per token, under `torch.no_grad()`. The modulator at the next modulation
step reads them fresh from within the same compiled graph — there is no
post-block Python update loop.

```
prev_readout[t-1] ──┐
                    ├──→ mem_head (weight-tied) ──→ logit[x_t]
input_ids[t] ───────┘                                   │
                                                         ▼
                                       s_mem_live(t) = -logit[x_t]
                                                         │
                                  ┌─── EMA (in-block) ───┘
                            s_mem_ema_fast
                                  │
readout[t] ─→ reshape cell ─→ |cell[t] - prev_cell| → readout_drift
                                  │                            │
                                  └────────────┬───────────────┘
                                               ▼
                                        per-cell modulator
                                    (global scalars broadcast,
                                     per-cell readout_drift local)
```

- **`s_mem_live`**: instant per-token memory-head surprise (negative
  unnormalized log-prob of the observed token). Computed inside the
  per-token loop from `prev_readout` (readout at `t-1`) and `input_ids[t]`.
  Lower = memory is confident, higher = memory was surprised.
- **`s_mem_ema_fast`**: short-horizon EMA of `s_mem_live`, updated
  per-token in the compiled block with `gain_ema_fast=0.3` (≈ 3-token
  effective horizon). Gives the modulator a smoothed reference next to
  the spiky live signal.
- **`readout_drift`**: per-cell local state churn —
  `mean(|readout_cell[t] - readout_cell[t-1]|)` per cell. Updated after
  every memory step. This is the only per-cell surprise channel; the
  two `s_mem_*` scalars are broadcast to every cell.

The in-loop computation is `no_grad`, so these signals are detached
inputs to the modulator. The training gradient for memory comes from
`mem_pred_loss` at segment end (see below), not from the in-loop path.

At inference (no backprop), the modulator continues to take writes driven
by these signals. No external reward, no RL, no labels needed.

## Memory-prediction Head Loss

At segment end, the memory head's full CE is computed over all T tokens,
**chunked along the time axis** to keep peak VRAM bounded (the full
`[BS, T, V]` logits tensor is never materialized at once):

```python
# Shift readouts: memory head uses readout[t-1] to predict input_ids[t].
shifted = concat([prev_readout_at_segment_start, readouts[:, :-1]])  # [BS, T, D]

loss_sum = 0
for s in range(0, T, block_size):                # block_size = tbptt_block = 8
    sub_logits = lm.mem_head_logits(shifted[:, s:s+block_size])  # [BS, 8, V]
    loss_sum += F.cross_entropy(
        sub_logits.reshape(-1, V), input_ids[:, s:s+block_size].reshape(-1),
        reduction="sum")
mem_pred_loss = loss_sum / (BS * T)
```

`prev_readout_at_segment_start` is the previous segment's final readout,
so the first token of each segment has a meaningful prediction — no
"free" loss on the first position.

`mem_pred_loss` is added to the total loss with weight `mem_pred_weight=0.1`:

```
total_loss = ce_loss + mem_pred_weight * mem_pred_loss
```

Gradient flow:
- `ce_loss` → LM head, upper scan, mem_scale, memory readout, memory params, modulator.
- `mem_pred_loss` → lm_head (tied weight), proj_down, ln_final, memory readout,
  memory params, modulator. Does **not** reach the LM scan layers because
  `H_mid.detach()` isolates the LM lower scan from the memory path.

## Default Dimensions

| Parameter | Symbol | Default | Source |
|-----------|--------|---------|--------|
| LM hidden dim | D | 2048 | config.D |
| Embedding dim | D_embed | 768 | config.D_embed |
| Scan layers | L_total | 4 | config.L_total |
| Scan split point | split_at | 2 | config.scan_split_at |
| Scan recurrence dim | d_inner | 1200 | config.d_inner |
| Neuron hidden dim | D_n | 256 | config.D_n |
| Cells | N_cells | 8 | D // D_n |
| Neurons per cell | Cn | 32 | config.neurons_per_cell |
| Total neurons | N | 256 | N_cells × Cn |
| Ports per cell | alpha | 4 | config.alpha |
| State MLP hidden | Hs | 256 | config.state_mlp_hidden |
| Msg MLP hidden | Hm | 256 | config.msg_mlp_hidden |
| Modulator hidden | Hmod | 2048 | config.cell_mod_hidden |
| Modulation interval | M | 4 | config.modulation_interval |
| TBPTT block | — | 8 | config.tbptt_block |
| W decay rate | — | 1e-3 | config.w_decay_rate |
| mem_pred_weight | — | 0.1 | config.mem_pred_weight |
| gain_ema_fast | — | 0.3 | config.gain_ema_fast |

## Memory Graph

### Connectivity: Dense W Matrix

Each cell has `W: [BS, NC, Cn, Cn] = [BS, 8, 32, 32]`. Runtime state
(not a learned parameter). Initialized sparse (K=8 nonzeros per row),
evolves via the neuromodulator. Message passing is a single bmm:

```python
received = torch.matmul(W, msg)    # [BS, NC, Cn, D_n]
```

Soft sparsity: `W *= (1 - w_decay_rate)` each step. This is *on-graph*
(no `torch.no_grad()`), so later-token loss can train the modulator
output that produced W via the persistence path ("write now, help later").

### Neuromodulator

Per-cell MLP, runs every `modulation_interval=4` steps.

**Input** `[BS, NC, mod_in = 2*D_n + 5 = 517]`:
```python
cat([h_mean, msg_mean,                     # 2*D_n — per-cell state
     W_stats, decay_mean,                   # 2     — per-cell stats
     readout_drift,                          # 1     — per-cell local surprise
     s_mem_live, s_mem_ema_fast,             # 2     — global surprise, broadcast
])
```

**Hidden**: `[BS, NC, Hmod=2048]` (tanh activation, per-cell einsum)

**Output** `[BS, NC, mod_out=1056]`:
```python
delta_W:      [BS, NC, N*N=1024]  → reshape to [BS, NC, 32, 32]  (direct)
delta_decay:  [BS, NC, N=32]
```

### Cell Layout

```
[0 : 4)      — input port neurons
[4 : 8)      — output port neurons
[8 : 32)    — internal neurons
```

### Inject

H_mid_t `[BS, D=2048]` reshaped to `[BS, NC=8, D_n=256]`, projected
through per-cell `inject_w: [N_cells=8, alpha*D_n, D_n]`, added to
input port neuron positions. No surprise augmentation — H_mid flows
in directly.

### State Update

Split first-layer optimization: no `cat([received, h])` tensor allocation.

```python
hidden = tanh(F.linear(received, w1_recv) + F.linear(h, w1_h, b1))
candidate = tanh(F.linear(hidden, w2, b2))
h_new = decay * h + (1 - decay) * candidate
```

Shared weights across all neurons.

### Emit Message

```python
msg_new = tanh(MLP(h)) + identity
```
Where `identity = neuron_id` (learned, static per neuron).

### Readout

Output port messages (neurons 4-7) summed and scaled by `1/sqrt(alpha)`,
reshaped to `[BS, D]`.

## Runtime State

```
h                  : [BS, NC, Cn, D_n]     — neuron hidden states
msg                : [BS, NC, Cn, D_n]     — neuron messages
W                  : [BS, NC, Cn, Cn]      — connectivity matrix
decay_logit        : [BS, NC, Cn]          — per-neuron temporal decay
s_mem_live         : [BS]                   — current memory-head surprise
s_mem_ema_fast     : [BS]                   — fast EMA
prev_readout       : [BS, D]                — last readout (for memory head)
prev_readout_cell  : [BS, NC, D_n]          — last per-cell readout (for drift)
```

All of these persist across segments and chunks — memory is lifelong.
Detached at TBPTT boundaries.

## Learned Parameters

| Component | Shape | Params | Notes |
|-----------|-------|--------|-------|
| neuron_id | [NC=8, Cn=32, D_n=256] | 65K | Per-neuron identity |
| state_w1/b1/w2/b2 | — | 198K | Shared state MLP |
| msg_w1/b1/w2/b2 | — | 131K | Shared message MLP |
| inject_w | [NC=8, alpha*D_n, D_n] | 2.1M | Per-cell, kaiming init |
| inject_b | [NC=8, alpha*D_n] | 8K | |
| mod_w1 | [NC=8, 517, 2048] | 8.5M | Per-cell modulator layer 1 |
| mod_b1 | [NC=8, 2048] | 16K | |
| mod_w2 | [NC=8, 2048, 1056] | 17.3M | Per-cell modulator layer 2 |
| mod_b2 | [NC=8, 1056] | 8K | |
| **Memory total** | | **~36.8M** | |
| **LM total** | | **~67.3M** | |
| **Grand total** | | **~104.1M** | |

Memory-prediction head adds **zero** parameters (tied to lm_head via
the embedding).

## Training

### Losses

```
total_loss = ce_loss + mem_pred_weight * mem_pred_loss
```

- **ce_loss**: standard next-token CE on LM output. Trains LM (scan
  layers, embedding, head), mem_scale, and the memory graph
  (state/msg MLPs, modulator, inject) via the readout path.
- **mem_pred_loss**: memory head's self-supervised CE against next
  tokens. Trains the memory graph (forces readout to carry predictive
  info) and refines lm_head/proj_down/ln_final (shared with main head).

### Gradient Boundaries

- `H_mid.detach()` before memory: the **LM scan layers** are isolated
  from the memory path. `mem_pred_loss` cannot reach scan-layer weights
  — verified: after backward on `aux_loss` alone, scan layer grads are
  zero. Note that `lm_head` is weight-tied to `embedding` and shared
  with the memory head, so `aux_loss` *does* reach `embedding.weight`
  and `proj_down`/`ln_final`. This is intended: memory and LM predict
  into the same vocabulary projection and co-adapt that shared space.
- `prev_readout_cell`, `s_mem_*` are detached (no_grad EMAs).
- TBPTT detach every 8 tokens.

### EOT Handling

- **LM scan**: resets carry at positions following EOT tokens. The reset
  is built from both in-chunk EOT positions *and* the previous chunk's
  last token (`prev_token` passed through from the dataloader), so
  document boundaries that fall between chunks are handled correctly.
- **Memory graph**: does NOT reset at EOT. Memory is lifelong — it
  persists across document boundaries. This is deliberate: the memory
  accumulates knowledge over the entire training sequence and continues
  to do so at inference.
- **CE loss**: tokens where `input_ids == eot_id` are excluded.

### Block structure

- TBPTT: detach all state and unroll the loop every `tbptt_block=8` tokens
- Optional activation checkpointing on each block (`checkpoint_memory=False`)
- Modulation: every `modulation_interval=4` tokens

## Initialization

- **W**: sparse random (K=8 nonzeros per row, value 0.1, no self-connections)
- **h**: small random (N(0, 0.01))
- **msg, decay_logit**: zeros
- **s_mem_live, s_mem_ema_fast**: zeros
- **prev_readout, prev_readout_cell**: zeros
- **mem_scale**: sqrt(alpha) = 2.0
- **mod_w2**: small init (× 0.01)
- **inject_w**: kaiming uniform

## Performance

BS=96, tier_a, RTX 4090:
- 235 ms/step, **52.3K tok/s**, 22.3 GB
- Training time estimates: 0.75B tokens ≈ 4h, 1.5B tokens ≈ 8h

The cost relative to the PCM-based predecessor (57K tok/s) comes from the
segment-level memory-head CE on `[BS, T, V]` readout logits — chunked
over time to keep peak VRAM bounded but still the dominant new cost. The
architectural simplification is worth it: no PCM, no split_mlp, no
surprise_proj, no detach gymnastics, and the surprise signal is
principled, interpretable, and genuinely live in the per-token loop.
