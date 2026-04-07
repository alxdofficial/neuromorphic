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
                 ┌─── per-token loop (t = 0..T-1) ───┐
                 │                                     │
                 │  memory_step(H_mid[t])              │
                 │     → W @ msg (dense bmm)           │
                 │     → inject + border exchange      │
                 │     → state MLP + decay blend       │
                 │     → message MLP + identity        │
                 │     → readout[t]                    │
                 │                                     │
                 │  (every 4 steps: modulate)          │
                 │                                     │
                 └─────────────────────────────────────┘
                           │
                           ▼
         H_enriched = H_mid + mem_scale * readouts
                           │
                           ├──→ memory head = lm_head(readouts)
                           │      → mem_pred_loss (segment-level CE)
                           │      → live s_mem (per-token, updates EMAs)
                           │
                           ▼
                 upper scan → H_upper[0..T-1]
                           │
                           ▼
                        LM head → logits
```

Note: H_enriched uses raw H_mid (no surprise augmentation MLP). Memory
predicts tokens directly via a weight-tied head. The memory head's CE is
both the auxiliary training loss *and* the source of the surprise signal
that drives the modulator.

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

All signals are scalars per batch sample (or per cell for `readout_drift`).

```
                          ┌─ mem head = lm_head(readout[t-1]) ─┐
memory readout ───────────┤                                     ├─→ logit[x_t]
                          └─ (weight-tied projection)           ┘
                                                                    │
                                       s_mem_live(t) = -logit[x_t] ─┘
                                                           │
                                         ┌──── EMA ────────┴────────┐
                                   s_mem_ema_fast         s_mem_ema_slow
                                         │                          │
                                         └──── slow - fast → s_progress
                                                                    │
readout reshape-to-cell → ||readout_cell[t] - prev_readout_cell|| → readout_drift
                                                                    │
                                                         ┌──────────┘
                                                         ▼
                                                   per-cell modulator
                                                   (broadcast 3 global
                                                    scalars to each cell)
```

- **`s_mem_live`**: instant per-token memory-head surprise (negative
  unnormalized log-prob of the observed token under the memory head).
  Lower = memory is confident. Higher = memory was surprised.
- **`s_mem_ema_fast / s_mem_ema_slow`**: EMAs of `s_mem_live`, with decays
  `gain_ema_fast=0.3` / `gain_ema_slow=0.05` (effective horizons ≈ 3 and 20
  tokens).
- **`s_progress = s_mem_ema_slow - s_mem_ema_fast`**: positive when fast is
  lower than slow, i.e., when memory's short-term prediction error is
  *below* its long-term average — learning progress. Literally the
  Schmidhuber-style curiosity signal, constructed from the memory-head
  loss instead of a separate world model.
- **`readout_drift`**: per-cell local state churn —
  `mean(|readout_cell[t] - readout_cell[t-1]|)` per cell. Captures how
  much each cell's output is moving. Updated after every memory step.

At inference (no backprop), the modulator continues to take writes driven
by these signals. No external reward, no RL, no labels needed.

## Memory-prediction Head Loss

At segment end, the memory head's full CE is computed over all T tokens:

```python
shifted = concat([prev_readout_at_segment_start, readouts[:, :-1]])  # [BS, T, D]
mem_logits = lm.mem_head_logits(shifted)                              # [BS, T, V]
mem_pred_loss = F.cross_entropy(mem_logits.reshape(-1, V),
                                 input_ids.reshape(-1))
```

The memory head uses `readout[t-1]` to predict the token at position `t`.
`prev_readout_at_segment_start` carries over from the previous segment,
so the first token of each segment has a meaningful prediction (no "free"
loss).

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
| Grid | H × W | 2 × 4 | config.grid_h × grid_w |
| Neurons per cell | Cn | 32 | config.neurons_per_cell |
| Total neurons | N | 256 | N_cells × Cn |
| Ports per cell | alpha | 4 | config.alpha |
| Border neurons per cell | B | 4 | config.border_per_cell |
| MLP groups (inject only) | G | 8 | config.mlp_groups |
| State MLP hidden | Hs | 256 | config.state_mlp_hidden |
| Msg MLP hidden | Hm | 256 | config.msg_mlp_hidden |
| Modulator hidden | Hmod | 2048 | config.cell_mod_hidden |
| Modulation interval | M | 4 | config.modulation_interval |
| TBPTT block | — | 8 | config.tbptt_block |
| Checkpoint every | — | 8 | config.checkpoint_every |
| W decay rate | — | 1e-3 | config.w_decay_rate |
| mem_pred_weight | — | 0.1 | config.mem_pred_weight |
| gain_ema_fast | — | 0.3 | config.gain_ema_fast |
| gain_ema_slow | — | 0.05 | config.gain_ema_slow |

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

**Input** `[BS, NC, mod_in = 3*D_n + 6 = 774]`:
```python
cat([h_mean, msg_mean, cell_context,       # 3*D_n — per-cell state
     W_stats, decay_mean,                   # 2     — per-cell stats
     readout_drift,                          # 1     — per-cell local surprise
     s_mem_live, s_mem_ema_fast, s_progress # 3     — global surprise, broadcast
])
```

**Hidden**: `[BS, NC, Hmod=2048]` (tanh activation, per-cell einsum)

**Output** `[BS, NC, mod_out=1316]`:
```python
delta_W:      [BS, NC, N*N=1024]  → reshape to [BS, NC, 32, 32]  (direct)
delta_decay:  [BS, NC, N=32]
delta_ctx:    [BS, NC, D_n=256]
delta_border: [BS, NC, B=4]
```

### Cell Layout

```
[0 : 4)      — input port neurons
[4 : 8)      — output port neurons
[8 : 12)     — border neurons
[12 : 32)    — internal neurons
```

### Inject

H_mid_t `[BS, D=2048]` reshaped to `[BS, NC=8, D_n=256]`, projected
through per-group `inject_w: [G=8, alpha*D_n, D_n]` with orthogonal
init, added to input port neuron positions. No surprise augmentation —
H_mid flows in directly.

### Border Exchange

Grid-reshape + slice shifts (N/S/E/W). Vectorized.

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
cell_context       : [BS, NC, D_n]         — per-cell context
border_gate_logit  : [BS, NC, B]           — cross-cell gates
s_mem_live         : [BS]                   — current memory-head surprise
s_mem_ema_fast     : [BS]                   — fast EMA
s_mem_ema_slow     : [BS]                   — slow EMA
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
| inject_w | [G=8, alpha*D_n, D_n] | 2.1M | Per-group orthogonal init |
| inject_b | [G=8, alpha*D_n] | 8K | |
| mod_w1 | [NC=8, 774, 2048] | 12.7M | Per-cell modulator layer 1 |
| mod_b1 | [NC=8, 2048] | 16K | |
| mod_w2 | [NC=8, 2048, 1316] | 21.6M | Per-cell modulator layer 2 |
| mod_b2 | [NC=8, 1316] | 11K | |
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

- `H_mid.detach()` before memory: LM lower scan is isolated from the
  memory path. This means `mem_pred_loss` cannot leak into the LM scan
  layers — the LM is trained purely by `ce_loss`. Verified: after
  backward on `aux_loss` alone, `lm.proj_in.weight.grad == 0`.
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

- TBPTT: detach all state every `tbptt_block=8` tokens
- Checkpointing: checkpoint blocks of `checkpoint_every=8` tokens
- Modulation: every `modulation_interval=4` tokens

## Initialization

- **W**: sparse random (K=8 nonzeros per row, value 0.1, no self-connections)
- **h**: small random (N(0, 0.01))
- **msg, decay_logit, cell_context, border_gate_logit**: zeros
- **s_mem_live, s_mem_ema_fast, s_mem_ema_slow**: zeros
- **prev_readout, prev_readout_cell**: zeros
- **mem_scale**: sqrt(alpha) = 2.0
- **mod_w2**: small init (× 0.01)
- **inject_w**: per-port orthogonal blocks

## Performance

BS=96, tier_a, RTX 4090:
- 235 ms/step, **52.1K tok/s**, 22.2 GB
- Training time estimates: 0.75B tokens ≈ 4h, 1.5B tokens ≈ 8h

The cost relative to the PCM-based predecessor (57K tok/s) comes from the
segment-level full-vocab CE on `[BS, T, V]` readout logits. The simplification
is worth it: no PCM, no split_mlp, no surprise_proj, no detach gymnastics,
and the surprise signal is principled and interpretable.
