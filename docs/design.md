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
                 │        logits = mem_head(prev_readout)         │
                 │        s_mem_live = logsumexp(logits)          │
                 │                     - logits[x_t]              │
                 │        (= per-token CE at target; scale-       │
                 │         invariant, fires everywhere)           │
                 │        update s_mem_ema_fast                   │
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
                 │  (4) update Hebbian trace, readout_drift,      │
                 │       prev_readout                              │
                 │                                                 │
                 │  (5) W stays at ~unit row RMS structurally:    │
                 │       the modulator EMAs it toward a            │
                 │       pre-RMSNormed delta_W, so no              │
                 │       per-matmul renormalization is needed.    │
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
                    ├──→ mem_head (weight-tied) ──→ logits
input_ids[t] ───────┘                                   │
                                                         ▼
                             s_mem_live(t) = logsumexp(logits) - logits[x_t]
                                             (proper CE, no EOT mask)
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

- **`s_mem_live`**: per-token memory-head cross-entropy at the observed
  token — `logsumexp(mem_head(prev_readout)) - target_logit`. Proper CE,
  so it's scale-invariant and bounded below at 0. Computed inside the
  per-token loop from `prev_readout` (readout at `t-1`) and `input_ids[t]`.
  Lower = memory is confident, higher = memory was surprised. **Fires at
  every position including document boundaries** — it's an observation
  signal for the modulator, not a training target, so EOT is not masked.
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
| mem_pred_weight | — | 0.1 | config.mem_pred_weight |
| gain_ema_fast | — | 0.3 | config.gain_ema_fast |
| Hebbian decay (per cell, learnable) | γ | sigmoid(2)≈0.88 init | hebbian_decay_logit |

## Memory Graph

### Connectivity: Dense W Matrix

Each cell has `W: [BS, NC, Cn, Cn] = [BS, 8, 32, 32]`. Runtime state
(not a learned parameter). Initialized sparse (K=8 nonzeros per row,
then row-RMSNormed), evolves via the neuromodulator. Message passing is
a plain bmm on the raw W — **no per-step renormalization**:

```python
received = torch.matmul(W, msg)            # [BS, NC, Cn, D_n]
```

The invariant that keeps W bounded is a **structural property of the
update**, not a renormalization hack. The modulator applies a convex EMA
toward a pre-RMSNormed delta:

```python
delta_W = F.rms_norm(delta_W_raw, normalized_shape=(N,))    # unit row RMS
W_new   = (1 - γ_W) * W + γ_W * delta_W                     # convex EMA
```

with `γ_W = sigmoid(W_decay_logit[cell])` — a learnable per-cell
plasticity rate, init `-3.0` (sigmoid ≈ 0.047, slow adaptation). Because
both operands of the convex combination are at ~unit row RMS, the result
stays at ~unit row RMS. No unbounded accumulator, no bf16 overflow, no
magic-number decay. The "write now, help later" credit path is preserved
because W is persistent runtime state updated additively (via the EMA)
across modulator calls.

### Neuromodulator

Per-cell MLP, runs every `modulation_interval=4` steps.

**Input** `[BS, NC, mod_in = 2*N + 4 + N*N = 1092]` for N=32:

```python
cat([h_norms, msg_norms,                  # 2*N    — per-neuron firing rates
     decay_mean,                          # 1      — per-cell average leakiness
     readout_drift,                       # 1      — per-cell volatility
     s_mem_live, s_mem_ema_fast,          # 2      — global surprise (broadcast)
     hebbian_flat,                        # N*N    — per-pair coactivation history
])
```

**Biological grounding**: the modulator reads only **rates** (per-neuron
firing magnitudes), **correlations** (Hebbian co-activation trace), and
**neuromodulatory signals** (surprise, drift). It does NOT have feature-
space "X-ray vision" into cell content — biological neuromodulators
(dopamine, ACh, norepinephrine) gate plasticity from rates and correlations,
not by reading individual neuron contents. The modulator output is also
restricted to plasticity controls (`delta_W`, `delta_decay`), never to
neuron identity vectors — identity is set during bootstrap and frozen.

**Hidden**: `[BS, NC, Hmod=2048]` (tanh activation, per-cell einsum)

**Output** `[BS, NC, mod_out=1056]`:
```python
delta_W_raw:      [BS, NC, N*N=1024]  → reshape [BS, NC, 32, 32]
delta_decay_raw:  [BS, NC, N=32]
```

Both outputs are **not applied directly**. They pass through bounded
convex-EMA updates:
```python
delta_W = F.rms_norm(delta_W_raw, normalized_shape=(N,))
W_new   = (1 - γ_W) * W + γ_W * delta_W
decay_new = (1 - γ_d) * decay + γ_d * sigmoid(delta_decay_raw)
```
with `γ_W = sigmoid(W_decay_logit[cell])` and `γ_d = sigmoid(decay_gamma_logit[cell])`
— learnable per-cell plasticity rates (both init `-3.0`, ≈ 0.047). This
guarantees W stays at ~unit row RMS and decay stays in `[0,1]` by
construction.

### Hebbian co-activation trace

A runtime state tensor `hebbian: [BS, NC, N, N]` (same shape as W) updated
after every memory step via:

```python
hebbian = (1 - γ) * hebbian + γ * (msg @ msg.T)
```

where `γ = sigmoid(hebbian_decay_logit[cell])` is a **learnable per-cell**
running-average rate (no magic number — each cell can learn its own
coactivation timescale). The update is on the autograd graph so the
modulator's downstream use of `hebbian` provides gradient back to
`hebbian_decay_logit`.

This is the biological "fire-together, wire-together" signal made
explicit. The modulator can learn to strengthen `W[i,j]` when
`hebbian[i,j]` is high, weaken when low — implementing classical Hebbian
plasticity gated by the global surprise signal.

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
W                  : [BS, NC, Cn, Cn]      — connectivity matrix (unit row RMS by construction)
decay              : [BS, NC, Cn]          — per-neuron temporal decay in [0,1]
hebbian            : [BS, NC, Cn, Cn]      — co-activation EMA trace
s_mem_live         : [BS]                   — current memory-head CE
s_mem_ema_fast     : [BS]                   — fast EMA of s_mem_live
readout_drift      : [BS, NC, 1]            — per-cell volatility
prev_readout       : [BS, D]                — last readout; reshaped per-cell on-demand for drift
```

All of these persist across segments and chunks — memory is lifelong.
Detached at TBPTT boundaries.

## Learned Parameters

| Component | Shape | Params | Notes |
|-----------|-------|--------|-------|
| neuron_id | [NC=8, Cn=32, D_n=256] | 65K | Per-neuron identity (frozen at inference) |
| state_w1/b1/w2/b2 | — | 198K | Shared state MLP |
| msg_w1/b1/w2/b2 | — | 131K | Shared message MLP |
| inject_w | [NC=8, alpha*D_n, D_n] | 2.1M | Per-cell, kaiming init |
| inject_b | [NC=8, alpha*D_n] | 8K | |
| mod_w1 | [NC=8, 1092, 2048] | 17.9M | Per-cell modulator layer 1 |
| mod_b1 | [NC=8, 2048] | 16K | |
| mod_w2 | [NC=8, 2048, 1056] | 17.3M | Per-cell modulator layer 2 |
| mod_b2 | [NC=8, 1056] | 8K | |
| hebbian_decay_logit | [NC=8] | 8 | Learnable per-cell Hebbian EMA rate (init 2.0 → γ≈0.88) |
| W_decay_logit | [NC=8] | 8 | Learnable per-cell W plasticity rate (init -3.0 → γ≈0.047) |
| decay_gamma_logit | [NC=8] | 8 | Learnable per-cell decay plasticity rate (init -3.0 → γ≈0.047) |
| **Memory total** | | **~37.7M** | |
| **LM total** | | **~67.3M** | |
| **Grand total** | | **~105.0M** | |

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
- `s_mem_*`, `readout_drift` are detached (no_grad observation signals).
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

Runtime state:
- **W**: sparse random (K=8 nonzeros per row at value 1.0, no self-connections), then row-RMSNormed to unit norm. Absolute value is irrelevant because of the immediate RMSNorm — only the sparsity pattern survives.
- **h**: small random (N(0, 0.01))
- **msg**: zeros
- **decay**: 0.5 (midrange in [0,1])
- **hebbian**: zeros
- **s_mem_live, s_mem_ema_fast**: zeros
- **readout_drift**: zeros
- **prev_readout**: zeros

Learned parameters:
- **mem_scale**: full(D, sqrt(alpha)) = 2.0 per dim
- **state_w1/w2, msg_w1/w2**: Xavier uniform with tanh gain (√2)
- **mod_w1, mod_w2**: Xavier uniform (linear gain = 1.0, via custom einsum-aware helper)
- **inject_w**: Xavier uniform
- **hebbian_decay_logit**: 2.0 (γ ≈ 0.88 — fast adaptation)
- **W_decay_logit, decay_gamma_logit**: -3.0 (γ ≈ 0.047 — slow adaptation)

## Architectural simplifications

Relative to earlier v9/v10 iterations:
- **No PCM** (predictive coding module) — the per-token memory-head CE
  serves the same "prediction-error → learning signal" role without a
  separate predictor network.
- **No split_mlp** combining H_mid with a surprise projection — the
  upper scan consumes `H_mid + mem_scale * readout` directly.
- **No surprise_proj** — surprise is a scalar EMA fed to the modulator,
  not a feature projected back into the LM residual stream.
- **No detach gymnastics beyond `H_mid.detach()`** isolating the LM
  lower scan from the memory path.

The surprise signal is principled (proper CE from the weight-tied
memory head), interpretable (scale-invariant, ≥0, matches LM CE units),
and live in the per-token compiled loop.
