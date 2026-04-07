# Neuromorphic Memory Graph — Dense-W Design

## Goals

1. Split-scan LM with interleaved PCM and memory graph.
2. One memory interaction per token.
3. GPU-friendly: all hot-path operations are batched matmuls (bmm) or F.linear.
4. Lifelong learning: neuromodulator continuously adjusts connectivity.
5. No document-boundary resets in the memory graph.
6. Surprise-driven memory management: PCM surprise feeds into the neuromodulator.

## Architecture Overview

```
tokens → embedding → lower scan → H_mid[0..T-1]  (parallel, fused_scan)
                                       │
                           ┌───────────┘
                           ▼
                 ┌─── per-token loop (t = 0..T-1) ───┐
                 │                                     │
                 │  PCM(H_mid[t], prev_readout)        │
                 │       → surprise[t] (detached)      │
                 │       → update surprise_ema          │
                 │                                     │
                 │  augment_single(H_mid[t], surprise)  │
                 │       → H_aug[t]                    │
                 │       (split_mlp: Linear(2D,128)    │
                 │        → SiLU → Linear(128,D))      │
                 │                                     │
                 │  memory_step(H_aug[t])              │
                 │       → W @ msg (dense bmm)         │
                 │       → inject + border exchange     │
                 │       → state MLP + decay blend     │
                 │       → message MLP + identity      │
                 │       → readout[t]                  │
                 │                                     │
                 │  update readout_ema                  │
                 │                                     │
                 │  (every 4 steps: modulate with      │
                 │   surprise_compressed as input)      │
                 │                                     │
                 └─────────────────────────────────────┘
                           │
                           ▼
         H_enriched = H_mid + mem_scale * mem_out
                           │
                           ▼
                 upper scan → H_upper[0..T-1]  (parallel, fused_scan)
                           │
                           ▼
                        LM head → logits
```

Note: H_enriched is computed from H_mid (not H_aug). The surprise
augmentation (H_aug) is used only inside the memory loop for the
inject step. The upper scan sees H_mid + scaled readout.

## Default Dimensions

| Parameter | Symbol | Default | Source |
|-----------|--------|---------|--------|
| LM hidden dim | D | 2048 | config.D |
| Embedding dim | D_embed | 768 | config.D_embed |
| Scan layers | L_total | 4 | config.L_total |
| Scan split point | split_at | 2 | config.scan_split_at |
| Scan recurrence dim | d_inner | 1200 | config.d_inner |
| Cortical columns (PCM) | C | 16 | config.C |
| Per-column dim | D_cc | 128 | D // C |
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
| Split MLP hidden | — | 128 | config.split_mlp_hidden |
| Surprise proj dim | proj_dim | 64 | config.surprise_proj_dim |
| PCM hidden | — | 256 | config.pcm_hidden |
| Modulation interval | M | 4 | config.modulation_interval |
| TBPTT block | — | 8 | config.tbptt_block |
| Checkpoint every | — | 8 | config.checkpoint_every |
| W decay rate | — | 1e-3 | config.w_decay_rate |
| Surprise EMA decay | — | 0.95 | config.surprise_ema_decay |

## Predictive Coding Module (PCM)

### Location

Inside the memory graph's per-token loop (`MemoryGraph._run_block`),
NOT in the LM. The PCM is owned by `MemoryGraph` as `self.pcm`.

### What it predicts

H_mid state transitions: `delta_hat[t] = PCM_MLP(H_mid[t], prev_readout)`.
Target: `H_mid[t+1] - H_mid[t]` (fixed, from lower scan).

### Input

Per cortical column: `cat([H_mid_cols[t], prev_readout_cols], dim=-1)`.
Shape: `[BS, C=16, 2*D_cc=256]`.

The prev_readout gives the PCM memory context — if memory already
"knows" about an upcoming transition, the PCM can predict it (low surprise).

### Forward

Per-column RMSNorm on the concatenated input, then per-column MLP:
`bmm([C, BS, 2*D_cc], [C, 2*D_cc, hidden]) → SiLU → bmm → delta_hat`.
PCM weights are f32, cast to compute dtype (bf16) inside `predict()`.

### Surprise computation

```python
if t > 0 and prev_delta_hat is not None:
    delta_actual = H_mid_cols[t] - H_mid_cols[t-1]
    surprise_t = (prev_delta_hat - delta_actual).detach()  # no CE grad into PCM
    pcm_loss += (prev_delta_hat - delta_actual.detach()).pow(2).mean()
```

The `.detach()` on surprise_t ensures CE loss cannot reach PCM through
the augment path. The pcm_loss computation keeps PCM gradients flowing.

### PCM loss

Self-supervised: `pcm_pred_loss = MSE(delta_hat, delta_actual.detach())`.
Added to total loss as `pcm_pred_weight * pcm_pred_loss` (default 0.1).

## Surprise Pipeline

```
PCM outputs surprise_t         [BS, C=16, D_cc=128]    (detached)
    ↓ reshape (C*D_cc = NC*D_n = 2048)
surprise_cell                  [BS, NC=8, D_n=256]
    ↓ EMA accumulation (no_grad, per step)
surprise_ema                   [BS, NC, D_n]            (runtime state)
    ↓ also accumulate readout
readout_ema                    [BS, NC, D_n]            (runtime state)
    ↓ at modulation time: learned projection (ON the CE graph)
    ↓ F.linear(cat([surprise_ema, readout_ema]), surprise_proj_w)
surprise_compressed            [BS, NC, proj_dim=64]
    ↓ concatenated into modulator input
modulator adjusts W, decay, context, gates
```

`surprise_proj_w: [proj_dim, 2*D_n]` is trained end-to-end via CE loss
through the modulator. It learns which aspects of surprise and readout
history are useful for memory management decisions.

## Memory Graph

### Connectivity: Dense W Matrix

Each cell has `W: [BS, NC, Cn, Cn] = [BS, 8, 32, 32]`. Runtime state
(not a learned parameter). Initialized sparse (K=8 nonzeros per row),
evolves via the neuromodulator. Message passing is a single bmm:

```python
received = torch.matmul(W, msg)    # [BS, NC, Cn, D_n]
```

Soft sparsity: `W *= (1 - w_decay_rate)` each step (no_grad).

### Neuromodulator

Per-cell MLP, runs every `modulation_interval=4` steps.

**Input** `[BS, NC, mod_in=834]`:
```python
cat([h_mean, msg_mean, cell_context, W_stats, decay_mean, surprise_compressed])
#    [D_n]   [D_n]     [D_n]        [1]      [1]         [proj_dim=64]
```

**Hidden**: `[BS, NC, Hmod=2048]` (tanh activation, per-cell einsum)

**Output** `[BS, NC, mod_out=1316]`:
```python
delta_W:      [BS, NC, N*N=1024]  → reshape to [BS, NC, 32, 32]  (direct, full resolution)
delta_decay:  [BS, NC, N=32]
delta_ctx:    [BS, NC, D_n=256]
delta_border: [BS, NC, B=4]
```

The modulator directly outputs all 1024 entries of delta_W — no
low-rank factorization. With Cn=32, the full W matrix is small enough
to predict directly.

### Cell Layout

```
[0 : 4)      — input port neurons
[4 : 8)      — output port neurons
[8 : 12)     — border neurons
[12 : 32)    — internal neurons
```

### Inject

H_aug_t `[BS, D=2048]` reshaped to `[BS, NC=8, D_n=256]`, projected
through per-group `inject_w: [G=8, alpha*D_n, D_n]` with orthogonal
init, added to input port neuron positions.

### Border Exchange

Grid-reshape + slice shifts (N/S/E/W). Vectorized, no Python loops:
```python
border = msg[:, :, 8:12].reshape(BS, grid_h, grid_w, 4, D_n)
incoming[:, 1:, :, 0] = border[:, :-1, :, 1]   # north
...
```
Gated by `sigmoid(border_gate_logit)`.

### State Update

Split first-layer optimization: no `cat([received, h])` tensor allocation.
```python
hidden = tanh(F.linear(received, w1_recv) + F.linear(h, w1_h, b1))
candidate = tanh(F.linear(hidden, w2, b2))
h_new = decay * h + (1 - decay) * candidate
```
Shared weights across all neurons. No group scale/bias.

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
h                 : [BS, NC, Cn, D_n]     — neuron hidden states
msg               : [BS, NC, Cn, D_n]     — neuron messages
W                 : [BS, NC, Cn, Cn]      — connectivity matrix
decay_logit       : [BS, NC, Cn]          — per-neuron temporal decay
cell_context      : [BS, NC, D_n]         — per-cell context
border_gate_logit : [BS, NC, B]           — cross-cell gates
surprise_ema      : [BS, NC, D_n]         — EMA of per-cell surprise
readout_ema       : [BS, NC, D_n]         — EMA of per-cell readout
prev_readout      : [BS, D]              — last readout for PCM input
prev_delta_hat    : [BS, C, D_cc] or None — last PCM prediction
```

## Learned Parameters

| Component | Shape | Params | Notes |
|-----------|-------|--------|-------|
| neuron_id | [NC=8, Cn=32, D_n=256] | 65K | Per-neuron identity |
| state_w1 | [Hs=256, 2*D_n=512] | 131K | Shared, split into recv/h halves |
| state_b1/w2/b2 | — | 66K | Shared state MLP |
| msg_w1/b1/w2/b2 | — | 131K | Shared message MLP |
| inject_w | [G=8, alpha*D_n, D_n] | 2.1M | Per-group orthogonal init |
| inject_b | [G=8, alpha*D_n] | 8K | |
| mod_w1 | [NC=8, 834, 2048] | 13.7M | Per-cell modulator layer 1 |
| mod_b1 | [NC=8, 2048] | 16K | |
| mod_w2 | [NC=8, 2048, 1316] | 21.6M | Per-cell modulator layer 2 |
| mod_b2 | [NC=8, 1316] | 11K | |
| surprise_proj_w/b | [64, 512] + [64] | 33K | Surprise compression |
| PCM (pcm_w1/b1/w2/b2, norm) | — | 1.6M | Per-column prediction MLP |
| split_mlp | Linear(4096,128)+Linear(128,2048) | 787K | Surprise augmentation |
| mem_scale | [D=2048] | 2K | Readout scaling |
| **Memory total** | | **~39.4M** | |
| **LM total** | | **~68.1M** | |
| **Grand total** | | **~107.5M** | |

## Training

### Losses

```
total_loss = CE_loss + pcm_pred_weight * pcm_pred_loss
```

CE loss trains: LM (scan layers, embedding, head), split_mlp, mem_scale,
memory graph (state/msg MLPs, modulator, inject, surprise_proj).

PCM pred_loss trains: PCM only (self-supervised transition prediction).

### Gradient boundaries

- `H_mid.detach()` before memory: LM lower scan isolated from memory
- `surprise_t.detach()` before augment_fn: CE loss cannot reach PCM
- `surprise_ema/readout_ema`: detached EMAs (no_grad), no gradient
- `surprise_proj`: ON the CE graph — trained through modulator
- `prev_delta_hat.detach()` at TBPTT boundaries

### EOT handling

- **LM scan**: resets carry at positions following EOT tokens (reset_mask).
  This prevents the scan from carrying hidden state across unrelated documents.
- **Memory graph**: does NOT reset at EOT. Memory is lifelong — it persists
  across document boundaries. This is deliberate: the memory accumulates
  knowledge over the entire training sequence.
- **CE loss**: tokens where `input_ids == eot_id` are excluded from the loss.

### Block structure

- TBPTT: detach all state every `tbptt_block=8` tokens
- Checkpointing: checkpoint blocks of `checkpoint_every=8` tokens
- Modulation: every `modulation_interval=4` tokens

## Initialization

- **W**: sparse random (K=8 nonzeros per row, value 0.1, no self-connections)
- **h**: small random (N(0, 0.01))
- **msg, decay_logit, cell_context, border_gate_logit**: zeros
- **surprise_ema, readout_ema, prev_readout**: zeros
- **mem_scale**: sqrt(alpha) = 2.0
- **split_mlp output layer**: depth-scaled init (× 1/sqrt(2*L_total))
- **mod_w2**: small init (× 0.01)
- **inject_w**: per-port orthogonal blocks
