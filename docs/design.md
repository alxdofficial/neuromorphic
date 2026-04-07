# Neuromorphic Memory Graph — Dense-W Design

## Goals

1. Split-scan LM with interleaved PCM and memory graph.
2. One memory interaction per token.
3. GPU-friendly: all hot-path operations are batched matmuls (bmm) or F.linear.
4. Lifelong learning: neuromodulator continuously adjusts connectivity.
5. No document-boundary resets in the memory graph.
6. No discrete structural plasticity — connectivity evolves continuously.
7. Surprise-driven memory management: PCM surprise feeds into the neuromodulator.

## Architecture Overview

```
tokens → embedding → lower scan → H_mid[0..T-1]  (parallel)
                                       │
                           ┌───────────┘
                           ▼
                 ┌─── per-token loop (t = 0..T-1) ───┐
                 │                                     │
                 │  PCM(H_mid[t], prev_readout)        │
                 │       → surprise[t]                 │
                 │       → update surprise_ema          │
                 │                                     │
                 │  augment(H_mid[t], surprise[t])     │
                 │       → H_aug[t]                    │
                 │                                     │
                 │  memory_step(H_aug[t])              │
                 │       → readout[t]                  │
                 │                                     │
                 │  H_enriched[t] = H_aug[t]           │
                 │       + mem_scale * readout[t]      │
                 │                                     │
                 │  (every M steps: modulate with      │
                 │   surprise_compressed as input)      │
                 │                                     │
                 └─────────────────────────────────────┘
                           │
                           ▼
                 upper scan → H_upper[0..T-1]  (parallel)
                           │
                           ▼
                        LM head → logits
```

The lower and upper scans run in parallel over T. The middle section
(PCM + memory) runs sequentially per token. The PCM is interleaved
with the memory graph, receiving the previous step's memory readout
as context for its predictions.

## Predictive Coding Module (PCM)

### What it predicts

The PCM predicts the LM's hidden state transition: `H_mid[t+1] - H_mid[t]`.

**Input**: `H_mid[t]` (current LM state) + `prev_readout` (what memory
contributed last step). The memory readout gives the PCM context — if
memory already "knows" about an upcoming transition, the PCM can predict
it, producing low surprise.

**Output**: `delta_hat[t]` (predicted transition) + `surprise[t]` (prediction
error = `delta_hat[t-1] - delta_actual[t]`).

**Target**: `H_mid[t+1] - H_mid[t]` — fully determined by the lower scan
before the memory loop starts. No circularity: surprise[t] affects H_aug[t]
and the upper scan, but NOT H_mid[t+1].

### PCM loss

Self-supervised: `pred_loss = MSE(delta_hat[t], (H_mid[t+1] - H_mid[t]).detach())`.
Weighted by `pcm_pred_weight` and added to CE loss as auxiliary loss.
The PCM trains to predict accurately, independent of how the memory
uses its surprise signal.

### Surprise flow into memory

```
PCM outputs surprise[t]           [BS, C, D_cc] = [BS, 16, 128]
    ↓ reshape to per-cell
surprise_per_cell[t]              [BS, NC, D_n] = [BS, 8, 256]
    ↓ EMA accumulation (detached, per step)
surprise_ema                      [BS, NC, D_n]
    ↓ also accumulate readout
readout_ema                       [BS, NC, D_n]
    ↓ learned projection (at modulation time, ON the CE graph)
surprise_compressed               [BS, NC, proj_dim=64]
    ↓ concatenated into modulator input
modulator decides how to adjust W, decay, context, gates
```

**surprise_ema** and **readout_ema** are runtime state (EMA, no grad).
They capture "what's been surprising recently" and "what memory has been
contributing recently." The temporal smoothing filters noise.

**surprise_proj** is a learned parameter `[proj_dim, 2*D_n]` trained
end-to-end via CE loss through the modulator. It learns which aspects
of surprise and readout are useful for memory management decisions.
The PCM itself (trained by aux_loss) doesn't know about this — clean
separation between "predict accurately" and "use predictions well."

## Memory Graph

### Layout

Grid of cells. Each cell contains N neurons with D_n-dim hidden states.
Connectivity within a cell is a dense N×N weight matrix W.

### Default Dimensions

| Parameter | Symbol | Default |
|-----------|--------|---------|
| LM hidden dim | D | 2048 |
| Neuron hidden dim | D_n | 256 |
| Cells | N_cells | 8 |
| Grid | H × W | 2 × 4 |
| Neurons per cell | N | 32 |
| Total neurons | N_total | 256 |
| Input/output ports per cell | alpha | 4 |
| Border neurons per cell | B | 4 |
| State MLP hidden | Hs | 256 |
| Msg MLP hidden | Hm | 256 |
| Modulator hidden | Hmod | 128 |
| Modulator rank | r | 16 |
| Surprise projection dim | proj_dim | 64 |

### Connectivity: Dense W Matrix

Each cell has a connectivity matrix `W: [BS, NC, N, N]`. Runtime state,
not a learned parameter. Initialized sparse (K nonzeros per row),
evolves continuously via the neuromodulator.

Message passing is a single batched matmul:
```python
received = torch.matmul(W, msg)
```

Soft sparsity via per-step decay:
```python
W = W * (1 - w_decay_rate)
```

### Runtime State

```
h                : [BS, NC, N, D_n]     — neuron hidden states
msg              : [BS, NC, N, D_n]     — neuron messages
W                : [BS, NC, N, N]       — connectivity matrix
decay_logit      : [BS, NC, N]          — per-neuron temporal decay
cell_context     : [BS, NC, D_n]        — per-cell context
border_gate_logit: [BS, NC, B]          — cross-cell exchange gates
surprise_ema     : [BS, NC, D_n]        — EMA of per-cell surprise (NEW)
readout_ema      : [BS, NC, D_n]        — EMA of per-cell readout (NEW)
prev_readout     : [BS, D]              — previous step's readout for PCM (NEW)
```

### Learned Parameters

```
neuron_id        : [NC, N, D_n]         — per-neuron identity embedding
state_w1         : [Hs, 2*D_n]         — shared state MLP (split into recv/h halves)
state_b1         : [Hs]
state_w2         : [D_n, Hs]
state_b2         : [D_n]
msg_w1/b1/w2/b2  : ...                 — shared message MLP
mod_w1           : [NC, mod_in, Hmod]   — per-cell modulator
mod_b1/w2/b2     : ...
inject_w/b       : [G, ...]            — per-group inject projection
surprise_proj_w  : [proj_dim, 2*D_n]   — surprise compression (NEW)
surprise_proj_b  : [proj_dim]          — (NEW)
mem_scale        : [D]                  — LM-side readout scale
pcm_w1/b1/w2/b2  : [C, ...]           — PCM prediction MLP (MOVED from LM)
```

## Per-Token Step (inside memory loop)

### 0. PCM (NEW location — inside the loop)

```python
# PCM input: current LM state + previous memory readout
pcm_input = cat([H_mid_cols[t], prev_readout_cols], dim=-1)  # [BS, C, 2*D_cc]
delta_hat[t] = pcm_mlp(pcm_input)                            # [BS, C, D_cc]

# Surprise (using previous prediction)
if t > 0:
    delta_actual = H_mid_cols[t] - H_mid_cols[t-1]
    surprise[t] = prev_delta_hat - delta_actual
prev_delta_hat = delta_hat[t]

# Accumulate per-cell surprise (detached EMA)
surprise_cell = surprise[t].reshape(BS, NC, D_n)
surprise_ema = 0.95 * surprise_ema + 0.05 * surprise_cell.detach()
```

### 0b. Augment

```python
H_aug[t] = H_mid[t] + split_mlp(H_mid[t], surprise[t])
```

### 1. Receive (dense bmm)

```python
received = torch.matmul(W, msg)
```

### 2. Inject

```python
received[:, :, :alpha] += inject(H_aug[t])
```

### 3. Border Exchange

```python
received[:, :, border_lo:border_hi] += border_exchange(msg, border_gate)
```

### 4. State Update

```python
h = decay * h + (1-decay) * tanh(mlp(received, h))
```

### 5. Emit Message

```python
msg = tanh(mlp(h)) + identity
```

### 6. Readout

```python
readout[t] = readout_from_output_ports(msg)
readout_cell = readout[t].reshape(BS, NC, D_n)
readout_ema = 0.95 * readout_ema + 0.05 * readout_cell.detach()
prev_readout = readout[t]

H_enriched[t] = H_aug[t] + mem_scale * readout[t]
```

### 7. Neuromodulate (every M steps)

```python
# Compress surprise for modulator (ON the CE graph via surprise_proj)
surprise_compressed = F.linear(
    cat([surprise_ema, readout_ema]), surprise_proj_w, surprise_proj_b)

mod_input = cat([h_mean, msg_mean, cell_context, W_stats, decay_mean,
                 surprise_compressed])

# Modulator predicts low-rank delta_W + deltas
delta_W = u @ v.T
W = W + delta_W
decay_logit += delta_decay
cell_context += delta_ctx
border_gate_logit += delta_border

# Soft W decay
W = W * (1 - w_decay_rate)
```

## Training

### Losses

```
total_loss = CE_loss + pcm_pred_weight * pcm_pred_loss
```

- **CE loss**: trains LM (lower/upper scan, embedding, head, split_mlp, mem_scale)
  AND memory graph (state/msg MLPs, modulator, inject, surprise_proj)
- **PCM pred_loss**: trains PCM only (self-supervised transition prediction)

### Gradient boundaries

- `H_aug.detach()` before memory: memory can't backprop into LM/PCM
- `surprise_ema` and `readout_ema`: detached EMAs, no grad
- `surprise_proj`: ON the CE graph — learns end-to-end through modulator
- PCM: trained ONLY by pred_loss, not by CE loss

### What each component learns

| Component | Trained by | Learns to... |
|-----------|-----------|--------------|
| PCM | pred_loss (self-supervised) | Predict LM transitions accurately |
| split_mlp | CE loss | Combine H_mid and surprise usefully |
| State/Msg MLPs | CE loss (via readout) | Update neuron states and messages |
| Modulator | CE loss (via readout) | Adjust W, decay, gates based on surprise + state |
| surprise_proj | CE loss (via modulator) | Select which surprise dims help memory management |
| W (runtime) | Not trained (modulator adjusts it) | Store connectivity patterns |

## Initialization

- **W**: sparse random (K=8 nonzeros per row, value 0.1)
- **h, msg**: small random / zeros
- **decay_logit**: zeros (sigmoid(0) = 0.5)
- **surprise_ema, readout_ema**: zeros
- **prev_readout**: zeros
