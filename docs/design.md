# Neuromorphic Memory Graph — Dense-W Design

## Goals

1. Preserve the split-scan LM + PCM architecture.
2. One memory interaction per token.
3. GPU-friendly: all hot-path operations are batched matmuls (bmm) or F.linear.
4. Lifelong learning: neuromodulator continuously adjusts connectivity.
5. No document-boundary resets in the memory graph.
6. No discrete structural plasticity — connectivity evolves continuously via the neuromodulator.

## Architecture Overview

```
tokens → embedding → lower scan → PCM → split_mlp(H_mid, surprise) → H_aug
                                                                     │
                                             ┌───────────────────────┤
                                             ▼                       ▼
                                       memory graph             upper scan
                                             │                       │
                                             └──► combine ──────────►│
                                                                     ▼
                                                                 LM head
```

The LM is unchanged. The memory graph uses a dense connectivity matrix W
per cell, updated by a neuromodulator with low-rank deltas.

## Memory Graph

### Layout

The memory graph is a grid of cells. Each cell contains N neurons with
D_n-dim hidden states. Connectivity within a cell is represented as a
dense N×N weight matrix W.

### Default Dimensions

| Parameter | Symbol | Default |
|-----------|--------|---------|
| LM hidden dim | D | 2048 |
| Neuron hidden dim | D_n | 128 |
| Cells | N_cells | 16 |
| Grid | H × W | 4 × 4 |
| Neurons per cell | N | 128 |
| Total neurons | N_total | 2048 |
| Input ports per cell | alpha | 4 |
| Output ports per cell | alpha | 4 |
| Border neurons per cell | B | 4 |
| MLP groups | G | 8 |
| State MLP hidden | Hs | 256 |
| Msg MLP hidden | Hm | 256 |
| Modulator hidden | Hmod | 128 |
| Modulator rank | r | 16 |

### Connectivity: Dense W Matrix

Each cell has a connectivity matrix `W: [BS, NC, N, N]`. This is runtime
state (not a learned parameter). The matrix is initialized sparse (K
random nonzeros per row) and evolves continuously via the neuromodulator.

Message passing within a cell is a single batched matmul:

```python
received = torch.matmul(W, msg)    # [BS*NC, N, N] @ [BS*NC, N, D_n]
```

This replaces the sparse gather + Triton kernels from previous designs.
The backward is also a bmm — no atomic_add scatter needed.

**Soft sparsity**: W is not masked. Instead, a small per-step decay
nudges entries toward zero. The neuromodulator must actively maintain
connections it wants to keep. Connections drift to zero if not reinforced.

```python
W = W * (1 - w_decay)    # e.g., w_decay = 1e-3
```

### Runtime State

Per batch element:

```
h              : [BS, NC, N, D_n]     — neuron hidden states
msg            : [BS, NC, N, D_n]     — neuron messages
W              : [BS, NC, N, N]       — connectivity matrix
decay_logit    : [BS, NC, N]          — per-neuron temporal decay
cell_context   : [BS, NC, D_n]        — per-cell context (modulator output)
border_gate_logit : [BS, NC, B]       — cross-cell exchange gates
```

### Learned Parameters

```
neuron_id      : [NC, N, D_n]         — per-neuron identity embedding
state_w1       : [Hs, 2*D_n]         — shared state MLP layer 1
state_b1       : [Hs]
state_w2       : [D_n, Hs]           — shared state MLP layer 2
state_b2       : [D_n]
state_gs/gb    : [G, Hs/D_n]         — per-group scale/bias
msg_w1/b1/w2/b2: ...                 — shared message MLP (same structure)
msg_gs/gb      : ...                  — per-group scale/bias
mod_w1         : [NC, mod_in, Hmod]   — per-cell modulator layer 1
mod_b1         : [NC, Hmod]
mod_w2         : [NC, Hmod, mod_out]  — per-cell modulator layer 2
mod_b2         : [NC, mod_out]
inject_w/b     : [G, ...]            — per-group inject projection
mem_scale      : [D]                  — LM-side readout scale
```

### Cell Layout

Within each cell, neurons are ordered:

```
[0 : alpha)                    — input port neurons
[alpha : 2*alpha)              — output port neurons
[2*alpha : 2*alpha + B)        — border neurons
[2*alpha + B : N)              — internal neurons
```

## Per-Token Step

### 1. Receive (dense bmm)

```python
received = torch.matmul(W, msg)         # [BS, NC, N, D_n]
```

One batched matmul. Tensor cores handle [128, 128] tiles optimally.
Backward is two more bmms (grad_msg = W.T @ grad, grad_W = grad @ msg.T).

### 2. Inject

LM signal enters through input port neurons. Same as previous design:
H_aug_t reshaped to [BS, NC, D_n], projected through inject_w, added
to port neuron positions in received.

### 3. Border Exchange

Fixed geometric exchange between adjacent cells (N/S/E/W). Border neurons
receive gated messages from neighboring cells' border neurons.

### 4. State Update

Shared MLP with per-group conditioning. Input is `cat([received, h])`:

```python
candidate = grouped_mlp(cat([received, h]))    # [BS, NC, N, D_n]
h = sigmoid(decay) * h + (1 - sigmoid(decay)) * candidate
```

Identity and cell_context are NOT in the MLP input — neurons are
differentiated by their state h, their row in W, and the modulator's
per-neuron outputs.

### 5. Emit Message

Shared MLP with identity residual:

```python
msg = grouped_mlp(h) + identity
```

### 6. Readout

Output port neuron messages summed and scaled, reshaped to [BS, D].

### 7. Neuromodulate (every M steps)

The modulator consumes cell-level statistics and predicts low-rank
updates to W, plus per-neuron decay deltas.

**Input** (per cell):
```python
mod_input = cat([h_mean, msg_mean, cell_context, W_stats, decay_mean])
```

Where `W_stats` is a compact summary of W (e.g., row norms, or top singular
values — exact form TBD, start with row-mean of abs(W)).

**Output**: low-rank delta to W, plus scalar deltas:
```python
u = output[..., :N*r].reshape(BS, NC, N, r)
v = output[..., N*r:2*N*r].reshape(BS, NC, N, r)
delta_W = u @ v.T                               # [BS, NC, N, N]
delta_decay = output[..., 2*N*r : 2*N*r + N]
delta_ctx = output[..., 2*N*r + N : 2*N*r + N + D_n]
delta_border = output[..., -B:]

W = W + delta_W
decay_logit = decay_logit + delta_decay
cell_context = cell_context + delta_ctx
border_gate_logit = border_gate_logit + delta_border
```

**W decay** (soft sparsity, every step):
```python
W = W * (1 - w_decay_rate)
```

This means every connection loses a small fraction of its weight each
step. The modulator must reinforce connections it wants to keep. Unused
connections naturally decay to zero — this IS structural plasticity,
but continuous rather than discrete.

**Modulator dimensions**:
```python
mod_in  = 3*D_n + N + 1 = 3*128 + 128 + 1 = 513
    # h_mean[D_n] + msg_mean[D_n] + ctx[D_n] + W_row_norms[N] + decay_mean[1]
mod_out = 2*N*r + N + D_n + B = 2*128*16 + 128 + 128 + 4 = 4356
```

## Initialization

**W**: sparse random init. For each neuron, K=8 random connections set to
small positive values (e.g., 0.1), rest zero.

**h, msg**: small random / zeros.

**decay_logit**: zeros (sigmoid(0) = 0.5, equal blend).

## Training

- Single optimizer for LM + memory parameters.
- W is NOT in the optimizer — it's runtime state adjusted by the modulator.
- Memory parameters at reduced LR (0.3x).
- f32 parameters, bf16 compute via autocast.
- Block TBPTT: detach every 8 tokens.
- Block checkpointing: checkpoint every 8 tokens.
- Modulation interval: every 1-4 tokens (cheap enough for every step).

## Why This Design

### Faster backward
The sparse gather backward used atomic_add scatter (contentious, slow).
Dense bmm backward is just another bmm — clean, fast, well-optimized.

### Simpler codebase
No Triton sparse gather kernels. No conn_idx buffers. No structural
plasticity rewiring. No CSR edge buffers. The entire connectivity is
one dense matrix per cell.

### Continuous plasticity
Instead of discrete rewiring every 1024 tokens, the modulator continuously
adjusts W. The W decay provides a forgetting pressure that keeps the
graph from accumulating stale connections. This is biologically realistic:
synaptic weights are constantly being adjusted, not periodically rewired.

### Better gradient signal
The modulator gets clean gradient through the bmm chain:
loss → readout → msg → h → received = W @ msg → W → delta_W → modulator.
No atomic_add noise in the backward pass.
