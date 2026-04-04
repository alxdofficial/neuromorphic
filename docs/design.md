# Neuromorphic Memory Graph — Cell-Grid Design

## Goals

1. Preserve the current split-scan LM + PCM architecture.
2. Keep memory interaction at least once per token.
3. Make the memory graph materially more GPU-friendly than the current flat global sparse graph.
4. Keep neuron-level expressivity through identity-conditioned dynamics.
5. Preserve lifelong memory: no document-boundary resets in the memory graph.

## Architecture Overview

```
tokens → embedding → lower scan → PCM → split_mlp(H_mid, surprise) → H_aug
                                                                     │
                                             ┌───────────────────────┤
                                             ▼                       ▼
                                       cell-grid memory         upper scan
                                             │                       │
                                             └──► combine ──────────►│
                                                                     ▼
                                                                 LM head
```

The language model remains a split recurrent scan stack. The change is entirely
in the memory system: we replace the current flat `N=8096, K=64` global sparse
manifold with a cell-grid memory graph laid out for locality.

## High-Level Design

### Core change

The memory graph is no longer one globally connected sparse neuron manifold.
Instead:

- the graph is divided into **64 cells**
- cells are arranged on an **8 x 8 grid**
- each cell owns a single `D_n=32` LM slice
- most connectivity is **strictly within-cell**
- cross-cell communication happens only through a small set of **border neurons**
  with fixed grid-local exchange

This makes memory access patterns much more regular while keeping the graph
stateful and recurrent at token resolution.

### Why this layout

The current flat design is dominated by:

- irregular global gather on `msg[:, conn_idx]`
- poor cache locality
- a giant `[BS, N, K, D_n]` intermediate
- too much logical specialization in the per-neuron modulator

The cell-grid design fixes that by:

- making most reads local to contiguous cell storage
- replacing arbitrary global sparse exchange with grid-local border exchange
- moving specialization from per-neuron parameter banks toward
  cell-conditioned / group-conditioned computation

## LM Side

The LM remains unchanged in structure:

- lower scan produces `H_mid`
- PCM predicts transitions and emits `surprise`
- `split_mlp(H_mid, surprise) -> H_aug`
- memory graph receives `H_aug.detach()`
- upper scan consumes `H_enriched = H_aug + mem_scale * mem_out`

`H_aug.detach()` remains a deliberate design choice. It keeps lower-scan
optimization cleaner while still allowing CE gradients to train the entire
memory graph through `mem_out`.

## Cell-Grid Memory Graph

### Default Dimensions

| Parameter | Symbol | Value |
|-----------|--------|-------|
| LM hidden dim | `D` | 2048 |
| Neuron hidden dim | `D_n` | 32 |
| Cells | `N_cells` | 64 |
| Grid | `H x W` | 8 x 8 |
| Neurons per cell | `C_n` | 128 |
| Total neurons | `N_total` | 8192 |
| Local neighbors per neuron | `K_local` | 32 |
| Input ports per cell | `alpha_in` | 4 |
| Output ports per cell | `alpha_out` | 4 |
| Border neurons per cell | `B` | 4 |
| Internal neurons per cell | `C_n - 12` | 116 |

Because `D / D_n = 2048 / 32 = 64`, we assign exactly one cell to each memory
slice. This gives a simple one-to-one mapping:

```
H_aug[BS, 2048] -> reshape [BS, 64, 32] -> one 32-dim slice per cell
```

### Cell Layout

Each cell stores neurons in a fixed contiguous order:

1. input port neurons: indices `0..3`
2. output port neurons: indices `4..7`
3. border neurons: indices `8..11`
4. internal neurons: indices `12..127`

This keeps inject/readout/border handling simple and branch-free.

### Runtime State

Each batch element maintains:

- `h[BS, N_cells, C_n, D_n]`
- `msg[BS, N_cells, C_n, D_n]`
- `w_local[BS, N_cells, C_n, K_local]`
- `decay_logit[BS, N_cells, C_n]`
- `cell_context[BS, N_cells, D_n]`
- `border_gate_logit[BS, N_cells, B]`
- `hebbian[BS, N_cells, C_n, K_local]`

The neuron identity vectors are learned parameters:

- `neuron_id[N_cells, C_n, D_n]`

Identity is static learned structure, not a runtime-reset state.

## Connectivity

### Within-cell connectivity

Each neuron chooses `K_local=32` presynaptic neighbors from the same cell only.

- no self-connections
- no cross-cell random edges
- structural plasticity rewires only within the cell

The local connection index tensor is:

- `conn_local[N_cells, C_n, K_local]`

### Border exchange

Cross-cell communication is fixed and geometric, not random.

Each cell has 4 border neurons corresponding to:

- north
- south
- west
- east

At each token step, each border neuron receives message input from the matching
opposite-direction border neuron in the adjacent grid cell:

- north border reads south border from the northern neighbor
- south border reads north border from the southern neighbor
- west border reads east border from the western neighbor
- east border reads west border from the eastern neighbor

Cells on the edge of the grid receive zeros for missing neighbors.

This is far more GPU-friendly than arbitrary cross-manifold sparse edges and
still allows information to move across the whole grid over time.

## Per-Token Memory Step

For each token `t`:

1. Optionally run the **per-cell modulator** if `t % modulation_interval == 0`
2. **Local receive**: each neuron gathers `K_local` neighbor messages from its own cell
3. **Inject**: project the cell’s LM slice into distinct port-specific views and add
   them to the input port neurons
4. **Border exchange**: add directional neighbor-cell messages to the 4 border neurons
5. **State update**: grouped state MLP computes candidate state from
   `(received, h, neuron_id, cell_context, decay)`
6. **Temporal blend**: `h = sigmoid(decay) * h + (1 - sigmoid(decay)) * candidate`
7. **Emit message**: grouped message MLP computes outgoing message from
   `(h, neuron_id, cell_context)`
8. **Readout**: collect output port messages from each cell, reduce replicas,
   and reshape back to `[BS, D]`
9. **Update Hebbian traces** using local gathered presynaptic messages and new messages

The graph still runs once per token. The design change is about locality and
parameter organization, not skipping memory interaction.

## Parameterization

### Grouped state and message MLPs

The fast per-token dynamics are no longer one globally shared MLP, and no
longer a separate MLP bank per neuron.

Instead:

- cells are assigned to a small number of **MLP groups**
- each group owns one state MLP and one message MLP
- neuron identity and cell context provide within-group specialization

This gives a better throughput / expressivity tradeoff than either extreme.

Recommended default:

- `mlp_groups = 8`
- each group serves 8 cells

### Per-cell modulator

The slow adaptation path is **per-cell**, not per-neuron.

Every `modulation_interval` tokens, each cell aggregates:

- mean hidden state
- mean message state
- mean Hebbian trace
- mean decay
- current cell context

and feeds those statistics into a per-cell MLP that outputs deltas for:

- local connection logits `w_local`
- per-neuron decay logits
- `cell_context`
- border gate logits

This keeps adaptation specialized while avoiding the current per-neuron
parameter explosion.

## Injection and Readout

### Injection

`H_aug_t[BS, D]` is reshaped to `[BS, N_cells, D_n]`.

For each cell, its 32-dim slice is not copied identically to all input ports.
Instead, a small grouped inject projection maps

```python
[BS, N_cells, D_n] -> [BS, N_cells, alpha_in, D_n]
```

so the 4 input-port neurons receive 4 different learned views of the same cell
slice.

This keeps the ingress path expressive without giving up locality. It is also
cheap: the inject projection operates on `[BS, N_cells, D_n]`, not on all
neurons, so its cost is tiny relative to local gather and state/message updates.

### Readout

For each cell, the 4 output port messages are summed and scaled by
`1 / sqrt(alpha_out)`, then reshaped back to `[BS, D]`.

Combination remains:

```
H_enriched = H_aug + mem_scale * mem_out
```

## Structural Plasticity

Structural plasticity remains slow and local.

- interval: every 1024 tokens by default
- signal: batch-averaged local Hebbian traces
- prune: lowest-scoring local connections
- regrow: only within the same cell
- invariant: `K_local` distinct presynaptic neighbors per neuron

No global rewiring is allowed in this design.

## Training

- single optimizer for LM + memory
- memory parameters use lower LR scale
- memory parameters remain in f32
- memory runtime state in bf16
- no document-boundary resets in the memory graph
- chunk boundary only detaches gradients, not state values
- use **block TBPTT** inside a segment, not detach every token
- use **block checkpointing** over token ranges instead of checkpointing every step

Recommended defaults:

- `tbptt_block = 8`
- `checkpoint_every = 8`
- `modulation_interval = 4`

## Why this design is more GPU-friendly

Compared with the current branch, this plan:

- replaces one giant global sparse gather with many local cell gathers
- removes arbitrary cross-manifold sparse edges
- uses fixed directional border exchange implemented with reshape / slice / roll
- reduces parameter specialization from per-neuron to per-cell or per-group
- aligns memory layout with the natural LM slice layout
- enables future fused kernels at the **cell** granularity rather than the
  entire flat manifold

This is still not as GPU-friendly as a pure dense transformer-style block, but
it is much better aligned with GPU execution than the current flat sparse graph.
