# v11 Architecture — Cell-Based Neuromorphic Memory Graph

**Branch**: `v11-cells`
**Updated**: 2026-04-03

---

## Overview

A flat world of 256 cells, each containing 256 thin neurons (D=8). Neurons
connect only within their cell. Multiple message-passing rounds per token step
compensate for thin per-neuron dimensionality. Dedicated inject/readout port
neurons interface with the language model without signal dilution.

**31,744 total neurons. ~55M memory params (per-neuron modulator). ~107M total.**

---

## Architecture Diagram

```
Input → Embedding → proj_up (768→2048)
  → LOWER SCAN (2 layers, d_inner=580)
  → PCM: predict transitions (H[t+1]-H[t]), surprise = predicted − actual
  → MEMORY GRAPH (cell-based, per-token interaction)
      Cell Modulator: per-cell MLP on aggregated stats → w_conn, decay, primitives
      For each token t:
        For R=4 rounds:
          1. Inject LM signal into inject neurons (4 per cell)
          2. Cell-local gather: each neuron reads K=16 neighbors in same cell
          3. State MLP (shared): update from (received, primitives, neuron_id, decay)
          4. Temporal integration: h = decay * h + (1-decay) * update
          5. Msg MLP (shared): msg from (h, primitives, neuron_id) + neuron_id
        Readout: average readout neurons (4 per cell) → D_lm
  → INJECT: H_enriched = H_mid + mem_scale * mem_out
  → Split-point MLP: H = H + split_mlp(cat(H, RMSNorm(surprise)))
  → UPPER SCAN (2 layers)
  → proj_down (2048→768) → ln_final → lm_head → logits
```

---

## Config (tier_a)

| Parameter | Value | Notes |
|-----------|-------|-------|
| D (LM) | 2048 | LM hidden dimension |
| D_embed | 768 | Embedding dimension |
| L_total | 4 | Scan layers (2 lower + 2 upper) |
| d_inner | 580 | Scan layer inner dimension |
| N_cells | 256 | Number of cells (= D / D_neuron) |
| C_neurons | 124 | Neurons per cell |
| D_neuron | 8 | Per-neuron state dimension |
| K_connections | 16 | Cell-local connections per neuron |
| N_border_per_cell | 4 | Border neurons per cell (inter-cell) |
| K_border | 4 | Inter-cell connections per border neuron |
| R_rounds | 4 | Message-passing rounds per token |
| alpha | 4 | Inject/readout redundancy factor |
| state_mlp_hidden | 16 | Shared state MLP hidden dim |
| msg_mlp_hidden | 16 | Shared message MLP hidden dim |
| cell_mod_hidden | 24 | Per-neuron modulator hidden dim |
| T | 128 | Tokens per segment |
| N_total | 31,744 | Total neurons (256 × 124) |

---

## Cell Structure

Each cell contains 124 neurons with designated roles:
- **4 inject neurons** (indices 0..3): receive LM signal via additive injection
- **4 border neurons** (indices 4..7): have K_border=4 connections to border
  neurons in OTHER cells, enabling inter-cell information flow
- **4 readout neurons** (indices 120..123): messages read out to LM
- **112 interneurons** (indices 8..119): internal computation only

All neurons participate in R rounds of message passing. Every round:
1. All neurons do intra-cell gather (K=16 cell-local connections)
2. Border neurons additionally gather from their K_border=4 inter-cell connections
3. Inject neurons get the additive LM signal
4. All neurons update state and generate messages

Information flow: LM → inject neurons → interneurons → border neurons ↔
border neurons in other cells → interneurons → readout neurons → LM.

### Why Port Neurons?

Broadcasting the LM signal to all 256 neurons and summing all 256 for
readout causes 1/sqrt(256) = 1/16 signal dilution. With alpha=4 port
neurons, the scaling is 1/4 — much cleaner gradients.

Biologically: cortical layer 4 receives external input, layers 5/6 project
output. Not every neuron both receives and sends external signals.

---

## Connectivity

All connections are **cell-local**: neuron n in cell c can only connect to
other neurons in cell c. Connection indices stored as uint8 (0..255).

- K=16 connections per neuron, randomly initialized within the cell
- No self-connections
- Structural plasticity rewires within cells based on co-activation
- No inter-cell connections (for now — future work)

### Why Cell-Local?

1. **GPU efficiency**: Cell data (~10 KB) fits in L1 cache. Gather reads
   contiguous memory, not random global addresses.
2. **Scalability**: Adding cells scales linearly with no cross-cell coupling.
3. **Biological**: Cortical connectivity is predominantly local.

---

## Inter-Cell Connectivity (Border Neurons)

Each cell has B=4 **border neurons** that additionally have K_border=4
connections to border neurons in other cells. These connections are global
(any border neuron in any cell) and initialized randomly.

### Implementation

All border neurons across all cells form a global pool of NC×B = 1024
border neurons. Each border neuron's K_border connections index into this
pool (excluding its own cell's border neurons).

The border gather is a small global memory read per token:
- 4 border neurons × 4 connections × 8 dims = 128 floats per cell
- 256 cells × 128 floats × 2 bytes = 65 KB total per token
- At 1 TB/s bandwidth: <0.001 ms — negligible

### Processing Order (per round)

Border gather happens **simultaneously** with intra-cell gather. Each border
neuron's received signal is the sum of its intra-cell received (from K=16
cell-local neighbors) plus its inter-cell received (from K_border=4 remote
border neurons). Same state MLP, same msg MLP, same modulator — the border
neuron is just a regular neuron with extra connections reaching outside.

### Structural Plasticity

Border connections are rewirable via co-activation-based structural
plasticity, using the same mechanism as intra-cell connections. This lets
the network learn which cross-cell connections are useful and prune/regrow
them based on Hebbian co-activation.

### Biological Analogy

Layer 2/3 pyramidal neurons in cortex have mostly local synapses plus
sparse long-range axonal projections to other cortical areas. Same neuron
type, same integration rule, same plasticity — just some connections reach
further. Our border neurons follow this pattern exactly.

---

## Inject / Readout (Parameter-Free)

### Inject
```
H_mid [BS, T, 2048] → reshape [BS, T, 256, 8]
→ replicate alpha=4 times → [BS, T, 1024, 8]
→ each group of 4 inject neurons gets the same 8-dim LM slice
→ added to inject neurons' received signal (interneurons get zero)
```

### Readout
```
readout_neurons' messages [BS, T, 1024, 8]
→ reshape [BS, T, 256, 4, 8]
→ average over alpha=4 replicas → [BS, T, 256, 8]
→ reshape [BS, T, 2048]
→ H_enriched = H_mid + mem_scale * readout
```

---

## Shared-Weight MLPs

### State MLP (shared across all 65K neurons)
```
Input: cat(input_vec[D=8], primitives[D=8], neuron_id[D=8], decay[1]) = 25 dims
Hidden: 16 units, tanh activation
Output: D=8, tanh activation
Params: [16, 25] + [16] + [8, 16] + [8] = 552
```

The MLP input is decomposed:
- `input_vec` = received (from cell-local gather) + inject (for port neurons)
- `primitives` = per-neuron features set by cell modulator
- `neuron_id` = learned per-neuron identity embedding (constant)
- `decay` = per-neuron decay rate set by cell modulator

### Message MLP (shared)
```
Input: cat(h[D=8], primitives[D=8], neuron_id[D=8]) = 24 dims
Hidden: 16 units, tanh activation
Output: D=8, tanh activation + neuron_id added
Params: [16, 24] + [16] + [8, 16] + [8] = 536
```

---

## Per-Neuron Modulator

Each neuron has its **own modulator MLP weights**. This is the dominant
parameter block (~55M of 107M total) but runs only **once per segment**
(every 128 tokens) — so the cost is negligible vs the T×R inner loop.

Each neuron's modulator learns its own modulation strategy: how to adjust
its connections, decay, and primitives based on its recent activity. This
gives the memory graph substantial learning capacity concentrated in the
slow neuromodulation pathway.

### Input (per neuron, 41 dims)
- `hebbian[K=16]`: this neuron's Hebbian traces
- `h[D=8]`: this neuron's hidden state
- `decay[1]`: this neuron's current decay logit
- `primitives[D=8]`: this neuron's current primitives
- `neuron_id[D=8]`: this neuron's identity embedding

### Output (per neuron, 25 dims)
- `w_conn_delta[K=16]`: adjustment to this neuron's connection weights
- `decay_delta[1]`: adjustment to this neuron's decay logit
- `prim_delta[D=8]`: adjustment to this neuron's primitives

### Implementation
Per-neuron einsum: `'bni,nhi->bnh'` on `[BS, N_total, mod_in] × [N_total, H, mod_in]`.
Runs once per segment. At BS=32, N=31744: a single kernel launch, ~1-2ms.

### Params: [31744, 24, 41] + [31744, 24] + [31744, 24, 29] + [31744, 29] ≈ 55M

### Why per-neuron?
The modulator is the memory graph's primary learning capacity. With shared
state/msg MLPs (~1K params), all the graph's expressiveness comes from:
1. Per-neuron modulator weights (55M) — each neuron learns its own strategy
2. Per-neuron identity embeddings (254K) — conditions the shared MLPs
3. Learned connectivity — structural plasticity reshapes the graph

The modulator runs once per 128 tokens, so the 55M params cost almost nothing
in compute. The params are "slow weights" that configure the fast dynamics.

---

## Parameter Budget

```
LM:                                             ~51.6M
  Embedding (32K × 768):                        24.6M
  proj_up/down (768↔2048):                      3.1M
  pos_embed (128 × 2048):                       0.3M
  4 scan layers (d_inner=580, GLU):              19.0M
  PCM (16 columns):                              1.1M
  split_mlp:                                     3.6M
  mem_scale [2048]:                              ~0

Memory:                                          ~55.3M
  Per-neuron modulator (31744 neurons):          55.0M
  neuron_id [256, 124, 8]:                       254K
  Shared state MLP:                              552
  Shared msg MLP:                                536

Grand total:                                     ~107M
```

---

## Temporal Dynamics

Each neuron's state evolves via leaky integration:
```
h_t = sigmoid(decay_logit) × h_{t-1} + (1 - sigmoid(decay_logit)) × update_t
```

The decay rate is set per-neuron by the cell modulator at segment boundaries.
Fast-decaying neurons respond to recent input; slow-decaying neurons maintain
persistent memory across tokens.

With R=4 rounds per token and T=128 tokens per segment, each neuron performs
512 state updates per segment. Information propagates ~4 hops per token
through the cell's sparse graph.

---

## Gradient Flow

```
CE loss → logits → upper scan → H_enriched = H_mid + mem_scale × readout
  → readout (average readout neurons' messages)
  → msg_MLP(h, primitives, neuron_id)
  → h from temporal scan: h = decay × h + (1-decay) × update
  → state_MLP(received + inject, primitives, neuron_id, decay)
  → received from cell-local gather with sigmoid(w_conn)
  → w_conn, decay, primitives from shared modulator
  → modulator MLP(mod_w1, mod_w2) ← per-neuron (hebbian, h, decay, prim, neuron_id)
```

All memory parameters receive gradients. TBPTT detaches h and messages at
segment boundaries.

---

## Triton Kernel

One program per (batch, cell). Processes all T=128 tokens × R=4 rounds ×
C=256 neurons entirely in L1 cache.

- Grid: (BS, N_cells) = (32, 256) = 8,192 programs
- Per-program data: ~10 KB (cell state in bf16)
- Shared MLP weights: ~1 KB (fit in registers)
- No global memory access during the inner loop

Forward: works at full 65K neurons.
Backward: uses PyTorch reference recomputation (currently OOMs at 65K —
needs fused Triton backward or reduced N).

---

## Structural Plasticity

Within-cell co-activation-based rewiring:
- Track per-cell co-activation matrix: [N_cells, C, C] = [256, 256, 256]
- Binary firing threshold at 75th percentile
- Pearson phi coefficient for co-activation
- Prune bottom 2% of connections (lowest phi)
- Regrow toward highest-phi unconnected pairs (80% guided, 20% random)

**Status**: Framework in place, full implementation pending.

---

## Files

| File | Purpose |
|------|---------|
| `src/v11/config.py` | V11Config with cell parameters |
| `src/v11/memory_graph.py` | CellMemoryGraph (Python fallback + Triton dispatch) |
| `src/v11/model.py` | V11Model wrapping V8LM + CellMemoryGraph |
| `src/v11/triton_kernels.py` | Fused cell forward kernel + autograd wrapper |
| `src/v8/lm.py` | V8LM (scan layers, PCM, inject_memory) — reused unchanged |
| `src/v8/pcm.py` | BatchedPCM — reused unchanged |
| `src/model/scan.py` | ScanLayer, fused_scan — reused unchanged |

---

## Implementation Status (2026-04-03)

- [x] Config with all cell parameters
- [x] CellMemoryGraph with Python reference implementation
- [x] V11Model integration with V8LM
- [x] Triton forward kernel (65K neurons)
- [x] Autograd wrapper (Triton fwd + PyTorch recompute bwd)
- [x] 28 tests passing (shapes, gradients, TBPTT, integration, Triton correctness)
- [x] Docs updated for v11
- [ ] Backward at 65K neurons (OOMs — needs Triton backward or reduced N)
- [ ] Training infrastructure (train.py, trainer.py, diagnostics.py)
- [ ] Structural plasticity within-cell implementation
- [ ] Training run + comparison with LM-only baseline
