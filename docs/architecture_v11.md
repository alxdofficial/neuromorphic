# v11 Architecture — Cell-Based Neuromorphic Memory Graph

**Branch**: `v11-cells`
**Updated**: 2026-04-03

---

## Overview

A flat world of 256 cells, each containing 256 thin neurons (D=8). Neurons
connect only within their cell. Multiple message-passing rounds per token step
compensate for thin per-neuron dimensionality. Dedicated inject/readout port
neurons interface with the language model without signal dilution.

**65,536 total neurons. ~526K memory params. ~52M total (LM dominates).**

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
| C_neurons | 256 | Neurons per cell |
| D_neuron | 8 | Per-neuron state dimension |
| K_connections | 16 | Cell-local connections per neuron |
| R_rounds | 4 | Message-passing rounds per token |
| alpha | 4 | Inject/readout redundancy factor |
| state_mlp_hidden | 16 | Shared state MLP hidden dim |
| msg_mlp_hidden | 16 | Shared message MLP hidden dim |
| cell_mod_hidden | 32 | Shared modulator hidden dim |
| T | 128 | Tokens per segment |
| N_total | 65,536 | Total neurons (256 × 256) |

---

## Cell Structure

Each cell contains 256 neurons:
- **4 inject neurons** (indices 0..3): receive LM signal via additive injection
- **4 readout neurons** (indices 252..255): messages read out to LM
- **248 interneurons** (indices 4..251): internal computation only

All neurons participate in R rounds of message passing. Inject neurons get
an extra additive signal from the LM. Only readout neurons' messages are
read out. Information flows from inject → interneurons → readout through
the cell's sparse connectivity.

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

## Shared Modulator (Per-Neuron Input, Shared Weights)

A shared-weight modulator MLP that runs **once per segment**. Unlike the
state/msg MLPs which run every round of every token, the modulator runs
once per 128 tokens — so even at 65K neurons the cost is negligible.

Each neuron provides its own state as input and receives a personalized
adjustment. The modulator learns general rules like "if this neuron's
Hebbian traces are strong on connection k, strengthen it" — applied with
per-neuron specificity via shared weights.

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
One `F.linear` call on `[BS × N_total, 41] × [41, 32]` — a single cuBLAS
GEMM. At BS=32, N_total=65536: `[2M, 41]` input matrix. Takes <1ms.

### Params: [32, 41] + [32] + [25, 32] + [25] ≈ 2.1K

### Why shared, not per-cell or per-neuron?
- **Per-neuron weights** (v9 approach): 65K × 32 × 41 = 85M params. Dominates
  the entire model. The throughput bottleneck we spent the session fixing.
- **Per-cell weights** (previous v11 design): 256 × 32 × 41 = 335K params.
  Gives cells personality but neurons in the same cell get identical adjustments
  — misses the point of per-neuron modulation.
- **Shared weights** (current): 2.1K params. Each neuron gets a unique output
  because its INPUT (hebbian, h, decay, prim, neuron_id) is unique. The shared
  MLP learns modulation RULES, not per-neuron lookup tables.

Biologically: neuromodulators act through shared receptor types (the shared
weights) but each neuron responds differently based on its own state (the
per-neuron input). Dopamine doesn't carry neuron-specific instructions — it
activates D1/D2 receptors identically, but the downstream effect depends on
each neuron's current activity and receptor distribution.

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

Memory:                                          ~526K
  Shared state MLP:                              552
  Shared msg MLP:                                536
  Shared modulator:                              2.1K
  neuron_id [65536, 8]:                          524K

Grand total:                                     ~52.1M
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
