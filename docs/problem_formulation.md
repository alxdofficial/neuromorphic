# Problem Formulation: Cell-Based Neural Memory Graph for Language Modeling

## The Problem

Can a persistent neuron graph, augmented onto a language model, learn to store
and retrieve information that improves next-token prediction — beyond what the
LM's fixed weights and recurrent state can capture alone?

## The Approach (v11)

Augment a split-scan linear recurrence LM with a **cell-based memory graph**:

```
Input tokens → Embedding → Lower Scan (2 layers)
  → PCM (surprise = transition prediction error)
  → Memory Graph (cell-based, 65K neurons, per-token interaction)
  → Inject memory output into H_mid
  → Split-point MLP (H_mid + surprise)
  → Upper Scan (2 layers) → Output head → Logits
```

The memory graph receives a detached copy of H_mid and returns a readout
signal that is added to H_mid via a learnable per-dimension scale (mem_scale).
The upper scan sees both the LM's own representation and the memory's
contribution.

## Key Design Decisions

### 1. Cell-Based Architecture

65,536 neurons organized into 256 cells of 256 neurons each.
- D_neuron = 8 (thin neurons — expressiveness from ensemble, not width)
- K = 16 cell-local connections per neuron (no global scatter/gather)
- R = 4 message-passing rounds per token step

**Why cells?** GPU efficiency: each cell fits in L1 cache. All memory access
is local. Biologically motivated: cortical columns have local connectivity.

### 2. Dedicated Port Neurons

4 inject neurons and 4 readout neurons per cell (alpha=4). The remaining
248 neurons are interneurons. This avoids the 1/sqrt(256) signal dilution
that killed gradient flow in earlier versions with broadcast inject/readout.

### 3. Shared-Weight MLPs + Per-Neuron Identity

State and message MLPs use shared weights conditioned on neuron_id.
This enables cuBLAS GEMM for the hot path while neurons still produce
different outputs (different connectivity, different identity, different
modulator-set parameters).

### 4. Per-Cell Modulator

Each cell has its own modulator MLP that runs once per segment, reading
aggregated cell statistics (mean h, mean hebbian trace, etc.) and outputting
connection weight adjustments, decay rates, and primitives broadcast to all
neurons in the cell.

### 5. Structural Plasticity

Within-cell co-activation-based rewiring. Neurons that fire together get
connected; weak connections are pruned and replaced with random or
co-activation-guided new connections. Constrained within cells to keep
the locality benefit.

### 6. Temporal Integration

Each neuron maintains persistent state h that evolves via leaky integration:
h_t = decay * h_{t-1} + (1-decay) * update. Decay is set by the cell
modulator, giving the modulator control over how quickly neurons forget.

## Open Questions

1. **Does the memory graph help?** The v9 training run showed the memory
   graph (58M params) provided zero benefit over the LM-only baseline.
   With v11's lightweight design (~1M params), the question is whether
   65K thin neurons can learn something the LM can't.

2. **Backward at scale.** The Triton forward kernel works at 65K neurons,
   but the backward (PyTorch reference recompute) OOMs due to materializing
   [BS, 256, 256, 16, 8] gather intermediates. Needs a fused Triton backward
   or reduced N.

3. **Optimal R (rounds per token).** R=4 allows full-cell information mixing
   but 4x the compute. R=2 might suffice. R=1 limits information to direct
   neighbors only.

4. **Inter-cell communication.** Currently cells are isolated. Adding sparse
   long-range connections between cells would enable cross-dimension-slice
   information flow, but complicates the local-only compute pattern.

5. **Scaling to millions of neurons.** The cell architecture is designed to
   scale. With D=8, a cell of 256 neurons is 2 KB. At 1M neurons (4K cells),
   the Triton kernel grid is (BS, 4096) — well within GPU limits.
