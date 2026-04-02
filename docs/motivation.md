# Motivation & Design Philosophy — Neuromorphic Language Model

## Why This Exists

Modern LLMs are powerful but structurally limited:

1. **LLMs are static** — once deployed, weights don't change. Improving requires
   finetuning, retraining, or bolting on retrieval (RAG). This leaves a gap between
   short-term learning, long-term personalization, and robust agent memory.

2. **RAG is not learning** — retrieval can fetch text but doesn't integrate knowledge
   into the model's computations, adapt behavior continuously, or compress and
   generalize experience.

3. **Transformers are expensive for long context** — O(N²) attention makes lifelong
   learning impractical. If we want AI embedded everywhere and eventually in silicon,
   we need O(1) memory that scales as state, not as quadratic attention.

4. **Agents need to improve by doing** — a practical agent should remember what it has
   done, refine tool-use patterns, adapt to workflows, and get better over time without
   cloud retraining.

## What We're Building

A **brain-inspired sequence model** with two components:

- **Language Model (LM)**: Split-scan linear recurrence (4 layers, split at 2).
  Lower scan produces H_mid. Predictive coding module (PCM) computes surprise
  (transition prediction error). Upper scan processes memory-enriched + surprise-
  modulated input. Trained by backprop. ~52M params.

- **Cell-Based Memory Graph** (v11): A flat world of 256 cells, each containing
  256 thin neurons (D=8). Neurons connect only within their cell (K=16 connections).
  R=4 message-passing rounds per token step. Dedicated inject/readout port neurons
  interface with the LM. Per-cell modulator adjusts connection weights, decay rates,
  and primitives once per segment. Structural plasticity rewires connections within
  cells based on co-activation. Trained end-to-end by backprop through the LM loss.
  ~1.1M params. 65,536 total neurons.

---

## Core Philosophy

### Memory as Persistent Signal Flow

The human brain doesn't do "memory read" and "memory write" operations. There is no
cosine similarity lookup. Instead:

- **Neurons constantly fire and receive signals.** Memory IS the pattern of activity
  that emerges from connection weights. Recall is pattern completion — a partial input
  activates connected neurons, which activate others, until the full pattern emerges.

- **Plasticity is continuous and local.** When two neurons co-activate, their connection
  strengthens (Hebbian learning). Neuromodulators gate HOW MUCH connections change.

- Our model captures this: memory is the evolving state of 65K neurons exchanging
  messages through learned sparse connections. No database, no keys/values, no
  attention. Just signal flow.

### Cell-Based Spatial Organization

The brain has spatial structure: cortical columns, topographic maps, local lateral
connections with sparse long-range tracts. Our cell-based design mirrors this:

- **Cells** = cortical minicolumns (~256 neurons each)
- **Cell-local connections** = lateral connections within a column
- **Inject neurons** = layer 4 (receives thalamic/external input)
- **Readout neurons** = layer 5/6 (projects output)
- **Interneurons** = layers 2/3 (local computation)

This spatial constraint also gives a massive throughput advantage: cell data fits
in GPU L1 cache, eliminating random global memory access.

### Thin Neurons, Rich Ensembles

Each neuron has only 8 dimensions — far less than the 256-dim neurons of v9.
But 256 neurons per cell collectively represent the same information through
their ensemble activity. Multiple message-passing rounds (R=4) let information
propagate across the cell each token step.

This is more biologically faithful: real neurons have ~1-dimensional output
(firing rate). Our D=8 is already 8x richer than biology. The expressiveness
comes from the NUMBER of neurons and their CONNECTION patterns, not from
per-neuron dimensionality.

### Shared Computation, Individual Identity

All neurons share the same state and message MLPs (shared weights). Neurons
produce different outputs because they receive different inputs (from different
neighbors), have different identity embeddings (neuron_id), and are configured
differently by their cell's modulator (decay, primitives, connection weights).

This is like how biological neurons share the same genetic program but develop
unique response properties through their connectivity and developmental history.

---

## What Makes This Fundamentally Different

### vs Transformers
- Fixed memory footprint vs O(N) KV cache growth
- Memory adapts at inference vs frozen weights
- O(N) compute vs O(N^2) attention

### vs SSMs (Mamba, RWKV)
- Structured persistent memory vs flat hidden state
- Spatial graph topology vs single state vector
- Per-neuron, per-token dynamics vs one recurrence per layer

### vs RAG
- Memory inside the model, not external
- Continuous automatic updates, not manually triggered
- Compressed/generalized knowledge, not verbatim storage

### vs Differentiable Memory (NTM, DNC)
- Memory as signal flow, not database read/write
- 65K neurons with spatial organization, not M memory slots
- Cell-local message passing, not global attention over memory

### vs Standard GNNs
- Persistent state across tokens and documents
- Structural plasticity that rewires the graph
- 128 steps of dynamics per segment, not 1-3 message-passing rounds

---

## Hardware Alignment

The cell-based design is well-suited to multiple hardware targets:

- **GPU**: Cell data fits in L1 cache. One Triton program per (batch, cell).
  Stencil computation pattern — highly optimized in scientific computing.

- **Neuromorphic chips (Intel Loihi)**: Cell-local connectivity maps to
  on-chip routing. Per-neuron dynamics map to physical neuron circuits.
  Hebbian/STDP plasticity is native to the hardware.

- **Analog computing**: Per-neuron weights stored as conductances in crossbar
  arrays. Cell-local connections = simple local routing. The sequential
  dynamics that are slow on GPU are native to analog continuous-time circuits.
