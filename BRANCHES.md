# Branch Guide — Neuromorphic Language Model

## Active Branches

### `cell-sequential` (current work)
**Design**: Cell-based memory graph with sequential per-token processing.
256 cells × 124 thin neurons (D=8). Cell-local connectivity (K=16) with
border neurons (K_border=4) for inter-cell communication. Dedicated inject
and readout port neurons. Per-neuron modulator with delta prediction.
Shared state/msg MLPs. Live message propagation — gather reads current
messages every token step, no frozen-message approximation.

**Status**: Functionally correct, 6K tok/s training at BS=8.
Bottlenecked by Python loop overhead (128 sequential steps × kernel launches).
Forward-only compute is 18K tok/s — autograd adds 3x overhead.

**Key files**: `src/v11/`, `tests/v11/`

---

### `cell-scan` (alternative)
**Design**: Same cell-based architecture as cell-sequential, but uses the
v9-style scan optimization: messages frozen per round, all T tokens' MLPs
batched into single F.linear calls, fused_scan for temporal integration.
R=2 refinement rounds (Newton-style fixed-point iteration on message field).

**Status**: 53K tok/s with shared I/O ports (shortcut that bypasses 97%
of neurons). 6.5K tok/s with separate inject/readout (honest full simulation).
The shared I/O shortcut was rejected because it defeats the purpose of
simulating all neurons.

**Tradeoff vs cell-sequential**: Faster (batches all T) but frozen messages
within each round means neurons can't react to each other per-token.

**Key files**: `src/v11/`, `tests/v11/`

---

### `shared-scan` (previous architecture)
**Design**: 512 wide neurons (D=256) with random sparse connectivity (K=32).
Shared-weight state/msg MLPs with base/inject decomposition. Per-neuron
modulator (H=112). fused_scan for temporal integration. 2-pass frozen-message
simulation.

**Status**: 24K tok/s at BS=32. Correctness-verified (f32 params fix,
autocast gradient fix). Never trained to completion — the memory graph
showed zero benefit over LM-only baseline in the v9 training run, which
motivated the v11 cell redesign.

**Key lesson**: Per-neuron weights (58M params) at D=256 can't be parallelized
as cuBLAS GEMM. Shared weights + scan was the solution for throughput. But
the memory graph didn't help the LM — open question whether this is a
capacity issue or a fundamental architecture problem.

**Key files**: `src/v8/`, `tests/v8/`

---

### `iterative-parallel` (fundamentally different)
**Design**: Cortical columns with iterative refinement (R=4 passes over
whole segment). No per-token sequential recurrence — all N tokens processed
simultaneously via parallel column operations. Memory via procedural memory
(PM, slot-based) and episodic memory (EM, cross-attention retrieval).

**Status**: Complete implementation, never trained against current baselines.
Much faster than sequential approaches because no per-token loop. But the
memory system is database-style (read/write slots), not dynamical-system-style
(persistent neuron graph). Fundamentally different design philosophy.

**Key lesson**: Parallel processing is fast on GPU. The per-token sequential
loop in cell-sequential is what GPUs are worst at. Worth revisiting if we
want to explore non-recurrent memory architectures.

**Key files**: `src/model/`, `legacy/`

---

### `main`
Default branch. Contains an older version of the codebase (pre-v11).
Not actively used for development.

---

## Abandoned Branches

These branches represent earlier attempts that were superseded or hit dead ends.
All code is preserved for reference.

### `abandoned/v9-backprop-correctness-fixes`
Just the f32 param + autocast gradient fixes, before scan optimization.
14K tok/s. Superseded by `shared-scan`.

### `abandoned/v9-backprop-2pass`
Earlier v9 variant with 2-pass simulation. Superseded by `shared-scan`.

### `abandoned/v9-backprop-neuron-redesign`
Experimental neuron architecture changes. Did not improve results.

### `abandoned/v9-es-backup`
Evolution Strategies for memory graph training (not backprop). ES sign
inversion bug caused it to fail. Abandoned in favor of backprop (v9).

### `abandoned/v9-es-neuron-mlps`
ES with per-neuron MLPs. Same ES approach, more expressive neurons.
Abandoned with ES.

### `abandoned/v9.1-triton-attempt`
D=16, N=8192 lightweight neurons. 0.3-2.1K tok/s — small D killed
Triton kernel performance. Led to the insight that D must be large
enough for efficient GPU computation.

### `abandoned/v10`
Spacetime grid brainstorm. Design exploration only, never implemented.

### `abandoned/v10-design-backup`
Design backup for v10. Never became a working implementation.

### `abandoned/v10-gnn`
GNN-based memory with per-neuron weights + upper scan. 8-16K tok/s.
Abandoned because per-token gather cost (128 gathers/segment) dominated
and the memory graph showed no benefit in preliminary tests.

### `abandoned/v10-shared-mlps`
Shared-weight variant of v10. Explored but superseded by v11 cell design.

### `abandoned/v8-rl-neuromod`
Original RL neuromodulator (GRPO). Failed due to credit assignment noise,
bf16 gradient rounding, shared neuromod can't differentiate neurons.
Led to v9 (backprop replaces RL).

### `abandoned/v8-broadcast-io`
Fix for dead internal neurons via broadcast inject/readout. Discovered
1/N readout kills gradients (use 1/sqrt(N) or average). Led to the
dedicated port neuron design in v11.

### `abandoned/v6-main-backup`
v6 baseline. Neuromorphic model lost to GPT-2 and Pythia baselines on NTP.

### `abandoned/v4-iterative-backup`
Same as `iterative-parallel` (kept as active branch with better name).

### `abandoned/v1-legacy`
Original prototype. Historical only.

### `abandoned/phase1-backprop-experiment`
Early backprop experiment. Superseded by v9.

---

## Key Lessons Across All Branches

1. **Per-neuron weights don't scale on GPU** — N separate matmuls can't match
   one cuBLAS GEMM. Shared weights + conditioning (neuron_id) is the solution.

2. **D=8 is too thin for GPU efficiency** — each F.linear row has only 8
   multiply-adds, making GEMMs memory-bandwidth-limited not compute-limited.

3. **Sequential Python loops are 3x slower than the actual compute** — 128
   steps × 8 kernel launches = 1024 launches at ~10μs each = 10ms overhead
   on 3ms of actual work.

4. **Scan (frozen messages) enables batching over T** but loses per-token
   inter-neuron communication. R>1 refinement rounds partially recover this.

5. **The memory graph has never been proven to help the LM** — v9 training
   showed 110M full model matched 52M LM-only baseline. The core hypothesis
   (persistent neuron graph improves language modeling) remains unvalidated.

6. **Hardware mismatch** — the cell-based design is optimized for spatial
   locality and sequential per-cell processing, which is native to FPGA/
   neuromorphic chips but adversarial to GPUs.
