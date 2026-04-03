# V11 Sequential Optimization Benchmark Report (2026-04-03)

## Baseline
- R=1, share_io_ports=False, separate inject/readout
- All 124 neurons per cell simulated with full temporal dynamics
- Live message propagation (gather every token step)
- 31,744 total neurons, 106.9M params

## Results

| Approach | BS | tok/s | VRAM | Notes |
|----------|---:|------:|-----:|-------|
| **0-Baseline** | 8 | 3,056 | 7.5 GB | No checkpointing, OOMs at BS=32 |
| **0-Baseline** | 16 | 2,840 | 14.3 GB | |
| **1-Checkpointed** | 8 | 2,384 | 2.3 GB | Trades speed for VRAM |
| **1-Checkpointed** | 32 | 2,241 | 7.2 GB | |
| **1-Checkpointed** | 96 | 2,034 | 20.3 GB | Max BS, still 2K tok/s |
| **2-Fewer neurons C=64** | 8 | 5,178 | 4.9 GB | 16K neurons, 80M params |
| **2-Fewer neurons C=64** | **16** | **5,927** | **8.6 GB** | **BEST throughput** |
| **2-Fewer neurons C=64** | 48 | 5,442 | 23.7 GB | |
| **3-Wider D=16 NC=128** | 8 | 4,608 | 6.4 GB | 15.8K neurons, 92M params |
| **3-Wider D=16 NC=128** | 16 | 4,435 | 11.7 GB | |
| **4-Shorter T=64** | 8 | 2,998 | 4.7 GB | Half the tokens per segment |
| **4-Shorter T=64** | 32 | 2,641 | 15.1 GB | |

## Analysis

### What worked:

**Fewer neurons (C=64) is the clear winner: 5.9K tok/s.** Halving C from 124 to 64
cuts the per-step compute and memory in half. The gather is smaller ([BS,NC,64,K,D]
vs [BS,NC,124,K,D]), the MLPs process fewer neurons, and autograd saves smaller
intermediates. Larger BS becomes possible (fits BS=48 in 24GB).

**Wider neurons (D=16, NC=128) also helps: 4.6K tok/s.** Fewer cells (128 vs 256)
means fewer gather operations, and the F.linear calls have larger last-dimension
(16 vs 8) which cuBLAS handles more efficiently.

### What didn't help:

**Gradient checkpointing** saves VRAM (2.3 GB at BS=8!) but the recomputation cost
means throughput drops to ~2K tok/s regardless of BS. Even at BS=96 it's 2K — the
checkpoint overhead dominates.

**Shorter segments (T=64)** halves the loop iterations but also halves the tokens
per step, so tok/s stays the same (~2.6-3K). Not a real improvement.

### What we didn't try yet:

- **torch.compile** — timed out in initial test, needs careful setup
- **CUDA graphs** — could eliminate Python loop overhead entirely
- **Custom Triton per-step kernel** — fuse gather+MLP+decay into one launch
- **Reducing both C and N_cells** — e.g., C=64, D=16, NC=128 = 8K neurons
- **Larger H (MLP capacity)** — might let fewer neurons do more work
- **Mixed approach** — checkpointing + fewer neurons for BS=64+

## Conclusion

The sequential loop's throughput is fundamentally limited by **128 Python iterations
with ~8 CUDA kernel launches each = 1024 launches**. At ~10μs per launch, that's
~10ms of pure overhead. The per-step compute at D=8 is tiny (<0.1ms), so we're
**90% overhead, 10% compute**.

The path to 20K tok/s requires eliminating the Python loop:
- torch.compile or CUDA graphs to fuse/replay the loop
- OR a Triton kernel that handles gather+MLP in one launch per step
- OR accepting the scan-based approach (frozen messages) with R=2

Reducing C to 64 neurons/cell is the best immediate win: 5.9K tok/s with honest
full simulation of all neurons, live message propagation, separate inject/readout.
