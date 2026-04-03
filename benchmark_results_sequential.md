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

## Additional Experiments (Approach 5-7)

### Approach 5: torch.compile on step function (forward-only)
- C=64, BS=16, `torch.compile(step, mode='max-autotune')` on MLP step
- Forward-only with compile: 12.2K tok/s (earlier test, full step compiled)
- Forward-only with compile in loop: 797 tok/s (WORSE — recompilation overhead)
- **torch.compile doesn't help in a Python loop** — it needs the full graph

### Approach 6: Raw forward-only (no autograd)
- C=64, BS=16, `torch.no_grad()`, inline loop
- **18.1K tok/s, 0.88ms per step** ← THIS IS THE COMPUTE SPEED
- Proves the GPU can handle the workload. Autograd is the bottleneck.

### Key Finding: Autograd Is the Bottleneck

| Mode | tok/s | Per-step | Ratio |
|------|------:|---------|-------|
| Forward only (no autograd) | 18,130 | 0.88ms | 1.0x |
| Forward with autograd | ~6,000 | ~2.7ms | 3.0x |
| Forward + backward | ~5,900 | ~2.7ms | 3.1x |
| Forward + backward + checkpoint | ~2,400 | ~6.7ms | 7.6x |

The GPU compute (gather + 4 F.linear + decay) takes 0.88ms per step.
PyTorch autograd tracking adds 2x overhead (building the graph).
Gradient checkpointing adds another 2.5x (recomputing the graph).

**To reach 20K tok/s training, we need to reduce autograd overhead.**
Options:
- Custom autograd.Function with manual backward for the T-step loop
- Compile the ENTIRE forward_segment (not per-step) so the compiler
  can optimize across the loop
- Accept 6K tok/s with C=64 and train at that speed

### Approach 7: Truncated BPTT within segment (C=64, BS=16)
- Detach h/msg every bptt_len steps, backprop only through last chunk
- bptt=8: 5.0K tok/s, 15.5 GB
- bptt=16: 4.8K tok/s, 15.5 GB
- bptt=128: 4.7K tok/s, 15.5 GB
- Minimal improvement — autograd per-step overhead dominates, not graph depth

### Approach 8: C=64 + checkpoint + large BS
- BS=16: 3.6K tok/s, 2.7 GB
- BS=32: 3.4K tok/s, 4.9 GB
- BS=96: 3.2K tok/s, 13.6 GB
- Ultra-low VRAM enables huge batches, but checkpoint overhead caps at ~3.5K

## Final Summary Table

| Approach | Best tok/s | BS | VRAM | Notes |
|----------|----------:|---:|-----:|-------|
| Baseline C=124 | 3,056 | 8 | 7.5 GB | |
| Checkpointed C=124 | 2,384 | 8 | 2.3 GB | VRAM-efficient |
| **C=64 no checkpoint** | **5,927** | **16** | **8.6 GB** | **Best throughput** |
| C=64 checkpointed | 3,642 | 16 | 2.7 GB | VRAM-efficient |
| C=64 + TBPTT(8) | 4,975 | 16 | 15.5 GB | |
| D=16 NC=128 | 4,608 | 8 | 6.4 GB | |
| T=64 | 2,998 | 8 | 4.7 GB | |
| **Forward-only (no autograd)** | **18,130** | **16** | **~3 GB** | **Compute ceiling** |
| torch.compile step | 797 | 16 | - | Worse (recompile overhead) |

## Conclusion

**Best honest training throughput: 5.9K tok/s** (C=64, BS=16, no tricks).

The compute ceiling is 18K tok/s (forward-only). Autograd adds 3x overhead.
No approach we tested breaks through this — torch.compile, checkpointing,
TBPTT, and larger BS all hit the same per-step autograd overhead wall.

**To reach 20K tok/s, the forward+backward needs a fundamentally different
approach**: either a custom autograd.Function that runs the T-loop in C++ 
without per-step Python overhead, or hardware that doesn't have kernel 
launch overhead (FPGA/neuromorphic chip).

**Practical recommendation**: Train at 5.9K tok/s with C=64, BS=16.
A 30K step run = 30K × 2048 / 5900 ≈ 2.9 hours. Acceptable.
