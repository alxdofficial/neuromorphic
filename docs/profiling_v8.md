# V8 Training Profiling Results

Profiled on RTX 4090 (24GB), BS=8, T=2048, no compile, gradient checkpointing enabled.
20 warmup steps, 10 measured steps in steady state. Date: 2026-03-23.

## Step Timing

| Step Type | Time | Throughput |
|-----------|------|-----------|
| Collect step (no RL update) | 307ms | 53.4K tok/s |
| Update step (with RL replay) | 384ms | 42.6K tok/s |
| Average (alternating) | 346ms | 47.4K tok/s |

RL updates happen every 2 chunks (`rl_collect_chunks=2`). Collect steps are ~20% faster
because they skip the neuromod replay + optimizer step.

## CUDA Time Breakdown (per step average, 10 steps)

| Component | CUDA Time | % of Total | Notes |
|-----------|-----------|------------|-------|
| Memory graph Triton kernel | 130ms | 38.4% | 2048 kernel launches/step (256/seg × 8 segs) |
| LM forward matmuls (mm + addmm) | 89ms | 26.4% | Scan layers, projections, output head |
| LM backward (cutlass GEMM) | 69ms | 20.2% | Gradient computation for all LM params |
| Element-wise (mul + add) | 40ms | 11.9% | Gates, activations, residuals, optimizer |
| Dtype copies (copy_) | 17ms | 4.9% | bf16 ↔ f32 conversions |

Total CUDA time per step: ~338ms (the rest is CPU overhead, kernel launch latency).

### Memory Graph Internal

The 130ms memory graph time breaks down as:
- **Triton kernel compute**: ~126ms (the sparse gather + integrate + tanh loop)
- **Post-segment stats**: ~4ms (firing threshold, co-activation phi on plasticity segments)

Each Triton kernel launch processes grid (BS=8, N=1024) = 8192 programs.
Per-token kernel time: ~63μs. Per-segment (256 tokens): ~16ms.

## Batch Size Scaling

| BS | ms/step | K tok/s | VRAM | Scaling efficiency |
|----|---------|---------|------|--------------------|
| 4 | ~220ms | ~37K | ~5GB | (baseline) |
| 8 | 346ms | 47.4K | 7.7GB | 1.28x throughput for 2x batch |
| 12 | 562ms | 43.7K | 11.2GB | 1.18x throughput for 3x batch |

Throughput does NOT scale linearly with batch size. At BS=8, the GPU is already
compute-bound on both the memory graph kernel and LM matmuls. Increasing batch size
adds proportional work with no idle GPU cycles to absorb it.

## Why Throughput Doesn't Scale with Batch Size

Two factors:

1. **Memory graph**: The Triton kernel grid is (BS, N). At BS=8, that's 8192 programs.
   Each program does a K=96 sparse gather over D=128 dimensions — significant register
   usage. The GPU's SMs are already occupied. BS=12 adds 50% more programs that queue
   behind the existing ones.

2. **LM matmuls**: The scan layers and projections are large matmuls that already
   saturate tensor cores at BS=8. Larger batch means larger matmuls, but throughput
   is already near peak for the given shapes.

## Optimization Opportunities (implemented)

| Optimization | Savings | Status |
|-------------|---------|--------|
| Fuse .norm() into Triton kernel | ~14ms/chunk | Done |
| Skip phi bmm on non-plasticity segments | ~5ms/chunk | Done |
| Sort conn_indices for cache locality | Minor | Done |
| Skip prepare_sparse_weights_kernel (all-active mask) | ~1ms/chunk | Done |

## Remaining Optimization Opportunities

| Optimization | Estimated Savings | Complexity | Design Change? |
|-------------|-------------------|------------|---------------|
| CUDA graph capture of 256-step loop | ~10-20% on kernel time | Medium | No |
| K-step analytical scan (K=4) | ~75% fewer kernel launches | High | Yes — reduces propagation resolution |
| Decouple D_mem from D_cc (projection) | Reduces kernel register pressure | Medium | Yes — adds projection layers |
| Multi-GPU DDP | Linear scaling with GPU count | Medium | No |

## Key Insight

The memory graph (38%) and LM (46%) are comparable costs. The sequential kernel loop
is a significant contributor but not an overwhelming bottleneck. Both components would
need to speed up for batch size scaling to improve. The most impactful single change
would be multi-GPU data parallelism (linear throughput scaling without code changes
to the model).
