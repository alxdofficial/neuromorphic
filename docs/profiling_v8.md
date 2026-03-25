# V8 Training Profiling Results

All measurements on RTX 4090 (24GB), T=2048, torch.compile enabled on LM methods,
gradient checkpointing OFF. 10 warmup + 10 measured steps in steady state.

> **Note**: These profiling numbers predate the switch from fixed L1-normalized
> conn_weights to key-based softmax routing. The Triton kernel structure is
> unchanged (it still uses precomputed scalar routing weights per neighbor),
> so per-token kernel cost is similar. The main difference is a small per-segment
> overhead for computing softmax(key . neighbor_messages) routing weights before
> the token loop begins.

## Tier Scaling Summary

| Tier | Params | D | L | N_neurons | K | D_mem | BS | w/ memory | w/o memory | Overhead | VRAM |
|------|--------|---|---|-----------|---|-------|-----|-----------|------------|----------|------|
| A | 103M | 2048 | 7 | 1024 | 96 | 128 | 12 | 57.8K tok/s | 154.2K tok/s | 167% | 13.9GB |
| B | 285M | 3072 | 12 | 2048 | 96 | 128 | 8 | 25.9K tok/s | 58.5K tok/s | 126% | 16.3GB |
| C | 1.01B | 4096 | 28 | 4096 | 128 | 256 | 2 | 7.2K tok/s | 15.2K tok/s | 110% | 16.0GB |

### Observations

- Memory graph overhead ranges from 110-167% across tiers
- At tier C (1B), the memory graph roughly doubles training time
- LM-only throughput scales as expected: 154K → 58K → 15K as params increase
- Tier C fits on a single 4090 but only at BS=2 (needs multi-GPU for practical training)
- D_mem=256 at tier C (vs 128 at A/B) makes the per-token kernel 4× more expensive per neuron

### Memory Graph Cost Scaling

The per-token sparse gather cost is `O(N × K × D_mem)`:

| Tier | N × K × D_mem | Relative to A |
|------|---------------|---------------|
| A | 1024 × 96 × 128 = 12.6M | 1.0× |
| B | 2048 × 96 × 128 = 25.2M | 2.0× |
| C | 4096 × 128 × 256 = 134.2M | 10.7× |

Tier C's memory graph is 10.7× more expensive per token than tier A, mainly due to
the wider D_mem (256 vs 128) and more neurons (4096 vs 1024).

## Tier A Detailed Profiling

### Step Timing (BS=8, compiled)

| Step Type | Time | Throughput |
|-----------|------|-----------|
| Collect step (no RL update) | 307ms | 53.4K tok/s |
| Update step (with RL replay) | 384ms | 42.6K tok/s |
| Average (alternating) | 346ms | 47.4K tok/s |

RL updates happen every 4 chunks (`rl_collect_chunks=4`). Collect steps are ~20% faster.

> **Note**: The counterfactual baseline (reverting K=96 neurons and re-running the memory
> graph) adds approximately 2x the memory graph cost per segment during Phase 2. This
> brings Phase 2 throughput to ~27K tok/s (vs ~68K tok/s in Phase 1 without neuromod).

### CUDA Time Breakdown (compiled, 10 steps)

| Component | CUDA Time | % of Total | Per Step |
|-----------|-----------|------------|----------|
| Memory graph Triton kernel | 1,304ms | 46.1% | 130ms |
| aten::mm (compiled matmuls) | 748ms | 26.4% | 75ms |
| cutlass GEMM (backward) | 255ms | 9.0% | 26ms |
| CompiledFunction (fused fwd+bwd) | 1,043ms | 36.9% | 104ms |
| aten::addmm (linear layers) | 149ms | 5.3% | 15ms |
| Total CUDA | 2,829ms | | 283ms |

With torch.compile enabled, the memory graph kernel is the dominant cost at **46%**.
Before compile, it was 38% (LM was uncompiled and slower).

### Memory Graph Internal

The 130ms memory graph time per step (8 segments × 256 tokens):
- **Triton kernel compute**: ~126ms (sparse gather + integrate + tanh, fused with norm)
- **Post-segment stats**: ~4ms (firing threshold, co-activation phi on plasticity segments)

Each kernel launch: grid (BS, N) programs. Per-token: ~63μs.

## Batch Size Scaling (Tier A, compiled)

| BS | ms/step | K tok/s | VRAM |
|----|---------|---------|------|
| 8 | 298ms | 54.9K | 9.5GB |
| 12 | 425ms | 57.8K | 13.9GB |

Throughput barely improves from BS=8 to BS=12 (+5%). The GPU is compute-bound
at BS=8 — both the memory graph kernel and LM matmuls saturate the SMs.

## torch.compile Impact

Compiling individual LM methods (forward_scan_lower, forward_scan_upper,
forward_output) and neuromod methods (get_action):

| Config | Before (broken compile) | After (method compile) | Improvement |
|--------|------------------------|----------------------|-------------|
| BS=8 | 43.3K tok/s | 54.9K tok/s | **+27%** |
| VRAM | 7.7GB | 9.5GB | +1.8GB |

The prior `torch.compile(model.lm)` was completely inert because V8LM has no
`forward()` method — all calls go through named methods. Compiling the individual
methods fixes this.

## Implemented Optimizations

| Optimization | Savings | Status |
|-------------|---------|--------|
| torch.compile on individual LM/neuromod methods | 27% total throughput | Done |
| Fuse .norm() into Triton kernel | ~14ms/chunk | Done |
| Skip phi bmm on non-plasticity segments | ~5ms/chunk | Done |
| Sort conn_indices for cache locality | Minor | Done |
| Skip prepare_sparse_weights when mask all-active | ~1ms/chunk | Done |
| Eliminate GPU-CPU sync points (reset_mask, ppl, etc.) | Reduced pipeline stalls | Done |
| Remove duplicate CE computation | ~0.5ms/step | Done |
| Fix benchmark grad_ckpt polarity (was inverted) | Accurate benchmarks | Done |

## Remaining Optimization Opportunities

| Optimization | Estimated Savings | Complexity | Design Change? |
|-------------|-------------------|------------|---------------|
| CUDA graph capture of 256-step loop | ~8% total | Medium | No |
| K-step message passing (K=4) | ~35% total | Medium | Yes — slower propagation |
| Decouple D_mem from D_cc (projection) | Reduces kernel cost at scale | Medium | Yes — adds projection layers |
| Multi-GPU DDP | Linear scaling with GPU count | Medium | No |

## Key Insights

1. The memory graph sequential loop (46% of CUDA time at tier A) is the single
   largest cost after compile optimization.

2. Memory overhead increases with model scale: 167% at tier A, but the ratio
   improves at tier C (110%) because the LM itself becomes the dominant cost
   at larger scales.

3. The practical scaling path is multi-GPU DDP — memory graph state is per-rank
   (no cross-GPU communication needed), just all-reduce LM gradients.

4. D_mem scaling is the key driver of memory graph cost. Tier C's D_mem=256
   makes the kernel 10× more expensive per token than tier A's D_mem=128.
   Decoupling D_mem from D_cc via a projection layer would allow the LM to
   scale width while keeping memory graph cost constant.
