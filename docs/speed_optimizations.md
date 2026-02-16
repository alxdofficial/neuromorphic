# Speed Optimizations

**Date:** 2026-02-16
**Hardware:** NVIDIA RTX 4090 (24GB VRAM)
**Model:** Tier A Wide (D=768, L=8, B=2, ~85M params)
**Phase B config:** BS=32, T=256, P=64 (4 spans/chunk)

---

## Current Performance

| Metric | Value |
|--------|-------|
| Training throughput | ~25K tok/s |
| Tokens per step | 8,192 (BS=32 x T=256) |
| Step time | ~0.33s |
| Peak VRAM | 10.3 GB |
| VRAM allocated (steady state) | 1.6 GB |
| VRAM reserved (PyTorch cache) | 10.7 GB |
| Compilation warmup | ~5.7 min (first 2 steps) |

### Baseline Comparison

| Model | Params | tok/s | Relative |
|-------|--------|-------|----------|
| Pythia-160M (transformer) | 134M | ~102K | 4.1x faster |
| Mamba-130M (SSM) | 115M | ~58K | 2.3x faster |
| **Neuromorphic LM** | **85M** | **~25K** | **1x** |

The speed gap is expected: baselines process T=256 tokens in one parallel pass, while our model splits into 4 sequential spans with PM/EM boundary operations. Three memory systems (PM, EM, WM) add per-token overhead that baselines lack.

---

## Optimizations Applied

### Phase 1: Sync Removal + Compilation (commits d7cfd6c, 8431ccb)

Removed GPU-CPU synchronization barriers that stalled the pipeline:

| Fix | Impact |
|-----|--------|
| Gate `_memory_budget_utils` behind log interval | Eliminated 18 `.item()` syncs per step |
| Remove `.item()` from loss functions | Keep loss as tensor until metrics dict |
| Remove `.any()` sync in model reset check | Unconditional reset (masked no-op when empty) |
| Fuse `gate_a + gate_b` into single GEMM | One kernel instead of two per layer |
| Compile PM `apply_batch` | Fused normalize + einsum + FFN |
| Precompute WM causal mask as buffer | Avoid per-span allocation |

**Result:** ~8.6K -> ~12K tok/s (1.4x speedup)

### Phase 2: Block-Level Compilation + WM -> GLA (commit ee32cb3)

| Change | Description |
|--------|-------------|
| Block-level `torch.compile` | Compile `block.forward_span` as single graph instead of individual functions |
| Pre-allocate all state tensors | `initialize_states()` removes lazy init guards from compiled paths |
| WM: softmax -> Gated Linear Attention | Pure PyTorch GLA recurrence replaces ring buffer + softmax attention |

**Block-level compilation** reduces compiled graphs from ~96 (3 functions x 8 layers x 4 blocks) to 4 (one per block). This dramatically reduces kernel launch overhead by allowing torch.compile to fuse entire block forward passes into optimized CUDA kernels.

**GLA Working Memory** eliminates ring buffer scatter/gather/cat ops (~24% of GPU time in Phase 1 profiling). The sequential GLA recurrence (P=64 steps) is efficiently fused by torch.compile within the block-level compilation scope.

**Result:** ~12K -> ~25K tok/s (2.1x speedup)

---

## Why We're Slower Than Baselines

1. **Sequential span processing:** 4 spans of P=64 instead of one parallel pass over T=256. PM commits and EM writes at span boundaries force serialization.

2. **Three memory systems:** PM (eligibility traces + Hebbian commits), EM (retrieval + write), WM (GLA recurrence) all add per-token and per-span overhead that baselines don't have.

3. **Recurrent computation:** GLA and PM eligibility are inherently sequential within a span (P=64 steps). Baselines use attention (fully parallel) or optimized CUDA scan kernels (Mamba).

4. **Kernel launch overhead:** Many small operations dispatched individually. Block-level compilation significantly reduced this but some overhead remains from span boundary operations.

---

## Potential Future Optimizations

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| FLA library GLA kernels | 1.5-2x WM speedup (fused Triton kernels) | Medium — requires `flash-linear-attention` dependency |
| Increase P to 128-256 | 1.3-1.5x (fewer span boundaries) | Low — config change, but fewer PM/EM commits |
| Custom CUDA kernel for GLA recurrence | 1.5-2x WM speedup | High — manual kernel development |
| `torch.associative_scan` for PM eligibility | 1.1-1.3x PM speedup | Low — drop-in replacement |
| Pipeline span boundary ops on CUDA streams | Variable | High — concurrent PM commit + next span forward |

### Realistic Speed Targets

| Scenario | tok/s | vs Mamba |
|----------|-------|---------|
| Current | 25K | 2.3x slower |
| + FLA kernels for GLA | ~35K | 1.7x slower |
| + P=128 + parallel scan | ~45K | 1.3x slower |

A 1.5-2x gap vs Mamba is the expected steady-state: three memory systems are the architectural cost for PM/EM/WM capabilities that baselines lack entirely.

---

## Profiling Results (2026-02-16)

Collected via `python -m scripts.profile_training --steps 10 --warmup 5` with PyTorch profiler.

### VRAM Usage

| Metric | Value |
|--------|-------|
| Model parameters | 85.1M |
| VRAM allocated (after warmup) | 1.59 GB |
| VRAM reserved (PyTorch cache) | 10.74 GB |
| Peak VRAM during training | 10.34 GB |
| Headroom on 24GB 4090 | ~13.7 GB |

The large gap between allocated (1.6 GB) and reserved (10.7 GB) is due to PyTorch's CUDA caching allocator retaining freed memory for reuse. Peak VRAM of 10.3 GB occurs during backward pass. The ~13.7 GB headroom means batch size could be increased (at the cost of throughput per token due to memory bandwidth).

### GPU Kernel Breakdown

Top operations by total CUDA time (10 profiled steps):

| Operation | CUDA Time | % of Total | Calls | Description |
|-----------|-----------|------------|-------|-------------|
| `aten::mm` | 408ms | 21.3% | 25,940 | Linear layer matmuls |
| CUTLASS bf16 GEMM (64x64) | 176ms | 9.2% | 5,240 | TensorCore matmul kernels |
| `aten::copy_` | 145ms | 7.6% | 35,360 | dtype conversions (bf16↔fp32) |
| `aten::addmm` | 120ms | 6.3% | 9,760 | Bias + matmul (linear layers) |
| `aten::add_` | 118ms | 6.2% | 43,760 | Residual additions |
| AdamW optimizer step | 112ms | 5.9% | 10 | Parameter updates |
| `aten::mul` | 101ms | 5.3% | 54,880 | Gating, scaling ops |
| Triton fused (GLA gates) | 99ms | 5.2% | 640 | Fused select/cat/mul/unsqueeze |
| `aten::bmm` | 89ms | 4.7% | 19,020 | Batched matmuls (PM/EM) |
| Triton fused (log_sigmoid) | 74ms | 3.9% | 40 | GLA gate activation backward |
| `aten::sum` | 73ms | 3.8% | 23,220 | Normalization, aggregation |

**Key observations:**
- Matrix multiplications dominate (~37% total: mm + CUTLASS + addmm + bmm), which is expected for a model with many linear projections per layer.
- dtype conversions (`copy_`) are 7.6% — the model uses bf16 autocast with fp32 state tensors (PM eligibility, GLA state), requiring frequent casts.
- Triton auto-generated kernels from `torch.compile` successfully fuse complex GLA gate computations into single kernels.
- The AdamW optimizer step is ~6% — relatively small thanks to fused AdamW.

### Kernel Launch Overhead

| Metric | Value |
|--------|-------|
| `cudaLaunchKernel` calls per step | ~28,600 |
| `cuLaunchKernel` calls per step | ~8,000 |
| Total kernel launches per step | ~36,600 |
| CPU time in kernel launch | 709ms / 13.5s total = 5.3% |

Down from ~56K kernel launches pre-block-compile (Phase 1). Block-level compilation reduced this by ~35%, but the remaining ~37K launches are from the many small operations within compiled graphs and span boundary ops.

### GPU Utilization

| Metric | Value |
|--------|-------|
| Self CUDA time (10 steps) | 1.912s |
| Wall clock (10 steps) | 3.25s |
| GPU compute utilization (profiled) | ~59% |

The ~59% utilization is measured under profiler overhead (profiling adds ~10-15% CPU overhead). Actual utilization during normal training is estimated at ~65-70%. The remaining gap is CPU-side kernel dispatch and span boundary operations between compiled graphs.

### Compiled Graph Summary

Block-level compilation produces these main graphs:

| Graph | Calls (10 steps) | CUDA Time | Role |
|-------|-------------------|-----------|------|
| Block forward (graph 1) | 60 | 2.56s | Main block forward pass (L=8 layers × 4 spans × ... ) |
| Block forward (graph 2) | 60 | 1.81s | Second block forward pass |
| Backward (graph 1) | 20 | 741ms | Backward through block 1 |
| PM eligibility | 420 | 227ms | Per-layer eligibility trace updates |
| Backward (graph 2) | 20 | 461ms | Backward through block 2 |
| Span boundary ops | 40 | 291ms | PM commit + EM write at span boundaries |

---

## Reproducing These Results

### Profiling

```bash
# Full profiling with PyTorch profiler (requires ~6 min compilation warmup)
python -m scripts.profile_training --steps 10 --warmup 5

# Chrome trace output: /tmp/neuromorphic_profile.json
# Open in chrome://tracing or https://ui.perfetto.dev/
```

### Baseline Speed Comparison

```bash
# Compare neuromorphic vs Mamba throughput on random data
python -m scripts.benchmark_speed_compare --warmup 5 --steps 15

# Output: JSON with tok/s for both models + ratio
```

### Quick Throughput Check

```bash
# Run actual training for a few steps with --compile
python -m src.train --phase B --steps 10 --compile --preset tier_a_wide
# Watch for "tok/s" in training logs after compilation warmup
```
