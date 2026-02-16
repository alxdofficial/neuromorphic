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
