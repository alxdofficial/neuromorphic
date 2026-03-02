# Redesign Options for GPU Throughput (v5 → v5.1+)

> **Decision (v5.1):** Implemented A1 (dense scan) with grouped PCM. Scan layers
> use dense nn.Linear for GPU efficiency. PCM and W_seed_w remain GroupedLinear
> with free .view() bridge between [BS,N,D] and [BS,N,C,D_col]. See commit history.
>
> **Result:** 92K tok/s train (trail=3, BS=24) — 2.9x speedup over v5's 31.8K tok/s.
> Matches Mamba-130M (90K), 37% fewer params, 38% less VRAM. With trail=1: 103K tok/s.
> Actual benchmarked baselines (same GPU, max BS): Pythia-160M=110K, Mamba-130M=90K.

**Original goal**: Match Mamba-130M (~90K tok/s) training speed.
**v5 baseline**: 31.8K tok/s on RTX 4090 (Tier A, BS=12, N=512, compiled).
**v5.1 result (trail=3)**: **92K tok/s** (BS=24, compiled) — matches Mamba, fewer params/VRAM.
**v5.1 result (trail=1)**: **103K tok/s** (BS=36, compiled) — exceeds Mamba.

## Sacred Invariants (preserved by ALL options)

1. PM + EM + PCM as distinct memory systems
2. Lifelong PM/EM as runtime state (not static params)
3. Vector-surprise-driven updates
4. EM trail-based soft composition (no hard top-k)
5. Bounded O(1)-w.r.t-context inference memory
6. Causal write-before-read (prefix-sum buffers)

## Root Causes of Slowness

| Cause | % CUDA time | Why |
|-------|-------------|-----|
| Element-wise kernels (sigmoid, silu, LN, mul, add) | 44% | ~0 arithmetic intensity, pure memory bandwidth |
| Grouped matmuls (16 narrow einsum) | 32% | 23% matmul efficiency (38 vs 163 TFLOP/s) |
| HGRN scan recurrence | 18% | Inherently memory-bound (~0.75 ops/byte) |
| Softmax, misc | 6% | Small |

Scan layers = 84% of forward time. Two equal scan stacks (Stage 1 + Stage 3) means
the most expensive component runs twice.

---

## Category A: Scan Layer Compute Efficiency

### A1. Go Fully Dense (C=1, nn.Linear)

Replace GroupedLinear(C=16, D_col=128) with nn.Linear(D=2048).

- **How**: ScanLayer uses nn.Linear + RMSNorm. Scan functions take 3D [BS,N,E]
  tensors. All 4D [BS,N,C,D_col] reshaping eliminated from model.py.
- **Param count**: Identical (d_inner=1024 matches current per-column E=1024).
- **GPU gain**: Matmul efficiency 23% → ~80-100%. HGRN batch drops from BS*C=192 to BS=12.
  Permute/reshape overhead (~10ms/fwd) eliminated entirely.
- **Tradeoff**: Complete loss of column independence. All features mix in every layer.
  Memory systems no longer serve as the only cross-feature mixer.
- **Estimated speedup**: 1.5-2x on scan layers alone.

### A2. Grouped GEMM Kernel (CUTLASS/Triton)

Keep GroupedLinear structure but replace einsum dispatch with a custom grouped GEMM.

- **How**: Write a Triton or CUTLASS kernel that processes all C groups in a single
  kernel launch with proper tiling. Drop-in replacement for GroupedLinear.forward().
- **Param count**: Identical.
- **GPU gain**: Better tensor core utilization for narrow matmuls. Reduces kernel
  launch overhead (1 launch vs C=16 implicit launches in einsum).
- **Tradeoff**: High engineering effort. Column independence fully preserved.
  Unclear how close to dense efficiency it can get for [128,2048] shapes.
- **Estimated speedup**: Unknown — depends on kernel quality. Maybe 1.2-1.5x on matmuls.

### A3. Grouped Conv1d (cuDNN)

Treat GroupedLinear as 1x1 grouped convolution.

- **How**: Replace proj_in/proj_out with nn.Conv1d(kernel_size=1, groups=C).
  Input layout: [BS, D, N] (channels-first). cuDNN handles grouped dispatch.
- **Param count**: Identical.
- **GPU gain**: cuDNN has heavily optimized grouped conv kernels that may be faster
  than einsum for this exact pattern. Low-effort experiment.
- **Tradeoff**: Requires layout transpose (channels-first vs channels-last). Column
  independence preserved. May or may not be faster — needs benchmarking.
- **Estimated speedup**: Unknown — quick to test.

### A4. Routed Wide-Column Experts (MoE-style)

Replace C=16 narrow columns (all tokens) with 2-4 wide columns (token subsets).

- **How**: Router assigns each token to 1-2 expert columns. Each expert is wide
  (D_col=512 or 1024), giving large matmuls. Load-balance regularization ensures
  even distribution. Memory-slice addressing preserved per expert.
- **Param count**: Similar (fewer wider experts ≈ same total params).
- **GPU gain**: Large matmuls saturate GPU. Fewer but bigger kernel launches.
- **Tradeoff**: Not all columns see every token (current invariant violated).
  Adds routing complexity + load-balance aux loss. Column independence partially
  preserved (each expert is independent). Cross-column communication only through
  memory (which the user likes).
- **Estimated speedup**: Potentially large (Mixtral-style MoE gets good GPU util).

---

## Category B: Scan Depth / Architecture

### B1. Eliminate Stage 3 Scan (parallel mixer)

Replace Stage 3's L_scan layers with a non-recurrent per-token mixer.

- **How**: After memory ops, integrate signals with:
  `integrated = H + pm_read + em_read + cum_em`
  Then apply 1-2 layers of `RMSNorm → SwiGLU MLP → residual` (parallel, not recurrent).
  Output through lm_head.
- **Param count**: Fewer (MLP layers cheaper than scan layers).
- **GPU gain**: Eliminates ~42% of scan compute (6 out of 12 layers). MLP layers
  are pure matmuls — much better GPU utilization than scan.
- **Tradeoff**: Memory signals don't propagate causally within the segment after
  integration. Token t's memory read influences t's logit but doesn't feed into
  a recurrence that affects t+1. Biggest architectural risk.
- **Estimated speedup**: ~1.5-2x on total forward time.

### B2. Asymmetric Dual-Scan (shallow Stage 3)

Keep Stage 3 as a scan but make it much shallower.

- **How**: Instead of L_scan=6 for both stages, use L_stage1=8-10, L_stage3=2.
  Total scan depth similar or slightly less. Stage 3 retains causal recurrence
  for memory integration but is cheaper.
- **Param count**: Adjustable (redistribute params from Stage 3 to Stage 1).
- **GPU gain**: Fewer total scan layers = less serial recurrence.
  Could also make Stage 3 narrower (project D → D_int, scan, project back).
- **Tradeoff**: Less causal processing of memory signals. Mitigated by memory
  state persisting across segments (slow path still works well).
- **Estimated speedup**: 1.3-1.5x depending on how aggressive.

### B3. Reduce Total Scan Depth + Widen

Fewer scan layers but each is more expressive.

- **How**: Drop total from 12 → 6-8 layers. Increase d_inner or add a per-token
  MLP to preserve model capacity. Fewer layers = fewer kernel launches, fewer
  intermediate tensors written/read.
- **Tradeoff**: Less recurrent depth may hurt sequence modeling. But current model
  is bandwidth-bound, not compute-bound, so fewer wider layers may be net positive.
- **Estimated speedup**: Proportional to depth reduction.

---

## Category C: Kernel Fusion / Data Movement

### C1. Scan Tensor Layout Fix

**Status: N/A — superseded by A1 (dense scan)**

Eliminate permute/copy overhead in fused_scan path.

- **How**: Change ScanLayer internal layout to [BS,C,N,D_col] (C before N) so the
  reshape to [BS*C,N,E] for HGRN is a view, not a copy. Or go 3D if using dense (A1).
- **GPU gain**: ~10ms/fwd saved (already measured: 1.29ms→0.43ms per layer × 12).
- **Effort**: Low (layout change only).
- **Estimated speedup**: ~5%.

### C2. Fused ScanLayer Custom Op (FlashAttention-style)

Fuse norm→proj→activation→scan→proj→residual into a single custom Triton kernel.

- **How**: Write a custom forward+backward Triton op that does the entire ScanLayer
  in one kernel. Recompute cheap intermediates in backward instead of saving them.
  Reduces memory traffic + saved activation memory.
- **GPU gain**: Eliminates most element-wise kernel launches within the layer.
  Reduces bytes/token significantly. May allow higher BS.
- **Tradeoff**: Very high engineering effort. Custom backward is complex.
  Debugging/maintenance burden. Would need reimplementation if architecture changes.
- **Estimated speedup**: 1.3-2x per scan layer if done well.

### C3. RMSNorm + Fused Epilogues

**Status: DONE**

Replace LayerNorm with RMSNorm, fuse bias+activation into single ops.

- **How**: RMSNorm is cheaper (no mean subtraction) and fuses more easily.
  Use torch.compile fused epilogues for bias+silu, bias+sigmoid patterns.
- **GPU gain**: Reduces element-wise kernel count. Small per-kernel savings.
- **Effort**: Low.
- **Estimated speedup**: 5-10%.

### C4. torch.compile Whole-Graph Capture

Ensure compilation captures forward_segment as a single graph.

- **How**: Fix dynamic shapes, avoid graph breaks. Enable CUDA graphs / reduce-overhead.
  Use static shapes for masks. Compile with fullgraph=True.
- **GPU gain**: Reduces Python overhead and kernel launch latency.
- **Effort**: Medium (debugging graph breaks).
- **Estimated speedup**: 5-10%.

---

## Category D: Memory Ops Optimization

### D1. Flash Attention for EM Reads

Route trail_read_all and compute_novelty_all through flash attention / SDPA.

- **How**: Both are essentially attention: query=[BS*B,N,D], key/value=[BS*B,M,D].
  Use torch.nn.functional.scaled_dot_product_attention with custom masking.
- **GPU gain**: Reduces softmax/intermediate materialization. Flash kernel is fused.
- **Tradeoff**: M=384 is small — may not benefit from flash attention's tiling.
  Shape doesn't match typical flash attention workloads well.
- **Estimated speedup**: Small (EM is only 12.5% of forward time).

### D2. Fuse EM Element-wise Chains

Fuse the scalar gate, expand, sum chains in trail_read_all.

- **How**: Custom Triton kernel or rely on torch.compile to fuse the chain:
  dot→sigmoid→mul→add within each trail step.
- **GPU gain**: Reduces element-wise kernel launches within EM.
- **Effort**: Medium.
- **Estimated speedup**: Small.

---

## Category E: Pipeline / Infrastructure

### E1. Non-blocking Data Pipeline

Async H2D transfers, pinned memory, prefetch ring buffer.

- **How**: Fix streaming.py/trainer.py to overlap data loading with compute.
- **Estimated speedup**: 10-15% if data pipeline is currently on critical path.

### E2. Benchmark Unification

Canonical speed comparison harness across neuromorphic/pythia/mamba.

- **How**: Single script, same tokenizer, same BS policy, same compile settings.
  NVTX markers for stage-level profiling.
- Not a speedup itself but required for measuring progress.

### E3. Multi-GPU (DDP/FSDP)

Data-parallel scaling with per-replica PM/EM state.

- **How**: Standard DDP for gradient sync. PM/EM state stays local per replica.
  Tensor-parallel over D for large tiers.
- Future work, after single-GPU efficiency is solved.

---

## Suggested Combinations (Pareto paths)

### Path 1: Conservative (low risk, moderate gain)
C1 (layout fix) + C3 (RMSNorm) + C4 (compile) + B2 (asymmetric scan) + E1 (pipeline)
**Expected**: ~45-55K tok/s. May not reach Mamba.

### Path 2: Dense rewrite (medium risk, large gain)
A1 (dense C=1) + B2 (asymmetric scan) + C4 (compile) + E1 (pipeline)
**Expected**: ~60-80K tok/s. Likely reaches Pythia, possibly Mamba.

### Path 3: Maximum throughput (higher risk)
A1 (dense) + B1 (eliminate Stage 3 scan) + C4 (compile) + E1 (pipeline)
**Expected**: ~70-90K tok/s. Likely matches both baselines.

### Path 4: Preserve columns (medium risk)
A4 (routed experts) + B2 (asymmetric scan) + C4 (compile) + E1 (pipeline)
**Expected**: ~55-75K tok/s. Preserves column concept. Higher implementation effort.

### Path 5: Long-term optimal (high effort)
A1 (dense) + B2 (asymmetric) + C2 (fused ScanLayer) + D1 (flash EM) + E1 (pipeline)
**Expected**: ~80-100K tok/s. Requires significant kernel engineering.
