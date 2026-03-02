# Performance Profile — Neuromorphic LM v5 Tier A

**Hardware**: RTX 4090 (24 GB VRAM, 165 TFLOP/s bf16 peak)
**Config**: D=2048, D_embed=384, C=16, D_col=128, L_scan=6, expansion=8, B=4, M=384, N=512, n_trail_steps=1
**Params**: 92.2M | **Compiled**: torch.compile enabled

## Current Throughput

| Metric | Value |
|--------|-------|
| Train tok/s (BS=12) | 31,863 |
| Infer tok/s (BS=12) | 99,711 |
| ms/step (train) | 192.8 |
| ms/step (infer) | 61.6 |
| Max train BS | 12 |
| Peak VRAM | 18.30 GB |
| VRAM split | 0.35 wt / 0.52 opt / 19.20 act |

## Baseline Comparison

| Model | Params | FLOPs/token | tok/s | GPU util |
|-------|--------|-------------|-------|----------|
| **Ours** | 92M | 205M | 31,863 | 4.0% |
| Mamba-130M | 130M | 260M | 89,000 | 14.0% |
| Pythia-160M | 160M | 320M | 69,000 | 13.4% |

Our model does **less** compute per token than either baseline but achieves only
4% GPU utilization. The bottleneck is arithmetic intensity, not total FLOPs.

## Forward Pass Time Breakdown

### By module (BS=12, N=512, single forward pass)

| Module | Time | % of forward |
|--------|------|-------------|
| Scan layers (12 total) | ~60 ms | 84% |
| EM (trail_read + novelty + commit) | ~8.9 ms | 12.5% |
| PM | ~0.5 ms | 0.7% |
| PCM | ~0.5 ms | 0.7% |
| Embed + head + misc | ~1.5 ms | 2.1% |

### Within a single ScanLayer (~5 ms each, 12 layers)

| Operation | Time | % of layer |
|-----------|------|-----------|
| proj_in (GroupedLinear D_col→2E) | 1.45 ms | 30% |
| fused_scan (HGRN Triton kernel) | 1.94 ms | 40% |
| proj_out (GroupedLinear E→D_col) | 0.74 ms | 15% |
| silu activation | 0.44 ms | 9% |
| LayerNorm | 0.37 ms | 7% |
| residual add | 0.04 ms | 1% |

### Within EM trail_read_all (~2.9 ms, called once)

| Operation | Time | % |
|-----------|------|---|
| matmuls (scores + attn@V) | 0.63 ms | 22% |
| softmax + masked_fill | 0.47 ms | 16% |
| scalar gate (dot + sigmoid) | ~0.5 ms | 17% |
| expand + sum | 0.3 ms | 10% |
| other element-wise | ~1.0 ms | 35% |

## Root Cause Analysis

### Why 4% GPU utilization?

Three structural factors:

1. **Narrow grouped matmuls**: GroupedLinear uses C=16 independent [BS×512, 128] × [128, 2048]
   matmuls. Each is too narrow to saturate the GPU. Measured: 38 TFLOP/s on proj_in vs
   163 TFLOP/s for an equivalent dense matmul (23% matmul efficiency).

2. **Memory-bound scan recurrence**: The HGRN Triton kernel does `h_t = a_t * h_{t-1} + b_t`
   — 2 FLOPs per element loaded. Arithmetic intensity ~0.75 ops/byte vs ~1500 for transformer
   attention matmuls. The kernel is already well-optimized; the recurrence is inherently
   memory-bandwidth-bound.

3. **Two scan stages + memory ops**: Processing each token through 12 scan layers plus
   memory operations means more serial work per token than a single-pass transformer or SSM.

### CUDA kernel breakdown (nsight profile)

| Kernel category | % of CUDA time |
|----------------|---------------|
| Element-wise (sigmoid, silu, mul, add, LN) | 44% |
| Matmul (bmm/einsum via GroupedLinear) | 32% |
| HGRN Triton recurrence | 18% |
| Softmax, masked_fill, misc | 6% |

Element-wise kernels dominate because they have near-zero arithmetic intensity
(1-2 FLOPs per memory access).

## GroupedLinear Dispatch Benchmark

Tested alternative dispatch methods for the largest matmul (proj_in: [16, 128] → [16, 3072]):

| Method | Time | Notes |
|--------|------|-------|
| einsum (current) | 2.02 ms | `'...ci,cio->...co'` |
| bmm | 2.02 ms | Permute → reshape → bmm |
| matmul broadcast | 2.53 ms | Permute overhead |
| block-diagonal F.linear | 7.14 ms | 94% zero weights, wastes bandwidth |
| compiled einsum | 2.01 ms | Negligible compile benefit |

**Verdict**: Current einsum dispatch is already optimal. No dispatch change helps.

## Column Width vs GPU Efficiency

Tested varying C (with D=2048 fixed) on proj_in:

| C | D_col | Wall time | TFLOP/s | Total FLOPs |
|---|-------|-----------|---------|-------------|
| 16 | 128 | **2.0 ms** | 38 | 77G |
| 8 | 256 | 2.4 ms | 64 | 155G |
| 4 | 512 | 3.3 ms | 94 | 309G |
| 1 | 2048 | 9.3 ms | 133 | 1237G |
| Dense | 2048 | 7.6 ms | 163 | 1237G |

C=16 is fastest in wall time despite lowest GPU utilization because total FLOPs
scale as D²/C. Fewer columns = more FLOPs per column = slower despite better
utilization. The grouped structure is already a net win.

## Config Optimization History

Tested various tier_a configurations (all compiled, max BS that fits):

| Config | tok/s | Notes |
|--------|-------|-------|
| Original (L=12, exp=4, B=6, trail=2) | ~29,000 | Starting point |
| L=6, exp=8 | 30,550 | +5.3% — wider layers, fewer depth |
| + n_trail_steps=1 | 31,702 | +3.8% — less EM iteration |
| + B=4, M=384 | 31,863 | +0.5% — fewer banks, same capacity |
| + scalar EM gate | 31,863 | ~same (gate is small fraction) |

### Rejected optimizations

| Change | Result | Why rejected |
|--------|--------|-------------|
| gradient_checkpointing=True | BS 12→36, tok/s 29K→20.4K | Compute-bound on recompute; flat throughput regardless of BS |
| B=2, M=768 | 32,163 tok/s but OOM at BS>14 | Marginal gain, fragile |
| N=256 (shorter segments) | 31,085 tok/s at BS=28 | HGRN kernel less efficient on shorter sequences |
| N=128 | 30,012 tok/s at BS=48 | Same — longer segments are faster |

## Segment Length Scaling

| N | Max BS | tok/s | Notes |
|---|--------|-------|-------|
| 512 | 12 | 31,703 | Best — HGRN kernel most efficient |
| 256 | 28 | 31,085 | -2% despite 2.3× batch |
| 128 | 48 | 30,012 | -5.3% despite 4× batch |

Longer segments win because the HGRN Triton kernel amortizes overhead better
over longer sequences.

## Identified Optimization Opportunities (not yet implemented)

### 1. Scan tensor layout change (~5% estimated)
The fused_scan path permutes [BS,N,C,E] → [BS,C,N,E] → [BS*C,N,E] before calling
HGRN, then permutes back. This triggers a full tensor copy each direction.
Microbenchmark: 1.29ms (with permute) vs 0.43ms (pre-permuted layout) — 3× faster
input preparation. At 12 layers, this is ~10ms saved (5% of 193ms forward pass).
Requires changing ScanLayer to use [BS,C,N,D_col] internal layout.

### 2. MoE-style column partitioning (architecture change)
Instead of 16 narrow columns each processing all tokens, use fewer wide "expert"
columns each processing a token subset. E.g., 2 experts × D=2048 × N/2 tokens.
Same total FLOPs but individual matmuls are much larger (better GPU utilization).
Tradeoff: loses "all columns see every token" property; memory system becomes
sole cross-column communication mechanism.

### 3. Custom Triton kernels for element-wise fusion
Fuse chains of element-wise ops (e.g., scan's LN→proj→silu→scan→proj→residual)
into single kernels to reduce memory bandwidth. Estimated difficulty: high.
Expected gain: up to 10-15% if element-wise overhead (44% of CUDA time) can be
halved.

## FLOPs Breakdown Per Token

| Component | FLOPs/token | % |
|-----------|-------------|---|
| Scan matmuls (12 layers) | 151M | 73.6% |
| EM trail_read | 12.6M | 6.1% |
| EM novelty | 12.6M | 6.1% |
| lm_head | 24.6M | 12.0% |
| proj_up + proj_down | 3.1M | 1.5% |
| W_seed_w | 1.0M | 0.5% |
| PM, PCM, misc | ~0.1M | <0.1% |
| **Total** | **205M** | |
