# Speed Optimizations & Scaling

**Last updated:** 2026-02-24
**Hardware:** NVIDIA RTX 4090 (24GB VRAM), benchmarked with Tier A Wide
**Model:** Tier A Wide (D=768, L=8, B=2, ~85M params), Phase B, BS=32, T=256, P=64

---

## Current Performance (Tier A Wide, compiled)

| Metric | Value |
|--------|-------|
| Training throughput | ~24K tok/s |
| Tokens per step | 8,192 (BS=32 × T=256) |
| Step time | ~340 ms |
| Peak VRAM | ~2.4 GB |
| 1.5B token training time | ~17h |

### Baseline Comparison (Tier A Wide, 1.5B tokens, RTX 4090)

| Model | Params | tok/s | 1.5B train time | Relative speed |
|-------|--------|-------|-----------------|----------------|
| Pythia-160M (transformer) | 134M | ~116K | ~2.7h | 4.9x faster |
| Mamba-130M (SSM) | 115M | ~52K | ~3.3h | 2.2x faster |
| **Neuromorphic LM** | **85M** | **~24K** | **~17h** | **1x** |

The neuromorphic model is slower than baselines due to its three memory systems (PM/EM/WM), sequential span processing (4 spans of P=64), and span-boundary operations (PM commits, EM writes). This is the cost of having persistent, adaptive memory — baselines are simpler feedforward or SSM architectures with no online learning.

---

## Model Tier Scaling

### Architecture configurations

| Tier | Params | D | L | B | D_h | Target Use |
|------|--------|---|---|---|-----|-----------|
| **A** | 39.7M | 512 | 8 | 4 | 128 | Debug/rapid iteration |
| **A Wide** | 85.1M | 768 | 8 | 2 | 384 | Development, 4090 training |
| **B** | 78.1M | 768 | 12 | 6 | 128 | Research (many shallow blocks) |
| **1B** | 1,070M | 2048 | 16 | 2 | 1024 | Conversational quality, cloud GPU |
| **C** | 164.0M | 1024 | 24 | 8 | 128 | Research (deep, many blocks) |

### Scaling pattern: D_h = D / B

The per-block hidden dimension D_h determines per-layer parameter count (~27×D_h² with PM readout FFN). Key implications:

- **Increasing B at fixed D** reduces D_h, which reduces per-layer capacity quadratically. B=2→4 at D=2048 drops from 1,070M to 599M params.
- **The correct scaling axis**: increase D proportionally with B. This is identical to how multi-head attention scales (head_dim = D/n_heads).

| Scale target | Recommended config | D_h |
|-------------|-------------------|-----|
| ~85M | D=768, L=8, B=2 | 384 |
| ~1B | D=2048, L=16, B=2 | 1024 |
| ~4B | D=4096, L=16, B=4 | 1024 |
| ~16B | D=8192, L=24, B=8 | 1024 |

D_h=1024 is the sweet spot for 1B+ models. Keep D_h >= 384 for meaningful per-layer capacity.

### VRAM estimates (training, bf16 + fp32 optimizer)

Rule of thumb: params × 18 bytes (bf16 weights + fp32 grad copy + fp32 optimizer states) + activations.

| Tier | Params | Param+Optim | Est. Total | Recommended GPU |
|------|--------|-------------|------------|----------------|
| A Wide | 85M | 1.5 GB | ~2.4 GB (BS=32) | RTX 4090 (24GB) |
| 1B | 1,070M | 19.3 GB | ~21 GB (BS=8) | A100 80GB |
| ~4B | ~4B | ~72 GB | ~85 GB | H100 80GB or multi-GPU |

Gradient checkpointing reduces activation memory by ~12% with negligible speed cost (~1%).

### Throughput scaling expectations

Throughput scales roughly as: tok/s ∝ (GPU_FLOPS / params_per_step). Larger models have more FLOPs per token, so tok/s decreases:

| Tier | Estimated tok/s | GPU | Notes |
|------|----------------|-----|-------|
| A Wide (85M) | ~24K | RTX 4090 | Measured (Phase B) |
| 1B | ~3-5K (est.) | A100 80GB | A100 has ~2x FLOPS of 4090, but 12x more params |
| 1B | ~5-8K (est.) | H100 80GB | H100 has ~3x FLOPS of 4090 |

---

## Optimizations Applied (Chronological)

### Phase 1: Sync Removal + Compilation

Removed GPU-CPU synchronization barriers:

| Fix | Impact |
|-----|--------|
| Gate `_memory_budget_utils` behind log interval | Eliminated 18 `.item()` syncs per step |
| Remove `.item()` from loss functions | Keep loss as tensor until metrics dict |
| Remove `.any()` sync in model reset check | Unconditional reset (masked no-op when empty) |
| Fuse `gate_a + gate_b` into single GEMM | One kernel instead of two per layer |
| Compile PM `apply_batch` | Fused normalize + einsum + FFN |
| Precompute WM causal mask as buffer | Avoid per-span allocation |

**Result:** ~8.6K → ~12K tok/s (1.4x)

### Phase 2: Block-Level Compilation + GLA WM

| Change | Description |
|--------|-------------|
| Block-level `torch.compile` | Compile `block.forward_span` as single graph instead of per-function |
| Pre-allocate all state tensors | `initialize_states()` removes lazy init guards from compiled paths |
| WM: softmax → Gated Linear Attention | Pure PyTorch GLA recurrence replaces ring buffer + softmax attention |

**Result:** ~12K → ~25K tok/s (2.1x)

### Phase 3: Decoder Compilation + Holographic PM

| Change | Description |
|--------|-------------|
| Compile spatial decoder | Register index buffers, stack tensors, `torch.compile(decoder.forward)` |
| einsum → matmul/bmm | Replace 9 einsum calls with direct matmul/bmm |
| GLA recurrence optimization | Hoist `torch.exp(g)` outside loop, preallocate output tensor |
| Holographic PM read | `y = x * (W @ x)` — quadratic in x, same compute cost |

**Result:** ~25K → ~26.4K tok/s (5.6%)

### Phase 4: Robustness + NaN Hardening + Batch Size Optimization

| Change | Description |
|--------|-------------|
| Pre-LayerNorm for Block Layer 0 | `input_norm = LayerNorm(D_h)` fixes gradient asymmetry (3.56x) between L0 and deeper layers |
| NaN hardening | Replace `nan_to_num(0.0)` with explicit `all_inactive` detection in WM/EM softmax paths |
| Gradient checkpointing support | `torch.utils.checkpoint` on FFN — 12% less VRAM, ~1% speed cost |
| Batch size sweep | Found BS=32 optimal (was previously using BS=32, confirmed) |
| torch.compile mode evaluation | Benchmarked `max-autotune-no-cudagraphs` — 3.5% slower than `mode="default"`, reverted |

**Result:** ~26.4K → ~24K tok/s (Phase B steady-state). The robustness fixes (pre-LayerNorm, NaN hardening) improved training stability without measurable throughput regression. The BS=32 sweep confirmed this was already optimal.

**Note on compile modes:** `max-autotune` tries hundreds of Triton kernel variants but at tier A Wide scale, the default autotuned kernels are already near-optimal. `max-autotune` may help at 1B+ scale where kernel selection matters more. CUDA graphs are incompatible with our stateful memory writes (PM/EM modify tensors in-place between forward passes).

---

## Why We're Slower Than Baselines

1. **Sequential span processing:** 4 spans of P=64 instead of one parallel pass over T=256
2. **Three memory systems:** PM eligibility, EM retrieval+write, WM recurrence add per-token overhead
3. **Span boundary operations:** PM commits and EM writes between spans force serialization
4. **Recurrent computation:** GLA and PM eligibility are inherently sequential within each span

---

### Phase 5: FLA Triton Kernel Integration (optional `--fla` flag)

| Change | Description |
|--------|-------------|
| FLA HGRN for layer scan | `fused_recurrent_hgrn` replaces `_sequential_scan` for `h_t = a*h + b` |
| FLA chunk_gla for GLA WM | `chunk_gla` replaces `_gla_recurrence` for working memory |
| Gate reparameterization | `sigmoid(a_raw)` → `logsigmoid(a_raw)` for HGRN log-space gates |
| Carry mask in log space | Doc boundary carry=0 → gate=-30 (exp(-30) ≈ 0) |

**Result:** FLA kernels are numerically correct (HGRN max diff 0.023, GLA max diff 0.001 in bf16). However, **no throughput improvement at P=64 with torch.compile**:

- FLA functions have `@torch.compiler.disable`, causing graph breaks inside compiled regions
- torch.compile already fuses the P=64 sequential loop into efficient kernels
- Net effect: ~23K tok/s with or without FLA when compile is active
- Without compile: FLA gives ~7K tok/s vs ~8K without (slightly slower due to kernel launch overhead)

**When FLA helps:** At larger P (128-256+), torch.compile's unrolled loop becomes less efficient while FLA's chunkwise parallel kernel maintains constant overhead. FLA may also help when their `@torch.compiler.disable` is relaxed in future versions.

**Usage:** `--fla` flag enables FLA kernels (automatically skips torch.compile for affected modules). Default is off (torch.compile preferred at P=64).

---

## Potential Future Optimizations

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| Increase P to 128-256 | 1.3-1.5x (fewer span boundaries, FLA shines here) | Low — config change |
| FlashAttention in spatial decoder | Minor (decoder is <5% of cost) | Low |
| `torch.associative_scan` for PM eligibility | 1.1-1.3x PM speedup | Low |
| Multi-GPU data parallelism | Linear scaling with GPU count | Medium |
| FLA torch.compile support | Would combine FLA speed with compile fusion | Depends on FLA upstream |

---

## Profiling

```bash
# Full profiling with PyTorch profiler
python -m scripts.profile_training --steps 10 --warmup 5

# Chrome trace output: /tmp/neuromorphic_profile.json
# Open in chrome://tracing or https://ui.perfetto.dev/
```

### Quick Throughput Check

```bash
# Run actual training for a few steps with --compile
python -m src.train --phase B --tier a_wide --steps 10 --compile
# Watch for "tok/s" in training logs after compilation warmup (~5 min)
```
