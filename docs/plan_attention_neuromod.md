# Attention-based Neuromodulator: Profile, Analysis, and Plan

> **SUPERSEDED (April 2026).** This doc captures the *initial* plan that
> forked the attention-neuromod branch from conv-grid, with shared-weights
> modulator and a per-token 2-layer state MLP. Many of those design choices
> (shared modulator weights, per-token state MLP, msg every token,
> modulation every 4 tokens) have since been replaced by the multi-
> timescale + per-cell + Triton-fused design described in `design.md`.
>
> Numbers in this doc reflect the intermediate state, not the current one.
> Keep for history / context only — for current architecture see `design.md`;
> for the latest throughput / param numbers see `RESULTS.md`.

Branch: `attention-neuromod` (forked from `conv-grid-modulator` @ `889f2a1`).

**Goal**: Keep the conv-grid branch's principled observations (per-edge + per-node
+ global) and full-rank actions, but replace the expensive Conv2d stack with
attention — which is genuinely GPU-friendly at our scale and already compiles to
the bmm patterns that tensor cores love.

## 1. Profile data

Both measured on the same RTX 4090 under `torch.autocast(bf16)`.

### 1.1 Throughput at BS=72 (production batch size)

| Branch | Step time | Throughput | Peak VRAM |
|---|---:|---:|---:|
| **main** (NC=8, N=32) | **239 ms** | **38.6K tok/s** | 22.1 GB |
| **conv-grid** (NC=1, N=256, DW-sep pyramid) | 5,854 ms | 1.6K tok/s | 17.0 GB |
| conv-grid at BS=8 | 712 ms | 1.4K tok/s | 3.0 GB |

**24× slower at production BS.**

### 1.2 Isolated-section timing at BS=8 on conv-grid branch

| Section | Time | Frequency |
|---|---:|---|
| build observation tensor | 1.06 ms | once/fire |
| encoder forward | **4.97 ms** | once/fire |
| decoder forward | **2.30 ms** | once/fire |
| W @ msg (bmm) | 0.023 ms | once/token |
| msg @ msgᵀ (hebbian bmm) | 0.027 ms | once/token |
| state MLP | 0.098 ms | once/token |
| msg MLP | 0.071 ms | once/token |
| full forward_chunk (T=128) | 143.5 ms | per segment |
| full step (fwd+bwd+opt) | 712.0 ms | per segment |

Forward budget: 32 fires × (4.97 + 2.30 + 1.06) ≈ **266 ms of modulator work per segment**.
Plus 128 steps × (0.023 + 0.027 + 0.098 + 0.071) ≈ **28 ms of memory step work**.
Plus LM (not separately profiled but small).

**Memory step (bmm-based) is ~10× faster per op than the modulator conv path.**

### 1.3 Per-stage encoder timing (single fire at BS=8)

| Stage | Cost | Spatial | Channels |
|---|---:|:-:|:-:|
| **stem (dense Conv2d k=7)** | **2.38 ms** | 256×256 | 78→48 |
| stage 0 (DW-sep stride 2) | 0.65 ms | 256→128 | 48→96 |
| stage 1 (DW-sep stride 2) | 0.26 ms | 128→64 | 96→192 |
| stage 2 (DW-sep stride 2) | 0.11 ms | 64→32 | 192→192 |
| stage 3 (DW-sep stride 2) | 0.09 ms | 32→16 | 192→192 |

**The stem alone is 48% of encoder time** despite being a single layer. It's the
only layer running at full 256×256 resolution — and at BS=8 with only 48 output
channels, there's not enough work to fill tensor cores.

### 1.4 Per-stage decoder timing

| Stage | Cost | Spatial | Channels |
|---|---:|:-:|:-:|
| init_proj (Linear) | 0.04 ms | — | 384→4096 |
| stage 0 (DW-sep upsample) | 0.28 ms | 4→8 | 256→128 |
| stage 1 | 0.16 ms | 8→16 | 128→96 |
| stage 2 | 0.14 ms | 16→32 | 96→64 |
| stage 3 | 0.14 ms | 32→64 | 64→48 |
| stage 4 | 0.21 ms | 64→128 | 48→32 |
| **stage 5 (final upsample)** | **1.08 ms** | **128→256** | 32→32 |

**Same story on decoder: the final full-resolution stage is 47% of decoder time.**

### 1.5 Top CUDA ops (conv-grid branch, 3 steps at BS=8)

| Op | CUDA time | % |
|---|---:|---:|
| `CompiledFunctionBackward` (bulk) | 1.63 s | 78% |
| `convolution_backward` | 0.62 s | 30% |
| `cudnn_convolution` (forward) | 0.36 s | 17% |
| `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16` | 0.23 s | 11% |
| `dgrad2d_c1_k1_nhwc_specialized` (pointwise 1×1 bwd) | 0.13 s | 6% |
| `triton_poi_fused__to_copy__unsafe_index_put_add_div` (F.interpolate backward) | 0.12 s | 6% |
| `nhwcAddPaddingKernel` (NCHW↔NHWC layout conversions) | 0.07 s | 3.5% |

**Conv (forward + backward) is ~47% of CUDA time. Everything else combined is 53%.**

**5472 conv kernel launches across 3 steps = 1824 per step.** At ~10-20 µs CPU
overhead each, that's ~20-35 ms of launch overhead per step.

### 1.6 Top CUDA ops (main branch for comparison)

| Op | CUDA time | % |
|---|---:|---:|
| `CompiledFunctionBackward` | 248 ms | 37% |
| `aten::mm` (main's per-cell logit-head einsum) | 183 ms | **27%** |
| `aten::linear` (state/msg MLPs) | 72 ms | 11% |
| `aten::addmm` | 69 ms | 10% |
| **`aten::bmm`** (W @ msg, msg @ msgᵀ) | **62 ms** | **9%** |
| `aten::copy_` (dtype casts) | 40 ms | 6% |

Main's top ops are **bmm / mm / addmm** — the tensor-core-friendly matrix
multiply family. No conv. No layout conversions.

### 1.7 FLOP accounting

| | Theoretical fwd (TF) | Fwd+bwd estimate (TF) | Measured throughput (TF/s) | Peak efficiency |
|---|---:|---:|---:|---:|
| **conv-grid @ BS=8** | 3.33 | ~10 | 14.0 | **8.5% of 165 TF/s peak** |

**We're running at 8.5% of peak. This is the honest diagnosis: the conv
encoder is dispatch-bound and bandwidth-bound, not compute-bound.**

## 2. Why "GPU-friendly" conv isn't actually fast here

Conventional wisdom: "conv is fast on GPUs." That's true in the image-classification
regime (C ≈ 256-2048, spatial 7-56, deep stacks). Our usage pattern violates every
assumption behind that intuition.

### 2.1 The stem: full-res × narrow-channel = worst case

Stem runs **Conv2d(78→48, k=7)** on `[8, 78, 256, 256]`. That's:

- Output positions: 8 × 256 × 256 = 524K per layer
- Per position compute: 78 × 48 × 49 = 183K FLOPs
- Total: 96 G FLOPs
- Memory read per position: ~78 × 49 × 2 bytes ≈ 7.6 KB
- Total memory read: 524K × 7.6 KB = 4 GB of activations to touch

At 1 TB/s 4090 memory bandwidth, that's **4 ms just for bandwidth**. We measured
2.38 ms — we're actually hitting the bandwidth limit (cuDNN probably reuses some
reads across the batch).

Tensor cores want at least M, N, K ≥ 64 to saturate. At K=48 input channels and
N=48 output channels, we're below the sweet spot. GPU does matrix instructions
but can't saturate.

### 2.2 DW-separable conv is fundamentally bandwidth-bound

Depthwise convolution has low arithmetic intensity by design:
- Per output element: `k²` FLOPs vs `(k² + 1)` memory reads
- At k=7: 49 FLOPs / ~50 reads ≈ 1 FLOP/byte
- 4090: 165 TF/s compute, 1 TB/s bandwidth → compute/bandwidth ratio = 165 FLOP/byte
- DW conv is **bandwidth-bound at ~1/165 of compute peak = ~1 TF/s effective**

This is *fundamentally* why DW-sep, despite having 40× fewer FLOPs than dense
conv at the same hidden size, barely speeds things up on GPU — the kernel runs at
near-peak-bandwidth, not near-peak-compute.

### 2.3 Many kernel launches

Conv-grid profile: 5472 conv calls / 3 steps = **1824 conv launches per step**.
Plus 768 NHWC layout conversions, plus all the `aten::copy_` etc.

At a typical 10 µs launch overhead each, 1824 × 10 µs = **18 ms** of launch
overhead per step just for conv calls.

Compare main: ~1735 `aten::mm` calls per step, but those are bigger per-call so
launch overhead is relatively smaller fraction of compute time.

### 2.4 Bilinear interpolate backward is slow

The decoder's bilinear upsampling has a slow `unsafe_index_put` backward kernel
that shows up at 6% of CUDA time. It's an inherent quirk of the bilinear backward.
Nearest-neighbor upsampling would be faster, but changes quality.

### 2.5 Why main's approach wins

Main's modulator uses one big `einsum("bni,nih->bnh", mod_input, w1)` — which
compiles to a single big `aten::mm` call per batch. Key differences:

| | main (MLP modulator) | conv-grid (conv encoder) |
|---|---|---|
| Primitive | `aten::mm` / `aten::bmm` / `aten::linear` | `cudnn_convolution` |
| Tensor-core-friendly? | **Yes** (designed around gemm) | Partially (only when C and spatial are large) |
| Arithmetic intensity | High (`k² · FLOPs/byte`) | Low (DW: 1 FLOP/byte) |
| Kernel launches | 1-2 per modulator call | 10+ per layer × 6 layers × 32 fires |
| Backward graph | compiled fused | cuDNN-specific kernels |
| Layout conversions | none | NCHW ↔ NHWC, 768 calls/step |

Main runs its modulator at ~40 GFLOPs/ms effective, ~25% of peak compute —
because every op saturates tensor cores.

Conv-grid runs its modulator at ~0.5 GFLOPs/ms effective, ~8% of peak compute —
because half the time is bandwidth-bound DW ops, layout conversions, and launch
overhead.

**This isn't fixable by swapping dense↔DW-sep or changing kernel size. The
primitive itself is mismatched with our workload.**

## 3. Redesign: attention over neurons

Keep conv-grid's principled observations and full-rank action emission. Replace
the conv with attention — which compiles to the same `bmm`-based pattern that
makes main fast.

### 3.1 Architecture

Each of the N=256 neurons is a token. The attention layers mix tokens through
self-attention with edge-feature biases. Output heads emit per-neuron ΔW rows and
per-neuron Δdecay directly.

```
INPUTS (same information content as conv-grid):
  Per-neuron:    h [N, D_n], msg [N, D_n], received [N, D_n], decay [N]
  Per-edge:      W [N, N], hebbian [N, N]
  Global:        s_mem_live, s_mem_ema_fast
  Role:          role_id [N] — 0=input-port, 1=output-port, 2=internal

STAGE 1 — Build tokens (per neuron features → F-dim token)
  tok[i] = MLP_in(concat[
      h_norm[i], msg_norm[i], decay[i],      # scalar rates
      role_emb[i],                            # learned role embedding
      h_proj[i], msg_emit_proj[i], msg_recv_proj[i],  # compressed content (d_proj=16)
      s_mem_live, s_mem_ema_fast,             # broadcast globals
  ])
  # shape: [BS, N, F]  where F=64 (tunable)

STAGE 2 — Build edge biases (per edge → scalar bias, per head)
  edge_features[i, j] = [W[i, j], hebbian[i, j], W[i, j] - W[j, i]]
  edge_bias[i, j] = MLP_edge(edge_features[i, j])   # [BS, n_heads, N, N]

STAGE 3 — Attention layers × L=2
  For each layer:
    Q, K, V = Linear projections of tok            # [BS, n_heads, N, F/n_heads]
    attn_scores = (Q K^T) / sqrt(F/n_heads) + edge_bias
    attn = softmax(attn_scores, dim=-1)
    tok_out = tok + attn @ V                        # residual
    tok_out = tok_out + FFN(LayerNorm(tok_out))     # FFN + residual

STAGE 4 — Per-neuron output heads
  ΔW_row[i] = W_out_dW  · tok[i]                    # [BS, N, N]   (full-rank!)
  Δdecay[i] = W_out_dec · tok[i]                    # [BS, N]
```

### 3.2 Why this works

- **Attention's Q·K^T and attn·V are `bmm` operations** — the exact primitive
  that makes main fast. Tensor cores saturate well at N=256 with F=64.
- **Edge bias is a per-pair scalar** (or per-head scalar). Computed once via a
  small MLP, then added to attention scores. This injects per-edge information
  without materializing a per-edge × channel tensor.
- **No spatial locality assumption.** Attention is permutation-equivariant over
  tokens — the correct inductive bias for unordered internal neurons.
- **Per-neuron output heads** emit full-rank ΔW via `tok @ W_out_dW` (shape
  `[F, N]`, shared across neurons). Each neuron directly produces its ΔW row.
- **Compute stays low**: attention is `O(N² · F)` per layer = 4M FLOPs at N=256,
  F=64. Compare: conv encoder is `O(N² · C²)` = 4M × C² ≈ hundreds of M FLOPs.

### 3.3 Compute budget (estimated)

Per fire per sample at N=256, F=64, 2 layers, 4 heads:

| Stage | FLOPs |
|---|---:|
| Build tokens (MLP_in) | 0.5M |
| Edge bias MLP | 0.3M |
| Attention × 2 layers | 2 × (3·N·F² + N²·F + N·F² + N²·F) ≈ 10M |
| FFN × 2 layers | 2 × N·4F² ≈ 4M |
| Output heads (tok → ΔW_row, Δdecay) | N·F·N + N·F ≈ 4M |
| **Total per fire per sample** | **~19M FLOPs** |

× BS=72 × 32 fires per step = **44 GFLOPs per step forward**.
With backward (~2-3×): **~120 GFLOPs total**.

That's **~200× cheaper than the DW-sep pyramid encoder** (which was ~3.15 TFLOPs
forward at N=256).

Compare to main's modulator compute: ~90 GFLOPs per step. Our attention modulator
is in the same ballpark. **No reason we shouldn't match main's throughput.**

### 3.4 Expected throughput

- LM work: ~80 TFLOPs effective per sec at BS=72 (measured from main: 40K tok/s
  × 128 tok/step × 9216 tok/step → 9216 tok / 0.24 s = 38K tok/s)
- Memory step (N=256, same as conv-grid): ~0.5 TFLOPs per step, scales fine at
  high arithmetic intensity via bmm.
- Attention modulator: ~0.1-0.2 TFLOPs per step.
- **Expected: 35-45K tok/s at BS=72 — roughly matching main.**

VRAM: attention memory is `BS × n_heads × N × N = 72 × 4 × 256² ≈ 19 MB per layer
in attention scores`. Tiny. No need for activation checkpointing.

### 3.5 What we preserve from conv-grid

- Single pool: N=256, one big W, permutation-equivariant over internal neurons.
- Per-edge observations: W[i,j], hebbian[i,j], asymmetry — now as attention biases.
- Per-node observations: h, msg, received — as token features.
- Role markers for port / internal distinction.
- Full-rank ΔW per event.
- Per-neuron Δdecay.
- bf16 everywhere + γ clamp to 0.97.
- Virtual I/O pools for LM interface.
- mem_pred_loss, surprise pipeline, TBPTT detach, Gumbel-softmax for phase 1
  (if we keep the codebook).

### 3.6 Open decision: keep the codebook or not?

**With codebook** (like conv-grid currently):
- Attention pools to a cell-level feature → code logits [K].
- Gumbel / Categorical sample code.
- Codebook → decoder (could reuse current conv-transpose but we'd have same
  slow-decoder problem). Or: decoder reads attention tokens directly +
  code-conditioning.
- Keeps GRPO compatibility.

**Without codebook**:
- Attention tokens directly drive ΔW_row and Δdecay emission.
- Simpler, faster, no Gumbel machinery.
- Breaks GRPO compatibility (no discrete action).

Given phase 2 GRPO is deferred behind the pretrained-LM pivot anyway, **drop the
codebook for this branch**. Phase-1-only training with direct-emission. If GRPO
comes back, codebook can be added later as an optional bottleneck.

### 3.7 Trade-offs to track

- **Attention is O(N²F) — still N².** We're not fixing the quadratic cost, just
  making the constant factor small (F ≪ C²). If N grows to 1024 this still bites.
- **Permutation equivariant** — this is a design benefit, but we need role
  embeddings to break symmetry where ports genuinely differ from internals.
- **No discrete bottleneck** — full-rank continuous emission. More expressive,
  but no information-bottleneck regularization. Whether this hurts or helps
  learning is empirical.

## 4. Implementation plan

### Step 1 — New module file

**`src/model/attention_modulator.py`** (replaces `grid_modulator.py`):

```python
class TokenBuilder(nn.Module):
    """Per-neuron feature projection into F-dim tokens."""
    # h_proj, msg_emit_proj, msg_recv_proj (D_n → d_proj, reuse from conv-grid)
    # role_emb (3 → role_dim)
    # token_mlp: concat(all features) → F

class AttentionLayer(nn.Module):
    """Multi-head self-attention with per-head edge bias."""
    # Q, K, V: Linear(F, F) — standard
    # out_proj: Linear(F, F)
    # norm: LayerNorm
    # ffn: Linear(F, 4F) → GELU → Linear(4F, F)
    # forward(tokens, edge_bias): apply attention + FFN with residuals

class EdgeBiasBuilder(nn.Module):
    """Build [BS, n_heads, N, N] from W, hebbian, asymmetry."""
    # mlp: Linear(3, n_heads) — extremely small MLP

class AttentionModulator(nn.Module):
    """Full pipeline: tokens → L attention layers → logits (if codebook)
    or direct output heads (if continuous)."""
    # If cfg.use_codebook: pool → logit_head → [BS, K]
    # Else: per-neuron output heads → (ΔW_row, Δdecay_raw)
```

### Step 2 — Update Config

**`src/model/config.py`**:
```python
# Attention modulator config
attn_token_dim: int = 64        # F
attn_n_heads: int = 4
attn_n_layers: int = 2
attn_ffn_mult: int = 4
use_codebook: bool = False       # direct emission by default

# Drop: conv_channels, conv_layers, conv_kernel, conv_groups, conv_dropout,
#       decoder_seed_spatial, decoder_seed_channels
```

### Step 3 — Update Memory

**`src/model/memory.py`**: swap `ConvGridModulator` + `ConvTransposeDecoder`
with `AttentionModulator`. The call site in `_modulate` becomes a single call
that returns `(ΔW_normed, Δdecay_raw)` directly.

### Step 4 — Drop checkpoint_memory default

At attention compute scale, we don't need activation checkpointing. Set
`checkpoint_memory: bool = False` (default). VRAM will be fine at BS=72.

### Step 5 — Test and benchmark

1. Smoke test (`tests/test_smoke.py`) on tier_tiny CPU path — must pass.
2. Smoke test at BS=8 CUDA — verify forward+backward runs, loss decreases.
3. Benchmark at BS=32 and BS=72.
4. **Target: ≥30K tok/s at BS=72.** If we don't hit this, investigate.

### Step 6 — (Optional) Add codebook back as bottleneck

If phase-1 results look good but we want the info bottleneck: add a
Gumbel-softmax over pool→K logits in parallel with the direct output head, and
let the downstream decoder read both (or gate between them).

### Step 7 — Update docs

- `docs/design_conv_modulator.md` → mark as superseded; keep for history.
- New doc: `docs/design_attention_modulator.md` with final architecture.

### Step 8 — Delete dead code

- `src/model/grid_modulator.py` — can go away once attention_modulator is in.

## 5. What "success" looks like

For this branch to be worth keeping:

1. **Speed ≥ 25K tok/s at BS=72** (a real ≥2×-main-matching-speed test).
2. **Trains stably**: loss decreases, no NaN, h/W/decay norms stay bounded.
3. **Matches or beats main on `mem_leverage_ce`** after 100M-300M tokens. Main
   verify_01 reached 1.19 nats leverage. If attention neuromod matches at
   comparable speed, keep and scale up. If it's worse, re-evaluate.

## 5a. Implementation notes and benchmark results

Implemented and benchmarked 2026-04-16. Summary:

- **22K tok/s at BS=72, T=128, N=256 on RTX 4090** (14× faster than conv-grid
  at 1.6K). Main branch is 40K for comparison.
- Memory module 7.3M params (vs conv-grid's 17M, main's 42M).
- Peak VRAM 8.5GB at BS=72 (vs main's 22GB).
- Attention modulator itself is tiny (300K params); decoder is 4.3M.
- All 4 regression tests pass on CPU tier_tiny.

### What's in the profile now

The `aten::mm / addmm / bmm` trio dominates CUDA time, as expected for an
all-tensor-core design. No conv, no layout conversions, no
`unsafe_index_put`. Profile looks like main's: clean bmm-heavy workload.

### Why we're at 22K instead of 40K (matching main)

The memory step at N=256 does 8× more W @ msg and hebbian work than main
(which has NC=8 × N=32). State/msg MLPs are SAME compute (same total neuron
count). Attention modulator is actually cheaper than main's per-cell MLP.

Net effect: memory-step compute is heavier but tensor cores handle it fine,
so we're compute-bound at ~2× main's step time. Additionally, activation
checkpointing is mandatory at BS=72 (without it we OOM at ~22GB) and adds
~1.5× backward cost.

### Sweeps performed

**Didn't help:**
- D_n 256 → 128 (tensor-core efficiency drops more than FLOPs saved)
- state_mlp_hidden 256 → 128 (same reason)
- N_total 256 → 384 / 512 (9K / 6K tok/s — N² cost dominates)
- D_n 256 → 512 (15K — tensor core doesn't like the bigger matmul shape)
- decoder_hidden 256 / 512 / 1024 (all same ~15-20K within noise)
- attn_token_dim 32 / 64 / 128 (all same within noise)
- num_codes 512 / 2048 / 4096 (no measurable difference)
- torch.compile mode=reduce-overhead, max-autotune (already compute-bound)

**DID help:**
- **Disable checkpoint_memory + BS=48 (fits 22.8 GB)**: 20.5K → 24.4K tok/s
  (+19%). Why: no forward-replay during backward. Tradeoff is smaller BS,
  but the 1.5× backward speedup wins.
- Direct emission decoder (vs rank-r factored): no speed change but
  removes a capability-limiting approximation.

### Final confirmed config

- BS=48, modulation_interval=4, checkpoint_memory=False
- N=256, D_n=256, attention F=64 × 4 heads × 2 layers
- K=2048, D_code=128, decoder_hidden=512, direct N²+N emission
- **24.4K tok/s steady state, 252ms/step, 22.8 GB peak VRAM**

### Why we can't easily beat this at NC=1

Profile: **GPU is 97% busy** (self CUDA time ≈ wall time). We're compute-
bound on `aten::mm` (state/msg MLP matmuls, already at peak tensor-core
throughput). Triton can't help — nothing to fuse that torch.compile
hasn't, and the dominant ops are already tensor-core-saturated.

The remaining 1.6× gap to main's 40K is **structural**: main's NC=8 × N=32
design has 8× fewer edges (8192 vs 65K), so W @ msg and hebbian work are
proportionally cheaper. To match main's speed at N=256 we'd need to go
multi-cell, which loses the "unified connectivity pool" of this design.

### What would beat main (future work, not implemented)

NC=8 cells with N_per_cell=32 (main's structure), BUT with an attention-
based **cross-cell modulator** observing all cell states at once. This:
- matches main's fast per-cell W structure (8192 edges total)
- adds cross-cell coordination via the attention modulator (capability win
  over main's independent per-cell modulators)
- preserves per-edge + per-node observation (within-cell)
- estimated ~35-40K tok/s

Substantial refactor (~1 day) but a clean architectural answer. Worth
revisiting if the current 24.4K training runs show mem_leverage_ce
comparable to main.

## 6. Honest risks

- **Attention might not learn the right thing.** Conv-grid's spatial processing
  wasn't tested long enough to know if the conv encoder had any learning
  advantage; replacing it with attention means trying a different architecture
  entirely. Could be better or worse.
- **Direct emission (no codebook) might be less sample-efficient.** Without the
  discrete bottleneck's capacity regularization, the modulator might overfit
  specific states.
- **`F=64` might be too small for real learning.** Easy to scale up if needed.

All of these are testable in the first few thousand training steps. None are
expensive to bail on.
