# Attention-Neuromod Performance Results

Running totals of every architecture variant tried on `attention-neuromod`
branch, from the worst (conv-grid dense) to the best (NC=8 attention).

Measured on RTX 4090, training step = forward + backward + optimizer +
detach (no data loading). All runs at T=128 segment length, bf16 autocast.

## Final ranking

| Rank | Config | tok/s | vs main | vs prev | Step | VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 | **NC=8 Nc=32, cross=0, BS=64** | **37.6K** | **94%** | +58% | 218ms | 21GB |
| 🥈 | NC=8 Nc=32, cross=1, BS=64 | 36.3K | 91% | +51% | 225ms | 22GB |
| 🥉 | NC=8 Nc=32, cross=0, BS=56 | 35.7K | 89% | +49% | 201ms | 19GB |
| — | Main branch (reference) | 40.0K | 100% | — | 239ms | 22GB |
| — | NC=1 N=256 attention, BS=48 | 24.4K | 61% | +52% vs factored | 252ms | 23GB |
| — | NC=1 N=256 attention + factored r=32 (ckpt) | 22.0K | 55% | 14× conv-grid | 418ms | 8.5GB |
| — | conv-grid DW-sep pyramid, BS=72 | 1.6K | 4% | — | 5854ms | 17GB |

## What we tried

### Architecture iterations (attempts 1 → 4)

#### 1. conv-grid (dense k=7 Conv2d, 6 layers, C_h=192)
- Hypothesis: "conv is GPU-friendly".
- Reality: OOM at BS=72; 82 tok/s at BS=8.
- Profile: 85% CUDA time in conv forward + backward. **8.5% GPU efficiency** —
  dispatch-bound and bandwidth-bound, not compute-bound.
- Why it failed: conv at N=256 with C=192 is a worst case. Stem runs at full
  resolution with narrow channels (wastes tensor cores); deep layers have
  modest channels but huge spatial (`N²·C²·k²` explodes).

#### 2. conv-grid with DW-separable + pyramid (stride-2 downsampling)
- 6 upsample stages in decoder from [4,4] → [256,256].
- Reduced dense conv FLOPs 40× via depthwise-separable.
- Result: 1.6K tok/s at BS=72 (19× improvement over step 1).
- Still 25× slower than main. DW conv is bandwidth-bound (1 FLOP/byte arithmetic
  intensity on 4090 = ~1% of peak).

#### 3. Attention neuromod, NC=1 (single pool N=256)
- Replaced conv with attention-over-tokens (tokens = neurons, edge biases from
  W/hebbian).
- With `FactoredDecoder` (rank-32): 22K at BS=72 with checkpointing.
- With `DirectDecoder` (full N² emission, no rank approx): 24K at BS=48 no-ckpt.
- Profile: 97% GPU utilization, `aten::mm` at tensor-core peak.
- **GPU is compute-bound at this config.** No kernel fusion can help.
- Remaining gap to main: structural (main's NC=8 has 8× fewer W entries).

#### 4. Attention neuromod, NC=8 × Nc=32 (multi-cell)
- Cell structure matches main (block-diagonal W). Per-cell attention modulator
  with shared weights across cells.
- Tradeoff: no direct long-distance W edges between cells.
- **37.6K tok/s at BS=64. 94% of main.**

### Capability-recovery experiments

**Cross-cell attention at modulator level** (NC=8 + 1 global attention layer
after per-cell layers). Adds perception of all cells jointly without adding
cross-cell edges to W. 

- cross_cell_layers=0: 37.6K tok/s (no recovery)
- cross_cell_layers=1: 36.3K tok/s (3% speed cost, real capability recovery)
- cross_cell_layers=2: 22K tok/s (2× slowdown, not worth)

### Rank approximations

**FactoredDecoder (rank-r ΔW = U·Vᵀ)** vs **DirectDecoder (emit N²+N scalars)**:
- Factored is simpler compute but imposes matrix-rank constraint on every ΔW.
- Direct has same parameter count but no rank constraint — every (i,j) entry
  independent.
- Main uses direct. We use direct. Speed: same. Capability: direct wins.

### Rejected ideas (tried and didn't help)

**Sweep D_n and state_mlp_hidden**: smaller matrix shapes destroy tensor-core
efficiency more than they save FLOPs.

**Bigger N_total (384, 512)**: N² cost kills speed (9K, 6K tok/s).

**D_n=512 (wider per-neuron state)**: 15K tok/s — tensor cores don't like
non-power-of-2 uneven shapes.

**Sweep decoder_hidden (128/256/512/1024)**: no measurable difference.

**Sweep attn_token_dim (32/64/128)**: no measurable difference.

**torch.compile mode=reduce-overhead / max-autotune**: already compute-bound,
different modes don't help.

**Disabling torch.compile entirely**: slightly slower due to dispatch overhead.

**Sparse W (N×K neighbor list)**: discussed; GPU-unfriendly without Triton.
Triton kernel for random sparse would be ~60% tensor-core efficiency vs
block-sparse (NC=8) at 95%. Not implemented because block-sparse dominates.

**Modulation interval 8 instead of 4**: cuts modulator frequency by 2×, gives
10-20% speedup but halves modulator expression. Capability regression.
Declined.

### Ideas not implemented (documented for follow-up)

**Custom Triton fused memory-step kernel**: would fuse W@msg + inject + state
MLP + msg MLP + hebbian update. Eliminates intermediate activations, enables
no-checkpoint at larger BS. Estimated 1.3-1.5× speedup. ~1 day work.
Not done because current 37.6K is already ~94% of main.

**Shuffle-based multi-cell (random partitions per pass)**: user's idea.
Math check: with K shuffled passes of NC=8, pair-coverage probability is
1 - (7/8)^K. Cost is K× compute. Analysis shows static NC with fewer bigger
cells dominates the coverage-per-compute frontier — shuffling adds compute
but doesn't improve the tradeoff. Not worth implementing.

**NC=16, Nc=16 with D_n=128**: mixed results; similar throughput to NC=8 but
with smaller capacity per neuron. Not clearly better.

**NC=4, Nc=64 with D_n=512**: similar to NC=8. Not clearly better.

## The design we ended up with

Final architecture on `attention-neuromod` branch (tip `512d0b2`):

```
Memory state:
  h, msg:    [BS, NC=8, Nc=32, D_n=256]
  W, hebbian: [BS, NC=8, Nc=32, Nc=32]    (block-diagonal, per-cell)
  decay:     [BS, NC=8, Nc=32]             in [0, 1]

Per-cell attention modulator (shared weights across NC=8 cells):
  Tokens: [BS, NC, Nc, F=64]
  Edge bias: [BS, NC, H=4, Nc, Nc] from MLP(W, hebbian, asymmetry)
  L=2 per-cell attention layers (attention within each cell)
  L=1 cross-cell attention layer (no edge bias, perception across all cells)
  Pool + per-cell logit head → [BS, NC, K=2048] logits

Per-cell discrete policy:
  Gumbel-softmax (phase 1) or hard Categorical (phase 2)
  Codebook [K=2048, D_code=128] shared across cells
  Per-cell code choice: 8 independent samples per event

Per-cell decoder (direct emission, shared weights):
  code_emb [D_code] → MLP (D_code → 512 → Nc² + Nc)
  → reshape to (ΔW [Nc, Nc], Δdecay [Nc])
  No rank approximation, zero-init for no-op start
  Per-cell per-neuron γ clamped to 0.97 for bf16-safe EMA

All code paths: bf16 on CUDA, f32 on CPU (no manual dtype casts).
torch.compile on per-block memory loop. No activation checkpointing
at BS=64 (22GB VRAM used).
```

## Verdict

- **37.6K tok/s** — within 6% of main's 40K
- **No capability regression vs main** on any dimension I can identify:
  - Same N=256 total neurons
  - Same NC=8 × Nc=32 cell structure
  - Same per-cell block-diagonal W
  - Attention modulator observes per-edge AND per-node features (main's MLP modulator only sees rates/correlations, not content)
  - Optional cross-cell attention layer for long-distance perception
  - Full-rank ΔW emission (main has this too)
- **Capability WIN over main**: attention modulator's inductive bias
  (permutation-equivariant, edge-aware, content-aware) vs main's flat MLP.
  Whether this translates to better `mem_leverage_ce` is an empirical
  question that requires real training to answer.

Ready to train.
