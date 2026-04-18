# Attention-Neuromod Performance Results

Running totals of every architecture variant tried on `main`
branch, from the worst (conv-grid dense) to the best (NC=8 attention).

Measured on RTX 4090, training step = forward + backward + optimizer +
detach (no data loading). All runs at T=128 segment length, bf16 autocast.

## Final ranking

| Rank | Config | tok/s | vs main | vs prev | Step | VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 | **NC=8 N=32, cross=0, BS=64** | **37.6K** | **94%** | +58% | 218ms | 21GB |
| 🥈 | NC=8 N=32, cross=1, BS=64 | 36.3K | 91% | +51% | 225ms | 22GB |
| 🥉 | NC=8 N=32, cross=0, BS=56 | 35.7K | 89% | +49% | 201ms | 19GB |
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

#### 4. Attention neuromod, NC=8 × N=32 (multi-cell)
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

**NC=16, N=16 with D_n=128**: mixed results; similar throughput to NC=8 but
with smaller capacity per neuron. Not clearly better.

**NC=4, N=64 with D_n=512**: similar to NC=8. Not clearly better.

## The pre-Triton milestone (historical; superseded by the Redesign below)

The "shared-weights attention modulator" milestone on the old
`attention-neuromod` branch hit **37.6K tok/s** at BS=64 (within 6% of
the old main's 40K). That intermediate milestone had F=64, H=4, 2
per-cell attention layers + 1 cross-cell attention layer, no Triton
fusion. It is documented for history; the *current* design below
supersedes it in every dimension (wider F, Triton-fused per-token
kernel, multi-timescale LIF, dropped cross-cell attention).

## Redesign (April 2026) — multi-timescale + per-cell + Triton

Everything below supersedes the shared-weights milestone above. See
`design.md` for the canonical description of the current architecture.

### Changes

1. **Per-token state update simplified to LIF** —
   `h = tanh(decay·h + (1−decay)·received)`. No state MLP; neurons are
   leaky integrators with a tanh threshold. Matches biological soma more
   accurately.
2. **Multi-timescale event schedule** —
   - Per-token: W @ msg + inject + LIF state update + readout (fast)
   - Every 4 tokens: msg = MLP(h), hebbian EMA update (event)
   - Every 16 tokens: neuromodulator → ΔW, Δdecay (slow)
3. **Triton-fused per-token kernel** — forward+analytical-backward
   `torch.autograd.Function` fusing W@msg + inject add + LIF + readout
   pool. 4.2× faster per-token in isolation; ~5-10% end-to-end gain after
   integration.
4. **Readout pools from `h`**, not `msg` (fresh every token).
5. **Shared modulator trunk + per-cell cell_emb** — attention, edge-bias
   MLP, logit head, and projections are SHARED across cells (batched over
   NC as an extra batch dim). The only per-cell parameter is
   `cell_emb [NC, d_cell=16]`, which is concatenated into per-neuron
   tokens and into the decoder input (FiLM-style conditioning). Same
   code ID can mean different plasticity programs in different cells,
   driven by the tiny `cell_emb` signal rather than duplicated weights.
   A per-cell-weights variant was tried separately and cost ~15%
   throughput for modest capacity gain — not kept.
6. **Shared-trunk decoder** — one MLP, conditioned on cell_emb at input.
   Same compute path regardless of cell count.
7. **Beefed modulator** — `attn_token_dim 64→128`, `attn_n_layers 2→3`,
   `attn_n_heads 4→8`, `d_proj 16→24`. ~3× modulator param count while
   staying shared-trunk.
8. **`tbptt_block` default 8→16.**

### Throughput progression (RTX 4090, BS=64, T=128, tbptt=16)

| Stage | tok/s | vs baseline | VRAM | Params (mem graph) |
|---|---:|---:|---:|---:|
| Shared-weights baseline | 37.8K | 1.00× | 21 GB | 3.6 M |
| + multi-timescale LIF | 67.3K | 1.78× | 10.4 GB | 3.4 M |
| + Triton fused kernel | 69.8K | 1.85× | 10.0 GB | 3.4 M |
| + beefed modulator (stayed shared trunk, **current**) | **~68K** | **~1.80×** | 12 GB | 4.1 M |

Total model: **71.2M params** (LM 67.1M + memory graph 4.1M).

"Beefed" in the current design is `attn_token_dim 64→128`, `n_layers 2→3`,
`n_heads 4→8`, `d_proj 16→24` — the modulator went 3× wider/deeper but
stayed a SHARED trunk across cells (not per-cell weights, which was tried
in a separate experiment that added ~5M params for ~15% throughput cost
and was not kept).

### At other batch sizes (current config)

| BS | tok/s | ms/step | VRAM |
|---:|---:|---:|---:|
| 64 | ~68K | ~120 | 11.5 GB |

### Knobs we deliberately did NOT change (for future reference)

- **Did NOT add more codes** (user preference — 2048 was felt adequate).
- **Did NOT widen decoder** (separate from widening modulator).
- **Did NOT drop hebbian** (though the modulator also sees W directly; this
  is a candidate optimization if VRAM gets tight).
- **Did NOT parallelize time dim** (incompatible with autoregressive
  inference / GRPO rollouts).
- **Did NOT touch LM** (stays at 67.1M, d_inner=1200, 4 scan layers).

### Open / future work

- Full Triton backward (current: analytical PyTorch backward). Would
  unlock the ~4× per-token speedup on the backward pass too.
- CUDA graph capture of `_run_block` to eliminate residual Python dispatch.
- Attach to a pretrained Llama (e.g. 3B) and measure autoregressive
  inference overhead. Memory graph's per-token cost should be negligible
  vs Llama's forward.
- GRPO training of the modulator policy (phase 2 in `discrete_policy`).

Ready to train.
