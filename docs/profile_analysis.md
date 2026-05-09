# Trajectory-Memory Performance Profile (2026-05-09)

Deep profile of the Phase 1 (Wave 1 long-doc TF NTP) training step at the
production config. Identifies what's unavoidable Llama cost vs what's
fixable in our code.

## Setup

- RTX 4090 24GB, bf16
- Config: medium tier (post-bump: N=4096, K=64, J=4, K_read=K_write=8, D=4,
  T_window=256, chunk=1024 tokens)
- BS=2, eager mode (compile gives same throughput at this BS)
- Trainable: 16.46M (61% bridge, 16% write, 15% read, 13% manifold)

## Whole-step breakdown

```
Step time:     241.8 ms/iter
Throughput:    8.47k tok/s
Peak VRAM:     11.19 GB
TFLOPS:        61.8 (37.5% of 4090 peak ~165 TFLOPS bf16)
```

## Component timing (CUDA events, best-of-5)

```
Per-window (chunk has D=4 windows + 1 backward):
  Llama forward + MemInjectLayer cross-attn        14.04 ms   (80% of fw_window)
  read_module only                                  3.16 ms   (18%)
  write_module only                                 3.69 ms   (21%)
  forward_window (full, no_grad)                   17.50 ms

  4× forward_window:                               70.02 ms   (29% of step)
  bw + opt step:                                  171.37 ms   (71% of step)
  Full step:                                      241.39 ms
```

**Backward + optimizer dominates the step at 71%.** This is typical for
fwd+bwd training (backward is ~2× forward in raw cost) but worth noting:
forward optimization gives diminishing returns because it's not the
dominant cost.

## FLOP accounting

Per forward pass through one chunk (D=4 windows × 1024 token-context each):

```
  Llama backbone:   4947.8 GFLOPS   (99.3% of total compute)
  Bridge MLP:         34.4 GFLOPS   (0.7%)
  read_module:         0.6 GFLOPS   (0.012%)
  write_module:        0.6 GFLOPS   (0.012%)
  TOTAL fwd:        4983.4 GFLOPS
```

**99.3% of compute is in Llama**, which is frozen and unavoidable. Even
making the entire memory module 100× faster would yield <0.7% speedup.

We achieve 61.8 TFLOPS at 37.5% of peak. Room to grow but not bad —
typical training workloads land at 30-50% of peak on consumer GPUs.

## Top CUDA kernels (torch.profiler)

```
aten::mm                              285.6 ms (62.6% CUDA time)  4148 calls
aten::copy_                            45.8 ms (10.0%)             2064 calls
ampere_bf16_s1688gemm various         165   ms (36%)               various
aten::to (dtype conversions)           22.0 ms (4.8%)              1840 calls
aten::slice_backward                   19.7 ms (4.3%)              278  calls
```

**Two flags worth investigating:**
1. **2064 `aten::copy_` calls (45.8 ms = 10% of step)** — high call count
   for one operation suggests dispatch overhead from many small copies.
2. **1840 `aten::to` calls (22 ms)** — dtype conversions dominate
   non-matmul time. Likely `current_hiddens.to(prev_states.dtype)` per
   window plus various detach/clone paths between TBPTT chunks.

## What's unavoidable

**The Llama backbone forward + backward** (240 ms × 99% ≈ 238 ms equivalent).

Even though only 9.4M params inside Llama are trainable (the
MemInjectLayer bridge), backward must propagate activation gradients
through all 16 layers to reach those params and the read/write modules
upstream. PyTorch can skip parameter gradient *accumulation* for frozen
params, but cannot skip activation gradient *computation* unless the
trainable params lie strictly on one side of the layer.

In our architecture, the trainable bridge sits at layer 8 with read/write
modules feeding it from below. So backward through layers 0-7 (to reach
read_module's gradient) and through layers 9-15 (to reach the lm_head and
back to the bridge) are both required.

## What's fixable (ranked by impact)

### 1. Reduce per-window data movement (~10-20 ms savings, 5-8% speedup)

The 2064 copies + 1840 dtype conversions add up to ~67 ms total. Sources
to audit:
- `prev_window_hiddens.to(prev_states.dtype)` per window (bf16↔fp32)
- `current_hiddens.to(prev_states.dtype)` per window
- `prev_states.detach()` between TBPTT chunks
- The closure capture in `_build_memory_fn` reshaping `read_visited`
- Slicing `full_logits[:, -T_window:]` and `full_hiddens[:, -T_window:]`
  per window — `slice_backward` shows up at 19.7 ms

Fix: keep `prev_states` in bf16 (matching Llama) so the per-window
conversions go away. The "fp32 manifold" rule in TrajMemConfig was for
weight updates (Adam needs fp32 for small-step accuracy), but the
forward computation can run in bf16 if we cast back for the optimizer.

### 2. Share d_lm↔D_concept projections across modules (~5-10 ms, 3-5% speedup)

Currently three independent learned projections between d_lm-space and
D_concept-space:
- `MemInjectLayer.W_in` (in bridge): d_lm → bridge_hidden → D_concept
- `read_module.cross_attn` K/V projections: d_lm → D_concept (1.18M
  params, 8% of all trainable)
- `write_module.cross_attn` K/V projections: d_lm → D_concept (1.18M
  params)

These run independently every window, each doing the same shape of
projection on related tensors. Sharing weights (or sharing the
pre-projected `MemInjectLayer.W_in(hidden)` output) saves both
parameters and per-window compute.

This is the architectural cleanup we discussed earlier (re-allocate
params from translation to mechanism). Speed benefit comes for free
alongside the param savings.

### 3. Compile read+write modules separately (~5-10 ms, 3-5% speedup)

`bench_trajmem.py --compile` compiles `forward_window`, which includes
Llama. Compile catches some operator fusion in the per-window bridge but
leaves read/write module kernels uncompiled if they're called from
non-traced code paths.

Adding `model.read_module = torch.compile(model.read_module)` and similar
for write_module forces inductor to fuse the small attention + MLP ops in
those modules into single kernels. Especially useful at low BS where
dispatch overhead is significant.

### 4. Gradient checkpointing on Llama backbone (no speed; saves memory)

Not a speed win — actually slightly slower (recomputes forward during
backward). Worth mentioning because it would let us push BS higher.
Currently OOM at BS=8; with gradient checkpointing we might fit BS=16 at
similar tok/s. Useful for the large config later.

### 5. Optimizer step: AdamW fused (~2 ms savings, 1%)

`build_optimizer` already passes `fused=True` (verified). No change here,
just confirming we're already on the fast path.

## Things NOT worth changing

- **Llama itself**: frozen, can't shrink, already using bf16 + Flash
  Attention via transformers ≥4.x. The 99% FLOP share is fundamental.
- **Manifold scatter_mean writes**: 0.6 GFLOPS / step, negligible. Even
  if it ran on CPU it would be invisible in the timing.
- **More aggressive compile modes** (`reduce-overhead`, `max-autotune`):
  reduce-overhead requires fixed shapes (breaks if any tensor varies);
  max-autotune costs minutes of compile time for marginal gains at this
  scale.

## Recommended optimization order

1. **Audit data-movement (item 1).** Highest expected return, no
   architecture change. Mostly removing redundant `.to()` and `.detach()`
   calls. Estimated 5-8% speedup.

2. **Compile read+write modules (item 3).** Trivial code change (1-2
   lines), modest win. Combine with #1 since they touch related code paths.

3. **Share d_lm projections (item 2).** Bigger refactor — touches read/
   write/bridge module structure — but addresses a real architectural
   issue (params over-allocated to translation) AND gives speed back.
   Worth doing alongside the param-allocation rebalance we discussed
   earlier.

Items 1+2+3 combined should land us at ~10-15% step speedup, which is
real but not transformative. The real ceiling is the 99% Llama backbone
share — to break through that we'd need either a smaller backbone (out
of scope) or a fundamentally different architecture (out of scope).

## Reproducing

```
PYTHONPATH=. python scripts/deep_profile_trajmem.py --bs 2
PYTHONPATH=. python scripts/deep_profile_trajmem.py --bs 2 --compile
```
