# Trajectory-Memory Performance Profile (2026-05-09)

> **SUPERSEDED 2026-05-10 — KV cache landed.** The headline finding
> below ("Llama re-encode is 70% of the slowdown") was correct
> diagnostically but the fix shipped: sliding KV cache eliminated the
> rolling-buffer re-encode entirely. Phase 1 went from 9.9k tok/s
> (here) to **17.7k tok/s** (current). Trajmem now beats vanilla
> per-token. See `docs/bench_results.md` 2026-05-10 PM section for
> current numbers. This doc is retained for the Llama-bound vs
> memory-bound decomposition methodology, which is still useful.

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

## What landed (2026-05-09)

**Wired `--compile` into `train_wave1.py` + `train_wave2.py`.** This was
the actual low-hanging fruit — the bench had already shown 28% speedup
at BS=2 from `torch.compile(model.forward_window)`, but that win was
unreachable from the trainer entrypoints. Now they pass `--compile` to
flip it on. ~2 min cold-start per first step; pays for itself within
~10 steps.

## What was attempted but deferred

**Audit data-movement (item 1) — turned out NOT to be 1-line fixes.**

The explicit `prev_window_hiddens.to(prev_states.dtype)` casts in
`forward_window` are *structurally* needed: read_module and write_module
both contain `cat([pooled_hiddens, head_query, ...])` operations, and
`head_query` is an `nn.Parameter` stored in fp32. PyTorch's `cat` with
mixed dtypes promotes to fp32 (it's in autocast's "promote" list), so
cat operands must already be same dtype.

To remove the casts we'd need to either:
- Refactor read/write modules to avoid the cat (e.g., split into two
  parallel projections + add), or
- Explicitly cast `head_query` per call to match input dtype

Neither is a 1-line change. They're worth doing alongside the broader
"share d_lm↔D_concept projections" architectural cleanup, not as a
standalone optimization.

**Autocast(bf16) wrapping — blocked on scatter_mean dtype.**

Tried `with torch.autocast("cuda", dtype=torch.bfloat16): trainer.step_wave1(chunk)`.
Hits `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`
inside `Manifold.write` — the scatter destination buffer is fp32
(per the manifold-fp32 design rule) but autocast made the source bf16.

Fixing requires either:
- Changing the manifold scatter to accept mixed dtypes (with explicit
  cast), or
- Keeping `concept_states` buffer in bf16 (drops fp32 accuracy benefit)

Both are real architectural decisions, not low-hanging. Documented for
future work.

**Compile read/write modules separately — already happens implicitly.**

`torch.compile(model.forward_window)` traces *through* `forward_window`,
which calls `read_module(...)` and `write_module(...)` directly. Inductor
already fuses kernels inside those modules as part of the traced graph.
Adding separate `torch.compile()` wrappers would either be redundant or
fight with the outer compile. Verified empirically — top kernels in the
profile are tensor-core gemms, which is what we'd want from inductor.

## Realistic ceiling

The 99% Llama-share ceiling stands. Items that genuinely could help
beyond the compile flag we just landed:

1. **Share d_lm↔D_concept projections** — saves both params and
   per-window compute. Real refactor (~1 day work). Pairs with the
   param-allocation rebalancing discussion. Estimated 3-5%.
2. **Manifold dtype refactor** — make scatter_mean dtype-agnostic, then
   wrap forward in autocast(bf16) for kernel-level wins. Real work but
   self-contained. Estimated 5-10%.
3. **Gradient checkpointing on Llama** — saves memory not speed; would
   let us push BS=8+, useful at large config later.

Items #1 and #2 combined would land another ~10% on top of the compile
win, but they're real engineering, not 1-line audits. Worth doing only
if the first training run shows we need more headroom.

---

# Update — 2026-05-09 (afternoon): Decomposing the 1.72× T1 vs V1.B gap

A second profiling pass with a "where does the slowdown actually live?"
question. Three new experiment scripts:

- `scripts/bench/experiment_memory_cost.py` — compares 5 paths (vanilla single-fwd,
  vanilla 4×growing-fwd, trajmem-no-memory, trajmem-bridge-only, full
  trajmem) at BS=4 to isolate memory module cost from per-window structure
  cost.
- `scripts/bench/experiment_lm_context.py` — sweeps `effective_lm_context`
  (256, 1024, 2048) × eager/compile to isolate rolling-buffer cost.
- `scripts/bench/experiment_compile_dynamic.py` — `torch.compile`'s `dynamic`
  flag and `mode='reduce-overhead'` at the production setting.

## Headline finding: **the memory module per se is essentially free**

| Path | tok/s | step ms | Notes |
|------|------:|--------:|-------|
| trajmem ec=256 eager | **19.0k** | 215 | Each window only sees its own 256 tokens; no rolling buffer |
| trajmem ec=256 compile | **23.4k** | 175 | Compile gives 23% at this shape |
| vanilla V1.B (T=1024 single fwd) | 17.0k | 237 | Reference baseline |
| trajmem ec=1024 eager | 9.9k | 414 | Rolling buffer growing to 1024 |
| trajmem ec=2048 eager | 9.9k | 414 | Same as ec=1024 (single chunk doesn't fill 2K) |
| trajmem ec=2048 compile dynamic=True | 11.3k | 363 | Avoids recompile thrashing |

**The trajmem trainer at ec=256 (no rolling buffer) is FASTER than vanilla
V1.B**: 19.0k vs 17.0k tok/s. The memory module's read/write/bridge ops
are net cheap because they let us run Llama at smaller per-window T,
which more than offsets their direct cost.

## What the rolling buffer actually costs

The `effective_lm_context = 2048` cap (the load-bearing knob in plan §4.1)
forces Llama to re-encode the rolling LM context buffer at every window:

- ec=256: per-window LM input is fixed at 256 → step 215 ms
- ec=1024: per-window LM input grows 256 → 1024 → step 414 ms (+92%)

That ~200 ms gap is **all** Llama re-encoding the prefix per window.
The memory module didn't get any more expensive between these two
configs — Llama got more expensive.

## Decomposition (`experiment_memory_cost.py`, BS=4):

| Path | step ms | Δ vs prior |
|------|--------:|----:|
| P0 vanilla single-fwd T=1024 (= V1.B) | 237 | — |
| P2 trajmem with memory bypassed (scale=0, read/write skipped) | 360 | +123 ms (per-window LM forward shape cost) |
| P3 trajmem with bridge cross-attn active (KV=zeros) | 360 | +0 ms (bridge cross-attn is noise) |
| P4 full trajmem (= T1) | 414 | +54 ms (read+write trajectory hops) |

So of the 177 ms gap between V1.B (237) and T1 (414):
- **123 ms (69%)** = running Llama 4 times with growing context instead
  of once at T=1024
- **54 ms (30%)** = read+write trajectory hops + scatter_mean
- **<1 ms (~0%)** = MemInjectLayer bridge cross-attn

The memory module accounts for ~30% of the gap, the rolling-buffer
structure accounts for ~70%.

## What this changes about the optimization plan

The original profile (above) said: 99% of FLOPs are in Llama, can't
optimize Llama. Still true. But the breakdown above tells us **the
optimization target is "reduce how much Llama runs"** — not "reduce
how much memory module runs". Three concrete lines:

### Tier 1 (architectural, 30-50% potential)

**Sliding KV cache for Llama.** The dominant lever. Each window today
re-encodes the full rolling buffer (256 → 2048 tokens by chunk 2). With
KV cache, each window's forward becomes ≈ ec=256 cost (~215 ms) +
small per-window cache append work. Approaches the vanilla speed ceiling.

Architectural complication: MemInjectLayer at layer 8 changes per window
(memory readout differs). Layers 0-7 KV cache is reusable; layers 8+
need to recompute the KV for cached positions because their input (h_inj
= h + scale·W_out(memory)) differs per window. Math is straightforward;
implementation is ~1-2 days of careful code in `forward_window`.

### Tier 2 (kept-promise compile, ~14% steady-state)

**`torch.compile(..., dynamic=True)`** — landed in this update, in both
`train_wave1.py` and `train_wave2.py`. The previous `dynamic=False`
recompiled per LM-input shape; with rolling-buffer growth across chunks,
a typical training run sees 8+ distinct shapes and hits `recompile_limit=8`
within a few chunks. `dynamic=True` compiles a single graph that
handles varying shapes — slightly less optimized per shape but stable
across long training runs.

Single-chunk synthetic bench (the only kind we currently have) doesn't
trigger the recompile thrashing because it only sees 4 shapes. Real
training will.

### Tier 3 (smaller, refactor-required)

- **Pre-allocate trajectory buffers** in read/write modules instead of
  `visited_list.append(...)` etc. The Python list-append pattern triggers
  CPU/GPU sync points → blocks `mode='reduce-overhead'` cudagraph (we
  observed 90+ "cudagraph partition due to non gpu ops" splits).
  Estimated 5-15% additional with cudagraph fully enabled.
- **Share d_lm↔D_concept projections** across MemInjectLayer / read-attn
  / write-attn (3 separate copies today). Estimated 3-5%.
- **autocast(bf16)** with manifold dtype refactor. Estimated 5-10%.
- **Gradient checkpointing** on Llama backbone — no speed; lets us push
  BS≥8 if gradient quality matters more than wall-clock.

## Recommendation

Launch Wave 1 with `--compile` (now `dynamic=True`) at the current speed
(~11.5k tok/s eager / ~13k+ compile expected). Watch the loss curve for
~1-2k steps. Only invest in KV cache if the loss curve is healthy AND
the wall-clock cost actually matters for the run length we want.

The memory graph itself has been verified-cheap. Further code-level
optimization on it has small ceilings; the architectural lever is
KV cache.

## Reproducing

```
# Original profiling
PYTHONPATH=. python scripts/bench/deep_profile_trajmem.py --bs 4

# New decomposition experiments
PYTHONPATH=. python scripts/bench/experiment_memory_cost.py
PYTHONPATH=. python scripts/bench/experiment_lm_context.py
PYTHONPATH=. python scripts/bench/experiment_compile_dynamic.py
```

---

## Original profile (BS=2 numbers, retained for history)

## Reproducing

```
PYTHONPATH=. python scripts/bench/deep_profile_trajmem.py --bs 2
PYTHONPATH=. python scripts/bench/deep_profile_trajmem.py --bs 2 --compile
```
