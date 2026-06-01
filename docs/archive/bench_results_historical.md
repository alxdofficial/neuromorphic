# Bench Results — Llama-3.2-1B + trajectory-memory

Tracks the headline throughput / VRAM numbers for the integrated model.
Each section dates the run and pins the configuration so future-me can
spot regressions or progress without re-running every time.

Vanilla Llama path numbers (forward / lm_head step / full step) are
**not re-run here** — they're hardware-dependent only and unchanged from
graph_walker's measurements (see `abandoned/graph-walker` →
`docs/bench_results.md`). We only bench paths that depend on
trajectory-memory specifically.

Per project memory:
- "Bench with fixed params, never sweep" — one config tier per run.
- "Bench at each path's own optimal BS" — peak-throughput BS per setting,
  not forced to share with other modes.

## Training-type labels used throughout

| label | meaning | trainer entry | code path |
|---|---|---|---|
| **TF-NTP** | Teacher-Forced Next-Token-Prediction (Wave 1 / Wave 2 SFT) | `Phase1Trainer.step_wave1` / `step_wave2` | `src/trajectory_memory/training/phase1.py` |
| **AR-GRPO** | Autoregressive GRPO with reward (Wave 3 / Wave 4 / MT-GRPO) | `Phase2Trainer.step` / `step_batched` | `src/trajectory_memory/training/phase2.py` |

Every bench section below is tagged with `[TF-NTP]` or `[AR-GRPO]` (or `[both]` for cross-cutting comparisons).

## Per-bench metadata convention

Every dated section MUST include:
- Date in UTC (`YYYY-MM-DDTHH:MMZ`)
- Git commit hash at time of run
- Hardware (GPU model + total VRAM)
- Base config (`TrajMemConfig.<tier>()` + explicit overrides)
- Bench scripts + raw output log paths
- Training-type label

Bench scripts live in `scripts/bench/`. Run benches via:
- `scripts/bench/bench_trajmem.py` — TF-NTP throughput sweep (default sweeps BS)
- `scripts/bench/bench_dconcept.py` — TF-NTP D scaling at fixed BS
- `scripts/bench/bench_dconcept_n.py` — TF-NTP D × N joint sweep
- `scripts/bench/profile_grpo.py` — AR-GRPO per-op profile with torch.profiler

---

## Index of dated entries

| date | training type | summary |
|---|---|---|
| 2026-05-13 (late PM) | [AR-GRPO] | **Negative result:** StaticCache + CUDA-graph A/B vs DynamicCache |
| 2026-05-13 (PM) | [AR-GRPO] | BS_outer ∈ {1,2,4,8} scaling at kl_coef=0, D=1024 default |
| 2026-05-13 (AM) | [TF-NTP] + [AR-GRPO] | Architectural scaling (D, N) + GRPO GPU-util profile |
| 2026-05-11 | [TF-NTP] + [AR-GRPO] | Phase 1 + Phase 2 after architectural perf push (closest pre-Gumbel-fix reference) |

Older entries (2026-05-09, 2026-05-10) deleted on 2026-05-13: the
architecture has changed substantially since (Gumbel→softmax-STE
routing, KV-cache cap enforcement, head_query std fix, logit_scale,
dynamic compile, D_concept default 256→1024). Old throughput numbers
aren't directly comparable to current code. See git log of this file
for history.

---

## 2026-05-13 (late PM) — StaticCache + CUDA-graph A/B (negative result)

**Training type:** [AR-GRPO]

**Run metadata.**

| field | value |
|---|---|
| Date (UTC) | 2026-05-13T05:30Z |
| Commit | `dcc61d4` + uncommitted Phase 2 StaticCache + CUDA-graph paths |
| Hardware | NVIDIA RTX 4090, 24564 MiB |
| Base config | `TrajMemConfig.medium()` (D=1024, N=4096, J=4, K_read=K_write=8, T_window=256, effective_lm_context=2048) |
| Model | Llama-3.2-1B (frozen, bf16) + cold-start trajectory memory |
| GRPO settings | K=8 rollouts, max_new_tokens=256, kl_coef=0, K_per_window=256 (one window per AR call) |
| Bench script | `scripts/bench/bench_grpo_cuda_graph.py` |
| Raw log | `outputs/bench_grpo_cuda_graph.log` |

### Setup

A/B'd three cache configurations for Phase 2's `_ar_sample_outer_batch`
inner per-token loop:

1. **`dyn_cache`** — current default. HF DynamicCache that grows from
   `cap=2048` to `cap+max_new=2304` over 256 AR tokens, then gets
   trimmed back to `cap` at end of window.
2. **`static_cache`** — HF StaticCache pre-allocated at
   `[M*K, n_kv_heads, 2304, head_dim]` once, then slot-filled from the
   M length-bucketed prefill caches. No per-window trim (cache stays at
   2304 by design).
3. **`static_cache+cg`** — same as (2) plus `torch.cuda.graph()` capture
   of the per-token Llama forward after a 2-token warmup. Captures once
   per AR window, replays for the remaining ~252 iterations.

### Numbers

| BS_outer | mode | warm s | s/step | tok/s | peak GB | per-sample s | Δ vs dyn_cache |
|---|---|---|---|---|---|---|---|
| 1 | dyn_cache | 6.43 | 5.82 | 352 | 5.51 | 0.728 | — |
| 1 | static_cache | 5.79 | 5.80 | 353 | 5.51 | 0.725 | ≈ (BS=1 uses `step()` not `step_batched()`, flags unused) |
| 1 | static_cache+cg | 5.89 | 5.88 | 349 | 5.51 | 0.734 | ≈ (same) |
| **4** | **dyn_cache** | **18.31** | **18.18** | **451** | **10.88** | **0.568** | — |
| 4 | static_cache | 24.33 | 24.34 | 337 | 12.72 | 0.761 | **−34% throughput** |
| 4 | static_cache+cg | 24.35 | 24.47 | 335 | 12.73 | 0.765 | −34% (CG didn't recover) |

(BS_outer=8 row stopped early — the decisive trend at BS_outer=4 was
already conclusive; no reason to expect a flip at BS_outer=8.)

### Diagnosis

DynamicCache wins because HF Llama-3.2-1B's attention path scales the
attention compute with the cache's actual tensor length — which for
DynamicCache grows incrementally and averages ~2176 positions over the
window, while StaticCache always exposes the full pre-allocated 2304
positions. The 2304 isn't padding "skipped by the kernel"; it's actual
key/value memory the attention QK^T and softmax run over (the causal
mask zeros out future positions only).

The ~5-10% extra attention work per token, compounded over 16 layers
× 256 tokens × M*K=32 parallel rollouts, plus larger memory-bandwidth
traffic on the bigger buffer, is the gap.

CUDA-graph capture cuts the per-iter kernel-launch overhead (~10-20µs
per launch × ~160 kernels per layer per token), saving ~5-10% of step
time at most — not enough to make up the 34% StaticCache compute tax.

### Decision

- Reverted defaults: `use_static_cache=False`, `use_cuda_graph=False`.
- Kept the code paths (both flags exist on `Phase2Trainer`) for future
  experiments — e.g. on a different backbone, or paired with a sliding
  attention mask that lets StaticCache also bound its compute to
  `min(seq_filled, cap)`.
- **DynamicCache remains the production path.** No regression in
  current BS_outer=8 numbers from earlier today (472 tok/s holds).

### Lessons

- StaticCache only pays off when (a) kernel-launch overhead dominates
  compute (typical at very small models / very long decode steps / very
  high BS), or (b) the model's attention path can mask its compute to
  the actual filled length. Llama-3.2-1B with HF eager/SDPA at our
  config is neither.
- `torch.cuda.graph()` capture is fast (~one forward pass), so the
  experiment was cheap to run — but the engineering effort (the
  StaticCache rewiring in `_ar_sample_outer_batch`, slot-fill helper,
  capture/replay scaffolding) was wasted on this backbone.
- A real next lever: **decouple rollout from training** by running
  vLLM in a separate process for AR generation, then loading the
  rolled samples back for Pass-2 grad-replay. That's how TRL/OpenRLHF
  get their 5-10× throughput — not from cache choices.

---

## 2026-05-13 (PM) — BS_outer scaling at the new D=1024 default

**Training type:** [AR-GRPO]

**Run metadata.**

| field | value |
|---|---|
| Date (UTC) | 2026-05-13T04:10Z |
| Commit | `dcc61d4` + uncommitted (D=1024 default, kl_coef=0 default, softmax_top1_ste assertion relaxed) |
| Hardware | NVIDIA RTX 4090, 24564 MiB |
| Base config | `TrajMemConfig.medium()` (N=4096, **D_concept=1024**, J=4, K_read=K_write=8, T_window=256, D=4, d_lm=2048, inject_layer=8, effective_lm_context=2048) |
| Model | Llama-3.2-1B (frozen, bf16) + trajectory-memory side-car. **Cold-start** of trajectory memory (Wave 2 ckpt was trained at D=256, shape-incompatible; Llama weights still loaded.) |
| GRPO settings | K=8 rollouts, max_new_tokens=256, clip_eps=0.2, **kl_coef=0** (no reference policy), temperature=1.0 |
| Reward | `bert_cosine` (SentenceBERT cosine) via narrativeqa prompts ≥ 2048 tokens |
| Bench script | `scripts/bench/bench_grpo_bsouter.py` (1 warmup + 3 timed steps per BS_outer, parallel `nvidia-smi` poll @ 100ms) |
| Raw log | `outputs/bench_grpo_bsouter.log` |

### Findings

**KV cache dtype confirmed bf16** (HF DynamicCache inherits Llama's dtype — `cache.layers[0].keys.dtype == torch.bfloat16`).

**Per-step throughput**:

| BS_outer M | s/step | rollout tok/s | peak VRAM | GPU util mean | Δ vs M=1 | per-sample s |
|---|---|---|---|---|---|---|
| 1 | 5.80 | 353 | 5.51 GB | 70% | 1.00× | 0.725 |
| 2 | 9.82 | 417 | 7.09 GB | 63% | 1.18× | 0.614 |
| 4 | 18.26 | 449 | 10.88 GB | 61% | 1.27× | 0.571 |
| 8 | 34.74 | 472 | 18.48 GB | 60% | **1.34×** | **0.543** |

Per-sample wall time drops 25% from M=1→8 (0.725 → 0.543 s/sample), but
the absolute throughput gain is only +34% — sub-linear. **Higher BS
gives smaller marginal speedup**, not bigger; the curve has plateaued
hard by M=8.

**Bottleneck.** GPU util sits at 60–70% across all M and *drops* as M
grows — meaning compute throughput is not the limit. The remaining 30–40%
idle is the AR per-token decode loop: 256 sequential calls to
`llama.model(input_ids=[BS,1], past_key_values=..., cache_position=...)`,
each launching ~16 layers × ~10 ops = ~160 kernels. At ~10–20 µs/launch
the kernel-launch tax is 1.6–3.2 ms/token regardless of BS, and Llama-1B
compute per token at BS≤8 finishes faster than that. Batch-merging
doesn't fix it; CUDA graphs or static-cache compilation would.

**Implications for Wave 3 GRPO training.**
- BS_outer=8 is the new operating point (50% more rollouts per
  optimizer step than BS=4 at +70% VRAM, +25% per-sample time).
- VRAM headroom at 18.5 GB peak / 24 GB cap → no room for a reference
  policy (would need ~2.5 GB extra) → kl_coef=0 default is load-bearing.
- Activation checkpointing remains deferred — no compute regime where it
  pays off when the GPU is already 30%+ idle on kernel-launch overhead.

**CUDA graphs — deferred.** The decode loop is shape-stable *after* the
KV cache reaches `effective_lm_context=2048`, but `DynamicCache` resizes
its internal tensors as it grows; capturing requires switching to
`StaticCache` with pre-allocated `[BS_outer*K, n_kv_heads, 2048, head_dim]`
buffers in the AR path, then `torch.cuda.graph()` or
`make_graphed_callables`. Estimated 2–3 hours engineering + bench;
projected gain (per HF Llama generation benchmarks) is 1.5–2× on the
AR loop alone, which is ~70% of step time, so ~1.3–1.6× end-to-end.
Worth doing once the architecture stabilizes; today's BS_outer=8 already
moves Wave 3 epoch time meaningfully and the protocol may evolve again
before this becomes the next bottleneck.

---

## 2026-05-13 — Architectural scaling + AR-GRPO utilization bench

**Training type:** [TF-NTP] (scaling) + [AR-GRPO] (utilization + profile)

**Run metadata.**

| field | value |
|---|---|
| Date (UTC) | 2026-05-13T03:30Z |
| Commit | `dcc61d4` (post-Gumbel-fix, KV-cache-cap, dynamic-compile-on-mem_inject; before today's D=1024 default bump) |
| Hardware | NVIDIA RTX 4090, 24564 MiB |
| Base config | `TrajMemConfig.medium()` (N=4096, J=4, K_read=K_write=8, T_window=256, D=4, d_lm=2048, inject_layer=8, effective_lm_context=2048). At bench time `D_concept` default was 256; D variants are explicit per-row. |
| Model | Llama-3.2-1B (frozen, bf16) + trajectory-memory side-car |
| Bench scripts | `scripts/bench/bench_dconcept.py`, `scripts/bench/bench_dconcept_n.py`, `scripts/bench/profile_grpo.py` |
| Raw logs | `outputs/dconcept_bench.log`, `outputs/dconcept_n_bench.log`, `outputs/grpo_speedbench_v2/`, `outputs/grpo_bs_outer2/`, `outputs/grpo_profile.log` |
| Wave 2 ckpt used for GRPO warm-start | `outputs/wave2_v2/ckpt.pt` |

### [TF-NTP] Bench 1 — `D_concept` scaling

Per-step time (warmup=2, timed=5) at BS=4, calling `Phase1Trainer.step_wave1` with full forward + backward + AdamW step on synthetic chunks of `D × T_window = 4 × 256 = 1024 tokens`. Synthetic data only affects content; compute is identical to real training.

| `D_concept` | ms/step | tok/s | peak VRAM | trainable M | slowdown |
|---|---|---|---|---|---|
| 256 | 489.7 | 8 365 | 12.87 GB | 16.3 M | 1.00× |
| 512 | 511.9 | 8 002 | 13.19 GB | 28.6 M | 1.05× |
| 1024 | 539.6 | 7 590 | 14.00 GB | 66.1 M | 1.10× |

**Takeaway.** D scaling is essentially free up to 1024 (+10% time, +1.2 GB VRAM). Llama-1B forward+backward dominates total compute; memory module's matmuls are small relative to that. Default committed to `D_concept=1024` immediately after this bench.

### [TF-NTP] Bench 2 — `D_concept × N` joint scaling

Same Phase 1 setup, scanning N ∈ {4096, 16384, 65536} and D ∈ {256, 1024}:

| D | N | ms/step | tok/s | peak VRAM | trainable M | slowdown vs (256, 4096) |
|---|---|---|---|---|---|---|
| 256 | 4096 | 489.7 | 8 365 | 12.87 GB | 16.3 M | 1.00× (baseline) |
| 256 | 16384 | 514.6 | 7 960 | 13.28 GB | 22.6 M | 1.05× |
| 256 | 65536 | 718.2 | 5 703 | 15.20 GB | 47.8 M | 1.47× |
| 1024 | 4096 | 538.3 | 7 609 | 14.00 GB | 66.1 M | 1.10× |
| 1024 | 16384 | 746.9 | 5 484 | 15.80 GB | 91.3 M | 1.53× |
| 1024 | 65536 | **OOM** | — | (>24 GB) | — | — |

**Takeaway.** N scaling is cheap up to 16K (+5%), real cost at 64K (+47%) — entry-routing's O(N·D) matmul starts to bite. N=65536 × D=1024 OOMs because `concept_states[BS=4, N=64K, D=1024]` runtime buffer alone is ~1 GB; with state_init + activations the total exceeds 24 GB.

### [AR-GRPO] Bench 3 — Step time + GPU utilization

Setup: `scripts/training/train_wave3.py` warm-started from `outputs/wave2_v2/ckpt.pt`, K=8 samples, max_new_tokens=256, narrativeqa source, kl_coef=0.001, clip_eps=0.2. Steady-state measured over steps 2-8 (step 1 is compile cold-start). GPU sampled every 1s during run via `nvidia-smi --query-gpu=utilization.gpu,memory.used`.

| variant | step time | rollouts/step | per-rollout time | peak VRAM | GPU mean util | GPU idle (<20%) | GPU high (≥80%) |
|---|---|---|---|---|---|---|---|
| K=8, BS_outer=1 | **4.4 s** | 8 | 0.55 s | 8.0 GB | **16.7%** | **77%** | 11% |
| K=8, BS_outer=2 | 7.1 s | 16 | 0.44 s | 9.0 GB | 20.7% | 68% | 10% |

**Takeaway — load-bearing for this session.** GPU is severely underutilized during AR-GRPO (~77% idle at BS_outer=1). The 4-second step time is *not* compute; it's primarily CPU work + kernel launch overhead + sync barriers during AR generation. BS_outer=2 recovers ~25% per-rollout throughput by amortizing overhead across more rollouts, but GPU is still 68% idle. Lots of headroom remaining.

Comparison vs the prior pre-fix reference (2026-05-11 entry below): that bench reported 2.37 s/step at K=8 on narrativeqa with 3.72 GB peak. Our current 4.4 s/step + 8 GB peak is regression on both axes; likely from the architectural fixes (logit_scale ops, KV-cache cap call per layer, dynamic-compile cache, reference-policy snapshot for KL).

### [AR-GRPO] Bench 4 — Per-op profile (torch.profiler)

[Appended once profile run completes — see `outputs/grpo_profile.log` + `outputs/grpo_profile.json` chrome trace.]

### Conclusions for the next round

1. **Default `D_concept` bumped to 1024** (post-bench, in working tree). Numerology aligns (id+state = 2 × 1024 = d_lm), bridge compression improves from 8× to 2×, per-concept capacity 4×. Cost is +10% step time + 1.2 GB VRAM.
2. **N=16K is cheap to reach** (+5% time). Worth keeping in back pocket if vocabulary size needs to scale.
3. **AR-GRPO's bottleneck is not raw GPU compute.** Reducing kernel-launch overhead (CUDA graphs on the AR step, BS_outer↑, K↑, async reward overlap) is the highest-leverage perf work. The Bench 4 profile data will identify which specific ops to target.

---

## 2026-05-11 — Phase 1 + Phase 2 after architectural perf push

**Training type:** [TF-NTP] (Phase 1 throughput sweep) + [AR-GRPO] (Phase 2 Strategy A data)

Headline: shipped 5 perf changes — Hopfield-tied entry projection,
cross-attn K/V sharing across J + across hops, bf16 cross-attn body,
trajectory-generator activation checkpointing, dropped per-slot KV cache
complexity. Plus 3 Phase 2 GRPO changes — batched K rollouts at BS=K,
selective log-softmax, per-sample backward in Pass 2.

### [TF-NTP] Wave 1 throughput sweep (RTX 4090, medium tier @ D_concept=256, KV cache ON)

| BS | compile | peak VRAM | tok/s | ms/iter |
|----|---------|-----------|-------|---------|
| 4  | OFF     | 9.44 GB   | 17.4k | 235     |
| 8  | OFF     | 16.2 GB   | 19.8k | 413     |
| 12 | OFF     | 22.9 GB   | 20.5k | 600     |
| **4**  | **ON**  | **7.16 GB**  | **26.0k** | **158** |
| **8**  | **ON**  | **11.6 GB**  | **27.2k** | **301** |
| 12 | ON      | 22.9 GB   | 20.5k | 599 (compile regressed here) |
| 16 | ON      | OOM       | —     | —       |

**Recommended production:** BS=8 with `--compile` → 27.2k tok/s,
11.6 GB peak, ~13 GB headroom. This is the new train_wave1.py default.
Previous baseline was BS=4 at 17k tok/s — **60% more useful throughput
with margin**.

Speed ceiling: throughput plateau between BS=4 and BS=8 (+4.6%) shows
we're at the GPU's per-token compute limit for Llama-3.2-1B forward+
backward on this hardware. Further gains would require Llama
quantization, vLLM-style rollout overlap (Phase 2 only), or moving to
a bigger GPU.

### [AR-GRPO] Wave 3 on REAL Strategy A data (RTX 4090, K=8, max_new=256)

Re-measured 2026-05-11 on actual parquet prompts, not synthetic random tokens. **This table is the reference our 2026-05-13 GRPO numbers compare against** — the 2.37 s/step on narrativeqa is the pre-Gumbel-fix benchmark we're regressed from.

| Source                          | N      | Avg prompt | Step time | Peak VRAM | Tok/s gen | Wall-time per epoch |
|---------------------------------|--------|------------|-----------|-----------|-----------|---------------------|
| narrativeqa                     | 4,764  | 8175       | 2.37s     | 3.72 GB   | 642       | **3.1 hr**          |
| musique                         | 10,000 | 1762       | 2.03s     | 3.72 GB   | 767       | **5.6 hr**          |
| hotpotqa (padded N=40)          | 12,000 | 6833       | 2.30s     | 3.72 GB   | 783       | **7.7 hr**          |
| 2wikimultihop (padded N=60)     | 8,000  | 6177       | 2.29s     | 3.72 GB   | 825       | **5.1 hr**          |
| quality                         | 2,523  | 6569       | 2.30s     | 3.72 GB   | 738       | **1.6 hr**          |
| **TOTAL (one Wave 3 epoch)**    | **37,287** | —      | —         | —         | —         | **23.2 hr**         |

Headline: full Wave 3 epoch on the Strategy A memory mix is **23.2 hours**
on a single 4090. Multi-epoch GRPO (2-4 epochs typical) → ~2-4 days.

VRAM is uniform at 3.72 GB across all sources (bounded by Llama's
sliding-window KV cache + manifold state + Pass 2 activation peak —
prompt length doesn't matter, only the cap and the K=8 BS-K rollout).
Plenty of headroom for K=16 (+10% step time) or BS_outer > 1.

Eval-only (held out from train mix per Strategy A — bench numbers kept
for reference):

| Dataset (eval-only)| Prompt | Gen cap | Step time | Peak VRAM | Notes              |
|--------------------|--------|---------|-----------|-----------|--------------------|
| gsm8k typical      | ~100   | 256     | 1.76s     | 3.42 GB   | reasoning capability eval only |
| numinamath p90     | ~141   | 256     | 1.79s     | 3.42 GB   | reasoning capability eval only |
| humaneval p90      | ~227   | 256     | 1.76s     | 3.43 GB   | reasoning capability eval only |

**BS_outer (multi-prompt batching) sweep at hotpotqa-typical
(prompt~6.6K, K=8, max_new=256, RTX 4090):**

| BS_outer M | Step time | Per-prompt | Peak VRAM | Speedup vs M=1 | Wave 3 epoch |
|------------|-----------|------------|-----------|-----------------|--------------|
| 1          | 2.15s     | 2.15s      | 3.72 GB   | 1.00×           | 23.2 hr      |
| 2          | 2.78s     | 1.39s      | 4.34 GB   | 1.55×           | 14.9 hr      |
| **4**      | **4.03s** | **1.01s**  | **6.00 GB** | **2.14×**     | **10.8 hr**  |
| 8          | 6.57s     | 0.82s      | 9.32 GB   | 2.62×           | 8.9 hr       |
| 12         | 9.24s     | 0.77s      | 12.64 GB  | 2.80×           | 8.3 hr       |

Production recommendation: **BS_outer=4 with K=8** (M*K=32 rollouts in
parallel per optimizer step). Cuts Wave 3 epoch from 23.2 → 10.8 hr,
fits at ~6 GB peak with plenty of headroom for a longer K or BS_outer
later. Diminishing returns past M=4 because prefill is still serial
(M sequential BS=1 prefills per step) — only the AR rollout + Pass 2
TF replay are batched.

**K-scaling at gsm8k-typical (prompt=100, gen=256):**

| K  | Step time | Peak VRAM |
|----|-----------|-----------|
| 8  | 1.76s     | 3.42 GB   |
| 16 | 1.94s     | 3.45 GB   |
| 32 | 2.42s     | 4.22 GB   |

K=16 is +10% time / +3% VRAM over K=8; K=32 is +37% time / +24% VRAM.
All fit comfortably; K=8 is the practical sweet spot, K=16 if more
samples per group is worth the cost.

**Pass-1 rollout speedup (R1, batched K rollouts vs serial K calls):**

| K | prompt | gen | serial | batched | speedup |
|---|--------|-----|--------|---------|---------|
| 4 | 1024   | 256 | 5.98s  | 1.56s   | **3.84×** |
| 8 | 512    | 128 | 5.28s  | 0.78s   | **6.79×** |
| 8 | 1024   | 256 | 11.28s | 1.57s   | **7.20×** |

Rollout went from dominating Phase 2 step time to ~half of it.

## Cross-references

- `scripts/bench/bench_trajmem.py` — Phase 1 sweep harness
- `scripts/bench/bench_compare.py` — Phase 1 + Phase 2 vs vanilla comparison
- `scripts/bench/_bench_common.py` — `bench()` timing primitive (warmup, sync,
  OOM cleanup, peak-mem stats)
- graph_walker's `docs/bench_results.md` (on `abandoned/graph-walker`) —
  reference numbers for an architecture with similar memory bridge.
