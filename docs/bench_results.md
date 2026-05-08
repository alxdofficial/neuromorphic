# Bench Results — Llama-3.2-1B + graph_walker

Tracks the headline throughput / VRAM numbers for the integration. Each
section dates the run and pins the configuration so future-me can spot
regressions or progress without re-running every time.

**All numbers below are at the production-target walker config** (the
perfected scaleup from the standalone graph_walker branch, currently on
`main`):

- **Topology:** N=4096 columns (64×64 flat torus), K=64 out-edges, p_rewire=0.5, radius=4 (capacity bump 2026-05-08; pre-bump runs were at N=1024, K=32)
- **Widths:** D_s=256, D_id=512, D_model=1024, content_mlp_depth=4, D_hid_content=1024, post_model_depth=2
- **Neuromod:** 6 layers / 8 heads / D_mod=512 / edge_hidden=384
- **Bridge (`MemInjectLayer`):** 2-layer MLP with `bridge_hidden=2048` (= d_lm), GELU activation; richer Llama↔graph translator than the prior single linear (~10.5M vs ~1.0M).
- **Trainable params:** ~40.0M total (29.5M walker [9.7M graph + 19.8M neuromod] + 10.5M bridge)

Run benches via:
- `scripts/bench_phase1.py` — Phase 1 paths A/B/C/D in one process (vanilla fwd / vanilla lm_head step / vanilla full step / Llama+GW step)
- `scripts/bench_phase2.py` — Phase 2 paths E/F (vanilla LM GRPO baseline / Llama+GW grpo_step)

CLI flags worth knowing:
- `--target-config` — load aspirational ~110M scaleup (eager-only, OOMs at BS≥4 on 4090; cudagraph compile bisect #276 blocks it)
- `--compile-block` — enable whole-block compile (production fast path; ~22 min first compile, ~3.7× over eager)
- `--regional-compile` — compile `walker_step_from_h` instead (1-2 min first compile, ~30% lower per-iter throughput; dev iteration only)
- `--dynamic-shapes` — pass `dynamic=None` to torch.compile for cross-shape compile reuse

---

## 2026-05-08 — Phase 1 + Phase 2 at bumped production config (N=4096, K=64, bridge=2048)

**Config:** RTX 4090 · bf16 autocast · production walker (N=4096, K=64, p_rewire=0.5, radius=4, D_s=256, D_id=512, neuromod 6L/8H/D_mod=512/edge_hidden=384) · `MemInjectLayer` 2-layer MLP bridge with `bridge_hidden=2048` · `--regional-compile`. Trainable total **40.0M** (29.5M walker + 10.5M bridge).

### Phase 1 — `bench_phase1.py --bs 8 --T 256 --regional-compile --skip-vanilla-full`

| Path | tok/s | Peak VRAM | ms/iter | Notes |
|---|---|---|---|---|
| A — vanilla Llama fwd (`no_grad`) | **54.5k** | 3.08 GB | 37.6 | Pure inference reference. |
| B — vanilla Llama lm_head step | **17.2k** | 10.24 GB | 119.1 | Backward through 16 frozen layers, only lm_head trainable. |
| C — vanilla Llama full step | OOM | — | — | All 1.24B trainable; BS=8 OOMs alongside walker activations. (Pre-bump runs at BS=8 fit because walker was smaller — re-bench in a separate process if needed.) |
| D — Llama + GW phase1 step | **1.7k** | 19.68 GB | 1215.3 | regional-compile mode. Vs 2026-05-02 BS=8 baseline (3.3k tok/s @ N=1024,K=32, single-linear bridge): ~2× walker slowdown from K=32→64 routing + bridge MLP. BS=20 OOMs at this config; max-fitting BS likely 12-14. |

### Phase 2 — `bench_phase2.py --B 2 --K 4 --T-pre 256 --gen-length 128 --reward placeholder --regional-compile`

Effective batch B·K=8, total seq T_pre+gen=384. Default B=8/K=8 OOMs at this config.

| Path | gpu-tok/s | Peak VRAM | ms/iter | Notes |
|---|---|---|---|---|
| E — vanilla LM GRPO | **3.5k** | 12.93 GB | 878.9 | Reference: frozen Llama + lm_head, autoregressive sample (K=4, gen=128) + replay grpo_step. |
| F — Llama + GW grpo_step | **1.0k** | 22.86 GB | 3031.9 | F vs E slowdown: **3.45×**. Near VRAM ceiling — no headroom to bump B or K at this config. |

**Takeaways:**
- N=4096 + K=64 + bridge=2048 nudged max-fitting BS down: Phase 1 BS=20 → ~12, Phase 2 B·K=64 → 8.
- Walker cost is dominated by routing (linear in K) — K=32→64 is ~2× of the slowdown vs prior bench.
- Bridge MLP is a small contributor: 10.5M params × 384 tokens × 2 directions ≈ 8 GFLOPs/iter (~0.1 ms on the 4090).
- Phase 2 GW is ~3.5× vanilla; gap is dominated by the autoregressive walker step (every gen token = one walker hop, launch-bound).
- For real waves at this config, plan on BS=8 Phase 1 / B·K=8 Phase 2 with `--regional-compile`; bigger BS requires either dropping `bridge_hidden` to 1024 (~5M bridge instead of 10M) or moving to whole-block compile.

---

## 2026-05-02 — Three-way comparison at production config

**Setup:** RTX 4090 · BS=4 · T=256 · bf16 autocast · fused AdamW · production walker config (N=1024, K=16, D_s=256, D_id=512, neuromod 6L/8H/D_mod=512/edge_hidden=384) · `_walker_cfg_for` overrides `mod_period = tbptt_block = T = 256` to satisfy the integration's external-surprise constraint · GW path uses `--compile-block`.

| Path | tok/s | Peak VRAM | ms/iter | Trainable params | Notes |
|---|---|---|---|---|---|
| Vanilla Llama fwd-only (`no_grad`) | **54.6k** | 2.78 GB | 18.8 | 0 | Reference: pure Llama inference. |
| Frozen Llama + gradients (lm_head only) | **18.6k** | 5.84 GB | 55.0 | ~262M | Backward flows through all 16 frozen layers; only lm_head receives param-grad. Isolates "activation memory + recompute through frozen Llama" cost. |
| Frozen Llama + GW (full `phase1_pretrained_step`) | **1.8k** | 6.50 GB | 576.3 | 25.3M (24.2M walker + 1.1M inject) | Production integration training step. `--compile-block` on. neuromod-only plasticity. |
| Hot Llama (all 1.24B params trainable) | **9.0k** | 11.52 GB | 113.2 | 1.24B | Full fine-tune reference: fused AdamW state for all params + per-layer param-grads. |

**Slowdown ratios:**
- Vanilla fwd → frozen+grad: **2.9×** slowdown, +3.1 GB VRAM. Cost of running backward through frozen layers (activation memory + recompute), no param-grad cost.
- Frozen+grad → frozen+GW: **10.3×** slowdown, +0.66 GB VRAM. Pure walker compute cost — the dominant term.
- Frozen+GW → hot-Llama: GW is **~5× SLOWER** than hot-Llama training. A 25M-trainable walker is more expensive per token than a 1.24B-trainable Llama.
- Vanilla fwd → frozen+GW: **30.3×** slowdown, +3.7 GB VRAM. End-to-end "walker overhead" vs pure inference.

**Key observations:**
- **Walker is launch-/compute-bound, not memory-bound.** Post the activation-checkpointing commit (`4a19103`) and vocab-agnostic refactor, peak VRAM for the GW path is only +0.66 GB over the lm_head-only baseline (6.50 vs 5.84 GB). The pain is throughput — 576 ms/iter for the walker vs 55 ms/iter for the Llama-only baseline.
- **Walker > Llama in compute cost.** At BS=4, T=256, the walker's per-step cost (~520 ms attributable, after subtracting Llama's 55 ms baseline) exceeds even the cost of training a full 1.24B-param Llama from scratch (113 ms/iter). The walker dominates the integration training step's wall-clock.
- **Walker speedup is the headline lever.** Per `walker_speedup_plan.md`, the target is 10–15× walker speedup via reduce-overhead torch.compile + Triton fusion. At 10× speedup the GW step would land at ~57 ms/iter, comparable to the lm_head-only baseline (55 ms) — i.e., the walker would be effectively free relative to running backward through frozen Llama.
- **VRAM headroom is healthy.** 6.50 GB peak at BS=4 leaves room for at least 2-4× BS increase before hitting the 24 GB 4090 limit, modulo the per-token compute scaling.

---

## 2026-05-02 — Max-BS / per-path optimum on a single 4090

The BS=4 table above is the "fair fixed-config comparison". This section shows each path measured at **its own optimal batch size**, so absolute throughput is what each path can actually deliver when allowed to fill the GPU.

**Setup:** RTX 4090 · T=256 · bf16 autocast · fused AdamW (where applicable) · production walker config · `--compile-block` on the GW path. Each row is a single `bench_pretrained_gw.py` or `bench_llama_full_training.py` subprocess.

| Path | Best BS | tok/s @ best | Peak VRAM | ms/iter | Notes |
|---|---|---|---|---|---|
| Vanilla Llama fwd-only (`no_grad`) | **8** | **54.5k** | 3.08 GB | 37.6 | Plateaus at BS=8; BS=16-256 stays in the 49-54k range — Llama-1B forward is compute-bound at small BS. |
| Frozen Llama + gradients (lm_head + opt.step) | **16** | **17.6k** | 16.94 GB | 232.2 | Sweep with `--mode lmhead-only`; BS=32 OOMs. |
| Frozen Llama + GW (full `phase1_pretrained_step`) | **12** | **4.5k** | 11.52 GB | 675.5 | BS=4: 1.9k → BS=8: 3.3k → BS=12: 4.5k (≈linear scaling, confirming launch-bound). BS=16 was unreliable in the same-process run (cross-test memory fragmentation; reported a degraded 1.3k); should be cleanly measurable with a fresh subprocess and could fit ~14 GB. |
| Hot Llama (all 1.24B trainable) | **16** | **12.7k** | 23.00 GB | 322.6 | At the 24 GB ceiling; BS=32 OOMs. |

**Slowdown ratios at each path's own optimum:**
- Vanilla fwd → frozen+grad: **3.1×** slowdown (54.5k → 17.6k)
- Frozen+grad → frozen+GW: **3.9×** slowdown (17.6k → 4.5k) — the walker's marginal cost
- Frozen+GW vs hot-Llama: walker is **2.8× slower** than full 1.24B fine-tune (4.5k vs 12.7k)
- Vanilla fwd → frozen+GW: **12.1×** slowdown (54.5k → 4.5k) — end-to-end walker overhead vs pure inference

**Comparison to the BS=4 fixed-config slowdowns:**
| Slowdown | At BS=4 fixed | At each path's optimum |
|---|---|---|
| Frozen+grad → frozen+GW | 10.3× | **3.9×** |
| Vanilla fwd → frozen+GW | 30.3× | **12.1×** |
| Frozen+GW vs hot-Llama | 5× slower | **2.8× slower** |

The walker scales much better with BS than the BS=4 numbers suggested — confirming the launch-bound diagnosis. Pure BS scaling alone (BS=4 → BS=12) gave **2.4× speedup** for the GW path with no code changes. The "walker is hopelessly slow" framing from the fair-BS table was overstated; at each path's optimum, the gap is closer to a 4× slowdown vs the lm_head baseline, which is the right neighborhood for an architecture this novel.

**Updated headline numbers:**
- **Achievable now (no code changes):** 4.5k tok/s GW step at BS=12 → 1B-token training run in **~62 hours** (vs ~6.4 days at BS=4).
- **After cudagraph + Triton fusion (10× walker speedup, plan):** ~22ms walker step → ~70 ms total step at BS=12 → ~44k tok/s → 1B-token run in **~6 hours**, comfortably in the iteration regime.

---

## 2026-05-03 — `_checkpoint_block=False` is now the integration default

**Diagnostic finding**: The BS=12 → BS=16 cliff in the GW step path was caused by activation checkpointing on the whole compiled walker block (`compile_walk_block_from_h`). When `_checkpoint_block=True`, backward re-runs the compiled forward to regenerate intermediates; that recompute triggers a separate inductor compilation of the backward gradient kernels. At BS≥16 that backward compile hits a cuBLAS autotuner failure — `select_algorithm.py: "Constructing input/output tensor meta failed for Extern Choice"` warnings fire — and falls back to a much slower kernel path.

**Diagnostic table (single 4090, T=256, --compile-block, BS=16):**

| Config | tok/s | ms/iter | Peak VRAM |
|---|---|---|---|
| `--compile-block` + `_checkpoint_block=True` (old default) | 2.2k | 1826 | 13.99 GB |
| `--compile` (per-step) + ckpt=True | 6.3k | 649 | 16.88 GB |
| `--compile-block` + **`_checkpoint_block=False`** (new default) | **8.8k** | **464** | **14.59 GB** |

Checkpoint OFF is **4× faster** than checkpoint ON at BS=16, with only +0.6 GB VRAM cost. The memory savings ckpt=True buys (~3 GB at BS=16) is not worth the wall-clock cost when there's 9+ GB of GPU headroom.

**Action taken:** `src/graph_walker/pretrained/integrated_lm.py` `__init__` now sets `self.memory._checkpoint_block = False` after constructing memory. Standalone walker training is unaffected (it uses the per-token `step_core` path, not the compile-block path, so `_checkpoint_block` is irrelevant there).

**Headline at the new default:**
- **Current best: 8.8k tok/s @ BS=16, T=256, 14.59 GB peak VRAM.**
- 69% of the hot-Llama bar (12.7k tok/s @ BS=16). Gap closed from 2.8× → 1.4×.
- 1B-token training run drops from ~62 hours (at the old BS=12 best) to **~32 hours**.
- 9.4 GB VRAM headroom remaining at BS=16 — could push to BS=20+ if scaling continues cleanly.


---

## 2026-05-04 — Phase-2 (GRPO) baseline + K sweep

The Phase-1 numbers above are parallel teacher-forced (`phase1_pretrained_step`).
Phase-2 GRPO has a different cost shape: `K` AR rollouts per step, prefix pass
with grad + per-token AR generation pass without grad + REINFORCE backward.
This section adds the matching baselines + a K sweep.

### Throughput metric framework (Phase-2 specific)

For Phase-1 (parallel teacher-forced) "tok/s" is unambiguous: each
forwarded token is a unique training-data token consumed. **For Phase-2
GRPO it's not.** The K rollouts per step share the same prefix and the
generated continuations are model samples, not data. So three different
metrics give three different answers, each useful for a different
question:

| Metric | Definition | What it measures | Right for |
|---|---|---|---|
| **steps/sec** | `1 / time_per_step` | Optimizer-step throughput | Sample-efficiency-bound training (Wave 3/4 with finite fact corpus) |
| **dataset-tok/sec** | `BS_outer × (T_pre + gen_length) / time` | Unique training tokens consumed per second; **does NOT scale with K** | "How long to consume Wave-N's dataset" |
| **gpu-fwd-tok/sec** | `BS_outer × K × (T_pre + gen_length) / time` | Raw GPU compute throughput; scales linearly with K | Comparing GPU work across configs (apples-to-apples with Phase-1 tok/s) |

K is a **variance-reduction lever, not a throughput lever**. All K
rollouts of a step act on the same prefix; K just gives you K samples
of the policy gradient to average. From a training-pace perspective,
going K=2 → K=16 buys you ~7% slowdown in steps/sec for an 8× variance
reduction in the gradient estimator — that's a great trade.

### AR baseline — frozen Llama, no walker (`bench_llama_full_training.py --mode ar`)

The "GRPO-shaped step without the walker" reference: forward prefix
[BS, T_pre=256] with grad enabled, AR-generate gen_length=128 tokens
with grad disabled, backward through a CE loss on the prefix, opt.step
on `lm_head` only.

For the AR baseline, BS is real data parallelism (each row is a
different prefix), so `dataset-tok/sec = BS × T_pre / time` (gen-tokens
are model samples, not data). The "tok/s" column below is `gpu-fwd-tok/sec`
= `BS × (T_pre + gen_length) / time` for direct comparison with the
GRPO `gpu-fwd-tok/sec` column.

| BS | steps/sec | dataset-tok/s | gpu-fwd-tok/s | Peak VRAM | ms/iter |
|---|---|---|---|---|---|
| 1  | 1.53 | 0.39k | 0.6k | 5.20 GB  | 653.2 |
| 2  | 1.45 | 0.74k | 1.1k | 5.41 GB  | 690.5 |
| 4  | 1.40 | 1.43k | 2.1k | 6.96 GB  | 715.8 |
| 8  | 1.31 | 2.68k | 4.0k | 10.35 GB | 763.4 |
| **16** | **1.14** | **4.67k** | **7.0k** | **17.15 GB** | **876.6** |
| 32 | OOM | — | — | — | — |

**Headline: 4.7k dataset-tok/sec at BS=16 (1.14 steps/sec)**, OOMs at 32.

### GRPO K-sweep (`bench_grpo.py`)

Production Phase-2 step at `BS_outer=1, T_pre=256, gen_length=128,
d_mem=256, neuromod-only trainable surface, --compile-block OFF`.

`--compile-block` was disabled because the AR generation path's
single-token forwards don't match the T=256 compile-block shape — first
attempt hung 10+ minutes in warmup before being killed. Eager-mode
numbers below.

K is the rollouts-per-step axis AND the effective rollout batch dim (K
prefix replicas forward in parallel). Headline table is K∈{4, 8, 16} —
production GRPO papers cap at K=8 for similar reasons:

| K  | steps/sec | dataset-tok/s | gpu-fwd-tok/s | Peak VRAM | ms/iter |
|---|---|---|---|---|---|
| 4  | 0.76 | 0.29k | 1.2k | 3.50 GB | 1308 |
| **8** | **0.75** | **0.29k** | **2.3k** | **3.80 GB** | **1336** |
| 16 | 0.72 | 0.28k | 4.4k | 4.42 GB | 1396 |

**Production headline: K=8 at 0.75 steps/sec, peak 3.80 GB.** K=4→8 is
~free (1% slowdown for 2× variance reduction in the gradient estimator);
K=8→16 also cheap (4% slowdown for another 2×). Beyond K=16 the marginal
trade gets worse.

#### Beyond K=16 (diminishing returns — for reference only)

K=32 and K=64 fit on the 4090 without OOM, but the "K-is-free" property
breaks once the AR-gen sequential bottleneck saturates the K batch dim:

| K  | steps/sec | Peak VRAM | ms/iter | Marginal cost |
|---|---|---|---|---|
| 16 | 0.72 | 4.42 GB | 1396 | (baseline) |
| 32 | 0.64 | 5.64 GB | 1554 | +11% slowdown for 2× variance reduction |
| 64 | 0.54 | 8.09 GB | 1867 | +16% slowdown for 2× variance reduction |

For production, K∈{4, 8, 16} is the right range — K=32+ is overkill
unless you're chasing the last bit of gradient variance. The 4090 has
plenty of headroom (~16 GB unused at K=64) so K isn't memory-bound here;
it's compute-bound on the AR-gen path.

**Observations:**
- **K barely affects steps/sec in the production range** (K=4..16): ~5%
  wall-clock growth across the range. Per-step cost is dominated by 128
  sequential AR-gen tokens, which K rollouts share by parallelizing
  across the K batch dim. This is the whole reason K=8-16 is a good
  production choice: 2-4× variance reduction for ~5% wall-clock cost.
- **Dataset-tok/sec is flat in K** (~280 tok/s across all K). Same prefix
  per step regardless of K — K is a variance-reduction lever, not a
  throughput lever.
- **GPU-fwd-tok/sec scales linearly in K** (1.2 → 2.3 → 4.4 → 7.9 → 13.2 k).
  This is what an outsider sees as "throughput", but it's pure GPU
  compute, not training pace.
- **Massive VRAM headroom on the 4090.** Even at K=64 we only use 8.09 GB,
  leaving 16 GB free. AR baseline at BS=16 uses 17.15 GB by contrast —
  `lm_head` trainable with backward through the full prefix holds ~3×
  more activation memory than GRPO's neuromod-only minimal surface.
- **GRPO vs Phase-1 GW step:** Phase-1 GW = 8.8k tok/s @ BS=16 (every
  token is a real training token in parallel TF) → 8800 dataset-tok/sec.
  Phase-2 GRPO @ K=8 = 290 dataset-tok/sec. Phase-2 is **~30× slower in
  dataset-tok/sec**. Or in steps/sec: Phase-1 ≈ 2.1 steps/sec at BS=16
  vs Phase-2 ≈ 0.75 steps/sec at K=8 → ~3× slower in steps/sec. The
  unavoidable cost of REINFORCE on actual generation via 128 sequential
  AR steps per rollout.
- **`--compile-block` hangs in mixed eager+AR paths.** When the wrapper
  has `--compile-block` enabled and then runs AR generation (single-
  token forwards through cached KV), inductor either recompiles per
  token or falls into a fallback path that doesn't terminate (10+ min
  observed before kill). Phase-2 stays eager until the rollout
  primitive is made compile-aware.

**Implications for training wave wall-clock:**

For Phase-2 waves (3 + 4), the right wall-clock unit is **steps/sec**
since the dataset is small relative to the number of revisits needed
for convergence. With Phase-2 ≈ 0.75 steps/sec at K=8 (production):

- **Wave 3** (passphrase chat-injected GRPO): ~500 facts × 5 questions
  × ~3 paraphrases = ~7500 unique (prefix, reference) pairs. Want
  ~10 visits per pair for convergence → ~75K steps. At 0.75 steps/sec
  (BS_outer=1) → ~28 hours per cycle. With BS_outer=8 (5.08× sess/s
  speedup, 2026-05-06 sweep) → **~3.3 hours per cycle pass**.
- **Wave 4** (WildChat overflow GRPO): K=8 default, turn-batched
  (Verlog-style). 30K sessions flatten to ~470K TurnPairs (15.7 avg
  assistant turns × 30K). At B=8 (Wave-3-equivalent speedup), one
  pass over the turn-pair pool ≈ 470K / (8 × ~2.5 sess/s) ≈ ~6.5 hours
  per pass. Multi-pass cycles for convergence — exact wall-clock
  pending first Wave 4 production-shape bench.

Earlier this doc reported "3 hr / 6 hr" estimates that used
`gpu-fwd-tok/sec` and overcounted by a factor of K. The new estimates
are honest. Pragmatic answer: Phase-2 cycles are multi-day; budget for
~1 cycle per week at the current rate.

---

## DeepSeek-style per-action credit assignment (Phase A — landed 2026-05-05)

GRPO now follows DeepSeek's sample → score → teacher-forced-replay-with-grad
structure. The previous implementation only credited **prefix-time**
routing decisions — gen-time routing decisions were made under `no_grad`
and got no gradient. Switching to DeepSeek-style gives per-routing-decision
log-π × advantage gradients across BOTH prefix and gen, analogous to
DeepSeek's per-generated-token credit (but for the walker's routing
action space).

**What landed:**
- `routing.py`: `route_or_replay` + `StepRoutingChoices` + `routing_log_pi_for_action`
- `graph_walker.py`: capture buffer (`start_capturing_routes` / `consume_routing_trace`), replay stash (`arm_replay_trace`), `walk_segment` consumes replay trace and threads per-step `replay_choices`, `walker_step_from_h(replay_choices=...)`, `_walker_step(replay_choices=...)`. `is_new_window` reconstructed from saved `anchor_idx` presence so sample/replay routing patterns match.
- `rollout.py`: `sample_grpo_rollout` (no-grad sample with capture armed) + `replay_grpo_rollout` (teacher-forced with-grad replay). Old `autoregressive_rollout` retained for inference.
- `train_phase2.py`: `grpo_step` rewritten as sample → score → replay → REINFORCE. Returns the same `GRPOStats` for telemetry-compat.
- `tests/test_routing_replay.py`: 6 tests covering math parity, grad propagation, walker capture+replay end-to-end, and DeepSeek full-flow with gradient reaching `memory.neuromod.*`.

128 tests pass (127 pre-existing + 1 new DeepSeek end-to-end).

**Performance impact:** replay does an extra parallel forward [K, T_pre + L_gen] through Llama with grad — compute increases vs the old "prefix-with-grad + gen-no-grad" path, but the gradient signal per step is now much richer (per-action across both prefix AND gen routing decisions, vs only-prefix-actions before). Expected steps/sec to drop 20-40% on the same hardware. Will re-bench post-merge and update this section with the new K-sweep numbers.

**Phase B + C deferred:** PPO clipping (`min(ρA, clip(ρ, 1±ε)A)`), KL penalty against a reference policy, entropy bonus. See `docs/plan_grpo_deepseek_style.md` for the design.

---

## 2026-05-05 — Wave 3 GRPO production-shape bench

Real Wave 3 stack: BERT-cosine reward (`sentence-transformers/all-mpnet-base-v2`)
+ real chat-injected loader (passphrase fact + UltraChat filler + question,
chat-templated). T_pre=256, gen_length=128, BS_outer=1, neuromod-only
trainable surface, eager mode.

| K | steps/sec | dataset-tok/s | gpu-fwd-tok/s | Peak VRAM | ms/iter |
|---|---|---|---|---|---|
| 4 | 0.50 | 192 | 0.8k | 4.10 GB | 2019 |
| **8** | **0.48** | **183** | **1.5k** | **4.55 GB** | **2098** |
| 16 | 0.45 | 174 | 2.7k | 5.46 GB | 2243 |

**Production headline: K=8 at 0.48 steps/sec, peak 4.55 GB.**

**~38% slower than bare GRPO** (which got 0.75 steps/sec at K=8 with
placeholder reward + synthetic prefixes). The extra ~700 ms/step is
BERT scoring (~50-100 ms per forward, K+1 sentences batched) + chat-
template tokenization in the loader. Loader cost can be amortized via
pre-batching; BERT cost is fundamental.

**Same K-sweep story as bare GRPO**: K=8 vs K=4 costs only ~4% steps/sec
for 2× variance reduction. K=16 adds another ~7% slowdown for the next
2× variance — only worth it if K=8 gradient signal is too noisy in
practice.

**Wave 3 wall-clock revisited:**

- ~7500 unique (prefix, reference) pairs × 10 visits/pair = 75K steps
- At 0.48 steps/sec → **~43 hours per cycle** (vs the ~28 hr earlier
  estimate using bare-GRPO numbers)

**Recommended config: K=8, BS_outer=1.** 4.55 GB / 24 GB peak leaves
plenty of room to push K higher, but K=8 is the DeepSeek-R1 production
default and the variance/cost trade flattens past it.

## 2026-05-06 — Wave 3 GRPO BS_outer (multi-prefix) sweep

`grpo_step` is now batch-aware: B independent prefixes per step, each
gets K rollouts, advantage-normalized within each K-group (no cross-
prefix mixing). Walker memory state is sized [B*K, ...] for the step.
See `train_phase2.py` docstring for layout details.

K=8, T_pre=256, gen_length=128, BERT-cosine reward, real chat data.

| B | steps/s | sess/s | gpu-tok/s | Peak VRAM | ms/iter | sess/s speedup |
|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.50 | 1.5k | 4.55 GB | 2010 | 1.00× |
| 2 | 0.46 | 0.92 | 2.8k | 5.46 GB | 2164 | 1.84× |
| 4 | 0.40 | 1.60 | 4.9k | 7.28 GB | 2508 | 3.20× |
| **8** | **0.32** | **2.54** | **7.8k** | **10.92 GB** | **3151** | **5.08×** |
| 12 | 0.26 | 3.14 | 9.6k | 14.56 GB | 3823 | 6.28× |
| 16 | 0.22 | 3.48 | 10.7k | 18.20 GB | 4594 | 6.96× |

**Production headline: B=8, K=8 → 2.54 sessions/sec at 10.9 GB peak.**
**5.08× session-throughput speedup vs B=1.**

Why the steps/s number goes DOWN with B: each step does B× more work
(B independent samples + B replays through the LM forward). The wall-
clock-relevant metric is `sess/s = B × steps/s` — the throughput at
which we consume distinct (prefix, ref) pairs. That number scales
near-linearly to B=8, then flattens hard.

**Diminishing returns past B=8**: B=12 only buys 1.24× more sess/s for
33% more VRAM, B=16 only 1.37× for 67% more. The flattening tracks
with GPU compute saturation: at B=8, gpu-tok/s = 7.8k tok/s, which is
already near the K-sweep ceiling (K-sweep at B=1 plateaus around
5-6k gpu-tok/s but BS_outer adds independent-batch headroom that the
K-axis doesn't expose).

**Wave 3 cycle wall-clock at B=8, K=8:**
- 30K (prefix, ref) pairs × 1 visit = 30K sessions / 2.54 sess/s = **3.3 hours per cycle pass**
- 10 visits/pair = 33 hours (vs ~43 hours at B=1) — same training data,
  ~25% wall-clock reduction at the same K-variance
- OR: same wall-clock budget = 5× more visits per pair at the same
  variance reduction = much better policy convergence

**Recommended config (updated): K=8, B=8.** 10.9 GB / 24 GB peak.
~5× wall-clock speedup over B=1 at the same K-variance.

**Implementation notes:**
- `prefix_ids[B, T_pre].repeat_interleave(K, dim=0)` — contiguous K-blocks
- BertCosineReward: list[Tensor] of B refs, K-block layout aware
- Per-group advantage normalization: rewards [B*K] → reshape [B, K] →
  per-row mean/std → flatten back. NO cross-prefix advantage mixing.
- `sample_grpo_rollout` and `replay_grpo_rollout` were already batch-
  dim-agnostic; no changes needed there.
- Back-compat: B=1 callers passing `reference_cont` as a Tensor still work.
