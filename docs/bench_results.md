# Bench Results — Llama-3.2-1B + trajectory-memory

Tracks the headline throughput / VRAM numbers for the integrated model.
Each section dates the run and pins the configuration so future-me can
spot regressions or progress without re-running every time.

Vanilla Llama path numbers (forward / lm_head step / full step) are
**not re-run here** — they're hardware-dependent only and unchanged from
graph_walker's measurements (see `abandoned/graph-walker` →
`docs/bench_results.md`). We only bench paths that depend on
trajectory-memory specifically.

Run benches via:
- `scripts/bench_trajmem.py` — Phase 1 (Wave 1 long-doc TF NTP) bench.
  Default sweeps BS via doubling from `--bs` anchor until OOM, picks
  peak. Pass `--no-sweep` for a fixed-BS run.

Per project memory:
- "Bench with fixed params, never sweep" — one config tier per run.
- "Bench at each path's own optimal BS" — peak-throughput BS per setting,
  not forced to share with other modes.

---

## 2026-05-09 — Phase 1 (Wave 1) at v1-default config (medium tier)

**Hardware:** RTX 4090 (25.3 GB) · bf16 (Llama backbone) · fp32 memory
params

**Config (`TrajMemConfig.medium`):**
- Manifold: N=4096 concepts · D_concept=256 · K_max_neighbors=64 ·
  radius=32 · p_rewire=0.5 · **262K directed edges**
- Trajectories: J=4 · K_read=8 · K_write=8
- Window: T_window=256 · D=4 (TBPTT depth) · **chunk = D × T_window =
  1024 tokens**
- Bridge: `MemInjectLayer` 2-layer MLP, `bridge_hidden=2048` (= d_lm),
  inject at layer 8
- Trainable params: **16.5M** (≪ Llama; backbone frozen)
  - bridge 9.44M · write 2.49M · read 2.43M · manifold 2.10M

(History: prior to 2026-05-09 bump, medium was N=2048, K=32 → 65K edges,
15.4M trainable. Bench numbers were within ~1% — manifold is too small
relative to Llama for the bump to dominate throughput.)

### Sweep — `bench_trajmem.py --config-tier medium --bs 1 --max-bs 32`

Numbers below are at the **post-bump medium config** (N=4096, K=64,
trainable 16.46M). Bench was re-run after the bump; throughput within
~1% of pre-bump values, peak GB up by ~0.05GB (manifold is small
relative to Llama).

| BS | eager tok/s | eager peak GB | compile tok/s | compile peak GB |
|----|------------:|-------------:|--------------:|---------------:|
| 1  | 7.1k | 7.5  | **9.1k** | **5.7**  |
| 2  | 8.1k | 13.1 | **9.1k** | 10.6     |
| 4  | **8.5k** | 21.6 | 8.5k | 22.3 |
| 8  | OOM  | —    | OOM      | —        |

`compile` flag: `torch.compile(model.forward_window, mode="default", dynamic=False)`.
Cold-start compile cost ~1-3 min per BS; reuses across iters within a run.

### Findings

- **Compile gives ~28% speedup at small BS** (BS=1: 7.1k → 9.1k tok/s).
  Wins from operator fusion in the per-window forward — peak memory
  also drops 7.4 → 5.7 GB at BS=1 (activation savings).
- **At BS=4 compile gives no benefit** — work is already kernel-bound
  (8.6k tok/s in both modes). Compile mainly helps when overhead-bound.
- **Both modes top out at BS=4 before OOM** on 24GB. BS=4 eager uses 21.5
  GB; BS=4 compile uses 22.2 GB (compile keeps a few extra graph buffers
  resident).

### Recommended production setting (UPDATED 2026-05-10)

This subsection was for the pre-KV-cache era. Current production
defaults (post-KV-cache + post-grad-checkpointing-removal):

- **`train_wave1.py`** asserts BS=1 (state-threading limitation). KV
  cache + compile both default ON. Real-world tok/s in BS=1 + state-
  threaded mode TBD (smoke shows ~5k tok/s steady-state).
- **`train_wave2.py`** runs `--batch-size 2` (TurnPair length-bucketed).
  KV cache + compile both default ON. `--prior-loss-weight 0.1`
  default (B12 fix).
- Use `--no-compile` for debug iteration (skip ~2 min cold-start).
- Use `--no-kv-cache` only for the rolling-buffer fallback path
  (slower but useful for reproducing legacy bench numbers).

NOTE: do NOT enable `gradient_checkpointing_enable()` while KV cache
is on — HF silently sets `use_cache=False` and discards
`past_key_values`. Verified empirically. They're mutually exclusive.

### Caveats

- Bench uses synthetic `randint` chunks at exactly `chunk_tokens=1024`.
  Real W1 training streams variable-length docs packed into 1024-token
  chunks via `LongDocDataset` — same shape, same speed.
- Wave 2 (TurnPair) and Wave 3 (GRPO) have different compute profiles.
  W2 chunks are length-bucketed across 1-12K-token priors; W3 multiplies
  by J=4-8 sample rollouts. Separate benches needed for those.
- We saw a Wave 2 OOM at BS=2 + config "small" during an earlier
  end-to-end smoke test (priors 1-3K tokens × N=4 chunks of 256 tokens).
  Cause is likely TBPTT activation accumulation across W2 chunks; flagged
  for follow-up before the first real W2 run.

---

---

## 2026-05-09 — Phase 1 + Phase 2 comparison vs vanilla Llama

Re-ran with all the post-audit fixes landed (chunk state threading,
output_hidden_states hook, logits-slicing, run_chunk strip, two-pass
GRPO). RTX 4090, eager, medium config, **BS=2** for Phase 1 / **K=4
samples** for Phase 2.

| Path | tok/s | peak GB | ms/iter |
|------|------:|--------:|--------:|
| **Phase 1** (BS=2, T=1024, lm_head-only trainable for vanilla) |  |  |  |
| V1.A — vanilla Llama forward (no_grad) | **52.0k** | 3.08 | 39.4 |
| V1.B — vanilla Llama lm_head TF step | **16.8k** | 10.26 | 122.0 |
| T1 — Llama + trajmem step | **9.3k** | 12.07 | 219.6 |
| **Phase 2** (T_pre=1024, T_gen=64, K=4 samples) |  |  |  |
| V2 — vanilla Llama GRPO step | ~43 | 10.31 | 5965 |
| T2 — Llama + trajmem two-pass GRPO step | ~35 | 22.00 | 7354 |

**Slowdowns:**
- T1 vs V1.B: **1.80×** — memory module costs ~7.5k tok/s (bridge MLP
  + read/write trajectory hops × D=4 windows).
- T2 vs V2: **1.23×** — smaller relative slowdown because Phase 2 is
  dominated by serial AR sampling (no KV cache in either path).

**Reproducing:** `PYTHONPATH=. python scripts/bench_compare.py --bs 2 \
--t-phase1 1024 --t-prompt 1024 --t-gen 64 --num-samples 4`

**Notes:**
- Phase 2 throughput is ~3 orders of magnitude lower than Phase 1 in
  tok/s. That's an architectural property of AR-without-KV-cache, not
  a memory-module problem — vanilla is also slow at this shape.
- Phase 2's 22 GB peak (T2) sits near our 24 GB ceiling. K=8 would not
  fit; T_gen=128+ marginal. KV caching for AR would help both.

---

## 2026-05-09 — Phase 1 + Phase 2 at each path's own max BS / K

Replaces the BS=2/K=4 shared-shape table above with a more honest
"each path at its own max-fitting size" comparison (per the project
convention "Bench at each path's own optimal BS"). Same hardware,
same medium config, eager mode. Per-path BS / K values are
hard-coded in `scripts/bench_compare.py`.

| Path | BS / K | tok/s | peak GB | ms/iter |
|------|-------:|------:|--------:|--------:|
| **Phase 1** (T=1024 chunk for trajmem; T=1024 for vanilla) |  |  |  |  |
| V1.A — vanilla Llama fwd (no_grad)        | BS=48 | **48.7k** | 16.90 | 1009 |
| V1.B — vanilla Llama lm_head TF step      | BS=5  | **17.0k** | 20.32 |  302 |
| T1   — Llama + trajmem step               | BS=4  | **9.9k**  | 18.46 |  415 |
| **Phase 2** (T_prompt=1024, T_gen=64) |  |  |  |  |
| V2   — vanilla Llama GRPO step            | K=10  | **42.5**  | 18.80 | 15050 |
| T2   — Llama + trajmem two-pass GRPO step | K=4   | **32.4**  | 22.00 |  7900 |

**Per-path slowdowns at each path's own optimal size:**
- **T1 vs V1.B: 1.72×** (vanilla 17.0k, ours 9.9k tok/s). Improved from
  the BS=2-shared 1.80× — vanilla doesn't gain from BS=2 → BS=5 since
  it was already throughput-saturated.
- **T2 vs V2: 1.31×** (similar story; V2 also saturated by K=8).

**Throughput-saturation observations:**
- V1.A vanilla forward has the same tok/s at BS=16 (7.29 GB) as at
  BS=48 (16.90 GB). Llama bf16 forward is GPU-saturated around
  ~49k tok/s on this card regardless of BS.
- V1.B and V2 are similarly saturated; bumping BS just trades VRAM
  headroom for nothing. Reported max-BS rows are for "we used the
  whole GPU" parity, not for throughput gains.
- T1 and T2 are at their actual VRAM caps. T1 BS=8 OOMs in eager
  (per `bench_trajmem.py` sweep); T2 K=5+ would push past 24 GB.

**The honest framing:** the memory module's overhead at peak is
**~7k tok/s for Phase 1** (V1.B 17.0k → T1 9.9k). Architecturally
that's mostly the per-window Llama forward re-encoding the rolling
LM context buffer (no sliding KV cache yet — see "Optimization
candidates" below).

**Reproducing:** `PYTHONPATH=. python scripts/bench_compare.py`

### Why is T1 ~1.7× slower than V1.B?

Not the memory modules themselves — they're <1% of FLOPs (per
`docs/profile_analysis.md`). The dominant overhead is Llama itself
running **multiple forwards per "chunk"** while the rolling LM
context grows.

- V1.B does **1 forward at T=1024**, full sequence, one shot.
- T1 does **4 forwards** at T_window=256 each, but with a rolling
  context buffer that grows window-by-window: 256 → 512 → 768 → 1024
  tokens fed to Llama. Total LM tokens processed per chunk = ~2560.
- Naive expected ratio: 2.5×. Measured 1.72×. T1 is actually
  *better-than-naive* per LM token (smaller forwards have lower
  attention cost than one big one).
- In real W1 training with state threading and a doc-long stream,
  the LM context fills to the 2048 cap and stays there → per-window
  forwards get bigger → real-training T1 is slower than this synthetic
  bench shows.
- The fix is sliding KV cache so each window's forward is just the
  new 256 tokens against cached 1792. That's the dominant lever.

### Optimization candidates (not yet tried)

| Lever | Est. win | Cost | Notes |
|-------|---------:|------|-------|
| Sliding KV cache for Llama | 30-50% | medium | Layer-8 inject changes per window so KV reuse needs custom logic for layers 0-7 + recompute for 8+. |
| `--compile` (bench is eager) | 28% at low BS | trivial | Already wired into `train_wave1.py`. Not yet tested at the new max BS. |
| Autocast bf16 (manifold scatter dtype refactor) | 5-10% | medium | Blocked on `scatter_mean` dtype mismatch — see profile_analysis.md. |
| Share d_lm↔D_concept projections (bridge / read-attn / write-attn) | 3-5% | medium | Three separate copies of the same shape projection today. |
| Llama-block gradient checkpointing | 0% speed, ~50% VRAM | small | Lets us push T1 to BS=8+ (gradient quality not throughput). |

---

## 2026-05-10 — KV cache landed: trajmem now matches/beats vanilla

Sliding KV cache implemented for both Phase 1 (rolling LM context buffer)
and Phase 2 (AR sampling + TF replay). HF DynamicCache, sliding-window
trimmed to `effective_lm_context`. Cache carries across windows of a
chunk; detached at chunk boundaries (mirrors `prev_states.detach()`).

Vanilla GRPO sampling in `bench_compare.py` was previously not using
KV cache — that handicapped vanilla unfairly. Fixed: V2 now uses HF
DynamicCache for AR. Both vanilla + trajmem benched at each path's
max-fitting BS / K (project convention).

| Path | BS / K | tok/s | peak GB | ms/iter |
|------|-------:|------:|--------:|--------:|
| **Phase 1** (T=1024) |  |  |  |  |
| V1.A — vanilla Llama fwd (no_grad)         | BS=48 | **48.7k** | 16.90 | 1009 |
| V1.B — vanilla Llama lm_head TF step       | BS=5  | **16.9k** | 20.32 |  302 |
| T1   — Llama + trajmem step (KV cache)     | BS=4  | **17.7k** | **15.07** |  232 |
| **Phase 2** (T_prompt=1024, T_gen=64) |  |  |  |  |
| V2   — vanilla Llama GRPO step (KV cache)  | K=12  | **169.5** | 21.65 | 4530 |
| T2   — Llama + trajmem two-pass GRPO (KV cache) | K=6   | **104.8** | 20.92 | 3664 |

**Per-path slowdowns:**
- **T1 vs V1.B: 0.96× — trajmem is now FASTER than vanilla**, with 5 GB
  less memory peak. Memory module pays for itself by letting Llama do
  smaller per-window forwards.
- T2 vs V2: 1.62× — vanilla Phase 2 fits twice the K (12 vs 6), reflecting
  the genuine memory cost of read+write per generation window. Both
  paths got 3-4× faster overall vs the no-KV baseline below.

**Comparison vs the no-KV-cache baseline (above table):**

| Path | Old tok/s | New tok/s | Speedup |
|------|----------:|----------:|--------:|
| T1 (Phase 1 trajmem) | 9.9k | **17.7k** | **1.79×** |
| V2 (vanilla GRPO) | 42.5 | **169.5** | **4.00×** |
| T2 (trajmem GRPO) | 32.4 | **104.8** | **3.23×** |

The Phase 1 1.72× T1-vs-V1.B gap from yesterday is **completely closed**
— trajmem now beats vanilla per-token AND uses less memory. The
prior measurement bottleneck (rolling buffer re-encoding per window)
turned out to be the entire architectural cost; sliding KV cache made
it disappear.

**Reproducing:** `PYTHONPATH=. python scripts/bench_compare.py`
**Trainer flag:** `train_wave{1,2}.py --use-kv-cache` (off by default
for backward compat; production runs should always set it).

### Phase 2 architectural fix bundled with KV cache

The pre-KV `_ar_sample_one` was also incorrect: it called
`forward_window` per generated token, which fired read+write per token
(should be per memory window per design). Fixed alongside the cache
work — read fires once per generation window, write fires once per
generation window (with surprise=0 per plan §5.4, since AR-generated
tokens have no NTP target). KV cache makes the per-token LM forwards
cheap.

---

## 2026-05-10 — Phase 2 GRPO correctness + shared-prefill efficiency

Big GRPO refactor landed across five phases (A-F). Adds the
correctness pieces production GRPO implementations all carry (KL term,
PPO importance-sampling clip, length-bias-free loss aggregation) AND
the shared-prefill efficiency win for pass 2.

### Final per-path table

| Path | BS / K | tok/s | peak GB | ms/iter |
|------|-------:|------:|--------:|--------:|
| **Phase 1** (T=1024) |  |  |  |  |
| V1.A — vanilla Llama fwd (no_grad)              | BS=48 | **48.7k** | 16.90 | 1009 |
| V1.B — vanilla Llama lm_head TF step            | BS=5  | **16.9k** | 20.32 |  302 |
| T1   — Llama + trajmem step (KV cache)          | BS=4  | **17.7k** | 15.07 |  231 |
| **Phase 2** (T_prompt=1024, T_gen=64) |  |  |  |  |
| V2   — vanilla Llama GRPO step (KV cache)       | K=12  | **173.6** | 21.65 | 4425 |
| T2   — Llama + trajmem GRPO (KV+shared prefill) | **K=16** | **132.8** | **12.78** | 7709 |

### Phase D shared-prefill memory + K wins

The dominant pass-2 cost was K full forwards through (prompt + sample)
each with their own activation graph. Phase D refactor: encode prompt
ONCE no_grad, then K per-sample TF forwards start from that shared
prefill state. Cost of "no_grad prefill": prompt-position writes don't
get gradient (only sample-position writes do). Acceptable — GRPO is
optimizing the rollout policy; sample-position writes are the ones
that directly affect reward.

| Stage | T2 tok/s | T2 peak GB | T2 K |
|-------|---------:|-----------:|-----:|
| Pre-KV-cache (yesterday) | 32.4 | 22.00 | 4 |
| KV cache only (today AM) | 104.8 | 20.92 | 6 |
| KV cache + shared prefill (today PM) | **132.8** | **12.78** | **16** |

Bigger K = better GRPO group-relative advantage estimates. K=16 matches
DeepSeek-R1's group size; the project was previously at K=4-6 which
the audit flagged as too noisy for stable GRPO.

### Phase A-B-D correctness summary (also landed today)

- **Dr.GRPO loss normalization** — divide by `K * max_new_tokens` not
  just `K` (removes documented length-bias pathology, arxiv 2503.20783).
- **PPO importance-sampling clip** — `clip(ratio, 1-ε, 1+ε)` with
  ε=0.2 default (matches TRL/verl). Optional asymmetric upper clip
  via `--clip-eps-higher` (DeepSeek-R1's `clip_higher` mode).
- **KL regularization to reference policy** — `β * D_KL(π_θ ‖ π_ref)`
  with β=0.001 (verl default). Reference is the loaded `--checkpoint-in`
  weights at start of Phase 2. K3 estimator. Param-swap pattern shares
  the ref forward across K samples (one swap-cycle per step).
- **Eval methods now use KV cache** — eval_wave1/eval_wave2 had the
  rolling-buffer fallback, busting compile cache on every val pass.
  Now matches training mode.
- **Partial-window write fix** — when AR stops mid-window, don't pad
  with zeros and write to manifold (would scatter zero into selected
  concept slots). If n_win < T/2, skip the write; else pad with the
  last real hidden.
- **Default temperature** 1.0 → 0.7 (matches DeepSeek-R1, verl).

### CLI flags added

```bash
--clip-eps        0.2     # PPO IS clip width (TRL/verl default)
--clip-eps-higher None    # DeepSeek-R1 asymmetric upper bound
--kl-coef         0.001   # KL term weight (verl default; 0 disables)
--no-compile              # opt out of torch.compile (default ON)
```

### Diagnostics added to `Phase2Metrics`

- `clip_fraction`: % of tokens where ratio was clipped this step
- `mean_ratio`: mean(ratio); should stay near 1.0
- `kl_to_ref`: mean per-token K3 KL estimate; rises if drift exceeds reg

Surfaced in trainer log lines.

**Reproducing:** `PYTHONPATH=. python scripts/bench_compare.py`

## Cross-references

- `scripts/bench_trajmem.py` — Phase 1 sweep harness
- `scripts/bench_compare.py` — Phase 1 + Phase 2 vs vanilla comparison
- `scripts/_bench_common.py` — `bench()` timing primitive (warmup, sync,
  OOM cleanup, peak-mem stats)
- graph_walker's `docs/bench_results.md` (on `abandoned/graph-walker`) —
  reference numbers for an architecture with similar memory bridge.
