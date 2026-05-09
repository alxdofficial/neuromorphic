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

### Recommended production setting

**`--config-tier medium --batch-size 2 --compile`** (compile flag is now
wired into `train_wave1.py` / `train_wave2.py`):

- 9.1k tok/s — peak throughput (tied with BS=1 + compile)
- 10.5 GB peak — leaves 14 GB headroom for variable-length W2 batches,
  optimizer state growth, occasional long needle docs
- Pays the ~2 min compile cold-start once; amortized over hours-long
  training runs

Alternative: **`--config-tier medium --batch-size 4`** (eager, no compile)
— 8.6k tok/s at 21.5 GB. 6% slower but no compile cold-start, useful for
debug iteration where startup time matters more than steady-state speed.

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

## Cross-references

- `scripts/bench_trajmem.py` — Phase 1 sweep harness
- `scripts/bench_compare.py` — Phase 1 + Phase 2 vs vanilla comparison
- `scripts/_bench_common.py` — `bench()` timing primitive (warmup, sync,
  OOM cleanup, peak-mem stats)
- graph_walker's `docs/bench_results.md` (on `abandoned/graph-walker`) —
  reference numbers for an architecture with similar memory bridge.
