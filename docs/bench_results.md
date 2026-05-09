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
- Manifold: N=2048 concepts · D_concept=256 · K_max_neighbors=32 ·
  radius=16 · p_rewire=0.5
- Trajectories: J=4 · K_read=8 · K_write=8
- Window: T_window=256 · D=4 (TBPTT depth) · **chunk = D × T_window =
  1024 tokens**
- Bridge: `MemInjectLayer` 2-layer MLP, `bridge_hidden=2048` (= d_lm),
  inject at layer 8
- Trainable params: **4.5M** (≪ Llama; backbone frozen)

### Sweep — `bench_trajmem.py --config-tier medium --bs 1 --max-bs 32`

| BS | eager tok/s | eager peak GB | compile tok/s | compile peak GB |
|----|------------:|-------------:|--------------:|---------------:|
| 1  | 7.1k | 7.4  | **9.1k** | **5.7**  |
| 2  | 8.1k | 13.1 | **9.1k** | 10.5     |
| 4  | **8.6k** | 21.5 | 8.6k | 22.2 |
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

**`--config-tier medium --batch-size 2 --compile`** (when compile lands in
trainer entrypoints):

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

## Cross-references

- `scripts/bench_trajmem.py` — the bench harness
- `scripts/_bench_common.py` — `bench()` timing primitive (warmup, sync,
  OOM cleanup, peak-mem stats)
- graph_walker's `docs/bench_results.md` (on `abandoned/graph-walker`) —
  vanilla Llama paths A/B/C reference numbers
