# Phase-0 Results — running log

Living, curated log of **Phase-0 (architecture bake-off)** experiments: training runs, band-gate
(REAL/SHUF/OFF) scores, other metrics, and — most importantly — **notes & observations**. Raw eval
dumps live in `outputs/memory/<run>_band_gate.json` and `<run>_<arm>_summary.json`; **this doc is the
interpreted record** so we don't re-derive conclusions from JSON each time.

> Append a new `## Run: <tag>` section per campaign. Keep the newest at the top of the Runs list.
> Always record: config, provenance (commit/date), the tables, and what we concluded.

---

## Metrics glossary (read before trusting a number)

The band-gate (`scripts/diagnostics/mixed/mixed_band_gate_eval.py`) reports, per arm × task:

- **REAL** — val loss with memory ON (lower = better).
- **OFF** — val loss with memory ZEROED (the arm's own no-memory control).
- **SHUF** — val loss with memory rolled across the batch (each example decoded with *another*
  example's memory).
- **OFF−REAL** (`>0` ⇒ memory is **used**) — how much turning memory off hurts.
- **SHUF−REAL** (`>0` ⇒ memory is **example-specific**) — **THE binding metric.** ≈0 means the memory
  is a generic task prior, not this-example's content (the "membership wall").
- **%band** = `(FLOOR−REAL)/(FLOOR−CEIL)·100`, where FLOOR = `vanilla_llama` (no memory), CEIL =
  `vanilla_full_context` (full raw context). Fraction of the no-mem→full-ctx gap closed.
- **TF-EM** (babi only) — teacher-forced exact-match over the answer span.

**Caveats (do not skip):**
- **%band is inflated / sometimes invalid.** CEIL is the *untrained-LoRA* full-context baseline, so a
  trained memory arm can beat it → %band > 100% (e.g. babi ~666%). If CEIL > FLOOR the band is flagged
  **INVALID** (reconstruct). Trust **OFF−REAL / SHUF−REAL** over raw %band.
- **All metrics are teacher-forced diagnostics**, internally comparable — **not** standard benchmark
  EM/F1. Benchmark scoring is a Phase-2 concern.
- Make causal "memory helps" claims from each arm's own **OFF** control, not from %band vs the fresh
  vanillas (which fold in decoder-adaptation).

---

## Runs

### Run: `localrun-0001` — first full cohort (2026-07-12)

**Config:** 6 trainable arms × **6000 steps** × B=6, `behavioral_kl`, frozen SmolLM2-135M + shared
read-LoRA, **M=96**, ctx 2048, 8×256 streaming windows. Local RTX 4090, ~7 h total. Eval: band-gate,
val-batches 32.
**Provenance / caveat:** trained on code **before** the peer-audit fixes (`cf3af21`) — notably the
**#6 position-padding fidelity fix** (which changes icae/autocompressor/gisting compression) and the
MemoryLLM relabel landed *after* this run. So these numbers are **indicative, not final** — a re-run
on `cf3af21`+ is pending (see Planned).

**Final in-training val loss (lower = better):**

| arm | reconstruct | babi | doc_qa | continuation | fact_recall |
|---|---|---|---|---|---|
| titans | 6.53 | 2.49 | 3.50 | 3.23 | 2.74 |
| icae | 6.54 | 2.59 | 3.81 | 3.29 | 3.02 |
| gisting | 6.47 | 2.33 | 3.43 | 3.07 | 2.45 |
| autocompressor | 6.60 | 2.58 | 3.74 | 3.29 | 2.96 |
| memoryllm | 6.51 | 2.42 | 3.62 | 3.07 | 2.60 |
| slotgraph | 6.61 | 2.56 | 3.75 | 3.25 | 2.94 |

**FLOOR / CEIL (for %band):** reconstruct 15.04/15.24 *(INVALID: ceil>floor)* · babi 9.61/8.57 ·
doc_qa 4.99/3.42 · continuation 3.47/2.35 · fact_recall 4.31/0.18.

**SHUF−REAL — example-specific binding (THE metric; >0 = binds):**

| task | icae | autocomp | titans | **gisting** | **memoryllm** | slotgraph |
|---|---|---|---|---|---|---|
| reconstruct | +0.13 | +0.01 | +0.04 | **+0.60** | **+0.42** | +0.01 |
| babi | +0.02 | +0.00 | −0.00 | +0.09 | +0.02 | +0.00 |
| doc_qa | −0.00 | −0.00 | +0.01 | **+0.11** | +0.08 | −0.00 |
| continuation | +0.06 | +0.04 | +0.15 | **+0.56** | **+0.60** | +0.19 |
| fact_recall | −0.00 | +0.00 | +0.00 | +0.08 | +0.04 | +0.00 |

**%band (valid tasks; higher = better):**

| task | icae | autocomp | titans | **gisting** | memoryllm | slotgraph |
|---|---|---|---|---|---|---|
| doc_qa | 73 | 75 | 92 | **95** | 83 | 74 |
| continuation | 14 | 15 | 21 | **35** | 35 | 19 |
| fact_recall | 32 | 33 | 39 | **45** | 42 | 34 |

**OFF−REAL — memory used (>0 = used; all arms use memory):**

| task | icae | autocomp | titans | gisting | memoryllm | slotgraph |
|---|---|---|---|---|---|---|
| reconstruct | +0.07 | +0.05 | +0.76 | +0.25 | +0.33 | +0.08 |
| babi | +1.49 | +3.56 | +5.77 | +6.24 | +5.84 | +3.58 |
| doc_qa | +0.18 | +0.23 | +0.95 | +1.11 | +0.53 | +0.57 |
| continuation | +0.12 | +0.08 | +0.22 | +0.38 | +0.48 | +0.31 |
| fact_recall | +0.06 | +0.10 | +0.82 | +1.17 | +1.02 | +0.49 |
| (babi TF-EM) | 2.1% | 2.6% | 1.6% | 4.2% | 3.1% | 1.6% |

**Observations:**
1. **Per-layer-KV read is the through-line for binding.** The *only* two arms with clearly positive
   SHUF−REAL are **gisting** and **memoryllm** — the two per-layer-KV arms. gisting wins %band across
   all three valid tasks. This is the strongest signal: read *surface* (per-layer KV vs single
   prepend) is what buys example-specific binding.
2. **The prepend arms (icae, autocompressor, titans, slotgraph) sit at SHUF−REAL ≈ 0** on the recall
   tasks — memory is *used* (OFF−REAL > 0) but as a **generic prior**, not this-example content. The
   membership wall persists.
3. **slotgraph is mid/bottom-cluster** — statistically tied with icae/autocompressor. Its graph
   structure is currently **inert** (only continuation shows life, +0.19). As a single-shot
   compressor it is not competitive; its differentiator (edges) isn't differentiating yet.
4. **titans is the loss-vs-binding cautionary case:** strong %band (2nd on doc_qa) via its test-time
   gradient write, but SHUF−REAL ≈ 0 — "loss-useful, not example-specific." Two different routes to
   helping the decoder; only per-layer-KV lights up binding.
5. **babi is the hardest** — SHUF−REAL ≈ 0 for *everyone*, TF-EM 1.6–4.2%.
6. **The current eval only tests single-shot compress-then-recall** — it does NOT test the
   persistent-update / overwrite axis slotgraph is meant to win. Read the slotgraph verdict with that
   in mind.

---

## Key findings so far (across runs)

- **Binding is driven by the READ (per-layer-KV), not by state persistence or write mechanism.**
  memoryllm (persistent) and gisting (per-chunk) both bind; the state axis barely matters at fixed
  read-richness. → the lever for slotgraph is a per-layer-KV read (see `slotgraph_kv_baseline`).
- **The graph edges are inert under the current loss-neutral objective** — SHUF−REAL ≈ 0 says they
  carry no example-specific bits. Two levers: (a) per-layer-KV read, (b) an objective that forces the
  edges to store this-example info.
- **The cohort tiles the `{state × read × write}` space with minimal redundancy** (icae≈autocompressor
  the only near-duplicate). slotgraph can't differentiate by re-picking a cell — only via the
  orthogonal *relational-structure* axis.

---

## In progress / planned

- **slotgraph_kv v1** (`slotgraph_kv_baseline`, commit `2bbea23`) — per-layer-KV read for slotgraph
  (reuses the write's per-layer edge injection). A 6000-step run was started then **stopped to profile**;
  **re-run pending.** Key number to watch: does its SHUF−REAL jump toward gisting/memoryllm (+0.1…+0.6)?
- **opt#1 A/B** (`slotgraph_inject_harvest_only`, commit `cdd641f`) — a ~1.44× write speedup, off by
  default; needs a train A/B (loss trajectory + SHUF−REAL sanity) before becoming default. Can be
  folded into the slotgraph_kv v1 re-run.
- **Cohort re-run on fixed code** (`cf3af21`+) — `localrun-0001` predates the position-padding fidelity
  fix; a clean re-run gives the *final* comparison numbers.

---

## Changelog

- **2026-07-13** — doc created; logged `localrun-0001` cohort + band-gate results and observations.
