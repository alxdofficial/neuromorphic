# Phase-2 Baselines — Results & Plan (START HERE)

Single entry point for the Phase-2 baseline evaluation: **established memory / long-context baselines on
LongMemEval-S + MemoryAgentBench, run BEFORE our own frozen-decoder memory layer.** This file is the index +
live status; the detailed docs are linked inline. _Last updated: 2026-07-21._

> **2026-07-21 — all MemoryAgentBench numbers were regenerated** after a scorer fix (`aab14e9`): two
> competencies (detective_qa / ICL) were structurally unscoreable and read 0.000 for **every** model.
> Any MAB figure in an older doc or note is stale; read them from [`PHASE2_REPORT.csv`](PHASE2_REPORT.csv).
> Details + before/after: [`PHASE2_REPORT.md`](PHASE2_REPORT.md) §"MAB scorer fix".

**Scoring policy:** deterministic only (EM + negation-guarded containment + BEM paraphrase for LongMemEval;
substring/exact per-competency for MAB). **No LLM-as-judge** for the panel; one GPT-4o cross-check was for
calibration only. Every number below is directly comparable across rows.

---

## 0. Status at a glance

| Layer | What | Status |
|---|---|---|
| **Tier-1** (API: long-context + RAG) | deepseek-v4-flash, llama-3.1-8b · floor/full_context/rag_bm25 | ✅ **DONE** (on `main`) |
| **Tier-2** (GPU memory *mechanisms*) | M+, H2O@2%, H2O@20%, KVzip, A-MEM (SnapKV/LCLM/`memoryllm-8b` dropped) | 🟡 **in progress** — H2O@2% done (both benchmarks); **KVzip MAB done (0.519)**, KVzip LongMemEval running; **M+ LongMemEval done (0.423)**, M+ MAB running; H2O@20% planned; A-MEM blocked on an API key (see §2) |
| **Our model** | frozen-decoder learned memory | ⏳ later (Phase-2 head-to-heads) |

---

## 1. RESULTS

### Tier-1 — DONE ✅
Full rendered tables (both datasets, every cell, per-competency + per-source), plus the **BEM-threshold
calibration finding** and the **Tier-1 cost log**: **[`PHASE2_REPORT.md`](PHASE2_REPORT.md)**
(+ [`PHASE2_REPORT.csv`](PHASE2_REPORT.csv)). Panel design + rationale: [`PHASE2_BASELINES.md`](PHASE2_BASELINES.md).

**LongMemEval-S (500 Q) — overall accuracy**
| model | floor | rag_bm25_k5 | full_context |
|---|---|---|---|
| **deepseek-v4-flash** | 0.006 | 0.594 | **0.687** |
| **llama-3.1-8b-instruct** | 0.002 | 0.404 | 0.462 |

**MemoryAgentBench (3,071 Q) — overall accuracy (strict / lenient)** — post-scorer-fix
| model | floor | rag_bm25_k5 | rag_bm25_k15 | full_context |
|---|---|---|---|---|
| **deepseek-v4-flash** | 0.356 / 0.382 | 0.712 / 0.773 | 0.778 / 0.822 | **0.854 / 0.885** |
| **llama-3.1-8b-instruct** | 0.221 / 0.252 | 0.491 / 0.497 | 0.524 / 0.531 | 0.636 (154 only)¹ |

_¹ llama full_context is N/A on MAB except the 154 shorter contexts that fit its 131k window (most are 158k–881k)._

**Reads:** deepseek-v4 > llama-3.1-8b in every condition; more context/retrieval helps monotonically
(floor → rag → full). deepseek full_context (0.687 LME) tops GPT-4o's published 0.606 — it's a 2026 model.
On MAB, deepseek full_context is strong across the board post-fix: Accurate-Retrieval 0.926,
Test-Time-Learning 0.828, Long-Range 0.789, Conflict-Resolution 0.723 (competency macro 0.816). The residual
hard cell is **multi-hop fact consolidation** — deepseek full_context still only reaches 0.200 at 262k
(`factconsolidation_mh_262k`) and RAG collapses to 0.041 there.
_(Pre-2026-07-21 docs claimed Test-Time-Learning and Long-Range "collapse to ~0 under deterministic
exact-match". That was our own prompt/metric mismatch, not a scorer-fidelity floor — see the fix note above.)_

**Published references (CITED, GPT-4o-judge — NOT our scale):** GPT-4o LME full 0.606 / oracle 0.870;
Llama-3.1-8B LME full 0.454 / oracle 0.710. Full tables in [`PHASE2_REPORT.md`](PHASE2_REPORT.md) §Published.

### Tier-2 — IN PROGRESS
H2O has completed locally on both full benchmark selections. M+ has completed LongMemEval-S on the pod fleet.
Length-capped generations are scored as emitted; natural EOS completion is reported separately in the
authoritative report.
**SCOPE — the five arms of interest (fixed 2026-07-21).** The runnable Tier-2 panel is **M+**, **H2O@2%**,
**H2O@20%**, **KVzip**, **A-MEM**. Everything else is cite-only: **SnapKV**, **LCLM**, and
**`memoryllm-8b`** (the non-M+ MemoryLLM) are **DROPPED** — do not schedule or report rows for them.

| method | LongMemEval | MemoryAgentBench |
|---|---|---|
| **M+** (`mplus-8b`) | **0.423** overall / 0.379 task-avg / **0.000** abstention (500; coverage 1.000, EOS 1.000) | 🔄 **RUNNING — no number yet** |
| **H2O @2% KV** (`cap2048`) | **0.066** overall / 0.056 task macro / 0.333 abstention (500; EOS 0.390) — see budget caveat | **0.371** strict micro / 0.287 competency macro / 0.377 lenient (3,071; EOS 0.892) |
| **H2O @20% KV** (`cap≈20k`) | ⏳ PLANNED — the paper's operating point | ⏳ PLANNED |
| **KVzip** (`Qwen2.5-7B-Instruct-1M`) | 🔄 RUNNING (4-pod shard fleet) | **0.519** strict micro / 0.519 competency macro / 0.526 lenient (3,071; EOS 1.000) |
| **A-MEM** (agent-memory) | — | — (blocked: OpenRouter key) |
| ~~SnapKV~~ | **DROPPED** | **DROPPED** |
| ~~LCLM~~ | **DROPPED** | **DROPPED** |
| ~~`memoryllm-8b`~~ | **DROPPED** (M+ only) | **DROPPED** |

**Tier-2 RESCORE (2026-07-21).** H2O and KVzip artifacts were written by their POD, whose scorer predates
the `parse_output` fix; `pure_rescore.py` only matches Tier-1's 4-field run tags and `merge_shards.py` only
matches `_shKofN` caches, so single-process Tier-2 runs fell through both and kept pre-fix `correct` values.
Rescoring the cached generations (no recompute) moved: **KVzip MAB 0.402 → 0.519** (Test_Time_Learning
0.000 → 0.634, Long_Range_Understanding 0.000 → 0.592) and **H2O MAB 0.278 → 0.371** (Test_Time_Learning
0.000 → 0.566). NOTE the earlier claim that H2O's two zeros were a genuine capability result is **retracted
for Test_Time_Learning** — H2O scores 0.566 there, consistent with its `recent1024` window retaining the
ICL labels that sit at the end of the context. Only Long_Range_Understanding (0.028) is genuinely bad,
which is the expected failure mode for recency-biased eviction.

**KVzip read (2026-07-21).** At 0.519 it is the strongest compression method on MAB, ahead of H2O@2%
(0.371), and the gap is concentrated exactly where the mechanisms differ: Long_Range_Understanding
**0.592 vs 0.028**. KVzip is query-agnostic and retains distant evidence; H2O's recency window discards it.

**H2O budget caveat (2026-07-21) — do not quote 0.066 bare.** That run is `rolling-cap2048` against a
~105k-token context = a **1.95% KV budget (~51× compression)**; H2O's paper evaluates at ~20%. At 2% it
refuses **95.8%** of LongMemEval questions, *more* than the no-context floor (93.0%; full-context is 26.8%),
and 0.066 vs the floor's 0.002 is barely above nothing. So it measures "H2O at 2%", not "H2O". Its best
category (abstention 0.333) is accidental — refusing is the correct answer there. Hence the **@20% arm**:
the pair is a compression-ratio ablation showing *where* KV eviction breaks, which is the informative
result. Sizing for the re-run: peak ≈ **23.8GB** (15.0 weights + 2.44 KV + 2.44 snapshot clone + ~2.0
attention matrix + ~1.95 activations) → **48GB pod (A40/A6000)**; a 24GB 4090 is borderline. Runtime is
~unchanged (attention is ~2% of per-chunk FLOPs at this scale).

**M+ read (new, 2026-07-21).** Artifact `outputs/baselines/longmemeval__memoryllm__mplus-8b__s__MERGED.json`
(28 shards merged, n=500). A fixed-size parametric memory lands **within 4 points of llama-3.1-8b
full-context** (0.423 vs 0.462) without reading the 115k history at query time, and **beats it on
temporal-reasoning** (0.472 vs 0.417); it crushes H2O (0.066) and stays well behind deepseek full-context
(0.687). Per-type: knowledge-update 0.694 · temporal 0.472 · single-session-user 0.406 · multi-session 0.364 ·
single-session-assistant 0.339 · preference 0.000 (low-confidence). **Abstention = 0.000** while every other
system scores ≥0.333 — M+ never declines to answer, it confabulates. That is a real, reportable failure mode
of the mechanism. Full per-type table: [`PHASE2_REPORT.md`](PHASE2_REPORT.md) §M+.

---

## 2. PLAN — Tier-2 campaign (SHARDED POD FLEET)

**How it actually runs (revised 2026-07-21; supersedes the earlier "one tailored pod, sequentially" plan).**
M+ cannot be batched within one model (single shared pool, batch=1 baked in), so throughput comes from
**horizontal sharding across many cheap pods**, not from a bigger GPU:

1. All code/script/config changes done **locally first** (no pod burning while editing).
2. Launch a **fleet** of single-GPU pods, one shard of the question set each, via
   **`scripts/pod/mpod.py`** (fleet state in `scripts/pod/.mpod_state.json`). The M+ LongMemEval campaign ran
   **14 shards** on a mixed RTX 4090 / A40 / RTX A6000 fleet at `rate_per_pod` **$0.69/hr**.
3. Poll actively; pods that die or lag get their shard re-cut. Runs are resumable (per-question JSONL cache).
4. **Pull → merge → terminate.** `scripts/baselines/tier2/merge_shards.py` merges shard artifacts
   (recursive cache glob; duplicate `question_id`s across partitionings resolved by newest mtime — the M+
   LongMemEval merge hit **72** such duplicates).

**Measured throughput (pod fleet, M+ LongMemEval):** **0.44–0.64 s/inject** and **4.1–4.5 s/answer**, against
the runner's built-in defaults of `--cost-inject-s 0.375` / `--cost-answer-s 2.3` (which came from a local
4090 n=1/n=2 smoke, `outputs/baselines/longmemeval__memoryllm__mplus-8b__s__n{1,2}__*.json`). **The runner's
own ETAs therefore read ~1.5× optimistic** — retune the flags per card. Also measured: **Ampere (A40 /
A6000) runs this workload at ~1.24× an isolated 4090, not slower** — the job is snapshot/PCIe-bandwidth-bound,
not compute-bound, so the cheap 48GB Ampere cards are the right buy.

**HARD CONSTRAINT — M+ on MemoryAgentBench needs a ≥116GB **RAM** container** (host RAM, not VRAM). MAB's
post-injection snapshot includes `cached_dropped_*` buffers that grow with inject count; 47GB pods are
OOM-killed (SIGKILL, rc=137, no traceback). LongMemEval is unaffected — singleton contexts skip the snapshot
entirely. Filter pods on RAM, not just GPU, when scheduling M+ MAB.

Ops rules for renting/driving pods (disk, cached images, flash-attn wheels, fleet hygiene):
**[`docs/ops/RUNPOD_RUNBOOK.md`](../ops/RUNPOD_RUNBOOK.md)**. Fleet driver: **`scripts/pod/mpod.py`**.
Per-model setup fixes are baked into `scripts/pod/tier2_bootstrap.sh`; restart checklist in
`scripts/pod/TIER2_RESUME.md`; single-pod tooling in `scripts/pod/tier2_pod.py`. Integration notes:
[`TIER2_GPU_INTEGRATION.md`](TIER2_GPU_INTEGRATION.md), [`TIER2_HOSTING.md`](TIER2_HOSTING.md). The methods &
how they work: [`FROZEN_COMPETITORS.md`](FROZEN_COMPETITORS.md).

| # | Model | Method (how) | Backbone | Scope | GPU used | Cost / time | Status |
|---|---|---|---|---|---|---|---|
| 1 | **H2O / SnapKV** | streaming/query-aware KV eviction | Llama-3.1-8B | LME + MAB (H2O) | local 4090 | $0 / 4h44m | **H2O complete**; SnapKV pending |
| 2 | **KVzip** | query-agnostic KV compression | Qwen2.5-7B-1M | LME + MAB | **H100/A100 80GB** | ~$5–10 _(estimate)_ | 🟡 local smoke passed; full run pending |
| 3 | **M+ / MemoryLLM** | recurrent parametric memory | mplus-8b | LME + MAB | 14-pod fleet, mixed 4090/A40/A6000 @ $0.69/pod-hr | see §3 | ✅ **LME DONE (n=500)**; 🔄 MAB running (needs ≥116GB RAM pods) |
| 4 | **A-MEM** | 2b agent-memory over a frozen LLM | OpenRouter panel | LME (+MAB) | none (API) | **PROJECTION only** — LME-S ~$14–36 expected / $181 strict-WandB, MAB ~$3.8–38.9; see [`A_MEM_FIDELITY_AUDIT.md`](A_MEM_FIDELITY_AUDIT.md) | ⏳ not started — cost table is a projection **blocked on a 401'd key** (needs key refresh to instrument) |
| 5 | **LCLM** | soft-token compression | 0.6b+4b | — | — | — | **DROPPED 2026-07-20** |

**Per-model notes (findings already established — do not re-discover):**

- **#1 H2O/SnapKV** — Llama-3.1-8B gate is **OPEN** (verified 2026-07-20, account `alxd219p1`). **H2O now uses
  the first-party infinite-streaming Transformers-5.1 adapter**, not KVCache-Factory: raw retained keys,
  position rolling, 2,048-token cache (1,024 heavy + 1,024 recent), 512-token chunks, query-head GQA mode.
  A real 105,146-token LME-S prefill completed locally in 23.29s (4,514 tok/s) at 17.62GB peak. The maximum
  744,639-token MAB prompt crossed the native 131k window and prefetched in 164.38s; query snapshot forks took
  ~2.5ms and peaked at 18.40GB. All 32 layers retained exactly 2,048 entries. Exact LME prompt stats: p50 105,531,
  p95 107,047, max 107,557 tokens; all fit Llama's 131,072 window with 64 generation tokens. SnapKV remains on
  KVCache-Factory/Transformers-4.44.2 and still needs ≥48GB because it materializes full prompt KV first.
  H2O supports exact context snapshot reuse on MAB; SnapKV remains LongMemEval-only. **Completed results:** LME
  0.066 overall (500/500 scored; EOS 39.0%); MAB 0.278 strict micro / 0.138 competency macro / 0.377 lenient
  (3,071/3,071 scored; EOS 89.2%). The 2,048-token budget is only ~1.9% of the average LME prompt and ~1.1%
  of the average MAB context, substantially below the original paper's main 4–60% budget sweep.
- **#2 KVzip** — needs the custom CUDA kernel (`cuda-nvcc` + **`cuda-cccl=12.4.*`** pin) + prebuilt flash-attn
  wheel (baked into bootstrap). Query-agnostic → correct KV baseline for MAB (36 encodes / 3071 Q). Tune
  `--ratio` (KV retained, default 0.3). The runner enforces the shared 64-token cap through upstream's
  `ModelKVzip.gen_kwargs` and records the actual value. Local 4090 validation passed on the longest LME-S
  context with prefill=2,048 / scoring=1,000 (paper ablation): 22.03GB allocated, 24.73GB reserved, 81.5s.
  Paper-default scoring=2,000 OOMs there. Full MAB reaches 745,586 tokens (~42.8GB raw KV plus 15.2GB
  weights), so use 80GB and smoke the maximum context before the complete run.
- **#3 M+ / MemoryLLM** — **batching within one model is impossible** (single shared `[32,10240,4096]` pool,
  batch=1 baked in). It's **overhead-bound, not compute-bound** (~0.4 s/inject wall vs ~15–25 ms compute) →
  the cost lever is **bigger `--inject-block-tokens`** (default 512; test {1024,2048} on a 50-item subsample
  first — free ~2× if quality holds). Because it can't batch, throughput comes from **sharding across many
  cheap pods** (`mpod.py`), not a bigger GPU: an H100 is wasted here, and A40/A6000 measured **~1.24× a
  4090**. Multi-GPU inside one pod buys wall-clock only, same $. Needs `flash_attention_2`. First-line answer
  truncation already in (base model, no EOS). Details: memory `project_mplus_batching_verdict`.
  **Prep DONE (local, off-GPU):** `run_memoryllm.py` records GPU-synced inject/answer timing into
  `meta.timing`; `scripts/baselines/tier2/sweep_memoryllm_blocksize.py` runs the block-size sweep on a fixed
  subsample and prints accuracy-vs-inject-time + a recommended block.
  **LongMemEval-S DONE (2026-07-21):** 14-shard fleet run, merged to n=500 at `blk512`, overall **0.423**.
  Two operational facts to carry forward: (a) the runner's ETA defaults (0.375 s/inject, 2.3 s/answer) are
  **~1.5× optimistic** vs the measured 0.44–0.64 / 4.1–4.5; (b) **MAB needs a ≥116GB-RAM container** — the
  post-injection snapshot's `cached_dropped_*` buffers scale with inject count and 47GB pods get SIGKILLed
  (rc=137). LongMemEval is immune (singleton contexts skip the snapshot). `run_memoryllm._inject_blocks` now
  tokenizes in character chunks so peak is O(chunk) rather than O(context) — necessary but not sufficient.
- **#4 A-MEM** — not started. The cost table in [`A_MEM_FIDELITY_AUDIT.md`](A_MEM_FIDELITY_AUDIT.md) is a
  **projection from one instrumented sample per dataset**, not an invoice, and re-instrumenting is **blocked
  on a 401'd key**. No accuracy number exists.
- **#5 LCLM — DROPPED 2026-07-20 by project decision.** Its runner and local clone remain for provenance,
  but no Phase-2 run is planned. Keep the paper in related work; do not include it in the active baseline list.

---

## 3. Budget

| | Cost | Notes |
|---|---|---|
| Tier-1 (done) | **~$17** | LongMemEval ~$14 (full_context ≈90%) + MAB ~$3.30 (deepseek 50× prefix-cache vs ~$66 naive) + judge $0.03 |
| Tier-2 M+ LongMemEval (done) | **fleet burn ≈$9.7/hr** | 14 pods × **$0.69/pod-hr** (`rate_per_pod` in `scripts/pod/.mpod_state.json`). Sharding buys wall-clock at roughly constant $/question — the total is set by the measured 0.44–0.64 s/inject × injects, not by the pod count. Exact invoice = RunPod billing, not reconstructible from the artifacts. |
| Tier-2 (remaining est.) | **~$15–25** _(estimate)_ | KVzip ~$5–10 · M+ MAB (needs ≥116GB-RAM pods, so pricier per pod than the LME fleet) · H2O complete locally · A-MEM API-only, see projection above · LCLM dropped |

---

## 4. Doc map
- **Results:** [`PHASE2_REPORT.md`](PHASE2_REPORT.md) / [`.csv`](PHASE2_REPORT.csv) (full tables + BEM calibration + cost log)
- **Panel design:** [`PHASE2_BASELINES.md`](PHASE2_BASELINES.md) · audit fixes [`PHASE2_AUDIT2_FIXES.md`](PHASE2_AUDIT2_FIXES.md)
- **Tier-2 setup:** [`TIER2_GPU_INTEGRATION.md`](TIER2_GPU_INTEGRATION.md) · [`TIER2_HOSTING.md`](TIER2_HOSTING.md) · `scripts/pod/TIER2_RESUME.md`
- **Pod ops:** [`docs/ops/RUNPOD_RUNBOOK.md`](../ops/RUNPOD_RUNBOOK.md) (portable rent/drive rules) · fleet driver `scripts/pod/mpod.py` (+ `scripts/pod/.mpod_state.json`) · shard merge `scripts/baselines/tier2/merge_shards.py`
- **Competitor landscape:** [`FROZEN_COMPETITORS.md`](FROZEN_COMPETITORS.md)
- **API runbook:** [`API_EVAL_RUNBOOK.md`](API_EVAL_RUNBOOK.md) · **MAB schema:** [`MEMORYAGENTBENCH_SCHEMA.md`](MEMORYAGENTBENCH_SCHEMA.md)
