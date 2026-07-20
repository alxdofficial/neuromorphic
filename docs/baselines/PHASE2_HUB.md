# Phase-2 Baselines — Results & Plan (START HERE)

Single entry point for the Phase-2 baseline evaluation: **established memory / long-context baselines on
LongMemEval-S + MemoryAgentBench, run BEFORE our own frozen-decoder memory layer.** This file is the index +
live status; the detailed docs are linked inline. _Last updated: 2026-07-20._

**Scoring policy:** deterministic only (EM + negation-guarded containment + BEM paraphrase for LongMemEval;
substring/exact per-competency for MAB). **No LLM-as-judge** for the panel; one GPT-4o cross-check was for
calibration only. Every number below is directly comparable across rows.

---

## 0. Status at a glance

| Layer | What | Status |
|---|---|---|
| **Tier-1** (API: long-context + RAG) | deepseek-v4-flash, llama-3.1-8b · floor/full_context/rag_bm25 | ✅ **DONE** (on `main`) |
| **Tier-2** (GPU memory *mechanisms*) | KVzip, H2O/SnapKV, M+/MemoryLLM, LCLM | 🔜 per-model campaign (see §2) |
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

**MemoryAgentBench (3,071 Q) — overall accuracy (strict / lenient)**
| model | floor | rag_bm25_k5 | rag_bm25_k15 | full_context |
|---|---|---|---|---|
| **deepseek-v4-flash** | 0.337 | 0.613 / 0.760 | 0.646 / 0.815 | **0.721 / 0.882** |
| **llama-3.1-8b-instruct** | 0.206 | 0.394 / 0.492 | 0.417 / 0.533 | 0.636 (154 only)¹ |

_¹ llama full_context is N/A on MAB except the 154 shorter contexts that fit its 131k window (most are 158k–881k)._

**Reads:** deepseek-v4 > llama-3.1-8b in every condition; more context/retrieval helps monotonically
(floor → rag → full). deepseek full_context (0.687 LME) tops GPT-4o's published 0.606 — it's a 2026 model.
On MAB, Accurate-Retrieval is strong (deepseek full 0.936) while Test-Time-Learning + Long-Range collapse to
~0 under deterministic exact-match (ICL/detective subtasks the string scorer can't credit — a scorer-fidelity
floor, disclosed in the report).

**Published references (CITED, GPT-4o-judge — NOT our scale):** GPT-4o LME full 0.606 / oracle 0.870;
Llama-3.1-8B LME full 0.454 / oracle 0.710. Full tables in [`PHASE2_REPORT.md`](PHASE2_REPORT.md) §Published.

### Tier-2 — NOT STARTED (no real numbers yet)
Only smoke tests have run. The table below fills in as each model completes its pod run.
| method | LongMemEval | MemoryAgentBench |
|---|---|---|
| KVzip | — | — |
| H2O / SnapKV | — | — (H2O only; SnapKV N/A) |
| M+ / MemoryLLM | — | — |
| LCLM | — | — |

---

## 2. PLAN — Tier-2 per-model campaign

**Workflow discipline (one model at a time):**
1. All code/script/config changes done **locally first** (no pod burning while editing).
2. Rent a pod **tailored to that model** → tune config until as fast as it'll go.
3. Get it running **steadily** → derive a clean ETA → know exactly when it's safe to pull artifacts.
4. **Retrieve → terminate → next model.** Runs are resumable (per-question JSONL cache).

Setup fixes are baked into `scripts/pod/tier2_bootstrap.sh`; restart checklist in `scripts/pod/TIER2_RESUME.md`;
agent↔pod tooling in `scripts/pod/tier2_pod.py`. Integration notes: [`TIER2_GPU_INTEGRATION.md`](TIER2_GPU_INTEGRATION.md),
[`TIER2_HOSTING.md`](TIER2_HOSTING.md). The 4 methods & how they work: [`FROZEN_COMPETITORS.md`](FROZEN_COMPETITORS.md).

| # | Model | Method (how) | Backbone | Scope | Cheapest GPU | Est. cost / time | Status |
|---|---|---|---|---|---|---|---|
| 1 | **H2O / SnapKV** | streaming/query-aware KV eviction | Llama-3.1-8B | LME + MAB (H2O) | local 4090 | TBD | 🟢 adapter ready; **codex** owns |
| 2 | **KVzip** | query-agnostic KV compression | Qwen2.5-7B | LME + MAB | H100/A40 | ~$5–10 | ⬜ pending |
| 3 | **M+ / MemoryLLM** | recurrent parametric memory | mplus-8b | LME + MAB | **A40 48GB** | **~$10 / overnight** | 🟡 **prep in progress (me)** |
| 4 | **LCLM** | soft-token compression | 0.6b+4b | LME + MAB | **local 4090** | **$0** (no rental) | ⬜ blocked on HALO GPU |

**Per-model notes (findings already established — do not re-discover):**

- **#1 H2O/SnapKV** — Llama-3.1-8B gate is **OPEN** (verified 2026-07-20, account `alxd219p1`). **H2O now uses
  the first-party infinite-streaming Transformers-5.1 adapter**, not KVCache-Factory: raw retained keys,
  position rolling, 2,048-token cache (1,024 heavy + 1,024 recent), 512-token chunks, query-head GQA mode.
  A real 105,146-token LME-S prefill completed locally in 23.29s (4,514 tok/s) at 17.62GB peak. The maximum
  744,639-token MAB prompt crossed the native 131k window and prefetched in 164.38s; query snapshot forks took
  ~2.5ms and peaked at 18.40GB. All 32 layers retained exactly 2,048 entries. Exact LME prompt stats: p50 105,531,
  p95 107,047, max 107,557 tokens; all fit Llama's 131,072 window with 64 generation tokens. SnapKV remains on
  KVCache-Factory/Transformers-4.44.2 and still needs ≥48GB because it materializes full prompt KV first.
  H2O supports exact context snapshot reuse on MAB; SnapKV remains LongMemEval-only.
- **#2 KVzip** — needs the custom CUDA kernel (`cuda-nvcc` + **`cuda-cccl=12.4.*`** pin) + prebuilt flash-attn
  wheel (baked into bootstrap). Query-agnostic → correct KV baseline for MAB (36 encodes / 3071 Q). Tune
  `--ratio` (KV retained, default 0.3). **TODO:** patch `wrapper.py` to honor `--max-new-tokens` (upstream
  hard-codes 512; fairness vs the 64-capped panel; currently disclosed in meta).
- **#3 M+ / MemoryLLM** — **batching within one model is impossible** (single shared `[32,10240,4096]` pool,
  batch=1 baked in). It's **overhead-bound, not compute-bound** (~0.4 s/inject wall vs ~15–25 ms compute) →
  the cost lever is **bigger `--inject-block-tokens`** (default 512; test {1024,2048} on a 50-item subsample
  first — free ~2× if quality holds). Cheapest = **1× A40** (H100 wasted on an overhead-bound job); multi-GPU
  buys wall-clock only, same $. Needs `flash_attention_2`. First-line answer truncation already in (base model,
  no EOS). Details: memory `project_mplus_batching_verdict`.
  **Prep DONE (local, off-GPU):** `run_memoryllm.py` now records GPU-synced inject/answer timing into
  `meta.timing`; `scripts/baselines/tier2/sweep_memoryllm_blocksize.py` runs the block-size sweep on a fixed
  subsample and prints accuracy-vs-inject-time + a recommended block. First pod action = run the sweep, set
  the block, then the full LME + MAB run.
- **#4 LCLM** — closest competitor (frozen-decoder + soft-token memory). Small enough to run **FULL locally on
  the 4090** (~9–10 GB; ~45–60 min full LME), **no rental**. Blocked only on HALO releasing the GPU.

---

## 3. Budget

| | Cost | Notes |
|---|---|---|
| Tier-1 (done) | **~$17** | LongMemEval ~$14 (full_context ≈90%) + MAB ~$3.30 (deepseek 50× prefix-cache vs ~$66 naive) + judge $0.03 |
| Tier-2 (est.) | **~$25–30** | KVzip ~$5–10 · M+ ~$10 (A40 overnight) · LCLM $0 (local) · H2O TBD |

---

## 4. Doc map
- **Results:** [`PHASE2_REPORT.md`](PHASE2_REPORT.md) / [`.csv`](PHASE2_REPORT.csv) (full tables + BEM calibration + cost log)
- **Panel design:** [`PHASE2_BASELINES.md`](PHASE2_BASELINES.md) · audit fixes [`PHASE2_AUDIT2_FIXES.md`](PHASE2_AUDIT2_FIXES.md)
- **Tier-2 setup:** [`TIER2_GPU_INTEGRATION.md`](TIER2_GPU_INTEGRATION.md) · [`TIER2_HOSTING.md`](TIER2_HOSTING.md) · `scripts/pod/TIER2_RESUME.md`
- **Competitor landscape:** [`FROZEN_COMPETITORS.md`](FROZEN_COMPETITORS.md)
- **API runbook:** [`API_EVAL_RUNBOOK.md`](API_EVAL_RUNBOOK.md) · **MAB schema:** [`MEMORYAGENTBENCH_SCHEMA.md`](MEMORYAGENTBENCH_SCHEMA.md)
