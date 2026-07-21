# Phase-2 Baseline Establishment — Plan & Runbook

**Status (updated 2026-07-21): Panel-B (native-scale) EXECUTING.** Tier-1 (API long-context + RAG) is **DONE**
on both benchmarks; Tier-2 (GPU memory mechanisms) runs as a **sharded pod fleet** (`scripts/pod/mpod.py`) —
H2O@2% complete on both; **M+ complete on LongMemEval-S (n=500)** with its MemoryAgentBench run still going;
KVzip running, H2O@20% planned, A-MEM pending. **SCOPE FIXED 2026-07-21 — the panel is exactly five arms:
M+, H2O@2%, H2O@20%, KVzip, A-MEM. SnapKV, LCLM and `memoryllm-8b` are DROPPED (cite-only).** **Panel-A
(matched-135M)** remains **DEFERRED** until our own model exists (it's the fair fight our number slots into).
This file is the design rationale; **live status + results = [`PHASE2_HUB.md`](PHASE2_HUB.md)**.
Establish reference numbers for established SOTA memory/long-context baselines on a fixed benchmark +
fixed harness *before* our memory layer exists, so our own number later drops into the same table.
Operationalizes the Phase-2 section of `docs/data/DATA_PHASES_PLAN.md` with concrete baselines, compute,
scoring.

> **REVISION 2026-07-18** (external cited peer-review, 2 load-bearing claims web-verified). Adopted, see
> memory `project_phase2_baseline_run` for full rationale:
> 1. **Whitespace narrower** — EpiCache (VERIFIED, arXiv:2509.17396) + TRIM-KV are already on LongMemEval
>    as *KV-cache* compression; our true claim = **first SOFT-TOKEN / trainable-encoder compressor on LME**.
>    Keep LME; cite them.
> 2. **MemoryAgentBench → judge-free CO-PRIMARY** (VERIFIED deterministic: AR+CR=`substring_exact_match`,
>    LRU+TTL=`exact_match`, Recsys=`Recall@5`; drop only `longmemeval`+`infbench_sum` judge subsets).
> 3. **Decoder: 135M for DEV now** (objective wall is scale-independent), **paper = 135M→360M→1B→3B ladder
>    + one 7–8B** confirmatory (matched-budget controlled study). Don't retrain at 3B until it binds.
> 4. **Panel B de-scoped** — cite published GPT-4o 60.6 / oracle 87, don't pay ~$144/pass; keep one local
>    Llama-3.1-8B ceiling. **MemoryLLM → M+ only** (8B can't hold 115k). **KV baselines → KVzip+SnapKV**
>    (KVCache-Factory ⚠ truncates to 7,500 tok by default). **Avoid LoCoMo.**

> **REVISION 2026-07-18b** (memory-MECHANISM competitor sweep; cited peer-review + **LCLM web-verified from
> the primary source** — [arXiv:2606.09659](https://arxiv.org/abs/2606.09659) + HF `latent-context`). The
> mechanism baselines are now split into **2a** (architectural / memory-as-state — our true head-to-heads)
> and **2b** (agent-memory frameworks — RAG-adjacent, on a big frozen LLM); see **§2.5**. Headline addition:
> **LCLM** — an encoder→soft-token→adapter→**frozen-init-4B-decoder** compressor AT SCALE **with released
> weights**, which benchmarks the same SnapKV/KVzip baselines we do → our closest concurrent competitor;
> engage head-on (novelty note in §2.5). Corrections applied vs the source addendum: **Memory-R1 code is NOT
> released** (cite-only, not a runnable 2b); **A-MEM repo = `WujiangXu/A-mem`** (not `agiresearch`).

> **REVISION 2026-07-20c:** **LCLM dropped from the runnable Phase-2 panel by project decision.** Retain its
> related-work/novelty analysis below, but do not schedule or report an LCLM benchmark row.

Companion docs: `docs/baselines/FROZEN_COMPETITORS.md` (landscape), `docs/data/DATA_PHASES_PLAN.md`
(two-phase plan), `docs/RESULTS.md` (Phase-0 results log).

> **Directive (supervisor, 2026-07-17):** move past architecture-only work → take established SOTAs
> (parametric + nonparametric), pick a real-world established memory/long-context eval, and RUN those
> baselines first. Later drop our trained memory layer into the same harness.

Grounded by a 2026-07-17 three-agent research sweep (benchmark landscape / runnable baselines / scoring)
— primary-source-verified; every vendor self-report treated as noise until re-run.

---

## 1. Benchmarks

| role | benchmark | why |
|---|---|---|
| **headline** | **LongMemEval-S** (`xiaowu0162/longmemeval-cleaned`, ~115k tok, 500 Qs, 5-way taxonomy) | de-facto standard for multi-session memory (mid-2026); **whitespace: no soft-token-prepend compressor — our exact class — has ever published on it** |
| **thesis-axis support** | **MemoryAgentBench** (ICLR'26) | its Selective-Forgetting + Test-Time-Learning competencies map onto our streaming-write / forced-forgetting thesis — the axis LongMemEval doesn't test |
| deferred supports | RULER, BABILong | contamination-proof synthetics where RMT/Titans compete; add in a later pass |
| **cross-ref only** | LoCoMo | integrity-flagged (Penfield audit: 6.4% of answer key wrong; judge accepts 63% of wrong answers). Never headline. |

**Anchor numbers (primary paper ONLY — vendor numbers span 36–94% for one system):**
GPT-4o full-ctx **60.6%** / +Chain-of-Note 64.0% / RAG@top-k ~71% / oracle **87.0** (+CoN 92.4).
Llama-3.1-8B-128k full-ctx **45.4%** (70B *worse* at 33.4% = long-context degradation). Phi-3-med 38.0%.

---

## 2. Two baseline panels

The fairness problem: SOTAs run at native 7–8B / GPT-4o; our decoder is a frozen **SmolLM2-135M**. A raw
accuracy table isn't apples-to-apples. So we report two panels.

### Panel A — matched-decoder (the fair fight our model slots into)
Everything reads the **same frozen SmolLM2-135M**; win = accuracy at the smallest memory footprint
(Cartridges KV-ratio framing). At 135M you **cannot** read 115k tokens — compression IS the point —
so "full-context" is not a Panel-A entry (only meaningful on the oracle split / native scale).

| baseline | param? | mechanism | compute |
|---|---|---|---|
| no-memory floor | — | `vanilla_llama` (question only, no context) | 4090 ✓ (have) |
| RAG (BM25) | nonparam | BM25 over sessions → top-k → 135M reader | 4090 ✓ |
| RAG (dense) | nonparam | Stella-V5 / GTE-Qwen2 embed → top-k → 135M reader | 4090 ✓ |
| KV-eviction | nonparam | **H2O = first-party adapter `src/memory/eval/h2o_llama.py`** (ported policy, not a vendored repo), run at TWO budgets (2% and 20%). ~~SnapKV/StreamingLLM~~ DROPPED 2026-07-21 | 4090 ✓ (H2O@2%); 48GB for H2O@20% |
| **OURS** | parametric | streaming-compress → 96 slots + edges → 135M | 4090 ✓ |

### Panel B — native-scale reference (the "what's the ceiling" numbers)

**Reconciliation of REVISION 2026-07-18 §4 ("Panel B de-scoped"):** the de-scoping applies to the **GPT-4o
row only** — we do not pay ~$144/pass for it, we cite the published 60.6 / oracle 87.0. The rest of Panel B
**did run**, substituting a cheap 1M-context frontier model (`deepseek-v4-flash`) for the GPT-4o slot; see
[`PHASE2_HUB.md`](PHASE2_HUB.md) §1.

| baseline | param? | mechanism | compute | status |
|---|---|---|---|---|
| ~~GPT-4o full-ctx / RAG~~ | nonparam | official harness, OpenAI API | ~$144/pass | **CITE-ONLY (de-scoped 2026-07-18)** — published 0.606 full-ctx / 0.870 oracle |
| deepseek-v4-flash full-ctx / RAG (1M ctx) | nonparam | our harness, OpenRouter | API, ~$17 total | ✅ **DONE** — LME 0.687 full-ctx |
| Llama-3.1-8B-128k full-ctx | nonparam | 115k-tok single prompt | OpenRouter (served 131k) | ✅ **DONE** — LME 0.462 (94.4% coverage) |
| **M+** | parametric | `YuWangX/mplus-8b` — online session-by-session write. ~~`memoryllm-8b`~~ DROPPED 2026-07-21 (M+ only) | fits 24GB VRAM; **MAB needs ≥116GB host RAM** | ✅ **LME DONE — 0.423, the first LongMemEval number for M+**; MAB running |

Skipped as headline: Mem0/Zep/Letta (numbers 36–94% for one system; need an API backend; only worth it
re-run under our fixed harness).

> **Mechanism competitors (Cartridges / Larimar / MemoryLLM-M+ / one agent-memory system) are all
> native-scale too → they slot into THIS panel.** Their mechanism classification (2a vs 2b), priority, and
> the cite-only LCLM novelty note live in §2.5 to keep this scale-table uncluttered.

---

## 2.5 Memory-mechanism competitors — 2a architectural vs 2b agent-memory

Panels A/B are a **scale** cut (matched-135M vs native). Orthogonal to it, the memory MECHANISMS we compete
against split two ways — keeping them straight avoids fighting the wrong zoo:

- **2a — architectural / "memory-as-state":** a trainable memory read by a (mostly) frozen decoder, or the
  model itself IS the memory. **Our true head-to-heads** (same family; matched-decoder-able in principle).
- **2b — agent-memory frameworks:** an external text/graph store + retrieval bolted onto a big frozen LLM
  (RAG-adjacent). A different paradigm — include **ONE** as the paradigm reference; anchor with
  full-context + naive-RAG (already our Tier-1). Do **not** chase the tail (LiCoMemory / TiMem / SGMem / …
  — self-reported, mutually contradictory; the EMem "simple strong baseline" shows raw-text + good
  embeddings rivals the elaborate pipelines).

Every 2a **ADD** below is 0.6–8B → **Panel B (native)**, not matched-135M (a matched-135M version =
reimplementation). Rule for ALL of these: **re-run under our fixed harness; never quote a paper's number**
(Mem0/Zep moved 84→58 under re-evaluation).

### 2a — architectural (priority-ordered)
| baseline | status | mechanism | scale / GPU | why |
|---|---|---|---|---|
| **LCLM** | **DROPPED (cite-only)** | encoder → soft tokens → adapter → **frozen-init 4B** decoder, e2e continual-pretrain, 4×/8×/16× | Qwen3-Emb-0.6B enc + 4B dec; released weights | related-work comparator only; no Phase-2 run planned |
| Larimar | optional | Kanerva episodic-memory matrix conditioning a frozen-ish decoder | 1.3B/6B; train-your-own | classic architectural analog / "memory-module baseline" |
| KV-compression | **have** | KVzip + H2O (2% and 20% budgets) evict the 115k-tok KV cache. ~~SnapKV~~ DROPPED | Llama/Qwen 7–8B; 48GB pod | the KV-eviction branch |
| M+ | **have; LME DONE** | parametric in-weights memory, online session write | `mplus-8b`; 24GB VRAM, but **≥116GB host RAM for MAB** | only mechanism-with-released-weights parametric baseline. ~~`memoryllm-8b`~~ DROPPED — M+ only (an 8B pool can't hold 115k) |

**Cite-only** (a related-work STRENGTH: "prior work claims X, ships nothing runnable-for-our-task, so we
compare against released alternatives"): **Titans / ATLAS / Miras** (Google — code promised, never shipped),
MELODI, B'MOJO, CAMELoT (no weights); **Cartridges** (`HazyResearch/cartridges`, Apache-2.0 — has code, but
trains a KV cache **per corpus**: on LongMemEval's 500 private haystacks that's 500 cartridges = infeasible &
pointless; **dropped as a runnable baseline 2026-07-18b**, keep as the trainable-frozen-soft-KV citation).
Optional recurrent-state branch: RWKV-7-Goose 2.9B (runnable) — skip unless we add the delta-rule family.

### 2b — agent-memory (add exactly ONE)
| baseline | status | note |
|---|---|---|
| **A-MEM** (NeurIPS'25, `WujiangXu/A-mem`, MIT) | **ADD (default)** | canonical Zettelkasten agentic memory; runs over a frozen LLM (our OpenRouter path) → minimal GPU |
| MemoryOS (EMNLP'25 Oral, `BAI-LAB/MemoryOS`) | alt | cleanest citable "memory-OS"; swap for A-MEM if preferred |
| Memory-R1 (ACL'26) | **cite-only** | code "coming soon" (NOT released); already cited in OBJECTIVES.md as a GRPO ref |

### LCLM — cite-only novelty note
LCLM overlaps our core (soft-token compression of a long context into a decoder's latent input) and ships
at 4B with weights → it **dents** the "first soft-token compressor at scale" framing and must be **addressed,
not cited in passing.** What is STILL ours: (1) LCLM **trains the decoder end-to-end** (frozen-init → 350B-tok
continual pretrain); **we keep the decoder FROZEN** — compose a memory onto an unmodified LM (cheaper, no
decoder retrain, composable). (2) LCLM is a **static** compressor; our axis is **streaming write/update +
forced forgetting** (stability–plasticity) — LCLM doesn't test it. (3) our compositional discrete-vocab-over-
graph angle. One-liner: LCLM = "compression-at-scale by training everything"; ours = "frozen-decoder implicit
memory cache with streaming update." **Read the paper directly before finalizing the framing.**

---

## 3. Scoring — deterministic, policy-compliant (+ judge cross-check)

LongMemEval ships **only** a GPT-4o judge (`src/evaluation/evaluate_qa.py`), which violates our
no-LLM-judge policy. We use `src/memory/eval/longmemeval_score.py` (built 2026-07-17, validated):

- **factual** types → normalize (SQuAD-style + number-word canonicalization + parenthetical alternates)
  → EM | containment | **BEM** paraphrase (`kortukov/answer-equivalence-bem`, threshold 0.5)
- **preference** → rubric keyword-coverage (proper-noun gate + ≥50% salient-token coverage)
- **abstention** (`_abs`, ~30/500) → fixed refusal-lexicon detector (recall-biased)
- report the three official-harness numbers: **overall** (micro, non-abstention) / **task-averaged**
  (mean of per-type means) / **abstention** accuracy

**+ one-time GPT-4o-judge cross-check on ~100 Qs** to quantify & disclose the offset (our deterministic
number sits *below* the judge number — the judge credits paraphrase/superset answers). Report ours as
"deterministic (EM+containment+BEM)", explicitly *not* the official metric.

**Freeze for fairness** (config alone swings published numbers 10–40 pts): cleaned Sept-2025 data
revision · retrieval top-k + granularity (session vs turn) · reader prompt (`direct` vs Chain-of-Note) ·
max-gen-tokens · denominator (overall vs task-avg vs abstention).

---

## 4. Infra: have / reuse / build

- **Have:** `src/memory/data/longmemeval.py` (real reader, 500 Qs, taxonomy + abstention); floor/ceiling
  `vanilla_llama` / `vanilla_full_context`; H2O eval-only; **the deterministic scorer (done).**
- **Have (built 2026-07-18, audit-hardened; suite now 133 tests, `.venv/bin/python -m pytest tests/ -q`):**
  the **Tier-1 API reference harness** —
  `scripts/baselines/run_api_eval.py` (floor / full-context / RAG-bm25 / RAG-dense over OpenRouter, token-
  accurate budgeting, resumable per-question store, coverage/cost accounting) + `src/memory/eval/` scorers
  for BOTH LongMemEval and MemoryAgentBench (judge-free, per-competency prompts verbatim from the MAB repo)
  + `scripts/baselines/tier2/` runners (KVzip/H2O + M+; the SnapKV and LCLM code paths are retained but
    INACTIVE — both dropped from the panel)
  + `run_agentmem.py`
  (**A-MEM/MemoryOS**, no GPU). Cartridges dropped (cite-only).
- **Reuse, don't build:** official harness `xiaowu0162/LongMemEval` (MIT) = retrieval + generation +
  judge with pluggable vLLM readers → Panel-B GPT-4o/8B mostly config.
- **Build:** Panel-A matched-decoder runner (135M reader + BM25/dense RAG + KV-eviction) · MemoryLLM/M+
  adapter (online write loop) · our-model adapter (last).

---

## 5. Execution order (cheapest-first)

1. ✅ **Deterministic scorers + Tier-1 API harness** (`src/memory/eval/`, `scripts/baselines/run_api_eval.py`)
   — built + audit-hardened (suite now 133 tests). Shared prerequisite. **DONE.**
2. ✅ **Tier-1 API reference run** — floor / full-context / RAG-bm25 (k5 + k15) over the OpenRouter panel on
   LongMemEval **and** MemoryAgentBench. No GPU. **DONE** (`deepseek-v4-flash` + `llama-3.1-8b-instruct`;
   RAG-dense not run). Numbers: [`PHASE2_REPORT.md`](PHASE2_REPORT.md).
3. **Panel-A on 4090** — 135M reader: floor → BM25 RAG → dense RAG → KV-eviction (matched-decoder fight).
   **DEFERRED** until our own model exists.
4. **2a mechanism competitors** (native / Panel-B), cheapest-first — **partly DONE**:
   a. ✅ **H2O** (local 4090, both benchmarks). ✅ **MemoryLLM/M+ on LongMemEval-S** (14-pod sharded fleet).
      🔄 **M+ on MemoryAgentBench** running (needs ≥116GB-RAM pods). 🔄 **KVzip** running on a Blackwell pod.
      ⏳ **H2O@20%** planned (#224) — the current H2O is a 1.95% KV budget, ~10× more aggressive than the
      paper, and sits at the no-context floor; the 2%/20% pair is the compression-ratio ablation.
   b. ~~SnapKV~~, ~~LCLM~~, ~~Cartridges~~, ~~`memoryllm-8b`~~ and Larimar are NOT runnable panel members —
      cite-only. Scope fixed 2026-07-21: the panel is M+, H2O@2%, H2O@20%, KVzip, A-MEM.
5. **2b — ONE agent-memory system** (**A-MEM** default) over a frozen LLM via the OpenRouter path (min GPU).
   ⏳ not started; cost table in `A_MEM_FIDELITY_AUDIT.md` is a projection blocked on a 401'd key.
6. **Panel-B native ceilings** — ✅ **DONE via the Tier-1 API path** (llama-3.1-8b + deepseek-v4-flash
   full-context) with the one-time GPT-4o judge cross-check for BEM calibration. **GPT-4o itself is cite-only**
   (de-scoped 2026-07-18; ~$144/pass) — do not schedule it.
7. ✅ **MemoryAgentBench** across the same Tier-1 set (thesis-axis support) — **DONE, 3,071 Q**; rescored
   2026-07-21 after the detective_qa/ICL scorer fix.
8. **Our trained model** into the same harness → the headline row.

Rule for steps 4–5: **re-run under our harness, never quote paper numbers.** Results land in
`docs/RESULTS.md` (Phase-2 section) as they complete.

---

## Changelog
- **2026-07-17** — doc created; decisions approved (both panels, deterministic+judge, LongMemEval +
  MemoryAgentBench); scorer built + validated.
- **2026-07-18** — Tier-1 API reference harness built (`scripts/baselines/`, `src/memory/eval/`) + Tier-2
  GPU scaffolds; two external audits (harness fidelity) fixed + re-verified. See
  `PHASE2_AUDIT2_FIXES.md` (test counts there are historical snapshots; the current suite is **133**).
- **2026-07-18b** — memory-MECHANISM competitor sweep adopted: **§2.5** splits mechanism baselines into 2a
  (architectural) / 2b (agent-memory); **LCLM added (must)** as our closest concurrent competitor
  (web-verified, novelty note); Cartridges added; A-MEM = the one 2b; Larimar optional; Memory-R1 → cite-only.
- **2026-07-20c** — LCLM dropped from the runnable panel by project decision; retained as cite-only related
  work.
- **2026-07-21b** — **PANEL SCOPE FIXED at five arms: M+, H2O@2%, H2O@20%, KVzip, A-MEM.** SnapKV and
  `memoryllm-8b` dropped by project decision (joining LCLM). H2O split into two budgets after the 2% run was
  found to sit at the no-context refusal floor (95.8% refusals vs 93.0% for no context at all).
- **2026-07-21** — §5 execution order reconciled with reality (Tier-1 + MAB marked DONE; H2O + M+/LongMemEval
  done); Panel-B contradiction resolved (**GPT-4o row cite-only, rest of Panel B ran**); H2O attribution fixed
  (first-party `src/memory/eval/h2o_llama.py`, not `apple/ml-epicache`); test count synced to 133. All MAB
  numbers regenerated after the detective_qa/ICL scorer fix (`aab14e9`) — see `PHASE2_REPORT.md`.
