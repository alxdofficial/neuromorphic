# Phase-2 Baseline Results — living log

**Updated as runs land.** Deterministic scoring (no LLM-judge for the panel); one-time GPT-4o-judge
cross-check for calibration only. Raw per-question stores: `outputs/baselines/cache/*.jsonl`;
per-cell JSON: `outputs/baselines/*.json`; regenerate the table with `python scripts/baselines/report.py`.

## Panel + config
- **Models:** `llama-3.1-8b-instruct` (recognized reproducible anchor; published LongMemEval judge #) +
  `deepseek-v4-flash` (current 2026 frontier-flash, 1M ctx). GPT-4o = **cited**, not run (see below).
- **Conditions:** floor (no memory) · full_context (whole history in prompt) · rag_bm25 (top-5 retrieval).
  `rag_dense` dropped (near-duplicate of bm25). Tier-2 memory *mechanisms* (KVzip/SnapKV/H2O, MemoryLLM-M+,
  LCLM) are the actual head-to-heads — separate, GPU/pod.
- **Scoring:** LongMemEval = EM + containment(negation-guarded) + BEM paraphrase; abstention = refusal
  detector; deterministic, reproducible. MemoryAgentBench = deterministic substring/exact-match per
  competency (NO BEM). Full context budget reserves 16k tokens under the served window.

## ⚠ Scoring-calibration finding (GPT-4o judge cross-check, 100 full_context items)
- deterministic **BEM@0.5 = 0.720** vs **GPT-4o judge = 0.590** → we run **+0.13 LENIENT** (opposite of the
  usual "deterministic is stricter" assumption). Agreement 81%. FN (we-wrong/judge-right) 3%; **FP
  (we-right/judge-wrong) 16% — ALL from BEM** crediting clear errors ("5" for gold "6", refusals,
  `Dark Souls 3` for `…3 DLC`).
- Threshold sweep on full_context: BEM@0.5 over-credits; **BEM@0.9 ≈ the published judge for llama**
  (0.446 vs 0.454); EM+containment-only is too strict.
- **DECISION PENDING (user):** raise default BEM threshold 0.5 → ~0.85 and re-score (free, cached), OR
  report EM+containment as conservative primary with BEM@0.5 as a lenient upper bound. Numbers below show
  the range until this is locked.

## LongMemEval-S (500 Q) — DONE (deterministic)
overall_accuracy (non-abstention), then abstention accuracy separately. `full_context` shown as a range
across BEM thresholds (off / @0.5-as-run / @0.9); floor & rag are BEM@0.5-as-run pending the threshold lock.

| model | floor | rag_bm25 | full_context (off / @0.5 / @0.9) | abstention (floor/full/rag) | coverage |
|---|---|---|---|---|---|
| **llama-3.1-8b** | 0.013 | 0.462 | 0.345 / **0.588** / 0.446 | 0.97 / 0.50 / 0.80 | full_context 94.4% (28 histories > llama 131k ctx) |
| **deepseek-v4** | 0.009 | 0.635 | 0.581 / **0.728** / 0.672 | 1.00 / 0.97 / 1.00 | ≥99.8% |

Reads: floor≈0 (no memory → can't answer, both abstain well) → RAG → full_context, monotonic. **deepseek >
llama everywhere.** deepseek full_context (even judge-calibrated ~0.67) tops GPT-4o's published 0.606 — it's
a 2026 model. llama full_context @0.9 (0.446) ≈ the published Llama-3.1-8B (0.454) → our scorer, tuned,
reproduces the reference.

## Published reference — CITED (paper GPT-4o judge, NOT our scale)
_Wu et al., LongMemEval-S, arXiv:2410.10813 — gpt-4o-2024-08-06 judge:_
| model | full-context | oracle |
|---|---|---|
| GPT-4o | 0.606 | 0.870 |
| Llama-3.1-8B | 0.454 | 0.710 |
| Llama-3.1-70B | 0.334 (worse than 8B) | 0.744 |
| Phi-3-Medium-128k | 0.380 | 0.702 |

(MemoryAgentBench published table lives in `src/memory/eval/published_baselines.py`, rendered by report.py.)

## MemoryAgentBench (3,071 Q, 4/5 competencies) — ⏳ RUNNING
Deterministic string-match (substring/exact per competency; no BEM). Recsys excluded (opt-in Recall@5).
**No `--max-context-chars` cap** — every question runs (the earlier cap SKIPPED whole competencies; removed).

**Context sizes are large** (deepseek-tokenizer): AR ~158k (max 881k), TTL ~109k, LRU ~133k, CR ~57k tok —
and MAB is "inject once, query many": only **36 distinct contexts** underlie all 3,071 Q (AR: 17 ctx→1,700 Q).
So the run matrix is scoped to what is *meaningful and affordable* per model:

| | floor | rag_bm25 | full_context |
|---|---|---|---|
| **llama-3.1-8b** | ✅ all | ✅ all | ❌ N/A — MAB contexts (158k–881k) exceed its 131k window; truncation drops the needle → RAG is llama's condition |
| **deepseek-v4** | ✅ all | ✅ all | ✅ all (1M window; the long-context ceiling) |

**full_context caching (the cost lever):** re-sending each 158k context ~85× would cost ~$66. Instead we pin
the OpenRouter provider to **`deepseek` first-party** (`allow_fallbacks:false`) — the only endpoint with
implicit prefix caching at a **50× cache-read discount** ($0.0028/M vs $0.14/M) — and **warm each context
once** before parallelizing, so repeats bill at the cached rate. Probe: **100% cache hits, answers correct.**
Requires the OpenRouter account to allow the DeepSeek provider (data-policy toggle; public-benchmark data).
Cache-aware cost accounting in `run_api_eval.py` (`--pin-provider`, `cached_tokens`). _Table on completion._

## Cost + coverage log
| run | cost | notes |
|---|---|---|
| LongMemEval llama+deepseek (floor/full/rag) | ~$14 | full_context is ~90% of it; 74 llama errors first pass (46 transient rate-limit fixed on rerun, 28 too-big remain) |
| judge cross-check (100 items, gpt-4o) | $0.03 | calibration |
| MemoryAgentBench floor+rag (both models, 3,071 Q) | ~$1.30 | running |
| MemoryAgentBench deepseek full_context (3,071 Q, deepseek-pinned 50× cache) | ~$2–3 | running; naive re-send would be ~$66 |

## Open items
- Lock the BEM threshold + re-score LongMemEval uniformly.
- 28 llama full_context context-length errors: accept @94.4% coverage, or bump reserve once more.
- Tier-2 mechanism baselines (GPU/pod) + our own model — later.
