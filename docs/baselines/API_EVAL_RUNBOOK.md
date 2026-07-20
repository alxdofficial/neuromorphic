# Phase-2 API reference baselines — runbook

Tier-1 reference baselines (floor / full-context / RAG) over the OpenRouter API. No local GPU → runs in
parallel with Phase-0 architecture work. Deterministic scoring only (no LLM judge). See
`docs/baselines/PHASE2_BASELINES.md` for the overall plan and `TIER2_GPU_INTEGRATION.md` for the
GPU/matched-decoder baselines.

## Code layout
| Piece | Location |
|---|---|
| dataset text accessors | `src/memory/data/longmemeval.py::load_longmemeval_text` (MemoryAgentBench: TODO) |
| deterministic scorers | `src/memory/eval/longmemeval_score.py` (MemoryAgentBench: TODO) |
| async OpenRouter client + price table | `src/memory/eval/api_client.py` |
| BM25 + dense retrievers | `src/memory/eval/retrieval.py` |
| prompt builders (the 4 modes) | `src/memory/eval/baselines.py` |
| runner CLI | `scripts/baselines/run_api_eval.py` |
| Tier-2 GPU scaffolding | `scripts/baselines/tier2/` |

## Conditions (modes)
- `floor` — question only, no history (parametric + abstention lower bound)
- `full_context` — entire history + question (long-context ceiling; truncated to the model's *served* window)
- `rag_bm25` — BM25 top-k sessions + question
- `rag_dense` — dense (MiniLM/CPU) top-k sessions + question

## Model panel (default, `src/memory/eval/api_client.py::DEFAULT_MODELS`)
`meta-llama/llama-3.1-8b-instruct` (131k) · `google/gemini-2.5-flash-lite` (1M) ·
`deepseek/deepseek-v4-flash` (1M) · `qwen/qwen3.5-flash-02-23` (1M). Two frontier 1M-ctx slots TBD.
Full-context sweep ≈ $17 (floor/RAG add pennies). Prices live in `PRICING`; override with `--models`.

> **Executed panel (2026-07-20):** the completed Tier-1 results used **two** models — `deepseek-v4-flash`
> (1M-ctx frontier anchor) and `llama-3.1-8b-instruct` (reproducible published anchor). gemini/qwen were not
> run. Final numbers: [`PHASE2_REPORT.md`](PHASE2_REPORT.md) · index: [`PHASE2_HUB.md`](PHASE2_HUB.md).

## Usage
```bash
export OPENROUTER_API_KEY=sk-or-...

# 1) preview cost + prompt shapes, NO API calls, no valid key needed:
python scripts/baselines/run_api_eval.py --dataset longmemeval --dry-run

# 2) cheap smoke on a live key (3 questions, all modes):
python scripts/baselines/run_api_eval.py --dataset longmemeval --max-examples 3

# 3) full reference sweep (500 questions, default panel, all modes):
python scripts/baselines/run_api_eval.py --dataset longmemeval --max-examples 500

# subset a model / mode:
python scripts/baselines/run_api_eval.py --dataset longmemeval \
  --models deepseek/deepseek-v4-flash --modes full_context rag_bm25
```
Flags: `--concurrency` (default 8), `--bm25-topk` (5), `--max-tokens` (256), `--no-bem` (skip the BEM
paraphrase model for speed), `--dry-run`. Outputs per-(model,mode) JSON + `<dataset>_api_summary.json`
under `outputs/baselines/`, each with the deterministic aggregate + exact token cost.

## Scoring
`overall_accuracy` (mean over non-abstention), `abstention_accuracy` (the ~30 `_abs` questions),
`per_type` (5-way). EM + containment + BEM paraphrase-equivalence + abstention detection — no LLM judge
(project policy). One-time GPT-4o judge cross-check to disclose the offset is a separate, optional step.

## Notes / gotchas
- **Served context ≠ listed context.** The runner budgets by `top_provider.context_length` (e.g. a model
  listing 131k but serving 32k truncates a 115k history) and logs `n_truncated`. Watch it in the output.
- Dense retrieval pins MiniLM to **CPU** (no GPU contention with Phase-0).
- `cache_read` pricing does NOT help — LongMemEval histories are unique per question (no shared prefix).
- **`:free` models (tencent/hy3, nvidia/nemotron-ultra) are $0 but rate/daily-capped** (~20 req/min; ~50/day
  without ≥$10 credit, ~1000/day with). A full 500×4-mode sweep = 2000 req/model → will hit the daily cap.
  Run free models on a reduced scope: e.g. `--modes full_context` (500 req) or `--max-examples 200`, and
  `--concurrency 2` to avoid 429 storms (the client retries/backoffs, but can't beat a hard daily cap).
- **Nemotron-Ultra is a reasoning model** — it may emit chain-of-thought; raise `--max-tokens` (e.g. 1024) so
  the final answer isn't truncated. Containment/BEM scoring tolerates extra text, but verbose CoT can dilute
  exact-match — consider it a soft caveat for that model.
