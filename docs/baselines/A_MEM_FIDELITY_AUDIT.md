# A-MEM Phase-2 fidelity and cost audit (updated 2026-07-21)

Scope: `scripts/baselines/run_agentmem.py` against the NeurIPS 2025 paper and the evaluation repository at
`/home/alex/code/neuromorphic/baselines/A-mem`, commit `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`.

## Fidelity verdict

The default LongMemEval path now preserves the substantive A-MEM algorithm:

1. Each conversation turn becomes one atomic note with its real session timestamp.
2. The configured LLM generates note keywords, context, and tags.
3. Every new note retrieves five candidate neighbors; the LLM decides link creation and neighbor evolution.
4. A question is rewritten to LLM-generated keywords, embedded, and used for top-k retrieval.
5. Retrieved notes expand through their learned links before the reader answers.
6. `all-MiniLM-L6-v2` is the local text encoder; the configured chat LLM is used for metadata, evolution,
   query rewriting, and answering.

Intentional benchmark differences are recorded in every artifact. LongMemEval replaces the paper's LoCoMo
dataset. The final answer prompt is the shared Phase-2 benchmark prompt, not the LoCoMo evaluation prompt, so
the reader comparison remains controlled. MemoryAgentBench has no paper-defined A-MEM ingestion protocol; it
uses bounded 800-character document notes. That size is an explicit adaptation chosen to leave room for
generated attributes under MiniLM's 256-token embedding limit.

Llama 3.1 8B Instruct is a valid model-agnostic deployment choice, but it is not one of the six main-table
models in the paper (GPT-4o-mini, GPT-4o, Qwen2.5 1.5B/3B, and Llama 3.2 1B/3B).

## Bugs found and fixed

- The previous adapter ingested whole sessions rather than atomic turns.
- It retrieved with the raw question rather than A-MEM's LLM keyword rewrite.
- It called plain top-k retrieval and omitted learned-link expansion.
- MAB answer prompts lost the benchmark system/task templates and were double-wrapped by the generic prompt.
- MAB's old 8,000-character notes were mostly invisible to the 256-token embedding encoder.
- The supposedly CPU-only path could allow SentenceTransformer/Torch to select a GPU.
- Cache names omitted generation-affecting A-MEM settings and could resume incompatible outputs.
- Token usage for A-MEM's internal calls was not observable.
- A redirected `/dev/null` handle was not closed.
- OpenRouter authentication failures were masked by an upstream `UnboundLocalError`; preflight now reports the
  credential failure before loading the embedder or dataset.

The runner now also supports context-preserving LPT process shards (`--num-shards`, `--shard-idx`), with
`merge_agentmem_shards.py` validating and merging their results and token ledgers. Contexts are independent,
but note insertion inside one context remains sequential because memory evolution depends on prior notes.

## Token workload

The runner records API-reported prompt, completion, cached-prompt tokens, and provider-reported cost by phase.
The live measurement remained blocked on 2026-07-21 because the available OpenRouter key returned HTTP 401 from
OpenRouter's key endpoint. The following is therefore a tokenizer measurement/projection, not mislabeled API
usage.

### LongMemEval-S (500 questions)

- 246,744 atomic notes; 494,488 LLM calls (two per note, plus query rewrite and answer per question).
- Gold answer tokens (Llama 3.1 tokenizer): median 3, p95 54, p99 85, maximum 104. Existing Phase-2 Llama
  generations reached 659 tokens without a length cutoff, so the A-MEM final-answer default is 1,024 tokens.
- Exact Llama 3.1 chat-formatted metadata inputs: 115,163,829 tokens.
- Exact query-rewrite inputs: 58,016 tokens.
- Evolution-input proxy using five preceding representative neighbors: 486,754,384 tokens.
- Answer inputs depend on learned links; top-10 plus modest expansion adds roughly 1.5-5M tokens.
- Planning input total: approximately 603-607M tokens.
- Planning output assumption: 80 tokens/metadata response, 220/evolution, 25/query rewrite, 64/answer =
  74,067,700 output tokens. The strict configured output ceiling is 494,500,000 tokens (upstream's 1,000
  token cap for each internal structured call plus 1,024 per final answer). The expectation must be replaced
  by the instrumented API measurement after key refresh.

Price projections, separating input and output spend:

| LongMemEval-S scenario | input tokens | output tokens | $0.02/$0.03 input/output | $0.05/$0.08 input/output |
|---|---:|---:|---:|---:|
| expected | ~605M | ~74.1M | $12.10 + $2.22 = **$14.32** | $30.25 + $5.93 = **$36.18** |
| output-cap planning ceiling | ~607M | 494.5M | $12.14 + $14.84 = **$26.98** | $30.35 + $39.56 = **$69.91** |

The second row is a billing ceiling under the configured completion caps and high input estimate, not an
expected generation volume. Provider choice and routing belong in the final artifact.

### MemoryAgentBench (3,071 scored questions; explicit adaptation)

- 36 reusable contexts, 41,647 document notes, 89,436 calls.
- Across all acceptable golds, answer tokens have median 6, p95 18, p99 23, and maximum 41. The A-MEM
  final-answer default is 256 tokens; this leaves more than 6x headroom while bounding pathological rambling.
- Exact metadata inputs: 17,175,859 tokens; exact query inputs: 689,655.
- Five-neighbor evolution proxy: 68,606,165 input tokens.
- Before learned-link answer expansion: 86,471,679 input tokens; planning total is roughly 94-108M.
- The same completion assumptions (80 metadata, 220 evolution, 25 query, 40 answer) give about 12.7M expected
  output tokens. The strict configured output ceiling is 87,151,176 tokens.

| MemoryAgentBench scenario | input tokens | output tokens | $0.02/$0.03 input/output | $0.05/$0.08 input/output |
|---|---:|---:|---:|---:|
| expected | ~101M | ~12.7M | $2.02 + $0.38 = **$2.40** | $5.05 + $1.02 = **$6.07** |
| output-cap planning ceiling | ~108M | 87.2M | $2.16 + $2.61 = **$4.77** | $5.40 + $6.97 = **$12.37** |

## Verification

`python -m py_compile` passes. The complete repository suite passes: 126 tests, including ten focused A-MEM
protocol, prompt, metering, cache-signature, and shard tests. A paid one-example live run must be repeated after
refreshing `OPENROUTER_API_KEY`; the current runner will fail cleanly at preflight until then.
