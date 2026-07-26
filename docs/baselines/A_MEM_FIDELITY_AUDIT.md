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
6. `all-MiniLM-L6-v2` is the text encoder, either in-process or behind the shared local GPU embedding
   service. The controller LLM performs metadata generation, evolution, and query rewriting; the separately
   configured reader LLM performs the final benchmark-standard answer.

Intentional benchmark differences are recorded in every artifact. LongMemEval replaces the paper's LoCoMo
dataset. The final answer prompt is the shared Phase-2 benchmark prompt, not the LoCoMo evaluation prompt, so
the reader comparison remains controlled. MemoryAgentBench has no paper-defined A-MEM ingestion protocol; it
uses bounded 800-character document notes. That size is an explicit adaptation chosen to leave room for
generated attributes under MiniLM's 256-token embedding limit.

The default controller, Mistral Nemo, is a model-agnostic deployment choice rather than one of the six
main-table models in the paper (GPT-4o-mini, GPT-4o, Qwen2.5 1.5B/3B, and Llama 3.2 1B/3B). Llama 3.1 8B
remains the shared Phase-2 reader. Both roles and their providers are recorded explicitly, so controller-model
quality is not silently attributed to the reader.

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
- Upstream commit `0c8039f` calls `re.sub` while parsing note metadata but does not import `re`. This silently
  replaced valid generated metadata with empty keywords and `General` context. The adapter injects the missing
  standard-library module at runtime without modifying the pinned checkout.
- Internal metadata/evolution/query calls had no retry even though the final answer call did. They now receive
  the same five-attempt exponential-backoff protection at the OpenAI client boundary.
- OpenRouter provider routing was implicit. Separate `--controller-provider` and `--answer-provider` pins now
  disable fallbacks, enter the cache identity, and receive role-appropriate preflights. The controller must
  advertise strict structured output; the ordinary final-answer endpoint only has to exist. Legacy
  `--llm-model` and `--openrouter-provider` flags still configure both roles together, but cannot be mixed
  ambiguously with the split-role flags.
- The upstream evolution schema permits any integer in `suggested_connections`, but linked retrieval assumes
  every integer is an existing note index. A hallucinated out-of-range target therefore caused an
  `IndexError`. The adapter now preserves every valid learned edge, drops only non-existent targets, and
  reports `invalid_links_dropped` in the artifact.
- A provider can occasionally return a successful HTTP response whose structured body is `null`, malformed,
  missing required fields, or truncated at the original 1,000-token limit. The runner validates the actual
  body against the requested upstream schema, meters every billed attempt, retries bad responses, and only
  raises the internal cap (up to 4,000 by default) when the provider explicitly reports a length cutoff.
- Importing evaluation helpers previously initialized the entire local PyTorch model stack, and every process
  separately loaded MiniLM. The memory package now lazily exposes its public model classes, while A-MEM can
  use one GPU embedding service with lightweight, Torch-free orchestration clients. The evaluated note/link
  implementation still comes from the pinned upstream A-MEM checkout.

The runner now also supports context-preserving LPT process shards (`--num-shards`, `--shard-idx`), with
`merge_agentmem_shards.py` validating and merging their results and token ledgers. Contexts are independent,
but note insertion inside one context remains sequential because memory evolution depends on prior notes.

## Shared GPU embedding service

Start one MiniLM service on the 4090:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/baselines/amem_embedding_service.py \
  --device cuda --port 8765 --batch-size 128 --max-wait-ms 5
```

Each A-MEM shard then adds `--embedding-service-url http://127.0.0.1:8765`. The service accepts concurrent
requests, dynamically batches them, and returns compact base64 float32 matrices. Workers retain their own
independent note graphs and API orchestration; only embedding inference is centralized. Its health endpoint
reports operational counters, while artifacts retain only stable service configuration so separately started
shards remain merge-compatible.

Measured on the local RTX 4090, the service reserves about 0.6 GiB of VRAM. A live oracle worker used about
107 MiB steady-state RSS, versus roughly 2.4 GiB before the lightweight path. CPU and GPU MiniLM vectors had
maximum absolute difference `1.34e-7`, cosine similarity `1.0`, and identical sample retrieval ranking. A
64-client load test embedded 256 sentences in 1.03 seconds: five GPU batches, maximum batch 128, zero
failures. Dataset parsing still has a temporary high-water mark before heap trimming, so large fleets should
stagger process starts rather than launch every shard in the same instant.

## Token workload

The runner records API-reported prompt, completion, cached-prompt tokens, and provider-reported cost by phase.
OpenCode's separately stored OpenRouter credential was valid after trimming its trailing newline, enabling the
following paid measurements on 2026-07-21.

### Live provider compatibility and billing measurement

Strict A-MEM controller calls use `response_format.type=json_schema`. Exact-schema probes pinned with
fallbacks disabled found that WandB was the only current Llama 3.1 8B endpoint advertising and accepting
`structured_outputs`; DeepInfra, Novita, Groq, and Cloudflare returned no compatible endpoint for that exact
request. This is an endpoint-capability mismatch for that OpenRouter model route, not evidence that those
providers never support structured output.

The selected default therefore separates the roles without relaxing A-MEM's schemas:

- **Controller:** `mistralai/mistral-nemo` pinned to DekaLLM. Both exact upstream metadata and evolution schema
  probes passed. Its listed input/output rates were $0.018/$0.03 per million tokens.
- **Final reader:** `meta-llama/llama-3.1-8b-instruct` pinned to DeepInfra at listed rates of $0.02/$0.03 per
  million tokens. Final answers are ordinary text and do not require structured-output support.

The earlier one-model Llama/WandB measurement below remains useful as an empirical workload measurement. A
three-context oracle pilot then directly compared the split-role controllers over the same 94 notes:

| controller / provider (Llama/DeepInfra reader) | calls | input | output | cache-read | billed cost | elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Mistral Nemo / DekaLLM | 193* | 311,838 | 54,993 | 0 | **$0.007293** | 30.8m |
| Mistral Small 24B / DeepInfra | 194 | 354,608 | 80,796 | 0 | **$0.022789** | 36.3m |

\* All 191 controller calls completed. One of three reader calls was skipped when the pilot exposed the
invalid-link bug above; the subsequent guard fixes that failure. The controller ledger, which dominates cost,
is complete.

A current split-role end-to-end oracle canary (36 notes) completed all 74 expected calls with zero item
errors. One malformed evolution response was detected and recovered by retry. OpenRouter reported 106,636
input tokens, 19,392 output tokens, no cache-read tokens, and **$0.002512084** total; elapsed time was 11.4
minutes. The final reader accounted for only $0.00010909 of that bill.

Extrapolating the direct Nemo phase rates gives approximately **782M input tokens, 144M output tokens, and
$18.4 for LongMemEval-S**. The measured MAB workload projects **148M input, 28.6M output, and about $3.5** at
the selected split rates: roughly **930M input, 173M output, and $21.9 total**. Reserve $25–30 for generation
variance and retries. These are planning estimates; the full artifacts use API-reported phase-level usage and
cost.

| paid end-to-end sample (WandB pin) | calls | input | output | cache-read subset | API-reported cost |
|---|---:|---:|---:|---:|---:|
| LongMemEval oracle item, 36 notes | 74 | 113,733 | 20,597 | 7,392 | **$0.02955260** |
| MAB fact context, 34 × 800-char notes, 1 Q | 70 | 102,937 | 23,269 | 9,216 | **$0.02776532** |

The cache-read tokens did not reduce these bills: WandB currently lists cache reads at the same $0.22/M as
ordinary input. DeepInfra and Cloudflare list no cache-read price for this model.

Extrapolating phase-specific per-note and per-question rates from those samples to the exact workloads gives:

| dataset | projected input | projected output | strict WandB | DeepInfra counterfactual* | Cloudflare counterfactual* |
|---|---:|---:|---:|---:|---:|
| LongMemEval-S | 682.0M | 140.9M | **$181.03** | $17.87 | $144.10 |
| MemoryAgentBench | 148.2M | 28.6M | **$38.91** | $3.82 | $30.74 |

\* Counterfactual pricing applies the provider's rates to a hypothetical all-Llama run. Those Llama endpoints
could not accept A-MEM's strict schemas at probe time. The selected Nemo controller avoids that incompatibility
without changing the protocol. The extrapolation is based on one complete sample per dataset, so it is a
planning estimate rather than a full-run invoice.

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

The complete repository suite passes: 148 tests, including 25 focused A-MEM protocol, prompt, metering,
structured-response recovery, embedding transport/batching, lightweight-loader equivalence, invalid-link,
split-role/provider-capability, cache-signature, and shard tests. Paid end-to-end samples succeeded with WandB
and the selected split-role providers pinned with fallbacks disabled. The current shared-GPU oracle canary also
completed successfully with the selected Mistral Nemo/DekaLLM controller and Llama 3.1 8B/DeepInfra reader.
