# MemoryAgentBench — schema & deterministic scoring (build reference)

Research-verified 2026-07-18 (sub-agent, against the live repo + HF dataset). Source of truth for
`src/memory/data/memoryagentbench.py` (reader) + `src/memory/eval/memoryagentbench_score.py` (scorer).
Repo: [HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) · dataset
[`ai-hyz/MemoryAgentBench`](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) (MIT) · arXiv:2507.05257.

## Access
`datasets.load_dataset("ai-hyz/MemoryAgentBench")` — no gating, Parquet, ~132 MB. 4 splits (config `default`):
`Accurate_Retrieval`, `Test_Time_Learning`, `Long_Range_Understanding`, `Conflict_Resolution`. Each **row =
one long context + ALL its questions** ("inject once, query many"). Sub-dataset selected by
`metadata.source` → `filter(lambda r: r["metadata"]["source"] == <name>)`.

## Row schema
```
context: string                       # the long history to memorize
questions: list[string]
answers: list[list[string]]           # per-question list of ACCEPTABLE paraphrase golds
metadata: { source: string,           # the sub-dataset key to filter on
            question_types: [...],     # longmemeval only
            question_ids/question_dates/haystack_sessions: longmemeval only
            keypoints: infbench_sum/detectiveQA only
            previous_events: eventqa only }
```

## Per-category deterministic scoring (drop the 2 judge subsets)
| Competency | source (filter) | reported metric | notes |
|---|---|---|---|
| Accurate Retrieval | `ruler_qa1_*`, `ruler_qa2_*` | substring_exact_match | context up to ~420k tok |
| Accurate Retrieval | `eventqa_*` | substring_exact_match | prompt embeds `previous_events` + candidate next-events |
| Accurate Retrieval | `longmemeval_s*` | **LLM-judge → SKIP** | filter out by source |
| Test-Time Learning | `icl_banking77_*`, `icl_clinic*`, `icl_nlu*`, `icl_trec_coarse*`, `icl_trec_fine*` | exact_match | gold = bare numeric label e.g. `["28"]` — brittle, want flexible parse |
| Long-Range Understanding | `detective_qa` | exact_match | gold **includes letter**, e.g. `["C. The Brandt couple"]`; model asked for JSON |
| Long-Range Understanding | `infbench_sum_*` | **LLM-judge (F1) → SKIP** | filter out by source |
| Conflict Resolution | `factconsolidation_mh_*`, `factconsolidation_sh_*` | substring_exact_match | mh=multi-hop, sh=single-hop |
| Recsys (under TTL) | `recsys_redial_*` | Recall@5 | needs `entity2id.json` + fuzzy edit-distance match — hard to repro exactly |

## Exact scoring code (reimplement in `memoryagentbench_score.py`)
```python
def normalize_answer(t):                       # DrQA/SQuAD style
    t=t.lower(); t=''.join(c for c in t if c not in string.punctuation)
    t=re.sub(r'\b(a|an|the)\b',' ',t); return ' '.join(t.split())
substring_exact_match(pred,gold) = normalize(gold) in normalize(pred)      # gold ∈ pred (lenient direction!)
exact_match(pred,gold)          = normalize(pred) == normalize(gold)
# score = max over the paraphrase gold list; and max over {raw output, parse_output(output)}
# parse_output: strip case-insensitive "Answer:" prefix, take first line.
```

## Querying (single-shot is valid)
The paper's `Long_context_agent` just concatenates `context + "\n" + query` into **one prompt** — so a naive
"whole context + question" API call reproduces that baseline for **every** category. RAG / agentic memory
agents are the other protocols (only needed if reproducing those specific baselines).

## Gotchas
- **Contexts far exceed 128k** for `ruler_qa2` (~420k), `recsys` (~1.48M, only 1 test sample), `eventqa_full`
  (multi-M). For API models: cap/truncate to the model's window and LOG it, or restrict to the ≤128k sub-sources.
- `exact_match` is deliberately brittle (README warns): ICL numeric labels + detectiveQA JSON need a flexible
  extractor for fidelity-to-intent; strict repro will under-score. **This bit us (fixed 2026-07-21, `aab14e9`):**
  our prompts mandate single-line JSON for detective_qa and `label: {label}` for ICL, neither of which the old
  `parse_output` could reach — **both competencies read 0.000 for every model**. `parse_output` now extracts
  the JSON `answer` field and strips a leading `label:` before the first-line fallback. **When adding a MAB
  prompt, check its mandated output shape is one the parser can score.**
- `detectiveQA` gold has the letter prefix — must match `"C. ..."` exactly.
- `recsys` Recall@5 not reproducible without `entity2id.json` (repo root, not in a split; hardcoded path
  `./processed_data/Recsys_Redial/entity2id.json`) + `find_nearest_movie` fuzzy match. Vendor it or mark
  non-canonical.
- Skip `longmemeval` + `infbench_sum` (LLM-judged) by `metadata.source` before scoring.

## Recommended first slice (single-shot, deterministic, ≤128k where possible)
Accurate-Retrieval (ruler_qa1, eventqa small) + Conflict-Resolution (fact_sh/_mh 6k/32k) + Test-Time-Learning
(the 5 ICL sets) + Long-Range (detectiveQA). Recsys later (needs entity2id + fuzzy match). This covers the
forced-forgetting (CR) + test-time-learning (TTL) thesis axes judge-free.
