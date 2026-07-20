# Phase-2 Baseline Report — rendered table (authoritative, deterministic scoring)

Generated 2026-07-20 by `python scripts/baselines/report.py` from `outputs/baselines/*.json`.
Deterministic scoring (EM + negation-guarded containment + BEM@0.5 paraphrase for LongMemEval; substring/exact
per-competency for MemoryAgentBench). **No LLM-as-judge** for the panel. Cell = `accuracy (n_scored)`.
Blank cell = not run (e.g. llama full_context on MAB: only 154 of the shorter contexts fit its 131k window).
CSV companion: `PHASE2_REPORT.csv`. Narrative + config + calibration notes: `PHASE2_RESULTS.md`.

| dataset-subtask | deepseek-v4-flash · floor | deepseek-v4-flash · full_context | deepseek-v4-flash · rag_bm25_k15 | deepseek-v4-flash · rag_bm25_k5 | llama-3.1-8b-instruct · floor | llama-3.1-8b-instruct · full_context | llama-3.1-8b-instruct · rag_bm25_k15 | llama-3.1-8b-instruct · rag_bm25_k5 |
|---|---|---|---|---|---|---|---|---|
| **longmemeval-OVERALL** | **0.006 (470)** | **0.687 (470)** |  | **0.594 (470)** | **0.002 (470)** | **0.462 (444)** |  | **0.404 (470)** |
| longmemeval-COVERAGE | 1.000 (500) | 1.000 (500) |  | 1.000 (500) | 1.000 (500) | 0.944 (500) |  | 1.000 (500) |
| longmemeval-abstention | 1.000 (30) | 0.967 (30) |  | 1.000 (30) | 0.967 (30) | 0.500 (28) |  | 0.800 (30) |
| longmemeval-knowledge-update | 0.000 (72) | 0.819 (72) |  | 0.792 (72) | 0.000 (72) | 0.783 (69) |  | 0.708 (72) |
| longmemeval-multi-session | 0.000 (121) | 0.612 (121) |  | 0.421 (121) | 0.000 (121) | 0.235 (115) |  | 0.182 (121) |
| longmemeval-single-session-assistant | 0.036 (56) | 0.911 (56) |  | 0.875 (56) | 0.018 (56) | 0.731 (52) |  | 0.696 (56) |
| longmemeval-single-session-preference | 0.000 (30) | 0.067 (30) |  | 0.000 (30) | 0.000 (30) | 0.111 (27) |  | 0.000 (30) |
| longmemeval-single-session-user | 0.000 (64) | 0.875 (64) |  | 0.797 (64) | 0.000 (64) | 0.541 (61) |  | 0.672 (64) |
| longmemeval-temporal-reasoning | 0.008 (127) | 0.638 (127) |  | 0.559 (127) | 0.000 (127) | 0.417 (120) |  | 0.276 (127) |
| **memoryagentbench-OVERALL** | **0.337 (3071)** | **0.721 (2986)** | **0.646 (2944)** | **0.613 (2881)** | **0.206 (3071)** | **0.636 (154)** | **0.417 (3030)** | **0.394 (3063)** |
| memoryagentbench-COMPETENCY_MACRO | 0.156 (3071) | 0.425 (2986) | 0.379 (2944) | 0.351 (2881) | 0.099 (3071) | 0.636 (154) | 0.223 (3030) | 0.211 (3063) |
| memoryagentbench-COVERAGE | 1.000 (3071) | 0.972 (3071) | 0.959 (3071) | 0.938 (3071) | 1.000 (3071) | 1.000 (154) | 0.987 (3071) | 0.997 (3071) |
| memoryagentbench-OVERALL_lenient | 0.382 (3071) | 0.882 (2986) | 0.815 (2944) | 0.760 (2881) | 0.249 (3071) | 0.636 (154) | 0.533 (3030) | 0.492 (3063) |
| memoryagentbench-comp:Accurate_Retrieval | 0.594 (1700) | 0.936 (1682) | 0.857 (1697) | 0.819 (1700) | 0.349 (1700) | 0.636 (154) | 0.624 (1700) | 0.592 (1700) |
| memoryagentbench-comp:Conflict_Resolution | 0.030 (800) | 0.765 (756) | 0.658 (682) | 0.585 (639) | 0.049 (800) |  | 0.269 (759) | 0.254 (792) |
| memoryagentbench-comp:Long_Range_Understanding | 0.000 (71) | 0.000 (67) | 0.000 (67) | 0.000 (64) | 0.000 (71) |  | 0.000 (71) | 0.000 (71) |
| memoryagentbench-comp:Test_Time_Learning | 0.000 (500) | 0.000 (481) | 0.000 (498) | 0.000 (478) | 0.000 (500) |  | 0.000 (500) | 0.000 (500) |
| memoryagentbench-detective_qa | 0.000 (71) | 0.000 (67) | 0.000 (67) | 0.000 (64) | 0.000 (71) |  | 0.000 (71) | 0.000 (71) |
| memoryagentbench-eventqa_131072 | 0.650 (500) | 0.961 (491) | 0.894 (500) | 0.850 (500) | 0.370 (500) |  | 0.652 (500) | 0.618 (500) |
| memoryagentbench-eventqa_65536 | 0.726 (500) | 0.972 (498) | 0.946 (500) | 0.936 (500) | 0.458 (500) | 0.636 (154) | 0.714 (500) | 0.710 (500) |
| memoryagentbench-eventqa_full | 0.496 (500) | 0.933 (494) | 0.829 (497) | 0.770 (500) | 0.270 (500) |  | 0.526 (500) | 0.476 (500) |
| memoryagentbench-factconsolidation_mh_262k | 0.000 (100) | 0.235 (85) | 0.062 (64) | 0.091 (66) | 0.010 (100) |  | 0.011 (87) | 0.031 (97) |
| memoryagentbench-factconsolidation_mh_32k | 0.010 (100) | 0.644 (90) | 0.212 (66) | 0.160 (50) | 0.000 (100) |  | 0.054 (92) | 0.061 (99) |
| memoryagentbench-factconsolidation_mh_64k | 0.010 (100) | 0.582 (91) | 0.156 (64) | 0.107 (56) | 0.010 (100) |  | 0.063 (95) | 0.050 (100) |
| memoryagentbench-factconsolidation_mh_6k | 0.010 (100) | 0.796 (93) | 0.767 (90) | 0.319 (69) | 0.000 (100) |  | 0.044 (91) | 0.062 (96) |
| memoryagentbench-factconsolidation_sh_262k | 0.050 (100) | 0.835 (97) | 0.778 (99) | 0.707 (99) | 0.050 (100) |  | 0.410 (100) | 0.370 (100) |
| memoryagentbench-factconsolidation_sh_32k | 0.040 (100) | 0.970 (100) | 0.919 (99) | 0.880 (100) | 0.110 (100) |  | 0.495 (97) | 0.460 (100) |
| memoryagentbench-factconsolidation_sh_64k | 0.050 (100) | 0.950 (100) | 0.870 (100) | 0.848 (99) | 0.140 (100) |  | 0.561 (98) | 0.530 (100) |
| memoryagentbench-factconsolidation_sh_6k | 0.070 (100) | 1.000 (100) | 0.970 (100) | 0.900 (100) | 0.070 (100) |  | 0.444 (99) | 0.450 (100) |
| memoryagentbench-icl_banking77_5900shot_balance | 0.000 (100) | 0.000 (100) | 0.000 (100) | 0.000 (100) | 0.000 (100) |  | 0.000 (100) | 0.000 (100) |
| memoryagentbench-icl_clinic150_7050shot_balance | 0.000 (100) | 0.000 (99) | 0.000 (100) | 0.000 (99) | 0.000 (100) |  | 0.000 (100) | 0.000 (100) |
| memoryagentbench-icl_nlu_8296shot_balance | 0.000 (100) | 0.000 (96) | 0.000 (100) | 0.000 (96) | 0.000 (100) |  | 0.000 (100) | 0.000 (100) |
| memoryagentbench-icl_trec_coarse_6600shot_balance | 0.000 (100) | 0.000 (96) | 0.000 (99) | 0.000 (96) | 0.000 (100) |  | 0.000 (100) | 0.000 (100) |
| memoryagentbench-icl_trec_fine_6400shot_balance | 0.000 (100) | 0.000 (90) | 0.000 (99) | 0.000 (87) | 0.000 (100) |  | 0.000 (100) | 0.000 (100) |
| memoryagentbench-ruler_qa1_197K | 0.290 (100) | 0.860 (100) | 0.660 (100) | 0.600 (100) | 0.210 (100) |  | 0.640 (100) | 0.590 (100) |
| memoryagentbench-ruler_qa2_421K | 0.450 (100) | 0.717 (99) | 0.560 (100) | 0.550 (100) | 0.230 (100) |  | 0.510 (100) | 0.450 (100) |

---

## Published reference (CITED — paper LLM-judge, NOT comparable to the deterministic columns above)

> These are the papers' own numbers, graded by a **GPT-4o judge** (which credits paraphrase/superset answers). Our deterministic scores sit **below** these on the same outputs — read them as recognized anchors, not head-to-head. Vendor self-reported numbers are excluded.

### LongMemEval-S — published
_Wu et al., LongMemEval-S, arXiv:2410.10813 (ICLR'25) — scoring: gpt-4o-2024-08-06 judge_

| model | condition | accuracy |
|---|---|---|
| GPT-4o | oracle (evidence session only) | 0.870 |
| GPT-4o | full-context (~115k) | 0.606 |
| GPT-4o | full-context +Chain-of-Note | 0.640 |
| Llama-3.1-8B-Instruct | oracle | 0.710 |
| Llama-3.1-8B-Instruct | full-context | 0.454 |
| Llama-3.1-70B-Instruct | oracle | 0.744 |
| Llama-3.1-70B-Instruct | full-context | 0.334 |
| Phi-3-Medium-128k | full-context | 0.380 |

_Llama-3.1-8B full-context (0.454) is the base number; verify base-vs-CoN split against Fig 3(b) before publishing. GPT-4o-mini / Mistral / Qwen / Claude are NOT in the primary -S table._

### MemoryAgentBench — published (long-context / RAG backbones)
_Hu et al., MemoryAgentBench, arXiv:2507.05257 — AR/TTL/CR deterministic; LRU = GPT-4o-judge F1_

| method | Accurate-Retrieval | Test-Time-Learning | Long-Range*(judge) | Conflict-Res(single-hop) |
|---|---|---|---|---|
| Long-context GPT-4o | 53.5–61.5 | 87.6 | 32.2 | 60.0 |
| Long-context GPT-4o-mini | 44.9–53.5 | 82.4 | 28.9 | 45.0 |
| Long-context Claude-3.7-Sonnet | 50.6–74.6 | 89.4 | 52.5 | 43.0 |
| RAG BM25 | 45.6–74.6 | 75.4 | 20.9 | 56.0 |
| RAG Embedding (NV-Embed-v2) | 51.4–83.0 | 69.4 | 20.7 | 55.0 |
| Mem0 | 22.4–37.5 | 3.4 | 0.8 | 18.0 |

_* LRU is LLM-judged (GPT-4o F1) upstream — our detective_qa (LRU) uses deterministic exact_match, so that column is the LEAST comparable. Conflict-Resolution MULTI-hop collapses to ≤6% for all methods upstream (only single-hop shown). AR is a per-subtask range._

