# REFERENCES — papers & sources for every baseline, technique, and dataset

Single source of truth so we never have to re-search a paper or dataset. Each row gives the
canonical reference (paper title + arXiv/URL) and, where relevant, the exact HuggingFace
identifier used in code and our in-repo location. arXiv IDs marked **(verify)** are best-effort —
the HF identifier / title is the authoritative anchor.

---

## Backbone (frozen decoder)

| Name | Reference | Link |
|---|---|---|
| SmolLM2-135M | "SmolLM2: When Smol Goes Big" (Allal et al., 2025), arXiv:2502.02737 | https://huggingface.co/HuggingFaceTB/SmolLM2-135M |
| Llama-3.2-1B (FineWeb source-tokenizer only) | Llama 3 (Grattafiori et al., 2024), arXiv:2407.21783 | https://huggingface.co/meta-llama/Llama-3.2-1B |

---

## Baseline compressors / memory techniques

Our reimplementations live in `src/memory/models/<name>/encoder.py`; each encoder docstring also
cites its paper. "Native read" = the memory-read mechanism from the paper; the baseline fairness /
asterisk policy (match M-slots + params, each arm uses its own paper's read) is in `docs/README.md`.

| Baseline | Paper | arXiv | Native read | Our code | Official repo |
|---|---|---|---|---|---|
| **ICAE** | In-Context Autoencoder for Context Compression in a LLM (Ge et al., ICLR 2024) | [2307.06945](https://arxiv.org/abs/2307.06945) | prepend soft tokens | `models/icae/` | getao/icae |
| **AutoCompressor** | Adapting Language Models to Compress Contexts (Chevalier et al., EMNLP 2023 Findings) | [2305.14788](https://arxiv.org/abs/2305.14788) | prepend (summary accumulation) | `models/autocompressor/` | princeton-nlp/AutoCompressors |
| **CCM** | Compressed Context Memory for Online LM Interaction (Kim et al., ICLR 2024) | [2312.03414](https://arxiv.org/abs/2312.03414) | per-layer KV (cond. LoRA) | `models/ccm/` | snu-mllab/Context-Memory |
| **Activation Beacon** | Soaring from 4K to 400K: Extending LLM Context with Activation Beacon (Zhang et al., 2024) | [2401.03462](https://arxiv.org/abs/2401.03462) | per-layer KV | `models/beacon/` | FlagOpen/FlagEmbedding |
| **Gisting** | Learning to Compress Prompts with Gist Tokens (Mu, Li & Goodman, NeurIPS 2023) | [2304.08467](https://arxiv.org/abs/2304.08467) | per-layer gist-KV | `models/gisting/` | jayelm/gisting |
| **MemoryLLM** | MemoryLLM: Towards Self-Updatable LLMs (Wang et al., ICML 2024) | [2402.04624](https://arxiv.org/abs/2402.04624) | per-layer KV pool (random-drop) | `models/memoryllm/` | wangyu-ustc/MemoryLLM |
| **Titans** | Titans: Learning to Memorize at Test Time (Behrouz et al., 2024) | [2501.00663](https://arxiv.org/abs/2501.00663) | prepend (MAC); deep-MLP autograd write | `models/titans/` | — |
| **H2O** | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs (Zhang et al., NeurIPS 2023) | [2306.14048](https://arxiv.org/abs/2306.14048) | raw-KV eviction (training-free) | `models/h2o/` (eval ref) | FMInference/H2O |
| **VQ-ICAE** | in-house = ICAE + vector quantization. VQ = Neural Discrete Representation Learning (van den Oord et al., NeurIPS 2017) | [1711.00937](https://arxiv.org/abs/1711.00937) | prepend soft tokens | `models/vqicae/` | — |

### Our own arms + their lineage
| Arm / concept | Reference | arXiv | Our code |
|---|---|---|---|
| slotgraph / slotgraph2 / slotgraph3 | in-house (STDP/delta-rule graph over a learned vocab) | — | `models/slotgraph{,2,3}/` |
| biomem | in-house (fast-Hebbian cortical-column) | — | `models/biomem/` |
| delta-rule write lineage | DeltaNet (Yang et al., 2024) / Gated DeltaNet (Yang et al., 2024) | [2406.06484](https://arxiv.org/abs/2406.06484) / [2412.06464](https://arxiv.org/abs/2412.06464) | — |
| Slot-Attention (anti-collapse read) | Object-Centric Learning with Slot Attention (Locatello et al., NeurIPS 2020) | [2006.15055](https://arxiv.org/abs/2006.15055) | — |
| EntNet (keyed slot memory) | Tracking the World State with Recurrent Entity Networks (Henaff et al., ICLR 2017) | [1612.03969](https://arxiv.org/abs/1612.03969) | — |
| RMT (recurrent memory) | Recurrent Memory Transformer (Bulatov et al., NeurIPS 2022) | [2207.06881](https://arxiv.org/abs/2207.06881) | — |
| vanilla_llama / vanilla_full_context | loss floor / ceiling (frozen SmolLM2-135M) | — | `models/vanilla/` |

---

## Training objective — behavioral-KL context distillation

`L = CE + α·KL(p_frozenLM(y|full ctx) ‖ p_frozenLM(y|memory))`, forward-KL, value-span masked
(`src/memory/training/objectives.py`). Lineage:

| Reference | arXiv |
|---|---|
| Deep Context Distillation (DCD) — Caccia et al., 2025 (output-KL + hidden-L1) | [2503.08727](https://arxiv.org/abs/2503.08727) |
| Wingate et al., 2022 — canonical KL(hard-prompt ‖ soft-prompt) | [2210.03162](https://arxiv.org/abs/2210.03162) |
| xRAG (Cheng et al., NeurIPS 2024) — CE + α·KL, frozen LLM | [2405.13792](https://arxiv.org/abs/2405.13792) |
| Cartridges / Self-Study (2025) — per-layer KV + context distillation | [2506.06266](https://arxiv.org/abs/2506.06266) |
| Kujanpää et al., 2024 — E[KL]=I(context;answer) theorem (forward-KL) | [2412.14964](https://arxiv.org/abs/2412.14964) |
| Padmanabhan et al., 2023 — score KL only downstream of injected content (value-span mask) | [2306.09306](https://arxiv.org/abs/2306.09306) |
| CCM — measures the posterior-collapse we avoid | [2312.03414](https://arxiv.org/abs/2312.03414) |

---

## Data sources

Loaders in `src/memory/data/sources/` (train) and `src/memory/data/*.py` (eval); build/ingest in
`scripts/data_build/`; on-disk in `data/<name>/`. See `DATASETS.md` for the wiring index.

### Phase-0 (architecture scrutiny — the ACTIVE mix)
| Source | Dataset / HF id | Paper | arXiv |
|---|---|---|---|
| babi (bAbI tasks) | `Muennighoff/babi` | Towards AI-Complete QA: bAbI tasks (Weston et al., 2015) | [1502.05698](https://arxiv.org/abs/1502.05698) |
| bio | in-house synthetic entity generator (`src/memory/data/bio_render.py`) | — | — |
| squad | `rajpurkar/squad_v2` | Know What You Don't Know: SQuAD 2.0 (Rajpurkar et al., 2018) | [1806.03822](https://arxiv.org/abs/1806.03822) |
| triviaqa | `mandarjoshi/trivia_qa` | TriviaQA (Joshi et al., ACL 2017) | [1705.03551](https://arxiv.org/abs/1705.03551) |
| hotpot_train | `hotpot_qa` | HotpotQA (Yang et al., EMNLP 2018) | [1809.09600](https://arxiv.org/abs/1809.09600) |
| musique_train | `dgslibisey/MuSiQue` | MuSiQue (Trivedi et al., TACL 2022) | [2108.00573](https://arxiv.org/abs/2108.00573) |
| multiwoz | MultiWOZ 2.2 | MultiWOZ (Budzianowski et al., EMNLP 2018) | [1810.00278](https://arxiv.org/abs/1810.00278) |
| quality | QuALITY | QuALITY (Pang et al., NAACL 2022) | [2112.08608](https://arxiv.org/abs/2112.08608) |
| fineweb | `HuggingFaceFW/fineweb-edu` | FineWeb / FineWeb-Edu (Penedo et al., NeurIPS 2024) | [2406.17557](https://arxiv.org/abs/2406.17557) |
| pile | The Pile | The Pile (Gao et al., 2020) | [2101.00027](https://arxiv.org/abs/2101.00027) |
| redpajama | RedPajama-Data | Together Computer, 2023 | github.com/togethercomputer/RedPajama-Data |
| mqar | generator (Zoology) | Zoology: associative recall / MQAR (Arora et al., 2023) | [2312.04927](https://arxiv.org/abs/2312.04927) |
| ruler_overwrite | fork of RULER | RULER (Hsieh et al., 2024) | [2404.06654](https://arxiv.org/abs/2404.06654) |

### Phase-1 (full-corpus training — registered, wired when Phase 1 starts)
| Source | Dataset / HF id | Paper | arXiv |
|---|---|---|---|
| wildchat | `allenai/WildChat-1M` | WildChat (Zhao et al., ICLR 2024) | [2405.01470](https://arxiv.org/abs/2405.01470) |
| lmsys_chat | `lmsys/lmsys-chat-1m` (gated) | LMSYS-Chat-1M (Zheng et al., 2023) | [2309.11998](https://arxiv.org/abs/2309.11998) |
| msc | `nayohan/multi_session_chat` | Multi-Session Chat (Xu et al., ACL 2022) | [2107.07567](https://arxiv.org/abs/2107.07567) |
| qasper | `allenai/qasper` | Qasper (Dasigi et al., NAACL 2021) | [2105.03011](https://arxiv.org/abs/2105.03011) |
| longcite | `zai-org/LongCite-45k` | LongCite (Zhang et al., 2024) | [2409.02897](https://arxiv.org/abs/2409.02897) |
| govreport | `ccdv/govreport-summarization` | GovReport (Huang et al., NAACL 2021) | [2104.02112](https://arxiv.org/abs/2104.02112) |
| pg19 | `emozilla/pg19` (PG-19) | Compressive Transformer / PG-19 (Rae et al., ICLR 2020) | [1911.05507](https://arxiv.org/abs/1911.05507) |
| ruler_niah | generator | RULER (Hsieh et al., 2024) | [2404.06654](https://arxiv.org/abs/2404.06654) |
| babilong_train | `RMT-team/babilong-train-5k-samples` | BABILong (Kuratov et al., NeurIPS 2024) | [2406.10149](https://arxiv.org/abs/2406.10149) |
| wikibigedit | `lukasthede/WikiBigEdit` | WikiBigEdit — lifelong knowledge editing (Thede et al., 2025) | (verify) |
| swe_trajectories | `nebius/SWE-agent-trajectories` | dataset by Nebius; SWE-agent (Yang et al., NeurIPS 2024) | [2405.15793](https://arxiv.org/abs/2405.15793) |
| perltqa | PerLTQA | PerLTQA (Du et al., 2024) | [2402.16288](https://arxiv.org/abs/2402.16288) |

### Phase-2 (test-eval benchmarks)
| Reader | Benchmark | Paper | arXiv |
|---|---|---|---|
| longmemeval | LongMemEval | Wu et al., ICLR 2025 | [2410.10813](https://arxiv.org/abs/2410.10813) |
| longbench | LongBench (v1 + v2) | Bai et al., ACL 2024 | [2308.14508](https://arxiv.org/abs/2308.14508) |
| infinitebench | ∞Bench / InfiniteBench | Zhang et al., ACL 2024 | [2402.13718](https://arxiv.org/abs/2402.13718) |
| niah | Needle-in-a-Haystack | Kamradt, 2023 | github.com/gkamradt/LLMTest_NeedleInAHaystack |
| babilong | BABILong | Kuratov et al., NeurIPS 2024 | [2406.10149](https://arxiv.org/abs/2406.10149) |
| ruler | RULER | Hsieh et al., 2024 | [2404.06654](https://arxiv.org/abs/2404.06654) |
| locomo | LoCoMo | Maharana et al., ACL 2024 | [2402.17753](https://arxiv.org/abs/2402.17753) |
| narrativeqa | NarrativeQA | Kočiský et al., TACL 2018 | [1712.07040](https://arxiv.org/abs/1712.07040) |
| hotpot / musique | (see Phase-0) | — | — |

### Competitor memory systems (Phase-2 head-to-heads — not yet in repo)
| System | Paper | arXiv |
|---|---|---|
| Larimar (Kanerva episodic memory) | Das et al., ICML 2024 | [2403.11901](https://arxiv.org/abs/2403.11901) |
| M+ (MemoryLLM + retriever) | Wang et al., ICML 2025 | [2502.00592](https://arxiv.org/abs/2502.00592) |
| WISE (lifelong editing) | Wang et al., NeurIPS 2024 | [2405.14768](https://arxiv.org/abs/2405.14768) |
| ATLAS (test-time memory) | Behrouz et al., 2025 | [2505.23735](https://arxiv.org/abs/2505.23735) (verify) |
| Mem0 / MemGPT / Zep | RAG / memory-OS competitors (external, mostly training-free) | — |

---

*Maintenance:* when you add a baseline or data source, add a row here **and** cite the paper in the
encoder/source docstring. See `docs/DATA_PHASES_PLAN.md` for the phase plan and `docs/README.md`
for the baseline cohort + fairness policy.
