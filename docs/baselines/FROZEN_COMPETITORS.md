# Frozen competitors — the off-the-shelf SOTA to compare against (Phase 2)

*Landscape survey, 2026-07-12. Purpose: decide which memory systems we line up against once we leave the small-scale Phase-0 matched-harness ablation and move to a real-world comparison. Companion to `docs/REFERENCES.md` (baseline provenance) and `docs/data/DATA_PHASES_PLAN.md` (the phase plan).*

## Framing: two comparisons, two baseline strategies
- **Phase 0 (now) — architecture ablation.** "Which memory-write *mechanism* wins at a fixed budget?" There is no released checkpoint of any competitor at *our* budget (frozen 135 M backbone, ~7 M trainable, M=96), so every arm is **re-implemented in a matched harness** (see `docs/REFERENCES.md`). This is the field norm for this question (Cartridges, CCM, RMT all do it), not a shortcut.
- **Phase 2 (later) — real-world comparison.** "Does our memory cache beat the alternatives on real tasks?" Here we compare against **frozen, off-the-shelf systems** — long-context LMs, RAG, agent-memory frameworks — plus the handful of memory mechanisms that actually ship weights. This doc is the menu for that phase.

## Headline finding: the SOTA memory *mechanism* research is almost all **unreleased**
This is the single most decision-relevant fact. Across the test-time / parametric / recurrent memory lineage, downloadable frozen checkpoints barely exist — which (a) confirms the Phase-0 re-implementation plan and (b) forces the Phase-2 headline to be against *paradigms*, not individual mechanisms.

| Mechanism | Released weights? | Note |
|---|---|---|
| Titans (Google) + follow-ups ATLAS, Miras/Moneta/Yaad/Memora | ❌ | "code soon" promised Jan 2025, still nothing; only unofficial `lucidrains/titans-pytorch` (arch, no weights) |
| Infini-attention (Google) | ❌ | HuggingFace **publicly failed to reproduce it** (gating collapses) |
| Memorizing Transformers / RMT / ARMT | ❌ | code only |
| Larimar, SEAL | ❌ | code only (train-your-own) |
| Gated DeltaNet (NVIDIA) | ❌ research weights | **but ships in `Qwen/Qwen3-Next-80B-A3B` (Apache-2.0, downloadable)** — 75% of its layers |
| **MemoryLLM-8B / M+** | ✅ | `YuWangX/memoryllm-8b`, `YuWangX/mplus-8b` — the ONLY parametric-memory mechanism with weights; closest research analog to us |
| **EM-LLM** | ✅ (n/a) | training-free — wraps any frozen LM's KV; trivially "frozen" |

## Tier 1 — the paradigms (must-have frozen baselines)
**Long-context LMs** ("just enlarge the window"):

| Model | Context | Access | Why |
|---|---|---|---|
| **GPT-4o @128K** | 128K | API | The model **LongMemEval itself benchmarked** → citable, reproduced numbers (not a vendor claim). Primary long-context baseline. |
| **Qwen2.5-1M / Qwen3(-Next)** | 1M / 256K | Open | Best **open** long-context with real 3rd-party RULER numbers (RULER 93.1 @1M) |
| **Gemini 2.5 Pro** | 1M–2M | API | Best-validated 1M utilization (LOFT/MRCR/NIAH) = API ceiling |

Avoid Llama-4's "10M" and MiniMax's claims as headline baselines — **marketing-length, no independent RULER/NoLiMa validation**. Effective context ≪ claimed context is the norm (RULER / HELMET / Chroma "context rot") — that gap *is* our thesis's argument, so GPT-4.1's advertised 1M (which underperforms GPT-4o-mini on LongMemEval) is a good adversarial *example*, not a baseline.

**RAG**: a standard retrieval pipeline (BM25 / Contriever / GTE-Qwen2 — LongMemEval's own configs).

**Anchors we already own**: `vanilla_full_context` (oracle) + `vanilla_llama` (no-memory floor).

## Tier 2 — memory systems with real released artifacts
| System | Mechanism | Artifact |
|---|---|---|
| **MemoryLLM-8B / M+** | self-updatable parametric latent memory pool | `YuWangX/memoryllm-8b`, `YuWangX/mplus-8b` (MIT, Llama-3.1-8B) — **highest-value: only mechanism-with-weights, closest analog to us** |
| **EM-LLM** | training-free episodic segmentation over frozen KV | `em-llm/EM-LLM-model` — easiest to run (wraps any frozen LM) |
| **Mem0 / Mem0^g** | LLM-extracted fact store (+ graph) | OSS; best-documented peer-reviewed agent-memory numbers (LoCoMo 66.9 / 68.4 J-score) |
| **Zep / Graphiti** | temporal knowledge graph | Graphiti core Apache-2.0; **strongest architecture story but its numbers are the most publicly disputed — re-verify, don't cite** |
| MemGPT / Letta | OS-style paging | the "paging" architectural family reference |

## Tier 3 — context compression (the family closest to our design)
**Downloadable frozen soft-token compressors:**

| Method | Ratio | Backbone | Checkpoint |
|---|---|---|---|
| **ARC-Encoder** (Kyutai 2025) | 4×/8× | frozen Mistral-7B / Llama-3.1-8B | `kyutai/arc-encoders` — **architecturally our NEAREST sibling** (bidir encoder → soft tokens → frozen decoder, multi-decoder portable) |
| **ICAE** (ICLR'24) | 4× | Llama-2-7B / Mistral | `sggetao/icae` — closest *classic* analog (LoRA enc → mem → frozen dec) |
| **AutoCompressors** (EMNLP'23) | ~40:1 | Llama-2-7B | `princeton-nlp/AutoCompressor-Llama-2-7b-6k` (7 ckpts) |
| **Gisting** (NeurIPS'23) | up to 26× | LLaMA-7B | `jayelm/llama-7b-gist-1` (CC-BY-NC, weight-diff) |
| **Activation Beacon / UltraGist** | 8× | Qwen-2 / Mistral | `namespace-Pt/beacon-qwen-2-7b-instruct` (orig llama-2 ckpt 404s) |
| **xRAG** (NeurIPS'24) | ~178× (1 token) | Mistral-7B | `Hannibal046/xrag-7b` |
| **PISCO** (ACL'25) | 16× | Mistral / Llama-3.1-8B | `naver/pisco-mistral` (CC-BY-NC) |
| **C3** (2025) | 20–40× | Qwen-3B | `liufanfanlff/C3-Context-Cascade-Compression` |
| **LLoCO** (EMNLP'24) | 30× | Llama-2-7B | `xiuyul/Lloco-7b-qasper` (task LoRAs) |
| **LLMLingua / -2** (text-pruning class) | 2–20× | XLM-R / phi-2 | `microsoft/llmlingua-2-...` (`pip install llmlingua`, MIT) — different mechanism (prunes NL text), but the easiest frozen compression baseline |

**Recipe-only (must retrain — not downloadable):** **Cartridges** (Stanford — per-corpus KV cache, no universal ckpt *by design*; our nearest thesis cousin: train a compact memory read by a frozen LM; 38.6× KV, LongHealth 48.4%), 500xCompressor (HF collection empty), UniICL (no code), CCM (adapters on GDrive only).

**KV-eviction band (training-free — apply to any LM, a "dumb compression" floor):** StreamingLLM, **H2O** (we already have it eval-only), SnapKV, PyramidKV.

## Benchmarks + anchor numbers
| Benchmark | Role | Anchors |
|---|---|---|
| **LongMemEval** (ICLR'25) | **Headline** | full-context **oracle** (GPT-4o) ≈ **87–91.8%**; **naive full-context (all sessions) ≈ 60.6%**. The ~30-pt gap = the "needle-in-noise tax" a good memory cache should close. Zep 63.8/71.2%, LightMem ~68–70%. |
| **BEAM** (ICLR'26) | Scale | 128K→**10M** tokens, 10 memory abilities; native long-ctx/RAG/memory-cache columns |
| **WikiBigEdit** (ICML'25) | Streaming / stability | 500K+ real sequential edits; RAG a first-class arm; the only good streaming-write fit |
| **LOFT** (DeepMind) / **HELMET** (Princeton) | RAG-vs-long-context | both have explicit RAG arms; LOFT built to test "does long-context subsume retrieval" |

## Red flags (bake into the eval protocol)
- **Every post-2024 LongMemEval/LoCoMo "SOTA" is a self-reported vendor number on a non-standard harness, and they are mutually contradictory.** Mem0's widely-cited "94.4% on LongMemEval" is **inflated → 40–54% on two independent harnesses**; Zep retracted 84%→58.4% on LoCoMo (scoring bug). **Cite only original-paper baselines; re-run anything else ourselves.**
- LongMemEval is **gameable**: a bare grep-over-files agent scores ~74% → needs our anti-Goodhart framing (see `docs/design/OBJECTIVES.md` / the curriculum).

## White space (our angle)
- **No soft-token / trainable-encoder compressor reports LongMemEval** (Cartridges uses LongHealth/MTOB; KV papers use LongBench/RULER). Our ~96-token compressor on LongMemEval would be a novel data point with no compression-based peer.
- **No purpose-built compression-vs-RAG-vs-long-context arena exists.** Cartridges is the nearest prior work and it is a recipe, not a standing benchmark. We could own that framing.

## Recommended Phase-2 lineup
_Original survey menu (2026-07-12):_ headline vs **GPT-4o/Qwen-long-context + RAG + MemoryLLM/M+ + EM-LLM**, on **LongMemEval** (+ BEAM for scale, + WikiBigEdit for stability), with **ARC-Encoder + ICAE + xRAG/PISCO + Cartridges** as compression cousins, bracketed by the **full-context oracle** and **no-memory floor**.

> **UPDATE 2026-07-21 — PANEL SCOPE FIXED at five arms:** **M+** (`mplus-8b`) · **H2O@2%** · **H2O@20%** ·
> **KVzip** · **A-MEM**. **SnapKV, LCLM and `memoryllm-8b` are DROPPED** (cite-only) — do not schedule or
> report rows for them. H2O is run at two KV budgets because the 2% run sits at the no-context refusal floor;
> the pair is the compression-ratio ablation. Live status: [`PHASE2_HUB.md`](PHASE2_HUB.md).
>
> **UPDATE 2026-07-20 — active lineup (superseded by the line above)** (see [`PHASE2_HUB.md`](PHASE2_HUB.md) for live status): **Tier-1** = deepseek-v4-flash + llama-3.1-8b (long-context + RAG + floor) — DONE. **Tier-2 GPU mechanisms** = **KVzip** (query-agnostic KV compression) · **H2O/SnapKV** (query-aware KV eviction) · **MemoryLLM/M+** (recurrent parametric). **A-MEM** is the separate agent-memory comparison. **LCLM was dropped from the runnable panel** and remains cite-only. EM-LLM/ARC-Encoder/xRAG/PISCO/Cartridges/GPT-4o/Qwen were not run (cite-only or deferred).

> **UPDATE 2026-07-21 — the M+ LongMemEval whitespace is now filled by us.** MemoryLLM/M+ reported **no** LongMemEval number; we establish it: **`mplus-8b` = 0.423 overall / 0.379 task-averaged / 0.000 abstention** (n=500, deterministic scorer, artifact `outputs/baselines/longmemeval__memoryllm__mplus-8b__s__MERGED.json`). H2O is done on both benchmarks. Full context: [`PHASE2_REPORT.md`](PHASE2_REPORT.md).
