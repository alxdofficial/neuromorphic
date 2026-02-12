# Baseline Comparison & Evaluation Plan for Neuromorphic LM

**Target model:** Neuromorphic LM, ~56M params (Tier A: D=512, L=8, B=4), recurrent architecture with parallel blocks, Procedural Memory (PM), Episodic Memory (EM), Working Memory (WM), and spatial decoder.

**Date:** 2026-02-11

---

## Table of Contents

1. [Transformer Baselines](#1-transformer-baselines)
2. [Recurrent / SSM Baselines](#2-recurrent--ssm-baselines)
3. [Memory-Augmented Models](#3-memory-augmented-models)
4. [Consolidated Benchmark Table](#4-consolidated-benchmark-table)
5. [Evaluation Benchmarks & Metrics](#5-evaluation-benchmarks--metrics)
6. [Memory-Specific Evaluations](#6-memory-specific-evaluations)
7. [Experimental Design](#7-experimental-design)
8. [Recommended Comparison Strategy](#8-recommended-comparison-strategy)
9. [References](#9-references)

---

## 1. Transformer Baselines

### 1.1 Pythia Series (EleutherAI) -- PRIMARY BASELINE

The gold standard for small-model research. All models trained on the Pile (deduplicated, ~207B tokens), with identical preprocessing and 154 public checkpoints per size. Published at ICML 2023.

| Model | Params | Non-embed | Layers | D_model | Heads | Training tokens |
|-------|--------|-----------|--------|---------|-------|-----------------|
| **Pythia-70M** | 70M | 18.9M | 6 | 512 | 8 | ~300B (1.5 epochs) |
| **Pythia-160M** | 160M | 85M | 12 | 768 | 12 | ~300B (1.5 epochs) |
| **Pythia-410M** | 410M | 302M | 24 | 1024 | 16 | ~300B (1.5 epochs) |

**Why ideal:** Same D=512 as our Tier A (Pythia-70M), reproducible, extensively benchmarked, public weights, public training data. Pythia-70M is the closest parameter-matched transformer baseline.

**Published zero-shot benchmarks (from Pythia paper, Table 4, lm-eval-harness):**

| Benchmark | Pythia-70M | Pythia-160M | Pythia-410M |
|-----------|-----------|-------------|-------------|
| LAMBADA (acc) | ~33% | ~43% | ~56% |
| HellaSwag (acc_norm) | ~27% | ~30% | ~34% |
| PIQA (acc) | ~62% | ~66% | ~69% |
| ARC-Easy (acc) | ~44% | ~49% | ~55% |
| ARC-Challenge (acc_norm) | ~23% | ~24% | ~27% |
| WinoGrande (acc) | ~51% | ~52% | ~53% |
| WikiText (ppl, word-level) | ~32.4 | ~21.8 | ~15.7 |

**Source:** Biderman et al., "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling," ICML 2023. arXiv:2304.01373.

**Note on benchmark numbers:** Published Pythia numbers use the EleutherAI lm-evaluation-harness. HellaSwag numbers at small scale are notably low (~27-34% for 70M-410M) because the task is hard for small models. Numbers vary by harness version; always specify version used.

### 1.2 GPT-2 Small (OpenAI)

| Model | Params | Layers | D_model | Heads | Training tokens |
|-------|--------|--------|---------|-------|-----------------|
| **GPT-2 Small** | 124M | 12 | 768 | 12 | ~40B (WebText) |

The original small transformer LM. Older training recipe, smaller dataset, no modern improvements (RoPE, SwiGLU, etc.). Useful as a "historical" reference but Pythia is a fairer modern comparison.

**Published benchmarks:**
- WikiText-103 perplexity: ~29.4 (word-level)
- LAMBADA accuracy: ~45%

**Source:** Radford et al., "Language Models are Unsupervised Multitask Learners," 2019.

### 1.3 SmolLM (HuggingFace, 2024)

Data-centric small models emphasizing data quality over scale. Trained on curated SmolLM-Corpus (FineWeb-Edu + StarCoder + Cosmopedia).

| Model | Params | Layers | D_model | Heads | Training tokens |
|-------|--------|--------|---------|-------|-----------------|
| **SmolLM-135M** | 135M | 18 | 576 | 9 | 600B |
| **SmolLM-360M** | 360M | 30 | 768 | 12 | 600B |

**Published benchmarks:**

| Benchmark | SmolLM-135M | SmolLM-360M |
|-----------|-------------|-------------|
| HellaSwag (acc_norm) | ~42% | ~54% |
| PIQA (acc) | ~68% | ~73% |
| ARC-Easy (acc) | ~52% | ~59% |
| ARC-Challenge (acc_norm) | ~28% | ~36% |

SmolLM demonstrates that aggressive data curation can compensate for small model size. At 135M params, it outperforms models trained on far more tokens with weaker data.

**Source:** Allal et al., "SmolLM - blazingly fast and remarkably powerful," HuggingFace Blog, 2024. arXiv:2502.02737.

### 1.4 TinyLlama (2024)

Llama-2 architecture at 1.1B params, trained on 3T tokens (150x Chinchilla-optimal). Demonstrates extreme over-training benefits.

| Model | Params | Layers | D_model | Heads | Training tokens |
|-------|--------|--------|---------|-------|-----------------|
| **TinyLlama-1.1B** | 1.1B | 22 | 2048 | 32 | 3T |

**Published benchmarks:**

| Benchmark | TinyLlama-1.1B |
|-----------|----------------|
| HellaSwag (acc_norm) | 59.2% |
| PIQA (acc) | 73.3% |
| ARC-Easy (acc) | 55.4% |
| ARC-Challenge (acc_norm) | 30.1% |
| WinoGrande (acc) | 59.1% |
| WikiText-2 (ppl) | 7.63 |

Too large for direct param-matched comparison, but useful as an "upper bound" reference.

**Source:** Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385, 2024.

### 1.5 OPT (Meta, 2022)

| Model | Params | Training tokens |
|-------|--------|-----------------|
| **OPT-125M** | 125M | ~300B |
| **OPT-350M** | 350M | ~300B |

Older training recipe, public weights. OPT-125M is a common comparison target.

**Source:** Zhang et al., "OPT: Open Pre-trained Transformer Language Models," arXiv:2205.01068, 2022.

---

## 2. Recurrent / SSM Baselines

### 2.1 Mamba (Gu & Dao, 2023-2024) -- PRIMARY RECURRENT BASELINE

Selective State Space Model (S6). Linear-time sequence modeling with input-dependent state transitions. The most prominent SSM architecture.

| Model | Params | Layers | D_model | Training data |
|-------|--------|--------|---------|---------------|
| **Mamba-130M** | 130M | 24 | 768 | Pile (~300B) |
| **Mamba-370M** | 370M | 48 | 1024 | Pile (~300B) |
| **Mamba-790M** | 790M | 48 | 1536 | Pile (~300B) |

**Published benchmarks (from Mamba paper, Table 3):**

| Benchmark | Mamba-130M | Mamba-370M | Mamba-790M |
|-----------|-----------|------------|------------|
| HellaSwag (acc_norm) | ~28% | ~35% | ~40% |
| PIQA (acc) | ~63% | ~68% | ~72% |
| ARC-Easy (acc) | ~48% | ~55% | ~61% |
| ARC-Challenge (acc_norm) | ~24% | ~26% | ~29% |
| WinoGrande (acc) | ~51% | ~52% | ~55% |
| LAMBADA (ppl) | 23.4 | 10.4 | 6.6 |

**Key properties:**
- 5x generation throughput vs transformers (constant-time inference per token)
- Linear training complexity in sequence length
- Struggles more than transformers on in-context retrieval / associative recall at small scale
- Excels at long-range dependencies when properly evaluated

**Source:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, 2023.

### 2.2 RWKV (BlinkDL, 2023-2025)

Linear attention RNN. Can be formulated as transformer (for training) or RNN (for inference). O(T) time, O(1) inference memory.

| Model | Params | Layers | D_model | Training data |
|-------|--------|--------|---------|---------------|
| **RWKV-4 169M** | 169M | 12 | 768 | Pile |
| **RWKV-4 430M** | 430M | 24 | 1024 | Pile |

**Published benchmarks (from RWKV paper):**

| Benchmark | RWKV-4 169M | RWKV-4 430M |
|-----------|-------------|-------------|
| Pile val (ppl) | ~10.97 | ~7.92 |
| LAMBADA (acc) | ~50% | ~59% |
| HellaSwag (acc_norm) | ~28% | ~33% |
| PIQA (acc) | ~65% | ~68% |

RWKV-5/6/7 introduced matrix-valued states and data-dependent gating. RWKV-7 3B achieves SoTA among recurrent models at 3B scale (2025).

**Source:** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era," arXiv:2305.13048, 2023.

### 2.3 xLSTM (Hochreiter et al., 2024-2025)

Extended LSTM with exponential gating (sLSTM) and matrix memory (mLSTM). Residual block architecture. Published at NeurIPS 2024.

| Model | Params | Architecture | Training data |
|-------|--------|-------------|---------------|
| **xLSTM 125M** | 125M | mLSTM + sLSTM blocks | SlimPajama (15B tokens) |
| **xLSTM 350M** | 350M | mLSTM + sLSTM blocks | SlimPajama (15B tokens) |

**Published benchmarks (from xLSTM paper, Section 4):**
- On SlimPajama validation, xLSTM matches or exceeds Transformer++, Mamba, and RWKV-5/6 at the 125M and 350M scales
- xLSTM shows strong length extrapolation (trained on 2048, tested up to 16K tokens)
- Achieves competitive rare-token prediction on WikiText-103

**Key properties:**
- Parallelizable (mLSTM) + fully recurrent (sLSTM) hybrid
- Exponential gating enables dynamic memory revision (addresses LSTM's inability to overwrite)
- Matrix memory (mLSTM) provides higher capacity than scalar cells
- xLSTM 7B (2025) demonstrates competitive inference speed vs Mamba and Llama

**Source:** Beck et al., "xLSTM: Extended Long Short-Term Memory," NeurIPS 2024. arXiv:2405.04517.

### 2.4 Griffin (Google DeepMind, 2024)

Hybrid: gated linear recurrence + local multi-query attention. Alternates 2 Hawk blocks (pure recurrence) + 1 attention block.

| Model | Params | Architecture |
|-------|--------|-------------|
| **Griffin** | Various (1B+) | Hybrid recurrent + local attention |

Griffin matches or exceeds transformer performance while achieving 5-7x higher throughput on long sequences. Primary results published at 1B+ scale, but the architecture is applicable at smaller scales.

**Source:** De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models," arXiv:2402.19427, 2024.

### 2.5 GLA / Gated Slot Attention (NeurIPS 2024)

Gated Linear Attention with bounded-memory-control. Efficient linear-time sequence modeling with explicit memory slots.

| Model | Params | Architecture |
|-------|--------|-------------|
| **GLA / GSA** | Various (125M-1.3B) | Linear attention + gating + memory slots |

Relevant because GSA explicitly uses a "slot" memory mechanism with gating, conceptually similar to our PM slots.

**Source:** Zhang et al., "Gated Slot Attention for Efficient Linear-Time Sequence Modeling," NeurIPS 2024.

### 2.6 Kimi Delta Attention / DeltaNet (2025)

Linear attention with delta rule updates. Kimi Linear (3B activated, 48B total MoE) outperforms full attention. Based on Gated DeltaNet with finer-grained gating.

Relevant as a state-of-the-art reference for recurrent models with memory update rules (delta rule is conceptually related to our eligibility-based PM writes).

**Source:** Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture," Technical Report, 2025.

---

## 3. Memory-Augmented Models

### 3.1 MemoryLLM / M+ (Wang et al., 2024-2025)

Compresses past information into hidden states across all layers, forming a memory pool of ~1B parameters. Extends context window of existing LLMs with latent-space memory.

| Model | Base | Memory pool | Architecture |
|-------|------|-------------|-------------|
| **MemoryLLM** | Llama-2 7B+ | ~1B params in memory pool | Latent memory injection at all layers |
| **M+** | Extension of MemoryLLM | Scalable long-term memory | Hierarchical memory with compression |

Too large for direct comparison, but conceptually relevant: both MemoryLLM and our model maintain persistent memory that accumulates across sequences. Their evaluation methodology (cross-document recall, long-range QA) is directly applicable.

**Source:** Wang et al., "MemoryLLM: Towards Self-Updatable Large Language Models," 2024. Wang et al., "M+: Extending MemoryLLM with Scalable Long-Term Memory," 2025.

### 3.2 LM2: Large Memory Models (Kang et al., 2025)

Augments transformer with explicit episodic memory banks. Processes information through memory read/write operations.

**Source:** Kang et al., "LM2: Large Memory Models," arXiv:2502.06049, 2025.

### 3.3 HMT: Hierarchical Memory Transformer (2024)

Hierarchical memory for long-context language processing. Uses memory segmentation with learned read/write.

**Source:** Bulatov et al., "HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing," arXiv:2405.06067, 2024.

### 3.4 Recurrent Memory Transformer (Bulatov et al., 2024)

Adds recurrent memory tokens to transformer. Achieves needle-in-haystack retrieval up to 11M tokens.

**Source:** Kuratov et al., "In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss," arXiv:2402.10790, 2024.

### 3.5 LifeSpan Cognitive Systems (Wang et al., 2025)

Framework for continuous, high-frequency interactions with environments. Emphasizes incremental updates and accurate recall -- conceptually aligned with our Phase E lifelong learning.

**Source:** Wang et al., "Towards LifeSpan Cognitive Systems," TMLR 2025.

### 3.6 Birdie: Dynamic Mixtures of Training Objectives for SSMs (EMNLP 2024)

Addresses SSM weakness in in-context retrieval by mixing training objectives (standard LM loss + synthetic copy/recall tasks). Shows that training procedure matters as much as architecture for memory tasks.

**Source:** Blouir et al., "Birdie: Advancing State Space Language Modeling with Dynamic Mixtures of Training Objectives," EMNLP 2024.

---

## 4. Consolidated Benchmark Table

Best available zero-shot numbers for models nearest to our ~56M parameter scale. All numbers from published papers using lm-evaluation-harness unless noted.

| Model | Params | Type | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | LAMBADA (acc) |
|-------|--------|------|-----------|------|-------|-------|------------|---------------|
| **Pythia-70M** | 70M | Transformer | ~27% | ~62% | ~44% | ~23% | ~51% | ~33% |
| **Pythia-160M** | 160M | Transformer | ~30% | ~66% | ~49% | ~24% | ~52% | ~43% |
| **GPT-2 Small** | 124M | Transformer | ~29% | ~65% | ~44% | ~23% | ~52% | ~45% |
| **OPT-125M** | 125M | Transformer | ~29% | ~63% | ~44% | ~23% | ~50% | ~38% |
| **SmolLM-135M** | 135M | Transformer | ~42% | ~68% | ~52% | ~28% | ~52% | -- |
| **Mamba-130M** | 130M | SSM | ~28% | ~63% | ~48% | ~24% | ~51% | -- |
| **RWKV-4 169M** | 169M | Linear RNN | ~28% | ~65% | -- | -- | -- | ~50% |
| **xLSTM-125M** | 125M | LSTM (ext.) | ~29%* | ~64%* | -- | -- | -- | -- |
| **Pythia-410M** | 410M | Transformer | ~34% | ~69% | ~55% | ~27% | ~53% | ~56% |
| **SmolLM-360M** | 360M | Transformer | ~54% | ~73% | ~59% | ~36% | ~55% | -- |
| **Mamba-370M** | 370M | SSM | ~35% | ~68% | ~55% | ~26% | ~52% | -- |
| **RWKV-4 430M** | 430M | Linear RNN | ~33% | ~68% | -- | -- | -- | ~59% |

*xLSTM-125M benchmarks estimated from SlimPajama scaling curves in the xLSTM paper; exact zero-shot numbers not individually reported.

**Important caveats:**
- SmolLM numbers appear higher due to 600B training tokens of curated data vs ~300B for Pythia/Mamba on less filtered data
- Benchmark numbers vary by lm-eval-harness version and prompt format
- Our model at ~56M is between Pythia-70M and the next tier down (no standard baselines below 70M)
- Direct parameter matching is Pythia-70M; architecture matching (recurrent) is Mamba-130M

---

## 5. Evaluation Benchmarks & Metrics

### 5.1 Language Modeling Perplexity (Primary)

| Dataset | Description | Why use it | Notes |
|---------|-------------|-----------|-------|
| **WikiText-103** | Wikipedia, 103M tokens | Standard LM benchmark, most-cited | Report word-level PPL; specify tokenizer |
| **WikiText-2** | Wikipedia, 2M tokens | Faster evaluation | Same family as WT-103 |
| **C4 validation** | Filtered Common Crawl | Diverse web text | Used by T5, PaLM papers |
| **LAMBADA** | Story completion | Tests long-range prediction | Report both PPL and accuracy |
| **Pile validation** | Diverse mix | Matches Pythia/Mamba training dist. | Enables direct comparison |
| **PG19** | Long books | Tests long-context ability | Critical for PM/EM evaluation |

**Recommendation:** Report WikiText-103 PPL (for broad comparison) + Pile validation PPL (for Pythia/Mamba comparison) + PG19 PPL (for memory advantage).

### 5.2 Zero-Shot Downstream Tasks

Use lm-evaluation-harness (EleutherAI) for reproducibility. These are the standard tasks that differentiate at small scale:

| Task | Type | Metric | Differentiates at <100M? |
|------|------|--------|--------------------------|
| **HellaSwag** | Commonsense reasoning | acc_norm | Weakly (most models ~25-30%) |
| **PIQA** | Physical intuition | acc | Yes (62-68% range) |
| **ARC-Easy** | Science QA | acc | Yes (44-55% range) |
| **ARC-Challenge** | Hard science QA | acc_norm | Barely (23-27% range) |
| **WinoGrande** | Coreference | acc | No (all ~50%, random) |
| **LAMBADA** | Word prediction | acc | Yes (33-56% range) |
| **BoolQ** | Yes/no QA | acc | Weakly |
| **OpenBookQA** | Open-book science | acc_norm | Weakly |

**Recommendation for ~56M model:** Focus on PIQA, ARC-Easy, LAMBADA (most signal). Report HellaSwag and ARC-Challenge for completeness but expect near-random. Skip WinoGrande (saturated at random for this scale).

### 5.3 Token-Level and Efficiency Metrics

| Metric | Description |
|--------|-------------|
| Tokens/second (training) | Throughput on target hardware (4090) |
| Tokens/second (inference) | Generation speed |
| Peak VRAM | Training and inference |
| FLOPs per token | Forward pass compute |
| Bits per byte (BPB) | Tokenizer-independent perplexity |

---

## 6. Memory-Specific Evaluations

These benchmarks specifically test whether PM/EM/WM provide value over vanilla architectures. This is the most important section for demonstrating the architecture "makes sense."

### 6.1 Synthetic Memory Tasks

#### 6.1.1 Multi-Query Associative Recall (MQAR)

The standard benchmark for evaluating in-context memory. Highly correlated with real language modeling performance (Arora et al., 2024; Okpekpe & Orvieto, 2025).

**Setup:** Present key-value pairs interleaved in a sequence, then query for specific values.
```
Input:  k1 v1 k2 v2 k3 v3 ... [query] k2 [answer] ?
Target: v2
```

**Why critical:** Tests whether our EM retrieval and WM attention can associate and recall specific key-value pairs. Transformers naturally solve this via induction heads; recurrent models struggle. Our explicit EM should provide an advantage.

**Protocol:**
- Vary number of KV pairs (4, 8, 16, 32, 64)
- Vary sequence length (128, 256, 512, 1024)
- Measure exact-match accuracy
- Compare: full model vs EM-off vs WM-off

**Source:** Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," ICLR 2024.

#### 6.1.2 Copy Task

**Setup:** Present a sequence of tokens, delimiter, then model must reproduce the sequence.
```
Input:  a b c d e [SEP] ?
Target: a b c d e
```

**Why critical:** Tests raw memory capacity. Our WM (sliding window attention) should provide direct copying ability. Mamba and pure recurrent models struggle with this.

**Protocol:**
- Vary copy length (32, 64, 128, 256, 512)
- Measure token-level accuracy
- Test extrapolation: train on length 128, test on 256+

**Source:** Graves et al., "Neural Turing Machines," 2014; Arjovsky et al., "Unitary Evolution Recurrent Neural Networks," ICML 2016.

#### 6.1.3 Induction Head Task

**Setup:** A B ... A ? (model should output B, pattern completion)

**Why critical:** Tests the formation of induction-head-like circuits. WM attention should enable this naturally; the question is whether our recurrent blocks can also learn this pattern through PM.

**Source:** Olsson et al., "In-context Learning and Induction Heads," Transformer Circuits, 2022.

### 6.2 Long-Range Dependency Tasks

#### 6.2.1 PG19 Perplexity at Distance

**Setup:** Process full books. Measure per-token perplexity as a function of position within the document.

**Why critical:** PM accumulates knowledge about patterns seen earlier in the book. EM stores specific episodes. Both should reduce perplexity at later positions compared to a vanilla recurrent model without memory.

**Protocol:**
- Process 100+ full books from PG19 validation
- Bucket perplexity by position: [0-1K], [1K-5K], [5K-10K], [10K-50K], [50K+]
- Compare: full model vs PM-off vs EM-off vs both-off
- Plot perplexity vs position curve

**Expected result:** Full model should show continued perplexity decrease at long distances where PM-off/EM-off plateau.

#### 6.2.2 Needle-in-a-Haystack

**Setup:** Insert a specific fact at position P in a long document, query for it at position Q >> P.

**Protocol:**
- Context lengths: 256, 512, 1K, 2K, 4K, 8K
- Needle positions: 10%, 25%, 50%, 75%, 90% of context
- Measure retrieval accuracy
- Compare architectures (transformer, Mamba, our model)

**Source:** Kamradt, 2023; Kuratov et al., "In Search of Needles in a 11M Haystack," 2024.

#### 6.2.3 Selective Copy / Selective Recall

**Setup:** Like copy task but with distractors. Model must selectively recall only specific items based on a cue.

**Why critical:** Tests whether EM can selectively retrieve relevant information while ignoring noise -- exactly what top-k retrieval with learned novelty scoring should enable.

### 6.3 Cross-Document Knowledge Retention (Phase E / Lifelong)

#### 6.3.1 Domain Adaptation Curve

**Setup:** Stream Wikipedia articles from a specific domain (e.g., chemistry). Measure per-document perplexity.

**Why critical:** In lifelong mode, PM should accumulate domain-specific patterns. Perplexity on later chemistry articles should decrease faster than a model without persistent memory.

**Protocol:**
- Stream 200 articles from a single domain
- Measure perplexity per article
- Compare lifelong mode vs reset-at-boundary mode
- Compare against transformer with same context window

**Already implemented:** `src/training/eval_lifelong.py`

#### 6.3.2 Cross-Document Fact Recall

**Setup:** Present facts in document A, test recall in document B (separated by EOT).

**Protocol:**
- Fact types: named entities, numerical values, definitions
- Distances: 0.5K, 1K, 2K, 5K, 10K tokens between fact and probe
- Measure exact-match and perplexity-delta at probe position
- Compare lifelong vs non-lifelong

#### 6.3.3 Forgetting Curve

**Setup:** After domain adaptation, measure perplexity on general text (FineWeb-Edu validation).

**Why critical:** Demonstrates that lifelong PM/EM learning does not cause catastrophic forgetting.

**Protocol:**
- Baseline: general text perplexity before domain adaptation
- After streaming N domain documents, re-evaluate general perplexity
- Acceptable threshold: <5% degradation

### 6.4 Few-Shot In-Context Learning

**Setup:** Standard few-shot evaluation with k=0,1,2,4,8 examples.

**Why critical:** WM (sliding window attention) enables attending to examples. EM could provide "pseudo-few-shot" by retrieving relevant past episodes.

**Protocol:**
- Tasks: simple classification (SST-2, TREC), arithmetic, word manipulation
- Compare k=0,1,2,4,8 performance curves across architectures
- Hypothesis: our model should benefit more from additional examples (steeper learning curve) due to WM attention

### 6.5 Memory Utilization Analysis

These are not benchmarks but diagnostic analyses to include in the paper:

| Analysis | What it shows |
|----------|--------------|
| PM slot activation heatmap | Which PM slots are active, how activation distributes across layers/blocks |
| PM commit rate over training | How often and where PM commits happen |
| EM write rate and novelty distribution | Whether EM learns to be selective |
| EM retrieval similarity histogram | Quality of EM retrievals |
| WM attention pattern visualization | What WM attends to |
| Surprise distribution over time | How surprise modulates the system |
| Memory state norm trajectories | Stability of PM/EM state over long sequences |

---

## 7. Experimental Design

### 7.1 Ablation Studies

The most important experiments for proving the architecture works. Each ablation disables one component while keeping everything else identical.

#### 7.1.1 Component Ablations

| Ablation | Config change | What it tests |
|----------|--------------|---------------|
| **Full model** | All enabled | Baseline |
| **No PM** | `pm_enabled=False` (zero pm_a) | Value of procedural memory |
| **No EM** | `em_enabled=False` (skip retrieval/writes) | Value of episodic memory |
| **No WM** | `wm_enabled=False` (skip WM attention) | Value of working memory |
| **No PM + No EM** | Both off | Value of memory systems vs bare recurrence |
| **No PM + No EM + No WM** | All memory off | Pure recurrent baseline |
| **No Spatial Decoder** | `snapshot_enabled=False` | Value of hierarchical decoding |
| **No Surprise Modulation** | Fixed surprise=1.0 | Value of neuromodulation |

**Protocol for each ablation:**
- Train from scratch with identical data, hyperparameters, and random seed
- Train for same number of tokens (controlled compute budget)
- Evaluate on all benchmarks above
- Report: perplexity difference, downstream task difference, memory-specific task difference

#### 7.1.2 Memory Capacity Ablations

| Ablation | Config change | What it tests |
|----------|--------------|---------------|
| PM r=2 | Reduce PM slots | PM capacity sensitivity |
| PM r=16 | Increase PM slots | Scaling returns |
| EM M=64 | Reduce EM capacity | EM capacity sensitivity |
| EM M=512 | Increase EM capacity | Scaling returns |
| WM W=64 | Reduce WM window | WM window sensitivity |
| WM W=512 | Increase WM window | Scaling returns |

#### 7.1.3 Phase-Wise Ablations (Neuromodulation)

| Ablation | What it tests |
|----------|---------------|
| Phase B (heuristic PM commit) vs Phase D (RL commit) | Value of learned commit policy |
| Phase C (heuristic EM write) vs Phase D (RL write) | Value of learned write policy |
| Fixed g_pm=0.5 vs learned g_pm | Value of learned write strength |
| Fixed g_em=0.3 vs learned g_em | Value of learned write strength |

### 7.2 Scaling Experiments

#### 7.2.1 IsoFLOP Comparison

Train multiple model sizes at the same total FLOP budget. This is the gold standard for architecture papers (used by Mamba, Griffin, xLSTM).

**Protocol:**
- FLOP budget levels: 1e17, 3e17, 1e18, 3e18, 1e19 FLOPs
- At each level, find optimal model size (vary D, L, B)
- Compare: our architecture vs transformer (same FLOP budget) vs Mamba (same FLOP budget)
- Plot: validation loss vs FLOPs for each architecture

**Implementation:**
```
For each architecture:
  For each FLOP budget:
    Grid search over (D, L) to find compute-optimal config
    Train to completion
    Record validation loss
Plot architecture scaling curves on shared axes
```

This demonstrates whether our architecture achieves better loss-per-FLOP than baselines -- the strongest possible evidence for architectural merit.

#### 7.2.2 Parameter-Matched Scaling

Train at 3-4 model sizes with matched parameter counts:

| Tier | Our model | Pythia | Mamba |
|------|-----------|--------|-------|
| ~50M | Tier A (D=512, L=8, B=4) | Pythia-70M | Mamba-130M* |
| ~150M | Tier B (D=768, L=12, B=6) | Pythia-160M | Mamba-130M |
| ~350M | Tier C (D=1024, L=24, B=8) | Pythia-410M | Mamba-370M |

*No Mamba at exactly 50M; closest is 130M.

**Plot:** Validation loss vs parameter count for each architecture family. If our curve is below baselines, the architecture is more parameter-efficient.

#### 7.2.3 Token-Matched Comparison

All models trained on same number of tokens (e.g., 10B tokens from FineWeb-Edu).

**Why important:** Eliminates confounds from different training data amounts. Pythia was trained on 300B tokens; if we train on 10B, the comparison is unfair. Train custom baselines or use early Pythia checkpoints (available every 2B tokens).

### 7.3 Fair Comparison Methodology

Following best practices from Mamba (2023), Griffin (2024), xLSTM (2024):

#### 7.3.1 Controlled Variables

| Variable | Strategy |
|----------|----------|
| **Training data** | Same dataset (FineWeb-Edu subset) for all models |
| **Tokenizer** | Same tokenizer (GPT-NeoX / Llama tokenizer with vocab ~32K) |
| **Training tokens** | Same total tokens for all models at each comparison point |
| **Optimizer** | AdamW with same hyperparameters (or per-model tuned LR with sweep) |
| **Batch size** | Same effective batch size (tokens per update) |
| **Precision** | bf16 for all models |
| **Evaluation** | Same lm-eval-harness version, same prompts |

#### 7.3.2 What Must Be Tuned Per-Model

| Variable | Strategy |
|----------|----------|
| **Learning rate** | Grid search per architecture (1e-4 to 1e-3) |
| **Warmup steps** | May need adjustment per architecture |
| **Weight decay** | Standard 0.01 unless architecture-specific recommendation exists |

#### 7.3.3 Baseline Implementation Options

**Option A: Train baselines from scratch** (best for fairness)
- Implement transformer and Mamba at matched param count
- Train on identical data with identical pipeline
- Most work but strongest claims

**Option B: Use published checkpoints + early checkpoints** (pragmatic)
- Use Pythia checkpoints at matched token counts
- Note: different training data (Pile vs FineWeb-Edu)
- Weaker claims but much less compute

**Option C: Hybrid** (recommended)
- Train small transformer baseline from scratch (matching our Tier A exactly)
- Use Pythia checkpoints for reference
- Use Mamba published numbers for SSM comparison

### 7.4 Statistical Rigor

- Report mean and standard deviation over 3 random seeds for key experiments
- Use bootstrap confidence intervals for downstream task accuracy
- Report training loss curves (not just final numbers)
- Show convergence: validation loss vs step/token count

---

## 8. Recommended Comparison Strategy

### 8.1 Minimum Viable Paper (MVP) Experiments

These are the experiments needed to convincingly demonstrate the architecture:

#### Priority 1: Core Claims (must have)
1. **Perplexity on WikiText-103 and Pile validation** at Tier A (~56M) vs Pythia-70M
2. **Component ablation table** (full, no-PM, no-EM, no-WM, bare recurrence) on WikiText-103
3. **PG19 perplexity vs position curve** showing memory advantage at long distances
4. **MQAR accuracy** at varying sequence lengths showing EM advantage
5. **PIQA, ARC-Easy, LAMBADA** zero-shot (with lm-eval-harness)

#### Priority 2: Architecture Understanding (should have)
6. **Copy task** showing WM advantage
7. **Memory utilization heatmaps** (PM activation, EM write patterns)
8. **Scaling curve** at 2-3 model sizes (Tier A, Tier B) vs Pythia at matched params
9. **Needle-in-haystack** retrieval accuracy vs context length
10. **Training throughput** comparison (tokens/sec vs Pythia, Mamba at matched params)

#### Priority 3: Advanced Claims (nice to have)
11. **Lifelong learning domain adaptation curve** (Phase E)
12. **RL ablation** (Phase D vs Phase C heuristics)
13. **IsoFLOP scaling** at 3+ compute levels
14. **Cross-document fact recall** at various distances
15. **Forgetting curve** post-domain-adaptation

### 8.2 Narrative Structure for a Paper

A paper proving this architecture "makes sense" should follow this structure:

1. **Motivation:** Biological inspiration, memory system decomposition (PM/EM/WM parallel to human memory)
2. **Architecture:** Full description with diagrams
3. **Language Modeling:** Perplexity results showing competitive or superior performance vs Pythia/Mamba at matched scale
4. **Ablation Studies:** Each component contributes; removing any one hurts
5. **Memory-Specific Tasks:** MQAR, copy, needle-in-haystack showing where memory systems excel
6. **Long-Range:** PG19 perplexity curves showing sustained improvement at distance
7. **Scaling:** Evidence that the architecture scales (2-3 points minimum)
8. **Efficiency:** Throughput analysis showing practical viability
9. **Lifelong Learning:** (if included) Domain adaptation demonstrating cross-document knowledge transfer

### 8.3 Key Claims to Support

| Claim | Evidence needed |
|-------|----------------|
| "PM improves pattern learning" | PM-on vs PM-off perplexity gap; PM utilization analysis |
| "EM enables episodic retrieval" | MQAR accuracy; needle-in-haystack; EM-on vs EM-off on PG19 |
| "WM enables in-context attention" | Copy task; few-shot learning curves; WM-on vs WM-off |
| "Memory systems compose" | Full model > any single component ablation |
| "Architecture scales" | Perplexity at 2+ model sizes tracks or beats baseline scaling |
| "Recurrence + memory is practical" | Training throughput within 2x of transformer baseline |

---

## 9. References

### Architecture Papers
- **Pythia:** Biderman et al., "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling," ICML 2023. arXiv:2304.01373
- **Mamba:** Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, 2023
- **Mamba-2:** Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," ICML 2024
- **RWKV:** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era," EMNLP 2023. arXiv:2305.13048
- **xLSTM:** Beck et al., "xLSTM: Extended Long Short-Term Memory," NeurIPS 2024. arXiv:2405.04517
- **xLSTM 7B:** Beck et al., "xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference," arXiv:2503.13427, 2025
- **Griffin:** De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models," arXiv:2402.19427, 2024
- **GPT-2:** Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI, 2019
- **SmolLM:** Allal et al., "SmolLM - blazingly fast and remarkably powerful," HuggingFace, 2024. arXiv:2502.02737
- **TinyLlama:** Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385, 2024
- **OPT:** Zhang et al., "OPT: Open Pre-trained Transformer Language Models," arXiv:2205.01068, 2022
- **OLMo:** Groeneveld et al., "OLMo: Accelerating the Science of Language Models," ACL 2024

### Memory-Augmented Models
- **MemoryLLM:** Wang et al., "MemoryLLM: Towards Self-Updatable Large Language Models," 2024
- **M+:** Wang et al., "M+: Extending MemoryLLM with Scalable Long-Term Memory," 2025
- **LM2:** Kang et al., "LM2: Large Memory Models," arXiv:2502.06049, 2025
- **HMT:** Bulatov et al., "HMT: Hierarchical Memory Transformer," arXiv:2405.06067, 2024
- **RMT:** Kuratov et al., "In Search of Needles in a 11M Haystack," arXiv:2402.10790, 2024
- **LifeSpan Cognitive Systems:** Wang et al., TMLR 2025
- **GSA:** Zhang et al., "Gated Slot Attention for Efficient Linear-Time Sequence Modeling," NeurIPS 2024
- **Kimi Linear:** Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture," 2025
- **DeltaNet:** Yang et al., "Parallelizing Linear Transformers with the Delta Rule," ICML 2024

### Benchmarks & Evaluation
- **MQAR / Zoology:** Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," ICLR 2024
- **Associative Recall Dynamics:** Okpekpe & Orvieto, "Revisiting Associative Recall in Modern Recurrent Models," arXiv:2508.19029, 2025
- **Birdie:** Blouir et al., "Birdie: Advancing State Space Language Modeling with Dynamic Mixtures of Training Objectives," EMNLP 2024
- **SLM-Bench:** "Small Language Models: Survey, Measurements, and Insights," arXiv:2409.15790, 2024
- **lm-evaluation-harness:** Gao et al., EleutherAI, github.com/EleutherAI/lm-evaluation-harness

### Scaling Laws
- **Chinchilla:** Hoffmann et al., "Training Compute-Optimal Large Language Models," NeurIPS 2022. arXiv:2203.15556
- **Scaling Law Estimation:** Choshen et al., "A Hitchhiker's Guide to Scaling Law Estimation," ICLR 2025
- **xLSTM Scaling Laws:** Beck et al., "xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity," arXiv:2510.02228, 2025
- **Compute-Optimal Scaling Discrepancies:** Porian et al., NeurIPS 2024
