# Training Dataset & Configuration Plan — Neuromorphic AI on a Single RTX 4090

## Overview

This plan covers datasets, configuration, and training phases for the ~40M parameter neuromorphic model (D=512, L=8, B=4) on a single RTX 4090 (24 GB VRAM). It addresses: basic LM skills, RL-based neuromodulation, lifelong learning / post-training adaptation, agentic capabilities (start small), and future multimodal expansion.

**Aligned with:** spec v2.0

### v2 Architecture Summary
- **B blocks** (default 6), each containing **L layers** (default 12)
- **Working Memory (WM):** sliding-window attention (W=256), 1 shared instance
- **Procedural Memory (PM):** fast low-rank weights + eligibility, B×L instances (each with `PMController`)
- **Episodic Memory (EM):** per-stream vector store, B instances (each with `EMController`)
- **Plasticity boundaries:** PM/EM writes every P=32 tokens (scan-friendly within spans)

---

## 1. VRAM Budget & Model Scaling

### Scaling Tiers (all fit on RTX 4090)

| Tier | Params | D | L | B | Target | 4090 BS |
|------|--------|---|---|---|--------|---------|
| **A (Debug)** | ~50M | 512 | 8 | 4 | Rapid iteration | 32–64 |
| **B (Competitive)** | ~150M | 768 | 12 | 6 | Match GPT-2 Small | 16–32 |
| **C (Strong)** | ~350M | 1024 | 24 | 8 | Match GPT-2 Medium | 8–16 |

**Note:** Early LLMs (GPT-1/GPT-2 Small at ~125M) showed meaningful language understanding. **Tier B is the recommended target** for demonstrating competitive results.

### VRAM Budget (Tier B, bf16 training)

| Component | Estimate |
|-----------|----------|
| Weights (bf16) | ~300 MB |
| Optimizer (fp32) | ~1.2 GB |
| Gradients (bf16) | ~300 MB |
| Activations (BS=16) | ~2 GB |
| PM/EM/WM state | ~100 MB |
| **Total** | **~4 GB** (16% of 24 GB) |

### Base Training Hyperparameters (Tier B)
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 (cosine decay to 1e-5) |
| Weight decay | 0.01 |
| Batch size (BS) | 16 (can increase for Tier A) |
| TBPTT chunk (T) | 256 tokens |
| Plasticity span (P) | 32 tokens |
| Gradient clipping | 1.0 (global norm) |
| Precision | **bf16** (PM/EM state in fp32) |
| Warmup | 1000 steps |

---

## 2. Training Phases & Datasets

### Phase A — Base Competence (WM + scan core, no PM/EM writes)

**Goal:** Verify the backbone + WM learns language; PM/EM disabled or read-only.

**Dataset:** TinyStories (small, clean, fast iteration)
- HuggingFace: `roneneldan/TinyStories`
- ~2.1M synthetic short stories, ~470M tokens
- Simple vocabulary, clear narrative structure
- Perfect for verifying loss goes down and the model generates coherent text

**Config:**
- WM enabled (sliding window attention)
- PM reads disabled (a=0) or entirely off
- EM retrieval/writes disabled
- BS=32, ~5–10K steps

**Exit criteria:** Loss decreases smoothly, generated text is coherent English, WM cache stable.

---

### Phase B — Enable PM (heuristic commits at plasticity boundaries)

**Goal:** Train general language competence with Procedural Memory active. `PMController` logs decisions but uses heuristic commit timing initially.

**Primary dataset:** FineWeb-Edu (sample-10BT)
- HuggingFace: `HuggingFaceFW/fineweb-edu` (use the `sample-10BT` subset)
- ~10B tokens of high-quality educational web text
- Filtered and deduplicated, good quality/size ratio
- Streaming-friendly (no need to download all at once)

**Secondary dataset:** SlimPajama-627B (for diversity)
- HuggingFace: `cerebras/SlimPajama-627B`
- Mix of web, books, code, Wikipedia, StackExchange, arXiv
- Use a 5–10B token subset via streaming
- Provides domain diversity the model needs

**Recommended mix:** 70% FineWeb-Edu + 30% SlimPajama (interleaved streaming)

**Long-context PM training:** PG19 (full-length books)
- HuggingFace: `deepmind/pg19`
- ~28K public domain books, avg ~69K tokens each
- Critical for exercising PM across long sequences
- Use as supplementary stream: process full books sequentially to test memory persistence

**Config:**
- BS=16, TBPTT T=256, plasticity span P=32
- PM enabled: reads always, commits at span boundaries (heuristic: elig_norm threshold or every span)
- EM still disabled (retrieval/writes off)
- `PMController` in logging-only mode (records what it would do)
- Base decay 0.999/span, eligibility decay ρ=0.95
- All stability rails active (clip, budget, normalize, commit cap)

**Training budget:** ~2–5B tokens total (streaming, no epoch needed)

**Evaluation:**
- Validation perplexity on held-out FineWeb-Edu
- PM ON vs OFF comparison (perplexity gap)
- PG19 long-range memory: fact recall at distance (>1000 tokens back)
- Stable PM budgets across blocks

---

### Phase C — Add EM Retrieval + Heuristic Writes

**Goal:** Enable Episodic Memory retrieval and writes. `EMController` uses heuristic write policy initially.

**Dataset:** Continue FineWeb-Edu + SlimPajama streaming

**Supplementary datasets for challenging memory scenarios:**
- **PG19** (continued): Long books stress-test when to write to EM vs. not
- **ProofPile-2** (subset): Mathematical/logical text where memory of definitions matters
  - HuggingFace: `EleutherAI/proof-pile-2`
  - Use ~1–2B token subset
- **RedPajama v2 — CommonCrawl news subset**: Factual text where storing key facts helps
  - Recent news articles with names, dates, facts

**Config:**
- PM continues with heuristic commits (same as Phase B)
- EM retrieval enabled: top k_ret=4 latent memory tokens → cross-attention aggregation
- EM writes at span boundaries: heuristic (surprise spike + novelty threshold)
- `EMController` in logging-only mode (records decisions for later RL)
- All stability rails active (EM budget, strength caps)

**Training budget:** ~2–3B tokens

**Evaluation:**
- Explicit recall improvements at long delays (>1000 tokens)
- EM usage statistics per block
- Compare EM ON vs OFF for long-context tasks
- Perplexity should improve on PG19 with EM enabled

---

### Phase D — Learn Gating via RL (optional)

**Goal:** Train `PMController` and `EMController` commit/write policies using RL (boundary-time decisions).

**Dataset:** Continue FineWeb-Edu + SlimPajama + ProofPile-2 streaming

**Config:**
- Train `commit_mask` (per layer per block) and `write_mask` (per block)
- Keep λ/g/slot defaults initially
- Reward: future loss delta − memory/drift penalties
- Backbone LR reduced to 1e-4, controller LRs at 3e-4
- Continue all stability rails

**Training budget:** ~2–3B tokens with RL events at span boundaries

**Evaluation:**
- Compare RL-trained policies vs. Phase C heuristics
- Commit/write rate statistics (should be sparse and targeted)
- Memory benchmark: delayed key-value recall, passage continuation
- Drift suite: perplexity with PM/EM OFF should remain stable

---

### Phase E — Lifelong Learning & Persistence

**Goal:** Demonstrate the model adapts to new domains/users during inference without retraining. Disable doc-boundary resets.

**Datasets for adaptation evaluation:**
- **Wikipedia** (latest dump): Domain adaptation — feed articles from a specific topic, test recall
  - HuggingFace: `wikimedia/wikipedia` (20231101.en)
  - Use specific topic subsets (e.g., all chemistry articles, then test chemistry knowledge)
- **Personalization corpus** (synthetic): Create user-profile sequences
  - Generate synthetic "user says X, later ask about X" test pairs
  - Test within-session personalization
- **NarrativeQA**: Reading comprehension over long narratives
  - HuggingFace: `deepmind/narrativeqa`
  - Tests whether PM/EM helps answer questions about previously-read content

**Config:**
- Deploy model from Phase D (or Phase C if skipping RL)
- `reset_on_doc_boundary = False` — PM/EM persist across documents
- No gradient updates to slow weights
- Only PM/EM state evolves during inference
- Use `save_state()`/`load_state()` for session persistence
- Measure adaptation quality over session length

**Evaluation:**
- Adaptation speed: how quickly does perplexity drop on new domain text?
- Fact recall: accuracy on injected facts at various distances (PM short-term, EM long-term)
- Cross-document memory: can the model reference facts from earlier documents?
- Stability: does the model degrade on general text after domain adaptation?

---

### Phase F (Future) — Agentic Learning (Start Small)

**Goal:** The model learns tool-use patterns and improves task execution within sessions using PM/EM.

**Datasets (start small, expand later):**

| Dataset | Description | Size | Use |
|---------|-------------|------|-----|
| `glaiveai/glaive-function-calling-v2` | Function calling conversations | ~113K examples | Basic tool-use patterns |
| `teknium/OpenHermes-2.5` | Diverse instruction-following | ~1M examples | General instruction competence |
| `nvidia/Nemotron-Agentic-v1` | Multi-turn agentic conversations | Varies | Advanced agentic patterns |

**Approach:**
- First: fine-tune on function-calling data (formatted as special tokens)
- Then: test whether PM/EM helps the model improve tool selection within a session
- PM: stores recent tool patterns (procedural)
- EM: stores tool-use episodes for later retrieval (episodic)
- Measure: does commit/write rate correlate with tool-use improvement?

**Note:** This phase requires defining a tool-use token format. Start with a simple `<tool_call>...</tool_call>` wrapper, expand later.

---

### Phase G (Future) — Multimodal Expansion

**Goal:** Extend the token interface to accept image and video tokens.

**Datasets (for future planning):**

| Dataset | Modality | Size | Notes |
|---------|----------|------|-------|
| `liuhaotian/LLaVA-CC3M-Pretrain-595K` | Image-text pairs | 595K | Good for initial vision alignment |
| COCO Captions | Image-text | 330K images | Standard benchmark |
| `PixArt-alpha/SAM-LLaVA-Captions10M` | Image-text | 10M | Scale-up dataset |
| WebVid-10M | Video-text | 10M clips | Future video understanding |

**Approach:**
- Add a vision encoder (e.g., SigLIP) that produces tokens fed into the same backbone
- PM/EM should help with visual context persistence:
  - PM: stores recent visual-semantic associations
  - EM: stores visual episodes for long-term retrieval
- This is a significant architectural extension — plan separately when Phase E is solid

---

## 3. Batching: Persistent Parallel Streams

### Why recurrent models need different batching
Unlike transformers (which sample independent sequences), a recurrent model needs **persistent streams** where hidden state, PM, EM, WM cache, and eligibility carry over between TBPTT chunks.

### Design
Maintain BS parallel token streams. Each stream is a continuous flow of concatenated documents separated by `<|endoftext|>` tokens. Each training step consumes one TBPTT chunk (T=256 tokens) from all streams simultaneously.

```
Stream 0: [doc1...] <|eot|> [doc2...] <|eot|> [doc3...]
Stream 1: [doc4...] <|eot|> [doc5...] <|eot|> [doc6...]
...
Step 0: streams[:, 0:256]   → [BS, 256]
Step 1: streams[:, 256:512] → [BS, 256]
```

### State isolation
All state tensors have a batch dimension at position 0. Per block b, per layer ℓ:
- `h`: `[BS, D_h]` — recurrent hidden per stream
- `pm_K, pm_V`: `[BS, r, D_h]` — PM per stream
- `pm_a`: `[BS, r]` — PM strengths per stream
- `elig_K, elig_V`: `[BS, r, D_h]` — eligibility per stream

Per block b (shared across layers):
- `em_K, em_V`: `[BS, M, D_em]` — EM per stream
- `em_S`: `[BS, M]` — EM strengths per stream

Shared:
- `wm_K, wm_V`: `[BS, W, D_wm]` — WM cache per stream

No cross-stream mixing. PyTorch broadcast handles this naturally.

### Document boundary handling
When `<|endoftext|>` appears in a stream, a per-stream boolean mask triggers reset (or not, depending on phase). Different streams hit boundaries at different positions.

### For pre-tokenized data
1. Tokenize all documents, concatenate with `<|eot|>` separators into flat array
2. Reshape into `(BS, total_tokens // BS)`
3. Iterate in strides of T=256

### For HuggingFace streaming
Maintain BS independent iterators, each fills T-length buffers by tokenizing on the fly.

### Mixed datasets (Phase 1: 70/30 FineWeb-Edu/SlimPajama)
Option A: Pre-tokenize both and interleave into the flat array at desired ratio.
Option B: Assign ~70% of streams to FineWeb-Edu and ~30% to SlimPajama.

---

## 4. Memory Reset Curriculum (Phased Persistence)

### The tension
- Lifelong memory needs **trained controllers** (otherwise garbage accumulates in PM/EM)
- Training the controllers needs **clean signal** (document isolation helps credit assignment)
- Solution: phased curriculum

### Phase A–C: Reset at document boundaries
- `reset_on_doc_boundary = True`
- PM (`pm_K/pm_V/pm_a`), EM (`em_K/em_V/em_S`), eligibility (`elig_K/elig_V`), hidden state (`h`), and WM cache cleared when `<|eot|>` is encountered
- Clean learning signal: model learns within-document memory patterns
- No risk of stale memory from unrelated text

### Phase D: Reset + RL training
- Same reset policy as Phase A–C
- RL counterfactuals compare commit/write-vs-no-commit/write within a single document
- Document isolation helps credit assignment for `PMController` and `EMController` policies

### Phase E+: Lifelong (no resets)
- `reset_on_doc_boundary = False`
- Controllers are now trained and selective
- PM/EM persist across documents indefinitely
- Base decay (0.999/span) provides natural forgetting
- Budget caps force overwriting of weak slots when capacity is full

### Implementation
Single boolean flag: `config.reset_on_doc_boundary`
Per-stream reset via boolean mask (different streams hit doc boundaries at different times).

---

## 5. Dataset Disk Space & Access Strategy

### Disk space requirements

| Dataset | Phase | On-disk Size | Strategy |
|---------|-------|-------------|----------|
| TinyStories | Phase 0 | ~1–2 GB | Full download |
| FineWeb-Edu sample-10BT | Phase 1–2 | ~10 GB (parquet) | Full download recommended |
| SlimPajama-627B | Phase 1–2 | ~627 GB full / **0 GB** | **Stream only** (never full download) |
| PG19 | Phase 1–3 | ~12 GB | Full download (need full books) |
| ProofPile-2 | Phase 2 | ~100–200 GB full / **0 GB** | **Stream only** |
| Wikipedia (en) | Phase 3 | ~20–25 GB | Full download |
| glaive-function-calling-v2 | Phase 4 | ~1–5 GB | Full download |
| OpenHermes-2.5 | Phase 4 | ~10–20 GB | Full download |

**Realistic disk budget for Phases 0–2:** ~25–50 GB total (TinyStories + FineWeb-Edu + PG19; rest streamed).

### Access strategy

All primary datasets are available on HuggingFace and support streaming:

```python
from datasets import load_dataset

# Example: streaming FineWeb-Edu
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

# Example: streaming PG19
ds = load_dataset("deepmind/pg19", split="train", streaming=True)
```

**Tokenizer:** Use GPT-2 tokenizer (vocab ~50K) or train a custom BPE on FineWeb-Edu subset. The spec lists vocab ~50K — adopting GPT-2's tokenizer directly is the fast-start option.

**Data pipeline:**
- Use HuggingFace `datasets` with streaming mode (no full download needed)
- Concatenate documents with `<|endoftext|>` separator
- Pack into fixed-length chunks of TBPTT length (256 tokens)
- Shuffle buffer of ~10K examples for diverse sampling in Phases 0–1 (with KVA reset at document boundaries)
- Sequential/streaming for long-context memory evaluation (PG19 books)

---

## 6. 4090-Specific Optimizations

| Optimization | Details |
|-------------|---------|
| **Precision** | **bf16** for forward/backward (better stability than fp16), fp32 for PM/EM state (`pm_K/pm_V/pm_a/elig_K/elig_V/em_K/em_V/em_S`) |
| **Batch size** | Tier A: 32–64, Tier B: 16–32, Tier C: 8–16 |
| **Gradient accumulation** | Use if effective BS needs to be larger |
| **TBPTT** | Chunk T=256, detach all recurrent state (`h`, `elig_K`, `elig_V`, `pm_K`, `pm_V`) between chunks |
| **Plasticity span** | P=32 tokens; PM/EM writes at span boundaries enable scan-friendliness within spans |
| **Compile** | `torch.compile(model)` for kernel fusion (RTX 4090 Ada Lovelace supports it well) |
| **Data loading** | `num_workers=4`, `pin_memory=True`, streaming from disk/HF |
| **Checkpointing** | Save every 1000 steps; checkpoint includes slow weights + PM/EM/WM state via `save_state()` |
| **Monitoring** | Log: loss, commit/write rates per block, surprise distribution, PM/EM norms, eligibility norms |

---

## 7. Implementation Order

1. **Set up data pipeline** — streaming FineWeb-Edu + PG19 with packing
2. **Phase A** — verify backbone + WM works (TinyStories)
3. **Phase B** — enable PM with heuristic commits (FineWeb-Edu + SlimPajama, bulk of training)
4. **Phase C** — add EM retrieval + heuristic writes (continue main mix + ProofPile-2)
5. **Phase D** — RL controller training (optional; train `PMController` and `EMController` policies)
6. **Phase E** — lifelong learning evaluation (Wikipedia, synthetic tests, no doc resets)
7. **Phase F** — agentic fine-tuning (function calling datasets)
8. **Phase G** — multimodal (future, requires architecture extension)

---

## 8. Key Decisions to Confirm

- **Tokenizer choice:** Use existing GPT-2/GPT-NeoX tokenizer (fast start) vs. train custom BPE (better fit, more work)?
- **Data mix ratios:** 70/30 FineWeb-Edu/SlimPajama is a starting point — adjust based on Phase B validation curves.
- **PG19 integration:** Interleave with main training or run as separate long-context evaluation passes?
- **EM candidate source:** Use final-layer output `h_final` for EM value candidates (v2 spec default) vs. earlier representations?
- **Agentic format:** Define tool-call token format before Phase F begins.
