# Training Dataset & Configuration Plan — Neuromorphic AI on a Single RTX 4090

## Overview

This plan covers datasets, configuration, and training phases for the current Tier-A wide neuromorphic model (~85M parameters, D=768, L=8, B=2) on a single RTX 4090 (24 GB VRAM). It addresses: basic LM skills, neuromodulated memory (PM + EM), lifelong learning / post-training adaptation, agentic capabilities (start small), and future multimodal expansion.

**Aligned with:** spec v2.0

### v2 Architecture Summary
- **B blocks** (default 2), each containing **L layers** (default 8)
- **Working Memory (WM):** sliding-window attention (W=256), 1 shared instance
- **Procedural Memory (PM):** fast low-rank weights + eligibility, B×L instances (each with `PMNeuromodulator`)
- **Episodic Memory (EM):** per-stream vector store, B instances (each with `EMNeuromodulator`)
- **Plasticity boundaries:** PM/EM writes every P=64 tokens (scan-friendly within spans)

---

## 1. VRAM Budget & Model Scaling

### Scaling Tiers (all fit on RTX 4090)

| Tier | Params | D | L | B | Target | 4090 BS |
|------|--------|---|---|---|--------|---------|
| **A (Wide)** | ~85M | 768 | 8 | 2 | Rapid iteration | 32–64 |
| **B (Competitive)** | ~103M | 768 | 12 | 6 | Match GPT-2 Small | 16–32 |
| **C (Strong)** | ~197M | 1024 | 24 | 8 | Match GPT-2 Medium | 8–16 |

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
| Weight decay | 0.01 (ndim>1 non-bias params only; biases and LayerNorm get 0.0) |
| Batch size (BS) | 16 (can increase for Tier A) |
| TBPTT chunk (T) | 256 tokens |
| Plasticity span (P) | 32 tokens |
| Gradient clipping | 1.0 (global norm) |
| Precision | **bf16** (PM/EM state in fp32) |
| Warmup | 1000 steps |

---

## 2. Training Phases & Datasets

### Phase A — Base Competence (WM + PM)

**Goal:** Verify the backbone + WM+PM learns language with stable PM commits; EM remains disabled.

**Dataset:** TinyStories (small, clean, fast iteration)
- HuggingFace: `roneneldan/TinyStories`
- ~2.1M synthetic short stories, ~470M tokens
- Simple vocabulary, clear narrative structure
- Perfect for verifying loss goes down and the model generates coherent text

**Config:**
- WM enabled (sliding window attention)
- PM enabled (reads + boundary commits with learned continuous heads)
- EM retrieval/writes disabled
- BS=32, ~5–10K steps

**Exit criteria:** Loss decreases smoothly, generated text is coherent English, WM cache stable.

---

### Phase B — Enable PM + EM with Learned Neuromodulators

**Goal:** Train general language competence with both Procedural Memory and Episodic Memory active. Both `PMNeuromodulator` and `EMNeuromodulator` create backbone + continuous heads trained via main loss backprop. PM uses learned continuous commit parameters (`p_commit`, `lambda`, `g`, `slot_logits`, `tau`). EM uses learned write parameters (`g_em`, `tau`, `ww`, `decay`) and `W_nov` learned novelty adjuster.

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

**Long-context memory training:** PG19 (full-length books)
- HuggingFace: `deepmind/pg19`
- ~28K public domain books, avg ~69K tokens each
- Critical for exercising PM and EM across long sequences
- Use as supplementary stream: process full books sequentially to test memory persistence

**Config:**
- BS=16, TBPTT T=256, plasticity span P=64
- PM enabled: reads always, commits at span boundaries with continuous learned parameters
- PM neuromod continuous heads (`p_commit`, `lambda`, `g`, `slot_logits`, `tau`) active and differentiable — trained via main optimizer
- EM retrieval enabled: top k_ret latent memory tokens → cross-attention aggregation
- EM writes at span boundaries with learned parameters:
  - `g_em`: write strength in [g_em_floor, g_em_ceil]
  - `tau`: softmax temperature in [tau_em_floor, tau_em_ceil]
  - `ww`: weakness weight in [ww_em_floor, ww_em_ceil]
  - `decay`: per-stream decay rate in [decay_em_floor, decay_em_ceil]
- `W_nov` learned novelty adjuster active — trained via main optimizer
- All neuromod params on main optimizer (no separate RL optimizer)
- Base decay 0.999/span, eligibility decay ρ=0.95
- All stability rails active (clip, budget, normalize, commit cap)

**Training budget:** ~2–5B tokens total (streaming, no epoch needed)

**Evaluation:**
- Validation perplexity on held-out FineWeb-Edu
- PM ON vs OFF comparison (perplexity gap)
- EM ON vs OFF comparison for long-context tasks
- PG19 long-range memory: fact recall at distance (>1000 tokens back)
- Stable PM and EM budgets across blocks

---

### Phase C — Lifelong Learning (persistent cross-document memory)

**Goal:** Enable persistent cross-document memory. PM/EM accumulate knowledge across document boundaries instead of resetting.

**Prerequisite:** Phase B (learned neuromodulator heads) should be stable. Without trained neuromodulators deciding commit/write strength, persistent memory would accumulate noise.

**Dataset:** Continue FineWeb-Edu + SlimPajama streaming. Add Wikipedia for domain adaptation evaluation.

**Config:**
- `config.lifelong_mode = True` (set by `config.set_phase("C")`)
- `reset_on_doc_boundary` remains True (loss masking still active)
- Soft reset at doc boundaries: h and eligibility traces reset; PM committed state and EM fully persist
- All neuromod params remain on main optimizer (same as Phase B)
- Existing learned decay + budget enforcement handle staleness naturally
- Continue all stability rails

**Training budget:** ~2–3B tokens

**Evaluation:**
- Domain adaptation: stream Wikipedia chunks, measure per-chunk perplexity decrease
- Drift monitoring: perplexity on general text stays within 5% of Phase B baseline
- Cross-document recall: stream factual text, probe at increasing distances
- Commit/write rate statistics (should be sparse and targeted)

---

### Phase E (Future) — Agentic Learning (Start Small)

**Note:** Phase C now implements lifelong learning (previously planned as Phase E). The phase letter "E" is reserved for future agentic capabilities.

---

### Phase F (Future) — Agentic Learning (Expanded)

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

### Phases A–B: Reset at document boundaries
- `reset_on_doc_boundary = True`
- PM (`pm_K/pm_V/pm_a`), EM (`em_K/em_V/em_S`), eligibility (`elig_K/elig_V`), hidden state (`h`), and WM cache cleared when `<|eot|>` is encountered
- Clean learning signal: model learns within-document memory patterns
- No risk of stale memory from unrelated text

### Phase C: Lifelong (soft resets)
- `config.lifelong_mode = True` (set by `config.set_phase("C")`)
- `reset_on_doc_boundary` remains True (loss masking still active)
- **Soft reset** at doc boundaries: h and eligibility traces reset; PM committed state and EM fully persist
- Neuromodulator continuous heads are trained (from Phase B) and selective about commit/write strength
- Learned per-stream decay (EM) + base decay (PM) provide natural forgetting
- Budget caps force overwriting of weak slots when capacity is full
- Runtime state checkpointed alongside model parameters

### Implementation
Two flags control reset behavior:
- `config.reset_on_doc_boundary`: controls loss masking at EOT (True in all phases)
- `config.lifelong_mode`: controls whether PM/EM state persists across doc boundaries (True only in Phase C)

`Block.reset_states()` branches on `lifelong_mode`:
- **lifelong_mode=False (Phases A–B):** Full reset — h, all PM state, and EM strengths zeroed
- **lifelong_mode=True (Phase C):** Soft reset — h zeroed, PM eligibility zeroed, but PM committed state (pm_K/pm_V/pm_a) and EM (em_K/em_V/em_S) persist

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
| **Plasticity span** | P=64 tokens; PM/EM writes at span boundaries enable scan-friendliness within spans |
| **Compile** | `torch.compile(model)` for kernel fusion (RTX 4090 Ada Lovelace supports it well) |
| **Data loading** | `num_workers=4`, `pin_memory=True`, streaming from disk/HF |
| **Checkpointing** | Save every 1000 steps; checkpoint includes slow weights, optimizer, scheduler, runtime state (PM/EM via `save_runtime_state()`), and `last_prev_token` per stream (prevents false doc-boundary resets on resume). Phase transitions detected automatically; optimizer state may be skipped when parameter groups change |
| **Monitoring** | Log: loss, commit/write rates per block, surprise distribution, PM/EM norms, eligibility norms |

---

## 7. Implementation Order

1. **Set up data pipeline** — streaming FineWeb-Edu + PG19 with packing
2. **Phase A** — verify backbone + WM + PM works (TinyStories)
3. **Phase B** — enable PM + EM with learned neuromodulators (FineWeb-Edu + SlimPajama, bulk of training)
4. **Phase C** — lifelong learning (soft resets: h + eligibility traces reset at doc boundaries, PM committed state and EM persist)
5. **Phase E (future)** — agentic fine-tuning (function calling datasets)
6. **Phase F (future)** — multimodal (requires architecture extension)

---

## 8. Key Decisions to Confirm

- **Tokenizer choice:** Use existing GPT-2/GPT-NeoX tokenizer (fast start) vs. train custom BPE (better fit, more work)?
- **Data mix ratios:** 70/30 FineWeb-Edu/SlimPajama is a starting point — adjust based on Phase B validation curves.
- **PG19 integration:** Interleave with main training or run as separate long-context evaluation passes?
- **EM candidate source:** Use final-layer output `h_final` for EM value candidates (v2 spec default) vs. earlier representations?
- **Agentic format:** Define tool-call token format before Phase F begins.
