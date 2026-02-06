# Neuromorphic AI Model Specification

**Version:** 1.7
**Status:** Implementation Ready
**Last Updated:** 2026-02-05

**Changelog:**
- v1.7: **Second review round (9 issues).** (1) Skip loss at cross-document EOT transitions (Phase 0–2): mask out positions where `input_tokens[:, t] == eot_id` in §6.5.5. (2) §5.1 state categories: renamed "BUFFERS (registered, saved)" to "STATE (runtime, optionally checkpointed)" with plain tensor attributes; added `save_state()`/`load_state()` methods to §5.2.5. (3) §5.2.1 table now says "zero K, V, and a" (was "zero a, keep K/V init"). (4) §6.1 diagram: eligibility label updated from "no_grad buf" to "differentiable within TBPTT". (5) §6.4 stale TBPTT loop deleted; replaced with redirect to §6.5.5. (6) Naming standardized: `self.embedding()` everywhere (was `model.embed()`), `forward_one_token` (was `_forward_one_token`), config uses `elig_decay_rho` (was bare `rho`). (7) §5.2.4 design note explaining `torch.where` gradient semantics in per-stream reset. (8) `elig_norm` clamped to [0,1]; threshold description clarified as "saturation upper bound". (9) Per-stream `commit_mask: [BS]` added to `soft_commit()`: non-commit streams skip decay/write/eligibility-reset. Decay, strength update, and eligibility reset all masked per-stream. Small corrections: §8.3 VRAM estimate softened ("monitor; use grad checkpointing if needed").
- v1.6: **Reviewer fixes (8 issues).** (1) Per-timestep doc-boundary resets in TBPTT loop (was per-chunk). (2) Online per-token loss accumulation replaces stacked [BS,T,vocab] logits (VRAM fix). (3) **Differentiable eligibility:** E_K/E_V are now differentiable recurrent state within TBPTT (detached at chunk boundaries), replacing no_grad buffer updates. P/Q/K_proj/V_proj now receive LM loss gradients through: eligibility → commit → fast memory read. ModNet outputs detached at commit (RL controls commit policy). K/V also detached at TBPTT boundaries. (4) KVA reset now zeros K/V in addition to a (prevents stale key/value leakage via EMA blend). (5) Normalized elig_commit_threshold: elig_norm now scaled to [0,1] range relative to steady-state; threshold changed from 2.0 to 0.5. (6) Corrected O(r·D) complexity claim to include GRU O(D_h²) term. (7) Added PyTorch implementation note: states must be plain attributes (not register_buffer) to support tensor reassignment in detach/reset/commit. (8) Per-block RL credit weighting by eligibility norm ratio. Vectorized commit with batch dimensions.
- v1.5: **Batching and training data update.** Added batch dimension (BS) to all buffer shapes (K, V, a, E_K, E_V, h). Introduced phased KVA reset curriculum: Phase 0–2 resets at document boundaries, Phase 3+ lifelong (no resets). Added §6.5 persistent parallel stream batching with per-stream document boundary detection. Added §7.6 Phase 3 (lifelong learning). Updated datasets: FineWeb-Edu + SlimPajama for Phase 1, PG19 + ProofPile-2 for Phase 2. Updated batch size to 16–32 (4090 VRAM analysis). Added training plan document (`docs/training_plan.md`).
- v1.4: **Major architectural update.** Reframed fast memory as dynamic low-rank adapter. Added multi-block width scaling (L layers × B blocks per layer, MHA-style). Replaced overwrite-one-slot commit with soft multi-slot EMA commits. Added teacher forcing vs autoregressive inference distinction. Added deployment modes (write-enabled, read-only). Updated snapshot system to M=L×B with layer+block identity embeddings. Documented eligibility trace design. Clarified Phase 2 as per-token p_commit decision. Rewrote early sections for intuitive narrative flow.
- v1.3: Fixed V=zeros NaN on normalize (init V orthogonal). Added slot selection tie-breaking noise. Added per-token base decay to NML step. Defined IID vs streaming state persistence modes. Normalize query in fast memory addressing (true cosine). Clarified snapshot reads post-GRU h_t^ℓ. Scoped Phase 2 RL to p_commit only. Fixed surprise to use log_softmax. Added bias=False for weight tying. Added elig_magnitude note.
- v1.2: Fixed eligibility K/V mapping (keys from x_t, values from h_t). Fixed GRU param shapes (D×D matrices). Rewrote snapshot system as spatial cross-layer reading (not temporal pyramid). Phase 1 now uses heuristic commits with ModNet logging-only. Removed inhibitory/negative al;rftttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
- v1.1: Added Snapshot Memory System with temporal pyramid compression and decoder cross-attention (superseded by spatial cross-layer design in v1.2).
- v1.0: Initial specification with NML, fast memory, eligibility, and ModNet.

This document is the **single source of truth** for the neuromorphic AI prototype. All implementation decisions should reference this specification.

### Issues Resolved (v1.2)

The following issues from the v1.1 review have been resolved:

| # | Issue | Resolution | Where |
|---|-------|-----------|-------|
| 1 | **Snapshot causality** — temporal pyramid could leak future info | Not applicable: replaced with spatial cross-layer snapshots. Model is recurrent and processes sequentially; snapshots read current-timestep states only. No future info is accessible. | §4.9 |
| 2 | **Eligibility K/V mapping** — keys derived from h_t but retrieval queries with x_t | Swapped: keys from x_t (pre-synaptic), values from h_t (post-synaptic). This matches the Hebbian outer product ΔW = h_t · x_tᵀ in factored form (K = input direction, V = output direction). | §4.5 |
| 3 | **GRU parameter shapes** — spec listed W_u, U_u as R^D but used matrix multiplies | Fixed: W_*, U_* ∈ R^{D_h×D_h}, b_* ∈ R^{D_h}. | §4.4 |
| 4 | **GatedPooling window size** — temporal pyramid pooling had unclear window param | Moot: temporal pyramid replaced by spatial cross-layer snapshot design. | §4.9 |
| 5 | **ModNet never-commit collapse** — penalizing commits during supervised training incentivizes never-commit | Phase 1 uses heuristic commits. ModNet runs forward but is logging-only. | §7.3 |
| 6 | **Inhibitory/negative a** — g was always positive so a could never go negative | Removed for MVP. Strength a ∈ [0, a_max]. | §4.3 |

### Issues Resolved (v1.3)

| # | Issue | Resolution | Where |
|---|-------|-----------|-------|
| 1 | **V=zeros NaN on normalize** — `F.normalize(zeros)` produces NaN | Initialize V to `random_orthogonal(r, D_h)` (like K). With `a=0`, slots are inactive regardless. | §4.3 |
| 2 | **Slot selection tie-breaking** — `argmax` always returns 0 when all slots tie | Replaced with soft multi-slot commits in v1.4 (no argmax needed). | §4.7 |
| 3 | **Base decay not applied per-token** — spec defined it but never called it | Added explicit Step 1 (base decay) to NML block step. | §4.2 |
| 4 | **Lifelong memory during shuffled training** — mixing unrelated docs creates spurious associations | Defined IID vs streaming modes. | §5.2 |
| 5 | **Snapshot content unclear** — didn't specify whether snapshot reads post-GRU or candidate | Snapshot reads post-GRU hidden state h_t^{ℓ,b} per block. | §4.9 |
| 6 | **Query not normalized** — K is unit-normalized but x_t is not | Added `x_q = F.normalize(x_t^b)` in fast_memory_apply. | §4.3 |
| 7 | **ModNet heads richer than RL trains** — Phase 2 RL only trains p_commit | Scoped Phase 2 MVP: only p_commit is RL-trained. λ/g/slot use heuristic defaults. | §7.5 |
| 8 | **Implementation nits** — surprise stability, weight tying bias, elig_magnitude | Fixed surprise to `log_softmax`, `bias=False` for weight tying, documented elig semantics. | §6.2, §4.8 |

### Changes in v1.5

| Change | What | Why |
|--------|------|-----|
| **Batch dimension on state tensors** | All state tensors (K, V, a, E_K, E_V, h) now `[BS, ...]` | BS parallel streams need independent state per stream |
| **Phased KVA reset** | Phase 0–2: reset at doc boundaries; Phase 3+: lifelong | Neuromodulator must be trained before lifelong memory is safe |
| **§6.5 Persistent stream batching** | New section: stream preparation, TBPTT loop with doc boundary detection | Recurrent models need different batching than transformers |
| **§7.6 Phase 3 (lifelong)** | New phase: no resets, evaluate online adaptation | Demonstrates the target deployment capability |
| **Dataset selections** | FineWeb-Edu, SlimPajama, PG19, ProofPile-2 | Concrete datasets sized for 4090 training |
| **Batch size 16–32** | Updated from 4–8 | VRAM analysis shows ~40M model uses <2 GB at BS=8 |

### Changes in v1.4

| Change | What | Why |
|--------|------|-----|
| **Multi-block width** | L layers × B parallel NML blocks per layer, MHA-style routing | Width scaling without inflating per-block parameter count; matches multi-head attention pattern |
| **Dynamic low-rank adapter framing** | Fast memory explicitly described as W_fast(t) = Σ a_i · (V_i ⊗ K_i) | Clearer mental model connecting to existing fast weight / adapter literature |
| **Soft multi-slot commits** | Replace overwrite-one-slot with softmax-weighted top-k EMA updates | Smoother gradients, less disruptive writes, better slot utilization |
| **Teacher forcing / inference** | Explicit distinction between training (ground truth input) and inference (model's own output) | Necessary for correct implementation of both modes |
| **Deployment modes** | Write-enabled (adapts) vs read-only (stable) | Practical deployment requirement |
| **Differentiable eligibility** | E_K/E_V are differentiable recurrent state within TBPTT (detached at chunk boundaries); commit preserves gradients through E→K/V path | P/Q/K_proj/V_proj learn from LM loss; autograd bounded by TBPTT chunk length |
| **Snapshot M = L × B** | One snapshot token per block, with layer + block identity embeddings | Richer spatial information; decoder sees all blocks, not just merged layers |
| **Per-token p_commit** | Phase 2 ModNet outputs p_commit every token (RL trained) | More granular commit control than event-driven-only |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation and Goals](#2-motivation-and-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Specifications](#4-component-specifications)
   - 4.1: Embedding Layer
   - 4.2: NML Layer (Multi-Block)
   - 4.3: Fast Memory (Dynamic Low-Rank Adapter)
   - 4.4: GRU Recurrence
   - 4.5: Eligibility Traces (Differentiable Within TBPTT)
   - 4.6: Neuromodulator (ModNet)
   - 4.7: Soft Multi-Slot Commit
   - 4.8: LM Head
   - 4.9: Snapshot Memory System
   - 4.10: Decoder with Cross-Attention
5. [State Management](#5-state-management)
   - 5.2: Phased KVA Reset Curriculum
   - 5.3: Deployment Modes
   - 5.4: TBPTT State Detachment
6. [Dataflow](#6-dataflow)
   - 6.5: Batching — Persistent Parallel Streams
7. [Training Strategy](#7-training-strategy)
   - 7.6: Phase 3 — Lifelong Learning
8. [Gradient Flow and Optimization](#8-gradient-flow-and-optimization)
9. [Stability Mechanisms](#9-stability-mechanisms)
10. [Hyperparameters](#10-hyperparameters)
11. [Evaluation Strategy](#11-evaluation-strategy)
12. [Implementation Plan](#12-implementation-plan)
13. [Future Directions](#13-future-directions)

---

## 1. Executive Summary

### What We're Building

A **neuromorphic language model** with two timescales of learning, organized around five core mechanisms:

**1. Slow weights** — standard `nn.Parameter` tensors trained by gradient descent. These encode general language competence (vocabulary, grammar, world knowledge) and are frozen at deployment. Analogous to genetics: what you're born knowing.

**2. Fast memory (dynamic low-rank adapter)** — a per-block online-updatable operator:

```
W_fast(t) = Σᵢ aᵢ(t) · (Vᵢ(t) ⊗ Kᵢ(t))
```

This is the model's "life experience." It stores learned associations as key-value slots (K, V) with strength scalars (a). The operator is read on every token (O(r·D_h) per block) but written only when the neuromodulator decides a commit is worthwhile. Crucially, W_fast is never materialized as a D×D matrix — it stays factored as r slots of dimension D_h.

**3. Eligibility traces (Hebbian evidence buffer)** — a running accumulation of "what could be learned" based on input-hidden correlations (Hebbian principle: fire together, wire together). The trace answers: "if I were to commit right now, what would I write?" Traces accumulate continuously but do NOT change memory — they are candidate updates waiting for approval. If no commit occurs, eligibility simply persists and fades via decay ρ (it is not immediately discarded). A commit converts the buffered evidence into actual fast-weight updates, then resets the trace.

**4. Neuromodulators (ModNet)** — a small per-block MLP that decides when and how to commit eligibility traces to fast memory. In Phase 1, commits are heuristic (periodic + threshold). In Phase 2, the commit probability p_commit is trained by RL to optimize future predictions. The model **learns when to learn**.

**5. Spatial snapshots** — a cross-layer, cross-block reading that gives the decoder access to what every NML block currently knows at the current timestep. With L layers × B blocks, the decoder sees M = L×B snapshot tokens, each tagged with layer and block identity embeddings.

### Key Architectural Features

- **Multi-block width scaling:** Each NML layer contains B parallel blocks (MHA-style input routing). Each block has its own GRU recurrence, fast memory, eligibility, and neuromodulator. Blocks operate in D_h = D/B space; an output projection merges them back to D.
- **Soft multi-slot commits:** Instead of hard-overwriting one slot, commits blend into the top-k most relevant slots via softmax-weighted EMA updates. Smoother, less disruptive.
- **Teacher forcing (training) vs autoregressive (inference):** Training inputs are ground truth tokens; inference feeds back model predictions.
- **Deployment modes:** Write-enabled (online adaptation with commits) or read-only (stable inference, no commits).
- **Hardware-friendly:** Fast memory apply is O(r·D_h) per block per token — the *novel* cost. GRU recurrence adds O(D_h²) per block. Total per-layer: O(B·(D_h² + r·D_h)) = O(D²/B + r·D). With r ≪ D_h, fast memory is a small overhead on top of the GRU.

### Constraints

- Single RTX 4090 (24GB VRAM)
- Mixed precision (BF16/FP16)
- TBPTT for memory-efficient training

---

## 2. Motivation and Goals

### 2.1 Problems with Current LLMs

| Problem | Description |
|---------|-------------|
| Static weights | Can't adapt without retraining |
| RAG limitations | Retrieves text but doesn't integrate into computation |
| Quadratic attention | Expensive for long context |
| No lifelong learning | Can't improve from experience |

### 2.2 Our Solution

A two-timescale learning system inspired by biological brains:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TWO TIMESCALES                               │
├─────────────────────────────────┬───────────────────────────────────┤
│         SLOW WEIGHTS            │          FAST MEMORY              │
│    (trained by gradients)       │   (dynamic low-rank adapter)      │
├─────────────────────────────────┼───────────────────────────────────┤
│ Analogy: "Genetics"             │ Analogy: "Life experience"        │
│ When: Training time             │ When: Every token (conditionally) │
│ How: Backprop + optimizer       │ How: Hebbian + neuromodulator     │
│ Encodes: General competence     │ Encodes: Lifelong associations    │
│ Persistence: Saved in weights   │ Persistence: Runtime state        │
│ Mechanism: nn.Parameter         │ Mechanism: W_fast(t) = Σ aᵢVᵢ⊗Kᵢ │
└─────────────────────────────────┴───────────────────────────────────┘
```

### 2.3 MVP Goals

**Functional Requirements:**
1. Next-token language modeling works (perplexity improves, coherent generation)
2. Model supports online adaptation without finetuning
3. Online adaptation is stable (no runaway drift)

**Differentiator:**
4. Neuromodulators learn when/how to write into fast memory to improve future predictions

**Constraints:**
5. Fits and trains on a single RTX 4090 with mixed precision and TBPTT

### 2.4 Success Metrics

| Metric | Target |
|--------|--------|
| Validation perplexity | Decreases during training |
| Memory benchmark (ON vs OFF) | Significant improvement with plasticity ON |
| Personalization accuracy | Higher after seeing user profile |
| Stability (plasticity OFF after long run) | No degradation in base perplexity |
| Commit rate | Sparse and bounded (< 5% of tokens) |

---

## 3. Architecture Overview

### 3.1 High-Level Structure

```
Input: token_ids [batch, T]
           │
           ▼
    ┌──────────────┐
    │  Embedding   │  W_embed ∈ R^{vocab × D}
    └──────────────┘
           │
           ▼ x_t ∈ R^D
    ┌───────────────────────────────────────────────────────────────────┐
    │  NML Layer 1                                                      │
    │  ┌──────────┐                                                     │
    │  │  W_in    │  Project + split: R^D → B × R^{D_h}                │
    │  └──────────┘                                                     │
    │       │                                                           │
    │  ┌────┴────┐  ┌─────────┐       ┌─────────┐                     │
    │  │ Block 1 │  │ Block 2 │  ...  │ Block B │  (each D_h wide)    │
    │  │ GRU+FM  │  │ GRU+FM  │       │ GRU+FM  │  FM = fast memory   │
    │  └────┬────┘  └────┬────┘       └────┬────┘                     │
    │       │             │                 │                           │
    │  ┌────┴─────────────┴────...──────────┘                          │
    │  │  W_o · concat + residual + LayerNorm → R^D                    │
    │  └──────────────────────────────────────────────────────────────┐│
    │       │                                              ↗ snapshot ││
    │       │  h_t^{1,1}, h_t^{1,2}, ..., h_t^{1,B} ─────  tokens   ││
    └───────┼──────────────────────────────────────────────────────────┘
            │ h_t^1 ∈ R^D
            ▼
           ...
            │
    ┌───────────────────────────────────────────────────────────────────┐
    │  NML Layer L                                                      │
    │  (same structure)                                                 │
    │       │  h_t^{L,1}, ..., h_t^{L,B} ──────────────→ snapshot     │
    └───────┼──────────────────────────────────────────────────────────┘
            │ h_t^L ∈ R^D
            ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Snapshot: M = L×B tokens, each [d_s]                           │
    │  token_m = proj(h_t^{ℓ,b}) + layer_emb[ℓ] + block_emb[b]      │
    │  ordering: m = ℓ × B + b                                        │
    └───────────────────────┬─────────────────────────────────────────┘
                            │
            ┌───────────────┘
            ▼ snapshot [M, d_s]
    ┌─────────────────────────────────────────┐
    │          Decoder Cross-Attention         │
    │  ĥ = h_t^L + CrossAttn(h_t^L, snapshot) │
    └─────────────────────────────────────────┘
            │
            ▼ ĥ_t
    ┌──────────────┐
    │   LM Head    │  W_vocab ∈ R^{D × vocab}
    └──────────────┘
            │
            ▼
    logits ∈ R^{vocab}
```

### 3.2 Multi-Block Design (MHA-Style)

Each NML layer contains B parallel NML blocks. This is analogous to multi-head attention:

| MHA Concept | NML Equivalent |
|-------------|---------------|
| Heads | NML blocks |
| Q/K/V projections | W_in (input projection + split) |
| Per-head attention | Per-block GRU + fast memory |
| Output projection | W_o (concat + project) |
| Head dimension d_k | Block dimension D_h = D/B |

**Input routing (two options; MHA-style is MVP default):**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **MHA-style (MVP default)** | `W_in` projects D→D, then reshape to B×D_h | Lower per-block cost, matches MHA convention | Requires learned projection |
| Duplicate-full-D | Each block receives the full D input, projects internally to D_h | Simpler, each block sees everything | Higher per-block input cost |

**MVP default:** `x_blocks = W_in(x_t).reshape(B, D_h)` — project then split.

**Output merge:** `h_out = LayerNorm(x_t + W_o(concat(h_blocks)))` — concat, project, residual, normalize.

Each block is self-contained: its GRU, fast memory, eligibility traces, and ModNet all operate in D_h space. Inter-block information flow happens through W_in (before blocks) and W_o (after blocks).

### 3.3 Token Flow: Training vs Inference

**Training (teacher forcing):**
```
Step t: model receives ground truth token_ids[t] as input
        model predicts token_ids[t+1]
        surprise = -log p(token_ids[t+1] | model output at t)
```

**Inference (autoregressive):**
```
Step 0: model receives <bos> token
Step t: model receives its own predicted token from step t-1
        model predicts next token (argmax or sample)
        surprise = -log p(selected_token | model output at t)
```

### 3.4 Deployment Modes

| Mode | Commits | Eligibility Updates | Use Case |
|------|---------|-------------------|----------|
| **Write-enabled** | Yes | Yes | Online adaptation, continual learning |
| **Read-only** | No | No | Stable inference without adaptation |

In read-only mode, fast memory is still read (y_fast contributes to GRU) but never written. This freezes the model's learned experience.

### 3.5 Decoder Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `simple` | LM head on final h_t^L only, no snapshots | Fallback if training unstable |
| `snapshot` | Cross-attend to M = L×B snapshot tokens | Default mode |

### 3.6 Model Dimensions (Tier A - MVP)

| Parameter | Value | Description |
|-----------|-------|-------------|
| D | 512 | Hidden dimension (full width) |
| L | 8 | Number of NML layers |
| B | 4 | NML blocks per layer (multi-head) |
| D_h | D/B = 128 | Per-block hidden dimension |
| r | 8 | Fast memory slots per block |
| d_p | 64 | Pre-synaptic compression (per block) |
| d_q | 64 | Post-synaptic compression (per block) |
| vocab | ~50k | Vocabulary size (tokenizer dependent) |
| d_s | 256 | Snapshot projection dimension |
| M | L×B = 32 | Snapshot tokens for decoder |

### 3.7 Model Dimensions (Tier B - Stronger Demo)

| Parameter | Value | Description |
|-----------|-------|-------------|
| D | 768 | Hidden dimension |
| L | 10-12 | Number of NML layers |
| B | 4-6 | NML blocks per layer |
| D_h | D/B | Per-block hidden dimension |
| r | 8 | Fast memory slots per block |
| d_p | 64-96 | Pre-synaptic compression (per block) |
| d_q | 64-96 | Post-synaptic compression (per block) |
| d_s | 256-384 | Snapshot projection dimension |
| M | L×B | Snapshot tokens |

---

## 4. Component Specifications

> **Notation convention:** Function signatures in this section show **per-stream shapes** (e.g., `[D_h]`, `[r, D_h]`) for clarity. The actual implementation includes a batch dimension at position 0 (e.g., `[BS, D_h]`, `[BS, r, D_h]`). See §5.1 and §6.5.2 for full batched shapes.

### 4.1 Embedding Layer

**Type:** Standard `nn.Embedding`

```python
embedding = nn.Embedding(vocab_size, D)
```

**Initialization Options:**
1. Random (default PyTorch)
2. Pretrained from GPT-2 (if D=768)

**Output:** `x_t ∈ R^D` for each token

---

### 4.2 NML Layer (Multi-Block)

Each NML layer wraps B parallel NML blocks with shared input/output projections.

#### 4.2.1 Layer Structure

```python
class NMLLayer(nn.Module):
    def __init__(self, D, B, D_h, r, config):
        super().__init__()
        self.B = B
        self.D_h = D_h

        # Input projection: D → D (then reshape to B × D_h)
        self.W_in = nn.Linear(D, D, bias=False)

        # B parallel NML blocks, each operating in D_h space
        self.blocks = nn.ModuleList([
            NMLBlock(D_h, r, config) for _ in range(B)
        ])

        # Output projection: D → D (after concatenating B blocks)
        self.W_o = nn.Linear(D, D, bias=False)

        # LayerNorm for residual connection
        self.layer_norm = nn.LayerNorm(D)

    def forward(self, x_t, surprise):
        """
        Args:
            x_t: [batch, D] - input from previous layer or embedding
            surprise: [batch, 1] - surprise signal (shared across blocks)

        Returns:
            h_out: [batch, D] - layer output (to next layer)
            block_states: List[Tensor] - B tensors each [batch, D_h]
                          (per-block hidden states for snapshot)
        """
        # 1. Project and split into blocks
        x_proj = self.W_in(x_t)                          # [batch, D]
        x_blocks = x_proj.reshape(-1, self.B, self.D_h)  # [batch, B, D_h]

        # 2. Process each block independently
        block_outputs = []
        block_states = []
        for b, block in enumerate(self.blocks):
            h_b = block(x_blocks[:, b, :], surprise)      # [batch, D_h]
            block_outputs.append(h_b)
            block_states.append(h_b)

        # 3. Concatenate and project back
        h_concat = torch.cat(block_outputs, dim=-1)        # [batch, D]
        h_proj = self.W_o(h_concat)                        # [batch, D]

        # 4. Residual connection + LayerNorm
        h_out = self.layer_norm(x_t + h_proj)              # [batch, D]

        return h_out, block_states
```

#### 4.2.2 NML Block (Per-Block, 6 Steps)

Each NML block processes a single token in D_h space:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NML BLOCK (one of B per layer)                          │
│                                                                             │
│  Input: x_t^b ∈ R^{D_h} (from W_in split)                                 │
│  State: h_{t-1}^b, (K^b, V^b, a^b), (E_K^b, E_V^b)                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Base Decay (every token, unconditional)                     │   │
│  │         a^b ← a^b * base_decay                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Fast Memory Apply                                           │   │
│  │         x_q = normalize(x_t^b)                                      │   │
│  │         y_fast = Σᵢ aᵢ^b · Vᵢ^b · (Kᵢ^{b,T} · x_q)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: GRU Recurrence                                              │   │
│  │         h_t^b = GRU(x_t^b, h_{t-1}^b, y_fast)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Eligibility Update (always, differentiable within TBPTT)     │   │
│  │         candidates computed with gradients                          │   │
│  │         E_K^b, E_V^b = ρ*E + candidate (differentiable recurrence)  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 5: Neuromodulator Decision                                     │   │
│  │         p_commit, g, λ, slot_logits = ModNet(x_t^b, h_t^b, ...)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 6: Soft Multi-Slot Commit (conditional on trigger)             │   │
│  │         if commit: EMA-blend top-k slots from eligibility           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  Output: h_t^b ∈ R^{D_h} (to layer merge + snapshot)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Two levels of decay:**
- **Base decay** (Step 1): Applied every token unconditionally. Memory slowly fades even without commits (`base_decay = 0.999`).
- **Commit-time decay** (Step 6): Applied only during commits via λ. Additional decay on top of base decay.

---

### 4.3 Fast Memory (Dynamic Low-Rank Adapter)

**Purpose:** Store and retrieve lifelong associations as a dynamic low-rank operator.

**Core idea:** Fast memory IS a low-rank adapter that evolves at runtime:
```
W_fast(t) = Σᵢ aᵢ(t) · (Vᵢ(t) ⊗ Kᵢ(t))  ∈ R^{D_h × D_h}  (never materialized)
```
Each slot (K_i, V_i, a_i) is one rank-1 component. The operator is applied as:
```
y_fast = W_fast(t) @ normalize(x_t) = Σᵢ aᵢ · Vᵢ · (Kᵢᵀ · normalize(x_t))
```
This is O(r·D_h) per block — no D²  operations.

**IMPORTANT: Fast memory is LIFELONG in Phase 3+ (deployment). During Phases 0–2, KVA is reset at document boundaries to provide clean learning signal while the neuromodulator is being trained (see §5.2).**

#### 4.3.1 State (Per Block)

```python
K ∈ R^{r × D_h}    # Key vectors (normalized to unit length)
V ∈ R^{r × D_h}    # Value vectors (normalized to unit length)
a ∈ R^{r}          # Strength scalars (clipped to [0, a_max])
```

#### 4.3.2 Initialization

```python
# At model creation (once, not per document)
K = random_orthogonal(r, D_h)  # Diverse addressing directions
V = random_orthogonal(r, D_h)  # Random unit vectors (safe to normalize; inactive because a=0)
a = zeros(r)                   # Zero strength (all slots inactive at init)
```

**Why V is not zeros:** `apply_stability_constraints` calls `F.normalize(V)` every step.
Normalizing a zero vector produces NaN. Since `a = 0`, the slot contributes nothing
regardless of V's content, so random orthogonal initialization is safe and avoids the issue.

#### 4.3.3 Apply Operation

```python
def fast_memory_apply(x_t_b: Tensor, K: Tensor, V: Tensor, a: Tensor) -> Tensor:
    """
    Retrieve from fast memory using content-based addressing.

    This is equivalent to: y = W_fast @ normalize(x)
    where W_fast = Σ_i a_i * (V_i ⊗ K_i) — but computed in factored form.

    Args:
        x_t_b: [D_h] - block input vector
        K:     [r, D_h] - key matrix (normalized)
        V:     [r, D_h] - value matrix (normalized)
        a:     [r] - strength vector

    Returns:
        y_fast: [D_h] - fast memory contribution

    Complexity: O(r·D_h)
    """
    x_q = F.normalize(x_t_b, dim=-1)    # [D_h] - unit-normalize query
    scores = K @ x_q                    # [r] - cosine similarity to each key
    weighted = a * scores               # [r] - strength-weighted similarities
    y_fast = weighted @ V               # [D_h] - weighted sum of values
    return y_fast
```

**Design Notes:**
- No softmax: all relevant memories contribute simultaneously
- `a[i]` is non-negative for MVP (inhibitory/negative writes deferred to future work)
- Both K and x_q are unit-normalized, so `scores` are true cosine similarities ∈ [-1, 1]
- W_fast is never materialized — factored computation is strictly O(r·D_h)

---

### 4.4 GRU Recurrence

**Purpose:** Update per-block recurrent hidden state, integrating block input, previous state, and fast memory.

#### 4.4.1 Parameters (Slow Weights, Per Block)

```python
# Per block
W_u, W_r, W_h ∈ R^{D_h × D_h}    # Input weight matrices
U_u, U_r, U_h ∈ R^{D_h × D_h}    # Recurrent weight matrices
b_u, b_r, b_h ∈ R^{D_h}          # Bias vectors
```

#### 4.4.2 Computation

```python
def gru_step(x_t_b: Tensor, h_prev: Tensor, y_fast: Tensor, params: dict) -> Tensor:
    """
    GRU step with fast memory injection (per block).

    Args:
        x_t_b:  [D_h] - block input from W_in split
        h_prev: [D_h] - hidden state from previous timestep (this block)
        y_fast: [D_h] - fast memory contribution
        params: GRU parameters

    Returns:
        h_t: [D_h] - new hidden state
    """
    # Update gate: how much to update vs keep
    u = sigmoid(W_u @ x_t_b + U_u @ h_prev + b_u)

    # Reset gate: how much of old state to use in candidate
    r = sigmoid(W_r @ x_t_b + U_r @ h_prev + b_r)

    # Candidate state with fast memory injection
    h_tilde = tanh(W_h @ x_t_b + U_h @ (r * h_prev) + b_h + y_fast)

    # Interpolate
    h_t = (1 - u) * h_prev + u * h_tilde

    return h_t
```

**Design Note:** `y_fast` is injected inside the tanh, before gating. The update gate `u` can still suppress it (stability feature).

#### 4.4.3 Initialization

```python
# Orthogonal initialization for recurrent weights (helps gradient flow)
nn.init.orthogonal_(U_u)
nn.init.orthogonal_(U_r)
nn.init.orthogonal_(U_h)

# Xavier for input weights
nn.init.xavier_uniform_(W_u)
nn.init.xavier_uniform_(W_r)
nn.init.xavier_uniform_(W_h)

# Zero biases
nn.init.zeros_(b_u)
nn.init.zeros_(b_r)
nn.init.zeros_(b_h)
```

---

### 4.5 Eligibility Traces (Differentiable Within TBPTT)

**Purpose:** Continuously accumulate Hebbian "what could be learned" signals as candidate key-value pairs.

**Differentiability:** E_K and E_V are differentiable recurrent state — the same category as h (GRU hidden state). Within a TBPTT chunk, the eligibility update `E = ρ*E + candidate` builds an autograd chain that allows gradients to flow back through P, Q, K_proj, V_proj via the LM loss whenever a commit writes eligibility into fast memory and that memory is subsequently read. At TBPTT chunk boundaries, E_K and E_V are detached (just like h) to bound the autograd graph length to T tokens.

#### 4.5.1 State (Per Block)

```python
E_K ∈ R^{r × D_h}    # Candidate keys (what inputs were active)
E_V ∈ R^{r × D_h}    # Candidate values (what outputs resulted)
```

#### 4.5.2 Parameters (Slow Weights, Per Block)

```python
P: nn.Linear(D_h, d_p)            # Pre-synaptic projection (input → key space)
Q: nn.Linear(D_h, d_q)            # Post-synaptic projection (hidden → value space)
K_proj: nn.Linear(d_p, r * D_h)   # Expand to candidate keys (from input)
V_proj: nn.Linear(d_q, r * D_h)   # Expand to candidate values (from hidden)
```

#### 4.5.3 Computation (Differentiable)

```python
def eligibility_update(
    x_t_b: Tensor,     # [BS, D_h] - current block input (pre-synaptic)
    h_t_b: Tensor,     # [BS, D_h] - current block hidden (post-synaptic)
    E_K: Tensor,       # [BS, r, D_h] - existing key eligibility (differentiable state)
    E_V: Tensor,       # [BS, r, D_h] - existing value eligibility (differentiable state)
    params: dict,
    rho: float = 0.95  # decay factor
) -> Tuple[Tensor, Tensor]:
    """
    Update eligibility traces with Hebbian candidates.

    Hebbian rule: ΔW = post · preᵀ = h_t · x_tᵀ
    In factored form: K from x_t (pre), V from h_t (post)
    This ensures keys live in input-space (matching retrieval query x_t).

    DIFFERENTIABILITY:
    - Candidates computed with gradients (P, Q, K_proj, V_proj in graph)
    - Trace update is a differentiable recurrence: E_new = ρ*E_old + candidate
    - This builds an autograd chain within the TBPTT chunk (max T=256 steps)
    - At TBPTT boundaries, E_K and E_V are detached (see §5.4)
    - Gradients flow: LM loss → fast memory read → committed K/V → eligibility → P/Q/K_proj/V_proj
    """
    # Compress pre and post signals (in autograd graph)
    p = P(x_t_b)                                  # [BS, d_p] - pre-synaptic
    q = Q(h_t_b)                                  # [BS, d_q] - post-synaptic

    # Expand to slot-shaped candidates (in autograd graph)
    k_candidate = K_proj(p).reshape(-1, r, D_h)   # [BS, r, D_h] - candidate keys (from INPUT)
    v_candidate = V_proj(q).reshape(-1, r, D_h)   # [BS, r, D_h] - candidate values (from HIDDEN)

    # Normalize candidates (in autograd graph)
    k_candidate = F.normalize(k_candidate, dim=-1)
    v_candidate = F.normalize(v_candidate, dim=-1)

    # Differentiable recurrent update (builds autograd chain within TBPTT chunk)
    E_K_new = rho * E_K + k_candidate             # [BS, r, D_h]
    E_V_new = rho * E_V + v_candidate             # [BS, r, D_h]

    return E_K_new, E_V_new
```

**Design Notes:**
- **Keys from x_t, values from h_t** matches the Hebbian outer product ΔW = h_t · x_tᵀ in factored form
- Keys live in input-space, so retrieval query (x_t) and stored keys are in the same representational space
- Bottleneck (d_p, d_q) forces learning of what's worth remembering
- Decay ρ=0.95 → trace "remembers" ~20 recent tokens
- Candidate normalization prevents trace explosion
- **Differentiable recurrence** builds at most T=256 steps of autograd chain per TBPTT chunk, then detached. This is the same cost as the GRU hidden state chain and well within memory budget.
- **Why not no_grad?** With `torch.no_grad()`, P/Q/K_proj/V_proj receive zero gradient from the LM loss — they can only learn through RL (Phase 2). Making eligibility differentiable lets these projections learn *what* to remember during standard LM training (Phase 0-1), while RL trains *when* to commit.

#### 4.5.4 Initialization

```python
E_K = zeros(r, D_h)
E_V = zeros(r, D_h)
```

---

### 4.6 Neuromodulator (ModNet)

**Purpose:** Decide when/how to commit eligibility to fast memory. One ModNet per block.

#### 4.6.1 Architecture

```python
class ModNet(nn.Module):
    def __init__(self, D_h: int, r: int, hidden: int = 64):
        super().__init__()

        # Input: x_t^b (D_h) + h_t^b (D_h) + surprise (1) + elig_norm (1) + mem_usage (1)
        input_dim = D_h + D_h + 1 + 1 + 1

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.commit_head = nn.Linear(hidden, 1)     # p_commit (per-token decision)
        self.decay_head = nn.Linear(hidden, r)       # λ per slot
        self.strength_head = nn.Linear(hidden, 1)    # g (scalar write strength)
        self.slot_head = nn.Linear(hidden, r)        # slot selection logits
```

#### 4.6.2 Inputs

| Input | Shape | Description |
|-------|-------|-------------|
| x_t^b | [D_h] | Current block input |
| h_t^b | [D_h] | Current block hidden state |
| surprise | [1] | -log p(y_t) from previous prediction (shared across blocks) |
| elig_norm | [1] | Normalized eligibility magnitude ∈ [0,1]. Computed as `(||E_K^b|| + ||E_V^b||) / (2 * √r / (1-ρ))` where the denominator is the expected steady-state Frobenius norm. At 0: no accumulated evidence. At 1: steady-state saturation. |
| mem_usage | [1] | Σ|a^b| (current memory utilization for this block) |

#### 4.6.3 Outputs

| Output | Shape | Range | Description |
|--------|-------|-------|-------------|
| p_commit | [1] | (0, 1) | Probability of committing (per-token decision) |
| λ | [r] | (λ_min, 1) | Decay factor per slot |
| g | [1] | (0, g_max) | Scalar write strength (scales all slot updates) |
| slot_logits | [r] | R | Slot selection logits (for soft commit weighting) |

#### 4.6.4 Forward Pass

```python
def forward(
    self,
    x_t_b: Tensor,
    h_t_b: Tensor,
    surprise: Tensor,
    elig_norm: Tensor,
    mem_usage: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # Concatenate inputs
    features = torch.cat([x_t_b, h_t_b, surprise, elig_norm, mem_usage], dim=-1)

    # MLP
    hidden = self.mlp(features)

    # Outputs with bounded activations
    p_commit = torch.sigmoid(self.commit_head(hidden))

    # λ ∈ (λ_min, 1) - never fully forget
    lambda_raw = torch.sigmoid(self.decay_head(hidden))
    λ = LAMBDA_MIN + (1 - LAMBDA_MIN) * lambda_raw

    # g ∈ (0, g_max) - bounded scalar write strength
    g = G_MAX * torch.sigmoid(self.strength_head(hidden))

    slot_logits = self.slot_head(hidden)

    return p_commit, λ, g, slot_logits
```

#### 4.6.5 Constants

```python
LAMBDA_MIN = 0.9    # Minimum decay (prevents catastrophic forgetting)
G_MAX = 1.0         # Maximum write strength
```

---

### 4.7 Soft Multi-Slot Commit

**Purpose:** Write eligibility traces to fast memory via smooth EMA blending across multiple slots.

**Key change from v1.3:** Instead of hard-overwriting one slot (argmax + replace), commits now blend into the top-k most relevant slots using softmax-weighted EMA updates. This is smoother, less disruptive, and avoids the tie-breaking issues of argmax selection.

#### 4.7.1 Algorithm

```python
def soft_commit(
    K: Tensor,           # [BS, r, D_h] - fast memory keys (may have gradients)
    V: Tensor,           # [BS, r, D_h] - fast memory values (may have gradients)
    a: Tensor,           # [BS, r] - fast memory strengths
    E_K: Tensor,         # [BS, r, D_h] - eligibility keys (differentiable)
    E_V: Tensor,         # [BS, r, D_h] - eligibility values (differentiable)
    λ: Tensor,           # [BS, r] - decay factors (from ModNet or heuristic)
    g: Tensor,           # [BS, 1] - scalar write strength
    slot_logits: Tensor, # [BS, r] - slot selection logits
    commit_mask: Tensor, # [BS] - bool: True = commit this stream, False = skip
    config: dict
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Soft multi-slot commit: EMA-blend eligibility into top-k slots.

    Per-stream commit masking: only streams where commit_mask=True are updated.
    Streams where commit_mask=False retain their current K/V/a/E unchanged.

    GRADIENT FLOW:
    - E_K, E_V are NOT detached: gradients flow back to P/Q/K_proj/V_proj
    - ModNet outputs (λ, g, slot_logits) ARE detached: commit policy under RL
    - After commit, K/V carry gradients through eligibility (detached at TBPTT)
    - Strength `a` remains gradient-free (RL controls write strength)

    Returns:
        K, V, a, E_K, E_V: updated states
    """
    # DETACH ModNet outputs: commit policy trained by RL, not backprop
    λ = λ.detach()
    g = g.detach()
    slot_logits = slot_logits.detach()

    # Per-stream masking: zero out g for non-committing streams.
    # This makes alpha=0 for those streams → EMA is a no-op (K/V unchanged).
    cm = commit_mask.float().unsqueeze(-1)   # [BS, 1]
    g = g * cm                               # [BS, 1] — zeroed for non-commit streams

    # 1. DECAY existing memory (commit streams only; non-commit streams keep a unchanged)
    λ_masked = torch.where(commit_mask.unsqueeze(-1), λ, torch.ones_like(λ))
    a = a * λ_masked

    # 2. COMPUTE soft write weights
    # Weakness bias: prefer weaker slots for natural turnover
    weakness_bonus = -torch.abs(a) * config['weakness_weight']
    adjusted_logits = slot_logits + weakness_bonus

    # Soft selection via temperature-scaled softmax
    w = F.softmax(adjusted_logits / config['tau'], dim=-1)   # [BS, r]

    # 3. TOP-K SPARSITY: zero out all but top-k slots
    topk_vals, topk_idx = torch.topk(w, config['commit_top_k'], dim=-1)
    mask = torch.zeros_like(w)
    mask.scatter_(-1, topk_idx, 1.0)
    w = w * mask
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)    # renormalize

    # 4. EMA UPDATE (vectorized, differentiable through E_K/E_V)
    alpha = (w * g).unsqueeze(-1)                    # [BS, r, 1]

    # Eligibility values flow WITHOUT detach → P/Q/K_proj/V_proj get gradients
    E_K_norm = F.normalize(E_K, dim=-1)              # [BS, r, D_h]
    E_V_norm = F.normalize(E_V, dim=-1)              # [BS, r, D_h]

    K = F.normalize((1 - alpha) * K + alpha * E_K_norm, dim=-1)
    V = F.normalize((1 - alpha) * V + alpha * E_V_norm, dim=-1)

    # Strength update (detach elig_magnitude to keep `a` out of autograd graph)
    elig_magnitude = torch.norm(E_V.detach(), dim=-1)   # [BS, r]
    a = a + (w * g) * elig_magnitude

    # 5. SAFETY RAILS

    # Clip strengths (non-negative for MVP)
    a = torch.clamp(a, 0, config['a_max'])

    # Global budget constraint (batch-safe)
    total_strength = torch.sum(torch.abs(a), dim=-1, keepdim=True)  # [BS, 1]
    a = torch.where(
        total_strength > config['budget'],
        a * (config['budget'] / (total_strength + 1e-8)),
        a
    )

    # Ensure normalization
    K = F.normalize(K, dim=-1)
    V = F.normalize(V, dim=-1)

    # 6. RESET eligibility for committing streams only
    cm_rd = commit_mask.float().reshape(-1, 1, 1)  # [BS, 1, 1]
    E_K = E_K * (1.0 - cm_rd)   # zero for commit streams, keep for non-commit
    E_V = E_V * (1.0 - cm_rd)

    return K, V, a, E_K, E_V
```

#### 4.7.2 Design Notes

- **Soft vs hard:** Softmax-weighted EMA blending is differentiable and avoids the discrete argmax selection problem (tie-breaking, slot thrashing)
- **Top-k sparsity:** Only k slots (recommended range k=2–4, default 2) are updated per commit, keeping writes sparse. This is a controllable knob: k=1 approximates the old overwrite-one-slot; larger k spreads the update more broadly
- **EMA semantics:** `K_i ← normalize((1-α_i)·K_i + α_i·normalize(E_K[i]))` smoothly rotates the key direction. With small α_i, this is a gentle update; with α_i close to 1, it's nearly a full overwrite
- **Differentiable eligibility, detached ModNet:** E_K/E_V flow into the commit WITHOUT detach, so gradients propagate back to P/Q/K_proj/V_proj through the LM loss. ModNet outputs (λ, g, slot_logits) ARE detached, keeping the commit *policy* under RL control. This separation means: backprop trains *what* to remember, RL trains *when/where* to commit
- **Vectorized with batch dimension:** The EMA update is written as vectorized operations over `[BS, r, D_h]` tensors, avoiding per-slot Python loops
- **Weakness bias:** Prefer overwriting weak slots for lifelong stability and natural turnover
- **Full eligibility reset:** After any commit, eligibility traces are reset via `torch.zeros_like()` (new tensor, not in-place `.zero_()`) to cleanly break the autograd graph. This avoids double-counting the same Hebbian evidence
- **Budget constraint:** Total memory strength is bounded per stream, using `torch.where` for batch-safe conditional. Prevents runaway growth
- **Elig_magnitude semantics:** `norm(E_V[i])` measures how much consistent signal accumulated in the trace. E_V accumulates normalized candidates but E_V itself is NOT normalized — it's a decayed sum. Large norm = many consistent updates, small norm = few or contradictory. Detached from graph so `a` stays gradient-free
- **Per-stream commit masking:** `commit_mask: [BS]` (bool) controls which streams are committed. For non-commit streams, `g` is zeroed (making alpha=0, so EMA is a no-op), decay is skipped (λ→1), and eligibility is preserved. In Phase 1 heuristic mode, `commit_mask = torch.ones(BS, dtype=torch.bool)` commits all streams simultaneously. In Phase 2+ with per-stream p_commit, `commit_mask = Bernoulli(p_commit).bool()`

---

### 4.8 LM Head

**Type:** Standard linear projection

```python
lm_head = nn.Linear(D, vocab_size, bias=False)  # bias=False when weight tying
```

**Optional (recommended):** Weight tying with embedding matrix

```python
lm_head.weight = embedding.weight  # Shared [vocab, D] parameter
# Note: bias must be False when tying, otherwise the bias would add an unshared
# offset that breaks the symmetry between embedding and unembedding.
```

**Note:** In `snapshot` mode, the LM head receives the decoder output ĥ_t (after cross-attention), not raw h_t^L. See Section 4.10.

---

### 4.9 Snapshot Memory System

**Purpose:** Provide the decoder with a **spatial cross-layer, cross-block reading** of what every NML block currently knows at the current timestep. This gives richer context than just the final hidden state, since different layers and blocks encode different information.

**Key concept:** Snapshots are NOT temporal (they don't summarize past tokens). They are SPATIAL — they read across layers and blocks at a single point in time. The recurrent state (h_t per block) already encodes temporal history.

**Causality:** No causality concern. The model is recurrent (GRU) and processes tokens sequentially. Each block's state h_t^{ℓ,b} at timestep t depends only on tokens ≤ t.

#### 4.9.1 Relationship to Other Mechanisms

| Mechanism | Purpose | What it captures |
|-----------|---------|-----------------|
| Fast Memory (K, V, a) | Online learning/adaptation | Lifelong associations from experience |
| Snapshot Memory | Spatial cross-layer/block reading | What each block currently knows |
| Recurrent State (h) | Temporal context | Sequential history within stream |

#### 4.9.2 Snapshot Projection

M = L × B tokens. One projection per layer (shared across blocks within a layer). Identity encoded via layer + block embeddings.

```python
class SnapshotProjection(nn.Module):
    """
    Spatial cross-layer, cross-block snapshot:
    Project each block's current state to snapshot dimension d_s.
    M = L × B tokens total.
    """

    def __init__(self, D_h: int, d_s: int, L: int, B: int):
        super().__init__()
        self.L = L
        self.B = B

        # One projection per layer (shared across blocks within a layer)
        self.projections = nn.ModuleList([
            nn.Linear(D_h, d_s) for _ in range(L)
        ])

        # Identity embeddings
        self.layer_embeddings = nn.Embedding(L, d_s)
        self.block_embeddings = nn.Embedding(B, d_s)

    def forward(self, all_block_states: List[List[Tensor]]) -> Tensor:
        """
        all_block_states: List of L lists, each containing B tensors of shape [batch, D_h]
                          — the post-GRU hidden state h_t^{ℓ,b} of each block at current timestep.

        Returns: [batch, M, d_s] where M = L × B
        Ordering: token m = ℓ × B + b (layer-major)
        """
        snapshot_tokens = []
        for ell in range(self.L):
            proj = self.projections[ell]
            for b in range(self.B):
                h = all_block_states[ell][b]                        # [batch, D_h]
                s = proj(h)                                          # [batch, d_s]
                s = s + self.layer_embeddings.weight[ell]            # layer identity
                s = s + self.block_embeddings.weight[b]              # block identity
                snapshot_tokens.append(s)

        # Stack: [batch, M, d_s] where M = L × B
        snapshot = torch.stack(snapshot_tokens, dim=1)
        return snapshot
```

**Shape:** `[batch, M, d_s]` where M = L × B = 32 (Tier A).

#### 4.9.3 Future: Pyramid Mode (Spatial Aggregation)

When L × B is large, pyramid mode can aggregate across layers/blocks to reduce M before decoder cross-attention. For MVP, dense mode (M = L × B = 32) is sufficient since cross-attention cost O(M·d_s) is negligible.

#### 4.9.4 Fallback: Simple Mode

If snapshot training is unstable, disable cross-attention entirely:

```python
# Simple mode: LM head reads only h_t^L (final layer merged output)
# No snapshot projection, no cross-attention
# decoder_mode = 'simple'
```

---

### 4.10 Decoder with Cross-Attention

**Purpose:** Turn the spatial snapshot [M, d_s] plus the final hidden state h_t^L into the next token prediction.

#### 4.10.1 Architecture

```python
class SnapshotDecoder(nn.Module):
    """
    Cross-attention from final hidden state to spatial snapshot tokens.
    Turns (h_t^L, snapshot[M, d_s]) → enhanced hidden state for LM head.
    """

    def __init__(
        self,
        D: int,             # Full hidden dimension
        d_s: int,           # Snapshot dimension
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project query from D to d_s
        self.query_proj = nn.Linear(D, d_s)

        # Cross-attention: query=h_final, key/value=snapshot
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_s,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Project back to D
        self.output_proj = nn.Linear(d_s, D)

        # Layer norm for residual
        self.norm = nn.LayerNorm(D)

    def forward(self, h_final: Tensor, snapshot: Tensor) -> Tensor:
        """
        h_final:  [batch, D] - final layer output at current timestep
        snapshot: [batch, M, d_s] - spatial snapshot (M = L × B tokens)

        Returns: [batch, D] - enhanced hidden state
        """
        # Project final hidden to query space
        q = self.query_proj(h_final).unsqueeze(1)   # [batch, 1, d_s]

        # Cross-attend to snapshot (M spatial tokens)
        context, _ = self.cross_attn(q, snapshot, snapshot)  # [batch, 1, d_s]

        # Project back and residual
        context = self.output_proj(context.squeeze(1))       # [batch, D]

        # Residual + norm
        h_out = self.norm(h_final + context)                 # [batch, D]

        return h_out
```

#### 4.10.2 Design Notes

- **M = L × B = 32** for Tier A — still small for cross-attention, O(M·d_s) per token
- **Layer + block embeddings** help the decoder learn which blocks carry which kinds of information
- **Single query** (h_final) cross-attending to M keys — this is O(M·d_s), negligible compute
- No sequence-level attention — the decoder is a "machine that turns (M, D) → next token"

---

## 5. State Management

### 5.1 State Categories

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL STATE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PARAMETERS (nn.Parameter) - Updated by optimizer                           │
│  ├── embedding.weight              [vocab, D]                               │
│  ├── Per layer:                                                             │
│  │   ├── W_in.weight               [D, D]                                   │
│  │   ├── W_o.weight                [D, D]                                   │
│  │   ├── layer_norm.*              [D]                                      │
│  │   └── Per block (B per layer):                                           │
│  │       ├── gru.W_*, U_*, b_*     [D_h, D_h], [D_h, D_h], [D_h]          │
│  │       ├── P.weight, bias        [d_p, D_h], [d_p]                       │
│  │       ├── Q.weight, bias        [d_q, D_h], [d_q]                       │
│  │       ├── K_proj.weight, bias   [r*D_h, d_p], [r*D_h]                   │
│  │       ├── V_proj.weight, bias   [r*D_h, d_q], [r*D_h]                   │
│  │       └── modnet.*              per-block ModNet params                  │
│  ├── Snapshot system (if decoder_mode='snapshot'):                          │
│  │   ├── snapshot_proj[ℓ]          [d_s, D_h] × L (one per layer)          │
│  │   ├── layer_embeddings          [L, d_s]                                │
│  │   ├── block_embeddings          [B, d_s]                                │
│  │   └── snapshot_decoder.*        cross-attention + projections            │
│  └── lm_head.weight                [vocab, D] (bias=False when tying)      │
│                                                                             │
│  STATE (runtime, optionally checkpointed) — plain tensor attributes         │
│  Stored as self.K = torch.zeros(...), NOT register_buffer (see §5.4).       │
│  Moved via custom init_state(device, BS). Saved via save_state()/           │
│  load_state() only when needed (Phase 3+). See §5.2.5.                     │
│  └── Per block (L × B total):                                              │
│      ├── K                         [BS, r, D_h]  (differentiable state)    │
│      ├── V                         [BS, r, D_h]  (differentiable state)    │
│      ├── a                         [BS, r]        (gradient-free)           │
│      ├── E_K                       [BS, r, D_h]  (differentiable state)    │
│      ├── E_V                       [BS, r, D_h]  (differentiable state)    │
│      └── h                         [BS, D_h]     (differentiable state)    │
│                                                                             │
│  NOTE: BS = batch size (number of parallel streams). Each stream            │
│  maintains fully independent state. All state operations are per-stream.    │
│                                                                             │
│  RUNTIME STATE (not saved, computed per step)                               │
│  ├── surprise                      [BS, 1] (from previous prediction)      │
│  └── snapshot                      [BS, M, d_s] (spatial, per step)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 State Persistence — Phased KVA Reset Curriculum

The model's KVA (fast memory) reset policy evolves across training phases. This is a **curriculum**: start with clean per-document isolation, then progressively enable lifelong persistence as the neuromodulator learns to manage memory selectively.

**Why not lifelong from day one?** Before the neuromodulator is trained (Phases 0–1), it cannot selectively commit useful information. Carrying fast memory across unrelated documents would accumulate garbage in KVA, injecting noise that corrupts learning. Resetting at document boundaries gives a clean learning signal. Once the neuromodulator is trained via RL (Phase 2), it is selective enough to manage persistent memory without corruption.

#### 5.2.1 Phase 0–1: Reset at Document Boundaries

| State | Reset at Doc Boundary | Rationale |
|-------|----------------------|-----------|
| K, V, a (fast memory) | **YES** — zero K, V, and a | Clean slate per document; no stale memory from unrelated text (zeroing K/V prevents stale key/value leakage via EMA blend at next commit) |
| E_K, E_V (eligibility) | **YES** — hard reset to zero | Eligibility is short-term; meaningless across documents |
| h (recurrent) | **YES** — reset to zero | No temporal context from previous document |
| surprise | **YES** — reset to zero | Previous document's surprise is irrelevant |

#### 5.2.2 Phase 2: Reset + RL Training

Same reset policy as Phase 0–1. The neuromodulator is now being trained by RL to make good commit decisions, but training still uses document-boundary resets for clean counterfactual signals. The RL rollouts compare commit-vs-no-commit within a single document, so isolation helps credit assignment.

#### 5.2.3 Phase 3+: Lifelong (No Resets)

| State | Reset at Doc Boundary | Rationale |
|-------|----------------------|-----------|
| K, V, a (fast memory) | **NO** — persists across documents | Lifelong learning: memory evolves through experience |
| E_K, E_V (eligibility) | **NO** — persists (fades via ρ decay) | Allows cross-document Hebbian accumulation |
| h (recurrent) | **NO** — persists | Continuous temporal context |
| surprise | Per token (always) | Always recomputed |

This is the end-goal mode. The trained neuromodulator is selective enough to manage memory lifetime. Base decay (0.999/token) provides natural forgetting — a slot loses ~50% strength after ~700 tokens without reinforcement. The budget cap forces overwriting of weak slots when capacity is full.

#### 5.2.4 Implementation: Batch-Aware Reset

Since the model processes BS parallel streams, document boundaries occur at different positions in each stream. Resets are applied **per-stream** using a boolean mask:

```python
def reset_at_doc_boundary(self, reset_mask: Tensor):
    """
    Reset state for streams that hit a document boundary.

    Args:
        reset_mask: [BS] boolean tensor. True = this stream hit a doc boundary.
    """
    if not self.config.reset_kva_on_doc_boundary:
        return  # Phase 3+: lifelong, no resets

    # reset_mask shape: [BS] → expand for broadcasting
    mask_r = reset_mask.unsqueeze(-1)          # [BS, 1]
    mask_rd = reset_mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]

    for layer in self.layers:
        for block in layer.blocks:
            # Full fast memory reset: K, V, and a
            # Why also K/V? With only a=0, K/V retain stale values.
            # At the next commit, EMA blend (1-α)*K_old + α*E_K_new
            # mixes stale keys from the old document into the new memory.
            block.K = torch.where(mask_rd.expand_as(block.K),
                                  torch.zeros_like(block.K), block.K)
            block.V = torch.where(mask_rd.expand_as(block.V),
                                  torch.zeros_like(block.V), block.V)
            block.a = torch.where(mask_r.expand_as(block.a),
                                  torch.zeros_like(block.a), block.a)

            # Hard reset eligibility
            block.E_K = torch.where(mask_rd.expand_as(block.E_K),
                                    torch.zeros_like(block.E_K), block.E_K)
            block.E_V = torch.where(mask_rd.expand_as(block.E_V),
                                    torch.zeros_like(block.E_V), block.E_V)

            # Reset recurrent state
            block.h = torch.where(mask_r.unsqueeze(-1).expand_as(block.h),
                                  torch.zeros_like(block.h), block.h)

    # Reset surprise for affected streams
    self.surprise = torch.where(mask_r.expand_as(self.surprise),
                                torch.zeros_like(self.surprise), self.surprise)
```

**Configuration flag:**
```python
reset_kva_on_doc_boundary: bool = True   # Phase 0–2: True. Phase 3+: False.
```

**Gradient behavior of per-stream reset:** `torch.where(mask, zeros_like(x), x)` provides correct per-stream gradient semantics. For reset streams (mask=True): they receive `zeros_like(x)` — a fresh tensor with no gradient history, cleanly cutting the computation graph. For non-reset streams (mask=False): they retain their original tensor `x` with its full gradient chain intact (mid-document, gradient flow must continue). This is the correct behavior — using `torch.no_grad()` on the entire reset operation would incorrectly kill gradients for non-reset streams too.

#### 5.2.5 Checkpointing

| Phase | What to Save |
|-------|-------------|
| Phase 0–1 | Parameters only (KVA resets per document, no value in saving it) |
| Phase 2 | Parameters + ModNet optimizer state |
| Phase 3+ | Parameters + **full KVA state** (lifelong memory is the model's "experience") |

**Saving state tensors:** Since K/V/a/E_K/E_V/h are plain attributes (not in `state_dict()`), use custom methods:

```python
def save_state(self) -> dict:
    """Collect runtime state for checkpointing (Phase 3+)."""
    state = {}
    for l, layer in enumerate(self.layers):
        for b, block in enumerate(layer.blocks):
            prefix = f'layer{l}_block{b}'
            state[f'{prefix}/K'] = block.K.detach().cpu()
            state[f'{prefix}/V'] = block.V.detach().cpu()
            state[f'{prefix}/a'] = block.a.detach().cpu()
            state[f'{prefix}/E_K'] = block.E_K.detach().cpu()
            state[f'{prefix}/E_V'] = block.E_V.detach().cpu()
            state[f'{prefix}/h'] = block.h.detach().cpu()
    return state

def load_state(self, state: dict, device: torch.device):
    """Restore runtime state from checkpoint."""
    for l, layer in enumerate(self.layers):
        for b, block in enumerate(layer.blocks):
            prefix = f'layer{l}_block{b}'
            block.K = state[f'{prefix}/K'].to(device)
            block.V = state[f'{prefix}/V'].to(device)
            block.a = state[f'{prefix}/a'].to(device)
            block.E_K = state[f'{prefix}/E_K'].to(device)
            block.E_V = state[f'{prefix}/E_V'].to(device)
            block.h = state[f'{prefix}/h'].to(device)
```

**Full checkpoint save/load:**
```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),       # nn.Parameters only
    'runtime_state': model.save_state(),           # K/V/a/E/h
    'optimizer_state_dict': optimizer.state_dict(),
}, path)

# Load
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
if 'runtime_state' in checkpoint:
    model.load_state(checkpoint['runtime_state'], device)
```

### 5.3 Deployment Modes

```python
def set_deployment_mode(self, mode: str):
    """
    Set deployment mode.

    Args:
        mode: 'write_enabled' or 'read_only'
    """
    self.deployment_mode = mode
    for layer in self.layers:
        for block in layer.blocks:
            block.commits_enabled = (mode == 'write_enabled')
            block.eligibility_enabled = (mode == 'write_enabled')
```

In **read-only** mode:
- Fast memory apply still runs (y_fast contributes to GRU)
- Eligibility update is skipped (no trace accumulation)
- Commit is skipped (no memory writes)
- GRU recurrence still runs (temporal context still evolves)

### 5.4 TBPTT State Detachment

During training, states must be detached at chunk boundaries to limit backprop length. All states have a batch dimension (BS) — detachment applies to all streams simultaneously:

```python
def detach_states(self):
    """Call after each TBPTT chunk. Operates on all BS streams.

    All recurrent/persistent state must be detached to bound autograd chain length.
    After a commit, K and V carry gradients from eligibility — must be detached here.
    """
    for layer in self.layers:
        for block in layer.blocks:
            block.h = block.h.detach()      # [BS, D_h] — GRU hidden
            block.E_K = block.E_K.detach()  # [BS, r, D_h] — eligibility keys
            block.E_V = block.E_V.detach()  # [BS, r, D_h] — eligibility values
            block.K = block.K.detach()      # [BS, r, D_h] — fast memory keys
            block.V = block.V.detach()      # [BS, r, D_h] — fast memory values
            # a: strength scalars, gradient-free (ModNet outputs detached at commit)
```

**PyTorch implementation note:** h, E_K, E_V, K, V, and a must be stored as **plain tensor attributes** (`self.h = torch.zeros(...)`) — NOT as registered buffers (`register_buffer`). Reason: `detach_states()`, `reset_at_doc_boundary()`, and `soft_commit()` all reassign these attributes (e.g., `block.h = block.h.detach()`), which creates a new tensor and rebinds the Python attribute. With `register_buffer`, the old tensor remains in the module's `_buffers` dict while the attribute points to the new tensor — breaking `state_dict()` and device tracking. Use custom `save_state()`/`load_state()` methods for checkpointing these tensors instead.

---

## 6. Dataflow

### 6.1 Single Token, Single Block

```
x_t^b ──────────────────────┬──────────────────────────────────────┐
                             │                                      │
                             │  (first: a^b ← a^b * base_decay)    │
                             ▼                                      │
               ┌─────────────────────────┐                         │
               │   Fast Memory Apply     │                         │
               │   x_q = normalize(x^b)  │                         │
               │   y = Σ aᵢVᵢ(Kᵢᵀx_q)   │                         │
               └───────────┬─────────────┘                         │
                           │                                        │
                           ▼ y_fast                                │
               ┌─────────────────────────┐                         │
               │   GRU Recurrence        │ ◄─── h_{t-1}^b          │
               │   h = GRU(x^b, h, y)   │                         │
               └───────────┬─────────────┘                         │
                           │                                        │
                           ▼ h_t^b                                 │
          ┌────────────────┼────────────────────┐                  │
          │                │                    │                  │
          ▼                ▼                    ▼                  │
┌─────────────────┐ ┌───────────┐    ┌─────────────────┐         │
│  Eligibility    │ │  ModNet   │    │  Output to      │         │
│  Update (diff.  │ │           │    │  Layer Merge    │         │
│  within TBPTT)  │ │  Inputs:  │    │  + Snapshot     │         │
└────────┬────────┘ │  x_t^b ───┼────┼─────────────────┼─────────┘
         │          │  h_t^b    │    │                 │
         │          │  surprise │    │                 │
         │          │  norms    │    │                 │
         │          └─────┬─────┘    └────────┬────────┘
         │                │                   │
         │                ▼                   │
         │    ┌─────────────────────┐         │
         │    │  Commit Decision    │         │
         │    │  p_commit, λ, g,   │         │
         │    │  slot_logits       │         │
         │    └─────────┬───────────┘         │
         │              │                     │
         │              ▼ (if commit)         │
         │    ┌─────────────────────┐         │
         └───►│  Soft Multi-Slot   │         │
              │  Commit (EMA)      │         │
              │  top-k slots       │         │
              └─────────────────────┘         │
                                              │
                                              ▼
                                           h_t^b (to layer merge + snapshot)
```

### 6.2 Single Token Forward (Core Building Block)

```python
def forward_one_token(self, x_t: Tensor) -> Tensor:
    """
    Process one token through all layers.

    Args:
        x_t: [BS, D] - embedded token

    Returns:
        logits_t: [BS, vocab] - prediction logits for this step
    """
    # Process through layers, collecting per-block states for snapshot
    all_block_states = []  # L lists of B tensors each [BS, D_h]
    for layer in self.layers:
        x_t, block_states = layer(x_t, self.surprise)
        all_block_states.append(block_states)

    h_final = x_t  # [BS, D] — merged output of final layer

    # Snapshot decoder (if enabled)
    if self.decoder_mode == 'snapshot':
        snapshot = self.snapshot_proj(all_block_states)              # [BS, M, d_s]
        h_final = self.snapshot_decoder(h_final, snapshot)          # [BS, D]

    # LM head
    logits_t = self.lm_head(h_final)  # [BS, vocab]
    return logits_t
```

**IMPORTANT — online loss, not stacked logits:** The model processes one token at a time and returns `[BS, vocab]` logits. The training loop (§6.5.5) computes loss per timestep and accumulates the scalar. Never stack logits into `[BS, T, vocab]` — with vocab=50K, BS=32, T=256, that would be ~820 MB in fp16.
```

### 6.3 Inference (Autoregressive)

```python
@torch.no_grad()
def generate(self, prompt_ids: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
    """
    Autoregressive generation.

    Args:
        prompt_ids: [1, T_prompt] - prompt token ids
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature

    Returns:
        generated_ids: [1, T_prompt + max_new_tokens]
    """
    generated = list(prompt_ids[0].tolist())

    # Process prompt (teacher forcing on known tokens)
    for t in range(prompt_ids.size(1)):
        x_t = self.embedding(prompt_ids[:, t])        # [1, D]
        logits_t = self.forward_one_token(x_t)        # [1, vocab]

        # Surprise from ground truth prompt token
        if t < prompt_ids.size(1) - 1:
            log_probs = F.log_softmax(logits_t, dim=-1)
            true_next = prompt_ids[:, t + 1]
            self.surprise = -log_probs.gather(-1, true_next.unsqueeze(-1))

    # Generate new tokens (autoregressive)
    for _ in range(max_new_tokens):
        # Sample or argmax
        probs = F.softmax(logits_t / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)           # [1, 1]
        generated.append(next_token.item())

        # Surprise from model's OWN predicted token
        log_probs = F.log_softmax(logits_t, dim=-1)
        self.surprise = -log_probs.gather(-1, next_token)  # [1, 1]

        # Feed back predicted token
        x_t = self.embedding(next_token.squeeze(1))         # [1, D]
        logits_t = self.forward_one_token(x_t)              # [1, vocab]

    return torch.tensor([generated])
```

### 6.4 TBPTT Training Loop

See §6.5.5 for the authoritative TBPTT training loop implementation. It uses per-timestep processing (not batch-forward) to support:
- Per-timestep document boundary resets
- Online loss accumulation (avoids materializing `[BS, T, vocab]`)
- Cross-document loss masking at EOT positions

### 6.5 Batching: Persistent Parallel Streams

The model is recurrent — hidden state, fast memory, and eligibility must persist across TBPTT chunks. This requires a different batching strategy than transformers (which sample independent sequences).

#### 6.5.1 Core Design: BS Parallel Streams

Training data is organized as BS independent, persistent token streams. Each stream is a continuous flow of concatenated documents separated by `<|endoftext|>` tokens:

```
Stream 0: [doc_1 tokens...] <|eot|> [doc_2 tokens...] <|eot|> [doc_3 ...] ...
Stream 1: [doc_4 tokens...] <|eot|> [doc_5 tokens...] <|eot|> [doc_6 ...] ...
  ...
Stream BS-1: [doc_N tokens...] <|eot|> [doc_N+1 ...] ...
```

Each training step consumes one TBPTT chunk (T=256 tokens) from each stream:

```
Step 0:  streams[:, 0:256]      → batch shape [BS, 256]
Step 1:  streams[:, 256:512]    → batch shape [BS, 256]
Step 2:  streams[:, 512:768]    → batch shape [BS, 256]
...
```

#### 6.5.2 State Isolation

All per-block state tensors have a batch dimension at position 0. Operations never mix across the batch dimension:

```
h:   [BS, D_h]       — GRU hidden, independent per stream
K:   [BS, r, D_h]    — Fast memory keys, independent per stream
V:   [BS, r, D_h]    — Fast memory values, independent per stream
a:   [BS, r]          — Fast memory strengths, independent per stream
E_K: [BS, r, D_h]    — Eligibility keys, independent per stream
E_V: [BS, r, D_h]    — Eligibility values, independent per stream
```

PyTorch handles this naturally — all operations (GRU update, fast memory apply, eligibility update, commit) broadcast correctly over the batch dimension.

#### 6.5.3 Document Boundary Handling

When a `<|endoftext|>` token appears at position t in stream i, the data loader marks that stream for potential reset. The model applies `reset_at_doc_boundary(reset_mask)` (see §5.2.4) which, depending on the training phase:
- **Phase 0–2:** Zeros out KVA, eligibility, and hidden state for stream i only (other streams unaffected)
- **Phase 3+:** Does nothing (lifelong persistence)

Document boundaries occur at different positions in different streams, so the reset mask varies per step.

#### 6.5.4 Preparing Stream Data (Pre-tokenized)

For local datasets, the efficient approach is:

```python
def prepare_streams(token_array: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Reshape flat token array into BS parallel streams.

    Args:
        token_array: [N] flat array of all tokens (docs separated by <|eot|>)
        batch_size: BS number of parallel streams

    Returns:
        streams: [BS, N // BS] — each row is one persistent stream
    """
    # Trim to exact multiple of batch_size
    usable = (len(token_array) // batch_size) * batch_size
    return token_array[:usable].reshape(batch_size, -1)
```

For streaming (HuggingFace), maintain BS independent iterators that each fill a T-length buffer by tokenizing on the fly.

#### 6.5.5 TBPTT Loop with Per-Timestep Reset and Online Loss

**Three critical implementation details:**

1. **Per-timestep resets, not per-chunk.** If `<|eot|>` appears at position t in stream i, reset stream i's state **before processing token t+1** (not before the whole chunk). Resetting at chunk granularity would wipe state for tokens still belonging to the previous document.

2. **Online loss accumulation.** Never materialize `[BS, T, vocab]` logits — with vocab=50K, BS=32, T=256, that's ~820 MB in fp16. Instead, compute loss per timestep and accumulate the scalar. The computation graph still supports TBPTT backprop.

3. **Skip loss at cross-document transitions (Phase 0–2).** When `input_tokens[:, t] == eot_token_id`, the model predicts the first token of the *next* document conditioned on the *previous* document's state. This is a cross-document transition and violates doc isolation. Skip loss for these positions. (Predicting `<|eot|>` as a *target* within a document is still trained — only the EOT→next-doc-start transition is skipped.)

```python
def train_tbptt(model, streams, chunk_size=256, eot_token_id=50256):
    """
    TBPTT training with per-timestep doc-boundary resets and online loss.

    Args:
        streams: [BS, total_tokens] pre-tokenized parallel streams
        chunk_size: T = TBPTT chunk length
        eot_token_id: end-of-text token ID for document boundary detection
    """
    BS, total_tokens = streams.shape
    num_chunks = (total_tokens - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size

        tokens = streams[:, start:end+1]           # [BS, T+1] (input + target)
        input_tokens = tokens[:, :-1]               # [BS, T]
        target_tokens = tokens[:, 1:]               # [BS, T]

        # Process token-by-token within the chunk for correct reset timing
        chunk_loss = torch.tensor(0.0, device=tokens.device)
        valid_tokens = 0

        for t in range(chunk_size):
            # Per-timestep doc boundary reset:
            # If the PREVIOUS token was <|eot|>, reset before processing this token.
            # At t=0, check the last token of the previous chunk.
            if t == 0 and chunk_idx > 0:
                prev_token = streams[:, start - 1]                # [BS]
                reset_mask = (prev_token == eot_token_id)         # [BS]
            else:
                reset_mask = (input_tokens[:, t - 1] == eot_token_id) if t > 0 \
                             else torch.zeros(BS, dtype=torch.bool, device=tokens.device)

            if reset_mask.any():
                model.reset_at_doc_boundary(reset_mask)

            # Forward one token
            x_t = model.embedding(input_tokens[:, t])             # [BS, D]
            logits_t = model.forward_one_token(x_t)               # [BS, vocab]

            # Compute loss — skip cross-document transitions (Phase 0–2)
            # When input is <|eot|>, the model predicts next-doc-start from prev-doc state.
            # This violates doc isolation, so mask those positions out of the loss.
            target_t = target_tokens[:, t]                        # [BS]
            is_eot_input = (input_tokens[:, t] == eot_token_id)   # [BS]
            loss_mask = ~is_eot_input                             # [BS]
            if loss_mask.any():
                token_loss = F.cross_entropy(logits_t[loss_mask], target_t[loss_mask],
                                             reduction='sum')
                chunk_loss = chunk_loss + token_loss
                valid_tokens += loss_mask.sum().item()

            # Update surprise for next step (teacher forcing: use ground truth)
            with torch.no_grad():
                log_probs = F.log_softmax(logits_t, dim=-1)
                model.surprise = -log_probs.gather(-1, target_t.unsqueeze(-1))

        # Average loss over the chunk
        if valid_tokens == 0:
            continue  # entire chunk was EOT transitions (extremely rare)
        chunk_loss = chunk_loss / valid_tokens
        chunk_loss = chunk_loss + compute_regularizers(model)

        optimizer.zero_grad()
        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Detach states for next chunk (h is detached; E_K/E_V are detached)
        model.detach_states()
```

**Note on VRAM:** By computing `logits_t` one step at a time (`[BS, vocab]`), peak logits memory is ~BS×vocab×2 bytes ≈ 3.2 MB (BS=32, vocab=50K, fp16) instead of ~820 MB for the full `[BS, T, vocab]` tensor. The computation graph for TBPTT backprop through the chunk is still maintained via the recurrent state `h`.

---

## 7. Training Strategy

### 7.1 Training Phases Overview

```
TIME ──────────────────────────────────────────────────────────────────────►

PHASE 0: Baseline (Plasticity OFF) — OPTIONAL
┌──────────────────────────────────────┐
│  Fast memory: disabled               │  Goal: Verify GRU + LM head works
│  ModNet: disabled                    │  Dataset: TinyStories
│  KVA reset: N/A (no fast memory)     │  Duration: Until perplexity stable
│  Training: Standard LM               │
└──────────────────────────────────────┘
                │
                ▼
PHASE 1: Supervised Plasticity (Heuristic Commits)
┌──────────────────────────────────────┐
│  Fast memory: enabled                │  Goal: Train slow weights with plasticity
│  Commits: HEURISTIC timing           │  Dataset: FineWeb-Edu + SlimPajama
│  ModNet: logging-only                │  Duration: Until memory bench improves
│  KVA reset: YES at doc boundaries    │
│  Training: LM + regularizers         │
└──────────────────────────────────────┘
                │
                ▼
PHASE 1.5: Advantage Warmup (Recommended)
┌──────────────────────────────────────┐
│  Compute counterfactuals             │  Goal: Reduce RL variance
│  Train advantage predictor           │  Duration: Brief (~10% of Phase 1)
│  KVA reset: YES at doc boundaries    │
└──────────────────────────────────────┘
                │
                ▼
PHASE 2: RL Training (Per-Token p_commit)
┌──────────────────────────────────────┐
│  Fast memory: enabled                │  Goal: Learn optimal commit policy
│  ModNet: per-token p_commit          │  Slow weights: Frozen initially
│  KVA reset: YES at doc boundaries    │  Dataset: FineWeb-Edu + PG19 + ProofPile
│  Training: Event-driven RL           │  Duration: Until commit policy improves
└──────────────────────────────────────┘
                │
                ▼
PHASE 3: Lifelong Learning (No Resets)
┌──────────────────────────────────────┐
│  Fast memory: enabled, lifelong      │  Goal: Demonstrate online adaptation
│  ModNet: trained (from Phase 2)      │  Slow weights: Frozen
│  KVA reset: NO (lifelong)            │  Evaluation: Wikipedia, personalization
│  config.reset_kva_on_doc_boundary    │
│    = False                           │
└──────────────────────────────────────┘
```

### 7.2 Phase 0: Baseline (Optional)

**Objective:** Verify the base model (without plasticity) can learn language. Can be skipped if starting from a working checkpoint.

**Dataset:** TinyStories (`roneneldan/TinyStories`, ~470M tokens)

**Configuration:**
```python
model.set_plasticity(enabled=False)
model.config.reset_kva_on_doc_boundary = True  # N/A since plasticity is off
# Standard LM training, BS=32, ~5–10K steps
```

**Success Criteria:**
- Validation perplexity decreases
- Generation is coherent

### 7.3 Phase 1: Supervised Plasticity (Heuristic Commits)

**Objective:** Train slow weights to work well with plasticity enabled. Commits are controlled by a simple heuristic — ModNet runs forward but its outputs are **logged, not used for commit decisions**.

**Dataset:** FineWeb-Edu 10BT (primary, 70%) + SlimPajama-627B (secondary, 30% via streaming). Supplementary: PG19 (long books for memory stress-testing). Training budget: ~2–5B tokens.

**KVA Reset Policy:** `reset_kva_on_doc_boundary = True`. Fast memory resets at document boundaries. This gives the model clean, isolated documents to learn within-document memory patterns.

**Configuration:**
```python
model.set_plasticity(enabled=True, commit_mode='heuristic')
model.config.reset_kva_on_doc_boundary = True

# Heuristic commit rule: commit every N tokens OR when eligibility norm exceeds threshold
def should_commit_heuristic(step_count, E_K, E_V, config):
    periodic = (step_count % config.heuristic_commit_interval == 0)

    # Normalized eligibility: ratio of current norm to expected steady-state norm
    # Steady-state Frobenius norm per trace = √r / (1-ρ) for unit-norm candidates
    # Use per-stream norms (dim=(-2,-1) over [r, D_h]), then mean across batch
    steady_state = math.sqrt(config.r) / (1.0 - config.elig_decay_rho)
    per_stream_K = E_K.norm(dim=(-2, -1))                   # [BS]
    per_stream_V = E_V.norm(dim=(-2, -1))                   # [BS]
    elig_norm = ((per_stream_K + per_stream_V) / (2 * steady_state)).mean()  # scalar
    elig_norm = torch.clamp(elig_norm, 0.0, 1.0)  # clamp: √r/(1-ρ) is an upper bound, not expected value
    elig_triggered = (elig_norm > config.elig_commit_threshold)
    return periodic or elig_triggered

# During heuristic commits, use fixed defaults instead of ModNet outputs:
#   λ = 0.95 (moderate decay of existing memory)
#   g = 1.0 (full write strength)
#   slot_logits = zeros(r) (rely on weakness bias for slot selection via softmax)
```

**Heuristic Commit Defaults:**
```python
heuristic_commit_interval: int = 32   # Commit every 32 tokens
elig_commit_threshold: float = 0.5    # Normalized: 0.5 = 50% of saturation upper bound. May need calibration.
heuristic_lambda: float = 0.95        # Moderate decay
heuristic_g: float = 1.0              # Full write strength
# slot_logits = zeros(r): with weakness bias, softmax naturally favors weak slots
```

**Loss Components (conceptual — actual training uses §6.5.5 online loop):**
```python
# These components are computed within the §6.5.5 TBPTT loop,
# NOT via batch-forward. Shown here for clarity of what's optimized.

loss_lm = cross_entropy(logits_t, target_t)  # per-timestep, online

# Regularizers (added to chunk_loss after accumulation):
loss_energy = config.alpha * mean_memory_strength(model)
loss_drift = config.gamma * memory_drift(model)

total_loss = loss_lm + loss_energy + loss_drift
```

**Success Criteria:**
- Memory benchmarks improve (plasticity ON vs OFF)
- Commit rate is sparse (<5% of tokens)
- Base perplexity (plasticity OFF) remains stable

### 7.4 Phase 1.5: Advantage Warmup (Recommended)

**Objective:** Train ModNet on counterfactual advantage labels — "would commit have helped at this point?" — using the logs from Phase 1.

```python
def advantage_warmup_step(model, batch, event_positions):
    for t in event_positions:
        state_t = deepcopy(model.states)

        # Rollout without commit
        model.states = deepcopy(state_t)
        model.force_no_commit = True
        loss_no_commit = rollout(model, batch, t, horizon=16)

        # Rollout with commit
        model.states = deepcopy(state_t)
        model.force_commit = True
        loss_commit = rollout(model, batch, t, horizon=16)

        # Advantage: positive means commit helped
        advantage = loss_no_commit - loss_commit

        # Train ModNet to predict advantage sign
        p_commit = model.layers[l].blocks[b].modnet(state_t)[0]
        label = (advantage > 0).float()
        loss = binary_cross_entropy(p_commit, label)

        loss.backward()
        optimizer.step()
```

### 7.5 Phase 2: RL Training (Per-Token p_commit)

**Objective:** Train ModNet's **commit decision** (`p_commit`) via RL.

**Dataset:** Continue FineWeb-Edu + SlimPajama. Add PG19 (long books) and ProofPile-2 subset (logical text where remembering definitions matters). Training budget: ~2–3B tokens.

**KVA Reset Policy:** `reset_kva_on_doc_boundary = True`. Still resetting at document boundaries. This gives clean counterfactual signals — RL rollouts compare commit-vs-no-commit within a single document, so isolation helps credit assignment.

**Key design:** ModNet outputs p_commit on **every token** (it's a per-token decision). RL rollouts may be **event-driven** (triggered at surprise spikes), but the ModNet policy is always available. Between rollout events, p_commit still gates commits — in Phase 1 via heuristic override, in Phase 2 via the learned policy.

**Scope (MVP):** Phase 2 trains only `p_commit` (commit vs no-commit). The other ModNet heads (λ, g, slot_logits) remain at their **heuristic defaults**. Training these heads requires defining action parameterization, sampling, and credit assignment — deferred to future work.

#### 7.5.1 Event Selection

```python
def select_events(surprise: Tensor, k: int = 2, min_spacing: int = 8) -> List[int]:
    """
    Select top-k surprise events with minimum spacing.
    """
    events = []
    sorted_indices = torch.argsort(surprise, descending=True)

    for idx in sorted_indices:
        idx = idx.item()
        if all(abs(idx - e) >= min_spacing for e in events):
            events.append(idx)
            if len(events) >= k:
                break

    return sorted(events)
```

#### 7.5.2 Counterfactual Rollouts

```python
def rl_training_step(model, batch, chunk_start, chunk_end):
    tokens = batch.tokens[:, chunk_start:chunk_end]

    # Forward pass to get surprise
    with torch.no_grad():
        _, surprise = model.forward_with_surprise(tokens)

    # Select events
    events = select_events(surprise, k=2, min_spacing=8)

    for t in events:
        state_t = model.save_states()

        # === Candidate A: No commit ===
        model.load_states(state_t)
        model.commit_mode = 'force_no'
        loss_A = 0
        for step in range(config.horizon):  # H = 16
            if t + step >= len(tokens) - 1:
                break
            logits = model.forward_token(tokens[t + step])
            loss_A += F.cross_entropy(logits, tokens[t + step + 1])

        # === Candidate B: Commit ===
        model.load_states(state_t)
        model.commit_mode = 'force_yes'
        loss_B = 0
        for step in range(config.horizon):
            if t + step >= len(tokens) - 1:
                break
            logits = model.forward_token(tokens[t + step])
            loss_B += F.cross_entropy(logits, tokens[t + step + 1])

        # === Compute reward ===
        reward = (loss_A - loss_B).item()  # Positive = commit helped

        # Penalties for memory usage
        delta_memory = model.get_memory_usage() - state_t.memory_usage
        delta_drift = model.get_drift()
        reward = reward - config.alpha_rl * delta_memory - config.beta_rl * delta_drift

        # === Update ModNet (per-block, weighted by eligibility contribution) ===
        model.load_states(state_t)

        # Collect per-block eligibility norms as credit proxy
        elig_norms = []
        for layer in model.layers:
            for block in layer.blocks:
                en = (block.E_K.norm() + block.E_V.norm()).item()
                elig_norms.append(en)
        total_elig = sum(elig_norms) + 1e-8

        block_idx = 0
        for layer in model.layers:
            for block in layer.blocks:
                p_commit, _, _, _ = block.modnet(...)

                # BCE with advantage weighting, scaled by block's contribution
                label = 1.0 if reward > 0 else 0.0
                base_weight = min(abs(reward), config.reward_clip)
                # Credit assignment: blocks with more eligibility had more influence
                block_credit = elig_norms[block_idx] / total_elig * len(elig_norms)
                weight = base_weight * block_credit

                loss_rl = weight * F.binary_cross_entropy(
                    p_commit, torch.tensor(label)
                )

                modnet_optimizer.zero_grad()
                loss_rl.backward()
                modnet_optimizer.step()
                block_idx += 1
```

#### 7.5.3 RL Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| k_events | 2 | Events per chunk |
| min_spacing | 8 | Minimum tokens between events |
| horizon (H) | 16 | Rollout length |
| reward_clip | 5.0 | Maximum reward magnitude |
| alpha_rl | 0.01 | Memory usage penalty |
| beta_rl | 0.01 | Drift penalty |

#### 7.5.4 RL Credit Assignment

**Per-block weighting:** The counterfactual rollout produces a single global reward (commit-all vs no-commit-all). To assign credit per block, each block's RL update is weighted by its **eligibility norm ratio**: blocks that accumulated more evidence (high elig_norm) had more influence on the commit outcome and receive proportionally stronger updates. The normalization `elig_norms[i] / total_elig * num_blocks` keeps the total weight equal to the unweighted case.

**Known limitation:** This is a heuristic proxy. True per-block credit would require per-block counterfactuals (commit this block only, keep others unchanged), which costs 2×L×B rollouts per event — prohibitively expensive for MVP. Per-layer counterfactuals (2×L rollouts) are a future middle ground.

### 7.6 Phase 3: Lifelong Learning (No Resets)

**Objective:** Demonstrate the model adapts to new domains and users during inference without retraining slow weights. This is the target deployment mode.

**Prerequisite:** Trained Phase 2 checkpoint with a functional neuromodulator commit policy.

**KVA Reset Policy:** `reset_kva_on_doc_boundary = False`. Fast memory persists across documents indefinitely. The trained neuromodulator manages memory lifetime — base decay (0.999/token) provides natural forgetting, and the budget cap forces overwriting of weak slots.

**Configuration:**
```python
model.set_plasticity(enabled=True, commit_mode='learned')
model.config.reset_kva_on_doc_boundary = False   # Lifelong
# Slow weights: frozen (no optimizer updates)
# Only fast memory (KVA) evolves through neuromodulated commits
```

**Evaluation datasets:**
- Wikipedia topic subsets (domain adaptation speed)
- Synthetic user-profile sequences (personalization recall)
- NarrativeQA (reading comprehension via fast memory)
- Drift suite: validate base competence with plasticity OFF remains stable

**Success Criteria:**
- Perplexity on new domain text drops faster than baseline (no fast memory)
- Injected facts are recallable at distance
- Base competence (plasticity OFF) remains within 5% of Phase 2 end

---

## 8. Gradient Flow and Optimization

### 8.1 What Gets Gradient-Trained

| Component | Type | Optimizer Updates? | Gradients Flow Through? |
|-----------|------|-------------------|------------------------|
| Embedding | Parameter | Yes | Yes |
| W_in, W_o (per layer) | Parameter | Yes | Yes |
| LayerNorm (per layer) | Parameter | Yes | Yes |
| GRU weights (per block) | Parameter | Yes | Yes |
| P, Q, K_proj, V_proj (per block) | Parameter | Yes | Yes (through differentiable eligibility → commit → fast memory read) |
| ModNet (per block) | Parameter | Logging-only (Ph1) / RL (Ph2) | Forward pass only (Ph1), RL signal (Ph2) |
| LM Head | Parameter | Yes | Yes |
| Snapshot projections | Parameter | Yes | Yes |
| Layer/block embeddings | Parameter | Yes | Yes |
| Snapshot decoder | Parameter | Yes | Yes |
| **K, V** (per block) | Differentiable state | **No** (Hebbian) | Read: yes. After commit: carry gradients from eligibility (detached at TBPTT) |
| **a** (per block) | Gradient-free state | **No** (Hebbian) | No (ModNet outputs + elig_magnitude detached at commit) |
| **E_K, E_V** (per block) | Differentiable state | **No** (Hebbian) | Yes within TBPTT chunk (detached at chunk boundaries, same as h) |

### 8.2 Gradient Flow Diagram

```
BACKWARD PASS

                    ∂L/∂logits
                         │
                         ▼
             ┌───────────────────────┐
             │      LM Head          │ ◄── ∂L/∂W_lm (UPDATES W_lm)
             └───────────┬───────────┘
                         │
                         ▼ ∂L/∂ĥ_t
             ┌───────────────────────┐
             │  Snapshot Decoder     │ ◄── UPDATES decoder params
             │  + Snapshot Proj      │ ◄── UPDATES projection + embeddings
             └───────────┬───────────┘
                         │
                         ▼ ∂L/∂h_t^L (merged layer output)
             ┌───────────────────────┐
             │  LayerNorm + W_o      │ ◄── UPDATES W_o, LayerNorm
             └───────────┬───────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          ∂L/∂h^{ℓ,1}  ...    ∂L/∂h^{ℓ,B}   (per block)
              │                     │
              ▼                     ▼
         GRU params            GRU params
         UPDATES               UPDATES
              │                     │
              ▼                     ▼
         Fast Memory Apply     Fast Memory Apply
         ∂L/∂x flows through   ∂L flows through K,V
         K,V (if post-commit:  to E_K,E_V → P,Q,
         grad → eligibility)   K_proj,V_proj (UPDATES)
              │                     │
              ▼                     ▼
         ∂L/∂(W_in split)     ∂L/∂(W_in split)
              │                     │
              └──────────┬──────────┘
                         ▼
              ┌───────────────────────┐
              │       W_in            │ ◄── UPDATES W_in
              └───────────┬───────────┘
                         │
                         ▼
                    ∂L/∂x_t (to prev layer)
```

### 8.3 Differentiable Eligibility

Eligibility traces are differentiable recurrent state, analogous to h (GRU hidden). The update builds an autograd chain within each TBPTT chunk:

```python
# Full autograd chain (within TBPTT chunk):
k_candidate = F.normalize(K_proj(P(x_t_b)).reshape(-1, r, D_h), dim=-1)
E_K_new = rho * E_K + k_candidate   # differentiable recurrence
```

**Gradient path for P, Q, K_proj, V_proj:**
When a commit happens at step `t_c` within a TBPTT chunk and the committed memory is read at step `t_r > t_c`:

```
LM loss at t_r  →  fast memory read (y_fast)  →  K[i], V[i] (from commit)
  →  E_K[i], E_V[i] (accumulated over t_0..t_c)  →  k_candidate, v_candidate
  →  K_proj, V_proj, P, Q  (receive gradient signal)
```

This means P/Q/K_proj/V_proj learn **what to remember** from the main LM loss during standard training (Phase 0-1), while RL trains **when to commit** (Phase 2).

**TBPTT detachment:** At chunk boundaries, E_K and E_V are detached (§5.4), bounding the autograd chain to at most T=256 steps — the same cost as the GRU hidden state chain.

**Memory cost:** The main additional autograd cost is retaining the F.normalize saved tensors and projection intermediates across all T steps (in the old no_grad version, these were freed after each step). Per block per step: ~2 × [BS, r, D_h] normalize saves + projection saves ≈ 130 KB at BS=32, fp16. Over T=256 steps: ~34 MB per block. Over all blocks (L=8, B=4 = 32): **~1.1 GB** at fp16 (or ~2.2 GB at fp32 if eligibility uses full precision). This is significant but eligible to fit in the 4090's 24 GB alongside model parameters, optimizer states, GRU activations, and snapshot decoder. Monitor actual VRAM usage during implementation — the estimate depends on AMP settings and whether eligibility uses fp16 or fp32. If VRAM becomes tight at larger BS, gradient checkpointing on the eligibility chain can trade recomputation for memory.

### 8.4 Selective Detaching in Commit

The commit operation uses a **split detach strategy** — eligibility values are differentiable, but ModNet outputs are detached:

```python
# ModNet outputs DETACHED (commit policy under RL control):
λ = λ.detach()
g = g.detach()
slot_logits = slot_logits.detach()

# Eligibility NOT detached (P/Q/K_proj/V_proj get LM loss gradient):
K = F.normalize((1 - alpha) * K + alpha * F.normalize(E_K, dim=-1), dim=-1)
```

**Why this split?**
- **E_K/E_V not detached:** Allows backprop to train P/Q/K_proj/V_proj to produce useful memory candidates. The gradient path: LM loss → fast memory read → committed K/V → eligibility → projection params.
- **ModNet outputs detached:** Prevents the LM loss from directly optimizing commit timing/strength, which would compete with the RL objective (Phase 2). RL trains *when/where* to commit via its own reward signal.
- **Strength `a` gradient-free:** `elig_magnitude` is detached when computing strength updates, so `a` stays out of the autograd graph. Strengths are under ModNet/RL control.
- **K/V detached at TBPTT boundaries:** After commit, K/V carry gradients from eligibility within the current chunk. These are detached in `detach_states()` (§5.4) to prevent cross-chunk gradient chains.

### 8.5 Gradient Health

```python
def training_step_with_monitoring(model, batch, optimizer):
    optimizer.zero_grad()

    loss = compute_loss(model, batch)
    loss.backward()

    # 1. Gradient clipping
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # 2. NaN detection
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise RuntimeError(f"NaN gradient in {name}")

    # 3. Logging
    wandb.log({
        'grad_norm': total_norm,
        'loss': loss.item(),
    })

    optimizer.step()
```

---

## 9. Stability Mechanisms

### 9.1 Overview

Since fast memory is LIFELONG in Phase 3+ deployment (and persists within documents during Phases 0–2), stability is critical. These mechanisms are NON-NEGOTIABLE.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STABILITY MECHANISMS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. NORMALIZATION                                                           │
│     K, V are always unit-normalized after updates                           │
│     Ensures dot products are bounded cosine similarities                    │
│                                                                             │
│  2. STRENGTH CLIPPING                                                       │
│     a ∈ [0, a_max]                                                          │
│     Prevents any single slot from dominating (non-negative for MVP)         │
│                                                                             │
│  3. GLOBAL BUDGET (per block)                                               │
│     Σ|a| ≤ budget                                                           │
│     Total memory influence is bounded                                       │
│                                                                             │
│  4. MINIMUM DECAY                                                           │
│     λ ∈ (λ_min, 1) where λ_min = 0.9                                       │
│     Memory always decays somewhat during commits                            │
│                                                                             │
│  5. BASE DECAY (every step)                                                 │
│     a ← a * base_decay (e.g., 0.999)                                       │
│     Even without commits, memory slowly fades                               │
│                                                                             │
│  6. COMMIT RATE CAP                                                         │
│     Max commits per N tokens per block                                      │
│     Prevents rapid memory thrashing                                         │
│                                                                             │
│  7. WEAKNESS BIAS (in soft commit)                                          │
│     Softmax logits biased toward weak slots                                 │
│     Natural turnover, prevents slot hoarding                                │
│                                                                             │
│  8. SOFT EMA UPDATES                                                        │
│     K_i ← normalize((1-α)K_i + α·new)                                      │
│     Gradual blending instead of hard overwrites                             │
│                                                                             │
│  9. TRAINING REGULARIZERS                                                   │
│     L_energy, L_drift (Phase 1); RL reward penalties (Phase 2)              │
│     Encourage sparse, stable memory usage                                   │
│                                                                             │
│  10. MONITORING                                                             │
│      Track plasticity-OFF perplexity throughout training                    │
│      Detect if base competence degrades                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Implementation

```python
class StabilityConfig:
    # Strength bounds
    a_max: float = 3.0

    # Budget (per block)
    budget: float = 4.0

    # Decay
    lambda_min: float = 0.9
    base_decay: float = 0.999  # Applied every step

    # Commit limits
    max_commits_per_chunk: int = 5  # Per block, per 256 tokens

    # Soft commit config
    weakness_weight: float = 0.5    # Bias toward weak slots in softmax
    tau: float = 1.0                # Softmax temperature for slot selection
    commit_top_k: int = 2           # Number of slots to update per commit

    # Regularizer weights (Phase 1)
    alpha: float = 1e-3   # Energy (memory strength)
    gamma: float = 1e-4   # Drift (K/V change rate)

    # Heuristic commit config (Phase 1)
    heuristic_commit_interval: int = 32
    elig_commit_threshold: float = 0.5   # Normalized: fraction of steady-state elig norm

    # Phased KVA reset (see §5.2)
    reset_kva_on_doc_boundary: bool = True  # Phase 0–2: True. Phase 3+: False.


def apply_stability_constraints(K, V, a, config):
    """Apply all stability constraints to fast memory (per block).

    Args:
        K: [BS, r, D_h], V: [BS, r, D_h], a: [BS, r]
    """

    # 1. Normalize K, V
    K = F.normalize(K, dim=-1)
    V = F.normalize(V, dim=-1)

    # 2. Clip strengths (non-negative for MVP)
    a = torch.clamp(a, 0, config.a_max)

    # 3. Budget constraint (per-stream)
    total = torch.sum(torch.abs(a), dim=-1, keepdim=True)  # [BS, 1]
    a = torch.where(
        total > config.budget,
        a * (config.budget / (total + 1e-8)),
        a
    )

    return K, V, a


def apply_base_decay(a, config):
    """Apply base decay every step (even without commit)."""
    return a * config.base_decay
```

---

## 10. Hyperparameters

### 10.1 Model Architecture (Tier A)

| Parameter | Value | Description |
|-----------|-------|-------------|
| D | 512 | Hidden dimension (full width) |
| L | 8 | Number of NML layers |
| B | 4 | NML blocks per layer (multi-head) |
| D_h | D/B = 128 | Per-block hidden dimension |
| r | 8 | Fast memory slots per block |
| d_p | 64 | Pre-synaptic compression (per block) |
| d_q | 64 | Post-synaptic compression (per block) |
| vocab_size | ~50k | Tokenizer dependent |
| d_s | 256 | Snapshot projection dimension |
| M | L×B = 32 | Snapshot tokens (one per block) |

### 10.2 Fast Memory (Per Block)

| Parameter | Value | Description |
|-----------|-------|-------------|
| a_max | 3.0 | Maximum strength magnitude |
| budget | 4.0 | Global budget per block (Σ|a|) |

### 10.3 Soft Commit

| Parameter | Value | Description |
|-----------|-------|-------------|
| tau | 1.0 | Softmax temperature for slot selection |
| commit_top_k | 2 | Number of slots to update per commit |
| weakness_weight | 0.5 | Bias toward weak slots in softmax logits |

### 10.4 Eligibility

| Parameter | Value | Description |
|-----------|-------|-------------|
| elig_decay_rho | 0.95 | Eligibility decay (per token). Local alias `rho` in function signatures. |

### 10.5 Neuromodulator

| Parameter | Value | Description |
|-----------|-------|-------------|
| lambda_min | 0.9 | Minimum decay factor |
| g_max | 1.0 | Maximum scalar write strength |
| modnet_hidden | 64 | ModNet hidden dimension (per block) |

### 10.6 Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| TBPTT chunk size | 256 | Tokens per backprop chunk |
| Batch size (BS) | 16–32 | Parallel persistent streams (4090 VRAM allows up to ~64) |
| Grad accumulation | 2–4 | Steps before optimizer update (effective BS = BS × accum) |
| Learning rate | 3e-4 | AdamW learning rate (cosine decay to 1e-5) |
| Weight decay | 0.01 | AdamW weight decay |
| Max grad norm | 1.0 | Gradient clipping threshold |

### 10.7 RL Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| k_events | 2 | Events per chunk |
| min_spacing | 8 | Minimum event spacing |
| horizon (H) | 16 | Rollout horizon |
| reward_clip | 5.0 | Maximum reward magnitude |
| alpha_rl | 0.01 | Memory usage penalty |
| beta_rl | 0.01 | Drift penalty |

### 10.8 Regularizers (Phase 1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| alpha | 1e-3 | Energy penalty weight (memory strength) |
| gamma | 1e-4 | Drift penalty weight (K/V change rate) |

**Note:** Phase 1 uses heuristic commits, so there are no write-strength or commit-probability penalties. ModNet penalties are handled by the RL reward signal in Phase 2.

### 10.9 Snapshot Memory

| Parameter | Value | Description |
|-----------|-------|-------------|
| decoder_mode | 'snapshot' | 'simple' or 'snapshot' |
| d_s | 256 | Snapshot projection dimension |
| M | L×B = 32 | Snapshot tokens (one per block) |
| decoder_heads | 4 | Cross-attention heads |
| decoder_dropout | 0.1 | Cross-attention dropout |

**Derived values:**
- Snapshot tokens M = L × B (one per block, dense mode)
- Cross-attention cost: O(M · d_s) = O(32 × 256) = negligible

**Fallback configuration (simple mode):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| decoder_mode | 'simple' | Disable snapshot cross-attention, use h_t^L only |

---

## 11. Evaluation Strategy

### 11.1 Core Metrics

| Metric | When | Purpose |
|--------|------|---------|
| Validation perplexity | Every N steps | LM quality |
| Perplexity (plasticity OFF) | Every N steps | Base competence preservation |
| Commit rate | Every step | Sparsity monitoring |
| Memory usage (Σ|a| per block) | Every step | Budget monitoring |
| Gradient norm | Every step | Training health |

### 11.2 Memory Benchmarks

#### 11.2.1 Key-Value Retrieval

```
Task: Store fact, retrieve after delay

Input: "The capital of France is Paris. [... 200 distractor tokens ...]
        Question: What is the capital of France?"

Measure: Accuracy at different delay lengths (64, 128, 256, 512 tokens)

Compare: Plasticity ON vs OFF
```

#### 11.2.2 Personalization

```
Task: Remember user preferences within session

Input: "My name is Alex. I prefer Python over JavaScript. [... interaction ...]
        What programming language do I prefer?"

Measure: Accuracy on preference queries
```

### 11.3 Stability Benchmarks

#### 11.3.1 Drift Detection

```
Protocol:
1. Evaluate perplexity with plasticity OFF
2. Run N tokens with plasticity ON
3. Evaluate perplexity with plasticity OFF again
4. Compare: should be stable (< 5% increase)
```

#### 11.3.2 Long-Run Stability

```
Protocol:
1. Run model on 1M+ tokens with plasticity ON
2. Monitor: commit rate, memory usage, drift per step
3. All should remain bounded
```

### 11.4 Logging

```python
# Per step
wandb.log({
    'loss/lm': loss_lm,
    'loss/energy': loss_energy,
    'loss/drift': loss_drift,
    'grad_norm': total_norm,
})

# Per layer, per block, per step
for l, layer in enumerate(model.layers):
    for b, block in enumerate(layer.blocks):
        wandb.log({
            f'layer{l}_block{b}/commit_rate': block.commit_rate,
            f'layer{l}_block{b}/memory_usage': block.a.abs().sum(),
            f'layer{l}_block{b}/mean_strength': block.a.abs().mean(),
            f'layer{l}_block{b}/max_strength': block.a.abs().max(),
            f'layer{l}_block{b}/elig_norm_raw': (block.E_K.norm(dim=(-2,-1)) + block.E_V.norm(dim=(-2,-1))).mean().item(),
            f'layer{l}_block{b}/elig_norm_ratio': ((block.E_K.norm(dim=(-2,-1)) + block.E_V.norm(dim=(-2,-1))) / (2 * math.sqrt(config.r) / (1 - config.elig_decay_rho))).mean().item(),
        })

# Periodic
wandb.log({
    'val/perplexity_on': eval_perplexity(model, val_data, plasticity=True),
    'val/perplexity_off': eval_perplexity(model, val_data, plasticity=False),
    'memory_bench/accuracy_64': memory_bench(model, delay=64),
    'memory_bench/accuracy_256': memory_bench(model, delay=256),
})
```

---

## 12. Implementation Plan

### 12.1 Repository Structure

```
neuromorphic/
├── spec/
│   └── SPECIFICATION.md          # This document
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py              # NeuromorphicLM (main model)
│   │   ├── nml_layer.py          # NMLLayer (multi-block wrapper)
│   │   ├── nml_block.py          # NMLBlock (single block: GRU+FM+elig+ModNet)
│   │   ├── fast_memory.py        # FastMemory (K, V, a, apply, soft commit)
│   │   ├── eligibility.py        # EligibilityTrace (differentiable E_K, E_V)
│   │   ├── modulator.py          # ModNet (neuromodulator, per block)
│   │   ├── gru.py                # GRU with y_fast injection (D_h)
│   │   ├── snapshot.py           # SnapshotProjection (L×B tokens)
│   │   └── decoder.py            # SnapshotDecoder (cross-attention)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # TBPTT training loop
│   │   ├── rl_trainer.py         # Phase 2 RL training
│   │   └── regularizers.py       # Stability regularizers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset loading
│   │   └── synthetic.py          # Memory benchmark generation
│   └── eval/
│       ├── __init__.py
│       ├── perplexity.py         # Perplexity evaluation
│       ├── memory_bench.py       # Key-value retrieval benchmark
│       └── drift_suite.py        # Stability evaluation
├── configs/
│   ├── tier_a.yaml               # Tier A hyperparameters
│   └── tier_b.yaml               # Tier B hyperparameters
├── scripts/
│   ├── train_phase0.py           # Phase 0 training
│   ├── train_phase1.py           # Phase 1 training
│   ├── train_phase2.py           # Phase 2 training
│   └── evaluate.py               # Run evaluation suite
├── tests/
│   ├── test_nml_block.py
│   ├── test_nml_layer.py
│   ├── test_fast_memory.py
│   ├── test_eligibility.py
│   ├── test_modulator.py
│   ├── test_soft_commit.py
│   └── test_snapshot.py
└── requirements.txt
```

### 12.2 Reusable Components (Use Libraries)

| Component | Library | Notes |
|-----------|---------|-------|
| Tokenizer | `transformers.GPT2Tokenizer` | Or LlamaTokenizer |
| Datasets | `datasets` (HuggingFace) | TinyStories, OpenWebText |
| Optimizer | `torch.optim.AdamW` | Standard |
| LR Scheduler | `transformers.get_scheduler` | Cosine, linear, etc. |
| Logging | `wandb` | Experiment tracking |
| Mixed Precision | `torch.cuda.amp` | autocast + GradScaler |
| Checkpointing | `torch.save/load` | Standard |
| Embedding | `torch.nn.Embedding` | Standard |
| LayerNorm | `torch.nn.LayerNorm` | Standard |
| MultiheadAttention | `torch.nn.MultiheadAttention` | For snapshot decoder |

### 12.3 Custom Components (Build)

| Component | Reason |
|-----------|--------|
| NMLBlock | Novel: GRU + fast memory + eligibility + ModNet + commit per block |
| NMLLayer | Novel: multi-block wrapper with W_in, W_o, residual, LayerNorm |
| FastMemory | Novel K/V/a representation as dynamic low-rank adapter |
| EligibilityTrace | Novel differentiable Hebbian accumulation with projections |
| ModNet | Custom architecture and per-block outputs |
| SoftCommit | Novel soft multi-slot EMA commit operation |
| GRU with injection | Need to inject y_fast into h_tilde |
| SnapshotProjection | Spatial cross-layer/block projection with identity embeddings |
| SnapshotDecoder | Cross-attend final state to M = L×B snapshot tokens |
| TBPTT Trainer | Custom state management with multi-block |
| RL Trainer | Custom counterfactual rollouts for per-token p_commit |
| Memory Benchmarks | Custom evaluation tasks |

### 12.4 Implementation Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M0 | Baseline multi-block GRU LM (no plasticity, simple decoder) | Perplexity decreases on TinyStories |
| M1 | FastMemory apply (no commits) | No regression from M0 |
| M1.5 | Spatial snapshot (M=L×B) + decoder cross-attention | No regression, snapshot mode works |
| M2 | Eligibility + heuristic soft commits | Memory bench shows improvement |
| M3 | ModNet with heuristic commits (logging-only) | Stable training, sparse commits |
| M4 | Phase 2 RL training of p_commit | Learned commits beat heuristic |
| M5 | Full evaluation suite | All metrics tracked |
| M6 | Tier B scaling | Improvements carry over |

### 12.5 Fallback Strategy

If training becomes unstable:

1. **Switch to simple decoder mode:**
   ```yaml
   decoder_mode: 'simple'  # Disable snapshots, use only h_t^L
   ```

2. **Reduce blocks:**
   ```yaml
   B: 1  # Single block per layer (no multi-head)
   ```

3. **Debug incrementally:**
   - First verify M0 works (baseline with simple decoder)
   - Add snapshot system (M1.5) and verify no regression
   - Then add fast memory (M1, M2)
   - Then add ModNet (M3, M4)

---

## 13. Future Directions

### 13.1 Hierarchical Organization

Current: Flat stack of L NML layers, each with B blocks.

Future: Hierarchical columns with different timescales, where groups of layers form "columns" with column-level communication.

```
┌─────────────────────────┐
│   Column 2 (abstract)   │  Slower adaptation, higher-level
│   ┌─────┐ ┌─────┐       │  concepts
│   │NML 5│→│NML 6│       │
│   └─────┘ └─────┘       │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│   Column 1 (concrete)   │  Faster adaptation, lower-level
│   ┌─────┐ ┌─────┐       │  patterns
│   │NML 1│→│NML 2│       │
│   └─────┘ └─────┘       │
└─────────────────────────┘
```

### 13.2 Multi-Timescale Memory

- Short-term: Current fast memory (tokens to minutes)
- Medium-term: Episodic store (hours to days)
- Long-term: Consolidation to slow weights ("sleep")

### 13.3 Parallelization

- Replace GRU with scan-friendly recurrence (Mamba-style)
- Custom Triton kernels for fast memory operations

### 13.4 Alternative Similarity

- Learned energy functions instead of dot product
- "Lock-and-key" bilinear compatibility

### 13.5 Multimodal

- Image tokens (patch embeddings) into same stream
- Cross-modal memory associations

### 13.6 Silicon Mapping

- Quantization-aware training
- Sparse memory access patterns
- Energy-efficient hardware implementation

### 13.7 Richer RL Action Space

- Train λ, g, slot_logits via RL (not just p_commit)
- Joint action space: commit probability + continuous λ/g + discrete slot
- Sampling from policy: Gumbel-softmax for slot, Beta for λ/g

---

## Appendix A: Quick Reference

### A.1 Tensor Shapes (Tier A)

| Tensor | Shape | Description |
|--------|-------|-------------|
| token_ids | [batch, T] | Input batch |
| x | [batch, T, 512] | Embeddings |
| x_blocks | [batch, B, D_h] = [batch, 4, 128] | Per-block input (after W_in split) |
| h^{ℓ,b} | [batch, 128] | Per-block hidden state |
| h^ℓ (merged) | [batch, 512] | Per-layer merged output |
| K (per block) | [BS, 8, 128] | Fast memory keys |
| V (per block) | [BS, 8, 128] | Fast memory values |
| a (per block) | [BS, 8] | Fast memory strengths |
| E_K (per block) | [BS, 8, 128] | Eligibility keys |
| E_V (per block) | [BS, 8, 128] | Eligibility values |
| x_q | [batch, 128] | Normalized query for fast memory |
| y_fast | [batch, 128] | Fast memory output (per block) |
| snapshot | [batch, 32, 256] | Spatial snapshot (M=L×B=32 tokens) |
| ĥ | [batch, 512] | Decoder output (after cross-attn) |
| logits | [batch, T, vocab] | LM output |

### A.2 Key Equations

**Dynamic Low-Rank Adapter (per block):**
```
W_fast(t) = Σᵢ aᵢ(t) · (Vᵢ(t) ⊗ Kᵢ(t))     # never materialized
```

**Base Decay (every token, unconditional):**
```
a ← a * base_decay                             # 0.999 — slow fade
```

**Fast Memory Apply:**
```
x_q = normalize(x_t^b)                          # unit-normalize query for true cosine
y_fast = Σᵢ aᵢ · Vᵢ · (Kᵢᵀ · x_q)
```

**GRU with Injection (per block, D_h space):**
```
u = σ(Wᵤx^b + Uᵤh^b + bᵤ)
r = σ(Wᵣx^b + Uᵣh^b + bᵣ)
h̃ = tanh(Wₕx^b + Uₕ(r⊙h^b) + bₕ + y_fast)
h'^b = (1-u)⊙h^b + u⊙h̃
```

**Layer Merge:**
```
h^ℓ = LayerNorm(x_t + W_o · concat(h^{ℓ,1}, ..., h^{ℓ,B}))
```

**Eligibility Update (Differentiable, Hebbian: K from input, V from hidden):**
```
k_candidate = normalize(K_proj(P(x^b)))         # in autograd graph
v_candidate = normalize(V_proj(Q(h^b)))         # in autograd graph
E_K = ρ·E_K + k_candidate                       # differentiable recurrence (detached at TBPTT)
E_V = ρ·E_V + v_candidate                       # differentiable recurrence (detached at TBPTT)
```

**Soft Multi-Slot Commit (per-stream: only where commit_mask=True):**
```
λ, g, slot_logits = detach(ModNet outputs)         # commit policy under RL
g ← g * commit_mask                               # zero g for non-commit streams
λ ← where(commit_mask, λ, 1)                      # skip decay for non-commit streams
a ← a ⊙ λ                                         # commit-time decay
w = softmax((slot_logits + weakness_bonus) / τ)    # soft slot weights
w = top_k_sparsify(w, k)                           # keep top-k, renormalize
α = w · g                                          # [r] effective write weights (0 for non-commit)
K ← normalize((1-α)·K + α·normalize(E_K))         # E_K NOT detached (grad→P,Q)
V ← normalize((1-α)·V + α·normalize(E_V))         # E_V NOT detached (grad→P,Q)
a ← a + α · ||E_V.detach()||                      # strength (a stays gradient-free)
E_K, E_V ← E * (1 - commit_mask)                  # reset only commit streams
```

**Snapshot Token (m = ℓ × B + b):**
```
s_m = proj_ℓ(h_t^{ℓ,b}) + layer_emb[ℓ] + block_emb[b]
```

### A.3 Training Loss

```
Phase 1 (heuristic commits, ModNet logging-only):
  L = L_LM + α·L_energy + γ·L_drift

Phase 2 (RL — p_commit only; λ/g/slot remain heuristic):
  L_LM trained separately; ModNet p_commit updated via RL reward signal

where:
  L_LM = CrossEntropy(logits, targets)
  L_energy = mean(|a|)           # penalize total memory strength (per block)
  L_drift = mean(||K_t - K_{t-1}|| + ||V_t - V_{t-1}||)  # penalize rapid changes
```

### A.4 RL Reward

```
R = (loss_no_commit - loss_commit) - α_rl·Δ|a| - β_rl·Δdrift
```

---

**END OF SPECIFICATION**
