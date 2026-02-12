````markdown
# Brain-Stack Neuromorphic LM — v2 Implementation Specification

**Version:** 2.6
**Status:** Phases A, B, D implemented; two-mode neuromodulators (heuristic → learned via main-loss backprop); enhanced EM with learned g_em, tau, ww, decay + novelty; no RL — all neuromod params trained by main optimizer; spatial decoder (hierarchical aggregation + deep cross-attention)
**Last Updated:** 2026-02-12
**Primary Constraint:** single RTX 4090 (24GB), mixed precision, persistent-stream TBPTT

This v2 spec is designed to be handed directly to an implementation agent. It intentionally preserves the training-correctness and state semantics learned from v1.7, while adding Working Memory (WM) + Episodic Memory (EM) and making the core **scan-friendly** via an **affine recurrence** and **plasticity updates only at fixed boundaries**.

---

## 0) Non-Negotiables to Carry Over from v1.7

These are required for correctness/stability/VRAM.

### 0.1 Persistent parallel streams + TBPTT correctness
- Training uses **BS independent persistent streams**, not independent sequences.
- **Per-timestep** doc-boundary reset (not per TBPTT segment). Note: the parallel `forward_span` path resets only at the first token of each span; PM/EM state leaks across mid-span doc boundaries (bounded to P-1 tokens). This is an intentional trade-off for scan efficiency — see `MODEL_EXPLAINER_PARALLEL.md` §10.
- **Skip loss** at cross-document transitions (mask positions where `input_tokens[:, t] == eot_id`) during phases that enforce doc isolation.
- **Online loss accumulation** — do **not** materialize full `[BS, T, vocab]` logits. Per-span `[BS, P, vocab]` materialization is acceptable (P ≪ T).

### 0.2 Runtime state semantics
- Memory/state tensors are **plain tensor attributes** (NOT `register_buffer`).
- Explicit `state.save_runtime_state(model)` / `state.load_runtime_state(model, state)` (free functions in `state.py`) for Phase 3+ persistence.
- Explicit `detach_states()` at TBPTT boundaries.

### 0.3 Per-stream masking everywhere
- Doc-boundary reset is per-stream.
- Commit/write decisions are per-stream.
- Any decay/write/reset must be masked per-stream.

---

## 1) Architecture Overview

### 1.1 Block/Layer Hierarchy

The model is organized as **B=4 parallel blocks**, each containing **L=8 sequential layers**:

```
Block 0 ─┬─ Layer 0 ─ Layer 1 ─ ... ─ Layer 7
Block 1 ─┬─ Layer 0 ─ Layer 1 ─ ... ─ Layer 7
Block 2 ─┬─ Layer 0 ─ Layer 1 ─ ... ─ Layer 7
Block 3 ─┬─ Layer 0 ─ Layer 1 ─ ... ─ Layer 7
```

Each block processes `D_h = D / B = 128` dimensions. Blocks run in parallel; layers within a block run sequentially.

### 1.2 Four Memory Types

v2 decomposes memory into four parts:

1) **Genetic memory (slow weights):** normal parameters trained by backprop. Frozen at deployment.
2) **Working Memory (WM):** bounded sliding-window attention over the last `W` tokens (precise copying / binding). No post-training evolution. **One WM shared across the model.**
3) **Procedural Memory (PM):** fast low-rank weights (`pm_K/pm_V/pm_a`) + differentiable eligibility. Updated via neuromodulated commits. **One PM per layer per block = B × L instances**, each with its own `PMNeuromodulator`.
4) **Episodic Memory (EM):** fixed-size per-stream vector store (`em_K/em_V/em_S`). Retrieves **latent vectors** (not tokens). Written via neuromodulation. **One EM per block = B instances**, each with its own `EMNeuromodulator`.

### 1.3 Memory Ownership Summary

| Memory | Count | Neuromodulator | Scope |
|--------|-------|----------------|-------|
| WM | 1 | None (standard attention) | Shared across model |
| PM | B × L | `PMNeuromodulator` per instance | Per layer within each block |
| EM | B | `EMNeuromodulator` per instance | Per block (shared across L layers in that block) |

### 1.4 Scan-Friendliness via Plasticity Boundaries

**Key design decision:**
- **PM/EM writes occur only at fixed "plasticity boundaries"** (every `P` tokens).
- Within each `P`-token span, PM and EM are **read-only** (frozen).
- This makes the core recurrence and eligibility updates **scan-friendly** within spans, while still allowing gradients from later tokens in the TBPTT segment to train eligibility projections (because commits happen *within* TBPTT, at span boundaries).

---

## 2) High-Level Dataflow (per token)

At timestep `t`:

1) Embed token: `x = Embedding(token_id)` → `[BS, D]`
2) Working memory: `y_wm = WM.step(x)` → `[BS, D]` (shared across model)
3) **Per block b** (B blocks in parallel):
   - EM retrieval: `y_em[b] = EM[b].retrieve_and_aggregate(x, y_wm)` → `[BS, D]`
   - **Per layer ℓ** (L layers sequentially within block):
     - PM read: `y_pm = PM[b][ℓ].apply(x_block)` → `[BS, D_h]`
     - Recurrence: `h = a ⊙ h_prev + b` (a, b from input features)
     - Eligibility: `elig_K = ρ * elig_K + k_cand`, `elig_V = ρ * elig_V + v_cand`
4) **Output pathway** (two modes):
   - **Simple** (`snapshot_enabled=False`): `logits = LMHead(concat(h^L for all blocks))` → `[BS, vocab]`
   - **Spatial decoder** (`snapshot_enabled=True`, the default): hierarchical aggregation of intermediate layer outputs + memory readouts, then deep cross-attention decoder → enhanced `h` → `LMHead` → `[BS, vocab]`. See §4.11. Supported in both sequential and parallel forward paths.
5) At plasticity boundaries (every P tokens):
   - `EMNeuromodulator[b]` decides EM writes per block
   - `PMNeuromodulator[b][ℓ]` decides PM commits per (block, layer)

---

## 3) Model Scaling Tiers

All tiers designed to train on a single RTX 4090 (24GB) with bf16 mixed precision.

### 3.1 Tier Comparison

| Tier | Params | D | L | B | D_h | Target | 4090 BS |
|------|--------|---|---|---|-----|--------|---------|
| **A (Debug)** | ~50M | 512 | 8 | 4 | 128 | Rapid iteration, sanity checks | 32–64 |
| **B (Competitive)** | ~150M | 768 | 12 | 6 | 128 | Match GPT-2 Small scale | 16–32 |
| **C (Strong)** | ~350M | 1024 | 24 | 8 | 128 | Match GPT-2 Medium scale | 8–16 |

**Note:** Early LLMs (GPT-1/GPT-2 Small at ~125M) showed meaningful language understanding. Tier B is our target for demonstrating competitive results.

### 3.2 VRAM Budget (bf16 training)

| Component | Tier A | Tier B | Tier C |
|-----------|--------|--------|--------|
| Weights (bf16) | ~100MB | ~300MB | ~700MB |
| Optimizer (fp32) | ~400MB | ~1.2GB | ~2.8GB |
| Gradients (bf16) | ~100MB | ~300MB | ~700MB |
| Activations (BS=16) | ~1GB | ~2GB | ~4GB |
| PM/EM/WM state | ~50MB | ~100MB | ~200MB |
| **Total** | **~1.7GB** | **~4GB** | **~8.5GB** |

All tiers fit comfortably on a 4090. Tier C leaves headroom for gradient checkpointing if needed.

### 3.3 Default Configuration (Tier B — Recommended)

**Core dimensions:**
- `D = 768` — model width
- `L = 12` — layers per block
- `B = 6` — parallel blocks
- `D_h = D / B = 128` — per-block hidden dimension
- `vocab` — vocabulary size (GPT-2 50257 recommended; `vocab_size` and `eot_id` set from chosen tokenizer at runtime)

**Procedural Memory (PM):**
- `r = 16` — slots per PM instance (B × L = 72 instances total)
- `ρ = 0.95` — eligibility decay

**Working Memory (WM):**
- `W = 512` — sliding window size (tokens)
- `D_wm = 192` — WM key/value dimension
- `n_heads_wm = 6` — attention heads

**Episodic Memory (EM):**
- `M = 512` — capacity per EM bank (B = 6 banks total)
- `D_em = 192` — EM key/value dimension
- `k_ret = 8` — retrieval count
- `C = 16` — candidates per span

**Spatial Decoder (Tier B):**
- `d_dec = 384` — decoder working dimension
- `n_heads_decoder = 6` — attention heads

Memory capacities scale with tier. Tier A uses smaller defaults (r=8, W=256, M=256) for rapid iteration. Tier C doubles again (r=32, W=1024, M=1024) for maximum capacity.

**Training:**
- `P = 32` — plasticity span (tokens)
- `T = 256` — TBPTT segment length (8 spans per segment)
- `BS = 16` — batch size (persistent streams)

---

## 4) Module Specs

### 4.1 Embedding + LM head
- Standard `nn.Embedding(vocab, D)`
- LM head `nn.Linear(D, vocab, bias=False)` (optionally tied)

### 4.2 Working Memory (WM) — sliding window attention
**Goal:** restore transformer-like precision on short context (copying, binding).

**State per stream:**
- `wm_K: [BS, W, D_wm]`
- `wm_V: [BS, W, D_wm]`
- `wm_valid: [BS, W]` (bool; cleared at doc resets in phases A–C)
- `wm_ptr: [BS]` (ring buffer index)

**Computation (per token MVP streaming):**
- `q = W_q(x_t)` → `[BS, D_wm]`
- `K_t = W_k(x_t)`, `V_t = W_v(x_t)` → `[BS, D_wm]`
- write `(K_t, V_t)` into ring buffer at `wm_ptr`
- attend over valid positions in last W: `y_wm = Attn(q, wm_K, wm_V, causal=True)`
- project to `D`: `y_wm = W_o(y_wm)` → `[BS, D]`

**Note:** For a later optimized path, compute WM over each span `[BS, P, D]` using a block/windowed SDPA with prefix cache. MVP can be streaming per token.

### 4.3 Episodic Memory (EM) — per-block, per-stream bank
**Key decision:** one EM bank per block (B total), each controlled by its own `EMNeuromodulator`. Each EM is shared across all L layers within its block.

#### 4.3.1 EM State (per block, per stream)
Each of the B blocks has its own EM bank with tensors:
- `em_K: [BS, M, D_em]` (unit-normalized keys)
- `em_V: [BS, M, D_em]` (values; optionally normalized)
- `em_S: [BS, M]` (strength/importance, bounded)
- optional:
  - `em_age: [BS, M]` (int)
  - `em_valid: [BS, M]` (bool)

Total EM state: B × (above tensors) = B independent EM banks.

Initialize (per block):
- `em_K, em_V` random orthogonal or normal + normalize
- `em_S = 0` (inactive)

#### 4.3.2 EM Retrieval (fixed k_ret)
**Inputs per token:**
- query `q: [BS, D_em]` computed from **input-side features** (not from recurrent state), e.g.:
  - `q = W_q_em(concat(x, y_wm))` → `[BS, D_em]`

**Scores:**
- `scores = em_K @ normalize(q)` → `[BS, M]`

**Top-k retrieval:**
- `idx = topk(scores, k_ret)` → `[BS, k_ret]`
- `V_top = gather(em_V, idx)` → `[BS, k_ret, D_em]`
- also gather `scores_top` for weights.

**Return format:**
- Return `mem_tokens = V_top` as **k_ret latent "memory tokens"**.
- Aggregate into one vector with a single-query cross-attention:
  - `out = CrossAttn(query=W_q(x), keys=mem_tokens, values=mem_tokens)` → `[BS, D_em]`
  - Post-retrieval FFN: `out = out + FFN(LayerNorm(out))` (pre-norm residual, GELU, 4× expansion)
  - `y_em = W_o_cross(out)` → `[BS, D]`

The readout FFN (controlled by `em_readout_ffn`, default `True`) adds nonlinear processing after cross-attention aggregation, allowing the model to reason about what it retrieved before injecting into the recurrence.

This "return k tokens + cross-attend + process" avoids forcing premature averaging and preserves multi-item recall.

#### 4.3.3 EM Candidate Proposals (computed per token, written at boundaries)
At each token, compute:
- `k_cand = normalize(W_k_cand(concat(x, y_wm)))` → `[BS, D_em]`
- `v_cand = W_v_cand(h_final)` → `[BS, D_em]`
  **Implementation choice:** use top layer merged representation `h_final` (MVP stable).
- `novelty` → `[BS]` (novelty score):
  - Heuristic (`em_enabled=False`): `novelty = clamp(0.5 * surprise + 0.5 * (1 - max_cos_sim(em_K, k_cand)), 0, 1)`
  - Learned (`em_enabled=True`, Phase B+): `w_nov = sigmoid(W_nov(concat(x, y_wm)))`, then `novelty = clamp(w_nov * surprise + (1 - w_nov) * (1 - max_sim), 0, 1)`. `W_nov: nn.Linear(D + D, 1)` is on `EpisodicMemory`, trained by main loss backprop.

Store candidates into a per-span buffer:
- `cand_K: [BS, P, D_em]`
- `cand_V: [BS, P, D_em]`
- `cand_score: [BS, P]`

At span boundary, select top C candidates per stream:
- `cand_idx = topk(cand_score, C)` → `[BS, C]`
- `K_C, V_C, score_C` → `[BS, C, D_em]`, `[BS, C, D_em]`, `[BS, C]`

#### 4.3.4 EM Write Update (soft blended EMA, like PM)
We update EM with "soft multi-slot commits" inspired by PM.

Inputs at boundary:
- candidate set `(K_C, V_C)` and scores
- write strength `g_em: [BS]` (in [g_em_floor, g_em_ceil] when `em_enabled=True` (Phase B+); fixed 0.3 when `em_enabled=False`)
- softmax temperature `tau: [BS]` (learned in Phase B+; fixed `tau_em` in Phase A)
- weakness weight `ww: [BS]` (learned in Phase B+; fixed `weakness_weight_em` in Phase A)
- decay rate `decay: [BS]` (learned in Phase B+; fixed `decay_em` in Phase A)
- config: `k_write` slots to update per candidate

For each candidate `c` (C is small, e.g. 8), do:
1) `scores_slot = em_K @ K_C[:, c]` → `[BS, M]`
2) weakness bias: `scores_slot += -weakness_weight * em_S`
3) `w = softmax(scores_slot / τ_em)` → `[BS, M]`
4) sparsify top-k: keep only top `k_write`, renormalize
5) EMA update:
   - `α = w * g_em` → `[BS, M]`
   - `em_K = normalize((1-α)[...,None]*em_K + α[...,None]*K_C[:,c,None,:])`
   - `em_V = (1-α)[...,None]*em_V + α[...,None]*V_C[:,c,None,:]`
6) Strength update:
   - `em_S = clamp(em_S + α * f(score_C[:,c]), 0, S_max)`
7) Apply budget / decay:
   - per-stream learned decay each span: `em_S *= decay` (decay is per-stream from EMNeuromodulator, in [decay_em_floor, decay_em_ceil])
   - enforce `sum(em_S, dim=-1) <= budget_em` via scaling (per-stream)

**Batch-friendly:** everything is `[BS, ...]` and uses `gather`, `topk`, broadcasting.

---

### 4.4 Procedural Memory (PM) — fast low-rank weights + eligibility
PM is essentially v1.7 fast memory, with one additional constraint: **K/V/a are frozen within a span** and updated only at span boundaries.

#### 4.4.1 PM State (per layer per block)
Per block b and layer ℓ (B × L instances total):
- `pm_K: [BS, r, D_h]` (unit-normalized)
- `pm_V: [BS, r, D_h]` (unit-normalized)
- `pm_a: [BS, r]` (bounded strengths, gradient-free)
- `elig_K: [BS, r, D_h]` (differentiable within TBPTT)
- `elig_V: [BS, r, D_h]` (differentiable within TBPTT)
- `h: [BS, D_h]` (core recurrent state)

#### 4.4.2 PM Apply (read-only within span)
Same as v1.7 linear lookup, followed by a post-readout FFN:
- `x_q = normalize(x_block)` → `[BS, D_h]`
- `scores = pm_K @ x_q` → `[BS, r]`
- `y_raw = (pm_a * scores) @ pm_V` → `[BS, D_h]`
- `y_pm = y_raw + FFN(LayerNorm(y_raw))` → `[BS, D_h]` (pre-norm residual, GELU, 4× expansion)

The readout FFN adds nonlinear processing capacity to the linear key-value lookup, allowing the model to transform retrieved procedural knowledge before injecting it into the recurrence. Controlled by `pm_readout_ffn` (default `True`).

#### 4.4.3 Eligibility Update (differentiable, neo-Hebbian)
Follows the **neo-Hebbian three-factor learning rule**: `ΔW ∝ pre × post × neuromodulator`.

- `k_cand = normalize(W_k_pre(x))` — pre-synaptic (input-side)
- `v_cand = W_v_post(h)` — post-synaptic (state-side)
- `elig_K = ρ * elig_K + k_cand` (differentiable)
- `elig_V = ρ * elig_V + v_cand` (differentiable)
- candidates computed with gradients through projection layers
- The implicit outer product `v_cand ⊗ k_cand` is the Hebbian update; the neuromodulator (third factor) gates whether it is committed at span boundaries

#### 4.4.4 PM Commit at span boundary (soft top-k EMA)
Use v1.7 soft commit almost unchanged, but triggered only at span boundaries:
- `PMNeuromodulator` outputs: `commit_mask` detached (heuristic: `elig_norm > 1.0`); `lambda_vals`, `g`, `slot_logits` retain grad when continuous heads exist (Phase B+)
- Eligibility (`elig_K`, `elig_V`) not detached (gradients flow to projections)
- per-stream `commit_mask` controls which streams update
- decay/write applied only for commit streams

### 4.11 Spatial Decoder (Hierarchical Aggregation + Deep Cross-Attention)

**Toggle:** `snapshot_enabled` (config flag, default `True`, independent of training phase). When `False`, the model uses the original path: `concat(h_blocks) → lm_head`. When `True`, intermediate layer outputs from all blocks feed through a three-level decoder. Both the sequential and parallel forward paths support the decoder.

**Biological motivation:** NML blocks are analogous to cortical columns — each specializes in different patterns. Without the spatial decoder, only the top layer of each column reaches the LM head. The spatial decoder gives the decoder access to the full "column" of computation, with biologically-inspired hierarchical aggregation:

#### Level 1: Columnar Attention (per block)

Each block has L layers producing intermediate outputs `[BS, D_h]`. A learned summary query iteratively cross-attends to all L outputs (tagged with layer-position embeddings) through `columnar_layers` (default 2) refinement layers, producing a single **column summary** per block.

Each refinement layer: pre-norm cross-attention + pre-norm FFN (4× expansion, GELU).

- Input: `[BS, L, D_h]` (L layer outputs within one block)
- Output: `[BS, D_h]` (column summary)
- One `ColumnarAttention` instance per block (B total)
- Depth: `columnar_layers` attention+FFN layers per block (default 2)
- Analogy: cortical column integrating across its laminar layers

#### Level 2: Thalamic Integrator (across blocks + memory types)

Integrates B column summaries with explicit memory readouts:

| Input token | Shape | Source |
|-------------|-------|--------|
| B column summaries | `[BS, B, d_dec]` | Level 1 output, projected |
| PM readout | `[BS, 1, d_dec]` | Strength-weighted PM slot average, projected |
| EM readout | `[BS, 1, d_dec]` | Strength-weighted EM slot average, projected |
| WM output | `[BS, 1, d_dec]` | WM step output, projected |

Total: B+3 input tokens. Each tagged with type embeddings (cortical/PM/EM/WM) and block-position embeddings (cortical tokens only).

K learned output queries iteratively cross-attend to these B+3 tokens through `thalamic_layers` (default 2) refinement layers, producing K **integrated memory tokens**.

Each refinement layer: pre-norm cross-attention + pre-norm FFN (4× expansion, GELU).

- Input: B+3 tokens at `d_dec`
- Output: `[BS, K, d_dec]` (K = `thalamic_tokens`, default 4)
- Depth: `thalamic_layers` attention+FFN layers (default 2)
- Analogy: thalamus binding cortical regions with memory systems

#### Level 3: Deep Decoder

Standard pre-norm transformer decoder layers. A single query token (projected from `h_final`) cross-attends to the K integrated memory tokens:

```
q = query_proj(h_final)     [BS, 1, d_dec]
for each decoder layer:
    q = self_attn(q) + q
    q = cross_attn(q, memory) + q
    q = FFN(q) + q
context = output_proj(q)     [BS, D]
h_decoded = h_final + context
logits = lm_head(h_decoded)
```

- `decoder_layers` layers (default 2), each with self-attention + cross-attention + FFN
- `output_proj` is **small-initialized** (`std=0.01`) so that at activation time the decoder initially contributes negligible noise (`h_decoded ≈ h_final`). Zero-init would block all gradient flow through the decoder (chain rule: `grad × 0 = 0`), preventing it from learning.
- Analogy: language production area reading from organized memory

#### Dimensions (Tier A)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_dec` | 256 | Decoder working dimension |
| `n_heads_decoder` | 4 | Attention heads (all levels) |
| `decoder_layers` | 2 | Depth of Level 3 decoder |
| `columnar_layers` | 2 | Depth of Level 1 columnar attention |
| `thalamic_layers` | 2 | Depth of Level 2 thalamic integrator |
| `thalamic_tokens` | 4 | Output tokens from Level 2 (K) |

Parameter cost: ~6M new parameters on Tier A (~50M base). Compute: attention over 8, 7, and 4 tokens respectively — negligible per layer. Tier B scales `d_dec=384, n_heads_decoder=6`; Tier C scales `d_dec=512, n_heads_decoder=8, decoder_layers=3`.

#### Gradient flow

The spatial decoder creates a **shortcut gradient path** from the loss to all intermediate layer outputs, bypassing the serial GRU chain. This can significantly help training deep stacks (L=8+) by providing direct gradient signal to early layers.

---

## 5) Core Backbone — Scan-Friendly Recurrence

Replace GRU with an affine recurrence per block:

### 5.1 Block update equation
For each layer ℓ and block b:

**Inputs to block at token t:**
- `x_block_t: [BS, D_h]` (from W_in split)
- `y_pm_t: [BS, D_h]`
- `y_wm_t_proj_to_Dh: [BS, D_h]` (shared WM projected per block)
- `y_em_t_proj_to_Dh: [BS, D_h]` (shared EM projected per block)
- `surprise_t: [BS, 1]`

Fuse:
- `u_t = concat([x_block_t, y_pm_t, y_wm_t_b, y_em_t_b, surprise_t])`

Compute gates (NOTE: **no dependence on h_{t-1}**):
- `a_t = sigmoid(W_a u_t)` → `[BS, D_h]`
- `b_t = tanh(W_b u_t)` → `[BS, D_h]`

Affine recurrence:
- `h_t = a_t ⊙ h_{t-1} + b_t`

This form is scan-friendly (parallel prefix over time) within spans.

### 5.2 Doc-boundary reset integration (scan-friendly)
Let `reset_before_t: [BS]` be True if we must reset state before token t (because token t-1 was EOT, or span starts after EOT).

Implement:
- `carry = (~reset_before_t).float()[:, None]` → `[BS, 1]`
- `h_t = a_t ⊙ (carry * h_{t-1}) + b_t`
- same masking applies to eligibility carry if needed in phases 0–2.

### 5.3 Layer structure (per-layer within block)
Each `Layer` (within a block) computes:
- Input fusion: `u = concat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])` → `[BS, D_in]`
- Gates: `a = sigmoid(W_a u)`, `b = tanh(W_b u)` → `[BS, D_h]`
- Recurrence: `h = a ⊙ (carry * h_prev) + b` → `[BS, D_h]`
- Output projection + residual: `out = norm(W_o(h) + x_block)` → `[BS, D_h]`
- Post-recurrence FFN: `out = out + FFN(LayerNorm(out))` → `[BS, D_h]` (pre-norm residual, GELU, `ffn_expansion`× expansion, default 4×; set `ffn_expansion=0` to disable)

The recurrence mixes temporal information; the FFN adds per-position nonlinear processing depth. Without the FFN, each layer has limited capacity to transform what it retrieves from memory before passing it forward.

`W_o: nn.Linear(D_h, D_h)`, residual, and layernorm are **per-layer** operations. When `snapshot_enabled=False`, the model concatenates all block outputs (`concat(h_blocks)` → `[BS, D]`) and feeds directly to `lm_head`. When `snapshot_enabled=True` (the default), intermediate layer outputs from all blocks feed through the spatial decoder (§4.11) before the LM head.

---

## 6) Neuromodulators (Gating / Neuromodulation)

Each neuromodulator operates in one of two modes, determined by whether its memory system is enabled:
- **Heuristic mode** (memory system disabled): threshold-based decisions with fixed defaults. No MLP heads created, zero parameters.
- **Learned mode** (memory system enabled): MLP backbone + continuous heads, trained via main loss backprop (main optimizer). PM: Phase B+ (when `pm_enabled=True`). EM: Phase B+ (when `em_enabled=True`). All neuromod params stay on the main optimizer in all phases.

### 6.1 PMNeuromodulator (per layer per block, boundary-time)
- One `PMNeuromodulator` per (block, layer) pair — B × L instances total.
- Lives at `layer.pm_neuromodulator`.
- Called only at span boundary with summary stats.

**Inputs (3 features):**

| Feature | Source | Range |
|---------|--------|-------|
| `elig_norm` | `pm.elig_K.norm().mean()` | ~0–5 |
| `pm_usage` | `pm_a.sum() / budget_pm` | [0, 1] |
| `span_surprise` | mean surprise over span | ~0–10 |

**Outputs (5-element tuple):**

| Output | Shape | Heuristic (Phase A) | Learned (Phase B+) | Training |
|--------|-------|--------------------|---------------------------------|----------|
| `commit_mask` | `[BS]` | `elig_norm > 1.0` | `elig_norm > 1.0` (same heuristic) | — |
| `lambda_vals` | `[BS]` | `config.decay_pm` | sigmoid → scaled to `[decay_pm, 1.0]` | Main loss backprop |
| `g` | `[BS]` | `0.5` | sigmoid | Main loss backprop |
| `slot_logits` | `[BS, r]` or `None` | `None` | raw linear | Main loss backprop |
| `p_commit` | `[BS]` or `None` | `None` | `None` | — (reserved) |

Backbone + continuous heads (`lambda_head`, `g_head`, `slot_head`) created when `pm_enabled=True` (Phase B+): `Linear(3, H) → ReLU → {lambda_head, g_head, slot_head}`. `H = config.rl_controller_hidden` (default 32).

### 6.2 EMNeuromodulator (per block, boundary-time)
- One `EMNeuromodulator` per block — B instances total.
- Lives at `block.em_neuromodulator`.
- Called at span boundary for its block's EM bank.

**Inputs (3 features):**

| Feature | Source | Range |
|---------|--------|-------|
| `span_surprise` | mean surprise over span | ~0–10 |
| `em_usage` | `em_S.sum() / budget_em` | [0, 1] |
| `cand_novelty_mean` | mean candidate novelty | ~0–1 |

**Outputs (4-element tuple):**

| Output | Shape | Heuristic (Phase A) | Learned (Phase B+) | Training |
|--------|-------|----------------------|------------------------------|----------|
| `g_em` | `[BS]` | `config.g_em_default` (0.3) | `floor + (ceil - floor) * sigmoid(raw)`, in [g_em_floor, g_em_ceil] | Main loss backprop |
| `tau` | `[BS]` | `config.tau_em` (1.0) | `floor + (ceil - floor) * sigmoid(raw)`, in [tau_em_floor, tau_em_ceil] | Main loss backprop |
| `ww` | `[BS]` | `config.weakness_weight_em` (0.5) | `floor + (ceil - floor) * sigmoid(raw)`, in [ww_em_floor, ww_em_ceil] | Main loss backprop |
| `decay` | `[BS]` | `config.decay_em` (0.999) | `floor + (ceil - floor) * sigmoid(raw)`, in [decay_em_floor, decay_em_ceil] | Main loss backprop |

Backbone + heads created when `em_enabled=True` (Phase B+): `Linear(3+content_proj_dim, H) → ReLU → {g_head, tau_head, ww_head, decay_head}`. All heads use calibrated bias init so that zero input produces the config default. No `write_mask` or `gate_head` — writes always occur; `g_em` near-zero floor enables soft "don't write."

### 6.3 Neuromodulator Training

**All phases (B+):** Continuous outputs (`lambda_vals`, `g`, `slot_logits` for PM; `g_em`, `tau`, `ww`, `decay` for EM) are differentiable through PM/EM operations → trained by main loss backprop on the **main optimizer**. There is no separate RL optimizer — all neuromodulator params are part of the main optimizer in all phases.

**Phase transition checkpoint handling**: Model weights load with `strict=False` (new params init fresh). When transitioning A→B, PM and EM neuromodulator backbones + heads init fresh. Optimizer state loading is skipped when a phase transition is detected (checkpoint's `pm_enabled`/`em_enabled` differ from current config), since parameter group sizes change across phases.

**Ownership is unambiguous:** each block's `EMNeuromodulator` controls exactly one EM bank; each layer's `PMNeuromodulator` controls exactly one PM instance.

---

## 7) Training Loop (Persistent Streams + TBPTT + Spans)

### 7.1 Stream format
Same as v1.7:
- each stream is a long token array with `<|eot|>` separators

### 7.2 Two time scales during training
- TBPTT segment length `T` (e.g. 256): determines autograd truncation.
- Plasticity span length `P` (e.g. 32): determines how often PM/EM can update.

Within a TBPTT segment of length `T`, we do `T/P` spans.

### 7.3 Loss masking
During all phases (`reset_on_doc_boundary=True`):
- Skip loss for positions where `input_tokens[:, t] == eot_id`
- Still train predicting `<|eot|>` as a target inside documents; only skip **EOT → next-doc-first-token** transition.

### 7.4 Online loss accumulation
Do not stack logits. For each token step produce `[BS, vocab]` and accumulate scalar loss.

### 7.5 Per-timestep reset (must be correct)
Reset is triggered before processing token `t` if token `t-1` was EOT (or prev-chunk last token).

Reset affects (phases A–B, `lifelong_mode=False`):
- PM: `pm_K/pm_V/pm_a`, `elig_K/elig_V`, `h` — all zeroed for masked streams
- WM: clear cache validity (`wm_valid`)
- EM: only `em_S=0` for masked streams (keys `em_K` and values `em_V` are preserved; zeroing strengths makes old slots invisible to retrieval while keeping content for future overwrites)

Reset affects (phase D, `lifelong_mode=True` — soft reset):
- `h` — zeroed (short-term context, don't leak across docs)
- `elig_K/elig_V` — zeroed (in-progress learning, stale across docs)
- PM committed state (`pm_K/pm_V/pm_a`) — **persists** (consolidated knowledge)
- EM (`em_K/em_V/em_S`) — **persists** (memories remain retrievable)
- WM: clear cache validity (`wm_valid`) — same as phases A–B

`Block.reset_states()` branches on `config.lifelong_mode`. In lifelong mode, it calls `pm.reset_eligibility(mask)` instead of `pm.reset_states(mask)`, and skips `em.reset_states(mask)` entirely.

### 7.6 TBPTT boundary detachment
After each TBPTT segment:
- detach `h`, `elig_K`, `elig_V`, and any states that can carry gradients
- PM `pm_K/pm_V` may carry gradients within TBPTT due to eligibility → commit; detach them too
- EM tensors are gradient-free state; remain detached throughout

---

## 8) Reference Pseudocode (Authoritative)

### 8.1 Model state container (high level)
- `self.surprise: [BS, 1]`
- WM state: `wm_K`, `wm_V`, `wm_valid`, `wm_ptr` (1 shared WM)
- Per block b (B total):
  - EM state: `em_K[b]`, `em_V[b]`, `em_S[b]`
  - `EMNeuromodulator[b]`
  - Per layer ℓ (L total):
    - PM state: `pm_K[b][ℓ]`, `pm_V[b][ℓ]`, `pm_a[b][ℓ]`
    - Eligibility: `elig_K[b][ℓ]`, `elig_V[b][ℓ]`
    - Recurrent: `h[b][ℓ]`
    - `PMNeuromodulator[b][ℓ]`

### 8.2 Training step over one TBPTT segment
**Inputs:**
- `input_tokens: [BS, T]`
- `target_tokens: [BS, T]`
- `prev_token: [BS]` (token before this segment start, for reset timing)

```python
chunk_loss = 0.0
valid = 0

for span_start in range(0, T, P):
    span_end = min(span_start + P, T)

    # ---- span buffers for EM candidates (per block) ----
    cand_K = [[] for _ in range(B)]
    cand_V = [[] for _ in range(B)]
    cand_score = [[] for _ in range(B)]
    cand_token_valid = [[] for _ in range(B)]      # loss_mask per token
    span_surprise_accum = zeros(BS)
    span_valid_tokens = zeros(BS)                   # per-stream valid count
    span_last_reset = zeros(BS, dtype=long)         # last reset position in span

    for t in range(span_start, span_end):
        # --- per-timestep doc reset mask ---
        if t == 0:
            reset_mask = (prev_token == eot_id)
        else:
            reset_mask = (input_tokens[:, t-1] == eot_id)

        # --- clear span accumulators for streams resetting mid-span ---
        if reset_mask.any() and config.reset_on_doc_boundary:
            local_t = t - span_start
            span_last_reset[reset_mask] = local_t
            span_surprise_accum[reset_mask] = 0
            span_valid_tokens[reset_mask] = 0

        # --- forward one token (embedding + WM + blocks internally) ---
        logits, x_emb, y_wm = model.forward_one_token(
            input_tokens[:, t], reset_mask
        )

        # --- loss masking at cross-doc transition ---
        is_eot = (input_tokens[:, t] == eot_id)
        loss_mask = ~is_eot if config.reset_on_doc_boundary else ones_like(is_eot)
        token_loss, count = online_cross_entropy(logits, target_tokens[:, t], loss_mask)
        chunk_loss += token_loss
        valid += count

        # --- update surprise (teacher forcing during training, masked) ---
        # At inference, pass the sampled token instead of target for
        # self-supervised surprise with identical scale.
        model.update_surprise(logits, target_tokens[:, t], mask=loss_mask)
        span_surprise_accum += model.surprise.squeeze(-1) * loss_mask.float()
        span_valid_tokens += loss_mask.float()

        # --- buffer EM candidate proposals per block ---
        if config.em_enabled:
            for b in range(B):
                h_final = model.blocks[b].layers[-1].h
                k_c, v_c, novelty = model.blocks[b].em.propose_candidate(
                    x_emb, y_wm, h_final, model.surprise
                )
                cand_K[b].append(k_c)
                cand_V[b].append(v_c)
                cand_score[b].append(novelty)
                cand_token_valid[b].append(loss_mask)

    # ---- per-stream span surprise mean ----
    span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)

    # ---- span boundary: PM base decay + commits ----
    if config.pm_enabled:
        for block in model.blocks:
            for layer in block.layers:
                layer.pm.base_decay()
        model.commit_at_boundary(
            span_surprise=span_surprise_mean.detach()
        )

    # ---- span boundary: EM writes ----
    if config.em_enabled:
        for b, block in enumerate(model.blocks):
            stacked_K = stack(cand_K[b], dim=1)            # [BS, P, D_em]
            stacked_V = stack(cand_V[b], dim=1)
            stacked_score = stack(cand_score[b], dim=1)
            stacked_valid = stack(cand_token_valid[b], dim=1)

            # Candidate validity: at/after last reset AND loss-mask valid
            pos = arange(P).unsqueeze(0)                   # [1, P]
            cand_valid = (pos >= span_last_reset.unsqueeze(1)) & stacked_valid.bool()

            g_em, tau, ww, decay = block.em_neuromodulator(
                span_surprise_mean, em_usage, cand_novelty_mean
            )
            block.em.write_at_boundary(
                stacked_K, stacked_V, stacked_score,
                g_em, tau=tau, weakness_weight=ww, decay=decay,
                cand_valid=cand_valid
            )

# finalize loss
chunk_loss = chunk_loss / max(valid, 1)
chunk_loss += compute_regularizers(model)

optimizer.zero_grad()
chunk_loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

model.detach_states()  # TBPTT boundary
```

---

## 9) Phased Training Plan (MVP)

### Phase A — Base competence (WM + scan core, no PM/EM writes)

* Enable WM.
* Disable PM reads/writes (or reads only with a=0).
* Disable EM retrieval/writes.
  Goal: perplexity decreases; stable streaming.

### Phase B — Enable PM + EM with learned neuromodulators

* Enable PM reads always.
* Eligibility accumulates per token.
* Commit every P tokens using heuristic gate (`elig_norm > 1.0`).
* `PMNeuromodulator` creates backbone + continuous heads (`lambda_head`, `g_head`, `slot_head`) → trained via main loss backprop on main optimizer. Heuristic commit gate preserved; learned continuous outputs control commit strength/decay/slot selection.
* Enable EM retrieve (fixed k_ret).
* `EMNeuromodulator` creates backbone + heads (`g_head`, `tau_head`, `ww_head`, `decay_head`) → all trained via main loss backprop.
* `EpisodicMemory.W_nov`: learned novelty adjuster (`nn.Linear(D+D, 1)`) replaces hardcoded 0.5/0.5 weighting; main optimizer.
* All neuromodulator params on main optimizer (no separate RL optimizer).
  Goal: memory bench improvement; explicit recall at long delays; learned write/commit parameters; stable budgets.

### Phase D — Lifelong learning (persistent cross-document memory)

* `config.lifelong_mode = True` (set by `config.set_phase("D")`)
* Soft reset at doc boundaries: h + eligibility traces reset, PM committed state + EM persist
* `reset_on_doc_boundary` remains True (loss masking still active)
* `PM.reset_eligibility(mask)` zeros only `elig_K/elig_V` for masked streams
* Runtime state (PM/EM contents) saved/loaded in checkpoints via `save_runtime_state()`/`load_runtime_state()`
* Natural memory turnover via learned per-stream decay (EM) + base decay (PM) + budget enforcement
* All neuromodulator params remain on main optimizer (same as Phase B)
* Evaluation: domain adaptation, drift monitoring (<5% regression), cross-document recall

---

## 10) Stability Rails (Required)

### PM Rails

* normalize `pm_K`, `pm_V` after updates
* clamp `pm_a` to `[0, a_max]`
* per-stream budget: `sum(pm_a, dim=-1) <= budget_pm`
* base decay: `pm_a *= decay_pm` (per span)
* commit-time decay: `pm_a *= λ` on commit streams only
* commit rate cap (optional for MVP; budget enforcement provides indirect cap)

### EM Rails

* clamp `em_S` to `[0, S_max]`
* per-stream budget: `sum(em_S, dim=-1) <= budget_em`
* weakness bias to prefer overwriting low-strength slots
* optional decay: `em_S *= decay_em` per span

### Monitoring

* perplexity with plasticity OFF (PM/EM reads disabled) must not drift
* log: commit/write rates, memory usage, max strengths, novelty stats per block

---

## 11) Ownership and "Who Controls What?" (Final Answer)

* **PM (Procedural Memory):** B × L instances, each controlled by its own `PMNeuromodulator`. Each PM belongs to one specific (block, layer) pair.
* **EM (Episodic Memory):** B instances, each controlled by its own `EMNeuromodulator`. Each EM belongs to one block and is shared across all L layers in that block.
* **WM (Working Memory):** No neuromodulator (standard attention + cached state, not plastic). One WM shared across the model.

| Component | Count | Neuromodulator | Ownership |
|-----------|-------|----------------|-----------|
| PM | B × L | `PMNeuromodulator[b][ℓ]` (at `layer.pm_neuromodulator`) | Per (block, layer) |
| EM | B | `EMNeuromodulator[b]` (at `block.em_neuromodulator`) | Per block |
| WM | 1 | None | Shared |

This resolves ambiguity by construction.

---

## 12) Implementation Checklist (Files / Classes)

### Core Model

* `NeuromorphicLM`
  * `embedding: nn.Embedding(vocab, D)`
  * `lm_head: nn.Linear(D, vocab, bias=False)` (optionally tied)
  * `wm: WorkingMemory` (1 instance, shared)
  * `blocks: nn.ModuleList[Block]` (B blocks)
  * `spatial_decoder: SpatialDecoder` (if `snapshot_enabled`, else `None`)
  * `surprise: Tensor[BS, 1]` (runtime state)
  * `forward_one_token(input_id, reset_mask) → (logits, x_emb, y_wm)` or `(logits, x_emb, y_wm, stats)` when `collect=True`
  * `update_surprise(logits, target, mask=None)` — computes `-log p(target)`. During training, `target` is ground truth (teacher forcing). During inference, `target` is the sampled token — same formula, identical scale, no calibration needed.
  * `generate(prompt_ids, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0) → [BS, T_prompt + max_new_tokens]` — autoregressive generation with self-supervised surprise (uses `-log p(sampled_token)` for surprise signal). Supports temperature, top-k, and nucleus sampling.
  * `commit_at_boundary(span_surprise=None)` — triggers PM commits with actual span surprise
  * `reset_at_doc_boundary(mask: Tensor[BS])`
  * `detach_states()`
  * State persistence via `state.save_runtime_state(model)` / `state.load_runtime_state(model, state)`

### Spatial Decoder (when `snapshot_enabled=True`)

* `SpatialDecoder`
  * `columnar: nn.ModuleList[ColumnarAttention]` (B instances)
  * `thalamic: ThalamicIntegrator` (cross-block + memory integration)
  * `decoder_blocks: nn.ModuleList[DecoderBlock]` (decoder_layers instances)
  * `output_proj: nn.Linear(d_dec, D, bias=False)` — small-initialized (`std=0.01`)
  * `forward(block_layer_outputs, pm_summary, em_summary, wm_output, h_final) → h_decoded: [BS, D]`

### Block

* `Block` (B instances)
  * `layers: nn.ModuleList[Layer]` (L layers per block)
  * `em: EpisodicMemory` (1 per block)
  * `em_neuromodulator: EMNeuromodulator` (1 per block)
  * `step(..., return_layers=False)` — when `return_layers=True`, also returns `[BS, L, D_h]` stacked layer outputs

### Layer

* `Layer` (B × L instances)
  * `pm: ProceduralMemory`
  * `pm_neuromodulator: PMNeuromodulator`
  * `h: Tensor[BS, D_h]` (recurrent state)
  * `gate_a: nn.Linear`, `gate_b: nn.Linear` (scan-friendly recurrence)

### Working Memory

* `WorkingMemory`
  * `wm_K, wm_V: Tensor[BS, W, D_wm]`
  * `wm_valid: Tensor[BS, W]`, `wm_ptr: Tensor[BS]`
  * `step(x, reset_mask) → y_wm`

### Episodic Memory

* `EpisodicMemory`
  * `em_K, em_V: Tensor[BS, M, D_em]`
  * `em_S: Tensor[BS, M]`
  * `retrieve(x_emb, y_wm) → y_em: [BS, D]` (retrieval + cross-attention aggregation combined)
  * `propose_candidate(x_emb, y_wm, h_final, surprise) → (k_cand, v_cand, novelty)`
  * `write_at_boundary(cand_K, cand_V, cand_score, g_em, tau=..., weakness_weight=..., decay=..., cand_valid=None)` — `cand_valid: [BS, P]` masks out pre-reset and EOT candidates; tau/ww/decay are per-stream `[BS]` tensors from EMNeuromodulator
  * `reset_states(mask)` — overrides StateMixin: only zeros `em_S` (preserves `em_K`, `em_V`)

* `EMNeuromodulator` (B instances)
  * `forward(span_surprise, em_usage, cand_novelty_mean) → (g_em, tau, ww, decay)`
  * Heuristic mode (`em_enabled=False`): fixed defaults from config (g_em=0.3, tau=1.0, ww=0.5, decay=0.999)
  * Learned mode (`em_enabled=True`): MLP backbone + {g_head, tau_head, ww_head, decay_head}, all trained via main loss backprop

### Procedural Memory

* `ProceduralMemory`
  * `pm_K, pm_V: Tensor[BS, r, D_h]`
  * `pm_a: Tensor[BS, r]`
  * `elig_K, elig_V: Tensor[BS, r, D_h]`
  * `apply(x_block) → y_pm`
  * `update_eligibility(x, h)`
  * `base_decay()` — per-span `pm_a *= decay` on ALL streams
  * `reset_eligibility(mask)` — zero only `elig_K/elig_V` for masked streams (Phase D soft reset)
  * `commit(commit_mask, lambda_vals, g, slot_logits)` — soft top-k EMA update for committing streams

* `PMNeuromodulator` (B × L instances)
  * `forward(elig_norm, pm_usage, span_surprise) → (commit_mask, lambda_vals, g, slot_logits, p_commit)`
  * Heuristic mode (`pm_enabled=False`): threshold-based, `slot_logits=None`, `p_commit=None`, zero params
  * Learned mode (`pm_enabled=True`): MLP backbone + lambda/g/slot heads, heuristic gate (main optimizer)

### Training

* `TBPTTTrainer` (in `training/trainer.py`) with:
  * persistent streams (BS is a training config param, not in ModelConfig)
  * per-timestep reset correctness
  * span boundaries for PM base decay + commits + EM writes
  * loss masking at EOT inputs
  * online loss accumulation via `online_cross_entropy()` (in `training/loss.py`)
  * regularization via `compute_regularizers(model)` (free function in `training/loss.py`)
  * All neuromodulator params on the main optimizer in all phases

* `NeuromorphicLM` also provides:
  * `commit_at_boundary(span_surprise)` — triggers PM commits with actual span surprise

---

## 13) Defaults (Config Snippet — Tier B)

```yaml
# === Model Architecture ===
model:
  D: 768                    # model width
  L: 12                     # layers per block
  B: 6                      # parallel blocks
  vocab: 50257              # GPT-2 recommended; vocab_size and eot_id set from chosen tokenizer

# === Working Memory ===
wm:
  W: 512                    # sliding window size
  D_wm: 192                 # WM key/value dimension
  n_heads: 6                # attention heads

# === Episodic Memory (per block) ===
em:
  M: 512                    # capacity per EM bank
  D_em: 192                 # EM key/value dimension
  k_ret: 8                  # retrieval count
  C: 16                     # candidates per span
  k_write: 8                # slots updated per candidate
  τ_em: 1.0                 # softmax temperature
  weakness_weight: 0.5      # bias toward weak slots
  S_max: 3.0                # max strength per slot
  budget: 8.0               # sum(em_S) budget per stream
  decay: 0.999              # per-span strength decay (default; learned in Phase B+)
  decay_floor: 0.99         # fast forgetting (~2.2K token half-life)
  decay_ceil: 0.9999        # near-permanent (~220K token half-life)
  g_em_floor: 0.001         # minimum write strength (near-zero = soft "don't write")
  g_em_ceil: 0.95           # maximum write strength (learned mode)

# === Procedural Memory (per layer per block) ===
pm:
  r: 16                     # slots per PM instance
  ρ: 0.95                   # eligibility decay
  a_max: 3.0                # max strength per slot
  budget: 4.0               # sum(pm_a) budget per stream
  decay: 0.999              # per-span strength decay
  commit_top_k: 2           # slots updated per commit
  τ_pm: 1.0                 # softmax temperature
  weakness_weight: 0.5      # bias toward weak slots

# === Training ===
training:
  BS: 16                    # batch size (persistent streams)
  T: 256                    # TBPTT segment length
  P: 32                     # plasticity span
  precision: bf16           # bf16 for forward/backward, fp32 for state
  reset_on_doc_boundary: true
  lifelong_mode: false        # Phase D: PM/EM persist across doc boundaries
  rl_controller_hidden: 32    # MLP hidden size for neuromodulators
  eot_id: 50256             # GPT-2 <|endoftext|>
  lr: 3.0e-4                # peak learning rate
  lr_min: 1.0e-5            # cosine decay target
  warmup_steps: 1000
  max_grad_norm: 1.0
  weight_decay: 0.01          # applied only to ndim>1 non-bias params; biases and LayerNorm excluded
```

---

## 14) Major Differences vs v1.7

1. **GRU → affine scan-friendly recurrence** (per block):
   * `h_t = a_t ⊙ h_{t-1} + b_t`
   * `a_t, b_t` depend on input features only (enables parallel scan)

2. **Plasticity at span boundaries** (every `P` tokens):
   * PM commits and EM writes are boundary-time events
   * Within spans, PM/EM are read-only → scan-friendly

3. **Episodic Memory (EM) added**:
   * One EM bank per block (B total), each per-stream
   * Fixed `k_ret` latent memory tokens → cross-attention aggregation
   * Per-block `EMNeuromodulator` owns writes

4. **Working Memory (WM) added**:
   * Bounded sliding-window attention (W tokens)
   * No post-training parameter evolution

5. **Naming standardized**:
   * `pm_K/pm_V/pm_a` for PM state, `elig_K/elig_V` for eligibility
   * `em_K/em_V/em_S` for EM state
   * `wm_K/wm_V` for WM cache
   * `PMNeuromodulator` and `EMNeuromodulator` for gating (renamed from `PMController`/`EMController` in Phase D)

6. **Scaling tiers defined** (A/B/C) for 4090 training:
   * Tier B (~150M) recommended for competitive results
   * bf16 mixed precision throughout

7. **Learned neuromodulators** (implemented):
   * Two-mode architecture: heuristic (memory disabled) → learned (memory enabled, main optimizer)
   * Backbone + continuous heads created when memory system enabled (Phase B+)
   * All outputs trained via main loss backprop on the main optimizer
   * PM: commit_mask heuristic, lambda/g/slot learned; EM: g_em/tau/ww/decay all learned
   * No separate RL optimizer — simplifies training and eliminates phase C/D RL complexity

8. **Spatial decoder** (implemented):
   * Three-level hierarchical aggregation: columnar attention (per-block, across L layers) → thalamic integrator (across blocks + memory types) → deep cross-attention decoder
   * `snapshot_enabled` toggle — independent of training phase, `output_proj` small-initialized (`std=0.01`) for near-identity startup with gradient flow
   * ~3M additional params on Tier A (~6%), negligible compute (attention over 8, 7, and 4 tokens)
   * Creates shortcut gradient path from loss to intermediate layer outputs

---

## 15) Inference / Generation

### Surprise at inference

During training, surprise is teacher-forced: `surprise = -log p(ground_truth_target)`.
During inference, no ground truth exists. The solution is to use the model's own sampled
token: `surprise = -log p(sampled_token)`.

This works because:
- **Identical formula**: same `-log p(token)`, just a different token source.
- **Identical scale**: no calibration or normalization needed.
- **Semantically correct**: when the model is confident (picks a high-probability token),
  surprise is low. When it's uncertain, even the best token has moderate `-log p`, so
  surprise is appropriately elevated.

The `generate` method handles this automatically — after each sampled token, it calls
`update_surprise(logits, sampled_token)` to keep the surprise signal flowing through
all memory systems (PM eligibility gating, EM novelty scoring, recurrence gates).

### Prompt processing

During prompt processing, ground-truth next tokens are available (the rest of the prompt),
so `generate` uses teacher-forced surprise for prompt tokens: `update_surprise(logits, prompt[t+1])`.
Only the last prompt token and all generated tokens use self-supervised surprise.

### Memory during generation

All memory systems remain active during generation:
- **WM**: ring buffer continues accumulating context
- **PM**: eligibility traces accumulate (gated by surprise), but no span-boundary commits occur
  during generation (no explicit boundary triggers). PM reads still work.
- **EM**: retrieval works normally. No EM writes during generation (no boundary triggers).
- **Surprise**: self-supervised via `-log p(sampled_token)`, feeds into all gates and neuromodulators.

---

## End of v2 Spec
