# Neuromorphic LM -- Parallel Span Forward Pass Explainer

**Purpose:** Documents the scan-friendly parallel training path (`forward_span`), its relationship to the sequential path (`forward_one_token`), and every place where the two paths differ in logic or math. Read `MODEL_EXPLAINER.md` first for full architecture context; this document focuses on what changes.

**Last verified against code:** 2026-02-25

---

## Table of Contents

1. [Why a Parallel Path](#1-why-a-parallel-path)
2. [Architecture Comparison](#2-architecture-comparison)
3. [Span-Level Dataflow](#3-span-level-dataflow)
4. [Concrete Walkthrough: One Span, Token by Token](#4-concrete-walkthrough-one-span-token-by-token)
5. [Component-by-Component: Sequential vs Parallel](#5-component-by-component)
6. [The Affine Scan](#6-the-affine-scan)
7. [Surprise Freezing](#7-surprise-freezing)
8. [Doc-Boundary Resets in Parallel](#8-doc-boundary-resets-in-parallel)
9. [Post-Forward: EM Candidates, Loss](#9-post-forward)
10. [Training Loop Integration](#10-training-loop-integration)
11. [Exact Semantic Differences](#11-exact-semantic-differences)
12. [What Is NOT Supported](#12-what-is-not-supported)
13. [File Index](#13-file-index)

---

## 1. Why a Parallel Path

The neuromorphic LM's affine recurrence (`h_t = a_t * h_{t-1} + b_t`) was designed to be scan-friendly: gates `a` and `b` depend only on input features, never on `h_{t-1}`. Despite this, the original implementation processed tokens one at a time in a Python for-loop. This meant:

- Every token launched separate CUDA kernels for gate projections, PM reads, EM retrieval, FFN, etc.
- GPU utilization was poor: each kernel operated on `[BS, D_h]` tensors (e.g. `[32, 384]` = 12K elements), far below the threshold for saturating GPU compute.
- Training speed was ~250 tok/s on a 4090 for Tier A — viable for debugging, too slow for serious experiments.

The parallel path batches P=32 tokens (one plasticity span) into `[BS, P, ...]` tensors. The expensive operations (linear projections, attention, FFN) now operate on 32x more elements per kernel launch, while the recurrence itself remains a cheap sequential loop. Combined with `torch.compile` (§12), this yields ~100x throughput increase over the original sequential path (~26K tok/s on a 4090 for Tier A Wide).

**Design constraint:** The parallel path must produce identical (or near-identical) results to the sequential path. Both paths share the same `nn.Module` parameters. The sequential path remains available for inference and as a correctness oracle.

---

## 2. Architecture Comparison

This document covers the full architecture (all subsystems). In earlier phases some components are absent (see `MODEL_EXPLAINER.md` §17). The scan-friendly parallel path is used for all forward passes during training; the sequential path is retained for inference.

```
SEQUENTIAL (forward_one_token, called T times):
  for t in 0..T-1:
    reset → embed → WM.step → Block.step(layer loop) → lm_head → loss → surprise

PARALLEL (forward_span, called T/P times):
  for span in 0..T/P-1:
    embed_all [BS,P,D]
    → WM.forward_span [BS,P,D]
    → Block.forward_span(batched layers + scan) [BS,P,D_h]
    → spatial_decoder([BS*P,...]) → reshape [BS,P,D] (if snapshot_enabled)
    → lm_head_all [BS,P,vocab]
    → batched_loss
    → per-token surprise (post-forward)
    → EM candidate proposal (post-forward)
    → span boundary: neuromodulators decide PM commits + EM writes

  After all spans:
    → optimizer.step() (single main optimizer for all params)
```

The parallel path processes an entire span in one `forward_span()` call (which includes inline PM eligibility updates), then handles surprise and EM candidates as post-forward steps in the trainer.

---

## 3. Span-Level Dataflow

`NeuromorphicLM.forward_span(input_ids, reset_mask_first)` does:

```python
# 1. Compute per-token reset masks from input_ids
reset_mask_all = _compute_reset_masks(input_ids, reset_mask_first)  # [BS, P]
carry_all = (~reset_mask_all).float().unsqueeze(-1)                 # [BS, P, 1]

# 2. Handle first-token resets (full state reset)
if reset_mask_all[:, 0].any():
    self.reset_at_doc_boundary(reset_mask_all[:, 0])

# 3. Freeze surprise for this span
surprise_span = self.surprise   # [BS, 1] — same value for all P tokens

# 4. Embed all tokens
x_emb_all = self.embedding(input_ids)           # [BS, P, D]

# 5. Working memory (batched span path)
y_wm_all = self.wm.forward_span(x_emb_all, reset_mask_all)  # [BS, P, D]

# 6. Project and split across blocks
x_proj_all = self.W_in(x_emb_all)                            # [BS, P, D]
x_blocks_all = x_proj_all.view(BS, P, B, D_h)                # [BS, P, B, D_h]

# 7. Process each block (batched layers + scan)
for b, block in enumerate(self.blocks):
    h_b = block.forward_span(x_blocks_all[:,:,b], y_wm_all, x_proj_all,
                             surprise_span, carry_all)        # [BS, P, D_h]

# 8. Merge block outputs
h_final = concat(h_blocks, dim=-1)                            # [BS, P, D]

# 9. Spatial decoder or direct LM head
if snapshot_enabled:
    block_layer_flat = [reshape to [BS*P, L, D_h]]            # from cached _last_layer_stack
    pm_flat = pm_summary expanded to [BS*P, D_h]
    em_flat = em_summary expanded to [BS*P, D_em]
    wm_flat, h_flat = reshape to [BS*P, D]
    h_decoded = spatial_decoder(block_layer_flat, pm_flat,
                                em_flat, wm_flat, h_flat)     # [BS*P, D]
    logits_all = lm_head(h_decoded.reshape(BS, P, D))         # [BS, P, vocab]
else:
    logits_all = lm_head(h_final)                             # [BS, P, vocab]

return logits_all, x_emb_all, y_wm_all
```

**Key differences from sequential path visible here:**
- `surprise_span` is frozen (§7)
- `reset_at_doc_boundary` only called for first token (§8)
- All embeddings, projections, and logits computed in batch

---

## 4. Concrete Walkthrough: One Span, Token by Token

Trace the full lifecycle of one span (P=32 tokens) on Tier A wide (D=768, B=2, L=8, D_h=384, D_wm=192, D_em=128, r=8, M=256, k_ret=4). BS=32 streams. Assume Phase B (WM + PM + EM enabled). Assume a doc boundary (EOT) occurs at position t=12 within the span.

### 4.1 State entering this span

```
model.surprise:  [32, 1]    — mean surprise from previous span (e.g. 3.2 for each stream)
Layer.h:         [32, 384]  — per layer, per block (16 instances total). Carries forward from last span.
WM state:       wm_K [32, 128, 192], wm_V [32, 128, 192]  — sliding-window ring buffer (W=128 recent tokens)
PM state:        pm_K [32, 8, 384], pm_V [32, 8, 384], pm_a [32, 8]  — per layer per block (16 instances)
PM eligibility:  elig_K [32, 8, 384], elig_V [32, 8, 384]            — per layer per block (16 instances)
EM state:        em_K [32, 256, 128], em_V [32, 256, 128], em_S [32, 256], em_age [32, 256]  — per block (2 instances)
```

### 4.2 Phase 1: Forward pass (model.forward_span)

**Step 1 — Reset masks and carry:**
```
input_ids:       [32, 32]   — 32 streams × 32 tokens
reset_mask_all:  [32, 32]   — True at t=13 for streams where input_ids[:,12]==EOT
carry_all:       [32, 32, 1] — 0.0 at t=13 for those streams, 1.0 elsewhere
```
Token 13 gets the reset (because token 12 was EOT). `reset_at_doc_boundary` is called only if token 0 has a reset.

**Step 2 — Freeze surprise:**
```
surprise_span = model.surprise    # [32, 1], value ≈ 3.2 for all streams
```
This single value will be broadcast to all 32 tokens in every layer's gate computation.

**Step 3 — Embed all tokens:**
```
x_emb_all = embedding(input_ids)  # [32, 32, 768]
```
One kernel launch. 64× more work per launch than the sequential path.

**Step 4 — Working memory (softmax sliding-window attention over span):**
```
q_all = W_q(x_emb_all)    # [32, 32, 192]  — one matmul
k_all = W_k(x_emb_all)    # [32, 32, 192]  — one matmul
v_all = W_v(x_emb_all)    # [32, 32, 192]  — one matmul
```
`forward_span()` runs softmax attention over the W=128 sliding window for P=32 tokens:
- Per token: write k_t, v_t to ring buffer, attend over valid positions with ALiBi recency bias
- Mid-span resets: at doc boundaries (e.g. t=13), ring buffer is zeroed before that token
```
y_wm_all:  [32, 32, 768]  — WM output for all tokens
```

**Step 5 — Project and split across blocks:**
```
x_proj_all = W_in(x_emb_all)                          # [32, 32, 768]  — one matmul
x_blocks_all = x_proj_all.view(32, 64, 2, 384)        # [32, 32, 2, 384]
```

**Step 6 — Process each block (b=0..1):**

For block 0:
```
x_block_all = x_blocks_all[:, :, 0]                    # [32, 32, 384]

# EM retrieval (frozen em_K, em_V, em_S, em_age)
y_em_all = em.retrieve_batch(x_emb_all, y_wm_all)     # [32, 32, 768]

# Project shared signals to per-block dim
y_wm_proj_all = W_wm_proj(y_wm_all)                   # [32, 32, 384]  — one matmul
y_em_proj_all = W_em_proj(y_em_all)                    # [32, 32, 384]  — one matmul
```

Then for each layer (l=0..7):
```
  # PM read (frozen pm_K, pm_V, pm_a)
  y_pm_all = pm.apply_batch(x)                         # [32, 32, 384]  — batched einsum

  # Fuse all inputs + frozen surprise
  u = cat([x, y_pm_all, y_wm_proj_all, y_em_proj_all, surprise_all])
                                                        # [32, 32, 1537]

  # Batched gate computation (two matmuls instead of 64×2)
  a = sigmoid(gate_a(u))                                # [32, 32, 384]
  b = tanh(gate_b(u))                                   # [32, 32, 384]

  # Apply carry mask: at t=13 for reset streams, a_eff becomes 0
  a_eff = a * carry_all                                 # [32, 32, 384]

  # Affine scan: h_t = a_eff_t * h_{t-1} + b_t
  # At t=13 for reset streams: h_13 = 0 * h_12 + b_13 = b_13 (fresh start)
  h_all = parallel_affine_scan(a_eff, b, self.h)        # [32, 32, 384]
  self.h = h_all[:, -1]                                  # update state to token 63

  # Output projection + residual + norm + FFN (all batched)
  output = norm(W_o(h_all) + x)                         # [32, 32, 384]
  output = output + ffn(ffn_norm(output))               # [32, 32, 384]

  # Cache for EM candidate proposal (only final layer retained)
  self._last_h_all = output                              # [32, 32, 384]

  # Inline PM eligibility update (x_in = pre-layer input, output = h_out)
  pm.update_eligibility_batch(x_in, output, elig_surprise, reset_mask)

  x = output  # next layer's input
```

Block 0 returns `x: [32, 32, 384]`. Repeat for block 1.

**Step 7 — Merge and predict (with spatial decoder):**
```
h_final = cat([h_b0, h_b1], dim=-1)                    # [32, 32, 768]

# Spatial decoder (snapshot_enabled=True, the default):
# Collect cached per-layer outputs from each block
block_layer_outputs = [block._last_layer_stack for block in blocks]
                                                        # 2 × [32, 32, 8, 384]
# PM/EM summaries frozen within span
pm_summary = _compute_pm_summary(32, device)            # [32, 384]
em_summary = _compute_em_summary(32, device)            # [32, 128]

# Reshape everything to [BS*P, ...] = [2048, ...]
block_layer_flat = [blo.reshape(2048, 8, 384) for blo in block_layer_outputs]
pm_flat = pm_summary.unsqueeze(1).expand(32, 64, 384).reshape(2048, 384)
em_flat = em_summary.unsqueeze(1).expand(32, 64, 128).reshape(2048, 128)
wm_flat = y_wm_all.reshape(2048, 768)
h_flat = h_final.reshape(2048, 768)

# Run decoder on all 2048 positions at once
h_decoded = spatial_decoder(block_layer_flat, pm_flat,
                            em_flat, wm_flat, h_flat)   # [2048, 768]
logits_all = lm_head(h_decoded.reshape(32, 64, 768))   # [32, 32, 32000]
```

`forward_span` returns `(logits_all, x_emb_all, y_wm_all)`.

### 4.3 Phase 2: Post-forward in trainer

**Step 8 — Per-token surprise (no grad):**
```
logp = log_softmax(logits_all)                          # [32, 32, 32000]
token_surprise = -logp.gather(-1, targets)              # [32, 32, 1]
token_surprise *= loss_mask                             # zero at EOT positions
```

**Step 9 — Batched loss:**
```
span_loss = cross_entropy(logits_all.reshape(2048, 32000),
                          targets.reshape(2048),
                          mask.reshape(2048))           # scalar
```

**Step 10 — PM eligibility (handled inline in Block.forward_span, not post-forward):**

PM eligibility is now updated inline within `Block.forward_span`, immediately after each layer. This matches the sequential path's `Block.step()` where `update_eligibility` follows each layer. The inline approach eliminates a second pass over activations and allows intermediate `_last_h_all` to be freed (only the final layer's is retained for EM candidates).

```
  # Inside Block.forward_span, after each layer:
  # x_in = pre-layer input, x = post-layer output (h_out)
  pm.update_eligibility_batch(x_in, x, elig_surprise, reset_mask)
    └── elig_surprise = PCM's ‖δ‖/√D_pc [BS, P, 1] (or scalar surprise if PCM disabled)
    └── reset_mask = ~carry_all.squeeze(-1) [BS, P] — carry=0 at doc boundaries
```

Inside `update_eligibility_batch`, the same batched projections + fused K+V affine scan as before:
```
  k_cand_all = normalize(W_k_pre(x_in))                                # [32, 32, 384]
  v_cand_all = W_v_post(x_in * x)                                      # [32, 32, 384] — Hebbian
  gate = (elig_surprise / 5.0).clamp(0, 1)                             # [32, 32, 1, 1]
  a = rho * carry  (0 at resets)                                        # [32, 32, 8, 384]
  elig_KV_all = parallel_affine_scan(a_KV, b_KV, h_KV_init)           # fused K+V scan
  elig_K, elig_V = elig_KV_all.chunk(2, dim=-1)                        # update state
```

At t=13 for reset streams: carry=0 so `elig = 0 * elig_prev + gate * cand` (fresh start).

**Step 11 — EM candidate proposal (batched, post-forward):**

For each block b=0..1:
```
  h_final_all = block.layers[-1]._last_h_all            # [32, 32, 384]
  k_cand = normalize(W_k_cand(cat(x_emb_all, y_wm_all)))  # [32, 32, 128]
  v_cand = W_v_cand(h_final_all)                            # [32, 32, 128]
  novelty = blend(token_surprise, 1 - max_sim(k_cand, em_K))  # [32, 32]
```

Candidates are buffered. Pre-reset candidates (t < 13 for reset streams) are masked invalid later.

Note: mid-span resets can affect at most `P-1` tokens per boundary event (31 with the current default `P=32`).

### 4.4 Phase 3: Span boundary (same as sequential path)

```
span_surprise_mean = span_surprise_accum / span_valid_tokens   # [32]
model.surprise = span_surprise_mean.unsqueeze(-1)               # [32, 1] — for next span

# PM: base decay + neuromodulator commit decision
for each PM instance:
    pm_a *= 0.999                                               # decay
    neuromodulator decides p_commit, g, lambda, slot_logits, tau
    pm.commit(p_commit, lambda, g, slot_logits, tau)            # EMA update pm_K, pm_V, pm_a
    soft eligibility reset via p_commit

# EM: neuromodulator write decision
for each EM instance:
    neuromodulator decides g_em, tau, ww, decay
    em.age_tick(P)                                                                    # age active slots by P tokens
    em.write_at_boundary(cand_K, cand_V, scores, g_em, tau=tau, ww=ww, decay=decay)   # EMA update em_K, em_V, em_S, em_age
    em_S *= decay; budget_enforce(em_S)                         # per-stream learned decay + budget

# PCM: update hypothesis
for each PCM instance:
    z_mean = block._last_z.mean(dim=1)                          # [BS, D_pc] — span-mean evidence
    L_pred = MSE(z_hat, z_mean.detach())                        # train predictor from old hypothesis
    z_hat = predictor(cat([z_mean, ctx_b, pm_summary, em_summary]).detach())  # new hypothesis
```

State is now ready for the next span. The cycle repeats.

### 4.5 Where the speedup comes from

In the sequential path, steps 3-7 would each be called 32 times with `[32, ...]` tensors. In the parallel path, they're called once with `[32, 32, ...]` tensors. The key operations and their kernel launch counts:

| Operation | Sequential (32 tokens) | Parallel (1 span) |
|-----------|----------------------|-------------------|
| Embedding lookup | 32 launches | 1 launch |
| WM Q/K/V projection | 32 × 3 = 96 launches | 3 launches |
| WM attention (sequential either way) | 32 launches | 32 launches |
| W_in projection | 32 launches | 1 launch |
| Per-block EM retrieval | 32 × 2 = 64 launches | 2 launches |
| Per-block WM/EM projection | 32 × 2 × 2 = 128 launches | 2 × 2 = 4 launches |
| Per-layer PM read | 32 × 16 = 512 launches | 16 launches |
| Per-layer gate_a + gate_b | 32 × 16 × 2 = 1024 launches | 16 × 2 = 32 launches |
| Per-layer scan (sequential either way) | 32 × 16 = 512 element-wise | 16 × 32 = 512 element-wise |
| Per-layer W_o + FFN | 32 × 16 × 3 = 1536 launches | 16 × 3 = 48 launches |
| LM head | 32 launches | 1 launch |
| PM eligibility projections (inline) | 32 × 16 × 2 = 1024 launches | 16 × 2 = 32 launches |
| PM eligibility scan (inline, fused K+V) | 32 × 16 × 2 = 1024 element-wise | 16 × 32 = 512 element-wise |
| **Total kernel launches** | **~12,500** | **~230 + 64 seq** |

Each parallel launch does 64× more work, saturating GPU compute. The scan loops (layer recurrence + eligibility) are the same cost either way but trivial compared to the matmuls. `W_in` is computed once and shared across all blocks for both the forward pass and eligibility.

With `torch.compile` (§12), the scan loops, gate computations, and elementwise ops within each compiled region are further fused into a small number of optimized CUDA kernels, and the backward pass replaces ~200K autograd nodes with fused backward kernels.

---

## 5. Component-by-Component: Sequential vs Parallel

### 5.1 Embedding + LM Head + Spatial Decoder

**Sequential:** `embedding(input_id)` → `[BS, D]`, optionally `spatial_decoder(...)` → `[BS, D]`, then `lm_head(h)` → `[BS, vocab]`

**Parallel:** `embedding(input_ids)` → `[BS, P, D]`, optionally `spatial_decoder(...)` on `[BS*P, ...]` → `[BS*P, D]` reshaped to `[BS, P, D]`, then `lm_head(h)` → `[BS, P, vocab]`

**Difference:** None. When `snapshot_enabled=True` (the default), both paths run the spatial decoder. The parallel path reshapes `BS*P` into the batch dimension, runs the decoder once on all positions, and reshapes back. PM/EM summaries are frozen within a span (computed once and expanded), matching the sequential path where they don't change within a span either.

### 5.2 Working Memory

**File:** `src/model/working_memory.py` — `WorkingMemory.forward_span()`

**Sequential** (`step()`): Projects Q/K/V for one token, writes to ring buffer, attends over valid positions with ALiBi bias.

**Parallel** (`forward_span()`): Projects Q/K/V for all P tokens in one batched matmul, then runs a sequential attention loop over the P tokens (each token writes to the ring buffer, then attends). Mid-span doc-boundary resets zero the ring buffer before the reset token. torch.compile fuses the loop.

**Difference:** None in output/state vs sequential. Both paths produce identical results — the ring buffer update is sequential, but batched projections make it efficient.

### 5.3 Procedural Memory Read

**File:** `src/model/procedural_memory.py` — `apply_batch()`

**Sequential** (`apply()`): `[BS, D_h]` input → scores against `pm_K` → holographic modulation (input × values) → `[BS, D_h]`

**Parallel** (`apply_batch()`): Same holographic read, batched over P:
```python
x_q = unit_normalize(x_block_all)                                        # [BS, P, D_h]
scores = torch.matmul(x_q, self.pm_K.transpose(-1, -2))                  # [BS, P, r]
weighted = (self.pm_a.unsqueeze(1) * scores).unsqueeze(-1)               # [BS, P, r, 1]
y_pm = (weighted * x_q.unsqueeze(2) * self.pm_V.unsqueeze(1)).sum(2)     # [BS, P, D_h]
```

The input flows through stored modulation patterns: `y_d = x_d * sum_i(a_i * score_i * v_{i,d})`. This is quadratic in x — each slot applies an input-dependent transformation rather than returning a fixed vector.

**Difference:** None — PM state (`pm_K`, `pm_V`, `pm_a`) is frozen within a span in both paths.

### 5.4 Episodic Memory Retrieval

**File:** `src/model/episodic_memory.py` — `retrieve_batch()`

**Sequential** (`retrieve()`): `[BS, D]` query → top-k scoring against `em_K` → cross-attention over top-k values → `[BS, D]`

**Parallel** (`retrieve_batch()`): Same math, batched over P. The top-k and cross-attention operate on `[BS, P, ...]` tensors:
```python
q = unit_normalize(W_q_em(cat([x_all, y_wm_all], dim=-1)))   # [BS, P, D_em]
scores = einsum("bpd, bmd -> bpm", q, em_K)                   # [BS, P, M]
scores += age_gate * log1p(em_age).unsqueeze(1)                # temporal bias [BS, 1, M]
topk_scores, topk_idx = scores.topk(k_ret, dim=-1)            # [BS, P, k]
# ... cross-attention ...
y_em_all = W_o_cross(out)                                      # [BS, P, D]
```

**Difference:** None in math. EM state (`em_K`, `em_V`, `em_S`, `em_age`) is frozen within a span in both paths. The temporal age bias (`age_gate * log1p(em_age)`) is broadcast over P positions identically in both paths. Implementation uses `em_V.unsqueeze(1).expand(-1, P, -1, -1)` for the gather, which increases peak memory by `P * M * D_em` per block.

### 5.5 Layer (Core Recurrence)

**File:** `src/model/layer.py` — `forward_span()`

**Sequential** (`step()`):
```python
u = cat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])  # [BS, 4*D_h+S]
# S = D_pc if pcm_enabled (vector surprise δ), else 1 (scalar surprise)
a = sigmoid(gate_a(u))                                      # [BS, D_h]
b = tanh(gate_b(u))                                         # [BS, D_h]
self.h = a * (carry * self.h) + b                           # scalar recurrence
output = norm(W_o(self.h) + x_block)                        # projection + residual + norm
ffn_input = ffn_norm(output)
if ffn_gain is not None:                                     # PCM-provided gain
    ffn_input = ffn_input * ffn_gain                         # [BS, D_h] *= [BS, D_h]
output = output + ffn(ffn_input)                            # post-recurrence FFN
```

**Parallel** (`forward_span()`):
```python
# surprise_all: [BS, P_b, D_pc] if PCM (vector δ), or [BS, P_b, 1] if scalar
u = cat([x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
         surprise_all], dim=-1)                                     # [BS, P_b, 4*D_h+S]

a = sigmoid(gate_a(u))                                              # [BS, P_b, D_h] — batched
b = tanh(gate_b(u))                                                 # [BS, P_b, D_h] — batched
a_eff = a * carry_all                                               # zero at doc boundaries

h_all = parallel_affine_scan(a_eff, b, self.h)                     # [BS, P_b, D_h] — scan
self.h = h_all[:, -1]                                               # update state

output = norm(W_o(h_all) + x_all)                                  # batched projection + norm
ffn_input = ffn_norm(output)
if ffn_gain_all is not None:                                        # PCM-provided gain
    ffn_input = ffn_input * ffn_gain_all                            # [BS, P_b, D_h]
output = output + ffn(ffn_input)                                   # batched FFN

self._last_h_all = output                                          # cached for post-forward
```

**Key differences:**
1. **Surprise is frozen** — When PCM is disabled, `surprise_span` is the same `[BS, 1]` value broadcast to all P positions (§7). When PCM is enabled, the surprise is the full δ vector computed per-token.
2. **Gate projections batched** — one `gate_a` forward on `[BS, P_b, 4*D_h+S]` instead of P separate calls. This is where most of the speedup comes from.
3. **Recurrence via scan** — `parallel_affine_scan` computes all P_b hidden states given pre-computed `a` and `b` (§6).
4. **`_last_h_all` cached** — The full `[BS, P_b, D_h]` output sequence is stored for EM candidate proposals (only the final layer's is retained; intermediate layers are freed after inline eligibility update).
5. **FFN gain modulation** — When PCM is enabled, `ffn_gain_all = 1 + 0.1 * tanh(W_gain(δ))` modulates FFN input per-dimension, bounded to [0.9, 1.1].

### 5.6 Block

**File:** `src/model/block.py` — `forward_span()`

**Sequential** (`step()`): PCM encode → per layer: PM read → layer step (with δ + FFN gain) → eligibility update → next layer

**Parallel** (`forward_span()`): PCM encode → per layer: PM batch read → layer forward_span (with δ + FFN gain) → **inline eligibility update** → next layer. Eligibility is updated inside the block, matching the sequential path.

```python
def forward_span(self, x_block_all, y_wm_all, x_proj_all,
                 surprise_span, carry_all):
    BS, P, D_h = x_block_all.shape

    # EM retrieval (batched, frozen state, full resolution)
    y_em = self.em.retrieve_batch(x_proj_all, y_wm_all)   # [BS, P, D]

    # Project shared signals to per-block dimension
    y_wm_proj = self.W_wm_proj(y_wm_all)                  # [BS, P, D_h]
    y_em_proj = self.W_em_proj(y_em)                       # [BS, P, D_h]

    # --- PCM: encode evidence, compute surprise, FFN gain ---
    if self.pcm is not None:
        z = self.pcm.encode(x_block_all)                   # [BS, P, D_pc]
        delta = self.pcm.compute_surprise(z)               # [BS, P, D_pc]
        surprise_all = delta                               # vector surprise for gates
        ffn_gain_all = self.pcm.compute_ffn_gain(delta)    # [BS, P, D_h]
        # RMS-normalized scalar surprise for PM eligibility + EM
        elig_surprise = delta.norm(
            dim=-1, keepdim=True) / sqrt(D_pc)             # [BS, P, 1]
        self._last_token_surprise = elig_surprise
    else:
        surprise_all = surprise_span.unsqueeze(1).expand(BS, P, 1)
        ffn_gain_all = None
        elig_surprise = surprise_all                       # [BS, P, 1]

    # Reset mask for eligibility (derived from carry_all)
    reset_mask = ~carry_all.squeeze(-1).bool()             # [BS, P]

    # Pre-LayerNorm for Layer 0
    x = self.input_norm(x_block_all)
    for l_idx, layer in enumerate(self.layers):
        y_pm = layer.pm.apply_batch(x)                     # PM read (batched)
        x_in = x                                           # capture pre-layer input
        x = layer.forward_span(x, y_pm, y_wm_proj,
                               y_em_proj, surprise_all, carry_all,
                               ffn_gain_all=ffn_gain_all)

        # Inline PM eligibility update (matches sequential path)
        if self.config.pm_enabled:
            layer.pm.update_eligibility_batch(
                x_in, x, elig_surprise, reset_mask)

    # Cache per-layer outputs for spatial decoder (already at full P)
    if self.config.snapshot_enabled:
        self._last_layer_stack = torch.stack(
            [layer._last_h_all for layer in self.layers],
            dim=2)                                         # [BS, P, L, D_h]

    # Free intermediate _last_h_all (only final layer needed for EM candidates)
    for layer in self.layers[:-1]:
        layer._last_h_all = None

    return x
```

**Key differences from sequential:**
1. **PCM operates at full resolution** — Evidence encoding, surprise computation, and FFN gain are computed on `[BS, P, ...]` tensors. The scalar surprise is RMS-normalized (`‖δ‖/√D_pc`) to keep scale near ~1.
2. **Eligibility is updated inline** after each layer, matching the sequential path. Uses `update_eligibility_batch` (batched projections + fused K+V affine scan) instead of per-token `update_eligibility` calls.
3. **Intermediate `_last_h_all` freed** — After inline eligibility update, only the final layer's `_last_h_all` is retained (for EM candidate proposals). This reduces peak memory.
4. `_last_layer_stack` is cached on the block for the spatial decoder.

---

## 6. The Affine Scan

**File:** `src/model/scan.py`

Given pre-computed gates `a: [BS, P, D]` and `b: [BS, P, D]` plus initial state `h_init: [BS, D]`, compute the recurrence `h_t = a_t * h_{t-1} + b_t` for all t:

```python
def parallel_affine_scan(a, b, h_init):
    BS, P, D = a.shape
    out_dtype = torch.promote_types(
        torch.promote_types(a.dtype, b.dtype), h_init.dtype
    )
    h_all = torch.empty(BS, P, D, dtype=out_dtype, device=a.device)
    h = h_init
    for t in range(P):
        h = a[:, t] * h + b[:, t]
        h_all[:, t] = h
    return h_all  # [BS, P, D]
```

**This is intentionally a sequential loop.** The scan itself is NOT the bottleneck — P=32 iterations of element-wise ops on `[BS, D_h]` = `[32, 384]` tensors takes microseconds. The speedup comes from batching the expensive operations *around* the scan (gate projections, FFN, attention), and from `torch.compile` (§12) which unrolls the loop and fuses all elementwise ops into optimized CUDA kernels. A true CUDA parallel prefix scan could replace this later as a further optimization.

**Dtype promotion:** Uses `torch.promote_types` on dtype objects (not tensors) to ensure the output dtype accommodates all three inputs. This is compile-safe under `torch.compile(fullgraph=True)` — the older `torch.result_type(tensor, tensor)` approach caused graph breaks.

**Doc boundary handling:** The carry mask is pre-applied before the scan: `a_eff = a * carry_all`. At positions where `carry=0`, the effective gate becomes `a_eff=0`, so `h_t = 0 * h_{t-1} + b_t = b_t` — equivalent to starting from `h=0`.

---

## 7. Surprise Freezing

**The most significant semantic change from sequential to parallel.**

**Sequential path:** Surprise is updated after every token:
```
token 0: layer gates see surprise from previous span's last token
token 1: layer gates see surprise = -log p(target_0)
token 2: layer gates see surprise = -log p(target_1)
...
```

**Parallel path:** Surprise is frozen at the span-initial value for all layer gate computations:
```
token 0..P-1: layer gates ALL see surprise = model.surprise (from previous span)
```

This is frozen at `model.py:197` (`surprise_span = self.surprise`) and broadcast at `layer.py:141`.

**The frozen value is the previous span's mean surprise** — averaged over all valid (non-EOT) tokens in the span. This is more stable than a single token's surprise, which would be a noisy sample (and would be 0 if the last token happened to be EOT).

**Why freezing is acceptable:**
1. Surprise is 1 dimension out of `4*D_h + 1 = 1537` gate input dimensions.
2. The spec §1.4 explicitly designed plasticity boundaries (every P tokens) as the granularity for slowly-varying signals. PM and EM are already frozen within spans; surprise follows the same pattern.
3. Per-token surprise IS still computed post-forward for everything else: eligibility gating, EM candidate scoring, span-mean surprise for boundary controllers.

**Where per-token surprise is used (post-forward in trainer):**
```python
# Computed from logits after forward_span returns
logp = F.log_softmax(logits_all, dim=-1)
token_surprise = -logp.gather(-1, span_targets.unsqueeze(-1))   # [BS, P, 1]

# Used for:
pm.update_eligibility(x_in, h_out, surprise_for_elig[:, t])    # eligibility gating
em.propose_candidate_batch(x, y_wm, h, token_surprise)          # EM novelty
span_surprise_mean = span_surprise_accum / span_valid_tokens    # boundary controllers
model.surprise = span_surprise_mean                              # next span's frozen value
```

---

## 8. Doc-Boundary Resets in Parallel

**Sequential path:** At every token where `reset_mask=True`, calls `reset_at_doc_boundary(mask)` which zeros:
- `model.surprise` for masked streams
- `Layer.h` for all blocks/layers
- `pm_K, pm_V, pm_a, elig_K, elig_V` (non-lifelong) or just `elig_K, elig_V` (lifelong)
- `em_S`, `em_age` (non-lifelong only)
- WM validity and pointer

**Parallel path:** `reset_at_doc_boundary` is called **only for the first token** of the span (`model.py:184-185`). For mid-span doc boundaries (tokens 1..P-1):

| State | How it's handled | Equivalent to sequential? |
|-------|-----------------|--------------------------|
| `Layer.h` | Carry mask zeros `a_eff`, giving `h_t = b_t` | Yes |
| `elig_K, elig_V` | Reset via carry mask in `update_eligibility_batch` (`a=0` at reset positions) | Yes |
| `pm_K, pm_V, pm_a` | NOT zeroed mid-span | **No** (phases A-B) |
| `em_S`, `em_age` | NOT zeroed mid-span | **No** (phases A-C) |
| WM state | Handled internally by `wm.forward_span` | Yes |
| `model.surprise` | Frozen anyway | N/A |

**Impact of the PM/EM gap:** In non-lifelong mode (phases A-B), post-boundary tokens within the same span may read stale PM state and retrieve stale EM memories from the old document. This affects at most `P-1` tokens per boundary event (63 at `P=32`). In lifelong mode (Phase C), PM/EM state intentionally persists across documents, so there is no gap.

**Eligibility reset detail** (`ProceduralMemory.update_eligibility_batch`):
```python
carry = (~reset_mask).float()  # [BS, P]
a = (rho * carry).unsqueeze(-1).unsqueeze(-1)
# ...
elig_all = parallel_affine_scan(a_flat, b_flat, elig_init)
# at reset positions: a=0 => elig_t = b_t (equivalent to explicit zero-then-update)
```

---

## 9. Post-Forward: EM Candidates, Loss

After `forward_span()` returns `logits_all`, `x_emb_all`, `y_wm_all`, the trainer runs several post-forward steps. Note that PM eligibility is now handled **inline** within `Block.forward_span` (see §5.6), not as a post-forward step.

### 9.1 Loss (batched)

```python
span_loss, span_valid = batched_cross_entropy(logits_all, span_targets, loss_mask_all)
```

Reshapes to `[BS*P, vocab]` and computes masked cross-entropy. Numerically identical to the sequential `online_cross_entropy` called P times.

### 9.2 Per-Token Surprise

```python
with torch.no_grad():
    logp = F.log_softmax(logits_all, dim=-1)
    token_surprise = -logp.gather(-1, span_targets.unsqueeze(-1))  # [BS, P, 1]
    token_surprise = token_surprise * loss_mask_all.unsqueeze(-1).float()
    # model.surprise is updated later to span_surprise_mean (see §4.4)
```

This is the real per-token surprise used for everything except layer gates (which saw frozen surprise). The frozen surprise for the *next* span is set to the span mean (averaged over valid tokens), not the last token's value — see §4.4.

### 9.3 PM Eligibility (Inline in Block.forward_span)

PM eligibility is **no longer a post-forward step**. It is updated inline within `Block.forward_span`, immediately after each layer processes its tokens. This matches the sequential path's `Block.step()` where `update_eligibility` follows each layer.

```python
# Inside Block.forward_span:
for l_idx, layer in enumerate(self.layers):
    y_pm = layer.pm.apply_batch(x)
    x_in = x                                           # pre-layer input
    x = layer.forward_span(x, y_pm, ...)               # layer output (h_out)

    # Inline eligibility — same data flow as sequential path
    if self.config.pm_enabled:
        layer.pm.update_eligibility_batch(x_in, x, elig_surprise, reset_mask)
```

**Key details:**
- **Surprise source**: Uses PCM's RMS-normalized `‖δ‖/√D_pc` when PCM is enabled (same surprise that drives gates and EM candidates), or scalar surprise otherwise.
- **Reset mask**: Derived from `carry_all` (`reset_mask = ~carry_all.squeeze(-1).bool()`), same carry mask used for the layer recurrence scan.
- **Memory optimization**: After eligibility update, intermediate layers' `_last_h_all` are freed. Only the final layer's output is retained for EM candidate proposals.
- **Compilation**: `PM._update_eligibility_core` remains separately compiled with `fullgraph=True` — the inline call sites invoke this compiled function.

The K and V eligibility scans are fused into a single double-width scan (concatenated along last dimension). Mathematically equivalent to two separate scans, but halves scan kernel launches.

**Difference from sequential path**: Doc-boundary resets use carry mask (`a=0` at reset positions) instead of explicit `reset_eligibility()` calls — mathematically equivalent (`elig_t = 0 * elig_prev + gate * cand`).

### 9.4 EM Candidate Proposal

```python
for b, block in enumerate(model.blocks):  # b=0..1
    h_final_all = block.layers[-1]._last_h_all  # [BS, P, D_h]
    k_c, v_c, nov = block.em.propose_candidate_batch(
        x_emb_all, y_wm_all, h_final_all, token_surprise,
    )
```

Uses `propose_candidate_batch()` which batches the same math as `propose_candidate()` across P tokens. Uses the full layer output (`_last_h_all`) for value candidates, and real per-token surprise for novelty scoring. EM state (`em_K`, `em_S`, `em_age`) is frozen during scoring.

**Candidate validity:** The trainer tracks `span_last_reset` per stream and masks out candidates from tokens before the last doc-boundary reset in the span. This prevents writing stale cross-document content to EM.

---

## 10. Training Loop Integration

**Files:** `src/training/trainer.py` (orchestration), `src/training/span_ops.py` (shared span-boundary ops used by trainer, validation, and eval_lifelong)

### 10.1 Span Loop Structure (Full Architecture)

The walkthrough below shows the complete flow with all subsystems active (Phase B+). In Phase A, PM/EM neuromodulator steps use heuristic defaults.

```
For each span (8 spans per chunk, P=32 tokens each):
    Reset EM candidate buffers, span accumulators

    # --- Parallel forward (includes inline PM eligibility) ---
    logits_all, x_emb_all, y_wm_all = model.forward_span(span_ids, reset_first)
      └── Embedding → WM.forward_span → Block.forward_span (per block):
            EM.retrieve_batch → per-layer: PM.apply_batch → gates → scan → FFN
              → PM.update_eligibility_batch (inline after each layer)
          → SpatialDecoder (if snapshot_enabled) → LM head → logits [BS,P,vocab]

    span_loss, span_valid = batched_cross_entropy(logits_all, targets, mask)
    chunk_loss += span_loss

    # --- Post-forward (no grad for surprise) ---
    Compute per-token surprise from logits             # [BS, P, 1]
    Compute reset_mask_all for accumulators
    Accumulate span_surprise_mean per stream           # [BS] scalar
    Update model.surprise = span_surprise_mean         # frozen for next span's gates

    # (PM eligibility already updated inline in Block.forward_span)

    EM candidate proposal:                             # per block
      propose_candidate_batch(x_emb, y_wm, h_final, surprise)
        └── key/value candidates + novelty scores      # [BS, P, D_em]

    # --- Span boundary ---
    Compute span_surprise_mean for neuromodulators
    Pre-compute EM candidate stacks + mean novelty

    PM neuromodulators: (elig_norm, usage, surprise) → p_commit, lambda, g, slots, tau
      └── PM.commit(): update pm_K, pm_V, pm_a with EMA

    EM neuromodulators: (surprise, usage, novelty) → g_em, tau, ww, decay
      └── g_em  = floor + range * sigmoid(raw)  [0.001, 0.95]
      └── tau   = floor + range * sigmoid(raw)  [0.05, 5.0]   (slot-softmax temperature)
      └── ww    = floor + range * sigmoid(raw)  [0.0, 2.0]    (weakness weight)
      └── decay = floor + range * sigmoid(raw)  [0.99, 0.9999] (memory retention)
      └── EM.age_tick(P): increment em_age for active slots
      └── EM.write_at_boundary(): update em_K, em_V, em_S, em_age with alpha = g_em * weights

After all spans:
    Finalize loss, add regularizers (PM/EM/WM)
    optimizer.zero_grad()
    total_loss.backward()
      └── gradient flows through: neuromod heads → commit/write → PM/EM state
          → future PM.apply / EM.retrieve → layer outputs → logits → loss

    clip_grad_norm_(all_params)
    optimizer.step() + scheduler.step()

    Detach all states (TBPTT boundary)
```

### 10.2 What Changed vs Sequential Loop

| Aspect | Sequential | Parallel |
|--------|-----------|----------|
| Forward calls per span | P calls to `forward_one_token` | 1 call to `forward_span` |
| Loss computation | P calls to `online_cross_entropy` | 1 call to `batched_cross_entropy` |
| Surprise update | Per-token, between forward calls | Batched post-forward, from logits |
| Eligibility update | Inside `Block.step`, interleaved | Inline in `Block.forward_span`, after each layer |
| EM candidate proposal | Inside trainer token loop | Post-forward batch in trainer |
| Span boundary logic | Unchanged | Unchanged |
---

## 11. Exact Semantic Differences

These are the places where the parallel path produces different numerical results from the sequential path, even with identical inputs and initial state:

### 11.1 Surprise in Layer Gates

**Magnitude:** Small (1/1537 gate input dimensions)

Sequential: token t sees `surprise = -log p(target_{t-1})` (lagged by one token).
Parallel: all tokens see `surprise = mean_surprise(previous_span)` (frozen span-mean).

The frozen value is the mean surprise over all valid tokens in the previous span, which is a smoother signal than the sequential path's single-token lag. For tokens 1..P-1, the sequential path would have updated surprise per-token — the parallel path holds the mean constant.

### 11.2 PM Committed State at Mid-Span Boundaries (Phases A-B)

**Magnitude:** Low-medium. Bounded to at most 31 tokens per boundary.

Sequential: `reset_at_doc_boundary` zeros `pm_K, pm_V, pm_a` → post-boundary tokens read zero PM.
Parallel: PM state not zeroed mid-span → post-boundary tokens read stale PM from old document.

In practice: PM state is slow-changing and typically weak during early training. Self-corrects at next span boundary.

### 11.3 EM Strengths at Mid-Span Boundaries (Phases A-B)

**Magnitude:** Low-medium. Bounded to at most 31 tokens per boundary.

Sequential: `reset_at_doc_boundary` zeros `em_S` and `em_age` → post-boundary tokens retrieve nothing (all slots inactive, ages reset).
Parallel: `em_S` and `em_age` not zeroed mid-span → post-boundary tokens retrieve old-document memories with stale ages.

In practice: EM content from old document is unrelated to new document, but retrieval output passes through learned projections that can dampen irrelevant signals.

### 11.4 Eligibility Surprise Gating

**No difference.** Both paths now use the same surprise source inline: PCM's RMS-normalized `‖δ‖/√D_pc` (when PCM is enabled) or the frozen scalar surprise. The parallel path's eligibility is updated inline in `Block.forward_span` after each layer, matching the sequential path's `Block.step()`. The previous difference (sequential used lagged surprise, parallel used post-forward per-token surprise) no longer exists.

### 11.5 Summary Table

| Difference | Affects | Phases | Max tokens affected | Severity |
|-----------|---------|--------|-------------------|----------|
| Frozen surprise in gates | Gate values `a`, `b` | All | All P tokens in span | Low |
| PM state leak mid-span | PM read output | A-B | ≤31 per boundary | Low-medium |
| EM strength leak mid-span | EM retrieval output | A-B | ≤31 per boundary | Low-medium |
| ~~Eligibility surprise timing~~ | ~~Eligibility trace~~ | — | — | **Eliminated** (inline, same source) |

### 11.6 What Is Identical

Everything not listed above is numerically identical between the two paths:
- Embedding, spatial decoder, and LM head computation
- Working memory GLA recurrence and state
- Recurrent hidden state `h` (carry mask is equivalent to zeroing h)
- PM read math (`apply` vs `apply_batch`)
- EM retrieval math (`retrieve` vs `retrieve_batch`)
- EM candidate proposals and validity masking
- Loss computation
- Span boundary logic (PM commits, EM writes, neuromodulator decisions)
---

## 12. What Is NOT Supported / Compile Support

### torch.compile Support

Block-level compilation is the current strategy: `block.forward_span` is compiled as a single graph per block using `mode="default"`, covering EM retrieval + L layers × (PM apply + gates + scan + FFN) + layer output stacking. This reduces compiled graphs from ~96 (individual functions) to ~4 (one per block in Tier A Wide with B=2). CUDA graph capture (`max-autotune`) was benchmarked but is incompatible with the model's stateful memory writes (PM/EM modify tensors in-place between forward passes) and was 3.5% slower at current scale.

**Compiled:**
- **`Block.forward_span`**: Compiled with `mode="default"`. Covers the entire block forward pass — EM retrieval, per-layer PM reads, gate computation, affine scans, output projections, and FFN — into fused CUDA kernels. The scan loops are unrolled by the compiler, and the backward pass replaces autograd nodes with efficient fused backward kernels.
- **`PM._update_eligibility_core`**: Compiled with `fullgraph=True, mode="default"`. Covers eligibility projections, surprise gating, and fused K+V affine scan. Called inline from `Block.forward_span` after each layer (not as a post-forward step).
- **`GLAWorkingMemory.forward_span`**: Compiled with `mode="default"`. Fuses the GLA recurrence loop.
- **`SpatialDecoder.forward`**: Compiled with `mode="default"`. Fuses hierarchical cross-attention.

**Not compiled at top level:**
- **`NeuromorphicLM.forward_span`**: Not compiled at the model level — compilation is applied at the Block granularity.
- **`WorkingMemory.forward_span`**: Contains data-dependent dispatch for mid-span resets. Runs outside block compilation scope.

**Pre-initialization requirement:** `model.initialize_states(BS, device)` must be called before `compile_for_training()` to pre-allocate all runtime state tensors (Layer.h, PM state, EM state, WM state). This eliminates lazy init guards from the forward path, preventing graph breaks during compilation.

**Usage:** Enable with `--compile` on the training CLI or `use_compile: True` in config. Requires CUDA. First two steps are slow (~5.7 min total) due to tracing; subsequent steps use cached compiled kernels. Provides ~24K tok/s on RTX 4090 (Tier A Wide, BS=32, Phase B).

---

## 13. File Index

| File | Parallel Path Content |
|------|----------------------|
| `src/model/scan.py` | `parallel_affine_scan(a, b, h_init)` — sequential-loop scan |
| `src/model/layer.py` | `Layer.forward_span()` — batched gates + scan + `_last_h_all` cache (final layer only) |
| `src/model/procedural_memory.py` | `ProceduralMemory.apply_batch()` — batched PM read |
| `src/model/episodic_memory.py` | `EpisodicMemory.retrieve_batch()`, `propose_candidate_batch()` |
| `src/model/working_memory.py` | `WorkingMemory.forward_span()` — batched Q/K/V, batched no-reset path, sequential fallback |
| `src/model/block.py` | `Block.forward_span()` — PCM, batched layers + inline PM eligibility, caches `_last_layer_stack` |
| `src/model/predictive_coding.py` | `PredictiveCodingModule` — evidence encoder, hypothesis predictor, surprise/gain computation |
| `src/model/decoder.py` | `SpatialDecoder.forward()` — called with `[BS*P, ...]` tensors in span path |
| `src/model/model.py` | `NeuromorphicLM.forward_span()`, `apply_pcm_boundary()`, `get_pcm_token_surprise()` |
| `src/training/trainer.py` | Span loop orchestration, delegates to span_ops |
| `src/training/span_ops.py` | Shared span-boundary ops: loss masking, surprise, EM candidate accumulation and commits |
| `src/training/loss.py` | `batched_cross_entropy()` — span-level loss computation |
| `tests/test_scan.py` | 28 equivalence tests: parallel vs sequential output, state, gradients |
