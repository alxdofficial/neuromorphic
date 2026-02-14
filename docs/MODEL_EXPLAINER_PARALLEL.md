# Neuromorphic LM -- Parallel Span Forward Pass Explainer

**Purpose:** Documents the scan-friendly parallel training path (`forward_span`), its relationship to the sequential path (`forward_one_token`), and every place where the two paths differ in logic or math. Read `MODEL_EXPLAINER.md` first for full architecture context; this document focuses on what changes.

**Last verified against code:** 2026-02-14

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
9. [Post-Forward: Eligibility, EM Candidates, Loss](#9-post-forward)
10. [Training Loop Integration](#10-training-loop-integration)
11. [Exact Semantic Differences](#11-exact-semantic-differences)
12. [What Is NOT Supported](#12-what-is-not-supported)
13. [File Index](#13-file-index)

---

## 1. Why a Parallel Path

The neuromorphic LM's affine recurrence (`h_t = a_t * h_{t-1} + b_t`) was designed to be scan-friendly: gates `a` and `b` depend only on input features, never on `h_{t-1}`. Despite this, the original implementation processed tokens one at a time in a Python for-loop. This meant:

- Every token launched separate CUDA kernels for gate projections, PM reads, EM retrieval, FFN, etc.
- GPU utilization was poor: each kernel operated on `[BS, D_h]` tensors (e.g. `[32, 128]` = 4K elements), far below the threshold for saturating GPU compute.
- Training speed was ~250 tok/s on a 4090 for Tier A — viable for debugging, too slow for serious experiments.

The parallel path batches P=64 tokens (one plasticity span) into `[BS, P, ...]` tensors. The expensive operations (linear projections, attention, FFN) now operate on 64x more elements per kernel launch, while the recurrence itself remains a cheap sequential loop. Combined with `torch.compile` (§12), this yields ~40x throughput increase over the original sequential path (~10K tok/s on a 4090 for Tier A).

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
    → PM eligibility accumulation (post-forward)
    → EM candidate proposal (post-forward)
    → span boundary: neuromodulators decide PM commits + EM writes

  After all spans:
    → optimizer.step() (single main optimizer for all params)
```

The parallel path processes an entire span in one `forward_span()` call, then handles surprise, eligibility, and EM candidates as post-forward steps in the trainer.

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

# 5. Working memory (batched projections, sequential attention)
y_wm_all = self.wm.forward_span(x_emb_all, reset_mask_all)  # [BS, P, D]

# 6. Project and split across blocks
x_proj_all = self.W_in(x_emb_all)                            # [BS, P, D]
x_blocks_all = x_proj_all.view(BS, P, B, D_h)                # [BS, P, B, D_h]

# 7. Process each block (batched layers + scan)
for b, block in enumerate(self.blocks):
    h_b = block.forward_span(x_blocks_all[:,:,b], y_wm_all, x_emb_all,
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

Trace the full lifecycle of one span (P=64 tokens) on Tier A (D=512, B=4, L=8, D_h=128, D_wm=128, D_em=128, r=8, M=256, W=256, k_ret=4). BS=32 streams. Assume Phase B (WM + PM + EM enabled). Assume a doc boundary (EOT) occurs at position t=12 within the span.

### 4.1 State entering this span

```
model.surprise:  [32, 1]    — mean surprise from previous span (e.g. 3.2 for each stream)
Layer.h:         [32, 128]  — per layer, per block (32 instances total). Carries forward from last span.
WM ring buffer:  wm_K [32, 256, 128], wm_V [32, 256, 128], wm_ptr [32], wm_valid [32, 256]
PM state:        pm_K [32, 8, 128], pm_V [32, 8, 128], pm_a [32, 8]  — per layer per block (32 instances)
PM eligibility:  elig_K [32, 8, 128], elig_V [32, 8, 128]            — per layer per block (32 instances)
EM state:        em_K [32, 256, 128], em_V [32, 256, 128], em_S [32, 256]  — per block (4 instances)
```

### 4.2 Phase 1: Forward pass (model.forward_span)

**Step 1 — Reset masks and carry:**
```
input_ids:       [32, 64]   — 32 streams × 64 tokens
reset_mask_all:  [32, 64]   — True at t=13 for streams where input_ids[:,12]==EOT
carry_all:       [32, 64, 1] — 0.0 at t=13 for those streams, 1.0 elsewhere
```
Token 13 gets the reset (because token 12 was EOT). `reset_at_doc_boundary` is called only if token 0 has a reset.

**Step 2 — Freeze surprise:**
```
surprise_span = model.surprise    # [32, 1], value ≈ 3.2 for all streams
```
This single value will be broadcast to all 32 tokens in every layer's gate computation.

**Step 3 — Embed all tokens:**
```
x_emb_all = embedding(input_ids)  # [32, 64, 512]
```
One kernel launch. 64× more work per launch than the sequential path.

**Step 4 — Working memory (batched projections, sequential attention):**
```
q_all = W_q(x_emb_all)    # [32, 64, 128]  — one matmul
k_all = W_k(x_emb_all)    # [32, 64, 128]  — one matmul
v_all = W_v(x_emb_all)    # [32, 64, 128]  — one matmul
```
Then for each t=0..63: write (k,v) into ring buffer, attend over valid entries, output y_wm.
At t=13: streams with reset clear wm_valid and reset wm_ptr to 0.
```
y_wm_all:  [32, 64, 512]  — WM output for all tokens
```

**Step 5 — Project and split across blocks:**
```
x_proj_all = W_in(x_emb_all)                          # [32, 64, 512]  — one matmul
x_blocks_all = x_proj_all.view(32, 64, 4, 128)        # [32, 64, 4, 128]
```

**Step 6 — Process each block (b=0..3):**

For block 0:
```
x_block_all = x_blocks_all[:, :, 0]                    # [32, 64, 128]

# EM retrieval (frozen em_K, em_V, em_S)
y_em_all = em.retrieve_batch(x_emb_all, y_wm_all)     # [32, 64, 512]

# Project shared signals to per-block dim
y_wm_proj_all = W_wm_proj(y_wm_all)                   # [32, 64, 128]  — one matmul
y_em_proj_all = W_em_proj(y_em_all)                    # [32, 64, 128]  — one matmul
```

Then for each layer (l=0..7):
```
  # PM read (frozen pm_K, pm_V, pm_a)
  y_pm_all = pm.apply_batch(x)                         # [32, 64, 128]  — batched einsum

  # Fuse all inputs + frozen surprise
  u = cat([x, y_pm_all, y_wm_proj_all, y_em_proj_all, surprise_all])
                                                        # [32, 64, 513]

  # Batched gate computation (two matmuls instead of 64×2)
  a = sigmoid(gate_a(u))                                # [32, 64, 128]
  b = tanh(gate_b(u))                                   # [32, 64, 128]

  # Apply carry mask: at t=13 for reset streams, a_eff becomes 0
  a_eff = a * carry_all                                 # [32, 64, 128]

  # Affine scan: h_t = a_eff_t * h_{t-1} + b_t
  # At t=13 for reset streams: h_13 = 0 * h_12 + b_13 = b_13 (fresh start)
  h_all = parallel_affine_scan(a_eff, b, self.h)        # [32, 64, 128]
  self.h = h_all[:, -1]                                  # update state to token 63

  # Output projection + residual + norm + FFN (all batched)
  output = norm(W_o(h_all) + x)                         # [32, 64, 128]
  output = output + ffn(ffn_norm(output))               # [32, 64, 128]

  # Cache for post-forward eligibility/EM
  self._last_h_all = output                              # [32, 64, 128]

  x = output  # next layer's input
```

Block 0 returns `x: [32, 64, 128]`. Repeat for blocks 1, 2, 3.

**Step 7 — Merge and predict (with spatial decoder):**
```
h_final = cat([h_b0, h_b1, h_b2, h_b3], dim=-1)       # [32, 64, 512]

# Spatial decoder (snapshot_enabled=True, the default):
# Collect cached per-layer outputs from each block
block_layer_outputs = [block._last_layer_stack for block in blocks]
                                                        # 4 × [32, 64, 8, 128]
# PM/EM summaries frozen within span
pm_summary = _compute_pm_summary(32, device)            # [32, 128]
em_summary = _compute_em_summary(32, device)            # [32, 128]

# Reshape everything to [BS*P, ...] = [2048, ...]
block_layer_flat = [blo.reshape(2048, 8, 128) for blo in block_layer_outputs]
pm_flat = pm_summary.unsqueeze(1).expand(32, 64, 128).reshape(2048, 128)
em_flat = em_summary.unsqueeze(1).expand(32, 64, 128).reshape(2048, 128)
wm_flat = y_wm_all.reshape(2048, 512)
h_flat = h_final.reshape(2048, 512)

# Run decoder on all 2048 positions at once
h_decoded = spatial_decoder(block_layer_flat, pm_flat,
                            em_flat, wm_flat, h_flat)   # [2048, 512]
logits_all = lm_head(h_decoded.reshape(32, 64, 512))   # [32, 64, 32000]
```

`forward_span` returns `(logits_all, x_emb_all, y_wm_all)`.

### 4.3 Phase 2: Post-forward in trainer

**Step 8 — Per-token surprise (no grad):**
```
logp = log_softmax(logits_all)                          # [32, 64, 32000]
token_surprise = -logp.gather(-1, targets)              # [32, 64, 1]
token_surprise *= loss_mask                             # zero at EOT positions
```

**Step 9 — Batched loss:**
```
span_loss = cross_entropy(logits_all.reshape(2048, 32000),
                          targets.reshape(2048),
                          mask.reshape(2048))           # scalar
```

**Step 10 — PM eligibility accumulation (batched projections + affine scan, per layer per block):**

`W_in` computed once for all blocks:
```
  x_proj_all = W_in(x_emb_all).view(32, 64, 4, 128)                   # 1 matmul
```

For each block b=0..3, each layer l=0..7:
```
  x_in  = previous layer's _last_h_all (or x_proj_all[:,:,b] for l=0)  # [32, 64, 128]
  h_out = this layer's _last_h_all                                      # [32, 64, 128]

  # Batched projections (2 matmuls instead of 128)
  k_cand_all = normalize(W_k_pre(x_in))                                # [32, 64, 128]
  v_cand_all = W_v_post(h_out)                                         # [32, 64, 128]

  # Surprise gating + carry mask for resets
  gate = (token_surprise / 5.0).clamp(0, 1)                            # [32, 64, 1, 1]
  a = 0.95 * carry  (0 at t=13 for reset streams)                      # [32, 64, 8, 128]
  b_K = gate * k_cand_all  (broadcast over r=8)                        # [32, 64, 8, 128]

  # Affine scan: elig_t = a_t * elig_{t-1} + b_t (fused K+V double-width)
  elig_KV_all = parallel_affine_scan(a_KV, b_KV, h_KV_init)           # [32, 64, 2048]
  elig_K, elig_V = elig_KV_all.chunk(2, dim=-1)                        # split back
  elig_K = elig_K[:, -1].reshape(32, 8, 128)                           # update to token 63
```

At t=13 for reset streams: carry=0 so `elig = 0 * elig_prev + gate * cand` (fresh start, equivalent to zeroing).

Note: The K and V eligibility scans are fused into a single double-width scan (concatenated along the last dimension) to halve scan kernel launches. This is a pure performance optimization — mathematically equivalent to two separate scans since the recurrence is element-wise.

**Step 11 — EM candidate proposal (batched):**

For each block b=0..3:
```
  h_final_all = block.layers[-1]._last_h_all            # [32, 64, 128]
  k_cand = normalize(W_k_cand(cat(x_emb_all, y_wm_all)))  # [32, 64, 128]
  v_cand = W_v_cand(h_final_all)                            # [32, 64, 128]
  novelty = blend(token_surprise, 1 - max_sim(k_cand, em_K))  # [32, 64]
```

Candidates are buffered. Pre-reset candidates (t < 13 for reset streams) are masked invalid later.

Note: mid-span resets can affect at most P-1=63 tokens per boundary event.

### 4.4 Phase 3: Span boundary (same as sequential path)

```
span_surprise_mean = span_surprise_accum / span_valid_tokens   # [32]
model.surprise = span_surprise_mean.unsqueeze(-1)               # [32, 1] — for next span

# PM: base decay + neuromodulator commit decision
for each PM instance:
    pm_a *= 0.999                                               # decay
    neuromodulator decides commit_mask, g, lambda, slot_logits
    pm.commit(mask, lambda, g, slots)                           # EMA update pm_K, pm_V, pm_a
    reset eligibility for committing streams

# EM: neuromodulator write decision
for each EM instance:
    neuromodulator decides g_em, tau, ww, decay
    em.write_at_boundary(cand_K, cand_V, scores, g_em, tau=tau, ww=ww, decay=decay)   # EMA update em_K, em_V, em_S
    em_S *= decay; budget_enforce(em_S)                         # per-stream learned decay + budget
```

State is now ready for the next span. The cycle repeats.

### 4.5 Where the speedup comes from

In the sequential path, steps 3-7 would each be called 64 times with `[32, ...]` tensors. In the parallel path, they're called once with `[32, 64, ...]` tensors. The key operations and their kernel launch counts:

| Operation | Sequential (64 tokens) | Parallel (1 span) |
|-----------|----------------------|-------------------|
| Embedding lookup | 64 launches | 1 launch |
| WM Q/K/V projection | 64 × 3 = 192 launches | 3 launches |
| WM attention (sequential either way) | 64 launches | 64 launches |
| W_in projection | 64 launches | 1 launch |
| Per-block EM retrieval | 64 × 4 = 256 launches | 4 launches |
| Per-block WM/EM projection | 64 × 4 × 2 = 512 launches | 4 × 2 = 8 launches |
| Per-layer PM read | 64 × 32 = 2048 launches | 32 launches |
| Per-layer gate_a + gate_b | 64 × 32 × 2 = 4096 launches | 32 × 2 = 64 launches |
| Per-layer scan (sequential either way) | 64 × 32 = 2048 element-wise | 32 × 64 = 2048 element-wise |
| Per-layer W_o + FFN | 64 × 32 × 3 = 6144 launches | 32 × 3 = 96 launches |
| LM head | 64 launches | 1 launch |
| PM eligibility projections (post-fwd) | 64 × 32 × 2 = 4096 launches | 32 × 2 = 64 launches |
| PM eligibility scan (post-fwd, fused K+V) | 64 × 32 × 2 = 4096 element-wise | 32 × 64 = 2048 element-wise |
| **Total kernel launches** | **~23,700** | **~440 + 64 seq** |

Each parallel launch does 64× more work, saturating GPU compute. The scan loops (layer recurrence + eligibility) are the same cost either way but trivial compared to the matmuls. `W_in` is computed once and shared across all blocks for both the forward pass and eligibility.

With `torch.compile` (§12), the scan loops, gate computations, and elementwise ops within each compiled region are further fused into a small number of optimized CUDA kernels, and the backward pass replaces ~200K autograd nodes with fused backward kernels.

---

## 5. Component-by-Component: Sequential vs Parallel

### 5.1 Embedding + LM Head + Spatial Decoder

**Sequential:** `embedding(input_id)` → `[BS, D]`, optionally `spatial_decoder(...)` → `[BS, D]`, then `lm_head(h)` → `[BS, vocab]`

**Parallel:** `embedding(input_ids)` → `[BS, P, D]`, optionally `spatial_decoder(...)` on `[BS*P, ...]` → `[BS*P, D]` reshaped to `[BS, P, D]`, then `lm_head(h)` → `[BS, P, vocab]`

**Difference:** None. When `snapshot_enabled=True` (the default), both paths run the spatial decoder. The parallel path reshapes `BS*P` into the batch dimension, runs the decoder once on all positions, and reshapes back. PM/EM summaries are frozen within a span (computed once and expanded), matching the sequential path where they don't change within a span either.

### 5.2 Working Memory

**File:** `src/model/working_memory.py` — `forward_span()`

**Sequential** (`step()`): Projects Q/K/V for one token, writes to ring buffer, attends over valid entries, advances pointer.

**Parallel** (`forward_span()`): Projects Q/K/V for all P tokens in one batched matmul, then runs per-token attention sequentially (matching `step()` exactly):

```python
# Batch projections (the expensive part)
q_all = self.W_q(x_all)    # [BS, P, D_wm]
k_all = self.W_k(x_all)    # [BS, P, D_wm]
v_all = self.W_v(x_all)    # [BS, P, D_wm]

# Sequential attention per token (matches step() exactly)
for t in range(P):
    # Reset validity for masked streams
    # Write (k,v) into ring buffer at ptr
    # Multi-head attention over valid entries
    # Advance pointer
```

**Difference:** None in output. The per-token inner loop is identical to `step()`. Only the Q/K/V projections are batched. WM handles mid-span resets internally (clearing validity and resetting pointer).

### 5.3 Procedural Memory Read

**File:** `src/model/procedural_memory.py` — `apply_batch()`

**Sequential** (`apply()`): `[BS, D_h]` input → scores against `pm_K` → weighted sum from `pm_V` → `[BS, D_h]`

**Parallel** (`apply_batch()`): Same math, batched over P:
```python
x_q = unit_normalize(x_block_all)                             # [BS, P, D_h]
scores = torch.einsum("brd, bpd -> bpr", self.pm_K, x_q)     # [BS, P, r]
weighted = self.pm_a.unsqueeze(1) * scores                    # [BS, P, r]
y_pm = torch.einsum("bpr, brd -> bpd", weighted, self.pm_V)  # [BS, P, D_h]
```

**Difference:** None — PM state (`pm_K`, `pm_V`, `pm_a`) is frozen within a span in both paths. The einsum handles the extra P dimension natively.

### 5.4 Episodic Memory Retrieval

**File:** `src/model/episodic_memory.py` — `retrieve_batch()`

**Sequential** (`retrieve()`): `[BS, D]` query → top-k scoring against `em_K` → cross-attention over top-k values → `[BS, D]`

**Parallel** (`retrieve_batch()`): Same math, batched over P. The top-k and cross-attention operate on `[BS, P, ...]` tensors:
```python
q = unit_normalize(W_q_em(cat([x_all, y_wm_all], dim=-1)))   # [BS, P, D_em]
scores = einsum("bpd, bmd -> bpm", q, em_K)                   # [BS, P, M]
topk_scores, topk_idx = scores.topk(k_ret, dim=-1)            # [BS, P, k]
# ... cross-attention ...
y_em_all = W_o_cross(out)                                      # [BS, P, D]
```

**Difference:** None in math. EM state (`em_K`, `em_V`, `em_S`) is frozen within a span in both paths. Implementation uses `em_V.unsqueeze(1).expand(-1, P, -1, -1)` for the gather, which increases peak memory by `P * M * D_em` per block.

### 5.5 Layer (Core Recurrence)

**File:** `src/model/layer.py` — `forward_span()`

**Sequential** (`step()`):
```python
u = cat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])  # [BS, 4*D_h+1]
a = sigmoid(gate_a(u))                                      # [BS, D_h]
b = tanh(gate_b(u))                                         # [BS, D_h]
self.h = a * (carry * self.h) + b                           # scalar recurrence
output = norm(W_o(self.h) + x_block)                        # projection + residual + norm
output = output + ffn(ffn_norm(output))                     # post-recurrence FFN
```

**Parallel** (`forward_span()`):
```python
surprise_all = surprise_span.unsqueeze(1).expand(BS, P, 1)         # frozen!
u = cat([x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
         surprise_all], dim=-1)                                     # [BS, P, 4*D_h+1]

a = sigmoid(gate_a(u))                                              # [BS, P, D_h] — batched
b = tanh(gate_b(u))                                                 # [BS, P, D_h] — batched
a_eff = a * carry_all                                               # zero at doc boundaries

h_all = parallel_affine_scan(a_eff, b, self.h)                     # [BS, P, D_h] — scan
self.h = h_all[:, -1]                                               # update state

output = norm(W_o(h_all) + x_all)                                  # batched projection + norm
output = output + ffn(ffn_norm(output))                            # batched FFN

self._last_h_all = output                                          # cached for post-forward
```

**Key differences:**
1. **Surprise is frozen** — `surprise_span` is the same `[BS, 1]` value broadcast to all P positions (§7).
2. **Gate projections batched** — one `gate_a` forward on `[BS, P, 4*D_h+1]` instead of P separate calls on `[BS, 4*D_h+1]`. This is where most of the speedup comes from.
3. **Recurrence via scan** — `parallel_affine_scan` computes all P hidden states given pre-computed `a` and `b` (§6).
4. **`_last_h_all` cached** — The full `[BS, P, D_h]` output sequence is stored for the trainer's post-forward eligibility and EM candidate steps.

### 5.6 Block

**File:** `src/model/block.py` — `forward_span()`

**Sequential** (`step()`): Per layer: PM read → layer step → eligibility update → next layer

**Parallel** (`forward_span()`): Per layer: PM batch read → layer forward_span → next layer. **Eligibility is NOT updated** inside the block — it's deferred to the trainer (§9).

```python
def forward_span(self, x_block_all, y_wm_all, x_emb_all,
                 surprise_span, carry_all):
    # EM retrieval (batched, frozen state)
    y_em_all = self.em.retrieve_batch(x_emb_all, y_wm_all)     # [BS, P, D]

    # Project shared signals to per-block dimension
    y_wm_proj_all = self.W_wm_proj(y_wm_all)                   # [BS, P, D_h]
    y_em_proj_all = self.W_em_proj(y_em_all)                    # [BS, P, D_h]

    x = x_block_all
    for layer in self.layers:
        y_pm_all = layer.pm.apply_batch(x)                      # PM read (batched)
        x = layer.forward_span(x, y_pm_all, y_wm_proj_all,
                               y_em_proj_all, surprise_span, carry_all)

    # Cache per-layer outputs for spatial decoder
    self._last_layer_stack = torch.stack(
        [layer._last_h_all for layer in self.layers], dim=2
    )  # [BS, P, L, D_h]

    return x
```

**Key differences:**
1. Eligibility accumulation is decoupled from the forward pass and handled in the trainer post-forward step. This is necessary because eligibility needs per-token surprise (not available until after logits are computed).
2. `_last_layer_stack` is cached on the block for the spatial decoder. Each layer already caches `_last_h_all` during `forward_span`; the block stacks them into `[BS, P, L, D_h]`. The model reshapes to `[BS*P, L, D_h]` before passing to the decoder.

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

**This is intentionally a sequential loop.** The scan itself is NOT the bottleneck — P=64 iterations of element-wise ops on `[BS, D_h]` = `[32, 128]` tensors takes microseconds. The speedup comes from batching the expensive operations *around* the scan (gate projections, FFN, attention), and from `torch.compile` (§12) which unrolls the loop and fuses all elementwise ops into optimized CUDA kernels. A true CUDA parallel prefix scan could replace this later as a further optimization.

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
1. Surprise is 1 dimension out of `4*D_h + 1 = 513` gate input dimensions.
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
- `em_S` (non-lifelong only)
- WM validity and pointer

**Parallel path:** `reset_at_doc_boundary` is called **only for the first token** of the span (`model.py:184-185`). For mid-span doc boundaries (tokens 1..P-1):

| State | How it's handled | Equivalent to sequential? |
|-------|-----------------|--------------------------|
| `Layer.h` | Carry mask zeros `a_eff`, giving `h_t = b_t` | Yes |
| `elig_K, elig_V` | Zeroed in trainer post-forward loop | Yes |
| `pm_K, pm_V, pm_a` | NOT zeroed mid-span | **No** (phases A-D) |
| `em_S` | NOT zeroed mid-span | **No** (phases A-C) |
| WM state | Handled internally by `wm.forward_span` | Yes |
| `model.surprise` | Frozen anyway | N/A |

**Impact of the PM/EM gap:** In non-lifelong mode (phases A-B), post-boundary tokens within the same span may read stale PM state and retrieve stale EM memories from the old document. This affects at most P-1=63 tokens per boundary event. In lifelong mode (Phase C), PM/EM state intentionally persists across documents, so there is no gap.

**Eligibility reset detail** (span_ops.py: `apply_pm_eligibility_batch`):
```python
for t_local in range(span_P):
    reset_t = reset_mask_all[:, t_local]
    if reset_t.any():
        pm.reset_eligibility(reset_t)    # zero elig_K, elig_V
    pm.update_eligibility(x_in[:, t_local], h_out[:, t_local],
                          surprise_for_elig[:, t_local])
```

---

## 9. Post-Forward: Eligibility, EM Candidates, Loss

After `forward_span()` returns `logits_all`, `x_emb_all`, `y_wm_all`, the trainer runs several post-forward steps that are NOT inside the model's forward pass:

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

### 9.3 PM Eligibility Accumulation

In the sequential path, eligibility is updated inside `Block.step()` — one call per token per layer, interleaved with the forward pass. In the parallel path, eligibility uses `update_eligibility_batch()` which batches the projections and runs an affine scan:

```python
# W_in computed once (shared across all blocks)
x_proj_all = W_in(x_emb_all).view(BS, P, B, D_h)  # one matmul

for b, block in enumerate(model.blocks):
    for layer in block.layers:
        # Layer input: block input (layer 0) or previous layer's output
        if layer.layer_idx == 0:
            x_in = x_proj_all[:, :, b]
        else:
            x_in = block.layers[layer.layer_idx - 1]._last_h_all

        h_out = layer._last_h_all

        # Batched: projections + affine scan (replaces P-step loop)
        layer.pm.update_eligibility_batch(x_in, h_out, token_surprise, reset_mask_all)
```

Inside `update_eligibility_batch()`:
```python
# Batch projections (2 matmuls instead of 2*P)
k_cand_all = unit_normalize(W_k_pre(x_all))   # [BS, P, D_h]
v_cand_all = W_v_post(h_all)                   # [BS, P, D_h]

# Surprise gating
gate = (surprise_all / 5.0).clamp(0, 1)        # [BS, P, 1, 1]
b_K = gate * k_cand_all  (broadcast over r)     # [BS, P, r, D_h]

# Carry mask: a=0 at resets (equivalent to zeroing elig before accumulation)
a = rho * carry  (0 at resets, rho elsewhere)   # [BS, P, r, D_h]

# Fused K+V scan: one double-width scan instead of two separate
b_KV = cat([b_K_flat, b_V_flat], dim=-1)       # [BS, P, 2*r*D_h]
a_KV = cat([a_flat, a_flat], dim=-1)            # [BS, P, 2*r*D_h]
h_KV_init = cat([h_K_init, h_V_init], dim=-1)  # [BS, 2*r*D_h]
elig_KV_all = parallel_affine_scan(a_KV, b_KV, h_KV_init)  # [BS, P, 2*r*D_h]
elig_K_all, elig_V_all = elig_KV_all.chunk(2, dim=-1)
self.elig_K = elig_K_all[:, -1]                 # update to last token
self.elig_V = elig_V_all[:, -1]
```

The K and V scans are fused into a single double-width scan by concatenating along the last dimension. This is mathematically equivalent to two separate scans (the recurrence is element-wise), but halves the number of scan kernel launches.

**Differences from sequential path:**
1. **`_last_h_all` is the full layer output** (post W_o + residual + norm + FFN), matching what the sequential path passes as `h` to `update_eligibility`. This was a bug that was fixed — originally `_last_h_all` stored raw recurrent `h` before the output projection.
2. **Surprise gating uses per-token surprise**, not frozen surprise. In the sequential path, `update_eligibility` receives `model.surprise` which lags by one token (it's the previous token's surprise). In the parallel path, it receives the current token's actual surprise. This is arguably more correct.
3. **Doc-boundary resets via carry mask**: Instead of explicitly calling `reset_eligibility()` before each token, the carry mask sets `a=0` at reset positions — mathematically equivalent (`elig_t = 0 * elig_{t-1} + gate * cand = gate * cand`).

### 9.4 EM Candidate Proposal

```python
for b, block in enumerate(model.blocks):
    h_final_all = block.layers[-1]._last_h_all  # [BS, P, D_h]
    k_c, v_c, nov = block.em.propose_candidate_batch(
        x_emb_all, y_wm_all, h_final_all, token_surprise,
    )
```

Uses `propose_candidate_batch()` which batches the same math as `propose_candidate()` across P tokens. Uses the full layer output (`_last_h_all`) for value candidates, and real per-token surprise for novelty scoring. EM state (`em_K`, `em_S`) is frozen during scoring.

**Candidate validity:** The trainer tracks `span_last_reset` per stream and masks out candidates from tokens before the last doc-boundary reset in the span. This prevents writing stale cross-document content to EM.

---

## 10. Training Loop Integration

**Files:** `src/training/trainer.py` (orchestration), `src/training/span_ops.py` (shared span-boundary ops used by trainer, validation, and eval_lifelong)

### 10.1 Span Loop Structure (Full Architecture)

The walkthrough below shows the complete flow with all subsystems active (Phase B+). In Phase A, PM/EM neuromodulator steps use heuristic defaults.

```
For each span (4 spans per chunk, P=64 tokens each):
    Reset EM candidate buffers, span accumulators

    # --- Parallel forward ---
    logits_all, x_emb_all, y_wm_all = model.forward_span(span_ids, reset_first)
      └── Embedding → WM.forward_span → Block.forward_span (per block):
            EM.retrieve_batch → per-layer: PM.apply_batch → gates → scan → FFN
          → SpatialDecoder (if snapshot_enabled) → LM head → logits [BS,P,vocab]

    span_loss, span_valid = batched_cross_entropy(logits_all, targets, mask)
    chunk_loss += span_loss

    # --- Post-forward (no grad for surprise; grad for eligibility projections) ---
    Compute per-token surprise from logits             # [BS, P, 1]
    Compute reset_mask_all for accumulators
    Accumulate span_surprise_mean per stream           # [BS] scalar
    Update model.surprise = span_surprise_mean         # frozen for next span's gates

    PM eligibility accumulation:                       # per block, per layer
      update_eligibility_batch(x_in, h_out, token_surprise, resets)
        └── surprise gating → affine scan → elig_K, elig_V updated

    EM candidate proposal:                             # per block
      propose_candidate_batch(x_emb, y_wm, h_final, surprise)
        └── key/value candidates + novelty scores      # [BS, P, D_em]

    # --- Span boundary ---
    Compute span_surprise_mean for neuromodulators
    Pre-compute EM candidate stacks + mean novelty

    PM neuromodulators: (elig_norm, usage, surprise) → commit_mask, lambda, g, slots
      └── commit_mask: heuristic gate (elig_norm > 1.0)
      └── PM.commit(): update pm_K, pm_V, pm_a with EMA

    EM neuromodulators: (surprise, usage, novelty) → g_em, tau, ww, decay
      └── g_em  = floor + range * sigmoid(raw)  [0.001, 0.95]
      └── tau   = floor + range * sigmoid(raw)  [0.05, 5.0]   (soft top-k temperature)
      └── ww    = floor + range * sigmoid(raw)  [0.0, 2.0]    (weakness weight)
      └── decay = floor + range * sigmoid(raw)  [0.99, 0.9999] (memory retention)
      └── EM.write_at_boundary(): update em_K, em_V, em_S with alpha = g_em * weights

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
| Eligibility update | Inside `Block.step`, interleaved | Post-forward loop in trainer |
| EM candidate proposal | Inside trainer token loop | Post-forward batch in trainer |
| Span boundary logic | Unchanged | Unchanged |
---

## 11. Exact Semantic Differences

These are the places where the parallel path produces different numerical results from the sequential path, even with identical inputs and initial state:

### 11.1 Surprise in Layer Gates

**Magnitude:** Small (1/513 gate input dimensions)

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

Sequential: `reset_at_doc_boundary` zeros `em_S` → post-boundary tokens retrieve nothing (all slots inactive).
Parallel: `em_S` not zeroed mid-span → post-boundary tokens retrieve old-document memories.

In practice: EM content from old document is unrelated to new document, but retrieval output passes through learned projections that can dampen irrelevant signals.

### 11.4 Eligibility Surprise Gating

**Magnitude:** Tiny.

Sequential: `update_eligibility` receives `model.surprise` which is the *previous* token's surprise.
Parallel: `update_eligibility` receives the *current* token's surprise (computed post-forward).

The parallel path is arguably more correct — gating eligibility by how surprising the current token is, rather than using a one-step-lagged value.

### 11.5 Summary Table

| Difference | Affects | Phases | Max tokens affected | Severity |
|-----------|---------|--------|-------------------|----------|
| Frozen surprise in gates | Gate values `a`, `b` | All | All P tokens in span | Low |
| PM state leak mid-span | PM read output | A-B | ≤63 per boundary | Low-medium |
| EM strength leak mid-span | EM retrieval output | A-B | ≤63 per boundary | Low-medium |
| Eligibility surprise timing | Eligibility trace | B+ | All P tokens | Negligible (arguably better) |

### 11.6 What Is Identical

Everything not listed above is numerically identical between the two paths:
- Embedding, spatial decoder, and LM head computation
- Working memory attention and ring buffer state
- Recurrent hidden state `h` (carry mask is equivalent to zeroing h)
- PM read math (`apply` vs `apply_batch`)
- EM retrieval math (`retrieve` vs `retrieve_batch`)
- EM candidate proposals and validity masking
- Loss computation
- Span boundary logic (PM commits, EM writes, neuromodulator decisions)
---

## 12. What Is NOT Supported / Compile Support

The parallel path intentionally does not support:

1. **Gate stats collection** (`collect=True`): `forward_span` does not return per-layer gate statistics. The trainer handles this by passing `gate_stats or {}` to the collector.

### torch.compile Support

`torch.compile(fullgraph=True)` is supported and tested for the two performance-critical paths:

- **`Layer.forward_span`**: Compiles the gate computation, affine scan, output projection, and FFN into fused CUDA kernels. The scan loop is unrolled by the compiler, and the backward pass replaces ~200K `SelectBackward0` autograd nodes with efficient fused backward kernels.
- **`PM.update_eligibility_batch`**: Compiles the eligibility projections, surprise gating, and fused K+V affine scan.

**Not compiled:**
- **`WorkingMemory.forward_span`**: Contains a data-dependent branch (`reset_mask_all[:, 1:].any()`) that would cause graph breaks. WM's per-token attention loop is already the correct granularity.
- **`Block.forward_span`** / **`NeuromorphicLM.forward_span`**: Not compiled at the top level — compilation is applied at the Layer granularity to avoid graph breaks from WM and other control flow.

**Usage:** Enable with `--compile` on the training CLI or `use_compile: True` in config. Requires CUDA. First step is slow (~30-60s) due to tracing; subsequent steps use cached compiled kernels. Provides ~2.8× speedup (measured: 3,687 → 10,278 tok/s on RTX 4090, Tier A, BS=32).

---

## 13. File Index

| File | Parallel Path Content |
|------|----------------------|
| `src/model/scan.py` | `parallel_affine_scan(a, b, h_init)` — sequential-loop scan |
| `src/model/layer.py` | `Layer.forward_span()` — batched gates + scan + `_last_h_all` cache |
| `src/model/procedural_memory.py` | `ProceduralMemory.apply_batch()` — batched PM read |
| `src/model/episodic_memory.py` | `EpisodicMemory.retrieve_batch()`, `propose_candidate_batch()` |
| `src/model/working_memory.py` | `WorkingMemory.forward_span()` — batched Q/K/V projections, sequential attention |
| `src/model/block.py` | `Block.forward_span()` — orchestrates batched components, caches `_last_layer_stack` |
| `src/model/decoder.py` | `SpatialDecoder.forward()` — called with `[BS*P, ...]` tensors in span path |
| `src/model/model.py` | `NeuromorphicLM.forward_span()`, `_compute_reset_masks()` |
| `src/training/trainer.py` | Span loop orchestration, delegates to span_ops |
| `src/training/span_ops.py` | Shared span-boundary ops: loss masking, surprise, PM/EM accumulation and commits |
| `src/training/loss.py` | `batched_cross_entropy()` — span-level loss computation |
| `tests/test_scan.py` | 28 equivalence tests: parallel vs sequential output, state, gradients |
