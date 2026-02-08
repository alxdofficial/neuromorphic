# Neuromorphic LM -- Full Model Explainer

**Purpose:** Independent reference for verifying that all code changes remain aligned with the design intent. Covers every module, its intuition, its concrete implementation, and how modules compose during training.

**Last verified against code:** 2026-02-08 (post spatial decoder: hierarchical aggregation + deep cross-attention)

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Token-Level Dataflow](#3-token-level-dataflow)
4. [Embedding + LM Head](#4-embedding--lm-head)
5. [Working Memory (WM)](#5-working-memory-wm)
6. [The Core Recurrence (Layer)](#6-the-core-recurrence-layer)
7. [Procedural Memory (PM)](#7-procedural-memory-pm)
8. [Episodic Memory (EM)](#8-episodic-memory-em)
9. [Blocks: Putting It Together](#9-blocks-putting-it-together)
10. [Neuromodulators](#10-neuromodulators)
11. [Phase D -- Hybrid RL + Backprop Training](#11-phase-d----hybrid-rl--backprop-training)
12. [Spatial Decoder](#12-spatial-decoder)
13. [State Management](#13-state-management)
14. [Training Loop (TBPTT + Spans)](#14-training-loop-tbptt--spans)
15. [Data Pipeline (Persistent Streams)](#15-data-pipeline-persistent-streams)
16. [Loss and Regularization](#16-loss-and-regularization)
17. [Phased Training Plan](#17-phased-training-plan)
18. [Scaling Tiers](#18-scaling-tiers)
19. [Gradient Flow Map](#19-gradient-flow-map)
20. [Design Integrity Notes](#20-design-integrity-notes)

---

## 1. Design Philosophy

The neuromorphic LM decomposes language model memory into four biologically-inspired systems, each with different persistence and update mechanics:

| System | Biological Analogy | Persistence | Update Mechanism |
|--------|--------------------|-------------|------------------|
| **Genetic memory** (slow weights) | DNA / long-term evolutionary adaptation | Permanent after training | Standard backprop (frozen at deployment) |
| **Working memory** (WM) | Prefrontal cortex / scratchpad | Last W tokens | Sliding window attention (no plasticity) |
| **Procedural memory** (PM) | Cerebellum / skill memory | Across documents/lifelong | Eligibility traces + neuromodulated commits |
| **Episodic memory** (EM) | Hippocampus / event memory | Across documents/lifelong | Novelty-based writes to vector store + neuromodulation |

**Why not just use a transformer?** Transformers have a single memory mechanism (KV cache) that scales linearly with context. Our model separates *what* to remember from *how long* to remember it. WM handles precise short-range copying. PM learns reusable patterns (like skills). EM stores specific surprising events for later recall. Each system can be independently controlled, budgeted, and -- as of Phase D -- learned via RL.

**Key constraint:** Single RTX 4090 (24GB). Everything must fit in VRAM with bf16 mixed precision. This rules out large KV caches or dense attention over long contexts, motivating the fixed-size memory banks.

---

## 2. Architecture at a Glance

```
Token ID ──► Embedding ──► x: [BS, D]
                              │
                              ├──► Working Memory ──► y_wm: [BS, D]  (shared)
                              │
                              ├──► W_in projection ──► split into B blocks
                              │
                     ┌────────┴────────┐
                     │   Per Block (b)  │  × B blocks in parallel
                     │                  │
                     │  EM retrieve     │  y_em = EM[b].retrieve(x, y_wm)
                     │  Project to D_h  │  y_wm_proj, y_em_proj
                     │                  │
                     │  ┌─── Per Layer ─┐│  × L layers sequentially
                     │  │ PM read       ││  y_pm = PM[b][l].apply(x_block)
                     │  │ Gate compute  ││  a, b from concat(inputs)
                     │  │ Recurrence    ││  h = a * (carry * h_prev) + b
                     │  │ Residual+Norm ││  output = norm(h + x_block)
                     │  │ Elig update   ││  elig_K += k_cand, elig_V += v_cand
                     │  └───────────────┘│
                     │                    │
                     │  h_out: [BS, D_h]  │
                     │  (+ all L layer    │
                     │   outputs if       │
                     │   snapshot_enabled) │
                     └────────┬───────────┘
                              │
                     concat all blocks ──► h_final: [BS, D]
                              │
              ┌───────────────┤ (if snapshot_enabled)
              │               │
              ▼               │ (if not snapshot_enabled)
     Spatial Decoder          │
     ├─ Columnar attn         │
     ├─ Thalamic integr.      │
     ├─ Deep decoder           │
     └─► h_decoded: [BS, D]   │
              │               │
              └───────┬───────┘
                      │
                 LM Head ──► logits: [BS, vocab]
```

**Dimensions (Tier A defaults):** D=512, B=4, L=8, D_h=128, vocab=32000

**Instance counts:**

| Component | Class | Count | Location |
|-----------|-------|-------|----------|
| WM | `WorkingMemory` | 1 | `model.wm` (shared) |
| EM banks | `EpisodicMemory` | B=4 | `block.em` (one per block) |
| PM instances | `ProceduralMemory` | B*L=32 | `layer.pm` (one per layer per block) |
| PM neuromodulators | `PMNeuromodulator` | B*L=32 | `layer.pm_neuromodulator` |
| EM neuromodulators | `EMNeuromodulator` | B=4 | `block.em_neuromodulator` |
| Spatial decoder | `SpatialDecoder` | 0 or 1 | `model.spatial_decoder` (if `snapshot_enabled`) |

---

## 3. Token-Level Dataflow

For each token at time t, `NeuromorphicLM.forward_one_token()` does:

```python
# 1. Reset states if previous token was EOT (per-stream)
if reset_mask.any():
    self.reset_at_doc_boundary(reset_mask)

# 2. Embed
x = self.embedding(input_id)           # [BS, D]

# 3. Working memory (shared, sliding window attention)
y_wm = self.wm.step(x, reset_mask)    # [BS, D]

# 4. Project and split for parallel blocks
x_proj = self.W_in(x)                  # [BS, D]
x_blocks = x_proj.view(BS, B, D_h)    # [BS, B, D_h]

# 5. Each block processes its D_h slice with access to x, y_wm
carry = (~reset_mask).float()[:, None]  # [BS, 1]
for b, block in enumerate(self.blocks):
    h_b = block.step(x_blocks[:, b], y_wm, x, surprise, carry)

# 6. Merge and predict
h_final = concat(h_blocks)             # [BS, D]

if snapshot_enabled:
    # Hierarchical aggregation → deep cross-attention (§12)
    h_decoded = spatial_decoder(layer_outputs, pm_summary, em_summary, y_wm, h_final)
    logits = self.lm_head(h_decoded)   # [BS, vocab]
else:
    logits = self.lm_head(h_final)     # [BS, vocab]

return logits, x, y_wm
```

**Returns:** Always `(logits, x_emb, y_wm)`. The trainer needs `x_emb` and `y_wm` for EM candidate proposals (they cannot be recomputed cheaply because WM state has already advanced).

---

## 4. Embedding + LM Head

**Embedding:** `nn.Embedding(vocab_size, D)` -- standard lookup table. Output `x: [BS, D]` is the raw token representation used by all downstream systems.

**LM Head:** `nn.Linear(D, vocab_size, bias=False)` -- projects merged block outputs to vocabulary logits. Not weight-tied with embedding (optional future optimization).

**Input projection:** `W_in: nn.Linear(D, D, bias=False)` -- learned projection before splitting across blocks. This allows the model to learn how to distribute information across the B parallel tracks, rather than using a fixed partition of embedding dimensions.

---

## 5. Working Memory (WM)

**File:** `src/model/working_memory.py`

**Intuition:** WM is the model's scratchpad -- bounded sliding-window attention over the last W=256 tokens. It provides transformer-like precision for short-range patterns (copying a name from 50 tokens ago, binding an adjective to its noun) without the unbounded memory cost of full attention.

**State per stream:**
- `wm_K: [BS, W, D_wm]` -- key cache (ring buffer)
- `wm_V: [BS, W, D_wm]` -- value cache (ring buffer)
- `wm_valid: [BS, W]` -- bool mask for occupied slots
- `wm_ptr: [BS]` -- ring buffer write index

**Per-token operation:**
1. Project current token: `q = W_q(x)`, `k = W_k(x)`, `v = W_v(x)` all `[BS, D_wm]`
2. Write `(k, v)` into ring buffer at `wm_ptr` via functional scatter (differentiable within TBPTT chunk)
3. Multi-head attention (`n_heads=4`) over all valid positions in the buffer
4. Project output: `y_wm = W_o(attn_output)` -> `[BS, D]`
5. Advance pointer: `wm_ptr = (wm_ptr + 1) % W`

**Doc boundary reset:** Clears `wm_valid` and resets `wm_ptr` to 0 for masked streams. This prevents cross-document information leakage.

**Gradient flow through the KV buffer:** Within a TBPTT chunk, gradients flow through the ring buffer back to `W_k` and `W_v`. The write uses a functional scatter (`wm_K = wm_K * (1 - mask) + k * mask`) rather than in-place mutation, keeping the gradient path alive. At TBPTT chunk boundaries, all buffer state is detached. WM is *non-plastic* in that it has no explicit memory update mechanism (no commits, no decay, no RL controller) — it simply caches the W most recent token projections. But `W_k`, `W_v`, `W_q`, and `W_o` are all trained via backprop through the attention computation.

**Shared across model:** There is exactly one WM instance. Its output `y_wm: [BS, D]` is broadcast to all B blocks, where each block projects it down to `D_h` via its own `W_wm_proj`.

---

## 6. The Core Recurrence (Layer)

**File:** `src/model/layer.py`

**Intuition:** Each layer maintains a recurrent hidden state `h: [BS, D_h]` that is updated via a scan-friendly affine recurrence. This is the backbone of the model -- it processes the fused signal from all memory systems and propagates information forward in time.

**Why "scan-friendly"?** The gates `a` and `b` are computed from input features only -- they do **not** depend on the previous hidden state `h_{t-1}`. This means that given a span of P tokens, the entire recurrence `h_t = a_t * h_{t-1} + b_t` can be computed via a parallel prefix scan (like Mamba, RWKV, or S4). We currently run token-by-token for simplicity, but the math allows parallelization.

**Gate computation:**
```python
# Fuse all inputs into a single feature vector
u = concat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])
# u: [BS, 4*D_h + 1]

# Compute gates (NO dependence on h_prev)
a = sigmoid(W_a(u))   # [BS, D_h]  -- retention gate (how much to keep)
b = tanh(W_b(u))       # [BS, D_h]  -- update gate (what to add)
```

**Recurrence with doc-boundary carry:**
```python
h = a * (carry * h_prev) + b
# carry = 0.0 at doc boundaries, 1.0 otherwise
# When carry=0: h = a*0 + b = b (fresh start)
# When carry=1: h = a*h_prev + b (normal recurrence)
```

**Output:** `norm(W_o(h) + x_block)` -- output projection, residual connection from block input, plus LayerNorm. `W_o: nn.Linear(D_h, D_h)` is a per-layer learned projection that allows the layer to transform its hidden state before the residual add.

**What each input contributes:**
- `x_block: [BS, D_h]` -- the current token's representation for this block
- `y_pm: [BS, D_h]` -- what procedural memory recalls for this input (learned patterns)
- `y_wm_proj: [BS, D_h]` -- what working memory recalled (recent context)
- `y_em_proj: [BS, D_h]` -- what episodic memory recalled (long-range events)
- `surprise: [BS, 1]` -- how surprised the model was at the previous prediction (scalar signal)

---

## 7. Procedural Memory (PM)

**File:** `src/model/procedural_memory.py`

**Intuition:** PM is the model's "skill memory." It stores low-rank key-value pairs that represent learned associations -- like a lookup table of patterns the model has encountered. When the model sees input that matches a stored key, PM retrieves the corresponding value and injects it into the processing pipeline. Unlike slow weights (parameters), PM can be updated *during inference* via neuromodulated commits at span boundaries.

**Instance count:** One PM per (block, layer) pair. B*L = 32 instances total (Tier A). Each operates independently on its own `[BS, D_h]` slice.

### 7.1 PM State

Per PM instance, per stream:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `pm_K` | `[BS, r, D_h]` | Key bank (unit-normalized). What patterns to match. |
| `pm_V` | `[BS, r, D_h]` | Value bank (unit-normalized). What to retrieve when matched. |
| `pm_a` | `[BS, r]` | Slot strengths. How "confident" each slot is. Range [0, a_max]. |
| `elig_K` | `[BS, r, D_h]` | Key eligibility trace. Accumulates proposed key candidates over time. |
| `elig_V` | `[BS, r, D_h]` | Value eligibility trace. Accumulates proposed value candidates over time. |

`r=8` slots per instance. Total PM state: B*L*r = 256 slot-pairs.

### 7.2 PM Read (every token)

```python
x_q = normalize(x_block)                    # [BS, D_h]
scores = einsum("brd, bd -> br", pm_K, x_q) # [BS, r]
y_pm = einsum("br, brd -> bd", pm_a * scores, pm_V)  # [BS, D_h]
```

**Intuition:** The input is compared against all r stored keys. Each key that matches well contributes its value, weighted by both the match score and the slot strength `pm_a`. Slots with `pm_a=0` contribute nothing (invisible).

This is a read-only operation. PM keys and values are frozen within each plasticity span.

### 7.3 Eligibility Traces (every token, differentiable)

```python
k_cand = normalize(W_k_pre(x))    # [BS, D_h] -- from pre-synaptic (input)
v_cand = W_v_post(h)               # [BS, D_h] -- from post-synaptic (output)

elig_K = rho * elig_K + k_cand.unsqueeze(1)  # accumulate into all r slots
elig_V = rho * elig_V + v_cand.unsqueeze(1)
```

**Intuition:** Eligibility traces are a running average of "what the model wanted to write." The projections `W_k_pre` and `W_v_post` are learned parameters -- gradients from the LM loss flow back through them, teaching the model *what* constitutes a good key-value pair to store. But the *decision* of whether to actually commit these traces into PM is made by the neuromodulator at span boundaries.

`rho=0.95` is the eligibility decay -- recent tokens contribute more than older ones. This creates a recency-weighted summary of the span's proposed writes.

**Key insight:** Eligibility is **differentiable** (no `torch.no_grad()`). The projections `W_k_pre` and `W_v_post` are `nn.Linear` layers trained by backprop. The model learns what to propose; the neuromodulator learns when to commit.

**Relationship to Hebbian learning:** This follows the **neo-Hebbian three-factor learning rule** from neuroscience. Classical Hebbian learning ("fire together, wire together") updates synapses based on the correlation between pre-synaptic input and post-synaptic output: `ΔW ∝ post ⊗ pre`. Our eligibility traces accumulate exactly this -- `k_cand` from the pre-synaptic signal (layer input `x`) and `v_cand` from the post-synaptic signal (layer output `h`). Their implicit outer product `v_cand ⊗ k_cand` is the Hebbian update that *would* be applied.

The critical difference from pure Hebbian learning is the **third factor: the neuromodulator**. Rather than applying `post ⊗ pre` immediately at every token (which causes catastrophic drift), the eligibility trace buffers the proposed update with exponential decay, and the neuromodulator decides at span boundaries whether to commit it. This three-factor rule -- `ΔW ∝ post × pre × neuromodulator` -- is how biological synapses are believed to work: local activity creates eligibility for change, but a global modulatory signal (dopamine, norepinephrine) determines whether that change is consolidated.

### 7.4 PM Commit (span boundary only)

Called every P=32 tokens. The commit process:

1. **Base decay** (all streams): `pm_a *= decay_pm` (0.999). Ensures non-committing streams gradually lose strength, preventing stale slots from persisting forever. Called in the trainer BEFORE commit decisions.

2. **Neuromodulator decision:** `PMNeuromodulator.forward(elig_norm, pm_usage, span_surprise)` returns a 5-tuple `(commit_mask, lambda_vals, g, slot_logits, p_commit)`.

   In heuristic mode (Phase A — `pm_enabled=False`):
   - `commit_mask: [BS]` -- commit if `elig_norm > 1.0`
   - `lambda_vals: [BS]` -- fixed `config.decay_pm`
   - `g: [BS]` -- fixed 0.5
   - `slot_logits: None` -- slot selection uses similarity
   - `p_commit: None` -- no learned probability

   In continuous-learned mode (Phases B–C — `pm_enabled=True`, `rl_enabled=False`):
   - `commit_mask: [BS]` -- heuristic gate: `elig_norm > 1.0`
   - `lambda_vals: [BS]` -- sigmoid, scaled to `[decay_pm, 1.0]` (differentiable, main optimizer)
   - `g: [BS]` -- sigmoid output (differentiable, main optimizer)
   - `slot_logits: [BS, r]` -- raw linear output (differentiable, main optimizer)
   - `p_commit: None` -- no RL gate yet

   In fully-learned mode (Phase D+ — `pm_enabled=True`, `rl_enabled=True`):
   - `commit_mask: [BS]` -- `(p_commit > 0.5).detach()` (hard gate, no grad)
   - `lambda_vals: [BS]` -- sigmoid, scaled to `[decay_pm, 1.0]` (differentiable)
   - `g: [BS]` -- sigmoid output (differentiable)
   - `slot_logits: [BS, r]` -- raw linear output (differentiable)
   - `p_commit: [BS]` -- sigmoid probability (used for RL training)

3. **Slot selection** (soft top-k):
   ```python
   scores = einsum(pm_K, elig_K_norm)       # [BS, r] -- similarity
   scores -= weakness_weight * pm_a          # prefer overwriting weak slots
   if slot_logits is not None:
       scores += slot_logits                 # learned bias (Phase D)
   weights = soft_topk(scores, k=2, tau=1.0) # [BS, r] -- softmax over top-2
   ```

4. **EMA update** (for committing streams):
   ```python
   alpha = weights * g * commit_mask         # [BS, r]
   pm_K = normalize((1-alpha) * pm_K + alpha * elig_K_norm)
   pm_V = normalize((1-alpha) * pm_V + alpha * elig_V)
   pm_a = clamp(pm_a + alpha, 0, a_max)
   ```

5. **Budget enforcement:** `sum(pm_a) <= budget_pm` per stream, enforced by proportional scaling.

6. **Reset eligibility** for committing streams (multiply by zero).

**Why soft top-k?** Rather than hard-selecting one slot to overwrite, soft top-k distributes the write across the k=2 most suitable slots with softmax weights. This provides a smoother gradient signal and avoids catastrophic overwriting of a single slot.

**Differentiable continuous outputs:** Starting in Phase B (continuous-learned mode), `lambda_vals`, `g`, and `slot_logits` carry gradients through the commit operation. The `alpha` computation and the EMA update are differentiable tensor operations, so `total_loss.backward()` reaches the neuromodulator's continuous heads. In Phases B–C these heads are on the main optimizer; in Phase D they move to the RL optimizer. Only `commit_mask` is detached (it is a binary gate trained via RL, added in Phase D).

### 7.5 PM Eligibility-Only Reset (Phase E)

In lifelong mode (Phase E), doc boundaries call `reset_eligibility(mask)` instead of the full `reset_states(mask)`:

```python
def reset_eligibility(self, mask):
    # Zero only elig_K, elig_V for masked streams
    # pm_K, pm_V, pm_a persist (committed knowledge carries forward)
    expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
    self.elig_K = self.elig_K * (~expanded)
    self.elig_V = self.elig_V * (~expanded)
```

**Intuition:** Eligibility traces represent partial, uncommitted learning from the current document. They are stale across doc boundaries and should reset. But committed PM state (`pm_K`, `pm_V`, `pm_a`) has passed through the neuromodulator's selection logic and represents consolidated knowledge -- it should persist.

---

## 8. Episodic Memory (EM)

**File:** `src/model/episodic_memory.py`

**Intuition:** EM is the model's long-term event store. While PM learns reusable patterns (procedural knowledge), EM stores specific *episodes* -- particular token contexts that were surprising or novel. When the model encounters a situation similar to a stored episode, EM retrieves it, providing a form of "I've seen something like this before" recall.

**Instance count:** One EM per block. B=4 instances total (Tier A). Each EM is shared across all L layers within its block.

### 8.1 EM State

Per EM instance, per stream:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `em_K` | `[BS, M, D_em]` | Keys (unit-normalized). What to match against. |
| `em_V` | `[BS, M, D_em]` | Values. What to retrieve. |
| `em_S` | `[BS, M]` | Strengths. How "active" each slot is. Range [0, S_max]. |

`M=256` slots per bank. `D_em=128`.

### 8.2 EM Retrieval (every token)

```python
# 1. Compute query from input embedding + WM output
q = normalize(W_q_em(concat(x_emb, y_wm)))    # [BS, D_em]

# 2. Score against all M slots
scores = einsum("bd, bmd -> bm", q, em_K)      # [BS, M]
scores[em_S == 0] = -inf                        # mask inactive slots

# 3. Top-k retrieval (k_ret = 4)
topk_scores, topk_idx = scores.topk(k_ret)     # [BS, k_ret]
V_top = gather(em_V, topk_idx)                  # [BS, k_ret, D_em]

# 4. Cross-attention aggregation
q_cross = W_q_cross(x_emb)                     # [BS, D_em]
attn = softmax(q_cross @ V_top.T * scale)      # [BS, k_ret]
out = attn @ V_top                              # [BS, D_em]

# 5. Project to model dimension
y_em = W_o_cross(out)                           # [BS, D]
```

**Why cross-attention instead of weighted average?** Cross-attention lets the model learn a query-dependent weighting over the retrieved memory tokens, rather than averaging them by similarity score. This preserves the ability to selectively focus on one retrieved memory while ignoring others.

**Important:** The query uses `x_emb` (raw token embedding) and `y_wm` (working memory output) -- both are input-side features. The query does NOT use recurrent state `h`, which makes retrieval independent of the recurrence and avoids circular dependencies.

### 8.3 EM Candidate Proposals (every token, buffered)

During each span, the trainer buffers candidate (key, value, novelty) triples for later writing:

```python
k_cand = normalize(W_k_cand(concat(x_emb, y_wm)))  # [BS, D_em]
v_cand = W_v_cand(h_final)                          # [BS, D_em]

# Novelty = blend of surprise + distance from existing keys
cos_sim = k_cand @ em_K.T                            # [BS, M]
max_sim = cos_sim.max(dim=-1)                        # [BS]
novelty = clamp(0.5 * surprise + 0.5 * (1 - max_sim), 0, 1)
```

**Intuition:** A candidate is "novel" if it's both surprising (model predicted poorly) and dissimilar from existing EM entries (not a duplicate). The projections `W_k_cand` and `W_v_cand` are `nn.Linear` layers trained by backprop -- the model learns *what* to propose as memory candidates.

**Learned novelty weighting (Phase C+):** When `em_enabled=True`, `EpisodicMemory` creates `W_nov: nn.Linear(D + D, 1)` -- a per-token projection that replaces the hardcoded 0.5/0.5 weighting:

```python
w_nov = sigmoid(W_nov(concat(x, y_wm)))      # [BS, 1] -- learned weight
novelty = clamp(w_nov * surprise + (1 - w_nov) * (1 - max_sim), 0, 1)
```

`W_nov` lives on `EpisodicMemory` (not on `EMNeuromodulator`), so its name does not contain `"neuromodulator"` and it is excluded from `rl_parameters()`. It joins the **main optimizer** and is trained purely via backprop (differentiable through candidate selection -> EM writes -> retrieval -> loss). When EM is disabled (`em_enabled=False`), the hardcoded 0.5/0.5 weighting is preserved.

Candidates are **buffered for the entire span** (P=32 tokens), producing:
- `cand_K: [BS, P, D_em]`
- `cand_V: [BS, P, D_em]`
- `cand_score: [BS, P]`

### 8.4 EM Write (span boundary)

At the span boundary, candidates are selected and written:

1. **Neuromodulator decision:** `EMNeuromodulator.forward(span_surprise, em_usage, cand_novelty_mean)` returns a 3-tuple `(write_mask, g_em, p_write)`.

   In heuristic mode (Phases A–B — `em_enabled=False`):
   - `write_mask: [BS]` -- write if `cand_novelty_mean > 0.3`
   - `g_em: [BS]` -- fixed 0.3
   - `p_write: None`

   In continuous-learned mode (Phase C — `em_enabled=True`, `rl_enabled=False`):
   - `write_mask: [BS]` -- heuristic gate: `cand_novelty_mean > 0.3`
   - `g_em: [BS]` -- in `[g_em_floor, g_em_ceil]` (default [0.001, 0.95]) via `floor + (ceil - floor) * sigmoid(raw)` (differentiable, main optimizer; near-zero floor enables soft "don't write")
   - `p_write: None`

   In fully-learned mode (Phase D+ — `em_enabled=True`, `rl_enabled=True`):
   - `write_mask: [BS]` -- always True (every stream always writes)
   - `g_em: [BS]` -- in `[g_em_floor, g_em_ceil]` via same formula (differentiable, dual-trained: main loss backprop + RL counterfactual)
   - `p_write: None` -- removed (no binary gate in learned mode)

2. **For each candidate c** in the span (with `cand_valid` masking):
   - Score candidate key against all M existing slots (similarity + weakness bias)
   - Soft top-k selection of `k_write=4` slots to update
   - EMA blend: `em_K = normalize((1-alpha)*em_K + alpha*k_c)`, same for `em_V`
   - Strength update: `em_S += alpha * score`

3. **Decay + budget:** `em_S *= decay_em`, then scale to enforce `sum(em_S) <= budget_em`.

**Candidate validity:** The trainer tracks `cand_valid: [BS, P]` which masks out candidates from before a mid-span doc-boundary reset and from EOT positions. This prevents writing stale cross-document content.

### 8.5 EM Doc-Boundary Reset

**Critical design decision:** On doc boundary reset (Phases A-D), EM only zeros `em_S` (strengths). It preserves `em_K` and `em_V`. This is implemented via an override of `StateMixin.reset_states()` in `EpisodicMemory`.

**Why?** Zeroing strengths makes all slots invisible to retrieval (the `em_S > 0` mask filters them out) without destroying the actual key-value content. When new writes occur, they can reuse these slots via the weakness-biased slot selection (slots with `em_S=0` are preferred targets). This is more graceful than re-randomizing keys, which would destroy any useful patterns in the key space.

In Phase E (lifelong mode), EM is not reset at all -- `em_K`, `em_V`, and `em_S` all persist across document boundaries.

---

## 9. Blocks: Putting It Together

**File:** `src/model/block.py`

A Block is a self-contained processing unit: B blocks run in parallel, each handling `D_h = D/B` dimensions.

**Block.step(x_block, y_wm, x_emb, surprise, carry) per token:**
```python
# 1. EM retrieval (one EM per block, queries using x_emb and y_wm)
y_em = self.em.retrieve(x_emb, y_wm)           # [BS, D]

# 2. Project shared signals to per-block dimension
y_wm_proj = self.W_wm_proj(y_wm)               # [BS, D_h]
y_em_proj = self.W_em_proj(y_em)                # [BS, D_h]

# 3. Sequential layers (L=8)
x = x_block                                     # [BS, D_h]
for layer in self.layers:
    y_pm = layer.pm.apply(x)                     # PM read
    h = layer.step(x, y_pm, y_wm_proj, y_em_proj, surprise, carry)
    layer.pm.update_eligibility(x, h)            # accumulate elig traces
    x = h                                        # next layer's input

return x   # [BS, D_h]
```

**Block.commit_pm(force_mode, span_surprise) at span boundary:**
```python
def commit_pm(self, force_mode="normal", span_surprise=None):
    if force_mode == "force_off": return {}
    for layer in self.layers:
        if force_mode == "force_on":
            # Commit all streams with fixed defaults (g=0.5, lambda=decay)
            pm.commit(all_true_mask, default_lambda, default_g, None)
        else:
            # Neuromodulator decides
            surprise_input = span_surprise if span_surprise is not None else elig_norm
            mask, lam, g, slots, _p = layer.pm_neuromodulator(
                elig_norm, pm_usage / config.budget_pm, surprise_input
            )
            pm.commit(mask, lam, g, slots)
```

The `force_mode` parameter exists for Phase D RL counterfactual rollouts: `"force_on"` commits unconditionally with fixed defaults, `"force_off"` skips all commits. The `span_surprise` parameter receives the actual per-stream mean surprise over the span, computed in the trainer.

**Why `force_on` uses fixed defaults:** The counterfactual tests the binary question "did committing help or not?" -- it is not trying to evaluate the continuous parameters (lambda, g, slots). Using fixed defaults isolates the gate decision from continuous-output quality.

### 9.1 Block State Reset (Lifelong Mode)

`Block.reset_states(mask)` branches on `config.lifelong_mode`:

**Phases A-D (lifelong_mode=False):**
```python
for layer in self.layers:
    layer.reset_states(mask)     # zeros h
    layer.pm.reset_states(mask)  # zeros all PM state
self.em.reset_states(mask)       # zeros em_S
```

**Phase E (lifelong_mode=True):**
```python
for layer in self.layers:
    layer.reset_states(mask)         # zeros h (always)
    layer.pm.reset_eligibility(mask) # zeros only elig_K, elig_V
# EM fully persists -- no reset call
```

This is the core of the soft reset: transient state (h, eligibility) resets at doc boundaries, but committed PM weights and EM content persist across documents. Natural memory turnover is handled by existing decay (0.999/span) and budget enforcement.

---

## 10. Neuromodulators

Neuromodulators decide **when** and **how strongly** to write to memory. They operate at span boundaries only -- once every P=32 tokens. Each neuromodulator is an `nn.Module` that operates in one of three modes depending on whether its memory system is enabled and whether RL is active.

### 10.1 PMNeuromodulator

**File:** `src/model/procedural_memory.py`
**Attribute:** `layer.pm_neuromodulator`
**Count:** One per (block, layer) pair -- B*L=32 instances total.

**Inputs (3 features):**

| Feature | Source | Normalization |
|---------|--------|---------------|
| `elig_norm` | `pm.elig_K.norm().mean(dim=-1)` | Raw (typically 0-5) |
| `pm_usage` | `pm_a.sum() / budget_pm` | [0, 1] |
| `span_surprise` | Mean surprise over span | Raw (typically 0-10) |

**Outputs (5-tuple):**

| Output | Shape | Heuristic (Phase A) | Continuous-Learned (Phases B–C) | Fully-Learned (Phase D+) | Training |
|--------|-------|--------------------|---------------------------------|--------------------------|----------|
| `commit_mask` | `[BS]` | `elig_norm > 1.0` | `elig_norm > 1.0` (same heuristic) | `(p_commit > 0.5).detach()` | -- |
| `lambda_vals` | `[BS]` | `config.decay_pm` | sigmoid, scaled to `[decay_pm, 1.0]` | same | Main loss backprop |
| `g` | `[BS]` | `0.5` | sigmoid | same | Main loss backprop |
| `slot_logits` | `[BS, r]` | `None` | raw linear | same | Main loss backprop |
| `p_commit` | `[BS]` | `None` | `None` | sigmoid | RL (counterfactual) |

**Backbone + continuous heads** (created when `pm_enabled=True`, i.e. Phase B+):
```
3 inputs → Linear(3, H) → ReLU → {lambda_head, g_head, slot_head}
```

**Gate head** (added only when `rl_enabled=True`, i.e. Phase D+):
```
backbone → gate_head
```

Hidden size H = `config.rl_controller_hidden` (default 32). Without gate: ~458 params per instance. With gate: ~491 params per instance, ~15,700 total across 32 instances.

### 10.2 EMNeuromodulator

**File:** `src/model/episodic_memory.py`
**Attribute:** `block.em_neuromodulator`
**Count:** One per block -- B=4 instances total.

**Inputs (3 features):**

| Feature | Source | Normalization |
|---------|--------|---------------|
| `span_surprise` | Mean surprise over span | Raw |
| `em_usage` | `em_S.sum() / budget_em` | [0, 1] |
| `cand_novelty_mean` | Mean candidate novelty | Raw (typically 0-1) |

**Outputs (3-tuple):**

| Output | Shape | Heuristic (Phases A–B) | Continuous-Learned (Phase C) | Fully-Learned (Phase D+) | Training |
|--------|-------|----------------------|------------------------------|--------------------------|----------|
| `write_mask` | `[BS]` | `novelty > 0.3` | `novelty > 0.3` (same heuristic) | always True | -- |
| `g_em` | `[BS]` | `0.3` | `floor + (ceil - floor) * sigmoid(raw)`, in [0.001, 0.95] | same formula, dual-trained | Main loss backprop (Phase C); + RL weighted MSE (Phase D+) |
| `p_write` | `[BS]` | `None` | `None` | `None` (removed) | -- |

**Backbone + g_head** (created when `em_enabled=True`, i.e. Phase C+):
```
3 inputs → Linear(3, H) → ReLU → g_head
```
No `gate_head` -- fully-learned mode (Phase D+) always writes. The binary write gate was removed because writing is almost always beneficial when candidates have high novelty; the gate rarely learned to say "no." In continuous-learned mode (Phase C), the heuristic gate (`novelty > 0.3`) is preserved alongside the learned `g_em`. In Phase D, the RL signal additionally trains `g_em` via counterfactual rollouts.

**g_em safety rails:** `g_em = g_em_floor + (g_em_ceil - g_em_floor) * sigmoid(raw)` with defaults `g_em_floor=0.001`, `g_em_ceil=0.95`. The near-zero floor enables a soft "don't write" — the model can drive g_em close to zero for low-novelty spans, resulting in negligible alpha without requiring a binary gate. Ceiling prevents sigmoid saturation (leaves gradient room for weighted-MSE RL regression and backprop).

~161 params per instance, ~644 total across 4 instances.

### 10.3 Neuromodulator Parameter Handling

Since neuromodulators are `nn.Module` submodules of `Layer` and `Block` (which are in the model's `ModuleList`), their parameters are automatically included in `model.state_dict()` and `model.named_parameters()`. This means they are saved/loaded in checkpoints automatically.

The model provides `rl_parameters()` to yield only neuromodulator params (by filtering on `"neuromodulator" in name`). **Optimizer assignment depends on the phase:**

- **Phases B–C** (`rl_enabled=False`): `rl_param_ids` is empty, so neuromodulator params are NOT excluded from the main optimizer. The backbone + continuous heads are trained via main loss backprop alongside all other model params. No RL optimizer is created.
- **Phase D+** (`rl_enabled=True`): `rl_param_ids` is populated from `rl_parameters()`, so neuromodulator params ARE excluded from the main optimizer and assigned to a separate RL optimizer (Adam, `config.rl_lr`). They are also excluded from gradient clipping so continuous-head grads survive for combination with RL gradients.

---

## 11. Phase D -- Hybrid RL + Backprop Training

Neuromodulator backbones and continuous heads are created starting in Phases B/C (when their memory systems are enabled) and trained via main loss backprop. Phase D adds the RL-specific components on top:

- **PM gate output** (`p_commit`): Binary decision that cannot be differentiated through (commit_mask is detached). Created only in Phase D. Trained via **RL counterfactual rollouts**.
- **PM continuous outputs** (`lambda`, `g`, `slot_logits`): Created in Phase B. Flow through PM commit operations into future predictions. Trained via **main loss backprop**. In Phase D, they move from the main optimizer to the RL optimizer, where both main-loss backprop and RL gradients accumulate.
- **EM write strength** (`g_em`): Created in Phase C with heuristic gate. In Phase D, always writes (no binary gate). `g_em` is **dual-trained**: main loss backprop (through write -> retrieve -> loss) + RL counterfactual (weighted MSE regressing g_em toward whichever strength — chosen or baseline — performed better).
- **Learned novelty** (`W_nov`): Per-token projection on `EpisodicMemory`. Created in Phase C. Trained via **main loss backprop only** (main optimizer, not RL optimizer — stays on main optimizer even in Phase D).

In Phase D, both main-loss and RL gradient sources accumulate on neuromodulator parameters before a single RL optimizer step.

### 11.1 Why Not Backprop for Gates?

PM commits and EM writes update plain tensors (`pm_K`, `pm_V`, `pm_a`, `em_K`, `em_V`, `em_S`). The continuous outputs (`g`, `lambda`, `slot_logits`, `g_em`) flow through the EMA update math (which is differentiable), so backprop reaches them. But PM's `commit_mask` is a hard boolean -- you either commit or you don't. This binary gate is the `(p_commit > 0.5).detach()` call, which explicitly breaks the gradient. We need RL to train it.

**Note on EM:** The EM neuromodulator no longer has a binary gate. Instead of deciding "write or not," it decides "how strongly to write" via the continuous `g_em` output, clamped to [g_em_floor, g_em_ceil] (default [0.001, 0.95]). This is trained via deconfounded RL counterfactual rollouts comparing the neuromod's chosen `g_em` against a fixed baseline strength of 0.3, using weighted MSE to regress toward the better-performing strength.

### 11.2 Why Not REINFORCE?

REINFORCE would sample commit/write decisions, observe the resulting loss, and use `loss * log_prob` as the gradient. Problems:
- High variance: a single loss observation is noisy
- Credit assignment: which of the 32 PM instances and 4 EM instances caused the loss change?
- Baseline estimation: requires careful EMA baseline tuning

### 11.3 GRPO-Style Counterfactual Rollouts

Instead, we directly measure the causal effect of committing by comparing two futures:

**At selected span boundaries** (`config.rl_events_per_chunk=2` of the 8 boundaries per chunk, evenly spaced):

1. **Snapshot state** BEFORE PM commit + EM write (via `save_runtime_state()` + deepcopy). The snapshot also captures EM candidate buffers, eligibility norms, PM usages, EM novelties, and span surprise -- everything needed to re-run the neuromodulators during the update step.

2. **PM counterfactual — deconfounded per-target** (if PM enabled):
   - Select the top `rl_pm_targets_per_event` (default 1) PM controllers ranked by eligibility norm (`_select_pm_targets`).
   - **For each target `(b_idx, l_idx)`:**
     - **Rollout A (force_off):** Restore snapshot, force the target controller off (skip commit), all other controllers run normally. Run P=32 tokens forward. Measure per-stream loss.
     - **Rollout B (force_on):** Restore snapshot, force the target controller on (commit with fixed defaults g=0.5, lambda=decay), all other controllers run normally. Run P=32 tokens forward. Measure per-stream loss.
     - **Reward** = `loss_off - loss_on` (positive means committing helped)
     - Update **only the target controller** with this deconfounded reward.

3. **EM counterfactual — deconfounded per-target** (if EM enabled):
   - Select the top `rl_em_targets_per_event` (default 1) EM controllers ranked by candidate novelty (`_select_em_targets`).
   - Both arms always write (no more force_off). The counterfactual compares write *strength*.
   - **For each target `b_idx`:**
     - **Rollout A (baseline):** Restore snapshot, write at fixed default strength `g_em=0.3` for the target block, all other blocks run normally. Run P=32 tokens forward. Measure per-stream loss.
     - **Rollout B (chosen):** Restore snapshot, write at the neuromodulator's output `g_em` (captured in the snapshot) for the target block, all other blocks run normally. Run P=32 tokens forward. Measure per-stream loss.
     - **Reward** = `loss_baseline - loss_chosen` (positive means neuromod's strength was better than default)
     - Update **only the target controller** with this deconfounded reward.

4. **Gate/strength update:**
   - **PM (BCE):** Re-forward neuromodulator to get `p_commit` with grad. `label = (reward > 0).float()`, `weight = |reward| * credit`. `loss = BCE(p_commit, label, weight=weight)`. `loss.backward()`.
   - **EM (weighted MSE toward better policy):** Re-forward neuromodulator to get `g_em` with grad. The target is the empirically better strength: `target_g = chosen_g` if reward > 0 (neuromod was right), else `baseline_g` (default was better). Both `g_em` and `target_g` are normalized to [0,1] via `(g - floor) / (ceil - floor)`. `loss = weighted_mean((g_em_norm - target_norm)^2)`, weighted by `|reward| * credit`. `loss.backward()`. This regresses `g_em` toward whichever strength performed better, with gradient magnitude proportional to the performance gap.

5. **Restore** final real state and continue training.

### 11.4 Credit Assignment

Each rollout isolates a single controller (deconfounded), so the reward directly reflects that controller's contribution. Within a controller's update, the reward is further weighted by a **credit** term that reflects how active the controller was:

- **PM credit:** `credit = elig_norm / elig_norm.max()` — controllers with larger eligibility traces had more to commit.
- **EM credit:** `credit = novelty / novelty.max()` — blocks with more novel candidates had more to write.

**Target selection** (`_select_pm_targets`, `_select_em_targets`) picks the top-k most salient controllers per snapshot, controlled by `rl_pm_targets_per_event` and `rl_em_targets_per_event` (both default 1). This bounds rollout cost while focusing RL signal on the controllers with the most potential impact.

### 11.5 Optimizer Separation and Gradient Flow

The training step proceeds as:

```
1. Forward all T=256 tokens (normal training)
   → neuromodulators produce continuous outputs (lambda, g, slot_logits, g_em)
   → these flow through commit/write → affect PM/EM state → affect future predictions

2. total_loss.backward()
   → computes gradient on ALL params, including neuromodulator continuous heads
   → PM commit_mask was .detach(), so PM gate_head gets zero grad from this step
   → EM g_em is NOT detached — EM g_head gets continuous grad from this step too

3. clip_grad_norm_(main_params_only)
   → neuromodulator params EXCLUDED from clipping (their continuous grads preserved)

4. main_optimizer.step()
   → updates only main model params (neuromodulators excluded)

5. RL rollouts (if this chunk has rollout events)
   → snapshot was captured BEFORE commit/write during the forward pass
   → PM: per-target force_on/off rollouts measure deconfounded reward; BCE on p_commit
   → EM: per-target baseline/chosen rollouts measure deconfounded reward; weighted MSE on g_em
   → RL backward adds RL grad to existing continuous-head grad on same params

6. rl_optimizer.step()
   → updates neuromodulator params with COMBINED gradient:
     PM: continuous grads (lambda, g, slots) + gate grad (BCE on p_commit)
     EM: continuous grads (g_em via write→retrieve→loss) + RL grad (weighted MSE on g_em)
   → rl_optimizer.zero_grad()

   OR (if no rollout events this chunk):

6. rl_optimizer.step()
   → updates neuromodulators with continuous-head grads only
   → rl_optimizer.zero_grad()
```

This ensures continuous heads are always trained (every chunk), while gate heads receive RL signal on chunks that have rollout events.

### 11.6 Rollout Cost

Per rollout event: `2 * rl_pm_targets_per_event + 2 * rl_em_targets_per_event` forward passes (force_off + force_on per PM target, baseline + chosen per EM target), each running P=32 tokens. With defaults (1 PM target, 1 EM target) and 2 events per chunk: `2 * (2 + 2) * 32 = 256` extra forward tokens, ~100% overhead. With higher target counts the cost scales linearly. These are no-gradient forward passes (cheaper than training forward).

### 11.7 RL Metrics

The following metrics are logged to JSONL when RL is active:

| Metric | Description |
|--------|-------------|
| `rl_pm_reward_mean` | Mean reward across PM rollouts (positive = committing helped) |
| `rl_em_reward_mean` | Mean reward across EM rollouts (positive = neuromod's g_em beat baseline) |
| `rl_pm_gate_loss` | Mean BCE loss for PM gate heads |
| `rl_em_g_loss` | Mean weighted MSE loss for EM g_em (normalized to [0,1]) |
| `rl_pm_commit_rate` | Fraction of streams where `p_commit > 0.5` |
| `rl_em_write_rate` | Always 1.0 in learned mode (every stream writes) |
| `rl_pm_lambda_mean` | Mean learned lambda across all PM instances |
| `rl_pm_g_mean` | Mean learned g across all PM instances |
| `rl_em_g_mean` | Mean learned g_em across all EM instances |
| `rl_events` | Number of rollout events this step |
| `gnorm_b{b}_l{l}_pm_neuromod` | Per-instance PM neuromodulator grad norm |
| `gnorm_b{b}_em_neuromod` | Per-block EM neuromodulator grad norm |

Grad norms are captured BEFORE `rl_optimizer.zero_grad()` to ensure they reflect the combined gradient.

---

## 12. Spatial Decoder

**File:** `src/model/decoder.py`

**Toggle:** `config.snapshot_enabled` (default `False`). When off, the model uses the original path: `concat(h_blocks) → lm_head`. When on, intermediate layer outputs from all blocks feed through a three-level hierarchical decoder before the LM head.

### 12.1 Why a hierarchical decoder?

Without the spatial decoder, only the **final layer** of each block reaches the LM head. All intermediate layer outputs — which represent different levels of abstraction and carry different PM contributions — are discarded. This is like reading only the last page of each chapter.

The spatial decoder treats NML blocks as distributed specialized memory (like cortical regions) and applies a biologically-inspired aggregation hierarchy before language decoding.

### 12.2 Level 1 — Columnar Attention

**Analogy:** A cortical column integrating across its laminar layers.

Each block has L=8 layers. A learned **summary query** cross-attends to all L layer outputs, which are tagged with learned layer-position embeddings. Produces one column summary `[BS, D_h]` per block.

```
Layer outputs: [h_0, h_1, ..., h_7]  each [BS, D_h]
       + layer_emb(0..7)
                ↓
       Learned query cross-attends
                ↓
       Column summary: [BS, D_h]
```

One `ColumnarAttention` instance per block (B=4 total). Self-attention over 8 tokens — negligible compute.

### 12.3 Level 2 — Thalamic Integrator

**Analogy:** The thalamus binding cortical regions with different memory systems.

Integrates B column summaries (cortical processing) with explicit memory readouts:

| Token type | Count | Source | Embedding |
|------------|-------|--------|-----------|
| Cortical | B=4 | Column summaries from Level 1 | type=cortical + block position |
| PM | 1 | Strength-weighted PM slot average | type=procedural |
| EM | 1 | Strength-weighted EM slot average | type=episodic |
| WM | 1 | WM step output | type=working |

Total: B+3 = 7 input tokens, all projected to `d_dec`.

K learned output queries (default K=4) cross-attend to these 7 tokens, producing K **integrated memory tokens** `[BS, K, d_dec]`.

**PM summary:** For each PM instance (B*L=32), compute strength-weighted readout `sum(pm_a * pm_V) / sum(pm_a)`, then average across all instances → `[BS, D_h]`.

**EM summary:** For each EM instance (B=4), compute strength-weighted readout `sum(em_S * em_V) / sum(em_S)`, then average → `[BS, D_em]`.

When PM or EM is disabled/uninitialized, zero vectors are used (the thalamic integrator still has those token positions — they just carry no signal).

### 12.4 Level 3 — Deep Decoder

**Analogy:** Language production area (Broca's area) reading from organized memory.

A single query token (projected from `h_final`) passes through `decoder_layers` (default 2) pre-norm transformer decoder layers. Each layer does:

1. Self-attention (on the single query)
2. Cross-attention to K integrated memory tokens from Level 2
3. FFN (GELU, expansion factor 4x)

```
q = query_proj(h_final)                          [BS, 1, d_dec]
for each decoder layer:
    q = norm1(q + self_attn(q))
    q = norm2(q + cross_attn(q, memory))          memory: [BS, K, d_dec]
    q = norm3(q + ffn(q))
context = output_proj(q.squeeze(1))               [BS, D]
h_decoded = h_final + context                     [BS, D]
logits = lm_head(h_decoded)                       [BS, vocab]
```

### 12.5 Small-init and gradient flow

`output_proj` is initialized with small random weights (`std=0.01`). At activation time:

```
h_decoded = h_final + output_proj(decoded)
           ≈ h_final + noise          (initially negligible)
```

This means `snapshot_enabled=True` initially produces near-identical logits to the non-snapshot path, while allowing gradients to flow through the entire decoder. Zero-init would kill all upstream gradients via the chain rule (`grad × 0 = 0`), preventing the decoder from ever learning. The small-init lets the decoder gradually learn to contribute useful context as training progresses while keeping the initial perturbation negligible.

### 12.6 Dimensions (Tier A)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_dec` | 256 | Decoder working dimension |
| `n_heads_decoder` | 4 | Attention heads (all three levels) |
| `decoder_layers` | 2 | Depth of Level 3 decoder |
| `thalamic_tokens` | 4 | Output tokens from Level 2 (K) |

Additional parameters: ~3M on Tier A (~6% overhead). Compute: cross-attention over sequences of length 8, 7, and 4 — effectively free compared to the O(D_h^2) GRU recurrence per layer.

### 12.7 Gradient flow benefit

The spatial decoder creates a **shortcut gradient path** from the loss through the decoder, through the columnar attention, directly to every intermediate layer output. This bypasses the serial GRU chain (8 layers deep per block), giving early layers direct gradient signal. This is especially valuable for deep stacks (L=8+) where vanishing gradients through the recurrence chain can starve early layers.

---

## 13. State Management

**File:** `src/model/utils.py` (StateMixin), `src/model/state.py` (bulk operations)

### 13.1 StateMixin

All stateful modules inherit from `StateMixin`, which provides:

- **`_state_tensor_names`**: Class variable listing runtime state attributes (e.g., `["pm_K", "pm_V", "pm_a", "elig_K", "elig_V"]`)
- **`detach_states()`**: Detaches all state tensors from the computation graph (TBPTT boundary)
- **`reset_states(mask)`**: Zeros state for masked streams (doc boundary)
- **`state_dict_runtime()` / `load_state_runtime()`**: Serialize/deserialize for checkpointing and RL rollouts

**Convention:** Runtime state tensors are plain attributes, NOT `nn.Parameter` or `register_buffer`. They are updated explicitly -- either under `torch.no_grad()` (for PM/EM slot updates) or via differentiable operations (for eligibility, recurrent h).

### 13.2 State Reset Semantics

**Phases A-D (lifelong_mode=False):**

| Module | What resets on doc boundary | What is preserved |
|--------|---------------------------|-------------------|
| Layer | `h` (hidden state) | -- |
| PM | `pm_K, pm_V, pm_a, elig_K, elig_V` (all zeroed) | -- |
| EM | `em_S` only (strengths zeroed) | `em_K, em_V` preserved |
| WM | `wm_valid` cleared, `wm_ptr` reset | -- |
| Model | `surprise` zeroed for masked streams | -- |

**Phase E (lifelong_mode=True) -- Soft Reset:**

| Module | What resets on doc boundary | What persists |
|--------|---------------------------|---------------|
| Layer | `h` (hidden state) | -- |
| PM | `elig_K, elig_V` only | `pm_K, pm_V, pm_a` (committed knowledge) |
| EM | -- (nothing resets) | `em_K, em_V, em_S` (all persist) |
| WM | `wm_valid` cleared, `wm_ptr` reset | -- |
| Model | `surprise` zeroed for masked streams | -- |

The EM override in Phases A-D is deliberate: zeroing strengths makes old memories invisible without destroying key-value content, allowing graceful slot reuse. In Phase E, even strengths persist -- natural decay and budget enforcement provide forgetting pressure instead.

### 13.3 Bulk Operations (state.py)

```python
detach_all(model)          # Walk tree, detach all StateMixin states
reset_all(model, mask)     # Walk tree, reset masked streams
save_runtime_state(model)  # Serialize all state for checkpointing / RL rollouts
load_runtime_state(model, state)  # Restore saved state
```

**Path-based checkpoint keys:** `save_runtime_state()` uses stable module paths as dictionary keys (e.g., `"blocks.0.layers.1.pm"`, `"blocks.2.em"`). These keys are derived from the module tree structure and remain stable across unrelated model changes. `load_runtime_state()` first tries path-based keys, then falls back to legacy index-based keys for backward compatibility.

**Critical for RL:** `save_runtime_state()` returns references to live tensors. For counterfactual rollouts that mutate state, you **must** deepcopy the saved state before loading it. The trainer uses a dedicated `_detached_runtime_state()` helper that clones + detaches all tensors.

---

## 14. Training Loop (TBPTT + Spans)

**File:** `src/training/trainer.py`

### 14.1 Two Time Scales

| Scale | Length | What happens |
|-------|--------|--------------|
| **TBPTT chunk** | T=256 tokens | Autograd truncation. One backward pass per chunk. States detached after. |
| **Plasticity span** | P=32 tokens | PM/EM are read-only within. Commits and writes happen at span boundaries. |

A chunk contains T/P = 8 spans. Each span boundary triggers PM base decay + commits + EM writes.

### 14.2 Chunk Processing (train_chunk)

```
For each span (8 spans per chunk):
    Reset EM candidate buffers
    Reset per-stream span accumulators (surprise, valid token counts)
    Track per-stream last-reset position for candidate validity

    For each token in span (32 tokens):
        1. Compute doc-boundary reset mask
        2. Clear span accumulators for streams that reset mid-span
        3. forward_one_token() -> logits, x_emb, y_wm
        4. Compute + accumulate loss (online, masked at EOT)
        5. Update surprise signal (masked: EOT positions get 0)
        6. Accumulate per-stream span surprise (only valid positions)
        7. Buffer EM candidates with validity mask (if enabled)

    At span boundary:
        8.  Compute per-stream span_surprise_mean
        9.  Pre-compute EM candidate stacks + novelty (for snapshot + write)
       10.  [Phase D] Capture RL snapshot BEFORE commit/write
       11.  PM base decay (all streams): pm_a *= decay_pm
       12.  PM commits: neuromodulator decides, pm.commit()
       13.  EM writes: neuromodulator decides, write_at_boundary()
            filters candidates by validity (post-reset only)

After all spans:
    14. Finalize loss: avg_loss = chunk_loss / valid_count
    15. Add regularizers (PM/EM budget penalties)
    16. optimizer.zero_grad() + rl_optimizer.zero_grad()
    17. total_loss.backward()
    18. Clip gradients (main params only; neuromodulator grads excluded)
    19. main_optimizer.step() + scheduler.step()
    20. [Phase D] RL rollouts + rl_optimizer.step()
        OR rl_optimizer.step() (continuous grads only, no rollout events)
    21. Detach all states (TBPTT boundary)
    22. Save last token per stream (for checkpoint resume)
```

**Step 10 is critical:** The RL snapshot is captured BEFORE PM commit and EM write. This means the rollout can test different commit/write strategies and measure the downstream effect on the next span's tokens. PM rollouts use force_on/force_off. EM rollouts use baseline (g_em=0.3) vs chosen (neuromod's g_em). The snapshot includes:
- Full runtime state (deepcopy, detached)
- EM candidate buffers (detached clones)
- Per-instance eligibility norms and PM usages
- Per-block EM novelties and usages
- Per-block neuromod's chosen g_em (for EM "chosen" arm)
- Span surprise mean

### 14.3 Online Loss Accumulation

We never materialize `[BS, T, vocab]` logits. Each token step produces `[BS, vocab]`, and loss is accumulated as a running sum:

```python
token_loss, count = online_cross_entropy(logits, targets, loss_mask)
chunk_loss = chunk_loss + token_loss
valid_count += count
```

This is critical for VRAM: `[32, 256, 32000]` in float32 would be ~1GB per chunk.

### 14.4 Loss Masking

When `reset_on_doc_boundary=True` (all phases):
- Skip loss at positions where `input_tokens[:, t] == eot_id`
- This avoids training on cross-document transitions (predicting first token of next doc from EOT is meaningless)
- We still train on predicting EOT within documents
- Note: `reset_on_doc_boundary` remains True even in Phase E (lifelong mode). The `lifelong_mode` flag controls state persistence, not loss masking.

### 14.5 PM Commit Surprise Signal

The trainer computes `span_surprise_mean` (per-stream mean surprise over valid tokens in the span) and passes it to `model.commit_at_boundary(span_surprise=span_surprise_mean.detach())`. This flows through to each `block.commit_pm(span_surprise=...)`, where the neuromodulator receives it as its `span_surprise` input. The `.detach()` ensures the commit decision path does not interfere with the main LM gradient.

If `span_surprise` is not available (e.g., standalone evaluation without the trainer), the neuromodulator falls back to using `elig_norm` as a proxy.

### 14.6 Checkpoint Resume and prev_token Persistence

When training resumes from a checkpoint, the dataloader creates fresh streams with `prev_token = EOS`. This would trigger doc-boundary resets on the first batch, wiping restored PM/EM state. To prevent this:

1. The trainer tracks `_last_prev_token` (the final input token per stream from each chunk).
2. Checkpoints save this as `last_prev_token`.
3. On resume, `trainer.override_prev_token` is set from the checkpoint. On the first batch, this overrides the dataloader's `prev_token`, preserving memory state continuity.

---

## 15. Data Pipeline (Persistent Streams)

**File:** `src/data/streaming.py`

### 15.1 Persistent Parallel Streams

This is fundamentally different from transformer data loading:

- **Transformer:** Each batch item is an independent sequence. No state persists between batches.
- **Neuromorphic:** BS=32 independent *streams* that persist across TBPTT chunks. Model state (h, pm_K, em_K, etc.) carries forward. Different streams hit document boundaries at different positions.

Each stream is a continuous flow of documents separated by `<|endoftext|>`:

```
Stream 0: [doc1 tokens...] <eot> [doc2 tokens...] <eot> [doc3...]
Stream 1: [docA tokens...] <eot> [docB tokens...] <eot> ...
...
```

### 15.2 StreamBatch

```python
@dataclass
class StreamBatch:
    input_ids: Tensor    # [BS, T] -- input tokens
    target_ids: Tensor   # [BS, T] -- targets (shifted by 1)
    prev_token: Tensor   # [BS] -- last token from previous chunk
```

`prev_token` is needed to detect doc boundaries at position t=0 of a chunk: if `prev_token == eot_id`, then stream was at a doc boundary when the previous chunk ended.

### 15.3 Document Boundary Timing

For token at position t:
- If `t == 0`: check `prev_token == eot_id`
- If `t > 0`: check `input_ids[:, t-1] == eot_id`

Reset is triggered **before** processing token t. This means if token t-1 was EOT, we reset state before processing token t (the first token of the new document).

---

## 16. Loss and Regularization

**File:** `src/training/loss.py`

### 16.1 Online Cross-Entropy

```python
def online_cross_entropy(logits, targets, loss_mask):
    # Only compute loss for valid positions (masked at EOT)
    loss = F.cross_entropy(logits[loss_mask], targets[loss_mask], reduction="sum")
    return loss, count
```

### 16.2 Regularizers

```python
def compute_regularizers(model):
    reg = 0
    # PM budget penalty: penalize when sum(pm_a) approaches budget
    for each PM instance:
        excess = relu(sum(pm_a) - budget * 0.9)
        reg += excess.mean() * 0.01
    # EM budget penalty: same for sum(em_S)
    for each EM instance:
        excess = relu(sum(em_S) - budget * 0.9)
        reg += excess.mean() * 0.01
    return reg
```

**Intuition:** Soft penalty that pushes usage toward 90% of budget. The `budget_enforce()` function (hard cap via scaling) runs at commit/write time. The regularizer provides a gradient signal to the LM loss encouraging the model to not over-saturate memory.

---

## 17. Phased Training Plan

| Phase | Components | rl_enabled | lifelong_mode | Neuromod State | Goal |
|-------|-----------|------------|---------------|----------------|------|
| **A** | WM only | False | False | Empty shells (0 params) | Stable streaming, perplexity decreases |
| **B** | WM + PM | False | False | PM: backbone + continuous heads (main optimizer) | Memory bench improvement, learned commit strength |
| **C** | WM + PM + EM | False | False | PM + EM: backbone + continuous heads + W_nov (main optimizer) | Explicit recall, learned write strength + novelty |
| **D** | WM + PM + EM + RL | True | False | PM: + gate_head; EM: always-write + RL on g_em (RL optimizer) | Learned gating, RL counterfactual training |
| **E** | WM + PM + EM + lifelong | *inherited* | True | Inherited from prior phase | PM/EM persist across doc boundaries |

`config.set_phase("X")` toggles the appropriate flags:

```python
"A": wm=True,  pm=False, em=False, rl=False,     lifelong=False
"B": wm=True,  pm=True,  em=False, rl=False,     lifelong=False
"C": wm=True,  pm=True,  em=True,  rl=False,     lifelong=False
"D": wm=True,  pm=True,  em=True,  rl=True,      lifelong=False
"E": wm=True,  pm=True,  em=True,  rl=inherited,  lifelong=True
```

**Phase E inherits `rl_enabled`:** When transitioning from Phase D to Phase E, `set_phase("E")` does NOT force `rl_enabled` on or off. If the model was trained with Phase D neuromodulators, they remain active in Phase E. This allows lifelong mode to benefit from learned gating rather than falling back to heuristics. If transitioning from Phase C (no RL), Phase E uses heuristic neuromodulators.

**Phase transition via checkpoint resume:** `src/train.py` loads model checkpoints with `strict=False`, so new parameters are initialized fresh when they first appear. When transitioning A→B, PM neuromodulator backbone + continuous heads init fresh. B→C adds EM neuromodulator backbone + g_head + W_nov. C→D adds PM gate_head only — all existing backbone/continuous-head weights carry over. D→E preserves everything.

**Optimizer state across phase transitions:** When a phase transition is detected (checkpoint's `pm_enabled`/`em_enabled`/`rl_enabled` differ from current config), optimizer state loading is skipped because parameter group sizes change across phases (e.g., neuromod params move from main optimizer to RL optimizer in C→D). Both optimizers reinitialize with fresh Adam state. To mitigate the cold-start effect on the RL optimizer (which receives warm weights but has no momentum/variance history), a linear LR warmup ramps `rl_lr` from 0 to its configured value over `rl_warmup_steps` (default 500) steps. This prevents the effective learning rate spike that would otherwise destabilize well-tuned continuous heads.

---

## 18. Scaling Tiers

| Tier | D | L | B | ~Params | 4090 BS | Use Case |
|------|---|---|---|---------|---------|----------|
| **A** (Debug) | 512 | 8 | 4 | ~50M | 32-64 | Rapid iteration |
| **B** (Competitive) | 768 | 12 | 6 | ~150M | 16-32 | Match GPT-2 Small |
| **C** (Strong) | 1024 | 24 | 8 | ~350M | 8-16 | Match GPT-2 Medium |

All tiers keep D_h = D/B = 128 constant. Scaling comes from wider model (D), more layers (L), and more parallel blocks (B).

Neuromodulator overhead is negligible: ~16,500 params total (~0.04% of Tier A), because each MLP is tiny (3 inputs -> 32 hidden -> small output).

---

## 19. Gradient Flow Map

Understanding what learns via backprop, what needs RL, and how the hybrid training works:

```
                    ┌─────────────────────────────────────────────────┐
                    │             BACKPROP GRADIENT PATH               │
                    │                                                  │
  LM Loss ◄── logits ◄── lm_head ◄── h_decoded ◄── h_final ◄── Layer.step()
                                           ▲                          ▲
                                 (if snapshot_enabled)   ┌────────────┴────────────┐
                                 SpatialDecoder ◄────────┤   (shortcut to ALL      │
                                  ▲  ▲  ▲  ▲            │    intermediate layers)  │
                          columnar│  │  │  │wm           │                         │
                          attn    │pm│em│               ┌────────────┴────────────┐
                                         │                         │
                                    gate_a, gate_b            residual
                                    (W_a, W_b params)     from x_block
                                         ▲
                                         │
                              u = concat(x_block, y_pm,
                                         y_wm_proj, y_em_proj,
                                         surprise)
                                    ▲         ▲         ▲
                                    │         │         │
                               PM.apply   WM.step   EM.retrieve
                               (read only) (W_q, W_k, (W_q_em, W_q_cross,
                                           W_v, W_o)   W_o_cross params)
                                    ▲                      ▲
                           ┌────────┘                      │
                     PM eligibility:              EM candidates:
                     W_k_pre, W_v_post            W_k_cand, W_v_cand, W_nov
                     (nn.Linear params)           (nn.Linear params)
                           │                           │
                    elig_K, elig_V              cand_K, cand_V, novelty
                           │                           │
    ┌──────────────────────┴──────┐   ┌────────────────┴──────────────────┐
    │       PM COMMIT              │   │          EM WRITE                  │
    │  ┌────────────────────────┐  │   │  ┌─────────────────────────────┐  │
    │  │   PMNeuromodulator     │  │   │  │     EMNeuromodulator        │  │
    │  │                        │  │   │  │                             │  │
    │  │  backbone → 4 heads:   │  │   │  │  backbone → g_head only    │  │
    │  │  gate  g  lambda slot  │  │   │  │  (no gate — always writes) │  │
    │  │   │    │    │     │    │  │   │  │         │                   │  │
    │  │   │    ╎    ╎     ╎    │  │   │  │  g_em = floor + range *    │  │
    │  │ .detach ╎   ╎     ╎    │  │   │  │         sigmoid(raw)       │  │
    │  │   │    ╎    ╎     ╎    │  │   │  │         │                   │  │
    │  │ commit ╎ alpha = g *   │  │   │  │  alpha = g_em * weights    │  │
    │  │ mask   ╎  weights      │  │   │  │  (DIFFERENTIABLE ──────────┤  │
    │  │ (bool) ╎    │          │  │   │  │  through write_at_boundary │  │
    │  │   │    ╎    │          │  │   │  │  into em_K / em_V / em_S)  │  │
    │  │   │    ╎    ▼          │  │   │  │         │                   │  │
    │  │   │    ╎ DIFFERENTIABLE│  │   │  │  RL (weighted MSE on       │  │
    │  │   │    ╎ through commit│  │   │  │  g_em): baseline vs chosen │  │
    │  │   │    ╎ into pm_K/V/a │  │   │  │  strength counterfactual  │  │
    │  │   │    ╎    │          │  │   │  └─────────────────────────────┘  │
    │  │ RL(BCE ╎    │          │  │   │                                    │
    │  │ on     ╎    │          │  │   │   affects FUTURE EM.retrieve ──────┘
    │  │ p_commit)   │          │  │   │         │
    │  └────────╎────┼──────────┘  │   │   Layer.step() → logits → loss
    │           ╎    │              │   │         │
    │   affects FUTURE PM.apply ───┘   │   loss.backward() reaches
    │           ╎    │                  │   EM g_head via write → retrieve
    │   Layer.step() → logits → loss   │
    │           ╎    │                  │
    │   loss.backward() reaches        │
    │   g_head, lambda_head,           │
    │   slot_head via alpha            │
    └───────────╎──────────────────────┘
                ╎
          RL optimizer combines:
          PM: continuous grads + gate grad (BCE on p_commit)
          EM: continuous grads + RL grad (weighted MSE on normalized g_em)
          → single rl_optimizer.step()

    Main optimizer (separate):
    W_nov (on EpisodicMemory) trained via main loss backprop only
```

**Summary:**
- **Learns via backprop (main optimizer):** All `nn.Parameter` weights -- gate projections, WM projections, EM query/output projections, PM eligibility projections, embedding, lm_head, `EpisodicMemory.W_nov` (learned novelty adjuster, Phase C+). In Phases B–C, neuromodulator backbone + continuous heads also train here.
- **Learns via backprop (RL optimizer, continuous heads) — Phase D+ only:** PM neuromodulator `lambda_head`, `g_head`, `slot_head`; EM neuromodulator `g_head` -- gradients flow through commit/write operations into future predictions. (In Phases B–C, these same heads are on the main optimizer instead.)
- **Learns via RL (RL optimizer, gate/strength heads) — Phase D+ only:** PM neuromodulator `gate_head` (trained via counterfactual rollout BCE on p_commit); EM neuromodulator `g_head` (trained via counterfactual rollout weighted MSE on normalized g_em toward the better-performing strength -- baseline vs chosen)
- **Evolves via explicit rules (no parameter grad):** pm_K, pm_V, pm_a, em_K, em_V, em_S -- updated at span boundaries by commit/write procedures
- **Not plastic (no memory update mechanism):** WM KV cache (ring buffer of recent token projections; gradients flow through within TBPTT chunks, detached at chunk boundaries)

---

## 20. Design Integrity Notes

This section documents the design decisions that were verified during the Phase D bug-fix audit (2026-02-07) and confirms that the core architecture remains intact.

### 20.1 Verified Invariants

1. **Snapshot timing:** RL snapshots are captured BEFORE PM commit + EM write. This ensures the counterfactual rollout can test different strategies (PM: force_on/force_off; EM: baseline vs chosen g_em) and measure the downstream effect.

2. **Rollout structure:** Rollouts apply forced commit/write BEFORE running the next span's tokens. This matches the normal training flow where commit happens at span boundary, then the next span reads from the updated PM/EM.

3. **Phase E inherits rl_enabled:** `set_phase("E")` does not touch `rl_enabled`. This allows D->E transitions to preserve learned neuromodulators, and C->E transitions to use heuristic neuromodulators.

4. **Continuous-head gradient path:** PM: `lambda_vals`, `g`, and `slot_logits` carry gradients through `pm.commit()` because the EMA update (`alpha = weights * g * mask`, `pm_K = (1-alpha)*pm_K + alpha*elig_K`) is composed of differentiable tensor operations. `loss.backward()` reaches these heads through: neuromodulator output -> alpha -> pm_K/pm_V update -> future `pm.apply(x)` -> layer output -> logits -> loss. EM: `g_em` carries gradients through `write_at_boundary()` via the same mechanism: g_em -> alpha -> em_K/em_V update -> future `EM.retrieve()` -> layer output -> logits -> loss. This gradient path is active starting in Phase B/C (when continuous heads are created), not just Phase D.

5. **Gradient isolation (Phase D only):** When `rl_enabled=True`, neuromodulator params are excluded from main gradient clipping (`clip_grad_norm_`) so continuous-head gradients survive for combination with RL gradients (PM gate BCE, EM g_em weighted MSE) in the RL optimizer step. In Phases B–C, neuromodulator params are on the main optimizer and DO participate in gradient clipping — this is intentional, as there is no RL gradient to preserve.

6. **EM novelty in rollouts:** RL snapshots store real candidate novelty (from `propose_candidate()`), not a proxy. This ensures `_update_em_neuromodulators` re-forwards the EMNeuromodulator with the same inputs it saw during the normal forward pass.

### 20.2 Intentional Design Decisions (Not Bugs)

- **PM `force_on` uses fixed defaults (g=0.5, lambda=decay):** The counterfactual tests the binary "commit vs. skip" question. Using fixed defaults isolates the gate decision.
- **EM counterfactual uses baseline strength (g_em=0.3) vs chosen strength:** Both arms always write. The counterfactual tests whether the neuromod's chosen `g_em` outperforms a fixed default, teaching the neuromod to deviate from the default only when beneficial.
- **4 forward passes per RL event** (PM force_off/on + EM baseline/chosen): PM and EM need independent counterfactuals because their effects are different (PM affects layer computation, EM affects block-level retrieval).
- **Surprise as neuromodulator input** (not just novelty): Surprise reflects the model's overall prediction quality, which is a useful signal for deciding whether to consolidate learning. Novelty is specific to EM candidates.

---

## File Index

| File | Primary Content |
|------|----------------|
| `src/model/config.py` | `ModelConfig` dataclass, tier presets, phase toggles, RL config fields |
| `src/model/model.py` | `NeuromorphicLM` -- top-level: embedding, WM, blocks, lm_head, `rl_parameters()` |
| `src/model/block.py` | `Block` -- parallel unit: L layers + 1 EM + 1 `EMNeuromodulator` |
| `src/model/layer.py` | `Layer` -- affine recurrence + PM instance + `PMNeuromodulator` |
| `src/model/working_memory.py` | `WorkingMemory` -- sliding window attention |
| `src/model/procedural_memory.py` | `ProceduralMemory`, `PMNeuromodulator` (three-mode: heuristic / continuous-learned / fully-learned) |
| `src/model/episodic_memory.py` | `EpisodicMemory`, `EMNeuromodulator` (three-mode: heuristic / continuous-learned / fully-learned) |
| `src/model/utils.py` | `StateMixin`, `unit_normalize`, `soft_topk`, `budget_enforce` |
| `src/model/state.py` | `save_runtime_state`, `load_runtime_state`, `detach_all`, `reset_all` |
| `src/training/trainer.py` | `TBPTTTrainer`, `BoundarySnapshot` -- chunk processing, span boundaries, RL rollouts |
| `src/training/loss.py` | `online_cross_entropy`, `compute_regularizers` |
| `src/training/eval_lifelong.py` | Phase E evaluation: domain adaptation, drift, cross-doc recall |
| `src/data/streaming.py` | `PersistentStreamDataset`, `StreamBatch`, `DocumentStream` |
| `src/train.py` | Entry point -- config, main + RL optimizers, scheduler, training loop, checkpoint save/load |
| `src/debug/collector.py` | `MetricsCollector` -- two-tier JSONL logging, gate stats, memory stats, grad norms |
