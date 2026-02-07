# Neuromorphic LM -- Full Model Explainer

**Purpose:** Independent reference for verifying that all code changes remain aligned with the design intent. Covers every module, its intuition, its concrete implementation, and how modules compose during training.

**Last verified against code:** 2026-02-07

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
10. [Controllers (Neuromodulation)](#10-controllers-neuromodulation)
11. [State Management](#11-state-management)
12. [Training Loop (TBPTT + Spans)](#12-training-loop-tbptt--spans)
13. [Data Pipeline (Persistent Streams)](#13-data-pipeline-persistent-streams)
14. [Loss and Regularization](#14-loss-and-regularization)
15. [Phase D -- RL via Counterfactual Rollouts](#15-phase-d----rl-via-counterfactual-rollouts)
16. [Phased Training Plan](#16-phased-training-plan)
17. [Scaling Tiers](#17-scaling-tiers)
18. [Gradient Flow Map](#18-gradient-flow-map)

---

## 1. Design Philosophy

The neuromorphic LM decomposes language model memory into four biologically-inspired systems, each with different persistence and update mechanics:

| System | Biological Analogy | Persistence | Update Mechanism |
|--------|--------------------|-------------|------------------|
| **Genetic memory** (slow weights) | DNA / long-term evolutionary adaptation | Permanent after training | Standard backprop (frozen at deployment) |
| **Working memory** (WM) | Prefrontal cortex / scratchpad | Last W tokens | Sliding window attention (no plasticity) |
| **Procedural memory** (PM) | Cerebellum / skill memory | Across documents | Eligibility traces + neuromodulated commits |
| **Episodic memory** (EM) | Hippocampus / event memory | Across documents | Novelty-based writes to vector store |

**Why not just use a transformer?** Transformers have a single memory mechanism (KV cache) that scales linearly with context. Our model separates *what* to remember from *how long* to remember it. WM handles precise short-range copying. PM learns reusable patterns (like skills). EM stores specific surprising events for later recall. Each system can be independently controlled, budgeted, and (eventually) learned via RL.

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
                     └────────┬───────────┘
                              │
                     concat all blocks ──► h_final: [BS, D]
                              │
                         LM Head ──► logits: [BS, vocab]
```

**Dimensions (Tier A defaults):** D=512, B=4, L=8, D_h=128, vocab=32000

**Instance counts:**
- 1 WM (shared across model)
- B=4 EM banks (one per block)
- B*L=32 PM instances (one per layer per block)
- B*L=32 PMControllers
- B=4 EMControllers

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
logits = self.lm_head(h_final)         # [BS, vocab]

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
2. Write `(k, v)` into ring buffer at `wm_ptr` (detached -- KV cache is not differentiable)
3. Multi-head attention (`n_heads=4`) over all valid positions in the buffer
4. Project output: `y_wm = W_o(attn_output)` -> `[BS, D]`
5. Advance pointer: `wm_ptr = (wm_ptr + 1) % W`

**Doc boundary reset:** Clears `wm_valid` and resets `wm_ptr` to 0 for masked streams. This prevents cross-document information leakage.

**Why detached KV cache?** The keys and values written into the cache are detached from the computation graph. This is deliberate: WM is a *non-plastic* cache. It stores exact copies of recent token representations but does not learn to update itself post-training. Gradients flow through the query path and output projection, which is sufficient for the model to learn *how to read* from WM.

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

**Output:** `norm(h + x_block)` -- residual connection from block input plus LayerNorm.

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

**Intuition:** Eligibility traces are a running average of "what the model wanted to write." The projections `W_k_pre` and `W_v_post` are learned parameters -- gradients from the LM loss flow back through them, teaching the model *what* constitutes a good key-value pair to store. But the *decision* of whether to actually commit these traces into PM is made by the controller at span boundaries.

`rho=0.95` is the eligibility decay -- recent tokens contribute more than older ones. This creates a recency-weighted summary of the span's proposed writes.

**Key insight:** Eligibility is **differentiable** (no `torch.no_grad()`). The projections `W_k_pre` and `W_v_post` are `nn.Linear` layers trained by backprop. The model learns what to propose; the controller learns when to commit.

### 7.4 PM Commit (span boundary only, under no_grad)

Called every P=32 tokens. The commit process:

1. **Base decay** (all streams): `pm_a *= decay_pm` (0.999). Ensures non-committing streams gradually lose strength, preventing stale slots from persisting forever.

2. **Controller decision:** `PMController.forward(elig_norm, pm_usage, surprise)` returns `(commit_mask, lambda_vals, g, slot_logits)`.
   - `commit_mask: [BS]` -- which streams commit (heuristic: `elig_norm > 1.0`)
   - `lambda_vals: [BS]` -- commit-time decay (heuristic: `config.decay_pm`)
   - `g: [BS]` -- write strength (heuristic: 0.5)
   - `slot_logits: None` -- slot selection bias (heuristic: similarity-based)

3. **Slot selection** (soft top-k):
   ```python
   scores = einsum(pm_K, elig_K_norm)       # [BS, r] -- similarity
   scores -= weakness_weight * pm_a          # prefer overwriting weak slots
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

**Why under no_grad?** The commit updates `pm_K`, `pm_V`, `pm_a` which are plain tensors, not parameters. They evolve via explicit update rules, not gradient descent. This is by design -- these are "fast weights" that adapt at runtime. The gradient signal for *what* to commit flows through the eligibility projections (which are `nn.Parameter`).

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

**Intuition:** Eligibility traces represent partial, uncommitted learning from the current document. They are stale across doc boundaries and should reset. But committed PM state (`pm_K`, `pm_V`, `pm_a`) has passed through the controller's selection logic and represents consolidated knowledge -- it should persist.

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

Candidates are **buffered for the entire span** (P=32 tokens), producing:
- `cand_K: [BS, P, D_em]`
- `cand_V: [BS, P, D_em]`
- `cand_score: [BS, P]`

### 8.4 EM Write (span boundary, under no_grad)

At the span boundary, top C=8 candidates per stream are selected and written:

1. **Select top-C:** `topk(cand_score, C)` -- pick the 8 most novel candidates from the span.

2. **For each candidate c** (loop over C=8):
   - Score candidate key against all M existing slots (similarity + weakness bias)
   - Soft top-k selection of `k_write=4` slots to update
   - EMA blend: `em_K = normalize((1-alpha)*em_K + alpha*k_c)`, same for `em_V`
   - Strength update: `em_S += alpha * score`

3. **Decay + budget:** `em_S *= decay_em`, then scale to enforce `sum(em_S) <= budget_em`.

**Controller decision:** `EMController.forward(surprise, em_usage, novelty)` returns `(write_mask, g_em)`.
- `write_mask: [BS]` -- which streams write (heuristic: `novelty_mean > 0.3`)
- `g_em: [BS]` -- write strength (heuristic: fixed 0.3)

### 8.5 EM Doc-Boundary Reset

**Critical design decision:** On doc boundary reset, EM only zeros `em_S` (strengths). It preserves `em_K` and `em_V`. This is implemented via an override of `StateMixin.reset_states()` in `EpisodicMemory`.

**Why?** Zeroing strengths makes all slots invisible to retrieval (the `em_S > 0` mask filters them out) without destroying the actual key-value content. When new writes occur, they can reuse these slots via the weakness-biased slot selection (slots with `em_S=0` are preferred targets). This is more graceful than re-randomizing keys, which would destroy any useful patterns in the key space.

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

**Block.commit_pm(force_mode) at span boundary:**
```python
def commit_pm(self, force_mode="normal"):
    if force_mode == "force_off": return
    for layer in self.layers:
        if force_mode == "force_on":
            # Commit all streams unconditionally
            pm.commit(all_true_mask, default_lambda, default_g, None)
        else:
            # Controller decides
            mask, lam, g, slots = layer.pm_controller(elig_norm, pm_usage, surprise)
            pm.commit(mask, lam, g, slots)
```

The `force_mode` parameter exists for Phase D RL counterfactual rollouts.

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

## 10. Controllers (Neuromodulation)

Controllers decide **when** and **how strongly** to write to memory. They operate at span boundaries only -- once every P=32 tokens.

### 10.1 PMController (Heuristic, Phases A-C)

**File:** `src/model/procedural_memory.py`

One per (block, layer) pair. Currently a simple threshold:

```python
def forward(self, elig_norm, pm_usage, span_surprise):
    commit_mask = elig_norm > 1.0           # commit if enough eligibility
    lambda_vals = full(BS, config.decay_pm) # default decay
    g = full(BS, 0.5)                       # default write strength
    return commit_mask, lambda_vals, g, None
```

**Intuition:** "If the eligibility traces have built up enough signal (norm > 1.0), commit them to PM. Use default strength and decay."

### 10.2 EMController (Heuristic, Phases A-C)

**File:** `src/model/episodic_memory.py`

One per block:

```python
def forward(self, span_surprise, em_usage, cand_novelty_mean):
    write_mask = cand_novelty_mean > 0.3    # write if candidates are novel enough
    g_em = full_like(span_surprise, 0.3)    # fixed write strength
    return write_mask, g_em
```

**Intuition:** "If the average novelty of this span's candidates exceeds 0.3, write them to EM. Use a conservative write strength of 0.3."

### 10.3 RL Controllers (Phase D, planned)

Phase D replaces heuristic controllers with learned MLPs trained via counterfactual RL. See [Section 15](#15-phase-d----rl-via-counterfactual-rollouts).

---

## 11. State Management

**File:** `src/model/utils.py` (StateMixin), `src/model/state.py` (bulk operations)

### 11.1 StateMixin

All stateful modules inherit from `StateMixin`, which provides:

- **`_state_tensor_names`**: Class variable listing runtime state attributes (e.g., `["pm_K", "pm_V", "pm_a", "elig_K", "elig_V"]`)
- **`detach_states()`**: Detaches all state tensors from the computation graph (TBPTT boundary)
- **`reset_states(mask)`**: Zeros state for masked streams (doc boundary)
- **`state_dict_runtime()` / `load_state_runtime()`**: Serialize/deserialize for checkpointing and RL rollouts

**Convention:** Runtime state tensors are plain attributes, NOT `nn.Parameter` or `register_buffer`. They are updated explicitly under `torch.no_grad()` (for PM/EM writes) or via differentiable operations (for eligibility, recurrent h).

### 11.2 State Reset Semantics

**Phases A-D (lifelong_mode=False):**

| Module | What resets on doc boundary | What is preserved |
|--------|---------------------------|-------------------|
| Layer | `h` (hidden state) | -- |
| PM | `pm_K, pm_V, pm_a, elig_K, elig_V` (all zeroed) | -- |
| EM | `em_S` only (strengths zeroed) | `em_K, em_V` preserved |
| WM | `wm_valid` cleared, `wm_ptr` reset | -- |
| Model | `surprise` zeroed for masked streams | -- |

**Phase E (lifelong_mode=True) — Soft Reset:**

| Module | What resets on doc boundary | What persists |
|--------|---------------------------|---------------|
| Layer | `h` (hidden state) | -- |
| PM | `elig_K, elig_V` only | `pm_K, pm_V, pm_a` (committed knowledge) |
| EM | -- (nothing resets) | `em_K, em_V, em_S` (all persist) |
| WM | `wm_valid` cleared, `wm_ptr` reset | -- |
| Model | `surprise` zeroed for masked streams | -- |

The EM override in Phases A-D is deliberate: zeroing strengths makes old memories invisible without destroying key-value content, allowing graceful slot reuse. In Phase E, even strengths persist -- natural decay and budget enforcement provide forgetting pressure instead.

### 11.3 Bulk Operations (state.py)

```python
detach_all(model)          # Walk tree, detach all StateMixin states
reset_all(model, mask)     # Walk tree, reset masked streams
save_runtime_state(model)  # Serialize all state for checkpointing / RL rollouts
load_runtime_state(model, state)  # Restore saved state
```

**Critical for RL:** `save_runtime_state()` returns references to live tensors. For counterfactual rollouts that mutate state, you **must** `copy.deepcopy()` the saved state before loading it.

---

## 12. Training Loop (TBPTT + Spans)

**File:** `src/training/trainer.py`

### 12.1 Two Time Scales

| Scale | Length | What happens |
|-------|--------|--------------|
| **TBPTT chunk** | T=256 tokens | Autograd truncation. One backward pass per chunk. States detached after. |
| **Plasticity span** | P=32 tokens | PM/EM are read-only within. Commits and writes happen at span boundaries. |

A chunk contains T/P = 8 spans. Each span boundary triggers PM base decay + commits + EM writes.

### 12.2 Chunk Processing (train_chunk)

```
For each span (8 spans per chunk):
    Reset EM candidate buffers

    For each token in span (32 tokens):
        1. Compute doc-boundary reset mask
        2. forward_one_token() -> logits, x_emb, y_wm
        3. Compute + accumulate loss (online, masked at EOT)
        4. Update surprise signal
        5. Accumulate span surprise for controller decisions
        6. Buffer EM candidates (if enabled)

    At span boundary:
        7. PM base decay (all streams): pm_a *= decay_pm
        8. PM commits: controller decides, commit() updates pm_K/pm_V/pm_a
        9. EM writes: controller decides, write_at_boundary() updates em_K/em_V/em_S

After all spans:
    10. Finalize loss: avg_loss = chunk_loss / valid_count
    11. Add regularizers (PM/EM budget penalties)
    12. Backward + gradient clipping + optimizer step
    13. Detach all states (TBPTT boundary)
```

### 12.3 Online Loss Accumulation

We never materialize `[BS, T, vocab]` logits. Each token step produces `[BS, vocab]`, and loss is accumulated as a running sum:

```python
token_loss, count = online_cross_entropy(logits, targets, loss_mask)
chunk_loss = chunk_loss + token_loss
valid_count += count
```

This is critical for VRAM: `[32, 256, 32000]` in float32 would be ~1GB per chunk.

### 12.4 Loss Masking

When `reset_on_doc_boundary=True` (Phases A-E):
- Skip loss at positions where `input_tokens[:, t] == eot_id`
- This avoids training on cross-document transitions (predicting first token of next doc from EOT is meaningless)
- We still train on predicting EOT within documents
- Note: `reset_on_doc_boundary` remains True even in Phase E (lifelong mode). The `lifelong_mode` flag controls state persistence, not loss masking.

### 12.5 Why PM Commits Use elig_norm as Surprise Proxy

Currently, `block.commit_pm()` is called from `model.commit_at_boundary()`, which does not pass span surprise statistics. The PMController receives `elig_norm` as a proxy for `span_surprise`. This is a known simplification -- when Phase D RL controllers are implemented, they will receive proper surprise statistics computed in the trainer's rl_step method.

---

## 13. Data Pipeline (Persistent Streams)

**File:** `src/data/streaming.py`

### 13.1 Persistent Parallel Streams

This is fundamentally different from transformer data loading:

- **Transformer:** Each batch item is an independent sequence. No state persists between batches.
- **Neuromorphic:** BS=32 independent *streams* that persist across TBPTT chunks. Model state (h, pm_K, em_K, etc.) carries forward. Different streams hit document boundaries at different positions.

Each stream is a continuous flow of documents separated by `<|endoftext|>`:

```
Stream 0: [doc1 tokens...] <eot> [doc2 tokens...] <eot> [doc3...]
Stream 1: [docA tokens...] <eot> [docB tokens...] <eot> ...
...
```

### 13.2 StreamBatch

```python
@dataclass
class StreamBatch:
    input_ids: Tensor    # [BS, T] -- input tokens
    target_ids: Tensor   # [BS, T] -- targets (shifted by 1)
    prev_token: Tensor   # [BS] -- last token from previous chunk
```

`prev_token` is needed to detect doc boundaries at position t=0 of a chunk: if `prev_token == eot_id`, then stream was at a doc boundary when the previous chunk ended.

### 13.3 Document Boundary Timing

For token at position t:
- If `t == 0`: check `prev_token == eot_id`
- If `t > 0`: check `input_ids[:, t-1] == eot_id`

Reset is triggered **before** processing token t. This means if token t-1 was EOT, we reset state before processing token t (the first token of the new document).

---

## 14. Loss and Regularization

**File:** `src/training/loss.py`

### 14.1 Online Cross-Entropy

```python
def online_cross_entropy(logits, targets, loss_mask):
    # Only compute loss for valid positions (masked at EOT)
    loss = F.cross_entropy(logits[loss_mask], targets[loss_mask], reduction="sum")
    return loss, count
```

### 14.2 Regularizers

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

## 15. Phase D -- RL via Counterfactual Rollouts

**Plan file:** `.claude/plans/happy-sniffing-valley.md`

**Status:** Planned, not yet implemented. Infrastructure (force_mode in commit_pm, state save/restore) is in place.

### 15.1 Why Not Backprop?

PM commits and EM writes happen under `torch.no_grad()`. The *what* to write is learned via backprop (eligibility projections, candidate projections), but the *when* to write (binary commit/write decisions) cannot receive gradients. We need a different training signal.

### 15.2 Why Not REINFORCE?

REINFORCE (policy gradient) would sample commit/write decisions, observe the resulting loss, and use `loss * log_prob` as the gradient. Problems:
- High variance: a single loss observation is noisy
- Credit assignment: which of the 32 PM instances and 4 EM instances caused the loss change?
- Baseline estimation: requires careful EMA baseline tuning

### 15.3 GRPO-Style Counterfactual Rollouts

Instead, we directly measure the causal effect of committing by comparing two futures:

1. **Run the normal TBPTT forward pass.** Collect per-span mean surprise.

2. **Select events:** Pick the k=2 span boundaries with highest mean surprise. These are the moments where the model is most uncertain, and where memory writes are most likely to matter.

3. **For each event (span boundary s):**
   - Save full model runtime state (deepcopy)
   - **Rollout A (force OFF):** Reload state, skip all PM commits and EM writes at this boundary, then run P=32 tokens forward. Measure total loss.
   - **Rollout B (force ON):** Reload state, force all PM commits and EM writes, then run P=32 tokens forward. Measure total loss.
   - **Reward** = loss_A - loss_B (positive means memory actions helped)
   - Subtract penalties for memory usage growth and weight drift

4. **Controller update:**
   - Reload state at the boundary
   - Forward each controller to get `p_commit` / `p_write` (sigmoid outputs)
   - Train via **weighted BCE**: label = (reward > 0), weight = |reward| * per-instance credit
   - Gradient only flows through controller parameters

### 15.4 Credit Assignment

The reward from a single rollout reflects the *combined* effect of all PM commits and EM writes. To distribute credit:

- **PM credit:** Proportional to eligibility norm. Instances with larger eligibility traces contributed more to the commit.
- **EM credit:** Proportional to candidate novelty. Blocks with more novel candidates contributed more to the write.

```
credit_pm[i] = elig_norm[i] / sum(elig_norms) * num_pm_instances
credit_em[b] = novelty[b] / sum(novelties) * num_blocks
```

### 15.5 Controller Architecture

Shared MLPs (not per-instance -- too many parameters):

- **RLPMController:** 1 shared across B*L=32 instances. MLP(7->32->1->sigmoid).
  Features: elig_norm, pm_usage_ratio, surprise_mean, surprise_max, block_idx/B, layer_idx/L, bias=1
- **RLEMController:** 1 shared across B=4 instances. MLP(6->32->1->sigmoid).
  Features: novelty_mean, surprise_mean, em_usage_ratio, novelty_max, block_idx/B, bias=1

Block/layer indices as features let the shared network learn instance-specific behavior.

### 15.6 Separate RL Optimizer

Controller params (~2K total) get their own Adam optimizer (lr=1e-3). They are excluded from the main LM optimizer. This prevents:
- LM gradient noise from overwhelming the small RL signal
- The RL learning rate from destabilizing LM training

### 15.7 Computational Cost

Per chunk: 2 events * 2 rollouts * P=32 tokens = 128 extra forward passes (no backward). Main forward = T=256 tokens. Overhead ~50%, acceptable for the quality of the reward signal vs noisy REINFORCE.

### 15.8 Future: Learnable Continuous Parameters

Phase D learns binary gates only (commit yes/no, write yes/no). Several continuous parameters are currently heuristic constants behind `no_grad`:

| Parameter | Current | Future RL Output |
|-----------|---------|-----------------|
| g (PM write strength) | 0.5 fixed | sigmoid head on RLPMController |
| g_em (EM write strength) | 0.3 fixed | sigmoid head on RLEMController |
| lambda (PM commit decay) | config.decay_pm | sigmoid head on RLPMController |

Implementation: widen controller output heads. Only after binary gates are stable.

---

## 16. Phased Training Plan

| Phase | Components | Goal |
|-------|-----------|------|
| **A** | WM only (PM/EM disabled) | Stable streaming, perplexity decreases |
| **B** | WM + PM (heuristic commits) | Memory bench improvement, stable budgets |
| **C** | WM + PM + EM (heuristic writes) | Explicit recall at long delays |
| **D** | WM + PM + EM + RL controllers | Learned gating, better commit/write decisions |
| **E** | WM + PM + EM + lifelong | PM/EM persist across doc boundaries, soft reset |

Each phase inherits from the previous. `config.set_phase("X")` toggles the appropriate flags:

```python
"A": wm=True,  pm=False, em=False, lifelong=False
"B": wm=True,  pm=True,  em=False, lifelong=False
"C": wm=True,  pm=True,  em=True,  lifelong=False
"D": wm=True,  pm=True,  em=True,  lifelong=False  (+ RL controllers)
"E": wm=True,  pm=True,  em=True,  lifelong=True   (PM/EM persist across docs)
```

---

## 17. Scaling Tiers

| Tier | D | L | B | ~Params | 4090 BS | Use Case |
|------|---|---|---|---------|---------|----------|
| **A** (Debug) | 512 | 8 | 4 | ~50M | 32-64 | Rapid iteration |
| **B** (Competitive) | 768 | 12 | 6 | ~150M | 16-32 | Match GPT-2 Small |
| **C** (Strong) | 1024 | 24 | 8 | ~350M | 8-16 | Match GPT-2 Medium |

All tiers keep D_h = D/B = 128 constant. Scaling comes from wider model (D), more layers (L), and more parallel blocks (B).

---

## 18. Gradient Flow Map

Understanding what learns via backprop vs what needs RL:

```
                    ┌─────────────────────────────────────────┐
                    │          BACKPROP GRADIENT PATH          │
                    │                                         │
  LM Loss ◄── logits ◄── lm_head ◄── h_final ◄── Layer.step()
                                                      ▲
                                         ┌────────────┴────────────┐
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
                               (read only) (detached  (W_q_em, W_q_cross,
                                           KV cache)   W_o_cross params)
                                    ▲
                           ┌────────┘
                     PM eligibility:
                     W_k_pre, W_v_post      ◄── THESE learn what to propose
                     (nn.Linear params)
                           │
                    elig_K, elig_V          ◄── differentiable accumulation
                           │
                    ┌──────┴──────┐
                    │  no_grad    │
                    │  barrier    │         ◄── GRADIENT STOPS HERE
                    └──────┬──────┘
                           │
                    PM commit / EM write    ◄── needs RL to learn WHEN
                    (updates pm_K/V/a,
                     em_K/V/S)
                    │
                    └──► affects FUTURE reads (next span onward)
```

**Summary:**
- **Learns via backprop:** All `nn.Parameter` weights -- gate projections, WM projections, EM query/output projections, PM eligibility projections, embedding, lm_head
- **Evolves via explicit rules (no_grad):** pm_K, pm_V, pm_a, em_K, em_V, em_S -- updated at span boundaries by commit/write procedures
- **Needs RL:** Binary commit/write decisions (Phase D). Future: continuous parameters (g, lambda, g_em)

---

## File Index

| File | Primary Content |
|------|----------------|
| `src/model/config.py` | `ModelConfig` dataclass, tier presets, phase toggles |
| `src/model/model.py` | `NeuromorphicLM` -- top-level: embedding, WM, blocks, lm_head |
| `src/model/block.py` | `Block` -- parallel unit: L layers + 1 EM |
| `src/model/layer.py` | `Layer` -- affine recurrence + PM instance |
| `src/model/working_memory.py` | `WorkingMemory` -- sliding window attention |
| `src/model/procedural_memory.py` | `ProceduralMemory`, `PMController` |
| `src/model/episodic_memory.py` | `EpisodicMemory`, `EMController` |
| `src/model/utils.py` | `StateMixin`, `unit_normalize`, `soft_topk`, `budget_enforce` |
| `src/model/state.py` | `save_runtime_state`, `load_runtime_state`, `detach_all`, `reset_all` |
| `src/training/trainer.py` | `TBPTTTrainer` -- chunk processing, span boundaries |
| `src/training/loss.py` | `online_cross_entropy`, `compute_regularizers` |
| `src/training/eval_lifelong.py` | Phase E evaluation: domain adaptation, drift, cross-doc recall |
| `src/data/streaming.py` | `PersistentStreamDataset`, `StreamBatch`, `DocumentStream` |
| `src/train.py` | Entry point -- config, optimizer, scheduler, training loop (+ runtime state checkpointing) |
