# Neuromorphic LM -- Full Model Explainer (Sequential Path)

**Purpose:** Independent reference for verifying that all code changes remain aligned with the design intent. Covers every module, its intuition, its concrete implementation, and how modules compose during training. This document describes the **sequential** (`forward_one_token`) path. For the parallel scan-friendly training path, see `MODEL_EXPLAINER_PARALLEL.md`.

**Last verified against code:** 2026-02-10

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
11. [Neuromodulator Training (All Phases)](#11-neuromodulator-training-all-phases)
12. [Spatial Decoder](#12-spatial-decoder)
13. [State Management](#13-state-management)
14. [Training Loop (TBPTT + Spans)](#14-training-loop-tbptt--spans)
15. [Data Pipeline (Persistent Streams)](#15-data-pipeline-persistent-streams)
16. [Loss and Regularization](#16-loss-and-regularization)
17. [Phased Training Plan](#17-phased-training-plan)
18. [Scaling Tiers](#18-scaling-tiers)
19. [Gradient Flow Map](#19-gradient-flow-map)
20. [Design Integrity Notes](#20-design-integrity-notes)
21. [Training Cost and Performance Estimates](#21-training-cost-and-performance-estimates)

---

## 1. Design Philosophy

The neuromorphic LM decomposes language model memory into four biologically-inspired systems, each with different persistence and update mechanics:

| System | Biological Analogy | Persistence | Update Mechanism |
|--------|--------------------|-------------|------------------|
| **Genetic memory** (slow weights) | DNA / long-term evolutionary adaptation | Permanent after training | Standard backprop (frozen at deployment) |
| **Working memory** (WM) | Prefrontal cortex / scratchpad | Last W tokens | Sliding window attention (no plasticity) |
| **Procedural memory** (PM) | Cerebellum / skill memory | Across documents/lifelong | Eligibility traces + neuromodulated commits |
| **Episodic memory** (EM) | Hippocampus / event memory | Across documents/lifelong | Novelty-based writes to vector store + neuromodulation |

**Why not just use a transformer?** Transformers have a single memory mechanism (KV cache) that scales linearly with context. Our model separates *what* to remember from *how long* to remember it. WM handles precise short-range copying. PM learns reusable patterns (like skills). EM stores specific surprising events for later recall. Each system can be independently controlled, budgeted, and -- starting in Phase B -- learned via main-loss backprop through neuromodulators.

**Key constraint:** Single RTX 4090 (24GB). Everything must fit in VRAM with bf16 mixed precision. This rules out large KV caches or dense attention over long contexts, motivating the fixed-size memory banks.

---

## 2. Architecture at a Glance

The diagram below shows the **full architecture** (all subsystems active). In earlier phases, some components are absent: Phase A has WM + PM; Phase B adds EM; Phase C adds lifelong persistence. See [§17](#17-phased-training-plan) for the phase progression.

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
                     │  │ Gate compute  ││  a, b from concat(inputs + surprise)
                     │  │ Recurrence    ││  h = a * (carry * h_prev) + b
                     │  │ Proj+Res+Norm ││  output = norm(W_o(h) + x_block)
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
                                │
                           ┌────┘
                           ▼
                   Per-token surprise ──► span_surprise_mean
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
     PM Neuromodulators            EM Neuromodulators
     (at span boundaries)          (at span boundaries)
     ├─ inputs: elig_norm,         ├─ inputs: surprise,
     │  usage, surprise            │  usage, novelty
     ├─ outputs: commit_mask,      ├─ outputs: g_em,
     │  lambda, g, slots           │  tau, ww, decay
     ▼                             ▼
     PM commit: update             EM write: update
     pm_K, pm_V, pm_a             em_K, em_V, em_S
              │                          │
              └── All trained via ───────┘
                  main loss backprop
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

**What happens outside `forward_one_token`:** The model forward is read-only with respect to PM and EM — it reads from them but doesn't write. Memory updates happen at span boundaries in the trainer: PM eligibility traces are accumulated post-forward, then neuromodulators decide PM commits and EM writes (see [§10](#10-neuromodulators)). All neuromodulator heads are trained via main loss backprop (see [§11](#11-neuromodulator-training-all-phases)).

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

**Why "scan-friendly"?** The gates `a` and `b` are computed from input features only -- they do **not** depend on the previous hidden state `h_{t-1}`. This means that given a span of P tokens, the entire recurrence `h_t = a_t * h_{t-1} + b_t` can be computed via a parallel prefix scan (like Mamba, RWKV, or S4). The parallel span path (`forward_span`) exploits this; the sequential path described here runs token-by-token. See `MODEL_EXPLAINER_PARALLEL.md` for the parallel path.

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

**Post-recurrence FFN:** After the output projection + residual + LayerNorm, each layer applies a feed-forward network with pre-norm residual:
```python
output = output + ffn(ffn_norm(output))
# ffn: Linear(D_h, D_h*4) → GELU → Linear(D_h*4, D_h)
```
The recurrence mixes temporal information across time steps; the FFN adds per-position nonlinear depth, giving the model capacity to transform what it retrieves from memory before passing it to the next layer. Controlled by `ffn_expansion` (default 4; set to 0 to disable).

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

**Post-readout FFN:** After the linear lookup, the result passes through a pre-norm residual FFN:
```python
y_pm = y_raw + readout_ffn(readout_norm(y_raw))
# readout_ffn: Linear(D_h, D_h*4) → GELU → Linear(D_h*4, D_h)
```
This adds nonlinear processing to the linear key-value lookup, allowing the model to transform retrieved procedural knowledge before injecting it into the recurrence. Controlled by `pm_readout_ffn` (default `True`).

This is a read-only operation. PM keys and values are frozen within each plasticity span.

### 7.3 Eligibility Traces (every token, differentiable)

```python
k_cand = normalize(W_k_pre(x))    # [BS, D_h] -- from pre-synaptic (input)
v_cand = W_v_post(h)               # [BS, D_h] -- from post-synaptic (output)

# Gate by surprise: low surprise → near-zero accumulation
gate = (surprise / 5.0).clamp(0, 1)  # [BS, 1] → [BS, 1, 1]

elig_K = rho * elig_K + gate * k_cand.unsqueeze(1)  # accumulate into all r slots
elig_V = rho * elig_V + gate * v_cand.unsqueeze(1)
```

**Intuition:** Eligibility traces are a running average of "what the model wanted to write," gated by surprise. Only tokens the model predicted poorly contribute significantly to the trace -- this prevents the trace norm from saturating (which would make the commit gate always fire). The projections `W_k_pre` and `W_v_post` are learned parameters -- gradients from the LM loss flow back through them, teaching the model *what* constitutes a good key-value pair to store. But the *decision* of whether to actually commit these traces into PM is made by the neuromodulator at span boundaries.

`rho=0.95` is the eligibility decay -- recent tokens contribute more than older ones. This creates a recency-weighted, surprise-gated summary of the span's proposed writes.

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

   In learned mode (Phase B+ — `pm_enabled=True`):
   - `commit_mask: [BS]` -- heuristic gate: `elig_norm > 1.0`
   - `lambda_vals: [BS]` -- sigmoid, scaled to `[decay_pm, 1.0]` (differentiable, main optimizer)
   - `g: [BS]` -- sigmoid output (differentiable, main optimizer)
   - `slot_logits: [BS, r]` -- raw linear output (differentiable, main optimizer)
   - `p_commit: None` -- reserved (unused)

3. **Slot selection** (soft top-k):
   ```python
   scores = einsum(pm_K, elig_K_norm)       # [BS, r] -- similarity
   scores -= weakness_weight * pm_a          # prefer overwriting weak slots
   if slot_logits is not None:
       scores += slot_logits                 # learned bias (Phase B+)
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

**Differentiable continuous outputs:** Starting in Phase B (learned mode), `lambda_vals`, `g`, and `slot_logits` carry gradients through the commit operation. The `alpha` computation and the EMA update are differentiable tensor operations, so `total_loss.backward()` reaches the neuromodulator's continuous heads on the main optimizer. `commit_mask` is a heuristic binary gate (detached, no gradient).

### 7.5 PM Eligibility-Only Reset (Phase C)

In lifelong mode (Phase C), doc boundaries call `reset_eligibility(mask)` instead of the full `reset_states(mask)`:

```python
def reset_eligibility(self, mask):
    # Zero only elig_K, elig_V for masked streams
    # pm_K, pm_V, pm_a persist (committed knowledge carries forward)
    expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
    self.elig_K = self.elig_K * (~expanded)
    self.elig_V = self.elig_V * (~expanded)
```

**Intuition:** Eligibility traces represent partial, uncommitted learning from the current document. They are stale across doc boundaries and should reset. But committed PM state (`pm_K`, `pm_V`, `pm_a`) has passed through the neuromodulator's selection logic and represents consolidated knowledge -- it should persist.

### 7.6 PM: Complete Learnable Components Reference

This table covers **every learnable weight, runtime state, and control signal** that PM uses or produces. "Backprop" means gradients from `total_loss.backward()` reach the parameter. "Rules" means updated by explicit code (EMA, decay, budget enforcement) — no gradient.

#### Learned Parameters (nn.Parameter, saved in state_dict)

| Component | Location | Shape (Tier A) | What It Controls | Training | Phase | Parallelization | Lifelong Role |
|-----------|----------|----------------|------------------|----------|-------|-----------------|---------------|
| `W_k_pre` | `PM.W_k_pre` | `Linear(D_h, D_h)` per instance | Projects layer **input** into candidate keys — decides *what patterns to match* | Backprop (main opt) | B+ | Per-token in `update_eligibility_batch`; parallelized in `forward_span` | Learns general pattern-recognition; persists across docs |
| `W_v_post` | `PM.W_v_post` | `Linear(D_h, D_h)` per instance | Projects layer **output** into candidate values — decides *what to retrieve when matched* | Backprop (main opt) | B+ | Per-token in `update_eligibility_batch`; parallelized in `forward_span` | Learns general value-encoding; persists across docs |
| `readout_norm` | `PM.readout_norm` | `LayerNorm(D_h)` | Normalizes raw PM read output before FFN | Backprop (main opt) | B+ | Per-token in `apply`/`apply_batch` | Static after convergence |
| `readout_ffn` | `PM.readout_ffn` | `Linear(D_h, 4*D_h)` + `Linear(4*D_h, D_h)` | Nonlinear transform of retrieved PM content before injection into recurrence | Backprop (main opt) | B+ | Per-token in `apply`/`apply_batch` | Static after convergence |
| `gate_a, gate_b` | `Layer.gate_a/b` | `Linear(4*D_h+1, D_h)` | Retention/update gates consuming PM read output (among other signals) | Backprop (main opt) | A+ | Per-token, parallelized via scan | Static after convergence |
| `W_o` | `Layer.W_o` | `Linear(D_h, D_h)` | Output projection after recurrence | Backprop (main opt) | A+ | Per-token, parallelized via scan | Static after convergence |
| PMNeuromod `backbone` | `layer.pm_neuromodulator` | `Linear(3, H=32)` + ReLU | Shared representation for all PM neuromod heads | Backprop (main opt) | B+ | Boundary-only (once per P tokens) | Adapts to stream-specific surprise/usage patterns |
| PMNeuromod `lambda_head` | `layer.pm_neuromodulator` | `Linear(H, 1)` | **Decay rate** for pm_a after commit: controls how quickly slot strengths decay. Range `[decay_pm, 1.0]` via sigmoid | Backprop (main opt) | B+ | Boundary-only | Sets forgetting speed; learned per-stream |
| PMNeuromod `g_head` | `layer.pm_neuromodulator` | `Linear(H, 1)` | **Commit strength** (alpha blending factor): how strongly new eligibility overwrites existing slots | Backprop (main opt) | B+ | Boundary-only | Controls plasticity-stability tradeoff |
| PMNeuromod `slot_head` | `layer.pm_neuromodulator` | `Linear(H, r=8)` | **Slot selection bias**: additive logits for soft top-k slot choice — shifts which slots get overwritten | Backprop (main opt) | B+ | Boundary-only | Learns slot allocation strategy |

#### Runtime State (not nn.Parameter — updated by rules at boundaries)

| State | Shape | What It Stores | Update Rule | When Updated | Lifelong Behavior |
|-------|-------|----------------|-------------|--------------|-------------------|
| `pm_K` | `[BS, r, D_h]` | Key bank (unit-normalized) — *what patterns are stored* | EMA: `pm_K = normalize((1-α)*pm_K + α*elig_K)` where α = `soft_topk_weights * g * commit_mask` | Span boundary (commit) | Persists in Phase C; accumulates cross-doc knowledge |
| `pm_V` | `[BS, r, D_h]` | Value bank (unit-normalized) — *what to retrieve when matched* | Same EMA as pm_K but with elig_V | Span boundary (commit) | Persists in Phase C |
| `pm_a` | `[BS, r]` | Slot strengths [0, a_max] — *how confident each slot is* | `pm_a += α * score`, then base_decay (`*= decay_pm`), then budget_enforce | Span boundary | Persists in Phase C; base_decay prevents runaway |
| `elig_K` | `[BS, r, D_h]` | Key eligibility trace — *running average of what to write* | `elig_K = ρ * elig_K + gate * k_cand` where gate = `(surprise/5).clamp(0,1)` | Every token | Reset on doc boundary (Phase C: reset_eligibility) |
| `elig_V` | `[BS, r, D_h]` | Value eligibility trace | Same as elig_K with v_cand | Every token | Reset on doc boundary |
| `h` | `[BS, D_h]` | Layer hidden state | Affine recurrence: `h = a*h + b` from gates | Every token | Reset on doc boundary (even Phase C) |

#### Control Signals (computed, not stored — ephemeral per boundary)

| Signal | Source | What It Decides | Differentiable? |
|--------|--------|-----------------|-----------------|
| `commit_mask` | PMNeuromod | Whether to commit at all (heuristic `elig_norm > 1.0` in all phases) | No (detached bool) |
| `lambda_vals` | PMNeuromod | Per-stream decay rate for pm_a post-commit | Yes (Phase B+) |
| `g` | PMNeuromod | Per-stream commit strength (alpha multiplier) | Yes (Phase B+) |
| `slot_logits` | PMNeuromod | Additive bias on slot scores for soft top-k | Yes (Phase B+) |
| `soft_topk weights` | `soft_topk(scores, k=2)` | Which slots to overwrite (continuous [0,1] weights) | Yes |
| `alpha` | `weights * g * commit_mask` | Final per-slot blending factor | Partially (commit_mask is detached) |

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

# 5. Post-retrieval FFN (pre-norm residual)
out = out + readout_ffn(readout_norm(out))       # [BS, D_em]
# readout_ffn: Linear(D_em, D_em*4) → GELU → Linear(D_em*4, D_em)

# 6. Project to model dimension
y_em = W_o_cross(out)                           # [BS, D]
```

The readout FFN (controlled by `em_readout_ffn`, default `True`) adds nonlinear processing after cross-attention aggregation, allowing the model to reason about retrieved episodic content before injecting it into the recurrence.

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

**Learned novelty weighting (Phase B+):** When `em_enabled=True`, `EpisodicMemory` creates `W_nov: nn.Linear(D + D, 1)` -- a per-token projection that replaces the hardcoded 0.5/0.5 weighting:

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

1. **Neuromodulator decision:** `EMNeuromodulator.forward(span_surprise, em_usage, cand_novelty_mean)` returns a 4-tuple `(g_em, tau, ww, decay)`.

   In heuristic mode (Phase A — `em_enabled=False`):
   - `g_em: [BS]` -- fixed `config.g_em_default` (0.3)
   - `tau: [BS]` -- fixed `config.tau_em` (default 1.0)
   - `ww: [BS]` -- fixed `config.weakness_weight_em` (default 0.5)
   - `decay: [BS]` -- fixed `config.decay_em` (default 0.999)

   In learned mode (Phase B+ — `em_enabled=True`):
   - `g_em: [BS]` -- in `[g_em_floor, g_em_ceil]` (default [0.001, 0.95]) via `floor + (ceil - floor) * sigmoid(raw)` (differentiable, main optimizer; near-zero floor enables soft "don't write")
   - `tau: [BS]` -- in `[tau_em_floor, tau_em_ceil]` (default [0.05, 5.0]) via same sigmoid+clamp pattern. Controls soft top-k sharpness for slot selection.
   - `ww: [BS]` -- in `[ww_em_floor, ww_em_ceil]` (default [0.0, 2.0]) via same pattern. Controls preference for overwriting weak slots.
   - `decay: [BS]` -- in `[decay_em_floor, decay_em_ceil]` (default [0.99, 0.9999]) via same pattern. Controls per-stream memory retention timescale.

2. **For each candidate c** in the span (with `cand_valid` masking):
   - Score candidate key against all M existing slots (similarity + weakness bias)
   - Soft top-k selection of `k_write=4` slots to update
   - EMA blend: `em_K = normalize((1-alpha)*em_K + alpha*k_c)`, same for `em_V`
   - Strength update: `em_S += alpha * score`

3. **Decay + budget:** `em_S *= decay` (per-stream learned decay), then scale to enforce `sum(em_S) <= budget_em`.

**Candidate validity:** The trainer tracks `cand_valid: [BS, P]` which masks out candidates from before a mid-span doc-boundary reset and from EOT positions. This prevents writing stale cross-document content.

### 8.5 EM Doc-Boundary Reset

**Critical design decision:** On doc boundary reset (Phases A-C), EM only zeros `em_S` (strengths). It preserves `em_K` and `em_V`. This is implemented via an override of `StateMixin.reset_states()` in `EpisodicMemory`.

**Why?** Zeroing strengths makes all slots invisible to retrieval (the `em_S > 0` mask filters them out) without destroying the actual key-value content. When new writes occur, they can reuse these slots via the weakness-biased slot selection (slots with `em_S=0` are preferred targets). This is more graceful than re-randomizing keys, which would destroy any useful patterns in the key space.

In Phase C (lifelong mode), EM is not reset at all -- `em_K`, `em_V`, and `em_S` all persist across document boundaries.

### 8.6 EM: Complete Learnable Components Reference

This table covers **every learnable weight, runtime state, and control signal** that EM uses or produces. "Backprop" means gradients from `total_loss.backward()` reach the parameter. "Rules" means updated by explicit code (EMA, decay, budget enforcement).

#### Learned Parameters (nn.Parameter, saved in state_dict)

| Component | Location | Shape (Tier A) | What It Controls | Training | Phase | Parallelization | Lifelong Role |
|-----------|----------|----------------|------------------|----------|-------|-----------------|---------------|
| `W_q_em` | `EM.W_q_em` | `Linear(D+D, D_em)` | **Retrieval query** from `concat(x_emb, y_wm)` — decides *how to search* the memory bank | Backprop (main opt) | C+ | Per-token in `retrieve_batch`; parallelized in `forward_span` | Static after convergence |
| `W_q_cross` | `EM.W_q_cross` | `Linear(D_em, D_em)` | **Cross-attention query** over top-k retrieved values — decides *which retrieved memory to attend to* | Backprop (main opt) | C+ | Per-token in `retrieve_batch` | Static after convergence |
| `W_o_cross` | `EM.W_o_cross` | `Linear(D_em, D)` | **Output projection** from EM dim back to model dim — shapes *how retrieved content is injected* | Backprop (main opt) | C+ | Per-token in `retrieve_batch` | Static after convergence |
| `readout_norm` | `EM.readout_norm` | `LayerNorm(D_em)` | Normalizes cross-attention output before FFN | Backprop (main opt) | C+ | Per-token | Static after convergence |
| `readout_ffn` | `EM.readout_ffn` | `Linear(D_em, 4*D_em)` + `Linear(4*D_em, D_em)` | Nonlinear transform of retrieved EM content before projection to model dim | Backprop (main opt) | C+ | Per-token | Static after convergence |
| `W_k_cand` | `EM.W_k_cand` | `Linear(D+D, D_em)` | Projects `concat(x_emb, y_wm)` into **candidate keys** — decides *what patterns to store* | Backprop (main opt) | C+ | Per-token in `propose_candidate_batch` | Learns general pattern encoding |
| `W_v_cand` | `EM.W_v_cand` | `Linear(D_h, D_em)` | Projects final layer hidden state into **candidate values** — decides *what content to store* | Backprop (main opt) | C+ | Per-token in `propose_candidate_batch` | Learns general value encoding |
| `W_nov` | `EM.W_nov` | `Linear(D+D, 1)` | **Learned novelty weighting** — per-token blend of surprise vs dissimilarity for scoring candidate importance. Replaces hardcoded 0.5/0.5 | Backprop (main opt) | B+ | Per-token in `propose_candidate_batch` | Adapts novelty sensing to domain |
| EMNeuromod `backbone` | `block.em_neuromodulator` | `Linear(3+content_proj_dim, H=32)` + ReLU | Shared representation for all EM neuromod heads. Inputs: span_surprise, em_usage, cand_novelty_mean + content embedding | Backprop (main opt) | B+ | Boundary-only (once per P tokens) | Adapts write decisions to stream context |
| EMNeuromod `g_head` | `block.em_neuromodulator` | `Linear(H, 1)` | **Write strength** (g_em): how strongly new candidates overwrite existing slots. Range `[0.001, 0.95]` via `floor + range * sigmoid(raw)` | Backprop (main opt) | B+ | Boundary-only | Controls plasticity-stability tradeoff for episodic memory |
| EMNeuromod `tau_head` | `block.em_neuromodulator` | `Linear(H, 1)` | **Slot selection temperature** (tau): controls `soft_topk` sharpness. Low tau → sharper selection (few slots updated strongly). Range `[0.05, 5.0]` | Backprop (main opt) | B+ | Boundary-only | Learns whether to concentrate or spread writes |
| EMNeuromod `ww_head` | `block.em_neuromodulator` | `Linear(H, 1)` | **Weakness weight** (ww): how much to prefer overwriting weak slots vs similar slots. Range `[0.0, 2.0]` | Backprop (main opt) | B+ | Boundary-only | Learns slot replacement strategy |
| EMNeuromod `decay_head` | `block.em_neuromodulator` | `Linear(H, 1)` | **Decay rate** (decay): per-stream memory retention timescale. Range `[0.99, 0.9999]` | Backprop (main opt) | B+ | Boundary-only | Learns forgetting speed per stream |

#### Runtime State (not nn.Parameter — updated by rules at boundaries)

| State | Shape | What It Stores | Update Rule | When Updated | Lifelong Behavior |
|-------|-------|----------------|-------------|--------------|-------------------|
| `em_K` | `[BS, M, D_em]` | Key bank (unit-normalized) — *what patterns are stored* | EMA: `em_K = normalize((1-α)*em_K + α*k_cand)` where α = `soft_topk_weights * g_em` | Span boundary (per candidate in span) | Persists in Phase C; K/V content survives doc reset |
| `em_V` | `[BS, M, D_em]` | Value bank — *what to retrieve when matched* | Same EMA as em_K with v_cand | Span boundary | Persists in Phase C |
| `em_S` | `[BS, M]` | Slot strengths [0, S_max] — *how active each slot is* | `em_S += α * cand_score`, then `*= decay` (per-stream learned), then budget_enforce | Span boundary | Persists in Phase C (lifelong); learned decay controls retention. In Phases A–B: zeroed on doc boundary (makes slots invisible but preserves K/V) |

#### Control Signals (computed, not stored — ephemeral per boundary)

| Signal | Source | What It Decides | Differentiable? |
|--------|--------|-----------------|-----------------|
| `g_em` | EMNeuromod | Per-stream write strength (alpha multiplier). Near-zero floor enables soft "don't write" | Yes (Phase B+) |
| `tau` | EMNeuromod | Per-stream soft_topk temperature — sharpness of slot selection | Yes (Phase B+); batched `[BS]` tensor |
| `ww` (weakness_weight) | EMNeuromod | Per-stream bias toward overwriting weak slots (low em_S) | Yes (Phase B+); batched `[BS]` tensor |
| `decay` | EMNeuromod | Per-stream memory retention decay rate | Yes (Phase B+); batched `[BS]` tensor |
| `soft_topk weights` | `soft_topk(scores, k=k_write, tau)` | Which slots to overwrite (continuous [0,1] weights over top-k within top-C) | Yes |
| `alpha` | `weights * g_em` | Final per-slot blending factor (one per candidate per slot) | Yes |
| `cand_score` (novelty) | `propose_candidate_batch` | Per-token novelty score: blend of surprise + dissimilarity from existing keys | Yes (through W_nov, Phase B+) |
| `cand_valid` | Trainer | Mask for candidates within current doc and non-EOT | No (computed from token IDs) |

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
    layer.pm.update_eligibility(x, h, surprise)  # accumulate elig traces (surprise-gated)
    x = h                                        # next layer's input

return x   # [BS, D_h]
```

**Block.commit_pm(span_surprise) at span boundary:**
```python
def commit_pm(self, span_surprise=None):
    for layer in self.layers:
        # Neuromodulator decides
        surprise_input = span_surprise if span_surprise is not None else elig_norm
        mask, lam, g, slots, _p = layer.pm_neuromodulator(
            elig_norm, pm_usage / config.budget_pm, surprise_input
        )
        pm.commit(mask, lam, g, slots)
```

The `span_surprise` parameter receives the actual per-stream mean surprise over the span, computed in the trainer. If not available (e.g., standalone evaluation), the neuromodulator falls back to using `elig_norm` as a proxy.

### 9.1 Block State Reset (Lifelong Mode)

`Block.reset_states(mask)` branches on `config.lifelong_mode`:

**Phases A-B (lifelong_mode=False):**
```python
for layer in self.layers:
    layer.reset_states(mask)     # zeros h
    layer.pm.reset_states(mask)  # zeros all PM state
self.em.reset_states(mask)       # zeros em_S
```

**Phase C (lifelong_mode=True):**
```python
for layer in self.layers:
    layer.reset_states(mask)         # zeros h (always)
    layer.pm.reset_eligibility(mask) # zeros only elig_K, elig_V
# EM fully persists -- no reset call
```

This is the core of the soft reset: transient state (h, eligibility) resets at doc boundaries, but committed PM weights and EM content persist across documents. Natural memory turnover is handled by existing decay and budget enforcement.

---

## 10. Neuromodulators

Neuromodulators decide **when** and **how strongly** to write to memory. They operate at span boundaries only -- once every P=32 tokens. Each neuromodulator is an `nn.Module` that operates in one of two modes depending on whether its memory system is enabled: heuristic (zero params) or learned (backbone + heads, trained via main loss backprop).

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

| Output | Shape | Heuristic (Phase A) | Learned (Phase B+) | Training |
|--------|-------|--------------------|---------------------------------|----------|
| `commit_mask` | `[BS]` | `elig_norm > 1.0` | `elig_norm > 1.0` (same heuristic) | -- |
| `lambda_vals` | `[BS]` | `config.decay_pm` | sigmoid, scaled to `[decay_pm, 1.0]` | Main loss backprop |
| `g` | `[BS]` | `0.5` | sigmoid | Main loss backprop |
| `slot_logits` | `[BS, r]` | `None` | raw linear | Main loss backprop |
| `p_commit` | `[BS]` | `None` | `None` | -- (reserved) |

**Backbone + continuous heads** (created when `pm_enabled=True`, i.e. Phase B+):
```
3 inputs → Linear(3, H) → ReLU → {lambda_head, g_head, slot_head}
```

Hidden size H = `config.rl_controller_hidden` (default 32). ~458 params per instance, ~14,600 total across 32 instances.

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

**Outputs (4-tuple):**

| Output | Shape | Heuristic (Phase A) | Learned (Phase B+) | Training |
|--------|-------|----------------------|------------------------------|----------|
| `g_em` | `[BS]` | `config.g_em_default` (0.3) | `floor + (ceil - floor) * sigmoid(raw)`, in [g_em_floor, g_em_ceil] | Main loss backprop |
| `tau` | `[BS]` | `config.tau_em` (1.0) | `tau_floor + (tau_ceil - tau_floor) * sigmoid(raw)`, in [0.05, 5.0] | Main loss backprop |
| `ww` | `[BS]` | `config.weakness_weight_em` (0.5) | `ww_floor + (ww_ceil - ww_floor) * sigmoid(raw)`, in [0.0, 2.0] | Main loss backprop |
| `decay` | `[BS]` | `config.decay_em` (0.999) | `floor + (ceil - floor) * sigmoid(raw)`, in [0.99, 0.9999] | Main loss backprop |

**Backbone + heads** (created when `em_enabled=True`, i.e. Phase B+):
```
3 + content_proj_dim inputs → Linear(in, H) → ReLU → g_head
                                                     → tau_head
                                                     → ww_head
                                                     → decay_head
```
No `write_mask` or `gate_head` — writes always occur. The near-zero `g_em_floor` enables a soft "don't write" — the model can drive g_em close to zero for low-novelty spans, resulting in negligible alpha without requiring a binary gate.

**g_em safety rails:** `g_em = g_em_floor + (g_em_ceil - g_em_floor) * sigmoid(raw)` with defaults `g_em_floor=0.001`, `g_em_ceil=0.95`. Ceiling prevents sigmoid saturation (leaves gradient room).

**tau/ww/decay safety rails:** Same `floor + (ceil - floor) * sigmoid(raw)` pattern. `tau` in [0.05, 5.0] controls soft top-k temperature — low tau makes slot selection sharper, high tau spreads writes more evenly. `ww` (weakness weight) in [0.0, 2.0] controls how strongly the model prefers overwriting weak slots. `decay` in [0.99, 0.9999] controls per-stream memory retention timescale — low decay forgets quickly (~2.2K token half-life), high decay retains nearly permanently (~220K token half-life). All are per-stream `[BS]` tensors.

All heads use **calibrated bias init**: `bias = log(frac / (1 - frac))` where `frac = (default - floor) / (ceil - floor)`. This ensures zero input produces the config default, so the learned mode starts at the same operating point as the heuristic mode.

### 10.3 Neuromodulator Parameter Handling

Since neuromodulators are `nn.Module` submodules of `Layer` and `Block` (which are in the model's `ModuleList`), their parameters are automatically included in `model.state_dict()` and `model.named_parameters()`. This means they are saved/loaded in checkpoints automatically.

**All neuromodulator params are on the main optimizer in all phases.** There is no separate RL optimizer.

In all phases where neuromodulators exist (B+), the backbone and all heads are trained via main loss backprop alongside all other model params on the single main optimizer.

---

## 11. Neuromodulator Training (All Phases)

All neuromodulator parameters are trained via main loss backprop on the main optimizer. There is no separate RL optimizer or counterfactual rollout system.

- **PM continuous outputs** (`lambda`, `g`, `slot_logits`): Created in Phase B (when `pm_enabled=True`). Flow through PM commit operations into future predictions. Trained via **main loss backprop**.
- **PM commit gate** (`commit_mask`): Heuristic (`elig_norm > 1.0`) in all phases. The `p_commit` output is reserved but unused.
- **EM write parameters** (`g_em`, `tau`, `ww`, `decay`): Created in Phase B (when `em_enabled=True`). All four are trained via **main loss backprop** (through write → retrieve → loss). `g_em` controls write strength, `tau` controls soft top-k slot selection temperature, `ww` controls weakness-weight bias toward overwriting weak slots, `decay` controls per-stream memory retention timescale.
- **Learned novelty** (`W_nov`): Per-token projection on `EpisodicMemory`. Created in Phase B. Trained via **main loss backprop**.

### 11.1 Gradient Flow Through Memory Operations

PM commits and EM writes update plain tensors (`pm_K`, `pm_V`, `pm_a`, `em_K`, `em_V`, `em_S`). The continuous outputs (`g`, `lambda`, `slot_logits` for PM; `g_em`, `tau`, `ww`, `decay` for EM) flow through the EMA update math (which is differentiable), so backprop reaches them. PM's `commit_mask` is a hard boolean (`elig_norm > 1.0`) which breaks the gradient — but the continuous outputs that control *how* commits happen (strength, decay, slot selection) still learn via backprop.

**EM has no binary gate.** Instead of deciding "write or not," it decides "how strongly to write" via `g_em`, "how to write" via `tau` (temperature) and `ww` (weakness weight), and "how fast to forget" via `decay`. All four are continuous outputs clamped to their respective `[floor, ceil]` ranges, fully differentiable through the write operations.

### 11.2 Training Step

```
1. Forward all T=256 tokens (normal training)
   → neuromodulators produce continuous outputs (lambda, g, slot_logits, g_em, tau, ww, decay)
   → these flow through commit/write → affect PM/EM state → affect future predictions

2. total_loss.backward()
   → computes gradient on ALL params, including neuromodulator heads

3. clip_grad_norm_(all_params)

4. optimizer.step()
   → updates all model params including neuromodulators
```

---

## 12. Spatial Decoder

**File:** `src/model/decoder.py`

**Toggle:** `config.snapshot_enabled` (default `True`). When off, the model uses the original path: `concat(h_blocks) → lm_head`. When on, intermediate layer outputs from all blocks feed through a three-level hierarchical decoder before the LM head. Both the sequential (`forward_one_token`) and parallel (`forward_span`) paths support the decoder.

### 12.1 Why a hierarchical decoder?

Without the spatial decoder, only the **final layer** of each block reaches the LM head. All intermediate layer outputs — which represent different levels of abstraction and carry different PM contributions — are discarded. This is like reading only the last page of each chapter.

The spatial decoder treats NML blocks as distributed specialized memory (like cortical regions) and applies a biologically-inspired aggregation hierarchy before language decoding.

### 12.2 Level 1 — Columnar Attention

**Analogy:** A cortical column integrating across its laminar layers.

Each block has L=8 layers. A learned **summary query** iteratively cross-attends to all L layer outputs (tagged with learned layer-position embeddings) through `columnar_layers` (default 2) refinement layers. Each refinement layer applies pre-norm cross-attention followed by a pre-norm FFN (4× expansion, GELU). Produces one column summary `[BS, D_h]` per block.

```
Layer outputs: [h_0, h_1, ..., h_7]  each [BS, D_h]
       + layer_emb(0..7)
                ↓
       Learned query iteratively refines (columnar_layers rounds):
         cross-attn → residual → FFN → residual
                ↓
       Column summary: [BS, D_h]
```

One `ColumnarAttention` instance per block (B=4 total). Cross-attention over 8 tokens × `columnar_layers` rounds — negligible compute.

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

K learned output queries (default K=4) iteratively cross-attend to these 7 tokens through `thalamic_layers` (default 2) refinement layers. Each layer applies pre-norm cross-attention followed by a pre-norm FFN (4× expansion, GELU). Produces K **integrated memory tokens** `[BS, K, d_dec]`.

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
| `columnar_layers` | 2 | Depth of Level 1 columnar attention |
| `thalamic_layers` | 2 | Depth of Level 2 thalamic integrator |
| `thalamic_tokens` | 4 | Output tokens from Level 2 (K) |

Additional parameters: ~6M on Tier A (~12% overhead). Compute: cross-attention over sequences of length 8, 7, and 4 × their respective layer counts — effectively free compared to the O(D_h^2) recurrence per layer. Tier B scales `d_dec=384, n_heads_decoder=6`; Tier C scales `d_dec=512, n_heads_decoder=8, decoder_layers=3`.

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
- **`state_dict_runtime()` / `load_state_runtime()`**: Serialize/deserialize for checkpointing

**Convention:** Runtime state tensors are plain attributes, NOT `nn.Parameter` or `register_buffer`. They are updated explicitly -- either under `torch.no_grad()` (for PM/EM slot updates) or via differentiable operations (for eligibility, recurrent h).

### 13.2 State Reset Semantics

**Phases A-B (lifelong_mode=False):**

| Module | What resets on doc boundary | What is preserved |
|--------|---------------------------|-------------------|
| Layer | `h` (hidden state) | -- |
| PM | `pm_K, pm_V, pm_a, elig_K, elig_V` (all zeroed) | -- |
| EM | `em_S` only (strengths zeroed) | `em_K, em_V` preserved |
| WM | `wm_valid` cleared, `wm_ptr` reset | -- |
| Model | `surprise` zeroed for masked streams | -- |

**Phase C (lifelong_mode=True) -- Soft Reset:**

| Module | What resets on doc boundary | What persists |
|--------|---------------------------|---------------|
| Layer | `h` (hidden state) | -- |
| PM | `elig_K, elig_V` only | `pm_K, pm_V, pm_a` (committed knowledge) |
| EM | -- (nothing resets) | `em_K, em_V, em_S` (all persist) |
| WM | `wm_valid` cleared, `wm_ptr` reset | -- |
| Model | `surprise` zeroed for masked streams | -- |

The EM override in Phases A-B is deliberate: zeroing strengths makes old memories invisible without destroying key-value content, allowing graceful slot reuse. In Phase C, even strengths persist -- natural decay (per-stream learned) and budget enforcement provide forgetting pressure instead.

### 13.3 Bulk Operations (state.py)

```python
detach_all(model)          # Walk tree, detach all StateMixin states
reset_all(model, mask)     # Walk tree, reset masked streams
save_runtime_state(model)  # Serialize all state for checkpointing
load_runtime_state(model, state)  # Restore saved state
```

**Path-based checkpoint keys:** `save_runtime_state()` uses stable module paths as dictionary keys (e.g., `"blocks.0.layers.1.pm"`, `"blocks.2.em"`). These keys are derived from the module tree structure and remain stable across unrelated model changes. `load_runtime_state()` first tries path-based keys, then falls back to legacy index-based keys for backward compatibility.

---

## 14. Training Loop (TBPTT + Spans)

**Files:** `src/training/trainer.py` (orchestration), `src/training/span_ops.py` (shared span-boundary ops)

### 14.1 Two Time Scales

| Scale | Length | What happens |
|-------|--------|--------------|
| **TBPTT chunk** | T=256 tokens | Autograd truncation. One backward pass per chunk. States detached after. |
| **Plasticity span** | P=32 tokens | PM/EM are read-only within. Commits and writes happen at span boundaries. |

A chunk contains T/P = 8 spans. Each span boundary triggers PM base decay + commits + EM writes.

### 14.2 Chunk Processing (train_chunk) — Sequential Path

**Note:** This describes the original sequential training loop using `forward_one_token()`. The primary training path now uses the parallel span forward pass — see `MODEL_EXPLAINER_PARALLEL.md` §14.2.

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
        9.  Pre-compute EM candidate stacks + novelty
       10.  PM base decay (all streams): pm_a *= decay_pm
       11.  PM commits: neuromodulator decides, pm.commit()
       12.  EM writes: neuromodulator decides, write_at_boundary()
            filters candidates by validity (post-reset only)

After all spans:
    13. Finalize loss: avg_loss = chunk_loss / valid_count
    14. Add regularizers (PM/EM budget penalties)
    15. optimizer.zero_grad()
    16. total_loss.backward()
    17. Clip gradients
    18. optimizer.step() + scheduler.step()
    19. Detach all states (TBPTT boundary)
    20. Save last token per stream (for checkpoint resume)
```

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
- Note: `reset_on_doc_boundary` remains True even in Phase C (lifelong mode). The `lifelong_mode` flag controls state persistence, not loss masking.

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

| Phase | Components | lifelong_mode | Neuromod State | Goal |
|-------|-----------|---------------|----------------|------|
| **A** | WM + PM | False | PM: heuristic (0 params) | Stable streaming, perplexity decreases |
| **B** | WM + PM + EM | False | PM + EM: backbone + heads + W_nov (main optimizer) | Memory bench improvement, learned commit/write params |
| **D** | WM + PM + EM + lifelong | True | Same as B (inherited) | PM/EM persist across doc boundaries |

`config.set_phase("X")` toggles the appropriate flags:

```python
"A": wm=True,  pm=True,  em=False, lifelong=False
"B": wm=True,  pm=True,  em=True,  lifelong=False
"C": wm=True,  pm=True,  em=True,  lifelong=True
```

All neuromodulator params are on the main optimizer in all phases. There is no separate RL optimizer.

**Phase transition via checkpoint resume:** `src/train.py` loads model checkpoints with `strict=False`, so new parameters are initialized fresh when they first appear. When transitioning A→B, PM and EM neuromodulator backbones + heads + W_nov init fresh. B→C preserves everything (only `lifelong_mode` flag changes).

**Optimizer state across phase transitions:** When a phase transition is detected (checkpoint's `pm_enabled`/`em_enabled` differ from current config), optimizer state loading is skipped because parameter group sizes change across phases. The optimizer reinitializes with fresh Adam state.

---

## 18. Scaling Tiers

| Tier | D | L | B | ~Params | 4090 BS | Use Case |
|------|---|---|---|---------|---------|----------|
| **A** (Debug) | 512 | 8 | 4 | ~50M | 32-64 | Rapid iteration |
| **B** (Competitive) | 768 | 12 | 6 | ~103M | 16-32 | Match GPT-2 Small |
| **C** (Strong) | 1024 | 24 | 8 | ~197M | 8-16 | Match GPT-2 Medium |

All tiers keep D_h = D/B = 128 constant. Scaling comes from wider model (D), more layers (L), more parallel blocks (B), **and scaled memory capacities**:

| Component | Tier A | Tier B | Tier C |
|-----------|--------|--------|--------|
| PM slots (r) | 8 | 16 | 32 |
| WM window (W) | 256 | 512 | 1024 |
| WM dim (D_wm) | 128 | 192 | 256 |
| EM capacity (M) | 256 | 512 | 1024 |
| EM dim (D_em) | 128 | 192 | 256 |
| EM retrieval (k_ret) | 4 | 8 | 16 |
| Decoder dim (d_dec) | 256 | 384 | 512 |
| Decoder depth | 2 | 2 | 3 |

Memory capacities scale alongside core dimensions, preventing the memory systems from becoming a bottleneck at higher tiers.

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
    │  │  backbone → 4 heads:   │  │   │  │  backbone → 3 heads:       │  │
    │  │  gate  g  lambda slot  │  │   │  │  g_em  tau  ww  (no gate)  │  │
    │  │   │    │    │     │    │  │   │  │         │                   │  │
    │  │   │    ╎    ╎     ╎    │  │   │  │  all: floor + range *       │  │
    │  │ .detach ╎   ╎     ╎    │  │   │  │    sigmoid(raw) [BS]       │  │
    │  │   │    ╎    ╎     ╎    │  │   │  │         │                   │  │
    │  │ commit ╎ alpha = g *   │  │   │  │  alpha = g_em * weights    │  │
    │  │ mask   ╎  weights      │  │   │  │  (DIFFERENTIABLE ──────────┤  │
    │  │ (bool) ╎    │          │  │   │  │  through write_at_boundary │  │
    │  │   │    ╎    │          │  │   │  │  into em_K / em_V / em_S)  │  │
    │  │   │    ╎    ▼          │  │   │  │         │                   │  │
    │  │   │    ╎ DIFFERENTIABLE│  │   │  │  RL (weighted MSE on       │  │
    │  │   │    ╎ through commit│  │   │  │  g_em, tau, ww): baseline  │  │
    │  │   │    ╎ into pm_K/V/a │  │   │  │  vs chosen counterfactual │  │
    │  │   │    ╎    │          │  │   │  └─────────────────────────────┘  │
    │  │ RL(BCE ╎    │          │  │   │                                    │
    │  │ on     ╎    │          │  │   │   affects FUTURE EM.retrieve ──────┘
    │  │ p_commit)   │          │  │   │         │
    │  └────────╎────┼──────────┘  │   │   Layer.step() → logits → loss
    │           ╎    │              │   │         │
    │   affects FUTURE PM.apply ───┘   │   loss.backward() reaches
    │           ╎    │                  │   EM g/tau/ww_head via write→ret
    │   Layer.step() → logits → loss   │
    │           ╎    │                  │
    │   loss.backward() reaches        │
    │   g_head, lambda_head,           │
    │   slot_head via alpha            │
    └──────────────────────────────────┘

    All parameters on single main optimizer.
    W_nov, neuromod backbone + heads all trained via main loss backprop.
```

**Summary:**
- **Learns via backprop (main optimizer):** All `nn.Parameter` weights -- gate projections, WM projections, EM query/output projections, PM eligibility projections, embedding, lm_head, `EpisodicMemory.W_nov` (learned novelty adjuster, Phase B+), neuromodulator backbone + all heads (Phase B+). Single optimizer for everything.
- **Evolves via explicit rules (no parameter grad):** pm_K, pm_V, pm_a, em_K, em_V, em_S -- updated at span boundaries by commit/write procedures
- **Not plastic (no memory update mechanism):** WM KV cache (ring buffer of recent token projections; gradients flow through within TBPTT chunks, detached at chunk boundaries)

---

## 20. Design Integrity Notes

This section documents verified design decisions and confirms that the core architecture remains intact.

### 20.1 Verified Invariants

1. **Continuous-head gradient path:** PM: `lambda_vals`, `g`, and `slot_logits` carry gradients through `pm.commit()` because the EMA update (`alpha = weights * g * mask`, `pm_K = (1-alpha)*pm_K + alpha*elig_K`) is composed of differentiable tensor operations. `loss.backward()` reaches these heads through: neuromodulator output -> alpha -> pm_K/pm_V update -> future `pm.apply(x)` -> layer output -> logits -> loss. EM: `g_em`, `tau`, `ww`, and `decay` carry gradients through `write_at_boundary()` via the same mechanism: g_em -> alpha -> em_K/em_V update -> future `EM.retrieve()` -> layer output -> logits -> loss. `tau` and `ww` affect gradient flow through `soft_topk` slot selection weights. `decay` affects gradient flow through em_S strength decay. This gradient path is active starting in Phase B (when learned heads are created).

2. **Single optimizer:** All parameters (model weights + neuromodulator backbone + heads) are on a single main optimizer with unified gradient clipping. No separate RL optimizer.

3. **Calibrated bias init:** All neuromod heads use `bias = log(frac / (1 - frac))` where `frac = (default - floor) / (ceil - floor)`. Zero input produces the config default, so the learned mode starts at the same operating point as the heuristic mode.

4. **Surprise as neuromodulator input** (not just novelty): Surprise reflects the model's overall prediction quality, which is a useful signal for deciding whether to consolidate learning. Novelty is specific to EM candidates.

### 20.2 Intentional Design Decisions (Not Bugs)

- **PM commit_mask is heuristic (`elig_norm > 1.0`) in all phases:** The binary gate is not learned; only the continuous outputs (lambda, g, slot_logits) are learned via backprop.
- **EM decay is per-stream learned (Phase B+):** Each stream learns its own memory retention timescale in `[0.99, 0.9999]`, replacing the previous fixed scalar `config.decay_em`.
- **Surprise as neuromodulator input** (not just novelty): Surprise reflects the model's overall prediction quality, which is a useful signal for deciding whether to consolidate learning. Novelty is specific to EM candidates.

---

## 21. Training Cost and Performance Estimates

This section provides throughput estimates, training budgets, and hardware recommendations for different scenarios. All numbers are **estimates** based on standard GPU throughput formulas, cloud pricing as of early 2025, and the model's non-transformer architecture (which has different compute characteristics than standard attention-based models).

### 21.1 Key Architectural Differences from Transformers

Our model uses **affine recurrence** (not attention) as the core sequence mechanism, plus memory reads/writes at span boundaries. This means:

- **No quadratic attention cost:** Compute scales linearly with sequence length (within spans)
- **Parallelizable within spans:** `forward_span` uses a parallel scan over P=32 tokens
- **Boundary overhead:** PM commit + EM write + neuromodulator forward at every span boundary

The recurrence + memory architecture means throughput is **lower than a pure transformer of equal parameter count** but has the advantage of truly O(1) per-token memory and O(n) compute.

### 21.2 Parameter Counts by Tier and Phase

| Tier | D | L | B | Phase A (WM) | Phase C (all) | Neuromod overhead |
|------|---|---|---|-------------|---------------|-------------------|
| **A** (Debug) | 512 | 8 | 4 | 56,012,416 | 56,033,136 | 20,720 (0.04%) |
| **B** (Competitive) | 768 | 12 | 6 | 102,555,456 | 102,620,400 | 64,944 (0.06%) |
| **C** (Strong) | 1024 | 24 | 8 | 196,265,728 | 196,530,272 | 264,544 (0.13%) |

Neuromodulators add negligible parameter overhead. The cost difference across phases comes from compute, not parameters.

### 21.3 Estimated Throughput by Phase (Single GPU)

Throughput depends heavily on batch size, span length (P=32), and which phases are active. These are rough estimates for typical configurations:

**Tier A (~56M) on RTX 4090 (24GB), BS=32:**

| Phase | Components | Est. tok/s | Bottleneck |
|-------|-----------|-----------|------------|
| **A** | WM only | ~500 | Recurrence + WM attention |
| **B** | WM + PM | ~400 | + PM read/elig every token, commit at boundary |
| **C** | WM + PM + EM | ~300 | + EM retrieval every token, write at boundary |
| **D** | WM + PM + EM + RL | ~180 | + 4 rollout forward_span per RL event (~30% of boundaries) |
| **E** | Lifelong (inherited) | ~250 | Similar to C/D but amortized RL |

**Tier B (~103M) on A100 80GB, BS=32:**

| Phase | Est. tok/s |
|-------|-----------|
| **A** | ~800 |
| **B** | ~650 |
| **C** | ~500 |
| **D** | ~300 |

**Tier C (~197M) on A100 80GB, BS=16:**

| Phase | Est. tok/s |
|-------|-----------|
| **A** | ~500 |
| **B** | ~400 |
| **C** | ~300 |
| **D** | ~180 |

### 21.4 Training Budget: Tokens vs Time

How long to train on N tokens, given estimated throughput:

**Tier A on 1x RTX 4090 (weighted average ~300 tok/s across phases):**

| Tokens | Hours | Days | Cost @ $0.34/hr |
|--------|-------|------|-----------------|
| 1B | 926 | 39 | $315 |
| 5B | 4,630 | 193 | $1,574 |
| 10B | 9,259 | 386 | $3,148 |
| 20B | 18,519 | 772 | $6,296 |

**Tier B on 1x A100 80GB (weighted average ~500 tok/s):**

| Tokens | Hours | Days | Cost @ $1.40/hr |
|--------|-------|------|-----------------|
| 5B | 2,778 | 116 | $3,889 |
| 10B | 5,556 | 231 | $7,778 |
| 20B | 11,111 | 463 | $15,556 |

**Tier C on 1x A100 80GB (weighted average ~300 tok/s):**

| Tokens | Hours | Days | Cost @ $1.40/hr |
|--------|-------|------|-----------------|
| 10B | 9,259 | 386 | $12,963 |
| 20B | 18,519 | 772 | $25,926 |

### 21.5 Multi-GPU Scaling

For Tier B and C, multi-GPU training significantly reduces wall-clock time. With data-parallel training (each GPU processes different streams):

| Config | Effective tok/s | 10B tokens wall-clock |
|--------|----------------|----------------------|
| Tier B, 1x A100 | ~500 | ~231 days |
| Tier B, 4x A100 | ~1,800 | ~64 days |
| Tier B, 8x A100 | ~3,200 | ~36 days |
| Tier C, 4x A100 | ~1,000 | ~116 days |
| Tier C, 8x A100 | ~1,800 | ~64 days |
| Tier C, 8x H100 | ~4,000 | ~29 days |

Scaling efficiency is estimated at ~85-90% for data parallelism (independent streams, minimal communication overhead since each stream's memory state is local).

### 21.6 Recommended Training Configurations

#### Overnight Run (Proof-of-concept, single RTX 4090)

- **Tier A**, Phases A→B→C→D, ~12 hours total
- **Tokens:** ~10M (A: 5M, B: 2M, C: 2M, D: 1M) — enough for loss curves, not convergence
- **BS:** 32, **T:** 256
- **Cost:** ~$4

#### Weekend Run (Meaningful training, single RTX 4090)

- **Tier A**, all phases, ~48 hours
- **Tokens:** ~50M total (~15M per phase, D gets less due to slower throughput)
- **Cost:** ~$16
- Should show clear perplexity improvement and memory utilization

#### Serious Experiment (Publishable ablation, single 4090)

- **Tier A**, all phases, ~2 weeks
- **Tokens:** ~500M (A: 200M, B: 150M, C: 100M, D: 50M)
- **Cost:** ~$115
- Sufficient to demonstrate architecture viability and ablate components

#### Publishable Experimental Model (Tier A, full training)

- **Tier A** on 1x RTX 4090
- **Tokens:** 5B (comparable to Chinchilla-optimal for 56M params)
- **Duration:** ~193 days (~6.5 months)
- **Cost:** ~$1,574
- Alternative: 4x RTX 4090 → ~50 days, ~$1,600 (cluster pricing)

#### Competitive Model (Match GPT-2 Small 124M)

- **Tier B** (~103M params) on 4x A100 80GB
- **Tokens:** 10-20B (GPT-2 used ~40B tokens for 124M, but Chinchilla scaling suggests ~2B is optimal for 103M; more tokens help with memory systems)
- **Duration:** 36-64 days
- **Cost:** $7,800-$15,600 (at $1.40/hr/GPU)

#### Strong Model (Match GPT-2 Medium 345M)

- **Tier C** (~197M params) on 8x A100 80GB
- **Tokens:** 20B
- **Duration:** ~64 days
- **Cost:** ~$15,500 (at $1.40/hr/GPU)
- Alternative: 8x H100 → ~29 days, ~$11,500 (at $2.50/hr/GPU)

### 21.7 Comparison to Reference Models

| Model | Params | Tokens | Hardware | GPU-hours | Est. Cloud Cost |
|-------|--------|--------|----------|-----------|-----------------|
| **GPT-2 Small** | 124M | 40B | 256x V100 | ~43,000 | ~$43K (at V100 rates) |
| **GPT-2 Medium** | 345M | 40B | 256x V100 | ~120,000 | ~$120K |
| **LLaMA 1 7B** | 7B | 1.4T | 2048x A100 | ~82,000 | ~$115K (at A100 rates) |
| **LLaMA 2 7B** | 7B | 2T | A100-80GB cluster | 184,320 | ~$258K |
| **LLaMA 2 13B** | 13B | 2T | A100-80GB cluster | 368,640 | ~$516K |
| **LLaMA 2 70B** | 70B | 2T | A100-80GB cluster | 1,720,320 | ~$2.4M |

Our model's advantage: memory systems (PM, EM) and neuroplasticity mechanisms could achieve better performance-per-parameter than standard transformers, especially on tasks requiring long-context recall, domain adaptation, and continual learning. The publishable hypothesis is that a 56M neuromorphic model with trained memory can match or outperform a 124M standard transformer on relevant benchmarks.

### 21.8 Cloud Pricing Reference (Early 2025)

| GPU | VRAM | RunPod Community | RunPod Secure | Lambda Labs | Northflank |
|-----|------|-----------------|---------------|-------------|------------|
| RTX 4090 | 24GB | $0.34/hr | $0.69/hr | — | — |
| A100 40GB | 40GB | $1.19/hr | $1.39/hr | $1.29/hr | $1.42/hr |
| A100 80GB | 80GB | $1.39/hr | $1.49/hr | $1.79/hr | $1.76/hr |
| H100 SXM | 80GB | $2.69/hr | $2.69/hr | $2.99/hr | $2.74/hr |
| H200 | 141GB | $3.59/hr | $3.59/hr | — | $3.14/hr |

---

## File Index

| File | Primary Content |
|------|----------------|
| `src/model/config.py` | `ModelConfig` dataclass, tier presets, phase toggles |
| `src/model/model.py` | `NeuromorphicLM` -- top-level: embedding, WM, blocks, lm_head, `rl_parameters()` |
| `src/model/block.py` | `Block` -- parallel unit: L layers + 1 EM + 1 `EMNeuromodulator` |
| `src/model/layer.py` | `Layer` -- affine recurrence + PM instance + `PMNeuromodulator` |
| `src/model/working_memory.py` | `WorkingMemory` -- sliding window attention |
| `src/model/procedural_memory.py` | `ProceduralMemory`, `PMNeuromodulator` (two-mode: heuristic / learned) |
| `src/model/episodic_memory.py` | `EpisodicMemory`, `EMNeuromodulator` (two-mode: heuristic / learned) |
| `src/model/utils.py` | `StateMixin`, `unit_normalize`, `soft_topk`, `budget_enforce` |
| `src/model/state.py` | `save_runtime_state`, `load_runtime_state`, `detach_all`, `reset_all` |
| `src/training/trainer.py` | `TBPTTTrainer` -- chunk processing, orchestrates span loop |
| `src/training/span_ops.py` | Shared span-boundary ops: loss masking, surprise, PM eligibility/commit, EM candidates/write |
| `src/training/loss.py` | `online_cross_entropy`, `compute_regularizers` |
| `src/training/eval_lifelong.py` | Phase C evaluation: domain adaptation, drift, cross-doc recall |
| `src/data/streaming.py` | `PersistentStreamDataset`, `StreamBatch`, `DocumentStream` |
| `src/train.py` | Entry point -- config, optimizer, scheduler, training loop, checkpoint save/load |
| `src/model/scan.py` | `parallel_affine_scan` -- affine recurrence scan for `forward_span()` |
| `src/debug/collector.py` | `MetricsCollector` -- two-tier JSONL logging, gate stats, memory stats, grad norms |
