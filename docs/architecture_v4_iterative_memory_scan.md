# Architecture v7: Single Scan Stack — Dense Scan + Hebbian PM + Grouped PCM

> **Previous versions**: v6 (3-stage scan-memory-scan), v5 (grouped scan), v4 (iterative R-loop) on `v4-iterative-backup` branch, v1 on `v1-legacy` branch.

## The Core Insight

The brain processes sensory input through two interleaved systems:

1. **Fast neocortical stream** — feedforward, massively parallel, processes each
   input near-instantly. Cortical columns extract features, predict upcoming input,
   and prepare eligibility signals for memory.

2. **Slow memory stream** — hippocampal/thalamic pattern completion, slower,
   asynchronous. Reads from and writes to structured memory stores based on what
   the fast stream found surprising or novel.

We implement this as a **single scan stack** with memory injection at a configurable layer:

```
layers[0..L_mem-1]:  PRE-MEMORY SCAN (dense nn.Linear, parallel via parallel-scan)
  → Process N tokens causally through dense [BS,N,D] scan layers
  → PCM operates on grouped [BS,N,C,D_col] view for per-feature-group surprise
  → Produces: EM trail seeds, write candidates, surprise vectors

MEMORY INJECTION at layer L_mem:
  → PM: Hebbian fast-weight read (bank-summed)              — [BS,N,D]
  → EM trail read from frozen primitives, summed over B      — [BS,N,D]
  → Additive injection: H = H + pm_read + em_read

layers[L_mem..L_total-1]:  POST-MEMORY SCAN (dense nn.Linear, parallel via parallel-scan)
  → Integrates PM read + EM trail read with encoded representation
  → Produces final predictions (NTP logits)

SEGMENT-END COMMITS (once per segment):
  → PM: Hebbian update — surprise-gated autocorrelation
  → EM: Neuromodulated decomposition write across primitives
```

**Key properties:**
- **Causal within segments** (scans are causal recurrences)
- **Causal across segments** (memory state flows forward)
- **Parallel within segments** (parallel-scan algorithm, memory ops batched)
- **Single scan stack** — no stage separation needed; memory reads are just additive residuals
- **No within-segment writes** — memory commits once at segment end (N=128 makes this frequent enough)
- **NTP training** — causal scans naturally support next-token prediction

## Why This Architecture Exists

We face a fundamental tension:

1. **Softmax attention**: KV cache grows linearly with context length, and
   O(N²) compute per layer during prefill — too expensive for lifelong learning
2. **Linear recurrence / affine scans**: our bio-inspired memory updates (Hebbian PM,
   episodic EM) involve non-affine decisions (thresholds, similarity matching,
   slot selection) — can't naively parallelize within a scan
3. **Pure sequential processing**: O(N) serial depth — underutilizes GPU

Our resolution: **memory reads are just functions that produce additive residuals** —
they don't need their own stage. PM matmul and EM trail attention are injected at
a configurable layer boundary, with the post-memory layers integrating the signal.

Shorter segments (N=128) mean memory updates every 128 tokens, making within-segment
write buffers unnecessary. This dramatically simplifies the architecture.

## Architecture

### Components

**Scan layers** (dense, full D mixing):
- Dense nn.Linear projections for GPU matmul efficiency
- Element-wise linear recurrence: `h_t = a_t ⊙ h_{t-1} + b_t` on d_inner dims
- Optional SwiGLU output projection for nonlinear feature mixing (glu_output=True by default in Tier A): proj_out produces [d_inner → 2*D], split into gate+up, output = silu(gate)*up

**Feature groups** (C groups of D_col, for PCM and memory addressing):
- C = D / D_col feature groups, used by PCM and W_seed_w (GroupedLinear)
- Free .view() between [BS,N,D] and [BS,N,C,D_col] (contiguous memory)
- **PCM** — per-group predictive coding (predict next token's encoding per group)
- Trail seeds and write candidates produced per-group via GroupedLinear

**Shared memory systems** (B independent banks):
- **PM** — procedural memory: Hebbian fast-weight network (basal ganglia / habit system).
  State = W_pm [BS, B, D_pm, D_pm] per bank. Read: proj_in(H) → (Σ_b W_b) @ pre → proj_out.
  Write: surprise-gated Hebbian autocorrelation. Per-bank β gives multiple timescales.
- **EM** — episodic memory: dictionary of M primitive patterns (hippocampus).
  Trail-based compositional read — seed navigates primitive space.
- B memory banks. Banks are independent; PM banks differ by plasticity rate β_b.
- PM state shape: `[BS, B, D_pm, D_pm]` (fast-weight matrix per bank)
- EM state shape: `[BS, B, M, D_mem]` where D_mem is a compressed latent dimension (512 for Tier A). Learned projections `mem_proj_in` (D→D_mem) and `mem_proj_out` (D_mem→D) inside EpisodicMemory.

**Dense scan, grouped PCM/projections:**
- Scan layers use dense nn.Linear — all D features mix freely for GPU efficiency.
- PCM and W_seed_w operate on grouped [BS,N,C,D_col] views (free reshape) so
  surprise and memory-facing signals remain per-feature-group.
- Cross-feature-group integration in scans, per-feature-group structure in
  prediction and memory addressing.

### How It Works (High Level)

```
For each segment of N tokens:

  PRE-MEMORY LAYERS [0..L_mem-1] — (causal, parallel via parallel-scan):
    For t = 1..N (parallelized):
      h_t = a_t ⊙ h_{t-1} + b_t               # element-wise recurrence
    PCM: z_hat_t = W_pcm · H_t                   # prediction
          surprise_t = z_hat_{t-1} - encode(X_t)  # vector surprise (D_col)
    [seed_t, w_cand_t] = W_seed_w · H_t           # fused seed + write candidate

  MEMORY INJECTION:
    # PM: Hebbian fast-weight read (bank-summed, no [BS,N,B,D])
      pre = proj_in(H)                           # [BS, N, D_pm]
      pm_read = proj_out((Σ_b W_b) @ pre)        # [BS, N, D]
    # EM trail read (from frozen primitives, summed over B)
      em_read = trail_read_all(seed)             # [BS, N, D]
    # Additive injection
      H = H + pm_read + em_read

  POST-MEMORY LAYERS [L_mem..L_total-1] — (causal, parallel via parallel-scan):
    For t = 1..N (parallelized):
      h'_t = a'_t ⊙ h'_{t-1} + b'_t
    logits_t = LM_head(H'_t)

  SEGMENT-END COMMITS:
    # PM: Hebbian — G = (1/N) Σ_t σ(‖surp_t‖) · pre_t⊗pre_tᵀ
    W_b = W_b @ (decay·I + β_b·G)               # PM: Hebbian update per bank
    # EM: novelty = w_cand.norm() + surprise.norm()
    g_em = neuromod(mean_novelty, usage)          # segment-level gate
    em.commit_all(w_cand, novelty, g_em)          # decompose across primitives

  Carry memory state to next segment.
```

Pre-memory and post-memory layers are all parallelizable via parallel-scan.
Memory reads are a fixed-cost batch operation (all N positions independent).

### Pre-Memory Layers: Scan + PCM

Scan layers are fully dense (nn.Linear) for GPU efficiency — all D features
mix freely in each layer. PCM operates on a grouped [BS,N,C,D_col] view
(free reshape) so surprise/prediction remains per-feature-group.

```python
def pre_memory(self, input_ids):
    """Pre-memory layers: Dense causal scan + PCM.

    Scan is dense [BS,N,D]. PCM and W_seed_w use grouped [BS,N,C,D_col] views.

    Returns:
        H:        [BS, N, D]           — scan hidden states (gain-modulated)
        seed:     [BS, N, D]           — EM trail seeds (shared across banks)
        w_cand:   [BS, N, D]           — write candidates (shared across banks)
        surprise: [BS, N, D]           — vector surprise signals
    """
    x = embed(input_ids)             # [BS, N, D_embed]
    x = proj_up(x)                   # [BS, N, D]
    x = x + pos_embed                # [BS, N, D]

    # Pre-memory scan layers: nn.Linear projections, full D mixing
    H = x
    for layer in layers[0:L_mem]:
        H = layer(H)                 # [BS, N, D]

    # PCM: view as grouped for per-feature-group surprise
    H_col = H.view(BS, N, C, D_col)             # free view
    x_col = x.view(BS, N, C, D_col)             # free view
    surprise_col = pcm.compute_surprise(H_col, x_col)
    H_col = pcm.apply_gain(H_col, surprise_col)
    H = H_col.view(BS, N, D)                    # back to dense (free)
    surprise = surprise_col.reshape(BS, N, D)

    # Grouped projections (per-feature-group seeds + write candidates)
    H_col = H.view(BS, N, C, D_col)             # free view
    sw = W_seed_w(H_col)            # [BS, N, C, 2*D_col] — GroupedLinear
    seed, w_cand = split(sw)        # each [BS, N, D] — shared across banks

    return H, seed, w_cand, surprise
```

**Scan carry across segments:** Scan hidden states are carried across segment boundaries
via `_carries` list in model.py. Each stores per-layer h_last as runtime state and passes
it as h_prev to the next segment. Cross-segment context flows through both memory state
(PM, EM) and scan recurrence.

**GLU output projection:** When glu_output=True (default in Tier A), each scan layer's
output projection produces 2*D features, split into gate and up branches, combined as
`output = silu(gate) * up`. This SwiGLU provides nonlinear feature mixing within each
scan layer, filling the role a separate FFN would otherwise serve.

### Memory Injection

Memory reads use the **frozen** segment-start state. PM and EM reads are pure functions
that produce additive residuals injected between the pre-memory and post-memory layers.

```python
def memory_reads(self, H, seed):
    """Read-only memory operations at the injection point.

    Returns:
        pm_read:  [BS, N, D]  — PM fast-weight reads, summed over B
        em_read:  [BS, N, D]  — EM trail reads, summed over B
    """
    # PM: Hebbian fast-weight read (bank-summed)
    pre = proj_in(H)                                   # [BS, N, D_pm]
    W_sum = W_pm.sum(dim=B)                            # [BS, D_pm, D_pm]
    pm_read = proj_out(pre @ W_sum.T)                  # [BS, N, D]

    # EM trail read (summed over B internally)
    em_read = trail_read_all(seed)                     # [BS, N, D]

    return pm_read, em_read
```

**Additive injection:** `H = H + pm_read + em_read`. Parameter-free, gradient-friendly,
degrades gracefully when memory is empty (both start at zero).

### Post-Memory Layers: Integration Scan

Post-memory scan layers integrate the memory-enriched representation into final predictions.

```python
def post_memory(self, H):
    """Post-memory layers + output head."""
    for layer in layers[L_mem:L_total]:
        H = layer(H)                # [BS, N, D]

    out = proj_down(H)              # [BS, N, D_embed]
    logits = lm_head(out) * D_embed**-0.5  # [BS, N, vocab]
    return logits
```

### Segment-End Commits

Memory commits happen once per segment, after output computation. This is the only
point where memory state is modified.

```python
def memory_commits(self, pm_pre, surprise, w_cand):
    """Segment-end memory commits (PM Hebbian + EM neuromodulated)."""
    # PM: Hebbian update
    s = sigmoid(surprise.norm(dim=-1))                 # [BS, N]
    G = (sqrt(s)*pre).T @ (sqrt(s)*pre) / N           # [BS, D_pm, D_pm]
    T = decay * I + beta[:, None, None] * G            # [BS, B, D_pm, D_pm]
    W_pm = W_pm @ T                                    # batched matmul, per bank

    # EM: novelty from write candidate energy + surprise magnitude
    novelty = w_cand.norm() + surprise.norm()           # [BS, N, B]
    em.base_decay()                                     # decay em_S
    g_em = neuromod(mean_novelty, usage)                # [BS, B]
    em.commit_all(w_cand, novelty, g_em)                # decompose across primitives
```

**Novelty computation:** `novelty = w_cand.norm(dim=-1) + surprise.norm(dim=-1)`. The
write candidate energy ensures non-zero writes even without PCM (when surprise=0).
Surprise magnitude adds discriminative gating when PCM is active.

**Neuromodulator:** Runs once at segment end on `(mean_novelty, usage)` — not per-token.
This is much simpler than the old per-token neuromodulator and reduces compute cost.

**State vs parameters:** The fast-weight matrices (W_pm) and EM primitives (em_K, em_V) are
**state**, not parameters. Parameters (proj_in, proj_out, raw_beta, W_seed_w, W_gate,
neuromodulator, τ, σ) are frozen after training. State evolves at inference — this is lifelong learning.

## Predictive Coding: Vector Surprise

### PCM in the Scan

PCM operates within the pre-memory layers. Each column predicts what the **next
token's** encoding will look like. Surprise is the vector difference between
the prediction and the actual encoding.

```
Token t-1: column state H_{t-1} → z_hat_{t-1} = W_pcm · H_{t-1}  (prediction)
Token t:   column input X_t → z_t = W_enc · X_t                    (actual encoding)
           surprise_t = z_hat_{t-1} - z_t                           (VECTOR, D_col dims)
```

**Vector surprise** (not scalar): Each dimension of D_col carries its own
surprise signal. The model was wrong about feature 3 but correct about feature
7 — memory writes can be modulated per-feature.

### PCM Optimization

1. **Auxiliary prediction loss**: `L_pred = MSE(z_hat_t, z_{t+1}.detach())`
   Gradient flows to the prediction network but not the encoding target.
2. **Downstream (cross-segment)**: surprise → novelty → EM commit →
   next segment's trail reads → loss. PCM encoder learns what's worth predicting.
   PM proj_in/proj_out get same-segment gradient through pm_read.

## Memory System Details

### PM: Procedural Memory — Hebbian Fast-Weight Network

PM is a **per-bank fast-weight matrix** — a small neural network whose weights
ARE the state, updated by Hebbian learning.

```
# State: one weight matrix per bank
W_pm [BS, B, D_pm, D_pm]                    # evolves at inference

# Read (bank-summed — no [BS, N, B, D] intermediate):
pre = proj_in(H)                             # [BS, N, D_pm]
post = (Σ_b W_b) @ pre                      # [BS, N, D_pm]  — single matmul
pm_read = proj_out(post)                     # [BS, N, D]

# Segment-end commit (Hebbian autocorrelation, 1 batched matmul):
s_t = sigmoid(‖surprise_t‖)                 # gate: high surprise → more learning
G = (1/N) Σ_t s_t · pre_t⊗pre_tᵀ           # [BS, D_pm, D_pm]
W_b ← W_b @ (decay·I + β_b·G)               # per-bank update, β_b differs per bank
clip Frobenius norm to budget_pm
```

**Why a matrix, not a vector:** A matrix encodes input→output transforms —
"when I see pattern A, produce pattern B." A bias vector can only shift
all inputs uniformly.

**Multiple timescales via banks:** Each bank has its own β_b (plasticity rate).
Fast-updating bank ≈ recent context habits. Slow-updating bank ≈ deep long-term habits.

**Compute cost:** ~84M ops/segment — 0.24% of one scan layer. Negligible.

### EM: Primitive Dictionary with Trail-Based Composition

EM stores M **primitive patterns** — atomic building blocks of concepts. Reading
is done via a **trail**: a seed vector navigates primitive space through
iterative refinement. The output is a composition of activated primitives.

**State (in compressed latent space D_mem):**
```
em_K   [BS, B, M, D_mem]  — primitive keys (unit-norm): "what triggers this"
em_V   [BS, B, M, D_mem]  — primitive values: "what this contributes"
em_S   [BS, B, M]         — strengths (0 = inactive)
```

**Read (trail-based pattern completion):**
```
seed_mem = mem_proj_in(seed)                 # [BS, N, D_mem]
y = seed_mem
for step in range(n_steps):
    scores = y @ em_K[b].T / τ               # [BS, N, M]
    scores[em_S == 0] = -inf                 # mask inactive
    attn = softmax(scores)                   # sparse activation
    delta = attn @ em_V[b]                   # primitive composition
    gate = sigmoid(g_alpha * dot(y, delta) + g_bias)
    y = y + gate * delta                     # seed moves through space
y_em = mem_proj_out(y - seed_mem)            # [BS, N, D]
```

**Write (segment-end commit, full softmax routing):**
```
# Soft routing: which primitives absorb this signal?
route = softmax(normalize(w_cand_mem) @ em_K[b].T / τ_w)  # [BS, N, M]

# Aggregate across N tokens, weighted by novelty
update_K = sum_n(novelty * route * k_cand)   # [BS, B, M, D_mem]
update_V = sum_n(novelty * route * v_cand)   # [BS, B, M, D_mem]

# Neuromodulated EMA commit
g_em = neuromod(mean_novelty, usage)          # [BS, B]
α = clamp(g_em * route_aggregate)             # [BS, B, M] per-primitive
em_K = (1 - α) · em_K + α · normalize(update_K)
em_V = (1 - α) · em_V + α · update_V
em_S = clamp(em_S + α, max=S_max)
```

No top-k routing — full softmax over all M primitives.

### Memory Decay and Budget

W_pm decays toward (1/B)·I via the `decay` factor in each Hebbian commit.
`em.base_decay()` applies per-bank strength decay to em_S — runs once per segment
before commits. EM budget enforcement soft-prunes weakest primitives when total
strength exceeds the budget.

## EM Latent Compression

EM operates in a compressed latent space D_mem (512 for Tier A) rather than the full
model width D. Learned projections `mem_proj_in` (D→D_mem) and `mem_proj_out` (D_mem→D)
inside EpisodicMemory handle the mapping.

## Gradient Flow

### Within-Segment (direct)

```
loss → logits → post-memory layers →
  → pm_read (fast-weight):
      → proj_out → W_sum = Σ_b W_pm[b] → proj_in
      (W_pm itself is state — gradient for raw_beta flows cross-segment)
  → em_trail (frozen primitives):
      → through trail steps → softmax → em_K, em_V, W_gate, W_seed_w, τ, σ
  → pre-memory layers → projection params, PCM params
```

### Cross-Segment (requires TBPTT)

```
TBPTT chunk = K segments (e.g., K=16)
Segment 1: forward → memory committed → carry to segment 2
Segment 2: forward → loss flows back through trail reads → segment 1's commit
```

### What Gets Gradient From Where

| Component | Gradient source | Path |
|-----------|----------------|------|
| Scan params (a, b projections) | Same segment's NTP loss | Direct through scan |
| PCM encode/predict | L_pred (auxiliary) + downstream | Same segment |
| PM proj_in / proj_out | Same segment's loss through pm_read | Direct |
| PM raw_beta | **Next segment's** loss through committed W_pm → read | Cross-segment TBPTT |
| EM trail params (W_seed_w, τ, σ) | Same segment's loss through trail | Through softmax |
| EM primitives (em_K, em_V) | Same segment's loss through trail read | Through softmax attention |
| W_seed_w | **Next segment's** loss through commit → em_K/V → trail read | Cross-segment TBPTT |
| EM neuromodulator | **Next segment's** loss through commit → em_K/V → trail read | Cross-segment TBPTT |
| EM raw_decay | **Next segment's** loss through decay → em_S → usage → neuromod → alpha → em_K/V | Cross-segment TBPTT |

## Correctness Analysis

### 1. Within-Segment: Causal (NTP)

Both pre-memory and post-memory scans are causal recurrences — token t only sees
tokens 1..t-1 through the hidden state. Memory reads are from **segment-start**
state (frozen within segment).

### 2. Cross-Segment: Causal

Memory state at segment k reflects only tokens from segments 0..k-1. Writes from
segment k are committed at the end and only visible to segment k+1.

### 3. Train/Inference Equivalence

**Training**: Process N-token segments, NTP loss on all positions.
**Inference**: Process tokens through the scan stack. Can be done:
- Segment-at-a-time: process N tokens, get N predictions (most efficient)
- Token-at-a-time: update scan state per token, defer memory ops to segment
  boundaries (streaming mode)

Inference memory: scan states (d_inner per layer) + PM + EM. **Constant**,
does not grow with sequence length.

## Practical Design Decisions

### Column Dimensions

```
# Tier A
D = 2048          # internal model width
D_embed = 768     # embedding / LM head width
B = 4             # memory banks
C = 16            # columns per bank
D_col = 128       # = D / C — column width
N = 128           # segment length
K_segments = 16   # TBPTT chunk = 2048 tokens
L_total = 10      # total scan layers
L_mem = 5         # memory injection point
```

### Feature Mixing

Dense scan layers (nn.Linear) mix all D features freely for GPU efficiency. PCM and
W_seed_w retain grouped structure via free .view() reshaping.

### Design Decisions (Resolved)

1. **Scan formulation** → Simple linear recurrence (`h_t = a_t ⊙ h_{t-1} + b_t`).
   No SSM. Memory systems handle the role SSM's expanded state was designed for.

2. **Scan mixing** → Dense nn.Linear projections mix all D features.

3. **FFN** → GLU output projection provides nonlinear feature mixing within each
   scan layer: `output = silu(gate) * up`.

4. **Single scan stack** → L_total layers with memory injection at L_mem.
   Simpler than the old 3-stage cycle — memory reads are just additive residuals.

5. **Memory integration** → Additive at L_mem: `H = H + pm_read + em_read`.

6. **No within-segment writes** → Segment-end commits only. N=128 makes this
   frequent enough. Removes complexity of cum_em, per-token neuromodulator, W_nov.

7. **PM** → Hebbian fast-weight matrix (D_pm × D_pm per bank).

8. **EM** → Trail-based primitive composition. Full softmax routing (no top-k).

9. **No attention** → Keeps compute O(N) not O(N²).

### Implementation Clarifications

**Bank-column mapping (H ↔ memory):**
H is `[BS, N, D]` (dense). Memory is `[BS, B, ...]` (bank-indexed).
All B banks share the same H. Memory reads return `[BS, N, D]` signals (bank-sum
computed via fused algebra).

**Seed and w_cand are shared across banks (grouped projection):**
`W_seed_w` is a GroupedLinear(C, D_col, 2*D_col) producing `[BS, N, C, 2*D_col]`,
split via `chunk(2)` into seed and w_cand — each `[BS, N, D]`.

**w_cand serves as both routing key and write content** in EM commits.

**PM proj_out zero-initialized:** PM starts silent, gradually learns to contribute.

**em_K/em_V and W_pm: state in computation graph:**
These are plain tensors, NOT nn.Parameter. They are NOT detached within a
TBPTT chunk — gradient flows through them via the commit chain from the
previous segment. At TBPTT chunk boundaries, `detach_states()` breaks the chain.

**Scan layer structure (dense, with GLU output):**
Each scan layer: dense nn.Linear input projection `[D → 2*d_inner]` producing
(a_raw, b_raw), then `a = sigmoid(a_raw)`, `b = silu(b_raw)`. When glu_output=True,
output projection is `[d_inner → 2*D]`, split into gate and up, combined as
`output = silu(gate) * up` (SwiGLU). RMSNorm(D) follows.

## Configuration Reference

### Hyperparameter Glossary

| Symbol | Config field | Meaning |
|--------|-------------|---------|
| **D** | `D` | Model width — the main hidden dimension |
| **D_embed** | `D_embed` | Embedding/unembedding dimension |
| **D_mem** | `D_mem` | EM latent compression dimension |
| **C** | `C` | Number of cortical columns (feature groups) |
| **D_col** | `D_col` | Column width = D / C |
| **B** | `B` | Memory banks |
| **N** | `N` | Segment length in tokens |
| **K_segments** | `K_segments` | TBPTT chunk = K_segments × N tokens |
| **L_total** | `L_total` | Total scan layers in the single stack |
| **L_mem** | `L_mem` | Memory injection point (reads happen after this layer) |
| **d_inner** | `d_inner` | Scan recurrence hidden dim per layer |
| **M** | `M` | EM capacity — number of primitive patterns per bank |
| **D_pm** | `D_pm` | PM fast-weight matrix dimension |
| **n_trail_steps** | `n_trail_steps` | EM trail iteration depth |
| **decay_pm** | `decay_pm` | Per-segment multiplicative decay for W_pm |
| **decay_em** | `decay_em` | Per-segment multiplicative decay for em_S |
| **budget_pm** | `budget_pm` | Max Frobenius norm per PM bank |
| **budget_em** | `budget_em` | Max sum(em_S) per stream |
| **glu_output** | `glu_output` | SwiGLU output on scan layers |

### Tier Comparison

| | **Tier A** | **Tier B** | **Tier C** |
|---|---|---|---|
| **Total params** | **~116M** | **~250M** | **~844M** |
| D (model width) | 2048 | 3072 | 4096 |
| D_embed | 768 | 1024 | 2048 |
| D_mem (EM latent) | 512 | 512 | 1024 |
| C (columns) | 16 | 16 | 16 |
| D_col = D/C | 128 | 192 | 256 |
| B (banks) | 4 | 6 | 8 |
| L_total | 10 | 20 | 28 |
| L_mem | 5 | 10 | 14 |
| d_inner | 1024 | 1024 | 2048 |
| glu_output | True | True | True |
| M (EM slots/bank) | 384 | 512 | 768 |
| D_pm (PM matrix) | 64 | 64 | 64 |
| n_trail_steps | 3 | 2 | 3 |
| N (segment) | 128 | 128 | 128 |
| K_segments | 16 | 16 | 16 |
| **Comparable to** | Pythia-160M, Mamba-130M | Mamba-370M, Pythia-410M | Qwen3.5-0.8B, Mamba-1.4B |

### State vs Parameters

**Trained parameters** (fixed after training):
- Scan layer weights (proj_in, proj_out, RMSNorm)
- PCM weights, W_seed_w
- PM projection weights (proj_in, proj_out) and per-bank β (plasticity rate)
- EM neuromodulator MLP
- EM trail parameters (gate_alpha, gate_bias, tau, sigma, tau_w)

**Runtime state** (stream-specific, evolves during inference):
- `W_pm [BS, B, D_pm, D_pm]` — PM fast-weight matrix
- `em_K [BS, B, M, D_mem]` — EM primitive keys
- `em_V [BS, B, M, D_mem]` — EM primitive values
- `em_S [BS, B, M]` — EM primitive strengths
- `_carries [L_total]` — scan recurrence state per layer
