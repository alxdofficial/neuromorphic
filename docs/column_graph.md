# ColumnGraph — Plastic Concept Graph with Cross-Attention I/O

> **HISTORICAL** — superseded by `docs/graph_walker.md` as the active
> design direction. The column_graph code in `src/column_graph/` is
> kept in tree for reference but is not actively trained on.

**Branch:** TBD (from `main`), supersedes `gridworld`.
**Status:** design in progress (as of 2026-04-23). Sections 1-8 agreed;
section 9 lists what's still open.
**Date:** 2026-04-23.

## 1. Core thesis

> **Retrieval as trajectory through a plastic concept graph, not
> nearest-neighbor in high-dimensional space.**
>
> Standard vector RAG picks a query vector, searches for its k nearest
> neighbors in an embedding space, and concatenates them into a prompt.
> Compositionality is implicit in the embedding — you hope the dimensions
> encode the right semantic axes — and performance degrades with dimension
> because high-dim similarity is dominated by noise.
>
> We propose the opposite: a graph of **typed concept columns** (column =
> compound unit, identity vector encodes what the column means), with
> **plastic sparse local connectivity**, and retrieval realised as
> **propagating signals through the graph for T steps** after injecting the
> query at an input plane. The output is read via cross-attention from an
> output plane. The graph rewires itself over time through local Hebbian
> updates gated by a neuromodulator — surprise-driven plasticity shapes
> which columns talk to which.
>
> **Why this helps over RAG:** the "path" a signal takes through the graph
> before landing on the output plane encodes composition. An "eat" signal
> that passes through a "food" region before hitting a "colour" column
> biases the output toward food-specific colour meanings, not all white
> things. Co-activation history, frozen into plastic edge biases, makes
> frequently-coherent trajectories cheaper to traverse.
>
> **Why this is the same thesis as gridworld:** a learned neuromodulator
> policy gates local Hebbian plasticity on plastic weights; *that IS the
> learning rule,* and it runs forever at inference. What changed is the
> substrate — compound columns with shared internal weights (GPU-friendly,
> tensor-core native) instead of tiny per-neuron matrices, and explicit
> cross-attention I/O planes instead of scatter-based input positions.

## 2. TL;DR

> - **Columns**, not individual neurons. Each column is a compound unit
>   with `D_s = 128` state dim, a learned identity vector `id_c`, and
>   shared-weight internal MLPs. Tensor cores see real matmul shapes.
>
> - **3D graph.** L planes, each a 2D grid of columns. Per-column
>   connectivity is K fixed neighbours drawn from {Moore radius within
>   own plane, Moore radius to next plane, Watts-Strogatz shuffled for
>   long-range skips}. Locally dense, globally sparse.
>
> - **Directional message passing.** At each step every column computes
>   one content vector `m_out` and K scalar scores (one per out-edge).
>   Per-edge weight = `sigmoid(score + E_bias)`. Message sent is
>   `weight · m_out`. `E_bias[A, B]` and `E_bias[B, A]` are separate
>   plastic scalars, guaranteeing the two directions evolve independently.
>
> - **Single-hop-per-token streaming dynamics.** For each input token:
>   inject at input plane, run **one round** of (send → sparse gather →
>   update), read out from output plane. State persists across tokens
>   within a segment, so depth of computation comes from the token
>   sequence itself — over T_seq=256 tokens the state evolves 256 hops
>   through the graph, which with high p_rewire fully mixes signals
>   across the whole graph in ~log(N) tokens.
>
> - **I/O via cross-attention.** Token stream → cross-attention into
>   input-plane column states at t=0. Output-plane column states →
>   cross-attention → motor vector → multi-horizon factored readout →
>   `[B, K_horizons, V]` logits.
>
> - **Multi-horizon surprise.** Ring buffer of past K_horizons predictions;
>   compare against actual token; maintain per-horizon surprise EMA. Same
>   mechanism as current gridworld (`readout.py` reused).
>
> - **Neuromodulator gates plasticity, doesn't predict it.** Neuromod
>   observes surprise (primary), tile-level activity, and per-column
>   features; outputs global rate `η_global`, per-column gate `η[c]`, and
>   per-column LTP/LTD threshold shift `β[c]`.
>
> - **Hebbian edge-bias update** with Oja-style decay:
>   `ΔE_bias[A, B] = η_global · η[A] · η[B] · (coact[A, B] - β[B]·||E_bias[A, :]||)`
>   clipped to a bounded range. Done in fp32 under `autocast(enabled=False)`.
>
> - **Dual-use.** The column graph is an autonomous memory module with
>   abstract cross-attn I/O. In standalone mode, token_emb → inject and
>   readout → unembed. In attachment mode (later), LM hidden states →
>   inject and readout → back into LM stream. Same substrate, different
>   heads.

## 3. Scope

### 3.1 What's in scope for v1 (standalone)

- Column graph memory module with plastic edges + neuromod + multi-horizon
  surprise.
- Token-level standalone language model: `token_emb → memory.step → unembed`.
- Training phase 1: autoregressive next-token CE + TBPTT.
- bf16 autocast + fp32 optimizer; plasticity / surprise math forced fp32.

### 3.2 What's deferred to v2+

- **Attachment mode** — memory as a module hooked into a pretrained
  Llama-class LM via MemInjectLayer-style fusion. Designed-in but
  unimplemented.
- **Phase 2 (GRPO / RL)** — the inference-time neuromod diversity injection
  policy from gridworld is orthogonal and can be re-bolted once phase 1
  converges.
- **Typed edges (option b/c from design discussion)** — scalar edge bias
  is the starting point. Edge types with shared matrices are a follow-up
  only if expressiveness proves tight.

### 3.3 What is explicitly *not* the thesis

- Not claiming per-neuron plasticity (gridworld's failed ambition).
  Plasticity lives on edges; columns themselves have shared weights
  trained by normal backprop.
- Not a transformer. The topology is fixed-degree sparse, not all-to-all
  attention. The plasticity is explicit and slow, not re-computed per
  forward pass.

## 4. Architecture

### 4.1 Column

Each column `c` has:

- **State** `s_c ∈ ℝ^{D_s}`, persistent across ticks (with TBPTT detach at
  segment boundaries), bf16 storage, fp32 compute in plasticity paths.
- **Identity** `id_c ∈ ℝ^{D_id}`, learned by gradient, slow-moving.
  Stable semantic name for this column across the life of training.
  Initial plan: learned from scratch with small Gaussian init; see open
  question 9.2.
- **Position in the graph** — plane index, (row, col) within plane.
  Static. Used at construction to compute `out_nbrs[c]`.

**Column compute per propagation round** (same code for every column):

```
# Input: own state s[c], own identity id[c], incoming messages incoming[c]
# Output: updated state s'[c], outgoing content m_out[c], K edge scalars

s_pre    = RMSNorm(s[c])
s_in     = concat(s_pre, id[c], incoming[c])           # [D_s + D_id + D_s]
s'[c]    = s[c] + update_MLP(s_in)                     # residual update

m_pre    = RMSNorm(s'[c])
m_out[c] = content_MLP(concat(m_pre, id[c]))           # [D_s]

# Scores against each of K out-neighbours — batched across k
for k in 0..K-1:
    dest = out_nbrs[c, k]
    score[c, k] = score_MLP(concat(s'[c], id[c], id[dest]))
```

Shapes at starting scale (N = 4096, D_s = 128, D_id = 32, K = 32):

- `update_MLP`: `(D_s + D_id + D_s) → 4·D_s → D_s`, i.e. `[288, 512, 128]`.
  Shared across all N columns and all T rounds. Tensor cores fill.
- `content_MLP`: `(D_s + D_id) → 2·D_s → D_s`, i.e. `[160, 256, 128]`.
- `score_MLP`: `(D_s + 2·D_id) → D_s/2 → 1`, i.e. `[192, 64, 1]`. Batched
  across all `N · K` (source, out-neighbour) pairs.

FFN multiplier, activation (GELU by default), norm placement follow
transformer best practice. All MLPs are bf16-autocast.

### 4.2 Topology

- **L planes** (default L = 4), each a 2D torus of shape `(H, W)` with
  `H · W` columns per plane. Planes are labelled `0 = input`, `L-1 =
  output`, interior are hidden.
- **Per-column out-edges:** K fixed neighbours per column, drawn from:
  - Some fraction of K (default ~2/3) from intra-plane Moore radius r=2
    (25 candidates excluding self; pick the K_local lowest-index
    candidates deterministically).
  - Some fraction (default ~1/3) from inter-plane neighbours: same (row,
    col) on plane `p+1` (or `p-1` with wrap) plus small local radius
    around it.
  - **Watts-Strogatz shuffling**: `p_rewire = 0.30` of edges are replaced
    with uniformly random destinations anywhere in the full graph
    (bumped high specifically because the design uses T=1 per-token
    recurrence — the graph has to mix fast through shortcuts, since
    each token only gets one hop of propagation). Fixed at init,
    seeded deterministically.

    At p=0.30, N=4096, K=32, the probability that an input-plane column
    has at least one direct edge to the output plane is ≈1 −
    (1 − 0.30·1/4)^32 ≈ 0.93. Nearly every input column has multiple
    direct-to-output shortcuts, so injected signals reach the output
    plane within one hop (reduced amplitude; 1-few edges among K=32) and
    build amplitude over subsequent tokens as the state evolves.

**Physical layout is static after initialization.** No structural
plasticity in v1 — column positions, plane assignments, and the set of
K out-neighbour indices never change. Only edge *biases* `E_bias` move.
This keeps the forward pass's data layout stable, makes the inverse
adjacency a one-time computation, and isolates the "learning" story
cleanly inside plasticity of scalars.
- **Directionality of topology is symmetric by construction** (A in B's
  out-nbrs ⇔ B in A's out-nbrs, modulo the Watts-Strogatz random
  rewiring). The *weights* on the two directed edges evolve
  independently (see 4.3).
- Input-plane columns and output-plane columns participate in the
  topology just like hidden columns — they have K out-neighbours and get
  K in-neighbours. What distinguishes them is only that the input plane
  receives cross-attention injection and the output plane produces
  cross-attention readout.

Buffers registered on the module:

- `out_nbrs [N, K]` — int64, each row lists the K out-neighbour column
  indices for a source column.
- `in_nbrs_flat [N·K]`, `in_nbrs_offsets [N+1]` — CSR-style inverse
  adjacency for efficient gather during the receive phase.

### 4.3 Edge math (directional scoring)

The receive phase uses the CSR inverse adjacency. For each destination
column `c`, aggregate incoming messages as:

```
incoming[c] = Σ_{A : (A→c) ∈ edges} w_out[A, c] · m_out[A]
```

where

```
w_out[A, c] = σ(score[A, k_A_to_c] + E_bias[A, c])
```

and `k_A_to_c` is the edge index in A's out-neighbour list that points
to c (precomputed once at init).

**Directionality guaranteed by:**

1. `s[A] ≠ s[B]` in general, so m_out and score depend on different
   inputs when A and B are swapped.
2. `score_MLP(s_A, id_A, id_B) ≠ score_MLP(s_B, id_B, id_A)` in general,
   because the MLP is nonlinear and the identity arguments are in
   different positions.
3. `E_bias[A, B]` and `E_bias[B, A]` are stored as separate scalars in
   a `[N·K]` flat tensor (indexed by the directed edge identifier).
   Plasticity updates them independently.

**Storage:** `E_bias [N·K]` scalars, fp32, plastic state (not in the
optimizer — updated by Hebbian rule). Initialized to zero at segment
start.

**Per-edge content variation.** The content that A sends is not
uniform across A's out-edges — it gets a small additive correction
conditioned on the receiver's identity:

```
m_edge[A, k] = m_out[A] + delta_MLP(concat(m_out[A], id[out_nbrs[A, k]]))
```

`delta_MLP` is shared across all columns and all edges, batched over
`N·K` pairs. Shape: `(D_s + D_id) → D_s/2 → D_s`. Residual from the
main `m_out` so the starting behaviour is "same message to every
neighbour" and the delta learns tailoring over training. Keeps the
computational pattern a standard big batched matmul (tensor-core
friendly) while giving the column an expressive knob.

**Send phase produces** the per-directed-edge message tensor:

```
msg_edge [N, K, D_s] = w_out[:, :, None] * m_edge[:, :, :]
```

A single elementwise multiply + a batched MLP over the N·K pairs.
Tensor cores fill both the `delta_MLP` call and the `content_MLP`
earlier.

### 4.4 Dynamics

**Per-token sequence** (one token at a time, one propagation round per
token — no within-token recurrence):

```
1. receive token_t
2. h_input = token_emb(token_t)                                # [B, D_inject]
3. per-column gated input injection at input-plane columns
4. one round of column compute:
     compute m_out, score for all columns (batched over N)
     w_out = σ(score + E_bias[flat])                            # [N, K]
     gather incoming per column via CSR inverse adjacency
     s ← gated-tanh update (see 4.1)
5. cross_attn readout: motor ← (output-plane states → k, v; learned_q)
6. logits [B, K_horizons, V] = factored_readout(motor, horizon_emb, W_unembed)
7. multi_horizon_surprise(pred_buf, token_t, surprise_ema)
8. write_prediction_buffer(pred_buf, logits)
9. every mod_period=16 tokens: neuromod forward + plasticity update
10. advance to t+1
```

**T = 1** — one hop of propagation per token. The computational depth
comes from T_seq (sequence length). Across 256 tokens the state takes
256 hops; with p_rewire=0.30 the small-world mixing time is a few
tokens, so by token ~5 every column has been influenced by every
injected token's signal (diluted by distance).

Conceptually: **the graph is a streaming computational medium**, not a
pond that settles per input. Each token nudges the state one step;
thinking unfolds across the stream, not within one token. This is
closer to how real cortex operates (continuous processing of
streaming input) and much cheaper per token than T-round within-token
recurrence.

**Precision:**

- Column states `s`, identities `id`, MLP weights: bf16 autocast, fp32
  master weights (standard recipe).
- `E_bias`, plasticity math, surprise math: forced fp32 via
  `autocast(enabled=False)`.
- Neuromod features aggregation: fp32.
- Multi-horizon CE loss: fp32.

### 4.5 Cross-attention I/O

**Input injection** (run once per token, at t=0 before propagation
rounds):

```
h_input = token_emb(token_t)                        # [B, D_inject = D_s]
q = input_plane_states                              # [B, N_in, D_s]
k = input_proj_k(h_input)[:, None, :]               # [B, 1, D_s]
v = input_proj_v(h_input)[:, None, :]               # [B, 1, D_s]
Δs = scaled_dot_product_attention(q, k, v)          # [B, N_in, D_s]
input_plane_states ← input_plane_states + Δs        # residual injection
```

Multi-head (say 4 heads of 32 dim) is fine. Single-token injection means
the attention is a glorified broadcast with learned gating. If we go
chunked (see 9.5), this becomes a T_chunk × N_in attention matrix.

**Output readout** (run once per token, at t=T after propagation rounds):

```
kv = output_plane_states                            # [B, N_out, D_s]
q  = motor_query[None, None, :].expand(B, 1, D_s)   # learned query
motor = scaled_dot_product_attention(q, kv, kv)     # [B, 1, D_s]
motor = motor.squeeze(1)                            # [B, D_s]
motor = pred_head(motor)                            # optional residual head
logits_motor   = motor @ W_unembed.T                # [B, V]
logits_horizon = horizon_emb @ W_unembed.T          # [K_horizons, V]
logits         = logits_motor.unsqueeze(1) + logits_horizon  # [B, K_h, V]
```

Factored readout identical to current `readout.py::MultiHorizonReadout`.
Tied `W_unembed = token_emb.weight`.

### 4.6 Multi-horizon surprise

Reused unchanged from current gridworld (`src/gridworld/readout.py`):

- `pred_buf [B, K_buf, K_horizons, V]` ring buffer of past logit predictions.
- `write_prediction_buffer` at step end.
- `multi_horizon_surprise` compares each horizon k's past prediction
  against current token, updates `surprise_ema[B, K_horizons]`.
- `read_past_prediction` returns `.detach().clone()`'d views so ring
  buffer writes don't break autograd.

`K_horizons = 8, K_buf = 8`. Cross-entropy + EMA math in fp32 under
`autocast(enabled=False)`.

### 4.7 Plasticity (post-synaptic-local Hebbian + neuromod gating)

**Design principle: one-direction-at-a-time, post-synaptic-local.**
Every plastic edge `E_bias[A, B]` belongs, administratively, to its
destination column `B`. Column B's local neuromod is responsible for
adjusting B's *incoming* edges `E_bias[*, B]` based on signals
observable *at B*. Column A's outgoing edge `E_bias[A, B]` is not
updated by A — that direction is B's problem. Symmetrically, the
reverse edge `E_bias[B, A]` is updated by column A's local neuromod.
The two directed edges between A and B evolve through completely
separate decision processes.

Biologically this is the correct mapping: post-synaptic calcium and
post-synaptic spike timing gate LTP/LTD at the dendrite; the
presynaptic neuron doesn't "decide" how strongly to potentiate its
output synapse.

**Observables that column B's neuromod sees** (all local to B):

- B's own state `s[B]` and identity `id[B]`
- B's activation norm `post[B] = ||incoming[B]||`
- The K incoming edge weights `w_in[k→B] = w_out[in_nbrs[B, k], B]`
- The K pre-synaptic activity proxies `pre[in_nbrs[B, k]]`
- A **broadcast global context vector** `g` derived from aggregate
  observables (surprise_ema, Δsurprise, input_ctx_ema). This is shared
  across all columns — it carries "how surprised is the model right
  now," which has to be a global signal even though each column's
  decision is local.

**Cadence.** Neuromod fires once every `mod_period = 16` tokens, matching
the rhythm of other branches. Between firings, plastic state is frozen.

**Efficient vectorized computation** (important — per-edge work must
batch properly):

```
# All fp32, under autocast(enabled=False).

# --- Per-column activity proxies — O(N · D_s) ---
pre[c]  = || m_out[c] ||_2 / sqrt(D_s)       # [N]  — sender activity
post[c] = || incoming[c] ||_2 / sqrt(D_s)    # [N]  — receiver activity

# --- Neuromod forward pass — one big batched call over all N columns ---
local_feats[c] = concat(
    post[c],                                  # scalar
    w_in[c, :],                               # [K]
    pre[in_nbrs[c, :]],                       # [K]   — gather
    mag_ema[tile[c]],                         # tile broadcast
    var_ema[tile[c]],                         # tile broadcast
    traffic[tile[c]],                         # tile broadcast
)                                             # [D_local ≈ 2K + 4]

global_feats = concat(surprise_ema, Δsurprise, input_ctx_ema)
g            = trunk_MLP(global_feats)        # [D_trunk], shared

η[c]  = softplus( head_η(concat(g, local_feats[c], id[c])) )   # [N], scalar
β[c]  = tanh(     head_β(concat(g, local_feats[c], id[c])) )   # [N], scalar
η_global = softplus( head_global(g) )          # [scalar]

# --- Per-edge Hebbian update, post-synaptic-gated — O(N · K) ---
# For each directed edge (A → B), update using only B's neuromod output.
# Gather pre[A] via in_nbrs index.
pre_per_edge[B, k]  = pre[in_nbrs[B, k]]      # [N, K]

coact[B, k]         = pre_per_edge[B, k] · post[B] · w_in[B, k]
decay[B, k]         = β[B] · E_bias_in[B, k]  # Oja-like, per-edge

ΔE_bias_in[B, k]    = η_global · η[B] · ( coact[B, k] − decay[B, k] )

E_bias_in[B, k]    += ΔE_bias_in[B, k]
E_bias_in           = clamp(E_bias_in, −E_max, +E_max)   # E_max = 4

# Scatter E_bias_in [N, K] back into the flat directed-edge storage
# used by the forward pass (see 4.3) via a precomputed scatter index.
```

**Where the cost lives:**

- Gathers `in_nbrs[c, :]` and `pre[in_nbrs[c, :]]`: `N · K` reads, ~4 MB
  of HBM traffic per call. BW-bound but tiny absolute.
- Neuromod forward over N columns: the heads are small MLPs, the trunk
  fires once globally. Batched as `[N, D_local + D_trunk + D_id]` input
  → `[N, 2]` output (η, β). Tensor cores fill.
- Per-edge Hebbian: `N · K = 131 K` scalar FMAs. Trivial.

**Total plasticity step cost at dev scale: ~3 MFLOPs + ~5 MB HBM, fires
once per 16 tokens.** Negligible next to the forward propagation
(~260 MFLOPs per token). Plasticity is not a performance concern.

**Signs:**

- `η_global, η[B] ≥ 0` via softplus.
- `β[B] ∈ [−1, +1]` via tanh. Positive β → incoming edges with high
  pre·post correlation strengthen (LTP); negative β → those edges
  weaken (LTD).

**Storage layout:** `E_bias_flat [N · K]` in the directed-edge order
used by forward. The plasticity step uses `E_bias_in [N, K]` (indexed
by destination column + in-edge slot) which is just a view / reshape
of the same underlying storage given a fixed `scatter_idx` computed at
init. No memory duplication.

Detached at segment boundaries; TBPTT doesn't backprop through the
Hebbian update — plasticity is local, not a second-order signal for
the trunk.

### 4.8 Neuromodulator (column-local, with broadcast global context)

**Architecture is two-level:**

1. One **global trunk** fires once per plasticity step, consuming
   aggregate observables (surprise, Δsurprise, input-ctx, tile
   statistics). Produces a shared context vector `g`.
2. **Per-column heads** fire in parallel, one per column. Each head sees
   `g` (broadcast) plus the column's own local observables. Produces
   that column's `η[c], β[c]`.

This matches the biological picture: a diffuse modulatory signal
(dopamine / serotonin / acetylcholine) bathes the cortex uniformly, but
each post-synaptic neuron decides how to act on it locally based on its
own state. It's also computationally clean: trunk is one small MLP call,
heads are one big batched call over N columns.

**Trunk inputs (global, aggregate):**

- `surprise_ema [K_horizons]` — per-horizon surprise EMA. Primary signal.
- `Δsurprise [K_horizons]` — surprise_ema minus previous step's
  surprise_ema.
- `input_ctx_ema [D_ctx]` — EMA of the injected input vector.
- `mag_ema [num_tiles]`, `var_ema [num_tiles]`, `traffic [num_tiles]`
  — tile-level activity / variance / wiring-traffic summaries.

**Per-column head inputs (local):**

Everything observable *at column B*:

- `id[B] [D_id]`
- `post[B] [1]` — receiver activity norm
- `w_in[B, :] [K]` — current in-edge weights
- `pre[in_nbrs[B, :]] [K]` — per-in-edge sender activity
- `g [D_trunk]` — broadcast global context

**Head outputs:**

- `η[B]` via softplus — column B's plasticity rate, modulating all of
  B's incoming Hebbian updates.
- `β[B]` via tanh — LTP/LTD bias for B's incoming edges.

(Optional future: a second head producing a forward-compute gain that
multiplies `m_out[B]` — this is §9.8, deferred.)

**Module shape:**

- `trunk_MLP`: `(2·K_h + D_ctx + 3·num_tiles) → trunk_hidden → D_trunk`.
  `D_trunk = 384`, `trunk_hidden = 768`. Fires once per plasticity step.
- `head_η`, `head_β`: shared-weight MLPs of
  `(D_trunk + D_id + 1 + 2K) → head_hidden → 1`.
  `head_hidden = 64`. Batched over N columns. Tensor-core friendly.

Total neuromod params: ~1-3 M.

**Precision:** all neuromod feature extraction + forward in fp32 under
`autocast(enabled=False)`. Outputs feed the fp32 plasticity math in §4.7.

**Training:** parameters trained end-to-end by backprop through the
eventual loss (surprise ↔ next-token CE). No separate RL loop in v1.

**Why this beats "global neuromod with per-column heads is an
afterthought":** explicitly enforcing that the per-column decision uses
only locally-observable features from that column's perspective keeps
the plasticity rule faithful to the thesis ("learning is local"). A
neuromod that peeks at *other* columns' states to decide B's plasticity
would be laundering a global credit-assignment computation through the
"local" label. We forbid that: B's features only.

## 5. Dual-use interface

### 5.1 Module boundaries

```
ColumnGraphMemory(nn.Module)
    def __init__(self, cfg: ColumnGraphConfig)
    def step(
        self,
        inject: torch.Tensor,      # [B, D_inject]
        *,
        compute_plasticity: bool = True,
    ) -> torch.Tensor:             # [B, D_readout]
        ...
```

All internal state (column states, E_bias, pred_buf, surprise_ema,
input_ctx_ema, mod_counter) is owned by the module and persists across
`step` calls within a segment. Detach + reset on segment boundary.

### 5.2 Standalone LM (v1 target)

```
class StandaloneLM(nn.Module):
    memory:    ColumnGraphMemory
    token_emb: nn.Embedding(V, D_inject)
    unembed:   nn.Linear(D_readout, V, bias=False)   # tied to token_emb

    def forward(self, tokens: LongTensor) -> FloatTensor:
        for t in range(tokens.shape[1]):
            readout = self.memory.step(self.token_emb(tokens[:, t]))
            logits[:, t] = readout_to_logits(readout, ...)
```

### 5.3 Attachment LM (v2 target)

```
class AttachedMemoryLM(nn.Module):
    memory:  ColumnGraphMemory
    llm:     PreTrainedLlama       # frozen
    in_adapter:  nn.Linear(D_lm, D_inject)
    out_adapter: nn.Linear(D_readout, D_lm)

    def forward(self, tokens):
        # LM runs up to hook layer
        h_lm = self.llm.partial_forward(tokens, up_to=hook_layer)
        for t in range(T):
            inject = self.in_adapter(h_lm[:, t])
            readout = self.memory.step(inject)
            h_lm[:, t] = h_lm[:, t] + self.out_adapter(readout)
        logits = self.llm.finish_forward(h_lm)
```

The `ColumnGraphMemory` module is identical between the two modes. The
head swap is ~50 LOC.

## 6. Training recipe

### 6.1 Phase 1 (standalone)

- **Objective:** multi-horizon next-token CE, factored over K_horizons
  heads. Streaming across segment of length T_seq.
  ```
  loss = Σ_t Σ_k CE(logits[b, t, k], tokens[b, t + k])
  ```
  The k=1 loss is primary; higher-k are auxiliary and weighted with a
  small coefficient (default 0.2).
- **TBPTT:** detach column states + E_bias + surprise_ema + pred_buf
  every `tbptt_block = 32` tokens. Same as gridworld default.
- **Segment boundary:** at segment start, reset all persistent state to
  zero (states, E_bias, pred_buf, input_ctx_ema, surprise_ema).
- **Optimizer:** AdamW fused, bf16 autocast + fp32 master weights.
  Plastic `E_bias` is NOT in the optimizer (it's plastic state).
- **LR:** start at 1e-4 (same as gridworld experimentally-stable default).
- **Gradient clipping:** global norm clip at 1.0.

### 6.2 Phase 2 (autoregressive GRPO, reused from current branches)

Phase 1 is teacher-forced: at step t the memory sees `token_t` (ground
truth) and predicts `token_{t+1}`. Phase 2 switches to autoregressive
rollout:

1. At step t, the memory emits logits; we sample `token̂_{t+1}` from them
   (instead of taking the ground truth).
2. The *sampled* token is what gets injected on the next step.
3. Neuromod adds Gaussian noise to its own outputs (η, β) at each
   rollout step — this is the exploration signal.
4. Reward = reduction in multi-horizon surprise EMA over the rollout.
5. GRPO update: positive-advantage noise directions are reinforced,
   negative are discouraged. The neuromod policy learns to plasticize
   *in ways that reduce future surprise*.

The whole machinery — rollout harness, advantage computation, GRPO
update — is reusable from `src/gridworld/train_phase2.py`. Plastic
`E_bias` is the natural variable that receives phase-2 exploration
noise through the neuromod heads.

## 7. Scale / param budget

### 7.1 Starting configuration

| Knob | Value | Notes |
|---|---|---|
| N (columns total) | 4096 | 4 planes × 32×32 |
| L (planes) | 4 | 0 = input, 3 = output, 1-2 hidden |
| K (out-edges per column) | 32 | Fixed topology |
| D_s (state dim) | 128 | Matmul-friendly; tensor-core native |
| D_id (identity dim) | 32 | Small, learned per column |
| T (rounds per token) | 1 | Depth comes from T_seq, not T |
| p_rewire | 0.30 | Watts-Strogatz shuffle probability |
| K_horizons | 8 | Same as gridworld |
| D_ctx (input ctx embed) | 128 | |
| D_trunk | 384 | Neuromod trunk width |
| V (vocab) | 32000 | Same as gridworld |

### 7.2 Parameter counts

| Component | Params | Notes |
|---|---|---|
| token_emb / unembed (tied) | V · D_s = 4.1 M | |
| column identity table | N · D_id = 0.13 M | |
| update_MLP | (288 · 512 + 512 · 128) = 0.21 M | Shared |
| content_MLP | (160 · 256 + 256 · 128) = 0.07 M | Shared |
| score_MLP | (192 · 64 + 64 · 1) = 0.01 M | Shared |
| cross-attn input heads | ~0.1 M | |
| cross-attn output heads | ~0.1 M | |
| horizon_emb + pred_head | K_h · D_s + D_s² = 0.02 M | |
| neuromod trunk + heads | ~1.5 M | |
| **Total slow params** | **~6.3 M** | |
| **Plastic E_bias** | N · K = 131 K scalars | State, not params |

### 7.3 Compute per token (BS = 1, T = 1 rounds)

Per token (one propagation round, D_s=256):

- `update_MLP` batched over N: `N · D_s · 4·D_s · 2 = 65 MFLOPs`
- `content_MLP` batched over N: `N · D_s · 2·D_s · 2 = 16 MFLOPs`
- `delta_MLP` batched over N·K: `N · K · (D_s + D_id) · D_s · 2 ≈ 75 MFLOPs`
- `score_MLP` batched over N·K: `N · K · D_s · D_s/2 · 2 ≈ 17 MFLOPs`
- Gather + weighted sum: `N · K · D_s · 2 = 67 MFLOPs (BW-bound)`
- Cross-attn I/O + readout + horizon-emb: ~20 MFLOPs
- **Per token:** ~260 MFLOPs

Plasticity fires every `mod_period=16` tokens at ~5 MFLOPs; amortised
~0.3 MFLOPs/token. Negligible.

At BS = 32, T_seq = 256: `32 · 256 · 260 MFLOPs = 2.1 TFLOPs per training
step (forward)`. Backward ~2×, optimiser ~small → ~6.5 TFLOPs per full
training step. On RTX 4090 at 160 TFLOPs bf16 peak and realistic ~40%
utilization (~65 TFLOPs effective), **~100 ms per training step** →
**~80 K tok/s** at dev scale — better than main-class (68K).

### 7.4 VRAM per batch item

- Column states `s [N, D_s]`: 4096 · 128 · 2 B = 1 MB
- E_bias `[N·K]` fp32: 0.5 MB
- pred_buf `[K_buf, K_h, V]` bf16: 8 · 8 · 32000 · 2 B ≈ 4 MB
- Input-ctx EMA, surprise state: negligible
- Per-round activations saved for TBPTT: roughly `N · D_s · T · 2 B = 8 MB
  per batch item`

At BS = 32, tbptt_block = 32 tokens: activations ~250 MB, plastic state
16 MB. Fits in 24 GB easily.

## 8. Code reuse from gridworld

### 8.1 Drop-in unchanged

- `src/gridworld/readout.py` — multi-horizon factored readout, ring
  buffer, surprise EMA, pred_head.
- `src/gridworld/train_phase1.py` — multi-horizon CE loss, TBPTT
  streaming, phase1_step training loop.
- `src/gridworld/grid.py::torus_radius_nbrs`, `sparse_permutation`,
  `tile_assignment` — topology primitives. Operate on columns now.

### 8.2 Adapt with renames / structural changes

- `src/gridworld/config.py` — new `ColumnGraphConfig` dataclass, same
  pattern. `D_n → D_s`, add L/K/D_id/T/num_planes, remove per-neuron
  plastic-W fields.
- `src/gridworld/neuromod.py` — trunk-heads-gate skeleton stays; feature
  extraction changes (column/tile features instead of neuron features).
- `src/gridworld/plasticity.py` — same fp32-autocast-disabled pattern;
  replace matrix-W update with the `E_bias` scalar update from §4.7.

### 8.3 New modules

- `src/column_graph/column_graph.py` — `ColumnGraphMemory` main module,
  propagation loop, cross-attn I/O.
- `src/column_graph/edges.py` — directional scoring, sparse gather,
  CSR inverse adjacency.
- `src/column_graph/standalone.py` — `StandaloneLM` head wrapper.

### 8.4 Archived

- `src/gridworld/kernels.py` — Triton tick-core kernel. Obsolete; the
  new per-column compute is vanilla batched MLP that compiles into
  proper matmuls. No custom kernel needed.

## 9. Open questions

These are the design decisions we have not yet pinned down. Listed
roughly in order of "must resolve before coding" → "can defer to
empirical tuning."

### 9.1 Score → weight squashing: sigmoid vs softmax — *RESOLVED*

**Resolved: sigmoid for v1.** Softmax would be a competing routing
decision and introduces gradient-stability concerns we don't want to
entangle with the plasticity rule from day 1. Reconsider after training
dynamics are understood.

### 9.2 Per-edge content variation — *RESOLVED*

**Resolved: `m_out + delta_MLP(m_out, id[dest])` per edge.** See §4.3.
`delta_MLP` is a small shared module batched over N·K. Column gets an
expressive knob to tailor messages to receiver identity.

### 9.3 Plasticity cadence — *RESOLVED*

**Resolved: `mod_period = 16` tokens.** Matches current gridworld and
CortexNet convention. Neuromod fires once every 16 tokens; between
firings plastic state is frozen.

### 9.4 Input injection cadence — *RESOLVED*

**Resolved: once per token at t=0 of that token's T-round propagation.**
Token is injected as a pulse into the input plane, then the graph has
T rounds to digest it before the next token arrives.

### 9.5 Teacher-forced vs autoregressive — *RESOLVED*

**Resolved: teacher-forced phase 1, autoregressive phase 2 (GRPO).**
Reuses the two-phase structure from gridworld and prior branches. See
§6. Phase 1 is the default implementation target; phase 2 bolts on once
phase 1 converges.

### 9.6 Backward-plane connectivity — *RESOLVED*

**Resolved: no explicit previous-plane edges.** Rely on Watts-Strogatz
rewiring (p_rewire = 0.30) for all non-local flows including backward.
Shuffle percentage bumped from 5% to 15% specifically to ensure ample
skip paths.

### 9.7 Column identity initialization — *RESOLVED (with caveat)*

**Resolved: random Gaussian init (std=0.02).** **Monitor for
representation collapse** during training — if column identities fail
to differentiate (e.g., cosine similarity between pairs stays near 1.0
throughout training), consider (a) adding a diversity regulariser, (b)
seeding from structured positional encodings, or (c) aligning initial
identities to pretrained token embeddings.

### 9.8 Neuromod feedback into forward compute — *STILL OPEN*

Neuromod currently outputs rates `η, β` that gate *plasticity*. Should
it also gate forward computation — e.g., per-column gain on `m_out` that
modulates the propagated signal?

Biological motivation: real neuromodulators do both (gating plasticity
AND gain-modulating activity).

Current recommendation: **no for v1** — keep forward and plasticity
decoupled. Add a forward-gain head later if the model shows specific
pathology (e.g., runaway activity in some columns that plasticity alone
can't dampen fast enough). Still pending explicit confirmation.

### 9.9 Training signal — auxiliary losses?

Current plan: multi-horizon next-token CE. Options to add:

- Trajectory-smoothness: penalize abrupt changes in column states across
  propagation rounds (encourages signals to settle).
- Co-activation sparsity: L1 penalty on `w_out` to encourage selective
  routing.
- Surprise prediction: explicit loss that the neuromod's `η_global`
  correlates with actual surprise level (meta-learning signal).

Recommendation: multi-horizon CE only for v1. Add auxiliaries only if
training dynamics show a specific pathology that an auxiliary would fix.

### 9.10 E_bias clipping bound

Spec'd as `|E_bias| ≤ E_max = 4`. At `σ(4) ≈ 0.98`, an edge can dominate
or be nearly zeroed. Too low = plasticity has no effect; too high =
runaway.

Recommendation: 4.0 as starting bound. Tune empirically by watching the
distribution of `E_bias` values during training.

### 9.11 Segment boundary semantics

"Segment boundary" = when do we reset column states, E_bias, pred_buf?
Options:

- End of every training segment (default T_seq = 256 tokens).
- At document boundaries within a mixed-document segment.
- Never — let state persist across documents.

Recommendation: reset at segment boundaries AND within-segment document
boundaries if we can detect them (from tokenizer EOS). Matches gridworld
behaviour.

### 9.12 Attachment-mode hook layer

When we eventually build attachment mode: which layer of the pretrained
LM does the memory hook into?

Options: embedding layer, mid-depth (layer L/2), late-mid (layer ~3L/4),
right before LM head.

Recommendation: **defer.** This is a v2 decision, and the right answer
depends on what the memory is intended to contribute (late for
knowledge injection, early for input reformulation).

### 9.13 Empirically-tune-later knobs

Can default to reasonable starting values, revisit based on measurements:

- `T` (rounds per token): default 1, sweep 1-4 (if 1 shows per-token
  settling matters).
- `K` (out-edges per column): default 32, sweep 16-64.
- `p_rewire`: default 0.30, sweep 0.15-0.50.
- `num_tiles`, tile size for neuromod aggregation.
- FFN multipliers, hidden sizes in column MLPs.
- LR, warmup, weight decay.

## 9.14 Post-synaptic-local neuromod scope — *RESOLVED*

**Resolved.** Neuromod is column-local: column B's neuromod reads only
signals observable *at B* and only updates B's *incoming* edges
`E_bias[*, B]`. The symmetric edge `E_bias[B, *]` is updated by column
A's neuromod when A is processed. The two directed edges between A and
B evolve through completely separate neuromod decisions. A global
broadcast `g` (derived from aggregate surprise + tile statistics) is
shared across all columns as context, but the per-column decision uses
only B's own local features plus `g`. See §4.7 for the computational
layout and §4.8 for the two-level neuromod architecture.

## 10. Summary of what's agreed

Bullets below are the commitments of this design. Everything *not* in
this list is either in §9 (open) or out of scope (§3.2).

1. Compound columns with shared internal MLPs, state `s_c [D_s]` and
   identity `id_c [D_id]`.
2. 3D stack of L planes, each a 2D grid; per-column K fixed out-neighbours
   from {intra-plane Moore + next-plane Moore + Watts-Strogatz 15%
   rewire}. Physical layout static after init.
3. Directional message passing: column computes one `m_out` and K
   scalar scores; message per edge = `σ(score + E_bias) · (m_out +
   delta_MLP(m_out, id_dest))`.
4. `E_bias[A, B]` and `E_bias[B, A]` are separate plastic scalars
   updated independently.
5. T-step recurrent dynamics with shared weights across rounds; input
   pulse injection at t=0 per token.
6. Cross-attention injection at input plane; cross-attention readout
   from output plane at t=T; multi-horizon factored readout into
   `[B, K_h, V]` logits.
7. Multi-horizon surprise via ring buffer + per-horizon CE EMA, code
   reused from current `readout.py`.
8. **Post-synaptic-local** Hebbian plasticity on E_bias with Oja-style
   decay. Column B's neuromod updates only B's incoming edges.
9. Two-level neuromod: global trunk broadcasts context `g` + per-column
   heads consume `g` + local observables at each column to produce
   `η[c], β[c]`. Cadence: once every `mod_period = 16` tokens.
10. Dual-use via abstract `ColumnGraphMemory` module with
    `step(inject) → readout`. Standalone v1, attachment v2 via head
    swap.
11. Precision recipe: bf16 autocast, fp32 master weights, fp32 forced
    for plasticity / surprise / neuromod-feature math.
12. Training: teacher-forced TBPTT phase 1 (next-token CE), GRPO
    autoregressive phase 2 on neuromod outputs with Gaussian noise.
13. Starting scale: N=4096, L=4, K=32, D_s=128, D_id=32, T=8,
    p_rewire=0.30.
14. ~6 M trainable params, ~130 K plastic scalars, ~80 K tok/s target
    throughput at dev scale.
15. Code reuse: readout, grid topology primitives, train_phase1,
    train_phase2, neuromod trunk-heads pattern. New: column compute,
    edge math, delta_MLP, per-column neuromod heads, standalone LM
    head. Archive: `kernels.py`.
