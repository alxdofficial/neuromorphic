# GraphWalker — Trajectory-Routed Plastic Concept Graph

**Branch:** `graph-walker` (from `column-graph`), supersedes the dense column-graph design.
**Status:** design locked, implementation in progress.
**Date:** 2026-04-23.

## 1. Thesis

> **Retrieval is literal trajectory through a plastic concept graph.**
>
> Previous attempts ("column graph" with dense per-column updates) diffused
> every token's signal across all columns every tick. That ran N×K edges of
> routing work per token even when 99% of the graph was irrelevant to the
> current signal.
>
> **GraphWalker commits to a sharper thesis: per token, a bounded number of
> signals take paths through the graph. Each path is a sequence of concept
> columns, and that sequence IS the model's representation of the token.**
>
> At each hop, the current column softmax-routes over its K=32 outgoing
> edges and picks the most likely next column. The trajectory runs L hops.
> H parallel heads give H trajectories per token. Only columns on the
> union of trajectories (H·L columns out of N) have their state updated.
> Plastic edges co-evolve via Hebbian on which edges were actually
> traversed.
>
> The graph is sparse at two levels: (a) fixed K-neighborhood topology
> (Watts-Strogatz shortcuts give log diameter), (b) trajectory sparsity
> (only ~H·L columns touched per token). This drops per-token compute
> from ~10 GFLOPs (dense column-graph) to ~10 MFLOPs — transformer class.

## 2. Architectural positioning

**This is Mixture-of-Experts constrained to a graph.** Each column is an
expert. Routing is top-k softmax, restricted to the K out-edges of the
current column. Load balancing and exploration keep the graph from
collapsing to a few popular paths.

Differences from standard MoE:
- **Sparse connectivity** (not fully-connected experts). Each column can
  only route to K=32 neighbors, determined by grid + Watts-Strogatz.
- **Plastic edges.** `E_bias[A, B]` adds to the routing logit and
  evolves via local Hebbian rules shaped by trajectory co-visitation.
- **Multi-hop trajectories.** Unlike standard MoE's single expert-pick
  per token, we take L=4-6 hops forming a path.
- **Persistent per-column state.** Each column maintains a long-term
  state vector updated only when visited. This is the "memory" part.

## 3. TL;DR

> - **Columns** (same as prior column-graph): compound units with shared
>   MLP weights, per-column identity `id[c]`, per-column state `s[c]`,
>   per-column learned decay α[c].
>
> - **Topology**: L planes, each a 2D grid of columns. K=32 out-edges per
>   column via {Moore radius intra-plane, Moore next-plane, Watts-Strogatz
>   p=0.30 shuffled}. Static after init.
>
> - **H heads × L hops per token.** Per head: pick starting column via
>   input cross-attention softmax over input-plane columns. Walk L hops
>   via Gumbel top-1 softmax routing at each step.
>
> - **Non-visited columns: state frozen.** A column's state only changes
>   when a trajectory visits it. This gives per-column long-term memory.
>
> - **Readout**: cross-attention over all `H·L` visited column states →
>   motor vector → factored multi-horizon logits.
>
> - **Plasticity**: co-visitation Hebbian on traversed edges. Fires every
>   `mod_period = 128` tokens. Neuromod gates the rate.
>
> - **Compute**: ~10-20 MFLOPs per token at N=1024, D_s=512, H=4, L=4.
>   Transformer-class. Target throughput: 20-50K tok/s at BS=16.

## 4. Scope and non-scope

### 4.1 In scope (v1)

- Graph-walker per-token forward with Gumbel top-1 softmax routing.
- Straight-through estimator for training.
- Hard-argmax for inference.
- Input cross-attn softmax → per-head starting column.
- Output cross-attn over visited trajectory → factored readout.
- Co-visitation Hebbian plasticity.
- Load-balance aux loss + ε-greedy exploration.
- Multi-horizon surprise + ring buffer (unchanged from column-graph).
- TBPTT training with streaming CE.

### 4.2 Not yet (v2+)

- **Top-k routing with k>1.** Start top-1; add top-2 if single-head routing
  collapses.
- **Adaptive trajectory length** (stop when reaching output plane). Start
  with fixed L; add adaptive if useful.
- **Occasional dense broadcast rounds** ("wake up dead columns"). Start
  without; add if dead-column pathology emerges.
- **Phase 2 (GRPO).** Deferred to post-v1.
- **Attachment to pretrained LM.** Deferred to v2.

### 4.3 Explicit thesis concessions we accept

- **Plasticity becomes coarser-grained.** Each mod_period sees H·L·window
  traversal events, not N·K activation pairs. We're counting trajectories,
  not monitoring full graph activity.
- **Non-visited columns don't learn from current token.** Their state stays
  frozen, their `id[c]` gets gradient only via routing softmax (if they're
  neighbors of visited columns and were considered as destinations).
- **Hard routing introduces variance.** Gumbel-soft mitigates in training.

## 5. Architecture detail

### 5.1 Columns (unchanged from column-graph)

Per column `c`:
- **State** `s[c] ∈ ℝ^D_s`. Persistent; updated only on trajectory visits.
- **Identity** `id[c] ∈ ℝ^D_id`. Learned parameter, stable across training.
- **Decay** `α[c] = σ(decay_proj(id[c]))`. Per-column learned time constant.

Shapes at starting scale (N=1024, D_s=512, D_id=32):
- `s`: [B, N, D_s] = 16 × 1024 × 512 × 2B = 16 MB per batch item
- `id`: [N, D_id] = 33K trained params
- `E_bias`: [N·K] fp32 plastic state, 32K scalars (128 KB)

### 5.2 Topology (unchanged)

- 4 planes × 16×16 columns = 1024 total.
- Per column: 32 out-edges = 16 intra-plane Moore + 16 next-plane Moore.
- Watts-Strogatz rewiring with `p_rewire=0.30`.
- **NO inverse adjacency needed.** Graph-walker doesn't do scatter-gather.
- `out_nbrs[N, K]` is all we store (plus edge_src/edge_dst for plasticity).

### 5.3 Input plane: start-column softmax per head

Per token, for each of H=4 heads:

```
# 1. Token embedding
h_input = token_emb(t)                                  # [B, D_s]

# 2. Per-head query projection
query[h] = input_q_proj[h](h_input)                     # [B, D_q_in]

# 3. Score input-plane columns (keys are projections of their id)
input_keys = input_k_proj(id[input_plane_cols])         # [N_in, D_q_in]
scores[h] = (query[h] @ input_keys.T) / sqrt(D_q_in)   # [B, N_in]
         + input_E_bias[h]                             # plastic per-head bias

# 4. Gumbel top-1 softmax for starting column
logits[h] = (scores[h] + gumbel_noise) / τ
start_col[h] = argmax(logits[h])                       # [B]

# 5. Inject token content at starting column
v = input_v_proj(h_input)                              # [B, D_s]
s_update[start_col[h]] = α * s_prev + (1-α) * tanh(v)
```

- `D_q_in = 64` (query/key dim for input scoring)
- `H = 4` heads (independent trajectories)
- `N_in = 256` (input-plane columns = 16×16 of plane 0)
- Per token cost: H × N_in = 1024 dot products. ~0.3 MFLOPs. Trivial.
- `input_E_bias[h]` is **plastic** per-head per-column bias, Hebbian-updated
  based on which starts led to good predictions.

### 5.4 Trajectory walk: L hops per head

Per head `h`, starting from `start_col[h]`:

```
cur = start_col[h]
visited[h, 0] = cur

for hop t in 1..L-1:
    # 1. Emit message at current column
    m_out = content_mlp(concat(norm(s[cur]), id[cur]))     # [B, D_s]

    # 2. Score K=32 out-edges
    q_hop = q_proj(concat(s[cur], id[cur]))                # [B, H_score·D_q]
    # Reshape to [B, H_score, D_q]
    # Per-edge key = k_proj(id[out_nbrs[cur, k]]) for each k
    nbr_ids = id[out_nbrs[cur]]                            # [K, D_id]
    k_edges = k_proj(nbr_ids)                              # [K, H_score·D_q]
    scores = bilinear_multihead_dot(q_hop, k_edges)        # [B, K]
           + E_bias[flat_edge_idx(cur, :K)]                # plastic bias

    # 3. Gumbel top-1 softmax (straight-through in forward)
    logits = (scores + gumbel_noise) / τ
    next_col_soft = softmax(logits)                        # [B, K]
    next_col_hard = one_hot(argmax(logits))                # [B, K]
    # Straight-through: forward uses hard, backward uses soft
    next_col_ste = (next_col_hard - next_col_soft).detach() + next_col_soft

    # 4. Pick next column
    next_col = out_nbrs[cur, argmax(logits)]               # [B]
    # Record trajectory
    visited[h, t] = next_col

    # 5. Integrate into next column's state (aggregated at token-end)
    # Record contribution: (next_col, m_out, weight=next_col_ste)
    messages[h, t] = (next_col, m_out, next_col_ste)

    cur = next_col
```

Key details:
- **`L = 4` hops.** Enough to traverse input → hidden → output via grid + shortcuts.
- **`H_score = 4` heads**, `D_q = 64` for bilinear edge scoring.
- **Read-only on `s` during the walk.** All state updates happen at
  token-end via aggregation of messages. This decouples heads for parallelism.
- **Gumbel temperature** `τ` starts at 2.0, anneals linearly to 0.5 over
  the first 10K training steps, then holds. At inference: set `τ=0` (hard argmax).
- **ε-greedy exploration**: with probability `ε = 0.05 → 0.01`, replace
  softmax sample with uniform random neighbor. Anneals over 10K steps.
- **Straight-through estimator** for the hard argmax: backward pretends
  routing was soft. Standard MoE trick, keeps gradients flowing.

### 5.5 Aggregate updates at token end

After all H heads walked L hops, we have:
- `visited[H, L]` column indices
- `messages[h, t] = (dest, m_out, weight)` for h=1..H, t=1..L-1

Aggregate:

```
# Per destination column, sum up all incoming messages.
incoming = zeros(B, N, D_s)
for h in range(H):
    for t in range(1, L):
        dest = visited[h, t]
        incoming[dest] += weight[h, t] * m_out[h, t-1]

# LIF integrate per destination
for each visited column c (union over heads):
    s[c] = α[c] * s[c] + (1 - α[c]) * tanh(incoming[c])
# Non-visited columns: s unchanged
```

Implementation note: the `incoming` tensor is sparse (only H·L columns
have nonzero entries, ≤16 out of N=1024). Use `index_add` to accumulate
with atomics (tiny, no bottleneck).

### 5.6 Output: cross-attention over trajectory

After the walk, collect all visited column states into a sequence:

```
traj_states = gather(s, visited)                         # [B, H·L, D_s]

# Single-query cross-attention
motor_query = learnable_motor_vector.unsqueeze(0)        # [1, D_s]
k = out_k_proj(traj_states)                              # [B, H·L, D_s]
v = out_v_proj(traj_states)
motor = attention(motor_query, k, v)                     # [B, D_s]

motor = pred_head(motor)                                 # residual head
logits = factored_readout(motor, horizon_emb, token_emb) # [B, K_h, V]
```

### 5.7 Multi-horizon surprise (unchanged)

Ring buffer of past logits; compare to current token; maintain per-horizon
surprise EMA. Reused from column-graph with no changes.

### 5.8 Plasticity: co-visitation Hebbian

At each plasticity event (every `mod_period = 128` tokens), we have
accumulated per-edge traversal counts from the intervening window:

```
# During the window, for each head and hop:
# If trajectory went A→B in the window, increment traversal[A, B].
# Weight by the magnitude of message and state at time of visit.
co_visit[A, B] += ||m_out[A]|| * ||s_at_visit[B]||

# At mod_period boundary:
normalized_co_visit = co_visit / window_size

# Neuromod gates rate (same as before)
η_global, η[c], β[c] = neuromod(surprise_ema, Δsurprise, col_features)

# Hebbian with Oja decay (post-synaptic local)
ΔE_bias[A, B] = η_global * η[B] *
                (normalized_co_visit[A, B] - β[B] * E_bias[A, B])

E_bias += Δ  ; clip to [-E_max, E_max]

# Reset window counter
co_visit.zero_()
```

Why co-visitation counts (not co-activation):
- **Signal is direct**: edge was traversed → it was useful → strengthen.
- **Sparse naturally**: only H·L events per token × window_size per mod
  step. At W=128, H=L=4: ~2K events per plasticity update. Plenty of stats.
- **No atomics or scatter**: accumulate per visited edge during trajectory.

### 5.9 Neuromod (slightly adapted features)

Neuromod global trunk: same input (surprise_ema + Δsurprise).
Per-column head: same observables. Small difference: `w_in_proxy` replaced
by "visit frequency" — how many times this column was visited in recent
window — giving the head a signal of "this column is currently active."

```
col_features[c] = concat(
    g,                                 # broadcast global context
    id[c],
    visit_freq_recent[c],              # replaces w_out_proxy
    avg_pre_activity_at_c[c],          # avg ||m_out|| of visits to c
    avg_post_activity_at_c[c],         # avg ||incoming|| at c
)
```

### 5.10 Load balancing

**Auxiliary loss** added to per-step loss:

```
# Aggregate visit frequency per column over the training step
visit_freq = count_visits_per_column(visited) / (B * T * H * L)

# KL from uniform
kl = sum(visit_freq * log(visit_freq * N))     # KL(visit | uniform)
loss_balance = λ_balance * kl                   # λ_balance = 0.01
```

This penalizes concentrated visit distributions (most visits to a few
columns). Standard MoE trick.

**ε-greedy exploration** in routing:
- With probability ε, sample uniformly from K neighbors instead of softmax.
- `ε = 0.05` initially, anneal to `0.01` over first 10K steps.

### 5.11 Precision

- Column states, MLP weights, cross-attn heads: bf16 autocast + fp32
  master weights.
- Plasticity state `E_bias`: fp32.
- Plasticity math + neuromod features: fp32 under `autocast(enabled=False)`.
- Gumbel noise: fp32 for numerical stability of the softmax.

## 6. Compute and memory

### 6.1 Per-token FLOPs at N=1024, D_s=512, H=4, L=4

| Op | FLOPs/token |
|---|---:|
| `input_q_proj × H` + input-plane scoring | ~0.3 M |
| Token injection into H start columns | ~0.5 M |
| `content_mlp` at visited columns (H·L = 16 calls) | 16 × 2 M = 32 M |
| `q_proj` at visited columns (H·L = 16 calls) | 16 × 1.5 M = 24 M |
| Edge scoring (H·L hops, K=32 edges each) | 16 × 0.01 M = 0.2 M |
| Gumbel softmax sampling | negligible |
| Message aggregation + LIF update at visited cols | ~0.5 M |
| Cross-attn readout over H·L=16 visited states | ~0.5 M |
| Factored multi-horizon logits + unembed | ~8 M |
| **Total per token** | **~66 MFLOPs** |

At BS=16: per-step FLOPs = 16 × 128 × 66M = **135 GFLOPs per forward step**.
Backward ~2×: **~270 GFLOPs per step**. On RTX 4090 at 40% of 160 TFLOPs
peak → **~4 ms/step → ~500K tok/s theoretical ceiling**.

Realistic (with Python overhead, autograd, compile limits): **30-60K tok/s**.
Same class as Mamba at this scale.

### 6.2 Parameter count

| Component | Params |
|---|---:|
| token_emb (tied) | V·D_s = 32000 × 512 = 16.4 M |
| content_mlp | (D_s + D_id) × 2D_s + 2D_s × D_s = 1.08 M |
| q_proj | similar = 1.08 M |
| k_proj | D_id × 2HDq + 2HDq × HDq = 0.56 M |
| col_id | N × D_id = 33 K |
| decay_proj | D_id × 1 = 33 |
| input_q_proj (H heads) | H × D_s × D_q_in = 0.13 M |
| input_k_proj | D_id × D_q_in = 2 K |
| input_v_proj | D_s × D_s = 0.26 M |
| output cross-attn | 2 × D_s² + D_s = 0.52 M |
| pred_head + horizon_emb | D_s² + K_h·D_s = 0.27 M |
| neuromod | ~0.1 M |
| **Total trained** | **~20.5 M params** |

Plastic state:
- `E_bias`: N·K = 32K scalars
- `input_E_bias`: H·N_in = 1K scalars

### 6.3 VRAM per batch item at BS=16, T=128, tbptt=16

- Column state: 16 × 1024 × 512 × 2B = 16 MB  
- TBPTT activations (per tick saved for backward): small, mostly visited-column MLP hiddens
- pred_buf: 16 × 8 × 8 × 32000 × 2B = 64 MB
- Total estimated: **4-6 GB** at dev scale. Plenty of headroom for BS scaling.

## 7. Training

### 7.1 Phase 1 — teacher-forced multi-horizon CE

Same as column-graph:
- Segment = T_seq tokens, streamed one at a time.
- At each token: run H trajectories × L hops, readout, compute loss.
- TBPTT detach every `tbptt_block = 16` tokens.
- Plasticity + aux load-balance loss applied within the compiled loop.
- `lr = 1e-4`, AdamW fused, grad clip 1.0.

### 7.2 Loss

```
loss = multi_horizon_ce_loss(logits, tokens)       # primary CE
     + λ_balance * kl_from_uniform(visit_freq)     # load balance
```

`λ_balance = 0.01`. Monitor: if visit distribution stays uniform → model
isn't routing meaningfully; lower λ. If distribution collapses → raise λ.

### 7.3 Gumbel temperature schedule

- `τ = 2.0` for first 2000 steps (warmup, soft routing, gradient-rich).
- Linear anneal `τ: 2.0 → 0.5` over steps 2000-10000.
- `τ = 0.5` after step 10000.
- At inference: hard argmax (`τ → 0`).

### 7.4 ε exploration schedule

- `ε = 0.05` for first 2000 steps.
- Linear anneal `ε: 0.05 → 0.01` over steps 2000-10000.
- `ε = 0.01` after step 10000.
- At inference: `ε = 0`.

### 7.5 Phase 2 (deferred)

GRPO on neuromod output distributions, same as prior design. Not
implementing in v1.

## 8. Code reuse vs new from column-graph

### Keep (minor adaptation)
- `config.py`: add graph-walker-specific fields. Keep N, D_s, K, L, etc.
- `topology.py::build_topology`: unchanged. **Drop inverse adjacency
  (`in_src`, `in_edge_flat`, `in_mask`, `K_in_max`) — no longer needed.**
- `readout.py`: `MultiHorizonReadout`, `multi_horizon_surprise`, ring
  buffer. Unchanged.
- `neuromod.py`: Neuromod trunk + heads. Minor feature adaptation.
- `train_phase1.py::phase1_step` structure: unchanged. Streaming CE.
- `standalone.py`: shell unchanged.

### New
- `src/graph_walker/graph_walker.py`: `GraphWalkerMemory` module with
  `_trajectory_walk`. Replaces `ColumnGraphMemory`.
- `src/graph_walker/routing.py`: Gumbel softmax, straight-through
  estimator, ε-exploration helpers.
- `src/graph_walker/plasticity.py`: co-visitation Hebbian (replaces
  activation-based).

### Drop
- `column_graph/kernels.py`: Triton weighted-gather. **Obsolete —
  graph-walker has no scatter-gather.**
- `column_graph/column_graph.py::_propagate_pure`: replaced by
  `_trajectory_walk`.
- `column_graph/column_graph.py` dense I/O cross-attn: replaced by
  softmax-select input + trajectory-readout output.

## 9. Open questions / decisions locked in

### Locked in v1
- **Top-1 routing** per hop. No top-k yet.
- **Fixed L=4 hops.** No adaptive length.
- **H=4 heads.** Fewer than MoE norms but appropriate for our graph
  (K=32 means fewer routing choices than dense MoE).
- **Loops allowed.** If trajectory cycles through same column, it just
  integrates the signal multiple times.
- **Non-visited columns: state frozen.** No decay.
- **Gumbel straight-through.** Train with stochastic softmax, infer
  with argmax.
- **Plastic only on traversed edges.** Load balance via aux KL loss.
- **D_s = 512, N = 1024.** Matches best-bang-for-buck config from
  column-graph bench.

### Open (revisit during training)

- **τ schedule**: 2.0→0.5 over 10K steps. May need tuning.
- **λ_balance**: 0.01. May need tuning to prevent collapse.
- **ε_explore**: 0.05→0.01. May need wider range.
- **Dead column detection**: at what visit-frequency threshold do we
  declare a column "dead"? 0.1× uniform? Trigger what intervention?
- **Multi-head coordination**: do heads share `q_proj`/`content_mlp`
  weights (as spec'd) or have per-head weights (adds params but more
  specialization)?
- **Occasional dense broadcast**: whether to fire one full-graph update
  every 1000 tokens. Decide after seeing training dynamics.

## 10. Summary of commitments

1. Compound columns with shared MLP weights (content_mlp, q_proj, k_proj),
   per-column id, state, decay. Unchanged from column-graph.
2. Static topology: 4 planes × 16×16 = 1024 columns, K=32 out-edges,
   p_rewire=0.30. Unchanged.
3. Per-token forward: H=4 heads, each runs L=4 hops.
4. Start column per head: softmax over N_in=256 input-plane columns.
5. Per-hop routing: Gumbel top-1 softmax over K=32 out-edges with
   straight-through estimator + ε-exploration.
6. State update: only on visited columns, LIF on aggregated messages.
   Non-visited columns' state frozen.
7. Readout: cross-attention over H·L=16 visited states → motor → factored
   multi-horizon logits.
8. Plasticity: co-visitation Hebbian on E_bias, fires every
   mod_period=128. Neuromod gates.
9. Load balance: aux KL loss + ε-greedy exploration.
10. Precision: bf16 autocast + fp32 master; plasticity/surprise/neuromod
    forced fp32.
11. Target: 20-50K tok/s at dev scale. Same class as Mamba at comparable
    parameter count.
