# GraphWalker

**Branch:** `main` (graph-walker promoted)
**Status:** live code reference for the standalone walker.
**Last updated:** 2026-04-30

This document describes the **standalone walker** — the original training
target where the walker is itself the language model (`StandaloneLM` +
`phase1_step` in `src/graph_walker/`). This is the path that has its
own LM head, computes its own multi-horizon CE, and fires plasticity
inside the training flush.

For the **integration** path (frozen Llama + walker as a side-channel
memory module), see [docs/pretrained_graph_walker.md](pretrained_graph_walker.md).
The integration walker is **vocab-agnostic**: no LM head, no aux CE,
plasticity is driven externally via `walker.update_plasticity(per_token_ce)`
called by the trainer after `loss.backward()`. The standalone path
described below is unchanged by that integration refactor.

## Thesis

The branch keeps the original memory thesis intact:

- memory lives in the evolving spatial organization of a graph, not in a
  fixed-size hidden tensor and not in an external retrieval database
- retrieval happens through sparse trajectories through graph space
- plasticity stays active after training

Execution-model choices that follow from the thesis:

- H **persistent** walkers, each advancing by **one hop per token**
- each walker has its own persistent D_s-dim state (`walker_state`) in
  addition to its position on the graph
- the dense lexical readout lives off the per-token hot path — it runs
  only at TBPTT-block boundaries
- anchoring (input-plane Gumbel softmax) fires **once per plasticity
  window**, not once per token
- exact multi-horizon surprise is accumulated into an EMA from the CE
  we are already computing for training — no separate readout pass

## Current Code Layout

- [src/graph_walker/config.py](/home/alex/code/neuromorphic/src/graph_walker/config.py) — hyperparams + post-init validation
- [src/graph_walker/topology.py](/home/alex/code/neuromorphic/src/graph_walker/topology.py) — fixed Watts-Strogatz graph builder
- [src/graph_walker/routing.py](/home/alex/code/neuromorphic/src/graph_walker/routing.py) — Gumbel top-1 softmax with ε-exploration and STE
- [src/graph_walker/readout.py](/home/alex/code/neuromorphic/src/graph_walker/readout.py) — `PostModelStack`, `PredictionHead`, `MultiHorizonReadout` (with factorized CE)
- [src/graph_walker/graph_walker.py](/home/alex/code/neuromorphic/src/graph_walker/graph_walker.py) — `GraphWalkerMemory`, `ColumnCompute`, `_step_core_pure`
- [src/graph_walker/neuromod.py](/home/alex/code/neuromorphic/src/graph_walker/neuromod.py) — `NeuromodGraphTransformer` (target-predicting head) + subgraph helpers
- [src/graph_walker/train_phase1.py](/home/alex/code/neuromorphic/src/graph_walker/train_phase1.py) — TBPTT flush, factorized CE, surprise streaming, plasticity trigger
- [src/graph_walker/standalone.py](/home/alex/code/neuromorphic/src/graph_walker/standalone.py) — thin wrapper: token embedding + `GraphWalkerMemory` + tied unembed
- [src/graph_walker/triton_sparse_update.py](/home/alex/code/neuromorphic/src/graph_walker/triton_sparse_update.py) — `SparseLIFUpdate` autograd Function (Triton forward + backward) used for the sparse column-state update
- [tests/test_graph_walker.py](/home/alex/code/neuromorphic/tests/test_graph_walker.py), [tests/test_neuromod.py](/home/alex/code/neuromorphic/tests/test_neuromod.py), [tests/test_triton_sparse_update.py](/home/alex/code/neuromorphic/tests/test_triton_sparse_update.py)

## Current Default Shape

From [src/graph_walker/config.py](/home/alex/code/neuromorphic/src/graph_walker/config.py)
(verified 2026-05-04 — earlier versions of this section had stale numbers):

- topology:
  - `L = 4` planes
  - `16 × 16` columns per plane
  - `N = 1024` total columns
  - `K = 16` outgoing edges per column
- widths:
  - `D_model = 1024` (external lexical width)
  - `D_s = 256` (graph state + walker state)
  - `D_id = 512` (column identity)
- walker head:
  - `n_heads (H) = 4` persistent walkers
  - `n_score_heads = 4` multi-head edge scoring
  - `D_q_in = 64`, `D_q_per_head = 64`
- hot MLP:
  - `content_mlp_depth = 4` residual FFN blocks
  - `D_hid_content = 1024` (decoupled from `ffn_mult_content * D_s`; the
    older formula has been replaced by an explicit override since the
    scale-up)
- cold model-space stack:
  - `post_model_depth = 2`, `post_model_ffn_mult = 4`
- clocks:
  - `mod_period = 128` — plasticity window length (standalone walker)
  - `tbptt_block = 128` — gradient detach cadence (**must equal** `mod_period`)
  - `segment_T = 1024` — full segment length (standalone). The integration
    enforces `segment_T == mod_period == tbptt_block` (single-knob clock
    invariant under external-surprise plasticity)
- neuromod (on by default):
  - `neuromod_D_mod = 512`, `neuromod_n_layers = 6`, `neuromod_n_heads = 8`
  - `neuromod_edge_hidden = 384`, `neuromod_eta = 1.0`, `E_bias_max = 4.0`
- plasticity:
  - `plasticity_mode = "hebbian_plus_neuromod"` (standalone default).
    `PretrainedGWConfig` factories override to `"neuromod_only"` for the
    integration — see § Plasticity below.

Total parameter count at defaults: about **76M** (this number is
config-sensitive; for live count run
`sum(p.numel() for p in walker.parameters()) / 1e6`).

For up-to-date throughput numbers see `docs/bench_results.md`.

## State

`GraphWalkerMemory` owns the following per-segment tensors (allocated by
`begin_segment`, detached each `tbptt_block` by `detach_state`):

- `s [B, N, D_s]` — column state (LIF-integrated)
- `walker_pos [B, H]` int64 — persistent walker positions
- `walker_state [B, H, D_s]` — walker's private running state (EMA of
  per-step `m_out`)
- `prev_motor [B, D_s]` — last step's motor output, fed into the anchor
  query at window boundaries
- `surprise_ema [B, K_h]`, `surprise_prev [B, K_h]` — per-batch
  per-horizon surprise EMA (no motor-state buffer — surprise is
  streamed from training CE)

Persistent-across-segments (plastic) state:

- `E_bias_flat [N·K]` — plastic edge biases, shared by Hebbian and
  neuromod paths
- `co_visit_flat [N·K]` — accumulated traversed-edge counts in the
  current plasticity window
- `visit_count [N]` — per-column visit tally in the current window
- `_prev_snapshot_ids`, `_prev_snapshot_feats` — detached snapshot of
  the previous window's touched columns (consumed by neuromod at the
  next window start)
- `_active_delta_nm [N·K]` — grad-carrying neuromod delta, live for the
  whole current window, detached-and-committed into `E_bias_flat` at
  window close

Topology tensors (built once at `__init__`, `persistent=False` buffers):

- `out_nbrs [N, K]` int64 — outgoing neighbour indices
- `edge_src`, `edge_dst` — flat edge endpoint lists
- `plane_ids [N]` int64 — plane index per column
- `input_positions [N_per_plane]` int64 — global indices of input-plane cols

## Per-Token Flow

`_step_core_pure` (graph_walker.py) runs once per token. Its inputs are
current state + the token id. A boolean `is_new_window = (window_len == 0)`
splits the flow into two branches. `is_new_window` is True at segment
start and on the first token of each new plasticity window (every
`mod_period` tokens).

### Common prefix (every step)

1. Embed token: `h_input = token_to_state(embed(token))` ∈ ℝ^{D_s}.

### Window-boundary step only

2. **Anchor softmax.** Build `q_in = h_input + prev_motor_proj(prev_motor)`.
   Score against input-plane column identities: `scores_in = q_in·input_k_proj(col_id[input_positions])`.
   Gumbel top-1 STE picks H `anchor_cols`.
3. **Teleport.** `walker_pos ← anchor_cols`.
4. **Anchor injection.** `v_inject = input_v_proj(embed(token))` gated by
   the STE one-hot of the anchor pick. Stacked into the same sparse LIF
   deposit as the walker's own write (step 6).

Within a window (tokens 1 … mod_period-1) this whole block is skipped;
walkers stay at their persistent positions.

### Common suffix (every step)

5. **Walker message.** For each of B·H walkers at its current column c:
   read `s_cur_old = s[c]` via gather. Build the **steering input**
   `cat(s_cur_old, col_id[c], walker_state, token_per_walker)` of dim
   `3·D_s + D_id`. Feed through `content_mlp` → `m_out ∈ ℝ^{D_s}`.
6. **Sparse LIF deposit.** Walker writes `m_out` at **its current
   column** (not destination). On window-start steps, the anchor
   injection is stacked in the same kernel call. The LIF blend is
   `s_new[c] = α(c)·s_old[c] + (1-α(c))·tanh(Σ messages_to_c)` where
   `α(c) = σ(decay_proj(col_id[c]))` per-column. Implemented by
   `SparseLIFUpdate` (Triton forward + backward).
7. **Re-read post-update.** `s_cur_new = s_new[c]`. The walker's own
   contribution is now in the state it reads for routing.
8. **Routing query.** Build the **post-update steering input**
   `cat(s_cur_new, col_id[c], walker_state, token_per_walker)` → `q_proj` →
   per-score-head query `[B·H, n_score_heads, D_q]`.
9. **Hop scores.** Score against `k_proj(col_id[nbrs[c]])` for each of
   the K neighbours. Add `active_E_bias[edge_flat]` (plastic
   persistent bias + grad-carrying neuromod delta; see §Plasticity).
   Gumbel top-1 STE → `next_col`.
10. **Endpoint readout.** `end_state = s_cur_new + Σ_k ste[k] ·
    nbr_id_to_s(col_id[nbrs[c, k]])`. First term forwards equals
    `s[c]_new` and carries gradient to `content_mlp`, LIF α,
    `walker_state`, token embedding. Second term is the **routing
    gradient bridge**: in forward it equals `nbr_id_to_s(col_id[next_col])`
    (ste is one-hot); in backward it differentiates through ste →
    soft_probs → scores → `q_proj` / `k_proj` / `E_bias` / neuromod.
11. **Motor.** Cross-attention over H end-states (learned `motor_query`)
    → `motor_state ∈ ℝ^{D_s}`.
12. **Walker state update.** `walker_state_new = σ(α_h)·walker_state +
    (1-σ(α_h))·m_out`. `α_h` is a learnable per-walker scalar, init so
    σ(α_h) ≈ 0.9 (slow decay).
13. **Walker moves.** `walker_pos ← next_col`.

## Routing

[src/graph_walker/routing.py](/home/alex/code/neuromorphic/src/graph_walker/routing.py). Gumbel top-1 softmax with
ε-exploration. Straight-through: forward picks the argmax, backward
differentiates through the soft distribution. ε-exploration samples a
uniform neighbour with probability ε; gradients on exploration rows are
**detached** so random choices can't train the router to prefer random
edges.

Tau and ε are tensors (not floats) so dynamo doesn't recompile when
they anneal. Schedule: linear from `gumbel_tau_start=2.0 →
gumbel_tau_end=0.5` over 10k steps; ε from 0.05 → 0.01 similarly.

Load-balance aux loss (Switch-Transformer style): `λ_balance · N ·
Σ_c P(c)·f(c)` where P is the softmax routing mass per column and f is
the empirical visit frequency. Gradient flows through the P term into
`q_proj` / `k_proj` / `E_bias`.

## Readout

[src/graph_walker/readout.py](/home/alex/code/neuromorphic/src/graph_walker/readout.py). `MultiHorizonReadout` produces
`[..., K_h, V]` predictions from motor vectors via the factorisation

    logits[...,k,v] = (motor @ W^T)[...,v] + (horizon_emb[k] @ W^T)[v]

where W is the tied token embedding. There are **two entry points**:

- **`forward(motor, unembedding, horizon_logits=None)`** — materialises
  the full broadcast `[..., K_h, V]`. Used by the single-token `step()`
  path.
- **`cross_entropy_factorized(motor, unembedding, targets, valid,
  horizon_logits=None)`** — avoids the broadcast. For each horizon k
  in Python, compute `logits_k = motor_logits + horizon_logits[k]`
  ([..., V]) and `F.cross_entropy(logits_k, targets[..., k],
  reduction="none")`. Memory: one `[B, T, V]` tensor at a time instead
  of the full K_h-multiplied broadcast.

The training flush uses `GraphWalkerMemory.readout_ce_block`, which is
the memory-side wrapper around `cross_entropy_factorized`. At BS=200,
T=48, K_h=8, V=32000, bf16 this saves ~4 GB relative to the naive
broadcast path.

## Plasticity

`GraphWalkerConfig.plasticity_mode` selects between two distinct rules:

- **`"hebbian_plus_neuromod"`** (default for the standalone walker):
  both Path 1 (Hebbian) and Path 2 (neuromod) contribute additively
  to `E_bias_flat`. The neuromod sees only per-column features;
  Hebbian-flavored stats (co_visit, E_bias_old) are NOT fed to it.
- **`"neuromod_only"`** (used by `PretrainedGWConfig` factories for
  the integration): drops Path 1 entirely. Hebbian stats become
  per-edge inputs to neuromod's edge MLP and per-head attention bias
  ("Option C"), so the neuromod has the final say on every plastic
  update. Surprise also enters as a per-touched-column feature.

In standalone training, plasticity fires every `mod_period` tokens via
`_maybe_finalize_surprise_and_plasticity` (called from the trainer's
flush after CE is computed). In the integration, plasticity is fired
**externally** once per training step via
`memory.update_plasticity(per_token_ce)` after backward + opt.step
— see `docs/pretrained_graph_walker.md` for the full external-surprise
flow. Either way, the actual update is in `_plasticity_step` below.

### Path 1 — surprise-gated Hebbian (only in `hebbian_plus_neuromod` mode)

- `surprise_ema [B, K_h]` was **streamed in** from the training flush via
  `accumulate_block_ce` (see §Training below). No separate readout re-run.
- `η_global = plast_eta · σ(surprise_ema.mean() - plast_surprise_bias)`
- `δ_hebb = η_global · (co_visit_norm - plast_decay · E_bias_flat)`
- `E_bias_flat ← E_bias_flat + δ_hebb`
- Clamped to `[-E_bias_max, +E_bias_max]`

### Path 2 — graph-transformer neuromod (on by default)

[src/graph_walker/neuromod.py](/home/alex/code/neuromorphic/src/graph_walker/neuromod.py).

**Timing.** At the **start** of each plasticity window, neuromod reads
a detached snapshot of the previous window's per-touched-column
features and emits a grad-carrying `Δ_nm [N·K]`. Routing uses
`active_E_bias = E_bias_flat.detach() + neuromod_eta · Δ_nm` for the
whole next window. At window close, `Δ_nm.detach()` is folded into
`E_bias_flat`.

**Gradient story.** Loss from the window backprops through every
routing score → `active_E_bias` → `Δ_nm` → neuromod parameters. This
is "the neuromod is trained by the outcome of the decision it just
made" — classic policy-style signal but fully differentiable, with the
autograd graph spanning exactly one plasticity window.

**Architecture.** Per-touched-column features are `[U, D_s + D_id + 1]`
(mean state across batch, column identity, log visit count, all
detached). Pipeline:

1. `feature_proj: D_feat → D_mod`
2. `n_layers` graph-attention blocks — multi-head attention with an
   additive `[U, U]` adjacency bias (1.0 on existing topology edges, 0
   otherwise) plus a pre-norm FFN. The transformer sees which touched
   columns are already connected and can reason about nearby columns.
3. **Target head** (per touched edge `(i, j)`): a small MLP on
   `cat(x_i, x_j)` produces a scalar target bounded by tanh:
   `target[i→j] = E_bias_max · tanh(edge_mlp(cat(x_i, x_j)))`.
4. **EMA blend.** The caller converts target → delta via a learnable
   scalar blend rate `γ = σ(blend_logit)`:
   `Δ_nm[edge] = γ · (target[edge] - E_bias_flat.detach()[edge])`.
   Adding this to `E_bias_flat` realises `(1-γ)·E_bias + γ·target` —
   standard EMA blend toward the target.

**Day-0 safety.** `edge_mlp`'s output layer is zero-initialised so
`target = 0` at start; `blend_logit = -5` gives `γ ≈ 0.007`. Enabling
the neuromod on a fresh init is effectively a no-op until training
opens both knobs.

**Scope of write.** Only edges whose source AND destination are in the
touched set this window get a non-zero target. Unvisited parts of the
graph are unchanged — "a synapse you don't use doesn't move".

## Training

[src/graph_walker/train_phase1.py](/home/alex/code/neuromorphic/src/graph_walker/train_phase1.py).
Teacher-forced TBPTT over a `segment_T` sequence with `tbptt_block`
equal to `mod_period` (enforced by config assertion).

Per segment:

1. `begin_segment(B, device)` — resets s, walker_pos, walker_state,
   surprise_ema, co_visit, visit_count. `E_bias_flat` survives.
2. For each token: `step_core(token_id)` produces a differentiable
   `motor_state` and accumulates visit / co-visit counters. Motor state
   is buffered for the block-level readout.
3. When the TBPTT block boundary is hit (= plasticity window close, by
   config alignment), call `flush()`:
   - Stack the block's motor states: `motor_state_bt [B, T_block, D_s]`.
   - Build `targets [B, T_block, K_h]` and `valid_tk [T_block, K_h]`
     (valid iff `block_start_t + i + k < T_seq`).
   - `ce_masked [B, T_block, K_h]` = factorized CE (see §Readout).
   - `accumulate_block_ce(ce_masked.detach(), valid_tk.detach())` —
     streams per-token per-horizon CE into `surprise_ema` via
     closed-form EMA per horizon.
   - `_maybe_finalize_surprise_and_plasticity()` — fires Hebbian +
     neuromod commit using the fresh `surprise_ema`.
   - `loss = Σ_k w_k · per_horizon_mean[k] + Σ block_balance_sum`;
     `loss.backward()`.
   - `detach_state()` — cuts the autograd graph for the next block.

This keeps the dense lexical stack OFF the per-token clock. The hot path
only holds small per-walker tensors during forward; the
`[B·T_block, D_model]` matmuls and per-horizon CE happen once per block.

## Public APIs

### `step_core(token_id)`

Hot graph-only path. Returns `WalkerCoreReadout(motor_state,
visit_freq_step, load_balance_loss)`. No logits. Used by training.

### `step(token_id)`

Compatibility / debugging path. Runs `step_core()`, materialises logits
via `readout_from_state_block`, and triggers
`_maybe_finalize_surprise_and_plasticity` at the window boundary. Not
used by `phase1_step`; surprise on this path is populated from whatever
was last streamed by training (so plasticity on the `step()` path uses
a potentially stale `surprise_ema` — fine for interactive smoke tests).

### `readout_ce_block(motor_state, targets, valid)`

Factorised CE entry point used by `phase1_step.flush`. Returns `[B, T,
K_h]` float32 per-position per-horizon cross-entropy (masked by
`valid`).

## Tests

- [tests/test_graph_walker.py](/home/alex/code/neuromorphic/tests/test_graph_walker.py) — 16 tests covering shapes,
  gradient flow, load-balance, prev-motor chaining, ε-exploration
  gradient neutrality, plasticity firing, detach / reset semantics,
  CUDA bf16 smoke, K_buf validation, non-visited column preservation.
- [tests/test_neuromod.py](/home/alex/code/neuromorphic/tests/test_neuromod.py) — 8 tests covering subgraph
  helpers (enumerate_touched_edges, build_adjacency_bias), zero-init
  safety, non-zero targets after perturbation, tanh bound, first-segment
  identity, phase-1 integration with gradient flow to neuromod params,
  delta-snapshot roundtrip.
- [tests/test_triton_sparse_update.py](/home/alex/code/neuromorphic/tests/test_triton_sparse_update.py) — 9 tests covering
  the `SparseLIFUpdate` Triton kernel numerical parity and edge cases.

**Full suite: 33 passed.** 24 pass CPU-only (graph_walker + neuromod);
9 Triton tests are GPU-gated.

## Current Measured Throughput

For up-to-date numbers, see **`docs/bench_results.md`**. The table that
used to live here described a `mod_period = tbptt_block = 48` config
(BS=48-200, peak 461 ms/step, 20.8k tok/s @ BS=200, 21.1 GB ceiling)
that no longer matches production defaults — the production walker
config is now `mod_period = tbptt_block = 128`, and the integration
config locks `T == mod_period == tbptt_block` under external-surprise
plasticity. The headline numbers worth tracking now are:

- **Standalone walker** (per `scripts/bench_walker_at_integration_config.py`)
- **Integration step** (per `docs/bench_results.md`): 8.8k tok/s @
  BS=16 / T=256 / 14.6 GB peak, 4090

Both supersede the old table. Re-run the bench scripts whenever
the config changes; don't trust this section for current numbers.

## Remaining Perf Frontier

1. **`[B*H]-row hot matmuls vs Mamba-class [B*T]-row scan-parallel** —
   the model is fully recurrent on the token clock. We can't scan-
   parallelise time the way Mamba / transformers do; each walker hop
   depends on the previous hop's deposited state. Expect per-tick
   overhead to dominate until a custom fused Triton step-core kernel
   exists.
2. **Active-row overlay for column state** — the sparse-update
   kernel still clones the full `[B·N, D_s]` buffer per step. At high
   batch this is ~10% throughput overhead on top of factorized CE.
   Tried once; reverted because the implementation tangled with
   autograd's view+inplace and version-check rules; needs a dedicated
   redesign (detached base + sparse differentiable overlay).
3. **Fused tied-unembedding CE Triton kernel** — current factorized
   CE is a Python loop of K_h iterations each running
   `F.cross_entropy(logits_k [B·T, V], ...)`. A fused kernel that
   never materialises even the per-horizon `[B·T, V]` logits would
   save another ~2 GB at high batch, at the cost of engineering effort.

## What's Next

Integration target (not on this branch): mount `GraphWalkerMemory` as
a memory module inside a frozen Llama (1B or 3B). See
[docs/pretrained_lm_memory.md](/home/alex/code/neuromorphic/docs/pretrained_lm_memory.md) on `main` for the interface
(`MemInjectLayer`, `forward_segment`, phase-1 Gumbel-STE + phase-2
GRPO). The current single-token `step(token_id)` API needs to be
wrapped as `forward_segment(h_mem[BS,T,d_mem], input_ids, lm, ...)`,
the standalone-LM readout tower dropped, and a discrete-policy path
added for phase-2 (candidate: REINFORCE over walker Gumbel-top-1 hops,
accumulating log_π per fire).
