# GraphWalker

**Branch:** `graph-walker`  
**Status:** live code reference for the current branch  
**Date:** 2026-04-23

This document describes the code that exists now, not the older dense
`column-graph` design and not the earlier relaunch-`H×L` graph-walker draft.

## Thesis

The branch is trying to keep the original memory thesis intact:

- memory should live in the evolving spatial organization of a graph, not in a
  fixed-size hidden tensor alone and not in an external RAG database
- retrieval should happen through sparse trajectories through graph space
- plasticity should remain active after training

The current implementation keeps that thesis, but changes the execution model to
make it cheaper:

- walkers are **persistent**
- each token advances each walker by **one hop**
- dense lexical prediction is moved **off the hot token path**
- exact surprise is computed only on the **plasticity clock**

## Current Code Layout

- [src/graph_walker/config.py](/home/alex/code/neuromorphic/src/graph_walker/config.py)
- [src/graph_walker/topology.py](/home/alex/code/neuromorphic/src/graph_walker/topology.py)
- [src/graph_walker/routing.py](/home/alex/code/neuromorphic/src/graph_walker/routing.py)
- [src/graph_walker/readout.py](/home/alex/code/neuromorphic/src/graph_walker/readout.py)
- [src/graph_walker/graph_walker.py](/home/alex/code/neuromorphic/src/graph_walker/graph_walker.py)
- [src/graph_walker/train_phase1.py](/home/alex/code/neuromorphic/src/graph_walker/train_phase1.py)
- [src/graph_walker/standalone.py](/home/alex/code/neuromorphic/src/graph_walker/standalone.py)
- [tests/test_graph_walker.py](/home/alex/code/neuromorphic/tests/test_graph_walker.py)

## Current Default Shape

From [src/graph_walker/config.py](/home/alex/code/neuromorphic/src/graph_walker/config.py):

- topology:
  - `L = 4` planes
  - `16 x 16` columns per plane
  - `N = 1024` total columns
  - `K = 32` outgoing edges per column
- widths:
  - `D_model = 1024`
  - `D_s = 512`
  - `D_id = 32`
- routing:
  - `n_heads = 4`
  - `n_score_heads = 4`
  - `D_q_in = 64`
  - `D_q_per_head = 64`
- hot MLP:
  - `content_mlp_depth = 4`
  - `ffn_mult_content = 4`
- cold model-space stack:
  - `post_model_depth = 7`
  - `post_model_ffn_mult = 4`
- clocks:
  - `tbptt_block = 16`
  - `mod_period = 128`

Current total parameter count at defaults is about **104.4M**.

Important split:

- token embedding / tied unembedding: about **32.8M**
- hot `content_mlp`: about **8.7M**
- post-model stack: about **58.8M**

That split is intentional. The branch is explicitly moving parameter budget out
of the per-token per-hop recurrent core and into cheaper model-space modules.

## Core Runtime Design

### 1. Persistent graph state

[GraphWalkerMemory](/home/alex/code/neuromorphic/src/graph_walker/graph_walker.py)
owns persistent state:

- `s [B, N, D_s]`: per-column state
- `walker_pos [B, H]`: current persistent walker positions
- `prev_motor [B, D_s]`: previous graph-state readout
- `E_bias_flat [N*K]`: plastic edge biases
- `co_visit_flat [N*K]`: traversed-edge counts accumulated over the current
  plasticity window
- `surprise_ema [B, K_h]`
- delayed-surprise buffers:
  - `surprise_motor_window [B, mod_period, D_s]`
  - `surprise_token_window [B, mod_period]`
  - `surprise_tail_motor [B, K_h-1, D_s]`

`begin_segment()` resets segment-local working state but keeps `E_bias_flat`
unless `reset_plastic_memory()` is called.

### 2. Hot token path

The hot compiled path is [step_core()](/home/alex/code/neuromorphic/src/graph_walker/graph_walker.py),
which routes into `_step_core_impl`.

Per token it does:

1. embed token in `D_model`
2. project token to `D_s` with `token_to_state`
3. choose **input anchor columns** on plane 0 via input-plane softmax
4. inject token content at those anchor columns
5. advance each persistent walker **one hop**
6. aggregate only the sparse touched destinations
7. update only those rows of `s`
8. read out graph-state `motor_state [B, D_s]` from the new walker endpoints

This is the key current architectural change.

The older graph-walker draft relaunched `H × L_walk` hops from scratch every
token. The current code does **one hop per token** and persists the walkers
through time.

### 3. Input anchoring vs persistent motion

The two mechanisms are separate on purpose:

- each token still chooses `start_cols` on the input plane
- walkers themselves continue from `walker_pos`

So each token can inject fresh token-specific signal into the graph while the
walker trajectories continue through graph space over time.

### 4. Sparse state update

The state update only aggregates messages for:

- input anchor columns
- one-hop walker destinations

The code deduplicates visited `(batch, column)` rows, sums into a compact
`incoming_sparse`, updates only those rows, then writes them back into `s`.

Important limitation:

- the implementation still rewrites the full flattened state tensor via
  `index_copy`, so the semantics are sparse but the storage update is not yet
  fully sparse

That is still one of the main remaining performance bottlenecks.

## Routing

Routing is in [src/graph_walker/routing.py](/home/alex/code/neuromorphic/src/graph_walker/routing.py).

Current behavior:

- Gumbel top-1 softmax in training
- straight-through estimator for hard routing
- optional epsilon exploration
- explored rows have gradient zeroed so random exploration does not backprop as
  if it were model-selected

Load-balance regularization is live and gradient-carrying.

## Readout

The dense lexical stack is in [src/graph_walker/readout.py](/home/alex/code/neuromorphic/src/graph_walker/readout.py):

- `state_to_model: D_s -> D_model`
- `PostModelStack`
- `PredictionHead`
- tied unembedding
- horizon factorization via `horizon_emb`

The readout now supports `[..., D_model] -> [..., K_h, V]`, so it can run
either:

- per token through the public `step()` API
- batched over a TBPTT block during training

## Training

Phase 1 is in [src/graph_walker/train_phase1.py](/home/alex/code/neuromorphic/src/graph_walker/train_phase1.py).

Current protocol:

1. run `step_core()` token by token
2. buffer `motor_state` for the current TBPTT block
3. at block flush, run one batched lexical forward pass with
   `readout_from_state_block`
4. compute exact multi-horizon CE from those block logits
5. backprop
6. `detach_state()` at TBPTT boundaries

This is a mechanical optimization only. It does **not** change the training
objective. It only moves the large model-space prediction stack off the hot
token path.

## Surprise And Plasticity

The branch no longer computes exact surprise every token.

Current behavior:

- each token records detached `motor_state` and token ids into a window buffer
- exact multi-horizon surprise is computed only when the plasticity window
  closes, via `_finalize_surprise_window()`
- then `_plasticity_step()` uses the resulting `surprise_ema`

So there are now three practical clocks:

- token clock: sparse graph dynamics
- block clock: batched lexical prediction for CE
- plasticity clock: exact surprise + plastic write

Plasticity is still scalar-eta surprise-gated Hebbian on traversed-edge counts:

- `co_visit_flat` counts traversed edges over the window
- `eta_global = plast_eta * sigmoid(mean(surprise_ema) - plast_surprise_bias)`
- `E_bias += eta_global * (co_visit_norm - plast_decay * E_bias)`
- clamp to `[-E_bias_max, E_bias_max]`

This is simpler than the older richer neuromodulator sketches. The code does
not currently have a separate neuromod MLP.

## Public APIs

### `step_core(token_id)`

Hot graph-only path. Returns:

- `motor_state`
- `visit_freq_step`
- `load_balance_loss`

No logits are produced here.

### `step(token_id)`

Compatibility / debugging path.

Runs:

- `step_core()`
- immediate single-token readout
- delayed-surprise buffering
- possible plasticity-window finalization

This keeps the single-step tests and debugging workflow simple, but it is not
the path used by phase-1 training.

## Current Tests

[tests/test_graph_walker.py](/home/alex/code/neuromorphic/tests/test_graph_walker.py)
currently passes **16 tests**.

Covered items include:

- shape sanity
- non-visited column preservation
- persistent walker position changes
- gradient flow through routing
- load-balance gradient
- prev-motor chaining
- epsilon exploration gradient neutrality
- K_buf validation
- plasticity firing
- detach / reset semantics
- CUDA bf16 smoke

## Current Measured Throughput

Measured on RTX 4090 with [scripts/bench_graph_walker.py](/home/alex/code/neuromorphic/scripts/bench_graph_walker.py):

- config: `N=1024`, `D_model=1024`, `D_s=512`, `K=32`, `H=4`, depth 4
- total params: **104.4M**

At the measured sweet spot (`BS=48 T=128 tbptt=32`):

- compiled training path: about **548 ms/step**, **11,200 tok/s**, **17.0 GB**

At the old dev config (`BS=16 T=128 tbptt=16`):

- compiled training path: about **389 ms/step**, **5,258 tok/s**, **6.88 GB**

Historical baseline (pre–CE vectorization + fused sparse update):

- compiled, `BS=16`: about **1592 ms/step**, **1286 tok/s** — step time dominated
  by `SelectBackward` nodes from a `T_block × K_h` Python loop in the CE flush.

What changed:

- Fused Triton forward + backward for the sparse LIF state update
  (`src/graph_walker/triton_sparse_update.py`). Replaced
  `torch.unique + index_add + gather + LIF + index_copy` with an O(U)-touched-
  row kernel path.
- Vectorized multi-horizon CE in `phase1_step.flush`. One `F.cross_entropy`
  over `[B * T_block * K_h, V]` replaces ~380 Python-loop iterations per
  block flush, each of which was generating 2–3 `SelectBackward` autograd
  nodes with full-shape zeros allocations.
- Vectorized `_finalize_surprise_window`: per-horizon slice + closed-form
  EMA instead of an inner double loop.
- Per-segment scratch tensors (`batch_idx`, `k_range`, `ones_bh`) cached in
  `begin_segment` rather than rebuilt every token.
- Removed per-flush `.item()` CPU↔GPU syncs.

## What This Means

The branch is now much more faithful to the intended thesis than the older
relaunch-`H×L` version:

- trajectories are truly temporal
- the lexical stack is off the fast clock during training
- surprise is computed on the plasticity clock

And the step is no longer dominated by autograd-graph overhead from Python
loops in training code. Remaining performance frontier is structural:

1. the hot path is still token-sequential (truly recurrent; can't scan-
   parallelize like Mamba)
2. backward still traverses ~50 autograd nodes per token through
   `step_core`'s mixed routing + MLP + sparse update graph
3. a custom step_core Triton forward + manual backward would likely cut
   another 2–3× off training time, but at significant engineering cost

## Fairness Of Parameter Allocation

The branch is intentionally using the parameter budget in a way that is cheaper
to execute:

- keep the hot walker core reasonably sized (`D_s=512`, depth `4`)
- move most extra parameters into cold model-space modules

This is not treated as cheating on this branch. The comparison target is total
trainable parameters plus wall-clock performance, not “every parameter must be
paid on the per-token recurrent clock.”

## Open Work

Most important remaining systems tasks:

1. eliminate the full-state rewrite
2. improve compile/fusion of the hot core
3. decide whether `H=4` or `H=8` is the best accuracy/speed point
4. consider an optional richer slow-clock neuromodulator if scalar surprise
   gating proves too weak

Most important current research question:

- can the persistent-walker design recover enough memory/reasoning capacity to
  justify the still-high systems overhead, or does the branch need a more
  aggressive sparse-state representation as well
