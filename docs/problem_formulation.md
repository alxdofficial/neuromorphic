# Problem Formulation: Neural Memory Graph for Language Modeling

> **NOTE (2026-03-28):** This doc was written for v8 (RL/GRPO, 7 layers, port neurons).
> Current code (v9-backprop, branch `v9-backprop`) trains memory end-to-end by backprop.
> N=512 neurons, D_neuron=256, K=32, 4 scan layers split at 2, d_inner=580, 2-pass
> simulation, 110M params (LM=52M, Mem=58M), ~24.8K tok/s. No RL, no ES, no port
> neurons, no GRPO. The open problems in sections 1-3 are still broadly relevant.
> Sections on neuromodulator/GRPO are historical.

## What We Are Building

A language model augmented with a persistent, brain-inspired memory graph. The
system has three components that operate at different levels:

1. **Language Model (Cortical Columns)**: A scan-based (linear recurrence) LM with
   16 cortical columns, 7 layers, D=2048. Processes tokens causally. Trained by
   backprop. This is the "fast" system — standard next-token prediction.

2. **Neural Memory Graph**: A network of 1024 neurons with sparse random connectivity
   (96 presynaptic connections each). At every token, neurons receive messages from
   neighbors, integrate them with internal state, and broadcast new messages. 16 "port
   neurons" interface with the LM's cortical columns. This is the "slow" system —
   persistent memory that evolves with every token but is NOT trained by backprop.

3. **Neuromodulator**: An RL-trained policy network (2-layer MLP, 512 hidden) that
   observes every neuron's state and outputs a gate (controls Hebbian plasticity
   direction) and decay target every 128 tokens. Three-factor learning: eligibility
   traces accumulate what neurons encode/receive, the gate controls whether to
   consolidate or reverse. This is the "meta" system — it learns HOW to configure
   the memory graph.

## Current Architecture in Detail

### The LM (What Works Well)

The LM is a standard scan-based sequence model:
- Embedding → 7 scan layers (h[t] = a*h[t-1] + b[t], parallelizable via FLA kernel)
- Per-column predictive coding modules (16 independent PCMs)
- Memory injection: `H_enriched = H_mid + gate * mem_signals` (mid-scan, before upper layers)
- ~93M trainable parameters
- Throughput: ~161K tok/s without memory (RTX 4090, compiled)

This part is well-understood and works. The scan layers are fully parallel over T=2048
tokens. No issues here.

### The Memory Graph (Core Design Challenge)

1024 neurons, each with:
- `h` [128 dims] — internal state (persistent across tokens and chunks)
- `prev_messages` [128 dims] — last outgoing message
- `primitives` [128 dims] — modulates outgoing messages (neuromod-controlled)
- `decay` — state persistence (neuromod-controlled)
- `key` [128 dims] — routing selectivity (neuromod-controlled)

**Per-token dynamics (sequential, every token):**
```
1. Dendritic tree gather (3 levels, no trainable params):
   Level 0: 8 branches of 12 connections each -> tanh
   Level 1: 2 groups of 4 branches -> tanh
   Level 2: soma averages groups -> received
   received[:, :16] += cc_signals[:, t]      # port neurons also get LM signal
2. h = decay * h + (1-decay) * received      # temporal integration
3. prev_messages = tanh(h * primitives)       # compute outgoing message
```

The LM reads port neurons' messages at every token as `mem_signals`.

### The Neuromodulator (RL Agent)

Every 128 tokens (16 times per 2048-token chunk):
- Observes all 1024 neurons (516 dims each: primitives, key, mean input, mean output, msg_magnitude, decay, trace_prim_norm, trace_key_norm)
- Outputs per-neuron gate (tanh'd to [-1,1]) + decay_target (act_dim=2)
- Shared 2-layer MLP (hidden=512) applied to each neuron independently
- Gate controls Hebbian trace consolidation: primitives += hebbian_lr * gate * normalize(trace_prim), keys similarly

**RL training**: GRPO trajectory scoring (8 trajectories for K=96 neurons, scored across
ALL 4 collected chunks, ranked by CE loss). Non-K neurons get gate=0 (no plasticity) +
current decay_logit as target (no drift). Best trajectory's final state persists. No
value function or critic.

---

## Open Problems

### Problem 1: Throughput — Sequential Memory in a Parallel World

**The core tension**: The memory graph's per-token dynamics are inherently sequential.
At each token, every neuron's state depends on its neighbors' messages from the previous
token, which depend on THEIR neighbors from the token before that, and so on. The
nonlinearity (tanh) and inter-neuron coupling (A @ messages) break all parallelization.

**Current cost**: One `torch.bmm([BS, 1024, 1024] × [BS, 1024, 128])` per token.
At BS=4, that's ~2 GFLOP per token × 2048 tokens = ~4 TFLOP per chunk. The sequential
loop adds Python overhead on top. Total memory path: ~100ms per chunk.

**Result**: ~64K tok/s Phase 1 (memory, no neuromod), ~87K tok/s Phase 2 (memory +
neuromod, frozen LM), ~161K without memory. For a research model this is acceptable,
but scaling to larger N or longer T will make it worse (cost scales as O(T × N × K)).

**Why this can't be trivially parallelized**:
- The recurrence `h[t] = f(h[t-1], A @ messages[t-1])` has a nonlinear dependency on
  the full graph state at the previous timestep
- If we drop the nonlinearity (tanh) and the inter-neuron coupling (use diagonal
  recurrence instead of matrix), we get a standard linear scan — parallelizable via
  associative scan, but neurons can't communicate at every token
- If we drop inter-neuron coupling but keep nonlinearity, each neuron is an independent
  nonlinear RNN — parallelizable across neurons but still sequential over T per neuron

**Possible future directions**:
- Sub-segment message passing: diagonal scan for K tokens (parallel), then one round of
  message passing, repeat. Message passing happens every K tokens instead of every token.
  Tradeoff: K=1 (current, faithful) vs K=32 (fast, coarser communication).
- Custom CUDA kernel: fuse the per-token loop (bmm + element-wise ops) into a single
  kernel to eliminate Python loop overhead and kernel launch latency.
- CUDA graphs: capture the static computation graph and replay it.
- Smaller N: N=256 with a matrix-valued associative scan would allow true per-token
  parallelization, but reduces graph capacity.

### Problem 2: RL Credit Assignment — Which Actions Mattered?

**The setup**: The neuromodulator takes 16 actions per chunk (one every 128 tokens).
Each action outputs a gate (controls Hebbian plasticity direction) and decay_target
per neuron. The gate modulates eligibility trace application; the reward is the CE
loss across all collected chunks.

**What we currently do**:
- Collect RL data across `rl_collect_chunks=4` chunks. Save MG state at start of window.
- Choose K=96 neurons (fixed for all trajectories and all chunks in the RL step)
- GRPO trajectory scoring: sample 8 trajectories, each replaying ALL 4 chunks sequentially.
  Only K neurons get stochastic gate+decay_target, non-K get gate=0 (no plasticity) +
  current decay_logit as target (no drift). Score = mean CE across all 4 chunks.
  Z-score normalize → per-trajectory advantages.
- Best trajectory's final MG state + upper scan carries persist (best-of-N state selection)
- No value function, no critic

**What works**:
- Multi-chunk scoring captures long-range effects (64 segments of compounding memory changes)
- Zero delta for non-K: loss differences are 100% attributable to K neurons' actions
- Best-of-N state selection: free performance boost before neuromod learns anything
- No critic to train — avoids the explained-variance=0 failure mode seen in earlier runs

**Open questions**:

**Q1: Is the reward signal informative enough?**
The reward is negative CE loss on the NEXT segment's tokens. But the neuromod's gate
controls Hebbian plasticity direction (consolidate vs reverse eligibility traces) —
the mapping from "set neuron 500's gate to +0.8" to "segment CE loss decreases by 0.01"
is indirect. However, the 2-dim action space (gate + decay_target) is much simpler than
the old 257-dim action space, and the Hebbian traces provide biologically meaningful
update directions. The neuromod only needs to learn WHEN to consolidate vs explore,
not WHAT direction to update in.

**Q2: Is per-segment reward the right granularity?**
128 tokens per segment. The neuromod adjusts the graph, then the graph processes 128
tokens of text. The CE loss on those 128 tokens is the reward. But:
- The first ~10 tokens after an action barely reflect the structural change (signals
  haven't propagated far through the graph yet)
- The last tokens in the segment fully reflect the changed graph state
- Should we weight later tokens more heavily in the segment reward?

**Q3: Can the neuromod distinguish "my action helped" from "this text was easy"?**
**Addressed by GRPO trajectory scoring.** GRPO samples 8 alternative trajectories with
different stochastic actions on the same text, then ranks by CE loss. The z-score
normalization cancels out text difficulty — all trajectories see the same text, so
the advantage reflects action quality, not text ease. Each trajectory runs per-segment
observe → sample → apply, mirroring real deployment. Cost: 8 extra memory graph +
upper scan runs every 4 chunks.

**Q4: Is the action space right?**
Each action is 2 continuous dims per neuron (gate + decay_target) x 1024 neurons = 2,048
total action dimensions per step. The gate is tanh'd to [-1,1], decay_target is blended
into decay_logit. The Hebbian trace directions are determined locally (from h and
mean_input), not by the neuromod — the neuromod only controls the scalar gate magnitude
and sign. Questions:
- Is hebbian_lr=0.01 the right scale for the gated updates?
- The same MLP is applied to all neurons — is the obs sufficient for the MLP to
  differentiate neuron roles?

**Q5: Reward horizon — should collection go beyond 4 chunks?**
**Addressed.** GRPO now scores trajectories across ALL 4 collected chunks (not just the
last one). Each trajectory replays all 64 segments sequentially, with memory graph and upper
scan carries evolving naturally. The total CE across all 4 chunks determines the trajectory's
score. Remaining concern: the effective horizon is still limited by the collection window
size. A structural change that pays off 5+ chunks later still gets attenuated credit.

### Problem 3: Memory Graph Signal Dynamics — Does It Actually Help?

**The fundamental question**: Does the memory graph provide information to the LM that
the LM's own scan layers can't already capture?

The LM scan layers already have per-layer carries that persist across chunks. They
implement h[t] = a*h[t-1] + b[t] — a linear recurrence that can capture long-range
dependencies. The memory graph adds a SECOND recurrence that's nonlinear and graph-
structured, but communicates with the LM through only 16 port neurons.

**Potential value of the memory graph**:
- Nonlinear dynamics (tanh) can represent patterns that linear scan cannot
- Graph structure gives spatial organization — different neurons can specialize
- The neuromodulator can adapt the graph at inference time (the LM's scan weights are frozen)
- 1024 neurons × 128 dims = 131K state dimensions vs 2048 dims per scan layer

**Potential problems**:
- The 16 port neurons are a severe bottleneck: 2048 dims of LM state compressed to
  16 × 128 = 2048 dims of port messages. The memory graph's internal state (131K dims)
  can only influence the LM through this narrow interface.
- (OUTDATED — v9-backprop uses mem_scale, a learnable per-dim scale for direct
  addition: H_enriched = H_mid + mem_scale * mem_readout. No gate, no MLP.)
- The CC signal enters port neurons as a raw additive input. Scale mismatch: CC signals
  are O(1), neighbor messages are O(0.2). Port neurons are dominated by the LM signal.

### Problem 4: Structural Plasticity — Is It Meaningful?

**Current design**: Every 4 segments (~1024 tokens), co-activation-based plasticity runs.
Connections with negative phi (anti-correlated firing patterns) are pruned. New connections
form toward the highest-phi unconnected neuron (80%) or random (20%). Routing weights
are sigmoid over key-neighbor similarity (each connection independently gated [0, 1]).
Vectorized — no Python loop over neurons.

**Remaining concerns**:
- The co-activation EMA (decay=0.995) takes ~200 segments to converge, so early plasticity
  decisions are based on noisy phi estimates
- Graph topology is shared across batch elements — can't specialize per document
- Random exploration (20% of regrowth) may not be enough to discover useful long-range connections
  — it doesn't know WHICH specific connections are useful

### Problem 5: Document Boundary Semantics — RESOLVED

**Current behavior**: The memory graph is fully persistent — NO resets at document
boundaries. Only the LM scan carries reset at doc boundaries. The memory graph's
h, prev_messages, primitives, key, decay, and co_activation all persist across docs.

**The design intent**: The whole point of the memory graph is long-horizon storage.
Understanding what document boundaries or session boundaries look like is something
the memory graph should *learn* from the data, not have manually erased. The graph
sees document transitions implicitly through abrupt CC signal changes (since the LM
scan carries reset, H_mid changes character at boundaries).

**Why this is better than partial resets**: Any manual reset policy (zero h at doc
boundaries, partially decay key, etc.) bakes in assumptions about what information
should persist. The memory graph should discover these boundaries organically. If a
neuron's state from a previous document is useful, it should be retained. If it's
not, the decay mechanism and new incoming signals will naturally overwrite it.

---

## Summary of What's Working vs What's Uncertain

| Aspect | Status | Confidence |
|--------|--------|------------|
| LM scan layers | Working, well-understood | High |
| Memory graph per-token dynamics | Implemented, correct | High |
| Throughput | ~64K Phase 1, ~87K Phase 2, ~161K no memory | Acceptable |
| Multi-chunk RL rewards | Implemented (4 chunks, 64 segments) | Medium — informative enough? |
| Credit assignment | GRPO (8 traj, K=96, all 4 chunks, zero non-K, best state persists) | Medium-High — trajectory ranking controls for text difficulty |
| Primitives as message modulation | Implemented | Medium — do neurons differentiate? |
| Structural plasticity | Implemented | Low — random rewiring meaningful? |
| Memory actually helping LM | Not yet proven | Unknown — need training run |
| Scale mismatch (CC vs graph signals) | Known issue | Needs investigation |
| Cross-chunk RL credit | GRPO scores all 4 chunks per trajectory | Addressed |

## Next Steps

1. **Long training run** to see if memory helps (lower loss than no-memory baseline)
2. **Monitor neuromod behavior**: do primitives diversify? do routing keys develop structure?
3. **Ablation**: memory graph with random (untrained) neuromod — does RL matter?
4. **Scale investigation**: does the 16-port bottleneck limit memory usefulness?
5. **Speed optimization**: sub-segment message passing, custom kernels
