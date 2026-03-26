# Architecture v8: Neural Memory Graph + Cortical Columns

> **Status**: Implemented. Training entry point: `python -m src.v8.train`
> **Code**: `src/v8/`
> **Branch**: `main`

## Design Philosophy

1. **The cortical column is the universal compute unit.** Our CCs are the language
   model — scan layers + PCM. Trained by backprop.

2. **Memory is activation flowing through weighted connections.** A persistent
   network of neurons outside autograd. Signals propagate hop-by-hop at every token.

3. **Plasticity is local and neuromodulated.** Three-factor learning: Hebbian
   eligibility traces accumulate what neurons encode/receive, the neuromodulator
   (RL) gates whether to consolidate or reverse those traces, and controls decay.
   Structural plasticity (prune/regrow) is autonomous, driven by co-activation
   statistics.

4. **Per-token neuron dynamics.** Each neuron receives presynaptic messages,
   integrates with internal state, broadcasts a message — at every token.

5. **Split-scan with mid-injection.** Lower scan produces representations, memory
   processes them per-token, memory output is injected mid-scan. Upper scan layers
   see memory-enriched input and learn to use it.

---

## Architecture Overview

```
+---------------------------------------------------------------+
|                    CORTICAL COLUMNS (V8LM)                    |
|                                                               |
|  Lower scan: layers 0-3 (D=2048, d_inner=1024)   [parallel]  |
|  PCM: per-CC surprise (side input to upper scan)     [at split] |
|      |                                                        |
|      v  CC signals (H_mid sliced per CC, D_cc=128)            |
|      |                                                        |
|  +-- MEMORY GRAPH (per-token, sequential) -----------------+  |
|  |  1024 neurons, 96 presynaptic connections each          |  |
|  |  dendritic tree -> integrate -> message (every token)   |  |
|  |  Port neuron messages -> mem_signals                    |  |
|  +--------------------------------------------------------+  |
|      |                                                        |
|      v  H_enriched = H_mid + gate * mem_signals               |
|      |                                                        |
|  Upper scan: layers 4-6                            [parallel]  |
|  Output head: proj_down -> LayerNorm -> lm_head               |
|  Trained by backprop. ~93M params.                            |
+-----------------+---------------------------------------------+
                  | neuron observations (516 dims per neuron)
                  v
+---------------------------------------------------------------+
|                    NEUROMODULATOR                              |
|  GRPO trajectory scoring (8 trajectories, K=96 neurons)       |
|  Actor: 2-layer MLP (516->512->512 -> action heads)          |
|  Scores ALL 4 collected chunks per trajectory (no value fn)   |
|  Only K neurons get actions, non-K get gate=0 + no drift      |
|  Best trajectory's state persists after scoring               |
|  16 actions per chunk (every 128 tokens):                     |
|    gate[1] (tanh'd [-1,1]) + decay_target[1]                 |
|  Three-factor learning: Hebbian traces gated by neuromod      |
|  Trained by policy gradient.                                  |
+---------------------------------------------------------------+
```

**Total trained params: ~103M** (scan + embed + PCM + neuromod)

---

## Component Details

### Cortical Columns (V8LM) — `src/v8/lm.py`

**Split-scan architecture:**

1. **Lower scan** (`forward_scan_lower`): Embed -> pos_embed -> layers 0-3 -> PCM
   - PCM at the split point: computes surprise vector
   - H_mid passes through unchanged to memory graph (no gain modulation)
   - Surprise is passed as a side input to the first upper scan layer
   - Returns H_mid [BS, T, D] with autograd intact

2. **Memory injection** (`inject_memory`): H_enriched = H_mid + gate * mem_signals
   - `mem_gate[c]`: learnable scalar per CC, starts at sigmoid(0)=0.5
   - mem_signals from port neuron messages (detached — no grad through memory)

3. **Upper scan** (`forward_scan_upper`): layers 4-6
   - Sees memory-enriched representations
   - Learns to use memory information for prediction

4. **Output** (`forward_output`): proj_down -> LayerNorm -> lm_head
   - No memory injection here — it happens mid-scan

**CC<->memory interface — no projections:**
- D_mem = D_cc = 128 by design
- CC->memory: H_mid sliced per CC (128 dims) added to port neuron's received input
- Memory->CC: port neuron message `tanh(h * primitives)` gated by `mem_gate[c]`

### PCM — `src/v8/pcm.py`

Batched predictive coding (BatchedPCM) processing all C=16 columns in parallel
via `torch.bmm`. RMSNorm on inputs. Both encoding and prediction condition on
scan hidden state H AND input token x:
- `input_norm = RMSNorm(2*D_cc)` applied to `cat(H_t, x_t)` before projections
- `z_t = W_enc(input_norm(cat(H_t, x_t)))` — current position's full-context encoding
- `z_hat_t = W_pcm(input_norm(cat(H_t, x_t)))` — prediction of next position's encoding
- `surprise = z_hat[t-1] - z[t]` (prediction error vector, 128 dims)
- No gain modulation — H_mid passes through unchanged to memory graph
- Surprise is delivered as a side input to the first upper scan layer
  (additive to `proj_in` output via zero-init `proj_side`)

### Neural Memory Graph — `src/v8/memory_graph.py`

1024 neurons with 96 presynaptic connections each. Each neuron has:
- `h [128]`: internal state — persistent across tokens/chunks
- `prev_messages [128]`: last outgoing message
- `primitives [128]`: modulates outgoing messages (neuromod-controlled)
- `key [128]`: routing selectivity (neuromod-controlled)
- `decay_logit [1]`: state persistence (neuromod-controlled)

**Per-token neuron dynamics (sequential loop, every token):**

```
1. RECEIVE (dendritic tree):
   Level 0: 8 branches of 12 connections each -> tanh per branch  [BS, N, 8, D]
   Level 1: 2 groups of 4 branches -> tanh per group              [BS, N, 2, D]
   Level 2: soma averages groups -> received                       [BS, N, D]
   Port neurons: received[:, :C] += cc_signals[:, t]
2. INTEGRATE: h = decay * h + (1-decay) * received                [BS, N, D]
3. MESSAGE: prev_messages = tanh(h * primitives)                   [BS, N, D]
4. OUTPUT: mem_signals[:, t] = prev_messages[:, :C]  (port neurons)
```

The dendritic tree replaces the flat weighted sum with a 3-level nonlinear
gather (no trainable parameters). Each level applies tanh, giving neurons
richer nonlinear integration of presynaptic inputs.

**Key-based sigmoid routing:**
- Each neuron has a `key` vector (128 dims)
- At the start of each segment, compute: `sim = key . neighbor_messages` (raw dot product)
- Routing weights: `sigmoid(sim)` per connection — each independently gated [0, 1]
- Unlike softmax, strong connections don't suppress weak ones (no normalization)
- Fixes signal dilution: softmax spread port signals across 96 connections (241x gap)
- More biologically faithful: biological synapses have independent efficacies
- These scalar routing weights are used for all tokens in the segment
- Triton path: separate routing kernel computes weights inline (no [BS,N,K,D] temp tensor)
- Python path: `_compute_routing_weights()` materializes [BS,N,K,D] for reference

**Binary firing + percentile threshold:**
- `activation = message.norm()` per neuron per token
- `fired = activation > 75th percentile(activation)` within the segment
- No per-neuron EMA tracking needed — percentile computed per segment
- `msg_magnitude` tracked as EMA of mean message norm for neuromod observation

**Co-activation structural plasticity** (every 4 segments, vectorized):
- Phi coefficient (binary Pearson) computed from firing traces via bmm
- `co_activation_ema [N, N]`: slow EMA of phi matrix
- Prune: existing connections where phi < 0 (anti-correlated)
- Grow: 80% toward highest-phi unconnected neuron, 20% random
- Regrown connections are topology-only (no persistent edge weights)
- No magic number thresholds — phi < 0 is a natural boundary

**Document boundaries**: Memory graph is fully persistent — no resets at doc
boundaries. The graph learns to handle document transitions through abrupt
CC signal changes (since the LM scan carries DO reset). Only LM scan carries
reset at doc boundaries.

### Neuromodulator — `src/v8/neuromodulator.py`

Shared policy network across all 1024 neurons. GRPO scoring (no critic/value function).
Outputs a gate (controls Hebbian plasticity direction) and a decay target per neuron.

**Observation** per neuron (516 dims = D_mem*4 + 4):
- primitive[128] + key[128] + mean_input[128] + mean_output[128]
- msg_magnitude[1] + decay[1] + trace_prim_norm[1] + trace_key_norm[1]

**Action** per neuron (2 dims):
- gate[1]: tanh'd to [-1,1]. Controls Hebbian trace application direction.
  gate > 0: consolidate. gate < 0: reverse (explore). gate ~ 0: maintain.
- decay_target[1]: blended into decay_logit as `0.9 * decay_logit + 0.1 * target`

**Architecture**:
- Backbone: 2 layers of 512 hidden, Tanh activations. Zero-init heads.
- No critic/value function. Advantage via GRPO trajectory scoring (z-score normalized).

### Three-Factor Learning (Hebbian + RL gating)

**Eligibility traces** (accumulated via EMA, decay=0.95 per segment):
- `trace_prim = 0.95 * trace_prim + 0.05 * h` ("shift primitives toward what I encode")
- `trace_key = 0.95 * trace_key + 0.05 * mean_input` ("shift key toward what I receive")

**Gated plasticity** (applied after each segment):
- `primitives += hebbian_lr * gate * normalize(trace_prim)`, then RMS-normalize
- `key += hebbian_lr * gate * normalize(trace_key)`, then L2-normalize
- `decay_logit = 0.9 * decay_logit + 0.1 * decay_target`
- Normalization is SAFE because neuromod only controls 1-dim gate (not 128-dim direction)

**Segment loop order:**
1. forward_segment (128 tokens of dendritic tree neuron dynamics)
2. compute_eligibility_traces (EMA update of traces from h and mean_input)
3. neuromod observes + outputs gate & decay_target
4. apply_gated_plasticity (Hebbian update gated by neuromod)
5. structural_plasticity (every 4 segments, if needed)

### RL Training

GRPO trajectory scoring across all collected chunks, entropy bonus, LR schedule:
1. Lower scan (layers 0-3) + PCM surprise (parallel over T=2048)
2. 16 segments per chunk: forward_segment → eligibility traces → neuromod → gated plasticity
3. RL data collected across `rl_collect_chunks=4` chunks. MG state saved at start of window.
4. GRPO scoring (every 4 chunks): Choose K=96 neurons (fixed for all trajectories/chunks).
   Sample 8 trajectories. For each: replay ALL 4 chunks sequentially — only K neurons get
   stochastic actions (gate + decay_target), non-K get gate=0 (no plasticity) + current
   decay_logit as target (no drift). Score = mean CE across all 4 chunks.
   Z-score normalize → per-trajectory advantages.
5. Best trajectory's final MG state + upper scan carries persist as the real state
   (best-of-N selection — free performance boost from selection pressure alone).
6. Replay: for each above-average trajectory, forward K-neuron obs/actions through
   policy (with grad) → log_prob weighted by advantage → policy gradient.
7. Entropy bonus (rl_entropy_coef=0.01) always applied, prevents exploration collapse.
8. Neuromod LR: warmup + cosine decay, floor derived from LR_MIN/LR ratio.

---

## Training — `src/v8/train.py`

```bash
python -u -m src.v8.train --bs 12 --steps 61035 --no-compile              # Phase 2: with neuromod
python -u -m src.v8.train --bs 12 --steps 61035 --no-compile --no-neuromod # Phase 1: LM only
python -u -m src.v8.train --bs 12 --steps 61035 --no-memory                # baseline (no memory)
```

**Per step:**
1. Lower scan (layers 0-3) + PCM surprise at split point (parallel)
2. 16 segments of 128 tokens each:
   a. forward_segment (dendritic tree neuron dynamics)
   b. compute_eligibility_traces (EMA update)
   c. neuromod observes → outputs gate & decay_target
   d. apply_gated_plasticity (Hebbian update gated by neuromod)
   e. structural_plasticity (every 4 segments, vectorized)
3. Inject memory into H_mid, upper scan (layers 4-6, parallel)
4. LM loss backward (gradients flow through upper + lower scan + mem_gate)
5. RL data collected across rl_collect_chunks=4 chunks (64 segments total)
6. On RL update step: GRPO scoring (8 trajectories across all 4 chunks, K=96 neurons
   with gate=0 for non-K), replay for neuromod gradients + entropy bonus.
   Best trajectory's state persists.
7. Detach scan carries. Memory graph state persists.

---

## Tier A Config

| Component | Value |
|-----------|-------|
| D | 2048 |
| D_embed | 768 |
| C (cortical columns) | 16 |
| D_cc = D_mem | 128 |
| L_total | 7 |
| scan_split_at | 4 (lower: 0-3, upper: 4-6) |
| d_inner | 1024 |
| N_mem_neurons | 1024 |
| K_connections | 96 |
| action_every | 128 tokens (16 segments per chunk) |
| neuromod_hidden | 512 |
| neuromod_layers | 2 |
| obs_dim per neuron | 516 (D_mem*4 + 4) |
| act_dim per neuron | 2 (gate + decay_target) |
| hebbian_lr | 0.01 |
| trace_decay | 0.95 |
| dendrite_branch_size | 12 (Tier A), 3 (Tier Tiny) |
| RL | GRPO (8 trajectories, K=96 neurons, all 4 chunks scored, best state persists) |
| T | 2048 |
| Throughput | ~64K tok/s Phase 1, ~87K tok/s Phase 2 (RTX 4090) |
| **Total trained params** | **~103M** |

---

## Design Decisions

1. **Split-scan with mid-injection.** Memory is injected between lower (0-3) and
   upper (4-6) scan layers. The upper layers see memory-enriched representations
   and learn to use them. The LM cannot ignore memory.

2. **Per-token neuron dynamics.** Sequential loop with bmm per token. Inherently
   non-parallelizable (nonlinear recurrence + inter-neuron coupling). Signals
   propagate hop-by-hop through the graph.

3. **Key-based sigmoid routing.** Each neuron has a learned key vector.
   Routing weights are `sigmoid(key . neighbor_messages)`
   (raw dot product), computed once per segment. Each connection is
   independently gated [0, 1] — strong connections don't suppress weak ones.
   The neuromod controls routing selectivity by assigning new key values.
   Content-based gating over presynaptic neighbors.

4. **Primitives modulate outgoing messages.** `message = tanh(h * primitives)`.
   Primitives are updated by gated Hebbian plasticity: eligibility traces
   accumulate what neurons encode, and the neuromod gate controls whether
   to consolidate or reverse. Decay controls persistence. Keys control
   routing. Clean separation of concerns.

5. **Co-activation-based structural plasticity.** Phi coefficient matrix tracks
   temporal co-firing patterns. Anti-correlated connections pruned (phi < 0).
   New connections formed toward highest-phi unconnected neurons (80%) or
   random (20%). All thresholds relative to network state.

6. **PCM at the split point with surprise as side input.** PCM computes a
   surprise vector at the split point. Rather than multiplicatively modulating
   H_mid, surprise is passed as a side input to the first upper scan layer
   (additive to `proj_in` output via zero-init `proj_side`). H_mid flows
   unchanged to the memory graph. RMSNorm on PCM inputs stabilizes training.

7. **GRPO trajectory scoring across all chunks.** Collect 4 chunks of RL data,
   then score 8 trajectories across ALL 4 chunks (64 segments total). Only K=96
   neurons get stochastic gate+decay_target actions; non-K get gate=0 (no
   plasticity) + current decay_logit as target (no drift). Rank by total CE,
   z-score normalize → advantages. Best trajectory's final MG state + upper
   carries persist. No value function or critic needed. Entropy bonus prevents
   exploration collapse. Neuromod LR decays alongside LM LR.

---

## File Structure

```
src/v8/
+-- config.py              # V8Config dataclass + tier presets
+-- pcm.py                 # BatchedPCM (bmm across all C columns, RMSNorm + surprise side input)
+-- memory_graph.py        # MemoryGraph (dendritic tree, eligibility traces, gated plasticity, vectorized init, routing kernel dispatch)
+-- triton_kernels.py      # memory_graph_routing_kernel (inline routing) + memory_graph_step_kernel (dendritic tree step)
+-- lm.py                  # V8LM (split-scan + BatchedPCM + memory interface)
+-- neuromodulator.py      # Policy network (2-dim action: gate + decay, 2x512 backbone)
+-- model.py               # V8Model (segment loop ordering, _apply_neuromod_action with tanh gate)
+-- trainer.py             # V8Trainer (joint LM + RL training loop)
+-- train.py               # Training entry point with CLI
+-- diagnostics.py         # Per-step metrics + periodic snapshots

scripts/
+-- benchmark_v8.py        # Throughput + VRAM benchmarking
+-- profile_v8.py          # torch.profiler breakdown
+-- plot_training.py       # Training curves + memory health plots
+-- analyze_memory.py      # Deep per-token memory analysis (STALE — references removed APIs)
+-- find_max_bs.py         # Max batch size finder + leak check

tests/v8/
+-- test_integration.py    # Full forward + backward + RL
+-- test_memory_graph.py   # Neuron dynamics, plasticity, reference tests
+-- test_triton_kernel.py  # Triton vs Python equivalence
```
