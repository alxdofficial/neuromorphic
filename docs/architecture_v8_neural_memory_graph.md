# Architecture v8: Neural Memory Graph + Cortical Columns

> **Status**: Implemented. Training entry point: `python -m src.v8.train`
> **Code**: `src/v8/`
> **Branch**: `main`

## Design Philosophy

1. **The cortical column is the universal compute unit.** Our CCs are the language
   model — scan layers + PCM. Trained by backprop.

2. **Memory is activation flowing through weighted connections.** A persistent
   network of neurons outside autograd. Signals propagate hop-by-hop at every token.

3. **Plasticity is local and neuromodulated.** The neuromodulator (RL) controls
   primitives, routing keys, and decay. Structural plasticity (prune/regrow)
   is autonomous, driven by co-activation statistics.

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
|  PCM: per-CC surprise + learnable gain modulation  [at split] |
|      |                                                        |
|      v  CC signals (H_mid sliced per CC, D_cc=128)            |
|      |                                                        |
|  +-- MEMORY GRAPH (per-token, sequential) -----------------+  |
|  |  1024 neurons, 96 presynaptic connections each          |  |
|  |  receive -> integrate -> message (every token)          |  |
|  |  Port neuron messages -> mem_signals                    |  |
|  +--------------------------------------------------------+  |
|      |                                                        |
|      v  H_enriched = H_mid + gate * mem_signals               |
|      |                                                        |
|  Upper scan: layers 4-6                            [parallel]  |
|  Output head: proj_down -> LayerNorm -> lm_head               |
|  Trained by backprop. ~93M params.                            |
+-----------------+---------------------------------------------+
                  | neuron observations (514 dims per neuron)
                  v
+---------------------------------------------------------------+
|                    NEUROMODULATOR                              |
|  GRPO trajectory scoring (8 trajectories, K=96 neurons)       |
|  Actor: 3-layer MLP (514->2048->2048->2048 -> action heads)  |
|  GAE advantages + GRPO ranking (no value function)            |
|  Collects 4 chunks (64 segments) before RL update             |
|  16 actions per chunk (every 128 tokens):                     |
|    delta_primitive[128] + delta_key[128] + delta_decay[1]     |
|  ~10M params (actor). Trained by policy gradient.             |
+---------------------------------------------------------------+
```

**Total trained params: ~103M** (scan + embed + PCM + neuromod)

---

## Component Details

### Cortical Columns (V8LM) — `src/v8/lm.py`

**Split-scan architecture:**

1. **Lower scan** (`forward_scan_lower`): Embed -> pos_embed -> layers 0-3 -> PCM
   - PCM at the split point: computes surprise, applies learnable gain modulation
   - Surprising tokens get amplified before memory reads them
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

Per-column predictive coding with learnable gain modulation:
- `surprise = z_hat[t-1] - z[t]` (prediction error vector, 128 dims)
- `gain = sigmoid(W_gain(surprise)) * gain_scale`
- `gain_scale`: learnable parameter, init 2.0 (starts at gain=1.0, can grow)
- PCM runs at the split point so memory receives surprise-amplified signals

### Neural Memory Graph — `src/v8/memory_graph.py`

1024 neurons with 96 presynaptic connections each. Each neuron has:
- `h [128]`: internal state — persistent across tokens/chunks
- `prev_messages [128]`: last outgoing message
- `primitives [128]`: modulates outgoing messages, RMS-normalized (neuromod-controlled)
- `key [128]`: routing selectivity, L2-normalized (neuromod-controlled)
- `decay_logit [1]`: state persistence (neuromod-controlled)

**Per-token neuron dynamics (sequential loop, every token):**

```
1. RECEIVE: received = A @ prev_messages            [BS, N, D]
   Port neurons: received[:, :C] += cc_signals[:, t]
2. INTEGRATE: h = decay * h + (1-decay) * received  [BS, N, D]
3. MESSAGE: prev_messages = tanh(h * primitives)     [BS, N, D]
4. OUTPUT: mem_signals[:, t] = prev_messages[:, :C]  (port neurons)
```

**Key-based softmax routing:**
- Each neuron has a `key` vector (L2-normalized, 128 dims)
- At the start of each segment, compute: `sim = key . neighbor_messages` (raw dot product)
- Routing weights: `softmax(sim)` over K=96 neighbors per neuron
- These scalar routing weights are used for all tokens in the segment
- The neuromodulator controls routing selectivity by assigning new key values
- The Triton kernel uses precomputed scalar routing weights (same structure as old fixed-weight kernel)

**Binary firing + percentile threshold:**
- `activation = message.norm()` per neuron per token
- `fired = activation > 75th percentile(activation)` within the segment
- No per-neuron EMA tracking needed — percentile computed per segment
- `firing_rate` tracked as EMA for neuromod observation

**Co-activation structural plasticity** (every 4 segments, vectorized):
- Phi coefficient (binary Pearson) computed from firing traces via bmm
- `co_activation_ema [N, N]`: slow EMA of phi matrix
- Prune: existing connections where phi < 0 (anti-correlated)
- Grow: 80% toward highest-phi unconnected neuron, 20% random
- Regrown connections are topology-only (no persistent edge weights)
- No magic number thresholds — phi < 0 is a natural boundary

**Document boundary resets** (`reset_streams`):
- Zeros: h, prev_messages, firing_rate
- Preserves: primitives, key, decay, co_activation_ema, connectivity

### Neuromodulator — `src/v8/neuromodulator.py`

Shared policy network across all 1024 neurons. Counterfactual baseline (no critic).

**Observation** per neuron (514 dims):
- primitive[128] + key[128] + mean_input[128] + mean_output[128]
- firing_rate[1] + decay[1]

**Action** per neuron (257 dims):
- new_primitive[128] + new_key[128] + delta_decay[1]
- Additive deltas for primitives, keys, and decay, then RMS-normalized / L2-normalized

**Architecture**:
- Actor: 3-layer MLP (514->2048->2048->2048 -> action heads), Tanh activations. ~10M params.
- No critic/value function. Advantage via GRPO trajectory scoring + GAE.

### RL Training

GRPO trajectory scoring + GAE advantages, entropy bonus, and LR schedule:
1. Lower scan (layers 0-3) + PCM (parallel over T=2048)
2. 16 segments per chunk: neuromod observes -> acts -> per-token memory loop (128 tokens)
3. Per-segment CE loss as reward, collected across `rl_collect_chunks=4` chunks (64 segments)
4. GRPO scoring (every 4 chunks): sample 8 alternative trajectories for K=96 neurons on the
   last chunk's text. Each trajectory: observe → sample → apply per segment (mirrors real forward).
   Rank by CE loss, z-score normalize → per-trajectory advantages.
5. GAE (lambda=0.95) over all 64 segments with batch-mean baseline
6. Replay: GAE policy gradient (batched 8 segments at a time) + GRPO encouragement of
   best-scoring trajectories' actions
7. Entropy bonus (rl_entropy_coef=0.01) prevents exploration collapse
8. Neuromod LR: warmup + cosine decay, floor derived from LR_MIN/LR ratio

---

## Training — `src/v8/train.py`

```bash
python -u -m src.v8.train --bs 12 --steps 61035 --no-compile              # Phase 2: with neuromod
python -u -m src.v8.train --bs 12 --steps 61035 --no-compile --no-neuromod # Phase 1: LM only
python -u -m src.v8.train --bs 12 --steps 61035 --no-memory                # baseline (no memory)
```

**Per step:**
1. Lower scan (layers 0-3) + PCM at split point (parallel)
2. 16 segments of 128 tokens: neuromod observes → acts → per-token neuron loop
3. Inject memory into H_mid, upper scan (layers 4-6, parallel)
4. LM loss backward (gradients flow through upper + lower scan + mem_gate)
5. RL data collected across rl_collect_chunks=4 chunks (64 segments total)
6. On RL update step: GRPO scoring (8 trajectories on last chunk), GAE advantages
   over all 64 segments, replay for neuromod gradients + entropy bonus
7. Structural plasticity check (every 4 segments, vectorized)
8. Detach scan carries. Memory graph state persists.

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
| neuromod_hidden | 2048 |
| neuromod_layers | 3 |
| obs_dim per neuron | 514 |
| act_dim per neuron | 257 |
| max_action_magnitude | 1.0 (RMS/L2 normalization bounds effect) |
| RL | GRPO (8 trajectories, K=96 neurons) + GAE (4 chunks), entropy bonus |
| T | 2048 |
| Throughput | ~53K tok/s collect, ~16K avg with GRPO (RTX 4090, BS=8) |
| **Total trained params** | **~103M** |

---

## Design Decisions

1. **Split-scan with mid-injection.** Memory is injected between lower (0-3) and
   upper (4-6) scan layers. The upper layers see memory-enriched representations
   and learn to use them. The LM cannot ignore memory.

2. **Per-token neuron dynamics.** Sequential loop with bmm per token. Inherently
   non-parallelizable (nonlinear recurrence + inter-neuron coupling). Signals
   propagate hop-by-hop through the graph.

3. **Key-based softmax routing.** Each neuron has a learned key vector
   (L2-normalized). Routing weights are `softmax(key . neighbor_messages)`
   (raw dot product), computed once per segment. The neuromod controls
   routing selectivity by assigning new key values. Content-based attention
   over presynaptic neighbors.

4. **Primitives modulate outgoing messages.** `message = tanh(h * primitives)`.
   All 1024 neurons use their RMS-normalized primitives. Decay controls
   persistence. Keys control routing. Clean separation of concerns.

5. **Co-activation-based structural plasticity.** Phi coefficient matrix tracks
   temporal co-firing patterns. Anti-correlated connections pruned (phi < 0).
   New connections formed toward highest-phi unconnected neurons (80%) or
   random (20%). All thresholds relative to network state.

6. **PCM at the split point with learnable gain.** Surprise modulates H_mid
   before memory reads it. Surprising tokens get amplified. gain_scale is
   learnable (starts at 2.0, giving gain=1.0 at init).

7. **GRPO trajectory scoring + GAE.** Per-segment CE loss as reward, collected
   across 4 chunks (64 segments) before computing advantages and updating.
   GRPO: sample 8 alternative trajectories for K=96 neurons, each with per-segment
   observe → sample → apply (mirrors real forward). Rank by CE loss. GAE (lambda=0.95)
   over all segments with batch-mean baseline. No value function or critic needed.
   Entropy bonus prevents exploration collapse. Neuromod LR decays alongside LM LR.

---

## File Structure

```
src/v8/
+-- config.py              # V8Config dataclass + tier presets
+-- pcm.py                 # SingleColumnPCM (learnable gain, per-CC)
+-- memory_graph.py        # MemoryGraph (per-token + Triton kernel + plasticity)
+-- triton_kernels.py      # Fused sparse-gather + integrate + tanh + norm step kernel
+-- lm.py                  # V8LM (split-scan + PCM + memory interface)
+-- neuromodulator.py      # Policy network (no critic)
+-- model.py               # V8Model (top-level wiring, multi-chunk RL)
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
