# Architecture v8: Neural Memory Graph + Cortical Columns

> **Status**: Implemented. Training entry point: `python -m src.v8.train`
> **Code**: `src/v8/`
> **Branch**: `v7-single-scan-stack`

## Design Philosophy

1. **The cortical column is the universal compute unit.** Our CCs are the language
   model — scan layers + PCM. Trained by backprop.

2. **Memory is activation flowing through weighted connections.** A persistent
   network of neurons outside autograd. Signals propagate hop-by-hop at every token.

3. **Plasticity is local and neuromodulated.** The neuromodulator (RL) controls
   primitives, connection weight distribution, and decay. Structural plasticity
   (prune/regrow) is autonomous, driven by co-activation statistics.

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
                  | neuron observations (387 dims per neuron)
                  v
+---------------------------------------------------------------+
|                    NEUROMODULATOR                              |
|  Per-segment REINFORCE + entropy bonus                        |
|  Policy: 3-layer MLP (387->2048->2048->2048 -> action heads) |
|  Batch-mean baseline, per-step advantages, LR schedule        |
|  8 actions per chunk (every 256 tokens):                      |
|    delta_primitive[128] + delta_conn_weights[96] + decay[1]   |
|  ~10M params (actor only). Trained by policy gradient.        |
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
- `primitives [128]`: modulates outgoing messages (neuromod-controlled)
- `decay_logit [1]`: state persistence (neuromod-controlled)
- `conn_weights [96]`: presynaptic routing, L1-normalized (neuromod-controlled)

**Per-token neuron dynamics (sequential loop, every token):**

```
1. RECEIVE: received = A @ prev_messages            [BS, N, D]
   Port neurons: received[:, :C] += cc_signals[:, t]
2. INTEGRATE: h = decay * h + (1-decay) * received  [BS, N, D]
3. MESSAGE: prev_messages = tanh(h * primitives)     [BS, N, D]
4. OUTPUT: mem_signals[:, t] = prev_messages[:, :C]  (port neurons)
```

**Adjacency matrix** `A [BS, N, N]`:
- Built from sparse `conn_indices` + L1-normalized `conn_weights`
- L1 normalization: `sum |w| = 1` per neuron (energy conservation)
- Neuromod controls the distribution, not the magnitude
- Cached with dirty flag, rebuilt only when conn_weights change

**Binary firing + adaptive threshold:**
- `activation = message.norm()` per neuron per token
- `fired = activation > (activation_ema + activation_std_ema)`
- All relative to the neuron's own statistics, no magic numbers
- `firing_rate` tracked as EMA for neuromod observation

**Co-activation structural plasticity** (every 4 segments, vectorized):
- Phi coefficient (binary Pearson) computed from firing traces via bmm
- `co_activation_ema [N, N]`: slow EMA of phi matrix
- Prune: existing connections where phi < 0 (anti-correlated)
- Grow: 80% toward highest-phi unconnected neuron, 20% random
- Regrown weight: median of neuron's current |weights|
- No magic number thresholds — phi < 0 is a natural boundary

**Document boundary resets** (`reset_streams`):
- Zeros: h, prev_messages, activation_ema, activation_std_ema, firing_rate
- Preserves: primitives, conn_weights, decay, co_activation_ema, connectivity

### Neuromodulator — `src/v8/neuromodulator.py`

Shared policy network across all 1024 neurons. No critic.

**Observation** per neuron (387 dims):
- primitive[128] + mean_input[128] + mean_output[128]
- firing_rate[1] + decay[1] + routing_entropy[1]

**Action** per neuron (225 dims):
- delta_primitive[128] + delta_conn_weights[96] + delta_decay[1]
- Actions clamped to [-1.0, 1.0], then conn_weights L1-renormalized

**Architecture**: 3-layer MLP, hidden=2048, Tanh activations. ~10M params.

### RL Training

Per-segment REINFORCE with discounted returns, entropy bonus, and LR schedule:
1. Lower scan (layers 0-3) + PCM (parallel over T=2048)
2. 8 segments: neuromod observes -> acts -> per-token memory loop (256 tokens)
3. Per-segment CE loss -> discounted returns (gamma=0.99)
4. Batch-mean baseline per step -> per-step advantages
5. Replay: log_prob with stored per-segment obs, weighted by advantages
6. Entropy bonus (0.01) prevents exploration collapse
7. Neuromod LR: warmup + cosine decay to 10% of initial

---

## Training — `src/v8/train.py`

```bash
python -u -m src.v8.train --bs 12 --steps 61035 --no-compile
python -u -m src.v8.train --bs 12 --steps 61035 --no-memory  # baseline
```

**Per step:**
1. Lower scan (layers 0-3) + PCM at split point (parallel)
2. 8 segments of 256 tokens: neuromod -> per-token neuron loop
3. Inject memory into H_mid, upper scan (layers 4-6, parallel)
4. LM loss backward (gradients flow through upper + lower scan + mem_gate)
5. Neuromod replay: per-step advantages + entropy bonus, backward
6. Structural plasticity check (every 4 segments, vectorized)
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
| action_every | 256 tokens (8 segments per chunk) |
| neuromod_hidden | 2048 |
| neuromod_layers | 3 |
| obs_dim per neuron | 387 |
| act_dim per neuron | 225 |
| max_action_magnitude | 1.0 (L1 normalization bounds effect) |
| RL | Per-segment REINFORCE, discounted returns, entropy bonus |
| T | 2048 |
| Throughput | ~44K tok/s with memory, ~85K without (RTX 4090, BS=12) |
| **Total trained params** | **~103M** |

---

## Design Decisions

1. **Split-scan with mid-injection.** Memory is injected between lower (0-3) and
   upper (4-6) scan layers. The upper layers see memory-enriched representations
   and learn to use them. The LM cannot ignore memory.

2. **Per-token neuron dynamics.** Sequential loop with bmm per token. Inherently
   non-parallelizable (nonlinear recurrence + inter-neuron coupling). Signals
   propagate hop-by-hop through the graph.

3. **L1-normalized connection weights (energy conservation).** Each neuron's
   weights sum to 1.0 in absolute value. The neuromod controls the distribution,
   not the magnitude. Strengthening one connection weakens others.

4. **Primitives modulate outgoing messages.** `message = tanh(h * primitives)`.
   All 1024 neurons use their primitives. Decay controls persistence. Connection
   weights control routing. Clean separation of concerns.

5. **Co-activation-based structural plasticity.** Phi coefficient matrix tracks
   temporal co-firing patterns. Anti-correlated connections pruned (phi < 0).
   New connections formed toward highest-phi unconnected neurons (80%) or
   random (20%). All thresholds relative to network state.

6. **PCM at the split point with learnable gain.** Surprise modulates H_mid
   before memory reads it. Surprising tokens get amplified. gain_scale is
   learnable (starts at 2.0, giving gain=1.0 at init).

7. **Per-segment RL with entropy bonus.** Each of 8 segments gets its own CE
   loss as reward. Discounted returns (gamma=0.99) credit early actions for
   downstream improvements. Entropy bonus prevents exploration collapse.
   Neuromod LR decays alongside LM LR.

---

## File Structure

```
src/v8/
+-- config.py              # V8Config dataclass + tier presets
+-- pcm.py                 # SingleColumnPCM (learnable gain, per-CC)
+-- memory_graph.py        # MemoryGraph (per-token + Triton kernel + plasticity)
+-- triton_kernels.py      # Fused sparse-gather step kernel
+-- lm.py                  # V8LM (split-scan + PCM + memory interface)
+-- neuromodulator.py      # Policy network (no critic)
+-- model.py               # V8Model (top-level wiring, per-segment RL)
+-- trainer.py             # V8Trainer (joint LM + RL training loop)
+-- train.py               # Training entry point with CLI
+-- diagnostics.py         # Per-step metrics + periodic snapshots

scripts/
+-- benchmark_v8.py        # Throughput + VRAM benchmarking
+-- profile_v8.py          # torch.profiler breakdown
+-- plot_training.py       # Training curves + memory health plots
+-- analyze_memory.py      # Deep per-token memory analysis
+-- find_max_bs.py         # Max batch size finder + leak check

tests/v8/
+-- test_integration.py    # Full forward + backward + RL
+-- test_memory_graph.py   # Neuron dynamics, plasticity, reference tests
+-- test_triton_kernel.py  # Triton vs Python equivalence
```
