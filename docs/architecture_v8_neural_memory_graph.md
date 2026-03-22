# Architecture v8: Neural Memory Graph + Cortical Columns

> **Status**: Implemented. Training entry point: `python -m src.v8.train`
> **Code**: `src/v8/`
> **Branch**: `v7-single-scan-stack`

## Design Philosophy

The human brain's memory is not a database with read/write operations. It is a
**sea of neurons constantly emitting and receiving signals**, where the pattern of
activity IS memory, and the connection weights determine what patterns emerge.
Recall is pattern completion through signal propagation. Learning is plasticity
modulated by neuromodulators — without backpropagation.

We replicate the principles that matter for language modeling:

1. **The cortical column is the universal compute unit.** Same circuit everywhere.
   Our CCs are the language model — scan layers + PCM. Trained by backprop.

2. **Memory is activation flowing through weighted connections.** No separate
   memory store. Neurons fire, signals propagate, patterns emerge. Our neural
   memory graph is a persistent network of neurons outside autograd.

3. **Plasticity is local and neuromodulated.** How much synapses change is gated
   by neuromodulators. We substitute RL training for the neuromodulator.

4. **Per-token neuron dynamics.** Each neuron receives presynaptic messages,
   integrates them with its internal state, and broadcasts an outgoing message
   — at every token timestep. Signals propagate through the graph hop-by-hop.

5. **Structural plasticity reshapes the graph.** Connection utility (signal flow,
   co-activation correlation) is tracked as EMAs. The neuromodulator adjusts
   connection weights informed by these metrics. Dead connections are pruned and
   randomly regrown.

---

## Architecture Overview

```
+---------------------------------------------------------------+
|                    CORTICAL COLUMNS (V8LM)                    |
|  Full-D scan stack: 7 layers (shared), D=2048, d_inner=1024  |
|  16 per-CC PCMs (independent weights, D_cc=128, hidden=256)  |
|  Single-pass: all layers -> PCM -> H + surprise (once)        |
|  End-injection: logits = output_head(H + gate * mem_signal)   |
|  Trained by backprop. ~93M params.                            |
+-----------------+----------------------------+----------------+
                  | raw H_slice (D_cc=128)     ^ port neuron messages
                  | per CC per token           | (D_mem=128, gated)
                  v                            |
+---------------------------------------------------------------+
|                    NEURAL MEMORY GRAPH                         |
|  1024 neurons, 96 presynaptic connections per neuron          |
|  D_mem = D_cc = 128 (no projections between CC and memory)   |
|                                                               |
|  PER-TOKEN NEURON DYNAMICS (sequential, every token):         |
|    1. Receive: received = A @ prev_messages                   |
|       Port neurons also receive CC signal from LM             |
|    2. Integrate: h = decay * h + (1-decay) * received         |
|    3. Message: prev_messages = tanh(h * primitives)           |
|                                                               |
|  Neuromod acts every 256 tokens (8 segments per chunk)        |
|  Structural plasticity: prune + regrow every 4 segments (twice/chunk)      |
|  NOT in autograd. Runs with torch.no_grad().                  |
|  Zero trainable params -- neuromod controls everything.       |
+-----------------+---------------------------------------------+
                  | neuron observations + plasticity metrics
                  v
+---------------------------------------------------------------+
|                    NEUROMODULATOR                              |
|  Per-segment REINFORCE with discounted returns                |
|  Policy: 3-layer MLP (obs->2048->2048->2048 -> action heads) |
|  Batch-mean baseline, per-step advantages (no critic)         |
|  8 actions per chunk (every 256 tokens):                      |
|    delta_primitive[128] + delta_conn_weights[96] + decay[1]   |
|  ~10M params (actor only). Trained by policy gradient.        |
+---------------------------------------------------------------+
```

**Total trained params: ~103M** (scan + embed + PCM + neuromod)

---

## Component Details

### Cortical Columns (V8LM) — `src/v8/lm.py`

Full-D scan stack, same architecture as v7. Provides all cross-column mixing
and causal token processing. Memory is an add-on, not a dependency.

**Single-pass forward:**
1. **Scan** (`forward_scan`): Embed -> pos_embed -> ALL scan layers -> PCM per-CC -> H, surprise
2. **Memory**: 8 segments of 256 tokens, each: neuromod -> per-token neuron loop
3. **Output** (`forward_output`): logits = output_head(H + gate * mem_signal)

**CC<->memory interface — no projections:**
- D_mem = D_cc = 128 by design. No projections needed.
- CC->memory: raw `H_slice[:, :, c]` (D_cc=128) added to port neuron c's received input
- Memory->CC: port neuron message `tanh(h * primitives)` (D_mem=128) gated by `mem_gate[c]`
- `mem_gate[c]`: learnable scalar, starts at sigmoid(0)=0.5

### Neural Memory Graph — `src/v8/memory_graph.py`

1024 neurons with 96 presynaptic connections each. Each neuron has:
- `h [D_mem=128]`: internal state — the neuron's memory, persists across tokens/chunks
- `prev_messages [D_mem=128]`: last outgoing message, received by postsynaptic partners
- `primitives [D_mem=128]`: modulates outgoing messages (neuromod-controlled)
- `decay_logit [1]`: controls state persistence vs responsiveness (neuromod-controlled)
- `conn_weights [96]`: presynaptic connection strengths (neuromod-controlled)

**Per-token neuron dynamics (sequential loop, every token):**

```
For each token t in segment:
  1. RECEIVE: received = A @ prev_messages            [BS, N, D]
     Port neurons: received[:, :C] += cc_signals[:, t]
  2. INTEGRATE: h = decay * h + (1-decay) * received  [BS, N, D]
  3. MESSAGE: prev_messages = tanh(h * primitives)     [BS, N, D]
  4. OUTPUT: mem_signals[:, t] = prev_messages[:, :C]  (port neurons only)
```

- `A [BS, N, N]`: mean-normalized adjacency from sparse connectivity + conn_weights
- `decay = sigmoid(decay_logit)`: per-neuron, neuromod-controlled
- `primitives`: per-neuron, define what each neuron broadcasts
- Signal propagation: K tokens = K hops through the graph
- Internal state `h` persists — never thrown away, blended via decay

**Adjacency matrix:**
- Dense `[N, N]` built from sparse `conn_indices` + `conn_weights`
- Mean-normalized by active connections to prevent signal accumulation
- Cached with dirty flag, rebuilt only when conn_weights change
- `A[i, j]` = weight of presynaptic connection from neuron j to neuron i

**Plasticity metrics** (tracked at segment end):
- `flow_ema[i,k]`: EMA of `|conn_weight * neighbor_message|` — signal flow magnitude
- `corr_ema[i,k]`: EMA of `neuron_message * neighbor_message` — co-activation correlation
- Added to neuromod observation as per-neuron summaries

**Structural plasticity** (every 4 segments, twice per chunk):
- Connections with |weight| < threshold (0.01) are pruned
- Pruned connections randomly rewired to new neurons
- Neuromod drives weights toward zero for useless connections (implicit pruning)

**No learned weights in memory graph.** All behavior controlled by neuromodulator
through primitives, connection weights, and decay.

### Neuromodulator — `src/v8/neuromodulator.py`

Shared policy network across all 1024 neurons. No critic.

**Observation** per neuron (391 dims):
- primitive[128] + mean_input[128] + mean_output[128]
- usage[1] + decay[1] + routing_entropy[1]
- mean_flow[1] + std_flow[1] + mean_corr[1] + min_flow[1]

**Action** per neuron (225 dims):
- delta_primitive[128] + delta_conn_weights[96] + delta_decay[1]

**Architecture**: 3-layer MLP, hidden=2048, Tanh activations. ~10M params.

### RL Training (Per-Segment REINFORCE with Discounted Returns)

Dense per-segment rewards with discounted returns and batch-mean baseline:
1. Scan runs once (parallel over T=2048)
2. 8 segments: neuromod observes -> acts -> per-token memory loop (256 tokens)
3. Per-segment CE loss -> discounted returns (gamma=0.99)
4. Batch-mean baseline per step -> per-step advantages
5. Replay: evaluate log_prob for all 8 actions with correct per-segment obs
6. Policy gradient: each segment's actions weighted by its own advantage

Per-segment obs stored during forward for accurate replay. No critic needed.

---

## Training — `src/v8/train.py`

```bash
# Full v8 with memory
python -u -m src.v8.train --bs 8 --steps 10000

# LM-only baseline (no memory graph, no RL)
python -u -m src.v8.train --bs 8 --steps 10000 --no-memory
```

**Per step:**
1. Full scan over T tokens (parallel, once)
2. 8 segments of 256 tokens: neuromod observes -> acts -> per-token neuron loop
3. Per-segment CE loss, discounted returns, batch-mean baseline
4. LM loss backward through CCs
5. Neuromod replay: evaluate log_prob with stored obs, per-step advantages, backward
6. Structural plasticity check (every 4 segments)
7. Detach scan carries. Memory graph state persists.

**Data**: The Pile, same pipeline as v7. T=2048 tokens per chunk.

---

## Tier A Config

| Component | Value |
|-----------|-------|
| D | 2048 |
| D_embed | 768 |
| C (cortical columns) | 16 |
| D_cc = D_mem | 128 |
| L_total (scan layers, single pass) | 7 |
| d_inner | 1024 |
| N_mem_neurons | 1024 |
| K_connections | 96 |
| action_every | 256 tokens (8 segments per chunk) |
| neuromod_hidden | 2048 |
| neuromod_layers | 3 |
| obs_dim per neuron | 391 |
| act_dim per neuron | 225 |
| RL | Per-segment REINFORCE, discounted returns, batch-mean baseline |
| T | 2048 |
| **Total trained params** | **~103M** |

---

## Design Decisions

1. **D_mem = D_cc always.** No projections between CC and memory. The CC's raw
   hidden state is the memory input. Port neuron messages are the CC's signal.

2. **Per-token neuron dynamics.** Each neuron receives, integrates, and messages
   at every token. This is the core of the model — signals propagate through the
   graph hop-by-hop, enabling multi-hop communication within each segment. The
   loop is sequential (inherently non-parallelizable due to nonlinear recurrence
   with inter-neuron coupling).

3. **Primitives modulate outgoing messages.** `message = tanh(h * primitives)`.
   All 1024 neurons use their primitives. The neuromodulator controls what each
   neuron broadcasts by adjusting primitives. Decay controls temporal persistence.
   Connection weights control routing. Clean separation of concerns.

4. **Random sparse connectivity.** 96 random presynaptic connections per neuron.
   Fixed topology at init. Structural plasticity (prune + regrow) adapts the
   graph over time, driven by neuromodulator weight adjustments informed by
   flow/correlation metrics.

5. **Dense matmul for message passing.** Sparse connectivity is represented as a
   dense `[N, N]` adjacency matrix (~2MB at N=1024). One `torch.bmm` call per
   token step. Cached with dirty flag, rebuilt only when conn_weights change.

6. **Mean-normalized message passing.** Each neuron's incoming signals are averaged
   (not summed) over active connections. Prevents signal accumulation at high-fan-in
   neurons.

7. **Per-segment RL with discounted returns.** Each of 8 segments gets its own CE
   loss as reward. Discounted returns (gamma=0.99) credit early actions for
   downstream improvements. Batch-mean baseline provides zero-parameter variance
   reduction. Per-segment obs stored for accurate replay.

---

## Open Questions

1. **Speed optimization**: The per-token sequential loop (2048 bmm calls per chunk)
   is the main throughput cost. Future options: sub-segment message passing
   (diagonal scan within sub-segments, message pass between), custom CUDA kernels,
   or CUDA graphs to reduce Python overhead.

2. **Lifelong mode**: Memory graph state persists across TBPTT boundaries by default.
   Doc boundary resets via EOT detection.

3. **Learned baseline (Phase 2 RL)**: If per-segment REINFORCE plateaus, a small
   learned value head (single linear layer) could replace the batch-mean baseline
   for better variance reduction.

---

## File Structure

```
src/v8/
+-- __init__.py
+-- __main__.py            # python -m src.v8.train
+-- config.py              # V8Config dataclass + tier presets
+-- pcm.py                 # SingleColumnPCM (per-CC, independent weights)
+-- memory_graph.py        # MemoryGraph (per-token recurrence + sparse graph)
+-- lm.py                  # V8LM (scan stack + PCM + memory interface)
+-- neuromodulator.py      # Policy network for plasticity control (no critic)
+-- model.py               # V8Model (top-level wiring, per-segment RL)
+-- trainer.py             # V8Trainer (joint LM + RL training loop)
+-- train.py               # Training entry point with CLI

tests/v8/
+-- test_memory_graph.py   # Neuron dynamics, message passing, plasticity
+-- test_integration.py    # Full forward + backward + RL
```
