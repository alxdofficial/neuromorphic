# v9-backprop — Differentiable Memory Graph with 2-Pass Simulation

**Branch**: `v9-backprop`
**Updated**: 2026-03-28

---

## Overview

A neuromorphic memory graph augments a split-scan language model. 512 neurons with
256-dimensional state vectors communicate through 32 sparse connections each. A
per-neuron modulator MLP predicts connection weights, decay rates, and message
primitives once per segment. The neuron dynamics run for T steps per segment with
full backprop, using a 2-pass optimization that reduces inter-neuron gathers from
T to 2.

**110M params** (LM 52M + Memory 58M). **24.8K tok/s** at BS=48 on RTX 4090.

---

## The Neuron Model (Sequential Mental Image)

Imagine 512 neurons arranged in a graph. Each neuron has:
- A **hidden state** h [256 dims] — what the neuron "knows"
- An **outgoing message** msg [256 dims] — what it broadcasts to neighbors
- **32 incoming connections** to other neurons, each with a learned weight

Every token step, each neuron:

```
1. LISTEN:   Read 32 neighbors' messages, weight by connection strengths,
             reduce through dendritic tree → received [256]
2. INJECT:   Add the LM's hidden state for this token → input_vec [256]
3. THINK:    State MLP updates hidden state: h_new = MLP(input_vec, h_old, decay)
4. SPEAK:    Message MLP generates outgoing message: msg = MLP(h_new, primitives)
5. IDENTIFY: Add learnable neuron ID to message
6. RECORD:   Track hebbian correlation (how active am I × connection strength)
```

After all T steps, the neurons' messages are read out and injected into the LM's
upper scan layers. This is the "true" sequential simulation — each step, every
neuron sees its neighbors' latest messages.

---

## The 2-Pass Optimization

The sequential simulation requires one **gather** (reading 32 neighbors' messages)
per step. At T=128, that's 128 gathers — the dominant cost.

The key insight: **inter-neuron messages change slowly** relative to the inject
signal (which changes every token). We can freeze the messages for an entire pass
and still get a good approximation.

### Pass 1: Approximate Trajectory

```
1. GATHER ONCE: Read all neighbors' messages from initial state
2. FREEZE: Use this same "received" for all T=128 steps
3. RUN: Each step, neurons update h and msg using frozen received + varying inject[t]
4. RESULT: Approximate trajectory — neurons responded to LM input but not to each other
```

**What this gets right:** Each neuron's response to the LM input (inject) at every
token position. The state MLP and message MLP process 128 different inject signals.

**What this misses:** Neuron A fires strongly at step 5 → Neuron B (connected to A)
should react at step 6. But B is reading A's initial message, not step 5's message.

### Pass 2: Refined Trajectory

```
1. GATHER ONCE: Read all neighbors' messages from Pass 1's FINAL state
2. FREEZE: Use this updated "received" for all T=128 steps
3. RUN: Same dynamics again, but now with better inter-neuron messages
4. RESULT: Refined trajectory — accounts for how neurons influenced each other
```

**What this fixes:** Neuron B now sees that A ended up in a high-activity state
(from Pass 1). This is a first-order correction — B knows A was active, though
it doesn't know exactly when during the 128 steps A became active.

**What remains approximate:** Second-order effects — A's reaction to B's reaction
to A. With sparse connectivity (K=32 out of N=512 = 6%), these higher-order
interactions are small.

### Cost Comparison

```
Sequential: 128 gathers + 128 MLP steps = slow (gathers dominate)
2-Pass:     2 gathers   + 256 MLP steps = fast (gathers negligible, MLPs dominate)
```

The 2-pass approach trades gather compute (expensive, scattered memory access) for
MLP compute (efficient, regular batched matmuls). The quality tradeoff is mild:
neurons see their neighbors' end-of-pass state rather than per-step state.

---

## Architecture

```
Input → Embedding → proj_up (768→2048)
  → LOWER SCAN (2 layers, d_inner=580)
  → PCM: predict H_{t+1} directly, surprise = predicted − actual
  → MEMORY GRAPH (2-pass, T steps per pass)
      Modulator: predict w_conn, decay, primitives (once per segment)
      Pass 1: frozen gather → T MLP steps → approximate trajectory
      Pass 2: updated gather → T MLP steps → refined trajectory
      Readout: average replicas → mem_out [BS, T, D]
  → Split-point MLP: H_combined = H_mid + MLP(cat(H_mid, surprise))
  → INJECT: H_enriched = H_combined + sigmoid(gate) × mem_out
  → UPPER SCAN (2 layers)
  → proj_down (2048→768) → ln_final → lm_head → logits
```

---

## Config (Tier A)

| Parameter | Value | Notes |
|-----------|-------|-------|
| D | 2048 | LM hidden dimension |
| D_embed | 768 | Embedding dimension |
| L_total | 4 | Scan layers (2 lower + 2 upper) |
| d_inner | 580 | Scan layer inner dimension |
| N_mem_neurons | 512 | Number of neurons |
| D_neuron | 256 | Per-neuron state dimension |
| K_connections | 32 | Sparse connections per neuron |
| neuromod_hidden | 80 | Modulator MLP hidden dimension |
| state_mlp_hidden | 24 | State update MLP hidden dimension |
| msg_mlp_hidden | 24 | Message MLP hidden dimension |
| T | 128 | Tokens per chunk (= segment length) |
| Total params | 110M | LM=52M, Memory=58M |

---

## Parameter Budget

```
LM:                                          52M
  Embedding (32K × 768):                     24.6M
  proj_up/down (768↔2048):                   3.1M
  pos_embed (128 × 2048):                    0.3M
  4 scan layers (d_inner=580):               19.0M
  PCM:                                       1.0M
  split_mlp:                                 3.5M
  mem_gate [16]:                             ~0

Memory:                                      58M
  Modulator (mod_w1/w2 per neuron):          7.5M
  State MLP (state_w1/w2 per neuron):        9.6M
  Message MLP (msg_w1/w2 per neuron):        9.6M
  Dendritic tree (branch_w + group_w):       4.5M
  Neuron ID [512, 256]:                      0.1M
```

---

## Gradient Flow

```
CE loss → logits → upper scan → inject_memory(gate) → readout(mean over replicas)
  → msgs[T steps, pass 2] → msg_MLP(msg_w1, msg_w2)
  → state_MLP(state_w1, state_w2)
  → input_vec = frozen_received + inject[t]
  → frozen_received ← gather(prev_msg, w_conn_sig)
  → w_conn_sig = sigmoid(w_conn)
  → w_conn ← modulator MLP(mod_w1, mod_w2) ← [hebbian, h, decay, primitives]
```

All memory graph parameters get gradients through this chain. The modulator's
gradient comes through w_conn/decay/primitives which affect all T steps in both
passes.

---

## Hebbian Traces and Structural Plasticity

**Hebbian traces** [N, K] track per-connection correlation during the last pass:
```
For each step: trace[k] += |msg| × sigmoid(w_conn[k])
```
High trace = neuron fired strongly AND connection weight was high = this connection
was useful. The modulator reads hebbian traces at the next segment to decide
updated connection weights.

**Structural plasticity** runs between chunks (after backward):
- Track co-activation: which neuron pairs fire together
- Weakest connections (lowest hebbian) get replaced by random new connections
- K stays fixed at 32 per neuron
- Non-differentiable topology change

---

## Performance

```
RTX 4090 (25GB VRAM):
  BS=32: 21.4K tok/s, 14.6GB
  BS=48: 24.8K tok/s, 21.8GB

Memory segment (BS=48):
  Forward:  82ms  (2 gathers: 0.3ms, 256 MLP steps: ~67ms, overhead: ~15ms)
  Backward: 102ms (2.0× forward)
  LM:       28ms
```

---

## Files

| File | Purpose |
|------|---------|
| `src/v8/config.py` | Configuration |
| `src/v8/memory_graph.py` | 2-pass neuron simulation, modulator, dendrites, plasticity |
| `src/v8/model.py` | LM + memory integration |
| `src/v8/lm.py` | Split-scan LM, PCM, split_mlp, inject_memory |
| `src/v8/pcm.py` | Predictive coding: predict H_{t+1} |
| `src/v8/trainer.py` | Training loop |
| `src/v8/train.py` | Entry point, optimizer setup |
| `src/v8/diagnostics.py` | Metrics and snapshots |
