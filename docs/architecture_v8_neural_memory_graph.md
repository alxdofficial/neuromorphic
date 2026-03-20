# Architecture v8: Neural Memory Graph + Cortical Columns

> **Status**: Implemented. Training entry point: `python -m src.v8.train`
> **Code**: `src/v8/`
> **Previous versions**: v7 (single scan stack) on `v7-single-scan-stack` branch.

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
   by neuromodulators. We substitute RL (PPO) training for the neuromodulator.

4. **Energy conservation constrains signal routing.** Output magnitude is divided
   among connections proportionally, creating natural sparsity.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                    CORTICAL COLUMNS (V8LM)                     │
│  Full-D scan stack: 8 layers (shared), D=2048, d_inner=1024   │
│  16 per-CC PCMs (independent weights, D_cc=128, hidden=256)   │
│  Two-pass forward:                                             │
│    Pass 1: layers[0..3] + PCM → H [BS,T,D], surprise          │
│    Pass 2: layers[4..7] with memory injection → logits         │
│  Trained by backprop. ~99M params.                             │
└────────────────────┬────────────────────┬──────────────────────┘
                     │ raw H_slice        │ raw mem_signal
                     │ (D_cc=128, no      │ (D_mem=D_cc=128,
                     │  projection)       │  gated by mem_gate)
                     ▼                    │
┌───────────────────────────────────────────────────────────────┐
│                    NEURAL MEMORY GRAPH                          │
│  8192 neurons in 8 blocks × 1024 per block                    │
│  D_mem = D_cc = 128 (no projections between CC and memory)    │
│  160 connections per neuron (128 random intra + 32 inter)      │
│  2 CCs per block (16 CCs / 8 blocks)                          │
│                                                                │
│  Each token: receive → modulate: silu(input * primitive) → route│
│  No learned weights in memory — neuromod controls everything   │
│  Per-neuron temperature (routing sharpness) + decay (persistence)│
│  Energy-conserving: output magnitude = sum of outbound signals │
│                                                                │
│  NOT in autograd. Runs with torch.no_grad().                   │
│  ~18 MB state per stream. Zero trainable params.               │
└────────────────────┬───────────────────────────────────────────┘
                     │ neuron observations
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                    NEUROMODULATOR                               │
│  PPO actor-critic, shared across all 8192 neurons              │
│  Actor: 3-layer MLP (obs→1024→1024→1024 → action heads)       │
│  Critic: 3-layer MLP (same dims → scalar value)               │
│  Actions every 8 tokens: delta_primitive + delta_thresholds    │
│    + delta_temperature + delta_decay (290 dims per neuron)     │
│  Reward: per-token CE loss (block-level, gamma=0.99)           │
│  ~8M params. Trained by PPO.                                   │
└───────────────────────────────────────────────────────────────┘
```

**Total trained params: ~107M** (67.2M scan + 24.6M embedding + 4.2M pos + 3.1M proj + 2.4M PCM + ~8M neuromod)

---

## Component Details

### Cortical Columns (V8LM) — `src/v8/lm.py`

Full-D scan stack, same architecture as v7. Provides all cross-column mixing
and causal token processing. Memory is an add-on, not a dependency.

**Two-pass forward:**
1. **Pass 1** (`forward_pre_memory`): Embed → pos_embed[0..T] → layers[0..L_mem-1] → PCM per-CC → H, surprise
2. **Memory loop**: per-token CC↔memory signal exchange (sequential, cheap, no_grad)
3. **Pass 2** (`forward_post_memory`): inject memory → layers[L_mem..L_total-1] → proj_down → logits

**CC↔memory interface — no projections:**
- D_mem = D_cc = 128 by design. No projections needed.
- CC→memory: raw `H_slice[:, t, c]` (D_cc=128) sent directly to port neuron
- Memory→CC: raw `mem_signal[:, :, c]` (D_mem=128) gated by `mem_gate[c]`
- `mem_gate[c]`: learnable scalar, starts at sigmoid(0)=0.5
- `pcm_modules[c]`: `SingleColumnPCM(D_cc, hidden=256)` — independent weights per CC

### Neural Memory Graph — `src/v8/memory_graph.py`

8192 neurons in 8 blocks of 1024. Blocks are independent of CC count
(16 CCs / 8 blocks = 2 CCs per block). Each neuron has:
- `primitive [D_mem=128]`: stored information, gating what neuron responds to
- `thresholds [160]`: energy costs per connection (128 random intra + 32 inter)
- `temperature [1]`: per-neuron routing sharpness (neuromod-controlled)
- `decay [1]`: per-neuron activation persistence (neuromod-controlled)

**Per-token step:**
1. **Inject** CC signals into port neurons (2 ports per block, one per CC)
2. **Gather** inputs from connected neurons' previous outputs (double-buffered)
3. **Modulate**: `output = silu(input * primitive)` — element-wise, no learned weights
4. **Route**: softmax(-thresholds / neuron_temperature) → sparsify → scatter (energy-conserving)
5. **Read** port neuron activations → return to CCs
6. **Decay**: `prev_output = sigmoid(decay) * old + (1-sigmoid(decay)) * outputs`

**Connectivity**: random sparse within block (K_intra=128 of 1024), random sparse across blocks (K_inter=32). Fixed topology at init.

**No learned weights in memory graph.** All behavior controlled by neuromodulator through primitives, thresholds, temperature, and decay.

### Neuromodulator — `src/v8/neuromodulator.py`

Shared PPO actor-critic across all 8192 neurons.

**Observation** per neuron (516 dims): primitive + mean_input + mean_output + usage + temperature + decay + routing_entropy + CC_surprise
**Action** per neuron (290 dims): delta_primitive[128] + delta_thresholds[160] + delta_temperature[1] + delta_decay[1]
**Architecture**: separate 3-layer actor/critic, hidden=1024, Tanh activations

### PPO Training — `src/v8/ppo.py`

- Rollout buffer: `[T//8, BS×8192, ...]` — 256 action steps × 65K+ parallel environments
- GAE: gamma=0.99, lambda=0.95, with dones mask at doc boundaries
- Reward: negative per-token CE loss, weighted by per-block surprise, shifted by action_every
- Update: 4 epochs, minibatch=512, clip=0.2, entropy coef=0.003

---

## Training — `src/v8/train.py`

```bash
# Full v8 with memory
python -u -m src.v8.train --bs 8 --steps 10000 --compile

# LM-only baseline (no memory graph, no PPO)
python -u -m src.v8.train --bs 8 --steps 10000 --compile --no-memory
```

**Per step:**
1. Forward: Pass 1 (parallel) → memory loop (sequential, cheap) → Pass 2 (parallel)
2. LM loss backward through CCs only
3. PPO update on neuromodulator using collected experience
4. Detach scan carries. Memory graph state persists.

**Data**: The Pile, same pipeline as v7. T=2048 tokens per chunk, BS=8.

---

## Tier A Config

| Component | Value |
|-----------|-------|
| D | 2048 |
| D_embed | 768 |
| C (cortical columns) | 16 |
| D_cc = D_mem | 128 |
| L_total (scan layers) | 8 |
| L_mem (injection point) | 4 |
| d_inner | 1024 |
| N_blocks | 8 |
| M_per_block | 1024 |
| N_neurons (total) | 8192 |
| K_intra (random within block) | 128 |
| K_inter (random across blocks) | 32 |
| CCs per block | 2 |
| neuromod_hidden | 1024 |
| neuromod_layers | 3 |
| action_every | 8 tokens |
| action_dims per neuron | 290 |
| T | 2048 |
| **Total trained params** | **~107M** |

**Param breakdown:**

| Component | Params | % | Trained by |
|-----------|--------|---|------------|
| Scan layers (8 shared) | 67.2M | 62.7% | Backprop |
| Embedding (tied) | 24.6M | 22.9% | Backprop |
| Positional embedding [2048, 2048] | 4.2M | 3.9% | Backprop |
| proj_up + proj_down | 3.1M | 2.9% | Backprop |
| PCM (16 independent, hidden=256) | 2.4M | 2.2% | Backprop |
| mem_gate (16 scalars) | 16 | <0.01% | Backprop |
| Neuromodulator (actor+critic) | ~8M | 5.3% | PPO |

**Memory graph state** (per stream, not trained params): ~18 MB
- Zero trainable parameters in memory graph
- All behavior controlled by neuromodulator through primitives, thresholds, temperature, decay

---

## Design Decisions

1. **D_mem = D_cc always.** No projections between CC and memory. The CC's raw
   hidden state is the memory input. Memory's raw output is the CC's signal.
   Eliminates untrained projection layers.

2. **Blocks independent of CCs.** 8 blocks × 1024 neurons, 16 CCs → 2 CCs per
   block. Blocks are local neighborhoods of neurons; CCs are slices of the LM.
   The mapping is fixed (each CC has a home block).

3. **No learned weights in memory.** Neuron modulation is `silu(input * primitive)`.
   The neuromodulator (PPO) controls everything: what neurons store (primitives),
   where signals route (thresholds), how sharply (temperature), how persistently
   (decay). Replaces explicit Hebbian learning, W_mod MLPs, etc.

4. **Random sparse connectivity.** 128 random intra-block + 32 inter-block
   connections per neuron. Uniform max_connections keeps tensors rectangular
   for GPU efficiency. Fixed topology at init.

---

## Open Questions

1. **Memory graph as CUDA kernel**: current implementation is pure PyTorch with
   `torch.no_grad()`. Custom Triton/CUDA kernel would reduce per-step overhead.

2. **Lifelong mode**: memory graph state persists across TBPTT boundaries by default.
   Doc boundary resets via EOT detection. Phase B disables resets.

---

## File Structure

```
src/v8/
├── __init__.py
├── __main__.py            # python -m src.v8.train
├── config.py              # V8Config dataclass + tier presets
├── pcm.py                 # SingleColumnPCM (per-CC, independent weights)
├── memory_graph.py        # MemoryGraph (no autograd, SIMD-parallel)
├── lm.py                  # V8LM (scan stack + PCM + memory interface)
├── neuromodulator.py      # PPO actor-critic for plasticity control
├── ppo.py                 # PPORolloutBuffer, GAE, PPOTrainer
├── model.py               # V8Model (top-level wiring)
├── trainer.py             # V8Trainer (joint LM + PPO training loop)
└── train.py               # Training entry point with CLI

tests/v8/
├── test_memory_graph.py   # Graph step, routing, energy conservation
└── test_integration.py    # Full forward + backward + PPO
```
