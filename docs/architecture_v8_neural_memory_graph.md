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
│  Trained by backprop. 109M params.                             │
└────────────────────┬────────────────────┬──────────────────────┘
                     │ CC signals         │ memory signals
                     │ (H_slice+surprise  │ (per CC, D_mem=256)
                     │  → 2-layer MLP     │
                     │  → D_mem=256)      │
                     ▼                    │
┌───────────────────────────────────────────────────────────────┐
│                    NEURAL MEMORY GRAPH                          │
│  4096 neurons in 16 blocks × 256 per block                    │
│  D_mem=256 primitive vectors, 288 connections per neuron       │
│  (256 intra-block + 32 inter-block)                            │
│                                                                │
│  Each token: receive → modulate (2-layer W_mod) → route        │
│  Energy-conserving: output magnitude = sum of outbound signals │
│  Sparsity: bottom 50% of routing weights zeroed                │
│                                                                │
│  NOT in autograd. Runs with torch.no_grad().                   │
│  ~17 MB state per stream. W_mod: 394K params (RL-trained).     │
└────────────────────┬───────────────────────────────────────────┘
                     │ neuron observations
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                    NEUROMODULATOR                               │
│  PPO actor-critic, shared across all 4096 neurons              │
│  Actor: 3-layer MLP (898→1024→1024→1024 → action heads)       │
│  Critic: 3-layer MLP (same dims → scalar value)               │
│  Actions every 8 tokens: delta_primitive + delta_thresholds    │
│  Reward: per-token CE loss (block-level, gamma=0.99)           │
│  6.6M params. Trained by PPO.                                  │
└───────────────────────────────────────────────────────────────┘
```

**Total trained params: 115.4M** (67.2M scan + 24.6M embedding + 16.3M memory-related + 6.6M neuromod)

---

## Component Details

### Cortical Columns (V8LM) — `src/v8/lm.py`

Full-D scan stack, same architecture as v7. Provides all cross-column mixing
and causal token processing. Memory is an add-on, not a dependency.

**Two-pass forward:**
1. **Pass 1** (`forward_pre_memory`): Embed → pos_embed[0..T] → layers[0..L_mem-1] → PCM per-CC → H, surprise
2. **Memory loop**: per-token CC↔memory signal exchange (sequential, cheap, no_grad)
3. **Pass 2** (`forward_post_memory`): inject memory → layers[L_mem..L_total-1] → proj_down → logits

**Per-CC interface:**
- `mem_proj_in[c]`: 2-layer MLP `(D_cc+D_cc → 512 → D_mem)` — projects H_slice + surprise into memory signal
- `mem_proj_out[c]`: 2-layer MLP `(D_mem → 512 → D_cc)` — projects memory signal back to CC width. Zero-initialized last layer.
- `mem_gate[c]`: learnable scalar, starts at sigmoid(0)=0.5
- `pcm_modules[c]`: `SingleColumnPCM(D_cc, hidden=256)` — independent weights per CC

### Neural Memory Graph — `src/v8/memory_graph.py`

4096 neurons organized in 16 blocks of 256. Each neuron has:
- `primitive [D_mem=256]`: stored information
- `thresholds [288]`: energy costs for each connection (256 intra + 32 inter)
- `activation [D_mem]`: current step accumulation buffer

**Per-token step:**
1. **Inject** CC signals into block port neurons (neuron 0 of each block)
2. **Gather** inputs from connected neurons' previous outputs (double-buffered)
3. **Modulate**: `output = silu(W_mod2 @ silu(W_mod1 @ [input; primitive]))` — 2-layer shared MLP
4. **Route**: softmax(-thresholds/temp) → zero bottom 50% → renormalize → scatter (energy-conserving)
5. **Read** port neuron activations → return to CCs

**Connectivity**: fully connected within block, 32 random inter-block connections per neuron. Fixed topology (not learned).

### Neuromodulator — `src/v8/neuromodulator.py`

Shared PPO actor-critic across all neurons.

**Observation** per neuron (898 dims): primitive + mean_input + mean_output + usage + routing_entropy + CC_surprise
**Action** per neuron (544 dims): delta_primitive[256] + delta_thresholds[288]
**Architecture**: separate 3-layer actor/critic, hidden=1024, Tanh activations

### PPO Training — `src/v8/ppo.py`

- Rollout buffer: `[T//8, BS×4096, ...]` — 256 action steps × 65K parallel environments
- GAE: gamma=0.99, lambda=0.95. `0.99^256 ≈ 0.08` credit propagation across full chunk
- Reward: negative per-token CE loss, averaged per block per action interval
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
| C (cortical columns / blocks) | 16 |
| D_cc | 128 |
| L_total (scan layers) | 8 |
| L_mem (injection point) | 4 |
| d_inner | 1024 |
| N_neurons | 4096 |
| M_per_block | 256 |
| D_mem | 256 |
| inter_block_k | 32 |
| neuromod_hidden | 1024 |
| neuromod_layers | 3 |
| action_every | 8 tokens |
| T | 2048 |
| **Total trained params** | **115.4M** |

**Param breakdown:**

| Component | Params | % |
|-----------|--------|---|
| Scan layers (8 shared) | 67.2M | 58.3% |
| Embedding (tied) | 24.6M | 21.3% |
| Positional embedding [2048, 2048] | 4.2M | 3.6% |
| mem_proj_in (2-layer MLP × 16) | 4.2M | 3.6% |
| proj_up + proj_down | 3.1M | 2.7% |
| mem_proj_out (2-layer MLP × 16) | 3.2M | 2.8% |
| PCM (16 independent, hidden=256) | 2.4M | 2.1% |
| Neuromodulator (actor+critic) | 6.6M | 5.7% |

---

## Open Questions

1. **mem_proj_in training**: currently detached from LM autograd (goes into memory env).
   Should be trained by PPO alongside W_mod, or given a separate gradient path.

2. **Memory graph as CUDA kernel**: current implementation is pure PyTorch with
   `torch.no_grad()`. Custom Triton/CUDA kernel would reduce per-step overhead.

3. **Lifelong mode**: memory graph state persists across TBPTT boundaries by default.
   Doc boundary resets via EOT detection. Phase B disables resets.

4. **W_mod training**: the shared modulation MLP inside the memory graph is
   stored as plain tensors (not nn.Parameter). Needs to be included in PPO
   updates alongside the neuromodulator.

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
