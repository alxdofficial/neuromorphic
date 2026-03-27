# Architecture v9: Differentiable Neural Memory Graph

> **Status**: Implemented. Training: `python -m src.v8.train`
> **Code**: `src/v8/`
> **Branch**: `main` (v8 RL version: `v8-rl-neuromod`)

## Design Philosophy

1. **The cortical column is the universal compute unit.** CCs are the language
   model — scan layers + PCM. Trained by backprop.

2. **Memory is activation flowing through weighted connections.** A persistent
   neuron graph with learned parameters. Signals propagate hop-by-hop at every token.

3. **Plasticity is learned end-to-end.** Per-neuron modulators (each with its own
   MLP) predict gate and decay from internal state. Trained by backprop during
   training, active at inference without gradients for adaptation.

4. **Per-token neuron dynamics.** Each neuron receives presynaptic messages via
   learned dendritic FC layers, integrates with internal state, broadcasts a
   message — at every token. Fully differentiable.

5. **Split-scan with mid-injection.** Lower scan produces representations, memory
   processes them per-token, memory output is injected mid-scan.

---

## Architecture Overview

```
+---------------------------------------------------------------+
|                    CORTICAL COLUMNS (V8LM)                    |
|                                                               |
|  Lower scan: layers 0-2 (D=2048, d_inner=1024)   [parallel]  |
|  PCM: per-CC surprise (side input to upper scan)  [at split]  |
|      |                                                        |
|      v  CC signals (H_mid per CC, D_cc=128, detached)         |
|      |                                                        |
|  +-- MEMORY GRAPH (nn.Module, ~22.5M params) ---------------+ |
|  |  1024 neurons, 96 connections each (fixed topology)       | |
|  |  Per-neuron modulator: h -> gate_prim, gate_key, decay_mod| |
|  |  Per-neuron dendritic FC: learned branch + group weights  | |
|  |  Per-token: receive -> integrate -> message (differentiable)| |
|  |  TBPTT: h detached at segment boundaries                  | |
|  +-----------------------------------------------------------+ |
|      |                                                        |
|      v  H_enriched = H_mid + sigmoid(gate) * mem_signals      |
|      |                                                        |
|  Upper scan: layers 3-4                            [parallel]  |
|  Output head: proj_down -> LayerNorm -> lm_head               |
+---------------------------------------------------------------+

Single optimizer (AdamW). End-to-end backprop.
Total: ~104M params (LM ~81M + memory graph ~22.5M)
```

---

## Component Details

### Cortical Columns (V8LM) — `src/v8/lm.py`

**Split-scan architecture (5 layers, split at 3):**

1. **Lower scan** (`forward_scan_lower`): Embed -> pos_embed -> layers 0-2 -> PCM
   - Returns H_mid [BS, T, D] with autograd intact
   - Surprise as side input to first upper layer

2. **Memory injection** (`inject_memory`): H_enriched = H_mid + gate * mem_signals
   - `mem_gate[c]`: learnable scalar per CC, starts at sigmoid(0)=0.5
   - mem_signals from port neuron messages (differentiable through memory graph)

3. **Upper scan** (`forward_scan_upper`): layers 3-4

4. **Output** (`forward_output`): proj_down -> LayerNorm -> lm_head

### PCM — `src/v8/pcm.py`

Unchanged from v8. Batched predictive coding (BatchedPCM) processing all C=16
columns via `torch.bmm`. RMSNorm on inputs. Surprise as side input to first
upper layer.

### Neural Memory Graph — `src/v8/memory_graph.py`

Now an `nn.Module` with ~22.5M learned parameters:

**Per-neuron learned parameters:**
- `primitives [N, D_mem]`: modulates outgoing messages. RMS-normalized init.
- `key [N, D_mem]`: routing selectivity. RMS-normalized init.
- `decay_logit [N]`: state persistence. Zero init (sigmoid=0.5).
- `dendrite_branch_w [N, n_branches, branch_size, D_mem]`: per-neuron FC at branch level (12.6M)
- `dendrite_group_w [N, n_groups, bpg, D_mem]`: per-neuron FC at group level (1.0M)
- `fc1_w [N, D_mem, 64]`, `fc1_b [N, 64]`: per-neuron modulator hidden layer (8.4M)
- `fc2_w [N, 64, 3]`, `fc2_b [N, 3]`: per-neuron modulator output (zero-init, 196K)
- `mod_lr_logit [1]`: learnable modulation step size

**Per-batch state tensors (not nn.Parameters):**
- `h [BS, N, D_mem]`: hidden state (detached at segment boundaries)
- `prev_messages [BS, N, D_mem]`: last outgoing messages
- `trace_prim, trace_key [BS, N, D_mem]`: eligibility traces (always detached)

### Per-Segment Forward (128 tokens, differentiable)

```
1. MODULATOR: gate_prim, gate_key, decay_mod = modulator(h_prev_detached)
   - Per-neuron MLP: h -> tanh(fc1) -> fc2 -> (gate[-1,1], gate[-1,1], decay_mod)
   - Zero-init fc2: starts as no-op, grows contribution as it trains

2. EFFECTIVE PARAMS (on compute graph):
   eff_prim = primitives + mod_lr * gate_prim * normalize(trace_prim)
   eff_key  = key + mod_lr * gate_key * normalize(trace_key)
   eff_decay = sigmoid(decay_logit + decay_mod)

3. ROUTING (once per segment):
   routing = sigmoid(eff_key . neighbor_messages)  [BS, N, K]
   Per-connection independent gating, no normalization.

4. PER-TOKEN DYNAMICS (128 steps):
   For each token t:
     a. RECEIVE: dendritic FC gather with per-neuron learned weights
        Branch level: tanh(weighted_inputs * branch_w)  per branch
        Group level: tanh(branch_outputs * group_w)  per group
        Soma: mean across groups
     b. CC INJECTION: received[:, :C] += cc_signals[:, t]
     c. INTEGRATE: h = eff_decay * h + (1-eff_decay) * received
     d. MESSAGE: prev_msg = tanh(h * eff_prim)

5. OUTPUT: port neuron messages -> LM injection

6. TRACES (detached, no_grad):
   trace_prim = 0.95 * trace_prim + 0.05 * h.detach()
   trace_key  = 0.95 * trace_key  + 0.05 * mean_input.detach()
```

**Gradient flow**: loss -> messages -> h, eff_prim -> gate_prim -> modulator params + primitives.
Also through routing -> eff_key -> key. And through integration -> eff_decay -> decay_logit.

### TBPTT

h is detached at segment boundaries (128 tokens). Each segment is an independent
compute graph. Memory graph nn.Parameters accumulate gradients from all 16
segments in a chunk.

---

## Training — `src/v8/train.py`

```bash
python -u -m src.v8.train --bs 8 --steps 30000
```

**Per step:**
1. Lower scan (layers 0-2) + PCM surprise (parallel over T=2048)
2. 16 segments of 128 tokens each (sequential, differentiable):
   - modulator -> effective params -> per-token dynamics -> output
   - TBPTT: detach h between segments
3. Inject memory into H_mid, upper scan (layers 3-4, parallel)
4. CE loss + aux_loss (PCM prediction) -> single backward pass
5. Single optimizer step (AdamW, memory graph LR = 0.3x LM LR)
6. Detach all states for next chunk

---

## Tier A Config

| Component | Value |
|-----------|-------|
| D | 2048 |
| D_embed | 768 |
| C (cortical columns) | 16 |
| D_cc = D_mem | 128 |
| L_total | 5 |
| scan_split_at | 3 (lower: 0-2, upper: 3-4) |
| d_inner | 1024 |
| N_mem_neurons | 1024 |
| K_connections | 96 |
| action_every | 128 tokens (16 segments per chunk) |
| modulator_hidden | 64 (per-neuron MLP) |
| trace_decay | 0.95 |
| dendrite_branch_size | 12 |
| T | 2048 |
| LM params | ~81M |
| Memory graph params | ~22.5M |
| **Total** | **~104M** |

---

## Design Decisions

1. **Per-neuron modulators (not shared).** Each neuron has its own MLP weights
   (8.65M total). Allows each neuron to develop unique plasticity strategy.
   Implemented via einsum for batched per-neuron matmul.

2. **Per-neuron dendritic FC (not flat sum).** Each neuron's dendritic tree has
   learned weights at branch and group levels (13.6M total). Replaces unweighted
   sum-then-tanh with learned per-neuron mixing of incoming signals.

3. **Effective parameters = base + modulation.** Primitives/keys are nn.Parameters
   (base values, updated by optimizer). The modulator adds state-dependent
   modulations on top. At inference: base is frozen, modulator still runs.

4. **cc_signals detached.** Memory graph doesn't backprop into lower scan.
   Prevents lower scan from being destabilized by memory gradients.

5. **No structural plasticity.** Fixed random topology. Dendritic FC weights
   learn which connections matter (replaces co-activation-based rewiring).

6. **Modulator zero-init.** fc2 initialized to zeros = no modulation at start.
   Primitives/keys learn entirely via direct backprop first. Modulator
   contribution grows organically as it trains.

---

## File Structure

```
src/v8/
+-- config.py              # V8Config dataclass + tier presets
+-- pcm.py                 # BatchedPCM (bmm, RMSNorm, surprise)
+-- memory_graph.py        # MemoryGraph nn.Module (dendritic FC, per-neuron modulator)
+-- triton_kernels.py      # Triton kernels (unused in v9 Phase 1 — Python loops for differentiability)
+-- lm.py                  # V8LM (split-scan + PCM + memory interface)
+-- model.py               # V8Model (segment loop with TBPTT)
+-- trainer.py             # V8Trainer (single optimizer, pure backprop)
+-- train.py               # Training entry point
+-- diagnostics.py         # Per-step metrics + periodic snapshots

tests/v8/
+-- test_integration.py    # Full forward + backward + gradient tests
+-- test_memory_graph.py   # Neuron dynamics, modulator, dendritic FC, TBPTT
+-- test_triton_kernel.py  # Skipped (Triton not used in v9 Phase 1)
```
