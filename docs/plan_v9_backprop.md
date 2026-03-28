# v9-backprop — Fully Differentiable Memory Graph

**Branch**: `v9-backprop`
**Updated**: 2026-03-28
**Target**: 40K+ tok/s on RTX 4090, BS=8, T=2048

---

## Architecture Overview

```
Input → Embedding → proj_up (768→2048)
  → LOWER SCAN (2 layers)
  → PCM: predict H_{t+1} directly, surprise = H_hat - H_actual
  → MEMORY GRAPH (differentiable, 16 segments × 64 steps)
      Segment boundary: modulator(hebbian, h, decay, prim) → new w_conn, decay, prim
      Per step: gather → weight → dendritic_tree → inject → state_MLP → msg_MLP → +neuron_id
      After all segments: structural plasticity rewires connections
  → Split-point MLP: H_combined = H_mid + MLP(cat(H_mid, surprise))
  → INJECT: H_enriched = H_combined + gate * mem_readout
  → UPPER SCAN (2 layers)
  → proj_down (2048→768) → ln_final → lm_head → logits
```

**Total: 114M params** (53M LM + 61M memory)

---

## LM (Split-Scan)

- **D=2048**, D_embed=768, C=16 cortical columns, D_cc=128
- **4 scan layers** (split at 2): 2 lower + 2 upper
- **d_inner=512**, GLU output
- **PCM** at split point: single predictor MLP, `H_hat_{t+1} = pred(norm(H_t))`
  - Surprise = `H_hat_{t-1} - H_t`
  - Loss = MSE(prediction, actual.detach())
  - No encoder — predicts scan hidden state directly (avoids degenerate collapse)
- **Split-point MLP**: `H + MLP(cat(H, surprise))` — residual, zero-init final layer
  - Linear(4096, 512) → SiLU → Linear(512, 2048)
- **mem_gate** [C=16]: per-column sigmoid gate for memory readout

---

## Memory Graph

### Topology
- **N=4096 neurons**, D_neuron=32, **K=128 connections** per neuron
- Fixed sparse topology (conn_indices [N, K]), rewired by structural plasticity
- **Dendritic tree**: 8 branches × 16 synapses, 2 groups × 4 branches
  - branch→tanh→group→tanh→mean

### I/O (zero parameters)
- **Inject**: H_mid [BS,T,2048] → view [BS,T,64,32] → replicate 64× → [BS,T,4096,32]
  - Each of 64 slices shared by 64 neurons. No learned weights.
- **Readout**: msgs [BS,T,4096,32] → view [BS,T,64,64,32] → mean(dim=3) → [BS,T,2048]
  - Average 64 neuron replicas per slice. No learned weights.

### Per-Step Dynamics (64 steps per segment at stride=2)

```
1. GATHER:      neighbor_msgs = prev_msg[:, conn_indices]           → [BS, N, K, D]
2. WEIGHT:      weighted = sigmoid(w_conn) * neighbor_msgs           → [BS, N, K, D]
3. DENDRITIC:   received = dendritic_tree(weighted, branch_w, group_w) → [BS, N, D]
4. INJECT:      input_vec = received + cc_slice                      → [BS, N, D]
5. STATE MLP:   h_new = state_mlp(cat(input_vec, h))                 → [BS, N, D]
                Linear(64,24) → tanh → Linear(24,32) → tanh
6. MSG MLP:     msg = msg_mlp(cat(h_new, primitive))                 → [BS, N, D]
                Linear(64,24) → tanh → Linear(24,32) → tanh
7. NEURON ID:   msg = msg + neuron_id                                → [BS, N, D]
8. HEBBIAN:     trace_k += |msg| * sigmoid(w_conn_k)                 → [BS, N, K]
```

### Segment-Boundary Modulator (runs FIRST, once per segment)

```
Input:  cat(hebbian_traces[K=128], h[D=32], decay[1], primitive[D=32]) → [193]
Hidden: Linear(193, 16) → tanh                                        → [16]
Output: Linear(16, 161) → split                                       → [161]

new_w_conn      = output[..., :128]        → [BS, N, K=128]
new_decay_logit = output[..., 128]         → [BS, N]
new_primitives  = output[..., 129:]        → [BS, N, D=32]
```

Runs before the token loop so we observe modulator effects during the segment.

### Structural Plasticity (chunk boundary, non-differentiable)

- Accumulate pairwise co-activation: `co_act += outer(msg_mag_per_neuron)` per segment
- At chunk end: for each neuron, swap 8 weakest connections with 8 strongest non-connected
- K stays fixed at 128, conn_indices re-sorted after swaps
- N² = 16M entries (~64MB f32)

---

## nn.Parameters

| Parameter | Shape | Count | Purpose |
|-----------|-------|-------|---------|
| **Modulator** | | **23.6M** | |
| mod_w1 | [4096, 193, 16] | 12.6M | Input→hidden |
| mod_b1 | [4096, 16] | 0.07M | |
| mod_w2 | [4096, 16, 161] | 10.6M | Hidden→output |
| mod_b2 | [4096, 161] | 0.3M | |
| **State MLP** | | **9.6M** | |
| state_w1 | [4096, 64, 24] | 6.3M | Input→hidden |
| state_b1 | [4096, 24] | 0.1M | |
| state_w2 | [4096, 24, 32] | 3.1M | Hidden→output |
| state_b2 | [4096, 32] | 0.1M | |
| **Message MLP** | | **9.6M** | |
| msg_w1 | [4096, 64, 24] | 6.3M | Input→hidden |
| msg_b1 | [4096, 24] | 0.1M | |
| msg_w2 | [4096, 24, 32] | 3.1M | Hidden→output |
| msg_b2 | [4096, 32] | 0.1M | |
| **Dendrite** | | **17.8M** | |
| branch_w | [4096, 8, 16, 32] | 16.8M | Per-branch FC |
| group_w | [4096, 2, 4, 32] | 1.0M | Per-group FC |
| **Other** | | **0.1M** | |
| neuron_id | [4096, 32] | 0.1M | Learnable ID |

**Total memory: 61M params**

### Runtime State (NOT learned, per batch)

| Tensor | Shape | Purpose |
|--------|-------|---------|
| h | [BS, N, 32] | Neuron hidden state |
| prev_messages | [BS, N, 32] | Last outgoing messages |
| w_conn | [BS, N, 128] | Synaptic weights (set by modulator) |
| primitives_state | [BS, N, 32] | Message modulation (set by modulator) |
| decay_logit | [BS, N] | Leak rate (set by modulator) |
| hebbian_traces | [BS, N, 128] | Per-segment avg of |msg| × σ(w_conn) |
| co_activation | [N, N] | Pairwise co-fire for structural plasticity |

---

## Training

- **Single optimizer** (AdamW), 4 param groups:
  - LM decay/no-decay at base LR
  - Memory decay/no-decay at 0.3× base LR
- **Memory params in f32** (tiny gradients round to zero in bf16)
- **TBPTT**: detach h, prev_msg at segment boundaries
- **Gradient checkpointing**: 8-step groups within each segment
- **Loss**: CE + pcm_pred_weight × PCM prediction loss

### Gradient Flow
```
CE loss → logits → upper scan → inject_memory(gate)
  → readout(mean over replicas) → msg_all
  → msg_MLP(msg_w1, msg_w2) → state_MLP(state_w1, state_w2)
  → dendritic_gather(branch_w, group_w, w_conn_sig)
  → sigmoid(w_conn) ← modulator(mod_w1, mod_w2) ← [hebbian, h, decay, prim]
```

---

## Triton Kernel

**Fused dendritic gather** — forward + backward kernels:
- Fuses gather + weight + dendritic tree into one kernel per step
- Eliminates [BS, N, K, D] intermediate (~512MB at tier_a)
- Grid: (BS, N), one program per (batch, neuron)
- Backward: recomputes forward intermediates, atomic adds for shared grads
- Wrapped in `torch.autograd.Function` with Python fallback

The per-neuron MLPs (state, message) use PyTorch batched einsums (already fast).

---

## Throughput Analysis (2026-03-28)

Current: **2.1K tok/s** on RTX 4090, BS=8, T=2048.

| Component | Time/step | % |
|-----------|-----------|---|
| Fused gather (Triton) | 0.59ms | 81% |
| State MLP (einsum) | 0.07ms | 10% |
| Message MLP (einsum) | 0.07ms | 10% |
| **Total per step** | **0.73ms** | |

64 steps × 16 segments × fwd+bwd with checkpointing = ~8.3s/chunk.
LM alone: 0.088s (186K tok/s). Memory is 99% of compute.

**Bottleneck**: 128 random memory reads per neuron per step × 4096 neurons × 8 batch.
The gather is I/O bound — Triton eliminates the intermediate tensor but can't avoid the reads.

**Target: 40K tok/s** → need ~410ms/chunk → ~26ms/segment (fwd+bwd).

### Optimization paths:
1. **Fuse entire step** into one Triton kernel (gather + MLPs + hebbian). Eliminates per-step kernel launch overhead and intermediate tensors.
2. **torch.compile** on the step loop — could auto-fuse the Python loop.
3. **Increase stride**: 2→4 halves steps (32 instead of 64). Quality impact unknown.
4. **Persistent kernel**: One launch processes all 64 steps with barriers.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| PCM predicts H_{t+1} directly | Encoder+predictor both learned → degenerate collapse |
| Split-point MLP instead of side_input | Surprise modulates the representation, not just the gate |
| State MLP replaces leaky integration | More expressive — can fully rewrite state, not just blend |
| Message MLP replaces tanh(h×prim) | Richer message generation with nonlinear interaction |
| tanh at all MLP outputs | Bounded activations prevent explosion over 64 steps |
| Neuron ID embedding | Breaks symmetry — neurons develop unique identities |
| Hebbian traces → modulator | Modulator sees per-connection activity, enables credit assignment |
| Modulator runs FIRST | Observe effects during segment (not after) |
| Structural plasticity | Topology adapts to learned patterns; K fixed at 128 |
| D_neuron=32 (not 16) | D=16 killed Triton perf (v9.1 lesson), richer representations |
| K=128 (not 96) | More connections per neuron for richer communication |
| Memory params in f32 | bf16 gradients round to zero for modulator/MLP params |

---

## Files

| File | Purpose |
|------|---------|
| `src/v8/config.py` | Configuration (tier_a, tier_tiny) |
| `src/v8/memory_graph.py` | MemoryGraph: neurons, modulator, MLPs, dendrites, plasticity |
| `src/v8/model.py` | V8Model: LM + memory integration |
| `src/v8/lm.py` | V8LM: split-scan, PCM, split_mlp, inject_memory |
| `src/v8/pcm.py` | BatchedPCM: predict H_{t+1}, compute surprise |
| `src/v8/triton_kernels.py` | Fused dendritic gather (Triton fwd+bwd + autograd.Function) |
| `src/v8/trainer.py` | Training loop: single optimizer, joint backward |
| `src/v8/train.py` | Entry point: param groups, f32 conversion, LR schedule |
| `src/v8/diagnostics.py` | Per-step metrics + periodic snapshots |
| `tests/v8/test_memory_graph.py` | 27 tests: init, forward, gradient flow, plasticity |
| `tests/v8/test_integration.py` | 10 tests: end-to-end model, gradient flow |
| `tests/v8/test_triton_kernel.py` | 7 tests: Triton vs Python reference (GPU only) |
