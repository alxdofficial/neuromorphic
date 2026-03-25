# V8 Codebase Intent — 2026-03-25

Complete description of how the v8 neuromorphic language model works,
from data loading through training, RL, checkpointing, and diagnostics.
Written after the design review sweep that fixed PCM, RL trajectory
scoring, doc boundary resets, and best-trajectory state persistence.

---

## 1. Data Loading

**Source**: The Pile (deduplicated), tokenized with TinyLlama tokenizer (32K vocab).

**Preparation** (`scripts/prepare_data.py`): Streams from HuggingFace, saves as
raw `.parquet` (for inspection) and pre-tokenized `.bin` (flat uint16 array with
EOS between documents). Default: 2B train tokens, 5M validation tokens.

**Loading** (`src/data/streaming.py`): The primary path is `TokenShardDataset` —
memory-maps the `.bin` shard. BS independent streams start at evenly-spaced
offsets with seed-based jitter. Each step reads T+1 contiguous tokens (wrapping
at shard end), yielding:

- `input_ids[BS, T]` = tokens 0..T-1
- `target_ids[BS, T]` = tokens 1..T (shifted by 1)
- `prev_token[BS]` = last token of the previous chunk (for doc boundary detection)

Uses pinned memory + background prefetch thread for async CPU→GPU transfer.
On the first chunk, `prev_token` is initialized to `eos_token_id`, so the LM
scan carries reset at step 0 (but memory graph does NOT reset — see below).

---

## 2. Model Architecture

### 2.1 Split-Scan Language Model (`src/v8/lm.py`)

The LM is a stack of 7 causal linear recurrence layers (ScanLayers), split
into lower (0-3) and upper (4-6) with memory injection at the split point.

**ScanLayer** (`src/model/scan.py`): Each layer computes
`h_t = sigmoid(a) * h_{t-1} + silu(b)` where `[a, b] = proj_in(RMSNorm(x))`.
Uses the FLA HGRN Triton kernel on CUDA (O(N) fused), Hillis-Steele parallel
prefix scan on CPU. Output gate: optional SwiGLU (`proj_out` → split → silu(gate) × up).
GPT-2 depth scaling on `proj_out` weights. Residual: `out = dropout(proj_out(h)) + x`.
Returns `(output, h_last)` where `h_last` is the carry for TBPTT.

**Embedding**: `nn.Embedding(vocab, D_embed=768)` → `proj_up(768→2048)` +
learnable positional embedding `pos_embed[T, D]`. Tied weights:
`lm_head.weight = embedding.weight`.

**Lower scan** (`forward_scan_lower`): Embed → layers 0-3 → PCM at split point.
Returns `H_mid[BS, T, D]` (with autograd) + surprise + aux_loss.

**PCM** (`src/v8/pcm.py`): Per-CC predictive coding. Both encoding and prediction
condition on scan hidden state H AND input embedding x:
- `z_t = W_enc(cat(H_t, x_t))` — current position's encoding
- `z_hat_t = W_pcm(cat(H_t, x_t))` — prediction of next position's encoding
- `surprise = z_hat_{t-1} - z_t` — how unexpected this position's representation is
- Gain modulation: `H *= sigmoid(W_gain(surprise)) * gain_scale`
  (gain_scale starts at 2.0, W_gain zero-init → gain=1.0 at init)
- Aux loss: `MSE(z_hat[:-1], z[1:].detach())` × `pcm_pred_weight` (0.1)

**Memory injection** (`inject_memory`):
`H_enriched = H_mid + sigmoid(mem_gate)[per-CC] × mem_signals`
where `mem_gate` starts at 0 → sigmoid(0)=0.5 at init.

**Upper scan** (`forward_scan_upper`): Layers 4-6 on H_enriched. These layers
learn to use memory-enriched representations.

**Output** (`forward_output`): `proj_down(2048→768)` → LayerNorm → `lm_head × D_embed^{-0.5}`.

**Carries**: `_carries[L_total]` — one per scan layer. Detached at TBPTT
boundaries. Reset at document boundaries (LM only, not memory graph).

### 2.2 Memory Graph (`src/v8/memory_graph.py`)

A persistent graph of 1024 neurons with 96 sparse presynaptic connections each.
NOT an nn.Module — state tensors are outside autograd.

**State per neuron** (all `[BS, N, D_mem]` unless noted):
- `primitives`: what the neuron broadcasts (RMS-normalized, per-dim ≈ 1)
- `key`: what it listens for (L2-normalized unit vector)
- `decay_logit` `[BS, N]`: temporal persistence (sigmoid(0)=0.5 at init)
- `h`: internal state (temporal memory)
- `prev_messages`: last outgoing messages
- `mean_input`, `mean_output`: per-neuron segment averages
- `firing_rate` `[BS, N]`: EMA of activation rate
- `co_activation_ema` `[N, N]`: phi coefficient matrix (shared across batch)

**Connectivity**: `conn_indices[N, K=96]` — random sparse graph, sorted per neuron
for cache-friendly gather. No self-connections. First C=16 neurons are CC ports.

**Per-token dynamics** (`forward_segment`): Computed once per segment:
routing weights = `softmax(key · neighbor_messages)` over K=96 connections.
Then for each token t:
1. **Receive**: `received = A @ prev_messages` (dense bmm from sparse routing weights).
   Port neurons add CC signal: `received[:,:C] += cc_signals[:,t]`
2. **Integrate**: `h = decay × h + (1-decay) × received` (convex combination)
3. **Message**: `prev_messages = tanh(h × primitives)`

Output = port neuron messages `[BS, T_seg, C, D_mem]` — injected back into LM.

**Memory graph is persistent** — no resets at document boundaries. The memory
graph learns to handle document transitions through changes in CC signal
character (since the LM scan carries DO reset at doc boundaries, H_mid changes
abruptly, which the memory graph observes through CC input).

**Triton kernel** (`src/v8/triton_kernels.py`): Same dynamics, one kernel launch
per token step. Grid = (BS, N), each program handles all D dims for one neuron.
Fuses sparse gather + integration + tanh + message norm.

**Firing stats**: Binary firing at 75th percentile within each segment (no EMA
lag). Firing rate updated via EMA (decay=0.99).

**Co-activation** (`_post_segment_stats`): Phi coefficient matrix between all
neuron pairs, computed from binary firing patterns. Batch-averaged, updated
via slow EMA (decay=0.995). Only computed when structural plasticity is due.

**Structural plasticity** (`structural_plasticity`): Every 4 segments (configurable):
- Prune: connections where phi < 0 (anti-correlated neurons)
- Grow: connect to best unconnected neuron by phi (80%) or random (20%)
- At most 1 connection per neuron per step. Re-sorts for cache locality.

### 2.3 Neuromodulator (`src/v8/neuromodulator.py`)

RL-trained policy that modifies neuron primitives, key, and decay.
Shared MLP across all neurons (parameter sharing).

**Architecture**: `neuromod_layers` × (Linear → Tanh) backbone.
Three action heads: `prim_head(→D_mem)`, `key_head(→D_mem)`, `decay_head(→1)`.
Total act_dim = `D_mem × 2 + 1 = 257`.

**Log-std**: Three learnable parameters (state-independent):
`prim_logstd[1, D_mem]`, `key_logstd[1, D_mem]`, `decay_logstd[1, 1]`.
Init: -0.5 (std ≈ 0.6).

**Sampling**: `action = mean + std × randn()`. For replay (action provided):
compute log_prob of that action under current policy.

**Direct Gaussian math** (avoids torch.distributions overhead):
```
log_prob = sum(-0.5 × ((a-μ)²/σ² + log2π) - logstd)
entropy = sum(0.5 × (1 + log2π) + logstd)
```

**Observation** (`get_neuron_obs`): `[primitives, key, mean_input, mean_output,
firing_rate, sigmoid(decay)]` = obs_dim = D_mem×4 + 2 = 514.

**Apply actions** (`apply_actions`): Additive deltas, clamped to ±max_action (1.0).
After: RMS-normalize primitives, L2-normalize key. No decay clamp needed
(convex combination self-bounds h).

---

## 3. Forward Pass (`src/v8/model.py: forward_chunk`)

Per chunk (T=2048 tokens):

1. **Reset scan carries** at doc boundaries (LM only, not memory graph)
2. **Lower scan + PCM** → `H_mid[BS, T, D]` with autograd
3. **No-memory fast path**: if disabled, upper scan + output → return
4. **CC signals** = `H_mid.detach().view(BS, T, C, D_mem)` (detached — no gradient from memory into lower scan)
5. **Segment loop** (16 segments of 128 tokens):
   - Neuromod: observe → sample action (no_grad) → apply to memory graph
   - `mg.forward_segment(seg_cc)` → `[BS, 128, C, D_mem]`
   - Structural plasticity if due
6. **Inject memory**: `H_enriched = H_mid + gate × mem_out`
7. **Upper scan** → H
8. **Output head** → logits
9. **Collect RL data**: `{eot_at, cc_segments, H_mid, pre_upper_carries}`

---

## 4. RL System (GRPO)

### 4.1 Collection (trainer.py)

Every chunk with neuromod enabled, the trainer:
1. Computes per-segment CE (for logging)
2. Saves target_ids in rl_data
3. On the first chunk of a collection window, saves pre-collection memory graph state
4. Appends rl_data to buffer

After `rl_collect_chunks` (4) chunks: triggers GRPO scoring and update.

### 4.2 Trajectory Scoring (`score_trajectories`)

Scores N=8 trajectories across ALL collected chunks (not just the last one):

1. **Choose K=96 neurons** (fixed for all trajectories and all chunks)
2. For each of 8 trajectories:
   - Restore memory graph to pre-collection state
   - Restore upper scan carries to chunk 0's pre-forward state
   - For each of 4 chunks:
     - For each of 16 segments:
       - Observe all neurons → sample stochastic action from policy
       - **Only K neurons get actions, non-K get zero delta**
       - Apply action → run memory graph segment
     - Inject memory → upper scan → output → float32 CE (masked by EOT)
   - trajectory_loss = mean CE across all 4 chunks
3. **Z-score normalize** trajectory losses → advantages
4. **Best trajectory persists**: memory graph state AND upper scan carries
   from the lowest-loss trajectory become the real state going forward

**Why this design**:
- Scoring across 4 chunks (64 segments) captures long-range memory effects
- Zero delta for non-K neurons: any loss difference is 100% attributable to K neurons
- Fixed K throughout: evaluating a coherent strategy, not a patchwork
- Best-of-N state selection: free performance boost from selection pressure alone

### 4.3 Policy Gradient (`replay_for_neuromod_grads`)

For each trajectory with positive advantage:
1. Batch all segments' K-neuron obs and actions (across all 4 chunks)
2. Forward through neuromod (WITH grad) to compute log_prob
3. `grpo_loss = -(advantage × mean_log_prob)`
4. `entropy_bonus = -entropy_coef × mean_entropy`
5. Backward: `(grpo_loss + entropy_bonus) / n_positive`

If all trajectories tied: apply entropy bonus only (pushes toward exploration).

**Why replay works**: `log_prob(action | obs)` is differentiable w.r.t. policy
parameters. The gradient shifts the policy mean toward actions that got positive
advantage, and adjusts logstd based on how far good actions were from the mean.

---

## 5. Training Loop

### 5.1 CLI (`src/v8/train.py`)

- Always tier_a config. Vocab/eot from tokenizer.
- `--no-memory`: LM-only baseline (no memory graph)
- `--no-neuromod`: Phase 1 (LM + frozen memory, no RL)
- `--freeze-lm`: Phase 2 (freeze all LM, only neuromod trains via GRPO)
- `--resume`: loads LM + neuromod + optimizer. If `--freeze-lm`, resets logstd
  to config init (-0.5) so exploration isn't locked to Phase 1 values.
- Compiles individual methods: `forward_scan_lower`, `forward_scan_upper`,
  `forward_output`, `neuromod.get_action_and_value`

### 5.2 Optimizers

- **LM**: AdamW, lr=3e-4 → 3e-5 cosine, betas=(0.9, 0.95), weight decay 0.01
  on 2d+ params, fused on CUDA
- **Neuromod**: Adam, lr=3e-4, eps=1e-5, no weight decay. Same cosine schedule.

### 5.3 Per-step (`trainer.py: train_chunk`)

1. Move batch to GPU (non_blocking)
2. Doc boundary detection: `has_reset = (prev_token == eot_id).any()` on CPU
3. Forward chunk under AMP (bf16)
4. CE loss: per-token, masked by EOT positions, + aux_loss
5. LM backward + optimizer step (skipped if frozen)
6. RL: accumulate buffer → score → replay → neuromod step (every 4 chunks)
7. TBPTT: `detach_states()`

### 5.4 Checkpointing

Saves: LM state, neuromod state, both optimizer states, scheduler state,
memory graph state, step number, config. Rotates to keep last N checkpoints.

Memory graph state includes: primitives, key, decay_logit, connectivity
(conn_indices, conn_mask), h, prev_messages, stats, co_activation_ema.

---

## 6. Diagnostics (`src/v8/diagnostics.py`)

### Per-step metrics (cheap):
- Memory graph: h_norm, msg_norm, prim_std, key_diversity, decay mean/std,
  firing rates (port vs non-port), dead neuron fraction, tanh saturation,
  phi stats (mean, std, pos/neg fractions)
- LM coupling: mem_gate values per CC
- Neuromod: logstd values (8 decimal places), action magnitude stats

### Periodic snapshots:
- Per-neuron: norms, decay, firing rate
- Connectivity: conn_indices, routing weights
- Full co-activation matrix, primitive/key vectors

---

## 7. Plotting (`scripts/plot_training.py`)

- **Training curves**: loss, PPL (log), LR schedules, throughput (smoothed)
- **RL curves**: GRPO loss, trajectory spread (best/worst z-score), advantage
  std, logstd per group, neuromod grad norm
- **Memory health**: state norms, gate, primitive diversity, decay, firing
  rates, saturation, key/prim diversity, usage, plasticity rewires
- **Connectivity snapshot**: per-neuron bars, key heatmap, fan-in histogram
- **Neuron graph**: UMAP on primitives, edges by routing weight

---

## 8. Key Design Decisions

1. **Split-scan**: Lower layers build representation, memory injected mid-stack,
   upper layers learn to use memory. Memory never touches lower scan gradients
   (cc_signals are detached).

2. **Per-token sequential neuron dynamics**: Not a scan — sequential loop with
   sparse gather per token. Signals propagate hop-by-hop through the graph.

3. **Persistent memory across documents**: The memory graph never resets at doc
   boundaries. It learns document structure through CC signal changes. Only LM
   scan carries reset at doc boundaries.

4. **PCM conditions on full context**: Both encoding and prediction see (H, x),
   so surprise measures how unexpected the model's full representation is, not
   just the raw token identity.

5. **GRPO with spatial specificity**: Only K=96 of 1024 neurons get varied
   actions; non-K get zero delta. Credit assignment is spatially specific.

6. **Multi-chunk scoring horizon**: Score across all 4 collected chunks (64
   neuromod actions per trajectory) to capture long-range memory effects.

7. **Best-of-N state selection**: After scoring, the best trajectory's memory
   graph and upper scan carries become the real state. Free performance boost.

8. **Two-phase training**: Phase 1 trains LM + frozen memory by backprop.
   Phase 2 freezes LM, trains neuromod by GRPO to adapt memory at inference.
