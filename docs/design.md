# Neuromorphic Memory Graph — Current Design

Branch: `main`. This document reflects the architecture as of
April 2026. Older planning docs in this folder describe earlier iterations
(conv-grid modulator, shared-weights attention modulator) — those are kept
for history but are **superseded** by this file for current state.

## Goals

1. Language-model augmented with a plastic **memory graph** that learns what
   to store and when to store it, end-to-end via backprop. No RAG, no memory
   documents, no prompt engineering.
2. Biologically-inspired: neurons, messages, synapses (`W`), Hebbian traces,
   neuromodulation. Multi-timescale: fast integration, event-driven spikes,
   slow plasticity.
3. Compatible with attaching to a pretrained LM (e.g. Llama) and training via
   next-token prediction or GRPO. Autoregressive at inference — no time-dim
   parallelization tricks.
4. GPU-friendly: hot paths are bmm / einsum. Per-token loop is fused in
   Triton on CUDA bf16 (≈4× the pure-PyTorch per-token cost).

## Architecture overview

```
tokens → embedding → lower LM scan → H_mid [BS, T, D]
                                       │
                                       ▼
        ┌─────────── Memory graph per-token loop ───────────────┐
        │                                                         │
        │  FAST CLOCK (every token):                              │
        │    received = W @ msg_prev + inject(H_mid_t)           │
        │    h = tanh(decay · h_prev + (1−decay) · received)     │
        │    readout_t = pool_output_ports(h)                    │
        │                                                         │
        │  EVENT CLOCK (every msg_interval = 4 tokens):          │
        │    msg = tanh(msg_MLP(h)) + neuron_id                  │
        │    hebbian = (1−γ_h)·hebbian + γ_h·(msg · msgᵀ)        │
        │                                                         │
        │  SLOW CLOCK (every modulation_interval = 16 tokens):   │
        │    logits  = modulator(h, msg, received,               │
        │                         W, hebbian, decay,              │
        │                         s_live, s_ema, role_id)        │
        │    code    = sample(logits)          per-cell          │
        │    emb     = codebook[code]                            │
        │    ΔW, Δdec = decoder(emb)            per-cell         │
        │    W      = (1−γ_W)·W     + γ_W·ΔW                    │
        │    decay  = (1−γ_d)·decay + γ_d·σ(Δdec)                │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                   H_enriched = H_mid + mem_scale · readouts
                                       │
                                       ▼
                               upper LM scan
                                       │
                                       ▼
                                   LM head → token logits
```

## State tensors (lifelong, per batch element)

```
h        : [BS, NC=8, N=32, D_n=256]   membrane potential
msg      : [BS, NC, N, D_n]            spike output (event-driven update)
W        : [BS, NC, N, N]             per-cell plastic weights (block-diagonal)
hebbian  : [BS, NC, N, N]             per-cell co-activation trace
decay    : [BS, NC, N]                 per-neuron leak rate in [0, 1]
```

All state persists across training segments (lifelong). Detached at
`tbptt_block` (=16) boundaries so TBPTT doesn't grow unbounded.

`W` is block-diagonal: there is NO synapse between a neuron in cell A and a
neuron in cell B. Cells communicate ONLY via the LM readout/inject
interface.

## Neurons and cells

- **Cell**: `N = 32` neurons sharing a dense 32×32 `W`. Within a cell, all-
  to-all connectivity.
- **Graph**: `NC = 8` cells, total 256 neurons. Not interconnected at the `W`
  level — see inter-cell communication below.
- **Per-cell role layout** (fixed local indices within each cell):
  ```
  [0, 1, 2, 3]     input ports   (receive LM broadcast via inject)
  [4, 5, 6, 7]     output ports  (pooled into LM readout)
  [8, ..., 31]     internal      (do the interesting work)
  ```
  `alpha = 4` ports per direction.

## Per-token hot path (Triton-fused on CUDA)

Implemented in `src/model/triton_memory_step.py`.

```python
received = W @ msg_prev + inject_dense
h        = tanh(decay * h_prev + (1 - decay) * received)
readout  = Σ_{n ∈ output_ports} h[n] · α^(−1/2)
```

- `W @ msg`: batched matmul `[BS, NC, N, N] @ [BS, NC, N, D_n]`.
- `inject(H_mid_t)`: per-cell projection. H_mid split into NC_pools slices of
  `D_n`; each slice passes through per-cell `inject_w [NC, α·D_n, D_n]`
  → reshaped to `[BS, NC, α, D_n]` → scatter-added into `received` at
  the cell-local `input_port_idx` positions.
- LIF integration: leaky-integrate (decay-weighted mix of previous h and
  incoming received) then tanh as soft spike threshold.
- Readout: weighted sum of output-port neurons (via a pre-built `out_port_mask`
  buffer), scaled by `α^−0.5`.

Everything except the pre-scatter of `inject_dense` runs in a single Triton
kernel, one program per `(batch, cell)` pair. State tiles fit in SRAM. The
forward kernel is 4.2× faster than the equivalent 5-op PyTorch path per-token;
the backward uses analytical PyTorch ops in a `torch.autograd.Function`.

Fallback: CPU / non-bf16 / non-CUDA paths use `fused_memory_step_torch`
(same math, no Triton). Used by the test suite.

## Event clock — spike emission (every `msg_interval = 4` tokens)

```python
msg    = tanh(msg_MLP(h)) + neuron_id
hebbian = (1 - γ_h) · hebbian + γ_h · (msg · msgᵀ)
```

- `msg_MLP` is a shared 2-layer MLP: `D_n → Hm=256 → D_n`.
- `neuron_id [NC, N, D_n]` is a learned per-neuron identity vector — breaks
  symmetry so different-position neurons emit distinguishable spike patterns.
- Hebbian is a rolling EMA of intra-cell co-activation. `γ_h = 0.97·σ(hebbian_decay_logit)`
  per neuron.

Between msg events, the `msg` tensor is held constant. Message passing
(`W @ msg`) still runs every token, using the last-emitted msg.

## Slow clock — neuromodulator (every `modulation_interval = 16` tokens)

Implemented in `src/model/attention_modulator.py`.

### What the modulator sees (per cell, NC=8 run in parallel)

**Per-node features** (each neuron contributes one token of dim `F=128`):
```
h[n]        [D_n=256]  →  h_proj          (shared Linear 256→24)
msg[n]      [D_n=256]  →  msg_emit_proj   (shared 256→24)
received[n] [D_n=256]  →  msg_recv_proj   (shared 256→24)
||h[n]||                [scalar]
||msg[n]||              [scalar]
decay[n]                [scalar]
role_id[n]              [embedding → 4-dim]
cell_emb[c]             [d_cell=16 — per-cell identity, broadcast over N]
s_live                  [scalar — current-token CE surprise]
s_ema                   [scalar — EMA of surprise]
```
→ concat (1+1+1+3·24+4+16+2 = 97 dims) → shared 2-layer MLP → `F=128`-dim
tokens, one per neuron.

**Per-edge features** (each pair (i, j) in a cell):
```
W[i, j]
hebbian[i, j]
(W − Wᵀ)[i, j]
```
→ shared edge-bias MLP (3 → 8 → H=8) → 8 scalar biases per edge, one per
attention head.

### What the modulator computes

For each cell (NC=8 batched as extra dim):
1. 3 layers of multi-head attention over the 32 per-neuron tokens.
   Edge biases are added to attention scores BEFORE softmax (like AliBi / T5
   / Graphormer). FFN mult = 4 (so FFN hidden = 4·F = 512).
2. Mean-pool tokens per cell → `[BS, NC, F=128]`.
3. Logit head: `F=128 → K=2048` → **logits `[BS, NC, K=2048]`**
   over discrete plasticity codes.

**Weights are SHARED across the NC=8 cells** (attention, FFN, edge-bias
MLP, logit head, token projections — all shared). The only per-cell
parameter is `cell_emb [NC, d_cell=16]`, which is (1) concatenated into
each per-neuron token feature and (2) concatenated into the decoder's
input alongside the sampled code embedding. This FiLM-style conditioning
gives each cell an identity signal at both the observation and action
stages without duplicating trunk weights — keeps compiled matmuls fat
(better tensor-core utilization) and param count modest.

### What the modulator writes

Per cell, independently:
1. Sample a code `k` from the categorical over `K=2048`:
   - Training phase 1: Gumbel-softmax straight-through
   - Eval / phase 2 rollout: hard Categorical with `log π` for GRPO
2. Look up `emb = codebook[k]`. Codebook `[K=2048, D_code=128]` is SHARED
   across cells — common plasticity vocabulary.
3. Shared-trunk decoder MLP (`D_code=128 + d_cell=16 → 512 → N²+N=1056`)
   emits, conditioned on the per-cell `cell_emb`:
   - `ΔW [N, N]` — zero-diagonal, RMS-normed along rows
   - `Δdecay [N]` — sigmoid → target in [0, 1]
4. γ-clamped EMA blend (γ_max = 0.97 for bf16 safety):
   ```
   W_new     = (1 − γ_W) · W     + γ_W · ΔW
   decay_new = (1 − γ_d) · decay + γ_d · σ(Δdecay)
   ```
   where `γ_W, γ_d ∈ [NC, N]` are learned per-neuron plasticity rates
   (`gamma_max · σ(W_decay_logit)` and similarly for decay).

The decoder's output layer is zero-init (weight and bias), so ΔW and
Δdecay are exactly zero at step 0 and the EMA is a no-op until the
decoder learns to write. This costs one update step of dead-zone
gradient through the decoder trunk, codebook, and modulator — accepted
because any nonzero init gets amplified by the RMS-norm on ΔW and
undermines the no-op starting point. The penultimate decoder layer is
Xavier-init as usual, so it wakes up on step 1.

**The modulator does NOT directly adjust `h`, `msg`, `hebbian`, `neuron_id`,
or `inject_w`.** Those either evolve by their own dynamics (h, msg, hebbian)
or are learned via backprop from the mem_pred / CE losses (neuron_id,
inject_w).

## Inter-cell communication — via the LM

Cells don't share `W` edges. Information flows between cells only through
the LM's scan recurrence:

```
cell c readout_c = pool_output_ports(h[c])              [BS, D_n=256]
all readouts     = concat(readout_0, ..., readout_7)    [BS, D=2048]
H_enriched       = H_mid + mem_scale · readouts         (learnable per-dim scale)
H_upper          = LM_upper_scan(H_enriched)
                                                        next token t+1
H_mid[t+1]       = LM_lower_scan(...)                   [BS, T, D]
cell c inject    = inject_w[c] · H_mid[t+1][c·D_n:(c+1)·D_n]
                   → scatter into input ports 0..3 of cell c
```

The LM IS the inter-cell router. This mirrors cortical architecture: a
cortical area's neurons are densely interconnected locally; areas communicate
through long-range projections via hub structures (thalamus, etc.).

## Training objective

Total loss:
```
loss = ce_loss + mem_pred_weight · mem_pred_loss
```
- `ce_loss`: standard next-token CE at the LM head, averaged over non-EOT
  positions.
- `mem_pred_loss`: per-segment CE of `mem_head(shifted_readouts)` against
  `input_ids`. The mem head is weight-tied to the LM head; training it pushes
  the memory graph to carry token-prediction-useful information.
- `mem_pred_weight = 0.1` (config).

Gradient flows:
- CE flows through the LM and through `mem_scale · readouts` into the memory
  graph's state, plasticity gates, and modulator.
- mem_pred_loss flows directly through readouts into memory state + modulator.

### TBPTT

`tbptt_block = 16`: memory state (`h, msg, W, decay, hebbian, prev_readout`)
is detached at every 16-token boundary. This caps the backward graph depth
inside a segment of `T=128` tokens so VRAM stays bounded.

## Config defaults (tier_a)

| Field | Value | Notes |
|---|---:|---|
| D | 2048 | LM hidden |
| D_embed | 768 | token embedding dim |
| L_total | 4 | LM scan layers |
| scan_split_at | 2 | memory graph interposed between scan layers 2 and 3 |
| d_inner | 1200 | LM scan inner dim |
| vocab_size | 32000 | |
| N_cells | 8 | NC |
| neurons_per_cell | 32 | N |
| D_n | 256 | per-neuron state dim |
| alpha | 4 | input / output ports per cell |
| msg_interval | 4 | msg emission + hebbian event rate (tokens) |
| modulation_interval | 16 | neuromodulator fire rate (tokens) |
| msg_mlp_hidden | 256 | |
| d_proj | 24 | per-node projection dim in modulator |
| attn_token_dim | 128 | F — modulator token width |
| attn_n_heads | 8 | |
| attn_n_layers | 3 | |
| attn_ffn_mult | 4 | |
| num_codes | 2048 | K — codebook size |
| code_dim | 128 | D_code |
| decoder_hidden | 512 | shared decoder hidden dim |
| gamma_max | 0.97 | bf16-safe clamp on plasticity rates |
| T | 128 | tokens per segment |
| tbptt_block | 16 | TBPTT detach interval |
| checkpoint_memory | False | activation checkpointing disabled by default |

## Param count and performance (RTX 4090, bf16 autocast)

```
Total model params : 71.17 M
  LM               : 67.08 M
  Memory graph     :  4.09 M
    inject_w + b   :  2.11 M   per-cell LM→ports projection
    modulator      :  0.91 M   SHARED attention trunk (F=128, 3 layers, H=8)
    decoder        :  0.62 M   SHARED code → plasticity MLP (conditioned on cell_emb)
    codebook       :  0.26 M   K=2048, D_code=128 (shared across cells)
    state/bookkeeping : 0.20 M   neuron_id, msg MLP, plasticity gate logits

Throughput (BS=64, T=128, tbptt=16) : ~68-70K tok/s (Triton-fused per-token)
VRAM at BS=64                       : ~12 GB
```

Compared to the pre-redesign baseline on the same branch (37.8K tok/s at BS=64,
shared-weights modulator, full 2-layer state MLP, modulator every 4 tokens),
the current architecture is **~1.5× faster** AND carries a ~4× bigger
per-cell modulator and decoder.

## Biological reading

- **h** ~ membrane potential: leaky-integrates synaptic input, saturates at
  ±1 (tanh as spike threshold surrogate).
- **msg** ~ spike output: sparse in time (event-driven every 4 tokens),
  carries a D_n-dim "spike pattern" to connected neurons via W.
- **W** ~ synaptic weights: plastic, modulator-written at slow timescale.
- **hebbian** ~ co-activation trace: STDP-like rolling average.
- **decay** ~ per-neuron membrane time constant.
- **modulator** ~ neuromodulatory nucleus (VTA / basal forebrain): integrates
  top-down signals (surprise, current state) and releases a discrete
  "neurochemical" (code) that triggers a cell-specific plasticity program
  (decoder).
- **codebook** ~ the neuromodulator's vocabulary. Shared across brain areas
  (dopamine is dopamine everywhere), interpreted differently per area (per-
  cell decoder ≈ area-specific receptor profiles).
- **Cells** ~ cortical areas / columns. Densely interconnected internally;
  communicate externally through long-range projections (LM scan).
- **Multi-timescale**: fast membrane dynamics (ms), event-driven spikes
  (10s of ms), slow plasticity (minutes-hours). Our per-token / per-4 / per-16
  schedule preserves this hierarchy.

## Pointers to code

| Component | File / location |
|---|---|
| Top-level Model | `src/model/model.py` |
| Config | `src/model/config.py` |
| LM (split scan + mem head) | `src/model/lm.py`, `src/model/scan.py` |
| Memory graph | `src/model/memory.py` |
| Per-cell attention modulator | `src/model/attention_modulator.py:AttentionModulator` |
| Per-cell decoder | `src/model/attention_modulator.py:DirectDecoder` |
| Port layout | `src/model/attention_modulator.py:port_layout` |
| Codebook + sampling | `src/model/discrete_policy.py` |
| Triton fused per-token kernel | `src/model/triton_memory_step.py` |
| Smoke tests | `tests/test_smoke.py` |
| Triton kernel correctness + speed | `scripts/test_triton_step.py` |
| Throughput benchmark | `scripts/bench_tbptt_single.py` |
| Profile script | `scripts/profile_attn_neuromod.py` |
