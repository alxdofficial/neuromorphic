# Conv-Grid Modulator Redesign

Supersedes the modulator and cell-layout sections of `docs/design.md`. The rest
of that doc (memory lifecycle, surprise signal, mem_pred_loss, TBPTT, training
objective) continues to apply.

## TL;DR

| | Current (`main`) | This design |
|---|---|---|
| Physical cells (`NC_cells`) | 8 | **1** — single connectivity pool |
| Neurons per cell (`N`) | 32 | **256** — full N×N connectivity |
| Encoder ("logit head") | Per-cell MLP (1092 → 3072 → K), 39M params | **Conv stack over the N×N edge grid** |
| Encoder observation | Flattened scalars per cell | **Edge feature map** `[BS, N, N, ~52]` |
| Codebook | 512 × 64 | **Same** (or slightly bigger) |
| Decoder output | Dense `[NC, N²+N]` per event | **Low-rank factored** `U, V ∈ [N, r]` + `Δdecay ∈ [N]` |
| Policy per event | 8 categorical (one per cell) | **1 categorical** over K codes |
| Encoder params | 39M (scales with N²) | **~200K** (kernel weights, fixed-size) |
| Memory params total | ~42M | **~5M baseline** (budget freed to spend on decoder/codebook) |

The cell as a grid becomes load-bearing: we treat each lane's connectivity W
(plus hebbian, plus per-neuron features broadcast into the grid) as an N×N
feature map and run a conv-like kernel over it. The modulator observes local
windows of that map and produces both edge-level updates (ΔW entries) and
cell-level outputs (code logits, per-neuron decay).

## Motivation

Three observations from the verify_01 analysis forced this:

1. **Current encoder is pathologically allocated.** 94% of memory-module params
   live in the per-cell `[1092, 3072]` logit-head matrix, because the input
   includes `hebbian_flat = N²`. Growing N quadratically inflates the encoder.
2. **The 8-cell factored policy was justified only by phase-2 GRPO SNR.** With
   autoregressive GRPO (the future-work plan), rollouts diverge in token space
   and per-cell factoring is no longer load-bearing.
3. **Dense ΔW output is over-parameterized.** The current decoder goes through
   a 64-dim codebook then produces N² action entries, so it's already
   implicitly rank-≤64. Making that explicit (low-rank factored output) saves
   params and matches the true action manifold.

Collapsing the 8 cells into one N=256 connectivity pool, replacing the
dense-flatten MLP encoder with a conv over the edge grid, and factoring the
ΔW output all address the same underlying issue: **connectivity IS the
structure the modulator operates on, so the modulator architecture should
match that structure.**

## Memory state (per lane)

```
h        : [BS, N, D_n]          per-neuron hidden state
msg      : [BS, N, D_n]          per-neuron outgoing message
W        : [BS, N, N]             dense synaptic weights (any-to-any)
hebbian  : [BS, N, N]             EMA of pairwise msg co-activation
decay    : [BS, N]                per-neuron persistence gate in [0, 1]
```

At N=256, D_n=256:
- h, msg: 256·256·bf16 = 128 KB per sample per tensor
- W, hebbian: 256·256·bf16 = 128 KB per sample per tensor
- Total per-sample runtime state: ~0.5 MB

At BS=72: ~35 MB of runtime state. Fits easily.

## Cell / port layout

One connectivity pool, but the LM interface still wants `D = D_n · NC_pools`
dimensions flowing in and out. So we keep **virtual pools** purely for I/O,
not for connectivity:

```
N = 256 total neurons
NC_pools = 8                         (determined by D / D_n = 2048 / 256)
alpha    = 4 (input ports per pool, output ports per pool)

Role layout (indices into [0, N)):
  input ports:     [0, 32)            = 8 pools × 4 ports    (replicate H_mid slice)
  output ports:    [32, 64)           = 8 pools × 4 ports    (contribute to readout)
  internal:        [64, 256)          = 192 neurons
```

Inject: `H_mid[:, D_n*p : D_n*(p+1)]` is distributed across the 4 input-port
neurons of pool `p` (each input port sees the full D_n slice via per-cell
inject projection, unchanged from current).

Readout: for each pool p, sum the D_n vectors of its 4 output ports, divide by
√α, concat across pools → D-dim readout per sample per time step.

The critical change: **W is N×N dense**, not NC_pools separate N_per_cell × N_per_cell
matrices. Any neuron can connect to any other — ports can communicate directly
with internals across what used to be cell boundaries.

## What the modulator observes

For each grid position (i, j), per cell (NC=1), per time step at modulation
events:

| Channels | What | Source |
|---|---|---|
| 1 | W[i, j] | connectivity matrix, raw |
| 1 | hebbian[i, j] | correlation trace, raw |
| 1 | (W[i,j] − W[j,i]) | edge asymmetry |
| `d_proj` = 16 | h_proj[i] (broadcast down col) | receiver's state, compressed |
| `d_proj` = 16 | h_proj[j] (broadcast across row) | sender's state, compressed |
| `d_proj` = 16 | msg_emit_proj[i] (broadcast down col) | what i is sending, compressed |
| `d_proj` = 16 | msg_recv_proj[i] (broadcast down col) | what i is receiving (Σ_j W[i,j]·msg[j]), compressed |
| 1 | decay[i] (broadcast down col) | receiver's persistence |
| `role_dim` = 4 | role_emb[i] (broadcast) | port type: input / output / internal |
| `role_dim` = 4 | role_emb[j] (broadcast) | port type of sender |
| 2 | s_mem_live, s_mem_ema_fast (broadcast global) | surprise |

**Total channels**: `3 + 4·d_proj + 1 + 2·role_dim + 2 ≈ 78`.

Projection heads (new parameters):
- `W_h_proj`: `[D_n, d_proj] = [256, 16] = 4K params` — compresses per-neuron
  state for broadcasting.
- `W_msg_emit_proj`: `[D_n, d_proj]` — compresses per-neuron *outgoing*
  message.
- `W_msg_recv_proj`: `[D_n, d_proj]` — compresses per-neuron *aggregate
  incoming* message `received[i] = Σ_j W[i,j] · msg[j]`.
- `role_emb`: `[3, role_dim] = [3, 4] = 12 params` — learned one-hot for port
  type.

### In plain English

Within a single kernel window (say a 7×7 patch at some position), the
modulator sees, for each of the 7 receiver-side and 7 sender-side neurons
in that window:

- The neuron's projected hidden state.
- The neuron's projected outgoing message (what it's sending).
- The neuron's projected aggregate received message (what it's getting).
- Its role marker (input port / output port / internal).

Plus, for the 49 edges in the window:

- The raw W value (unprojected — full connectivity weight).
- The raw hebbian correlation value.
- The asymmetry (W[i,j] − W[j,i]).

Plus the global surprise signal broadcast to every position.

Across the full conv stack + global pool, the model integrates these local
views into a single cell-level feature that drives code-logit emission.

The projection is **crucial**. A naive broadcast of full D_n=256 vectors gives
~1028 channels, which at `[72, 256, 256, 1028]·bf16` is ~9.7 GB — doesn't fit.
Projection brings it to 52-64 channels, ~490 MB, comfortable.

### Why these observations

- **W and hebbian**: the current state of the thing we're about to edit, plus
  its running correlation partner.
- **Projected h / msg per neuron** (broadcast, NOT per-connection): the
  modulator can see "what each neuron is up to" without paying D_n per
  position. Projection is learned; msg at position (i, j) is the broadcast
  per-neuron value, not the weighted edge contribution `W[i,j] · msg[j]`.
  The per-edge information comes from W[i,j] + hebbian[i,j], which
  **already** cover structural weight and co-activation history — so
  per-connection message value would be largely redundant. If a future
  ablation shows the modulator can't see fine-grained "this edge is live
  right now" effects, add `W[i,j] · msg_norm[j]` as one extra scalar channel
  (cheap) before adding full-width per-edge message projections.
- **Role embedding**: the modulator can treat ports differently from internals
  (it needs to, since their semantics are different). Without this the conv
  would have to rediscover port roles from observation patterns.
- **Surprise**: unchanged from current design.

### Why these observations are NOT in the current design

- **Raw h / msg content**: current design is "rates + correlations only, no
  content peek." The new design flips this. Content at d_proj=16 compression
  is a much weaker information channel than full D_n, but it's non-trivial —
  we're betting that compressed content helps the modulator decide which code
  to pick.

## Modulator architecture (encoder)

The **encoder** — the piece that trains under phase-2 GRPO, and the part that
replaces the current `logit_w1, logit_w2` MLP — is the full stack from
observation tensor to code logits. Not just the conv layers; the conv + pool
+ logit head together.

```
             ┌─────────────── encoder (GRPO-trainable) ────────────────┐
observation  │                                                          │  code
tensor   ──► │  conv stack → pool → Linear logit head                   │──► logits
[N, N, C_in] │  [N,N,C_h]   [C_h]   C_h → K                             │    [BS, K]
             └──────────────────────────────────────────────────────────┘
                                                                             │
                                                            Gumbel / Categorical (sampling)
                                                                             │
                                                            codebook [K, D_code]
                                                                             │
                                                            decoder MLP
                                                                             │
                                                            U, V ∈ [N, r], Δdecay ∈ [N]
```

### Conv stack (perception over the edge grid)

```
INPUT: E ∈ [BS, N, N, C_in≈62]

Conv block × L_conv layers:
    E ← Conv2d(k=3, padding=1, in=C_h, out=C_h)(E)
    E ← GroupNorm(groups=8)(E)
    E ← GELU(E)

After conv stack: E ∈ [BS, N, N, C_h]
```

### Pooling → code logits

```
code_feat     = global_avg_pool_over_grid(E)         # [BS, C_h]
code_logits   = Linear(C_h → K)(code_feat)           # [BS, K]

# (optional for future per-neuron output heads, not used for codes:)
row_feat      = mean over j axis                     # [BS, N, C_h]  — per-receiver
col_feat      = mean over i axis                     # [BS, N, C_h]  — per-sender
```

Suggested hyperparameters (sized to hit the 100M total-param target):
- `C_h = 256` (conv hidden channels)
- `L_conv = 6`
- `kernel_size = 7` — deliberately "large" because at N=256 a small kernel
  sees a rounding-error fraction of the grid per step. 7×7 sees 49 neurons'
  features at once (7 receivers × 7 senders + 49 raw W and hebbian entries);
  6 layers of k=7 give ~37×37 effective receptive field before global pool.
- `groups = 32` for GroupNorm

Params for a 6-layer 7×7 conv at C_h=256, C_in=78:
- Layer 1 (C_in=78 → 256): 78·256·49 = 979K
- Layers 2–6 (256 → 256): 256·256·49 = 3.2M × 5 = 16.1M
- Total: **~5.4M params**

Compare to current per-cell logit head: 39.4M params. The conv is still
smaller than the old encoder, but deeper and more expressive per-param.

## Discrete bottleneck (unchanged semantics)

```
code_logits = [BS, K]

Phase 1 (Gumbel-softmax, differentiable):
    soft = gumbel_softmax(code_logits, τ, hard=True)  # [BS, K]
    emb  = soft @ codebook                             # [BS, D_code]

Phase 2 (hard categorical, GRPO):
    code = multinomial(softmax(code_logits), 1)       # [BS]
    emb  = codebook[code]                              # [BS, D_code]
    log_pi = log_softmax(code_logits).gather(code)    # [BS]
```

- `codebook`: `[K, D_code]` learned lookup, same structure as current.
- `K = 4096` (up from current 512) — bigger vocabulary of memory-update
  templates. Growing K is cheap: each extra code costs `D_code = 256` params.
- `D_code = 256` (up from current 64) — richer per-code intent vector;
  matters for downstream decoder capacity.

## Decoder: low-rank factored action

```
emb ∈ [BS, D_code]

action_head = MLP(emb):
    h1 = tanh(emb @ W_d1 + b_d1)                  # [BS, H_dec=256]
    raw = h1 @ W_d2 + b_d2                         # [BS, 2·N·r + N]

U         = raw[..., :N*r].reshape(BS, N, r)
V         = raw[..., N*r : 2*N*r].reshape(BS, N, r)
Δdecay    = raw[..., 2*N*r:]                       # [BS, N]

ΔW_raw    = U @ V.T                                # [BS, N, N]
ΔW_normed = F.rms_norm(ΔW_raw, normalized_shape=(N,))
```

Hyperparameters (sized for 100M budget):
- `r = 64` (rank of ΔW per event, up from 32)
- `H_dec = 2048` (3-layer MLP: D_code → H_dec → H_dec → flat output)

3-layer decoder shape:
- `W_d1`: 256·2048 = 524K
- `W_d2`: 2048·2048 = 4.2M
- `W_d3`: 2048·(2·256·64 + 256) = 2048·32896 = 67.4M — too big!

We rebalance to stay near 100M total. Two options to pick from:

**Option A (default, 48M decoder):** 2-layer decoder, H=1024, r=64.
- `W_d1`: 256·1024 = 262K
- `W_d2`: 1024·32896 = 33.7M
- Total decoder: **~34M params**

**Option B (slimmer decoder, smaller rank):** 3-layer, H=2048, r=32.
- `W_d1`: 256·2048 = 524K
- `W_d2`: 2048·2048 = 4.2M
- `W_d3`: 2048·16640 = 34.1M
- Total decoder: **~38.8M params**

Option A keeps rank=64 (more expressive per event, matches current implicit
rank-≤64); Option B gives more nonlinear capacity in the code→factor mapping
at the cost of halving the rank. Default to **Option A** — the rank is more
often the bottleneck than the decoder's nonlinearity.

At N=256, r=64: each modulation event's ΔW is a rank-64 perturbation of W.
Matches what the current design *implicitly* gets through the 64-dim
codebook, but cleanly decomposed and explicit.

## Applying the update

Unchanged EMA-blend semantics, but per-neuron (not per-cell) plasticity rates:

```
# Per-neuron learnable EMA rates (upgraded from per-cell):
W_gamma      = sigmoid(W_decay_logit)               # [N]
decay_gamma  = sigmoid(decay_gamma_logit)           # [N]
hebbian_gamma = sigmoid(hebbian_decay_logit)        # [N]

# Update W (EMA toward row-RMS-normed low-rank target):
W_new = (1 - W_gamma[None, :, None]) * W + W_gamma[None, :, None] * ΔW_normed

# Update decay:
target_decay = sigmoid(Δdecay)
decay_new    = (1 - decay_gamma[None, :]) * decay + decay_gamma[None, :] * target_decay
```

Per-neuron rate is a capacity bump (some neurons may want fast plasticity,
others slow). Cost: `3·N = 768` new scalar params vs. current `3·NC_cells = 24`.

## Memory step (mostly unchanged)

```
received = W @ msg                                  # [BS, N, D_n]
received[input_port_slots] += inject(H_mid)
h        = decay * h + (1 - decay) * state_mlp(received, h)
msg      = identity + msg_mlp(h)
readout  = pool_output_ports(msg) / sqrt(alpha)
hebbian  = (1 - γ_heb) * hebbian + γ_heb * (msg @ msgᵀ)
```

No NC dimension in these tensors. `W @ msg` is one matmul of shape
`[BS, N, N] @ [BS, N, D_n] → [BS, N, D_n]`. Cheaper overhead per step than the
current per-cell loop, slightly more compute total at N=256 vs NC·N=256.

## Cost accounting

Per modulation event (BS=72, N=256, C_h=64):

| Step | FLOPs | Notes |
|---|---:|---|
| Project h, msg | 72·256·(256·16·2) ≈ 150M | Linear proj |
| Build grid (concat + broadcast) | memory-bound, ~500 MB write | One-time per event |
| Conv layer × 4 | 72·65K·64·64·9 ≈ 170G total | Runs at ~0.7ms/layer on 4090 |
| Pool + code logit | 72·64·K ≈ 2M | Tiny |
| Decoder | 72·(64·256 + 256·16640) ≈ 300M | Once per event |
| ΔW construction | 72·256·32·256 ≈ 150M | U @ Vᵀ |

Wall time estimate: **~3 ms per modulation event** on 4090. At mod_interval=4
and T=128, that's 32 events × 3ms = **96 ms per segment**. Comparable to
current, with richer observation and larger cell.

Per step (non-modulation):

| Step | FLOPs | vs current |
|---|---:|---|
| `W @ msg` | 72·256·256·256 = 1.2G | vs 72·8·32·32·256 = 150M → **8× more** |
| State MLP | 72·256·(256·256·2) = 2.4G | vs 72·256·… = same (N_total same) |
| Msg MLP | 2.4G | same |
| Hebbian `msg @ msgᵀ` | 1.2G | vs 150M → 8× |
| Readout | tiny | tiny |

Per-step compute ~7G vs current ~5G (about 1.4× more), consistent with having
an 8× larger W matrix but the same total neuron count.

## Training plan

Unchanged from current two-phase plan (bootstrap → cycles of phase-1 + phase-2)
structurally. What changes:

- Phase 1 (Gumbel): single categorical per event, not NC. Gradient path is
  `CE → upper_scan → readout → memory dynamics → modulator action → decoder
  → codebook → Gumbel soft → conv encoder → node projections + edge features`.
- Phase 1 cycle: freeze codebook + decoder, train everything else. Same as
  before.
- Phase 2 (GRPO): future work. When re-introduced with autoregressive
  sampling on a pretrained LM, only the conv encoder + code-logit head train.
  Fewer log_pi per event (1 vs 8), but autoregressive rollout divergence
  restores the gradient signal that the teacher-forced setup killed.

Two-phase curriculum is same: bootstrap ~500M tokens, cycles ~50M each.

## What stays the same

- `MemoryGraph.forward_segment` signature and return values
- `Model.forward_chunk` flow (lower scan → memory → upper scan)
- `H_mid.detach()` boundary: memory doesn't see LM-input-path gradient
- Weight-tied `lm.mem_head_logits` for mem_pred_loss and surprise
- TBPTT detach semantics, modulation_interval, tbptt_block
- mem_scale interface on the LM side
- Dead-code reset during bootstrap
- Dataloader, tokenization, eval harness

## What's gone

- Per-cell logit head with mod_in = 2N + 2 + 2 + N² observation layout
- 8 independent code choices per event (replaced by 1)
- Per-cell plasticity rates (`W_decay_logit`, `decay_gamma_logit`,
  `hebbian_decay_logit` move to per-neuron)
- `neurons_per_cell` as an architectural knob (replaced by `N_total`)
- `N_cells` > 1 connectivity (cells become virtual port pools)

## Parameter budget (100M target total)

Current model: ~109M (LM ~52M + Memory ~42M).
Target: ~100M (LM ~52M + Memory ~48M).

Memory allocation under this design:

| Component | Params |
|---|---:|
| Projection heads (h_proj, msg_emit_proj, msg_recv_proj) | 12K |
| Role embeddings | tiny |
| Conv stack (6 layers, C_h=256, k=7) | 5.4M |
| Code logit head (C_h=256 → K=4096) | 1.0M |
| Codebook (K=4096 × D_code=256) | 1.0M |
| Decoder (Option A: 2-layer, H=1024, r=64) | 34.0M |
| `inject_w, inject_b` (8 pools × α×D_n×D_n) | 2.1M |
| `neuron_id` (N=256 × D_n=256) | 65K |
| State/msg MLPs | 260K |
| Per-neuron plasticity logits | 1K |
| **Memory total** | **~48M** |

LM (unchanged): 52M.

**Grand total: ~100M. ✓**

Where the params went compared to the old 42M:
- **Encoder**: 39.4M → 6.4M (conv + logit head). The giant MLP on flattened
  N² is gone; the conv is much smaller but deeper and spatially structured.
- **Decoder**: 143K → 34M. This is the big shift. We're paying for rich
  action synthesis — 4096 codes, each mapped to a rank-64 ΔW perturbation
  via a 1024-hidden MLP. The action manifold is where the modulator's
  expressiveness actually matters.
- **Codebook**: 33K → 1M. Bigger vocabulary of memory-update templates.
- **Conv stack**: new, 5.4M. Moderate compared to the decoder.

## Comparison in one table

| Axis | Current | This design |
|---|---|---|
| Modulator encoder | 39M params, MLP on flattened N² | 150K–600K params, 2D conv on edge grid |
| Observation content | Rates + correlations only | + compressed node content + edge asymmetry |
| Action space rank per event | ≤64 implicit via codebook | 32 explicit via factored U, V |
| Policy structure | 8 independent codes per event | 1 code per event |
| Neurons per cell | 32 | 256 |
| Cells | 8 | 1 |
| Total neurons | 256 | 256 |
| W matrices | 8 × 32² = 8192 entries | 1 × 256² = 65K entries (8× more) |
| Connectivity | within cells only | global across whole cell |

## Open questions

1. **Content vs content-free observation.** The compressed h/msg broadcast is
   a departure from "statistics only" — do we regress on the biologically
   principled stance, or keep it? Worth ablating.
2. **`r` (action rank).** 32 vs 64 vs 128. Start at 32; ablate if decoder
   capacity feels tight.
3. **Depthwise-separable conv vs dense conv.** Dense is simpler and fits;
   depthwise is faster but slightly more complex. Start dense.
4. **Kernel size.** 3×3 gives 3-neuron receptive field per layer; 5×5 halves
   needed depth to see globally. Start 3×3.
5. **One code per event vs a few (H=2 or 4).** Even without the 8-cell
   factoring, having a handful of parallel codes could matter for GRPO SNR.
   Deferred to post-phase-1 evaluation.
6. **Layer norm placement.** Between convs (GroupNorm) and around pooling
   heads. Pay attention if training instability appears.
7. **Role embeddings vs positional.** Role (port/internal) is the invariance
   we care about; we skip positional (neurons don't have a "position"
   meaning). Confirm by ablation.
8. **Should conv be replaced by attention-with-edge-bias?** Permutation-
   equivariant, but less hardware-optimal. If conv training is unstable, try
   attention as a fallback.

## References and related work

- Mordvintsev et al., "Growing Neural Cellular Automata" (2020) — the literal
  2D-conv-over-grid-state idea, applied to image synthesis but architecturally
  parallel.
- Message-passing neural networks (Gilmer et al. 2017) — the general framework
  for node + edge updates; conv-over-edge-grid is a specific instance when
  edges form a dense pairwise set.
- Transformer with edge biases (Graphormer, Dwivedi & Bresson 2020) — the
  permutation-equivariant fallback if conv misbehaves.
