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
| 1 | W[i, j] | connectivity matrix |
| 1 | hebbian[i, j] | correlation trace |
| 1 | (W[i,j] − W[j,i]) | edge asymmetry |
| `d_proj` = 16 | h_proj[i] (broadcast down col) | receiver's state, compressed |
| `d_proj` = 16 | h_proj[j] (broadcast across row) | sender's state, compressed |
| `d_proj` = 16 | msg_proj[i] OR msg_proj[j] (broadcast) | message (pick one; emit[j] preferred) |
| 1 | decay[i] (broadcast down col) | receiver's persistence |
| `role_dim` = 4 | role_emb[i] (broadcast) | port type: input / output / internal |
| `role_dim` = 4 | role_emb[j] (broadcast) | port type of sender |
| 2 | s_mem_live, s_mem_ema_fast (broadcast global) | surprise |

**Total channels**: `3 + 3·d_proj + 1 + 2·role_dim + 2 ≈ 62`.

Projection heads (new parameters):
- `W_h_proj`: `[D_n, d_proj] = [256, 16] = 4K params` — compresses per-neuron
  state for broadcasting.
- `W_msg_proj`: `[D_n, d_proj]` — same for messages.
- `role_emb`: `[3, role_dim] = [3, 4] = 12 params` — learned one-hot for port
  type.

The projection is **crucial**. A naive broadcast of full D_n=256 vectors gives
~1028 channels, which at `[72, 256, 256, 1028]·bf16` is ~9.7 GB — doesn't fit.
Projection brings it to 52-64 channels, ~490 MB, comfortable.

### Why these observations

- **W and hebbian**: the current state of the thing we're about to edit, plus
  its running correlation partner.
- **Projected h / msg per neuron**: the modulator can see "what each neuron is
  up to" without paying D_n per position. Projection is learned.
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

```
INPUT: E ∈ [BS, N, N, C_in≈62]

Conv block × L_conv layers:
    E ← Conv2d(k=3, padding=1, in=C_h, out=C_h)(E)
    E ← GroupNorm(groups=8)(E)
    E ← GELU(E)

After conv stack: E ∈ [BS, N, N, C_h]

Pooling heads:
    code_feat     = global_avg_pool(E)            # [BS, C_h]
    code_logits   = Linear(C_h → K)(code_feat)    # [BS, K]

    row_feat      = mean over j axis               # [BS, N, C_h]  — per-receiver
    col_feat      = mean over i axis               # [BS, N, C_h]  — per-sender
```

Suggested hyperparameters:
- `C_h = 64` (conv hidden channels)
- `L_conv = 4`
- `kernel_size = 3` (depthwise separable also viable)
- `groups = 8` for GroupNorm

Params for a 4-layer 3×3 conv at C_h=64:
- Layer 1 (C_in=62 → 64): 62·64·9 = 36K
- Layers 2–4 (64 → 64): 64·64·9 = 37K × 3 = 111K
- Total: **~150K params**

Compare to current per-cell logit head: 39.4M params.

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

- `codebook`: `[K, D_code]` learned lookup, same as current.
- `K = 512` initially (same as current). Can grow if the freed param budget
  is spent here.
- `D_code = 64` initially.

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

Hyperparameters:
- `r = 32` (rank of ΔW per event)
- `H_dec = 256` (decoder hidden; bumped from 128 since we have budget)

Decoder param count:
- `W_d1`: 64·256 = 16K
- `W_d2`: 256·(2·256·32 + 256) = 256·16640 = 4.26M
- Total: **~4.3M params**

At N=256, r=32: each modulation event's ΔW is a rank-32 perturbation of W.
That's 32 "directions of change" — substantially more expressive than the
current implicit rank-≤64 under per-cell factoring (since the current design
can't share rank budget across cells).

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

## Parameter budget (spending the freed capacity)

Current memory total: ~42M (logit heads dominated).

New design baseline (conservative):

| Component | Params |
|---|---:|
| Projection heads (h_proj, msg_proj) | 8K |
| Role embeddings | tiny |
| Conv stack (4 layers, C_h=64) | 150K |
| Code logit head (pool → K) | 33K |
| Codebook (K=512, D_code=64) | 33K |
| Decoder (D_code=64 → 256 → 2Nr+N) | 4.3M |
| `inject_w, inject_b` (8 pools × α×D_n×D_n) | 2.1M |
| `neuron_id` (N=256 × D_n=256) | 65K |
| State/msg MLPs | 260K |
| Per-neuron plasticity logits | 1K |
| **Memory total baseline** | **~7M** |

That leaves **~35M of freed budget** to spend. Recommended allocation:
- Bigger decoder hidden: `H_dec = 256 → 1024` → decoder grows to ~17M (better
  expressiveness per code).
- More codes: `K = 512 → 2048` → richer vocabulary (codebook 33K → 260K).
- Wider conv: `C_h = 64 → 128` → encoder grows to ~600K.

Target post-allocation memory total: **~25M** (about 60% of current), with
capacity shifted from "perceive observations via giant MLP" to "emit
expressive actions via rich decoder."

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
