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
| Encoder observation | Flattened scalars per cell | **Edge feature map** `[BS, N, N, ~78]` |
| Codebook | 512 × 64 | **4096 × 384** — bigger vocabulary, richer per-code intent vectors |
| Decoder | MLP, dense `[NC, N²+N]` output | **Conv-transpose stack**: code_emb → [4,4,C] seed → upsample to `[N,N,1]` ΔW (full-rank) + row-pooled Δdecay |
| Action rank | ≤64 implicit (via 64-dim codebook) | **Full rank** (no factoring constraint) |
| Policy per event | 8 categorical (one per cell) | **1 categorical** over K=4096 codes |
| Encoder params | 39M (scales with N²) | **~3.4M** (6 conv layers, kernel 7, C_h=192) |
| Decoder params | 143K (tiny MLP) | **~2.5M** (conv-transpose stack) |
| Memory params total | ~42M | **~10M** (lean; bump if training hits capacity ceiling) |

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
3. **Dense MLP decoder is over-parameterized.** The current decoder is a tiny
   MLP that emits `N² + N = 1056` scalars at once. With NC=1 and N=256, that
   becomes a 65K-entry flat output — the MLP's final layer would be ~34M
   params. A conv-transpose stack that upsamples a small spatial seed to
   `[N, N, 1]` is ~13× smaller and produces a full-rank ΔW with spatial
   structure that matches the encoder's.

Collapsing the 8 cells into one N=256 connectivity pool, replacing the
dense-flatten MLP encoder with a conv over the edge grid, and replacing the
MLP decoder with a conv-transpose generator all address the same underlying
issue: **connectivity IS the structure the modulator operates on, so the
modulator architecture should match that structure.**

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

Suggested hyperparameters (sized on merit, not to hit a param budget):
- `C_h = 192` (conv hidden channels)
- `L_conv = 6`
- `kernel_size = 7` — deliberately "large" because at N=256 a small kernel
  sees a rounding-error fraction of the grid per step. 7×7 sees 49 neurons'
  features at once (7 receivers × 7 senders + 49 raw W and hebbian entries);
  6 layers of k=7 give ~37×37 effective receptive field before global pool.
- `conv_groups = 32` for GroupNorm (shared between encoder and decoder)
- `dropout = 0.1` between conv layers (matches LM dropout)

**Pre-norm with residual connections** (stable at depth):

```python
# Per conv block (applied 6 times):
h_in = x
x = group_norm(x)
x = conv2d(x, k=7, padding=3)
x = gelu(x)
x = dropout(x)
x = x + h_in        # residual (same channel dim after layer 1)
```

The first layer projects `C_in=78 → C_h=192` and doesn't take a residual
(dim mismatch). Layers 2–6 are residual blocks.

Params for a 6-layer 7×7 conv at C_h=192, C_in=78:
- Layer 1 (C_in=78 → 192): 78·192·49 = 734K
- Layers 2–6 (192 → 192): 192·192·49 = 1.8M × 5 = 9.0M
- Total: **~9.7M** (or ~1–2M with depthwise-separable; see §Open Questions)

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
  templates. Growing K is cheap: each extra code costs `D_code` params.
- `D_code = 384` (up from current 64) — wider per-code intent vector.
  Decoupled from `C_h=192`: the encoder's pooled feature projects into the
  K-way logit head (unchanged), while the codebook lives in its own wider
  space so individual codes have room to be meaningfully distinct.

## Decoder: conv-transpose generator (full-rank ΔW)

The decoder upsamples the code embedding from a small spatial seed to the full
N×N grid, producing a dense (full-rank) ΔW map plus a per-neuron Δdecay head.

```
emb ∈ [BS, D_code=384]
    │
    ▼ Linear + reshape
[BS, 256, 4, 4]                    ← small spatial seed (wider than D_code's worth)
    │
    ▼ 6 × upsample stages (each scale_factor=2)
    │   Per stage (pre-norm with residual over upsampled input):
    │       y = F.interpolate(bilinear, scale_factor=2)(x)
    │       h = y + Conv2d(k=3)(GroupNorm(GELU(y)))       # residual
    │       project channels if in!=out via 1×1 conv on y before the add
[BS, 128, 8, 8]
[BS, 96, 16, 16]
[BS, 64, 32, 32]
[BS, 48, 64, 64]
[BS, 32, 128, 128]
[BS, 32, 256, 256]                 ← full N×N, per-edge feature vector
    │
    ├─► diagonal mask (ΔW[i,i] = 0)
    ├─► 1×1 Conv2d(32 → 1, zero-init) → ΔW_raw [BS, N, N]
    │   └─► F.rms_norm over row axis → ΔW_normed
    │
    └─► row-pool [N, 32] → small MLP → Δdecay_raw [BS, N]
```

### Why resize+conv instead of native ConvTranspose2d

`F.interpolate(mode='bilinear') + Conv2d` avoids the checkerboard artifacts
that native `ConvTranspose2d` with stride-2 is famous for. The learnable
refinement happens in the plain `Conv2d`, and the upsampling itself is a
fixed, artifact-free bilinear interpolation. Same representational power,
cleaner training dynamics.

### Zero-init the final 1×1 conv

The final `Conv2d(32 → 1)` that emits `ΔW_raw` has its weights zero-init'd.
At init, the decoder produces `ΔW_raw ≈ 0` regardless of code. After the
EMA blend with `γ_W`, this means `W_new ≈ W_old` at the start of training —
the modulator can't perturb the memory into a chaotic regime before it's
learned anything useful.

### Full-rank output, no factoring

Unlike the old design (which produced implicit-rank-≤64 via a 64-dim
codebook) or an earlier draft of this one (which explicitly produced
rank-r factors `U, V ∈ [N, r]`), the conv-transpose decoder outputs each
`ΔW[i, j]` independently. The ΔW map can be full-rank if the task wants
that; or it can be low-rank by learned choice. No hyperparameter commits
the model to a rank bound.

### Post-processing (mask / activation / norm)

Applied on the N×N map before feeding into the EMA blend:

| Step | What it does |
|---|---|
| Diagonal mask | `ΔW[i, i] = 0` — no self-weight changes. Matches the init convention (diag zeroed). |
| Role mask (optional) | Zero out edges that shouldn't exist structurally (e.g., within-pool input→input). Skip for v1. |
| Row RMSNorm | Bounds per-row magnitude (already in current design, retained). |
| EMA blend with `W_gamma` | `W_new = (1 - γ_W) · W + γ_W · ΔW_normed`. Same as current. |

### Hyperparameters

- Seed spatial: `[4, 4]`
- Seed channels: `256` (decoupled from D_code — gives the decoder extra
  bandwidth to unpack the richer code embedding)
- Upsample stages: 6 (to reach N=256 from 4)
- Upsample channels (in, out per stage): 256→128, 128→96, 96→64, 64→48, 48→32, 32→32
- Upsample conv kernel: 3×3, padding=1
- Norm: GroupNorm(groups=8), Activation: GELU

### Param count

- Initial Linear (D_code=384 → 4·4·256 = 4096): 384 · 4096 = 1.57M
- 6 upsample conv stages:
  - Stage 1 (256 → 128, k=3): 256·128·9 = 295K
  - Stage 2 (128 → 96, k=3): 128·96·9 = 111K
  - Stage 3 (96 → 64, k=3): 96·64·9 = 55K
  - Stage 4 (64 → 48, k=3): 64·48·9 = 28K
  - Stage 5 (48 → 32, k=3): 48·32·9 = 14K
  - Stage 6 (32 → 32, k=3): 32·32·9 = 9K
  - Subtotal: ~512K
- Final 1×1 Conv (32 → 1): 32 (tiny, zero-init)
- Δdecay head (row-pool [N, 32] → MLP 32 → 64 → 1): ~2K
- **Decoder total: ~2.1M params**

(Still ~16× smaller than the MLP decoder it replaces. Full-rank output. No
rank hyperparameter. The bulk of the growth from the earlier spec is the
wider init Linear, which is what makes the larger D_code actually useful.)

## Applying the update

Per-neuron plasticity rates, **clamped to a bf16-safe range** to avoid
precision loss in the EMA blend when `(1-γ)` gets tiny:

```
GAMMA_MAX = 0.97                                    # bf16-safe ceiling

W_gamma       = GAMMA_MAX * sigmoid(W_decay_logit)        # [N] ∈ [0, 0.97]
decay_gamma   = GAMMA_MAX * sigmoid(decay_gamma_logit)    # [N] ∈ [0, 0.97]
hebbian_gamma = GAMMA_MAX * sigmoid(hebbian_decay_logit)  # [N] ∈ [0, 0.97]

# All computation in bf16 now — no f32 casts:
W_new = (1 - W_gamma[None, :, None]) * W + W_gamma[None, :, None] * ΔW_normed

target_decay = sigmoid(Δdecay_raw)
decay_new    = (1 - decay_gamma[None, :]) * decay + decay_gamma[None, :] * target_decay
```

Why 0.97: at `γ = 0.97`, `(1-γ) = 0.03` has ~0.78% bf16 precision (7-bit
mantissa). At higher γ, `(1-γ)` compresses toward zero faster than bf16
can represent, and EMA errors compound across steps. 0.97 gives effective
integration windows up to ~33 modulation events ≈ 132 tokens, which covers
our full segment length. Longer-term memory happens through stable `W`
entries, not through slow-integrating γ.

Per-neuron rate is a capacity bump (some neurons may want fast plasticity,
others slow). Cost: `3·N = 768` new scalar params vs. current `3·NC_cells = 24`.

## Precision: bf16 compute, f32 parameters (PyTorch standard)

**Parameters and optimizer state stay in f32** (standard for AdamW
stability). **Runtime state and forward-pass compute are in bf16 on CUDA,
f32 on CPU.** Autocast handles the cast from f32 params to bf16 compute
on CUDA; on CPU, matching-dtype direct call works because runtime state
is f32 too.

No manual f32 casts anywhere in the memory module for the EMA blends —
the γ clamp handles precision. This is a deliberate simplification from
the current code, which has f32 casts for Hebbian and W/decay EMA blends.

The bf16-fragile operation in EMA blends (`(1-γ)·X + γ·Y` when γ ≈ 1) is
handled by **clamping γ to 0.97** via the activation `0.97 · sigmoid(logit)`
on all three plasticity rates. This keeps `(1-γ) ≥ 0.03`, which bf16
represents with ~0.78% precision — accurate enough that EMA accumulation
doesn't compound errors meaningfully.

Operations that PyTorch auto-promotes to f32 internally (so no manual
handling needed):
- `F.softmax`, `F.log_softmax`, `F.gumbel_softmax`, `F.cross_entropy`
- `F.group_norm`, `F.layer_norm`, `F.rms_norm` (mean-squared reduction)
- AdamW optimizer state (always kept in f32 regardless of param dtype)
- Tensor-core matmul accumulation on 4090 (inputs bf16, accum f32, output bf16)

The `W @ msg` and `msg @ msgᵀ` matmuls at N=256 are standard bmm operations
and use f32 accumulators on tensor cores — 256-element dot-product precision
is well within bf16's practical range.

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

Per modulation event (BS=72, N=256, C_h=192, C_dec_max=128):

| Step | FLOPs (order of magnitude) | Notes |
|---|---:|---|
| Project h, msg_emit, msg_recv | ~350M | 3 linear projections over N=256 |
| Build grid (broadcast + concat to [N,N,78]) | memory-bound, ~900 MB write | Materialize once per event |
| Conv encoder × 6 layers (k=7, C_h=192) | ~550G | Bulk of modulator compute |
| Pool + code logit (192 → K=4096) | ~55M | Tiny |
| Decoder initial Linear (192 → 3072) | ~40M | Small |
| Decoder 6 upsample stages (N²·C at output sides) | ~80G total | Upsample path |
| Decoder final 1×1 conv | ~5M | Reduce to scalar ΔW |
| Row RMSNorm + diag mask + EMA blend | O(BS·N²) | Memory-bound |

Wall time estimate on 4090: **~5-8 ms per modulation event**. At
mod_interval=4 and T=128 that's 32 events × 6ms ≈ **200 ms per segment**
dominated by the conv encoder. Non-modulation per-step memory work
(W·msg, state MLP, msg MLP, Hebbian update at N=256) adds another chunk;
overall throughput target **≥25K tok/s at BS=72**.

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

## Parameter budget (sized on merit)

We don't target a specific total-param number. Each component is sized for
what it plausibly needs to do the job. If phase-1 training hits capacity
ceilings we have clear levers to bump (wider encoder, more codes, etc.).

Memory allocation under this design:

| Component | Params |
|---|---:|
| Projection heads (h_proj, msg_emit_proj, msg_recv_proj): 3 × D_n · d_proj | 12K |
| Role embeddings (3 × role_dim=4) | tiny |
| Conv encoder (6 layers, C_h=192, k=7) | 9.7M |
| Code logit head (C_h=192 → K=4096) | 786K |
| Codebook (K=4096 × D_code=384) | 1.57M |
| Decoder (conv-transpose, init Linear + 6 upsample stages) | 2.1M |
| `inject_w, inject_b` (8 pools × α·D_n·D_n) | 2.1M |
| `neuron_id` (N=256 × D_n=256) | 65K |
| State MLP + msg MLP (shared) | 260K |
| Per-neuron plasticity logits (3 × N=256) | 1K |
| **Memory total** | **~16.6M** |

LM (unchanged): ~52M.

**Grand total: ~69M.**

Less than the current ~109M. That's fine — we're not padding to match
baselines, we're testing whether the conv-grid modulator works. If it trains
well at 67M, that's a cleaner result than one where we couldn't tell if it
was capacity or architecture doing the work.

Where the params moved compared to the old 42M-memory design:

- **Encoder**: 39.4M → 10.5M (conv + logit head). The per-cell MLP on
  flattened N² was the biggest waste; replacing it with a conv that
  shares weights across grid positions is the big structural win.
- **Decoder**: 143K → 1.0M. Slight increase, but now full-rank N×N ΔW
  via conv-transpose, no rank bottleneck.
- **Codebook**: 33K → 786K. 8× more templates, each 3× richer.
- **Everything else**: roughly unchanged.

### If it turns out we need more capacity

In priority order, the cheapest/highest-leverage bumps:
1. **Encoder depth** `L_conv` 6 → 8 (+3M).
2. **Encoder width** `C_h` 192 → 256 (+4M).
3. **Codebook size** `K` 4096 → 8192 (+786K).
4. **Decoder width** — bump the channel ladder in the upsample stages
   (e.g., 192 → 256 → 192 → ...) (+few hundred K).
5. **Depthwise-separable convs** to save compute if that becomes the
   binding constraint rather than params.

Cheap to scale up; don't scale preemptively.

## Comparison in one table

| Axis | Current | This design |
|---|---|---|
| Modulator encoder | 39M params, MLP on flattened N² | ~10M params, 2D conv on edge grid |
| Observation content | Rates + correlations only | + compressed node content (h, msg_emit, msg_recv) + edge asymmetry |
| Decoder | 143K MLP, produces implicit rank-≤64 N² action | 1M conv-transpose stack, full-rank N×N ΔW |
| Policy structure | 8 independent codes per event | 1 code per event |
| Neurons per cell | 32 | 256 |
| Cells | 8 | 1 |
| Total neurons | 256 | 256 |
| W matrices | 8 × 32² = 8192 entries | 1 × 256² = 65K entries (8× more connectivity) |
| Connectivity | within cells only | global across whole cell |
| Memory module params | 42M | ~15M |

## Open questions

1. **Content vs content-free observation.** The compressed h/msg broadcast is
   a departure from "statistics only" — do we regress on the biologically
   principled stance, or keep it? Worth ablating.
2. **Depthwise-separable vs dense conv in the encoder.** Dense is simpler
   and fits at ~10M params; depthwise is ~10× fewer params with similar
   expressiveness. Start dense, switch if compute becomes tight.
3. **One code per event vs a few (H=2 or 4).** Even without the 8-cell
   factoring, having a handful of parallel codes could matter for GRPO SNR.
   Deferred to post-phase-1 evaluation.
4. **Conv-transpose vs resize+conv in the decoder.** We default to
   resize+conv to avoid checkerboard artifacts. Native `ConvTranspose2d`
   could be tried if wall-time matters and artifacts turn out not to bite
   (the downstream RMSNorm + EMA blend may absorb them).
5. **Post-processing on ΔW.** Currently only diagonal mask + row RMSNorm.
   Role mask (zeroing structurally-impossible edges) is an option if we
   want to enforce hard port-layer constraints.
6. **Should conv be replaced by attention-with-edge-bias?** Permutation-
   equivariant, but less hardware-optimal. If conv training is unstable, try
   attention as a fallback.
7. **γ_max ceiling.** 0.97 is our bf16-safe cutoff. If the modulator learns
   to peg γ against the ceiling everywhere (suggesting it wants slower
   integration), consider upgrading the specific EMA blends to f32 to
   unlock γ up to ~0.998 — a localized exception to the bf16-everywhere
   rule. Monitor `γ_max` across neurons.
8. **Per-cell vs per-neuron plasticity γ.** We moved from per-cell (24
   scalars) to per-neuron (768 scalars). If this causes instability (e.g.,
   a subset of neurons learning pathological rates), back off to per-cell.
9. **Dropout on conv encoder.** Default 0.1 between layers; may be
   unnecessary if the LM's dropout already regularizes the full chain.
   Ablate.

## References and related work

- Mordvintsev et al., "Growing Neural Cellular Automata" (2020) — the literal
  2D-conv-over-grid-state idea, applied to image synthesis but architecturally
  parallel.
- Message-passing neural networks (Gilmer et al. 2017) — the general framework
  for node + edge updates; conv-over-edge-grid is a specific instance when
  edges form a dense pairwise set.
- Odena et al., "Deconvolution and Checkerboard Artifacts" (2016) — the
  canonical reference for why we use resize+conv instead of native
  ConvTranspose2d in the decoder.
- Transformer with edge biases (Graphormer, Dwivedi & Bresson 2020) — the
  permutation-equivariant fallback if conv misbehaves.
