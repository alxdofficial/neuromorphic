# CortexNet v0 — Wave-Grid Design

**Branch:** `memory-is-the-model` (from `main` @ 84a310f).
**Status:** design locked. Ready to scaffold.
**Date:** 2026-04-20.
**Full-vision roadmap:** `plan_memory_is_the_model_full_vision.md` (the
superset design with 6 laminae, ERR units, episodic buffer, replay,
structural plasticity — future extensions on top of v0).

> **TL;DR.** We delete the LM and replace it with a **2-D grid of
> cortical columns with wave-propagation dynamics**. 6×6 = 36 columns,
> each containing N=32 neurons. Tokens enter at sensory columns (left
> edge), waves propagate rightward via local 8-neighbor connectivity,
> output columns (right edge) feed a multi-horizon readout. A small
> number of random long-range skip connections (K_long=4 per column,
> fixed at init) keep the grid's effective diameter low.
>
> **All connection weights are scalars** (per-edge, not per-channel or
> matrix). Channel mixing happens inside each neuron's `state_MLP` and
> `spike_MLP`, not at the edges. This is biologically accurate and
> gives the neuromodulator a minimal action space: just `(η_c, dir_c)`
> per column per MOD event — one learning-rate and one LTP/LTD sign
> scalar. Hebbian eligibility traces per edge decide WHICH synapses
> actually update.
>
> **Fast weights are state, not parameters.** `W_intra`, `W_inter`,
> `W_long` are never backprop-trained. They load from a learned init
> template once, at day-0 of deployment; from there, neuromod-driven
> plasticity is the only thing that changes them. **At inference,
> nothing resets — ever.** Column states, Hebbian traces, and fast
> weights all evolve continuously from deployment forward. Years of
> running = years of accumulated plastic state. This is the
> "memory IS the model" thesis in its cleanest form and the true
> lifelong-learning property. Training-time resets (TBPTT detach,
> between-example boundaries) are artifacts of bounded-graph backprop,
> not architectural features.
>
> **Meta-learning framing.** Neuromod + MLP weights ARE the learning
> rule (frozen at inference). Fast weights are what gets learned
> (plastic at inference, reset per segment). Training shapes the
> rule; deployment lets the rule learn forever.
>
> **Two-phase training.** Phase 1 = differentiable-plasticity backprop
> with TBPTT (trains everything except the fast weights themselves).
> Phase 2 = GRPO long-horizon rollouts with Gaussian noise on neuromod
> scalar actions (trains neuromod policy only).
>
> **No hierarchy, no thalamus, no attention.** Signals propagate across
> the grid as waves. No shortcuts. Memory lives in plastic scalar
> weights + state that persists across ticks within a segment.

---

## 1. Why kill the LM

The current design (`docs/design.md`, `docs/pretrained_lm_memory.md`) has
a clean LM + memory split, but the split forces three structural problems:

1. **Gradient SNR is dominated by the LM.** Memory is additive on top of
   a near-perfect teacher (Llama-3.2-3B). `verify_01` on the from-scratch
   branch measured within-K rollout spread at 2.8×10⁻⁴ vs across-slot 2.4
   — memory is a rounding error on top of a frozen LM.
2. **The LM is not plastic at inference.** If memory adapts but the LM
   doesn't, the system's *behaviour* is still bounded by frozen weights.
3. **Two substrates = two throughput budgets.** The memory runs at 68K
   tok/s standalone; bolted onto Llama-3B, the LM dominates compute.

CortexNet removes the LM. Memory *is* the model.

---

## 2. Scope cuts vs the full vision

The full-vision design (see archived companion doc) stacks nine new
mechanisms at once. V0 keeps the minimum needed to test the core thesis.

### Kept in v0
- Plastic weights at inference (scalar W_intra, W_inter, W_long).
- Three-factor plasticity (eligibility × surprise × modulator).
- Shared-trunk neuromodulator (one per column, emits scalar action).
- 2-clock event loop (FAST every token, MOD every 16).
- Triton fusion path (reuse current kernel pattern).
- **2-D grid structure** (new): columns laid out spatially, local
  8-neighbor connectivity, sensory and motor regions at opposite edges.
- **Per-column long-range skip connections** (new): K_long=4 random
  targets per column, fixed topology, scalar weights.
- **Multi-horizon readout** (new): K=8 prediction heads from motor
  columns. Non-degenerate surprise in self-generation.
- **Fast weights as state** (new): W_intra, W_inter, W_long never
  trained by backprop — only neuromod-driven plasticity modifies them.

### Cut / deferred
- **LM scaffold** — gone entirely.
- **Recursive tree / hierarchy** — gone; flat 2D grid instead.
- **Thalamic broadcast buffer** — no global shortcuts; waves only.
- **Per-channel and matrix weight expressivity** — scalar weights
  everywhere. Channel mixing via MLPs.
- **6 laminae per column** — single homogeneous neuron pool per column.
- **ERR units per lamina** — v0 uses global scalar surprise.
- **Episodic buffer + replay** — defer to v3.
- **CONSOLIDATE clock** — not in v0.
- **Structural plasticity** — skip topology fixed at init.

---

## 3. Architecture

### 3.1 The wave-grid principle

Columns are laid out on a 2-D grid. Each column connects to its 8 grid
neighbors (Moore neighborhood) plus K_long=4 random long-range targets
chosen at init. Tokens arrive at sensory columns on the LEFT edge;
waves of activity propagate rightward through the grid; the RIGHT edge
(motor columns) pools into the readout.

```
      SENSORY                                             MOTOR
   ┌─────────────────────────────────────────────────────────┐
   │ S ─ · ─ · ─ · ─ · ─ · ─ · ─ M                           │ ← row 0
   │ │ ╳ │ ╳ │ ╳ │ ╳ │ ╳ │ ╳ │                               │
   │ S ─ · ─ · ─ · ─ · ─ · ─ · ─ M                           │ ← row 1
   │ │ ╳ │ ╳ │ ╳ │ ╳ │ ╳ │ ╳ │                               │
   │ S ─ · ─ · ─ · ─ · ─ · ─ · ─ M                           │ ← row 2
   │ ...                                                      │
   │ S ─ · ─ · ─ · ─ · ─ · ─ · ─ M                           │ ← row 5
   └─────────────────────────────────────────────────────────┘
     x=0  1   2   3   4   5   ← 6 columns wide
   Plus: each column has 4 random long-range skip edges (not drawn)
```

- Grid: **6 × 6 = 36 columns** at dev tier.
- Sensory columns: `x=0` (leftmost), 6 columns.
- Motor columns: `x=5` (rightmost), 6 columns.
- Grid diameter without skips: ~10 (Manhattan). With K_long=4 random
  skips: ~3 effective diameter (small-world).

### 3.2 One column

Each column has a consistent internal structure:
- **N=32 neurons**, each with a `D_n=128`-dim hidden state `h` and
  outgoing message `msg`.
- **Dense intra-column connectivity**: scalar `W_intra[c, i, j] ∈ R`
  per (sender j, receiver i) pair. 32² = 1024 scalar weights per column.
- **Hebbian trace** per intra edge: `hebbian_intra[c, i, j] ∈ R` — rolling
  EMA of sender-receiver co-firing.
- **Column identity embedding** `cell_emb[c] ∈ R^{32}` — learned (slow),
  feeds into the neuromod.

### 3.3 Inter-column connectivity (grid + skips)

Each column has:
- `K_grid = 8` Moore-neighborhood edges (within-grid local).
- `K_long = 4` random long-range edges to non-neighbor columns, fixed
  at init.

All inter-column weights are **scalars**:
- `W_inter[c, k] ∈ R` for `k ∈ 0..7` (grid neighbor index).
- `W_long[c, k] ∈ R` for `k ∈ 0..3` (long-range target index).

Plus Hebbian traces at every edge (same shapes).

Static index tensors fixed at init:
```
grid_nbrs[c] : Long[C, K_grid=8]    # column indices of grid neighbors
long_nbrs[c] : Long[C, K_long=4]   # column indices of random skips
```

**Compile-time optimization.** Both grid and skips have fixed shape per
column, so we concatenate them into one `[C, K_total=12]` target array
and run a single batched gather + scalar-weighted sum per tick.

### 3.4 Fast weights are state, not parameters

The three plastic weight tensors:

| Tensor | Shape | Life cycle |
|---|---|---|
| `W_intra` | `[C=36, N=32, N=32]` = 37 K scalars | Load from init template at day-0 → neuromod-plastic forever at inference |
| `W_inter` | `[C=36, K_grid=8]` = 288 scalars | Same |
| `W_long` | `[C=36, K_long=4]` = 144 scalars | Same |
| Hebbian traces (per edge) | same shapes as above | EMA every FAST tick; natural decay via γ_e |

These are **never in the optimizer.** Backprop flows through them
during training (they're part of the graph), but their gradients don't
get applied. They're generated by the neuromod's plasticity rule.

**At inference, nothing resets — ever.** Fast weights, column states,
Hebbian traces, prediction ring buffer all evolve continuously from
deployment onward. The learned init templates are loaded ONCE, at day
0; after that every state evolves under the plasticity rule. Years of
deployment = years of accumulated plastic changes. This is the
lifelong-learning property.

**At training, some resets are unavoidable** (you can't backprop
through a billion-step graph). These are training-time artifacts, not
architectural properties. See §7 for the detach/reset schedule during
phase 1 and phase 2.

**The learned init template** matters only at day 0 of deployment. It
is trained by backprop during phase 1 (it's an ordinary
`nn.Parameter`), because it serves as the starting point for every
training example. At inference it's only referenced once (loaded at
deploy time) and then irrelevant.

**Drift safety at inference.** Three natural defenses prevent runaway:

1. **γ_W-clamped EMA on plasticity writes:** each update blends only
   `γ_W < 1` fraction of Δw, so no single event dominates.
2. **Three-factor gate:** plasticity only fires when eligibility ×
   surprise × neuromod-signal agree. Irrelevant events do nothing.
3. **Hebbian decay γ_e:** stale eligibility fades. Edges not recently
   active stop getting modified.

Biology uses the same defenses (synaptic normalization, neuromodulator-
gated plasticity, Hebbian decay). Brains run 80+ years without
resetting; we inherit the principle.

Future option (v1): switch to purely random init (no learned template) —
the most honest "memory IS the model" statement, but harder to train.

### 3.5 One FAST tick

Fully vectorized across all 36 columns — no per-level loops, no
recursion, no tree traversal.

```python
def fast_tick(self, external_input):
    # 1. Intra-column scalar matmul (per column, batched)
    #    W_intra is [C, N, N] (scalar), msg_prev is [C, N, D_n]
    intra_out = einsum('cij,cjd->cid', self.W_intra, self.msg_prev)
                                                             # [C, N, D_n]

    # 2. Column-level pool (neurons → column message)
    col_msg = pool_head(self.msg_prev)                       # [C, D_n]

    # 3. Inter-column: one gather + scalar-weighted sum
    #    all_nbrs is [C, K_total=12] fixed indices (grid + skips)
    #    W_all is [C, K_total] scalar weights
    nbr_msgs = col_msg[self.all_nbrs]                        # [C, K_total, D_n]
    inter_contrib = einsum('ck,ckd->cd', self.W_all, nbr_msgs)
                                                             # [C, D_n]

    # 4. Sensory input injected at left edge ONLY
    received_col = inter_contrib                             # [C, D_n]
    received_col[self.sensory_cols] += self.W_sense @ external_input

    # 5. Broadcast column-level signal into each column's neurons
    received = intra_out + received_col.unsqueeze(1)         # [C, N, D_n]

    # 6. LIF state update (batched over all 36×32 neurons)
    self.h = decay * self.h + (1 - decay) * tanh(
        self.state_MLP(received, self.h))                    # [C, N, D_n]

    # 7. Emit messages (per-neuron)
    self.msg = self.spike_MLP(self.h)                        # [C, N, D_n]

    # 8. Hebbian trace updates — ALL THREE edge types, same pattern
    #    (post_new × pre_old correlation, EMA'd with γ_e).
    #
    # Intra-column: per-neuron co-firing [C, N, N]
    corr_intra = einsum('cid,cjd->cij', self.msg, self.msg_prev)
    self.hebbian_intra = ((1 - γ_e) * self.hebbian_intra
                          + γ_e * corr_intra)

    # Column-level pooled messages for inter-column Hebbian
    col_msg      = pool_head(self.msg)                     # [C, D_n]
    col_msg_prev = pool_head(self.msg_prev)                # [C, D_n]

    # Inter-column grid: per grid-neighbor co-firing [C, K_grid=8]
    grid_prev    = col_msg_prev[self.grid_nbrs]            # [C, K_grid, D_n]
    corr_inter   = einsum('cd,ckd->ck', col_msg, grid_prev)
    self.hebbian_inter = ((1 - γ_e) * self.hebbian_inter
                          + γ_e * corr_inter)

    # Long-range skips: per skip-target co-firing [C, K_long=4]
    long_prev    = col_msg_prev[self.long_nbrs]            # [C, K_long, D_n]
    corr_long    = einsum('cd,ckd->ck', col_msg, long_prev)
    self.hebbian_long  = ((1 - γ_e) * self.hebbian_long
                          + γ_e * corr_long)

    # 9. Multi-horizon readout from motor columns ONLY
    if len(self.motor_cols) > 0:
        h_motor = self.h[self.motor_cols]                    # [|motor|, N, D_n]
        h_motor_pool = pool_motor(h_motor)                   # [D_e]
        for k in range(K_horizons):
            logits_t[k] = unembedding(h_motor_pool + horizon_emb[k])
        self.pred_buf.append(logits_t)                       # [K, vocab]

    # 10. Surprise scoring against actual_t (materialized from external
    #     input, or own sample during self-generation)
    self.surprise_ema = score_surprise_against(actual_t)

    # Cache for next tick
    self.msg_prev = self.msg
```

Four batched einsums + one gather + standard MLPs. All tensor shapes
are fixed at compile time.

### 3.6 Neuromodulator — scalar action per column

The neuromod fires every MOD clock (every 16 tokens). Per column, it
emits just **two scalars**: the learning-rate magnitude and the LTP/LTD
direction.

```python
# Per-column pooled features
feats_c = concat([
    # --- state + messages: mean over neurons, keeps channels ---
    h[c].mean(dim=0),                   # [D_n=128] — column state summary
    msg_window[c],                      # [D_n]     — EMA of col_msg over last 16 ticks
                                         #              (already pooled over time)
    # --- Hebbian features: don't bottleneck to scalar ---
    intra_hebbian_moments(hebbian_intra[c]),
                                         # [4]   mean|·|, std|·|, max|·|, mean (signed)
    hebbian_inter[c],                   # [K_grid=8]  raw per-edge eligibility
    hebbian_long[c],                    # [K_long=4]  raw per-edge eligibility
    # --- identity + global signals ---
    cell_emb[c],                        # [32]  — learned column identity
    surprise_ema * broadcast,           # [1]   — global error, broadcast to each col
    recent_input_ctx * broadcast,       # [D_e=512] — global EMA of token embeddings
])                                       # → ~820 scalars per column

# Shared trunk (batched across all columns)
trunk_out = neuromod.trunk(feats)       # [C, D_trunk]

# Per-column scalar actions — only two outputs per column
η_c   = sigmoid(neuromod.head_η(trunk_out))   * η_max   # [C] ∈ [0, η_max]
dir_c = tanh   (neuromod.head_dir(trunk_out))           # [C] ∈ [-1, +1]
```

**Pool choice rationale.** Mean pooling for `h[c]` preserves 128 dims of
per-channel activity — not bottlenecked. Hebbian intra is `[N, N] = 1024`
scalars, so collapsing to one scalar would throw away both magnitude
and distribution shape. Four moments (mean, std, max of absolute values
plus signed mean) preserve the information the neuromod needs for
coarse per-column "how much / which sign" decisions without exploding
the feature size. Inter and long Hebbian are already small vectors
(8 / 4 scalars) — pass them raw, no pooling.

**None of these pools are shared.** `h`/`msg_window` pools reduce the
neuron axis; Hebbian-intra pools reduce a full `[N, N]` matrix to 4
scalars; inter/long Hebbian are passed through unchanged. The
`pool_head` used every FAST tick to produce `col_msg` is a separate
(learnable) module — it's on the hot path and has a different
semantic role ("how does this column summarize itself for outgoing
messages").

**Action space per MOD event:** 36 columns × 2 scalars = 72 scalars
network-wide. Tiny. The neuromod's entire job is to decide per column:
"learn harder / softer" and "LTP or LTD."

### 3.7 Plasticity rule — scalar Hebbian update

At each MOD event, for every edge:

```python
# Three-factor gate — sigmoid of pooled features, per edge
gate = sigmoid(gate_net(
    hebbian_at_this_edge,      # eligibility (factor 1)
    surprise_ema,              # error (factor 2)
    η_c * dir_c,               # neuromod (factor 3, signed)
))                              # scalar ∈ [0, 1]

# The update per edge is a scalar:
#   magnitude = η_c * dir_c * gate
#   "which edges change" is encoded in the Hebbian trace
ΔW_edge = η_c * dir_c * gate * hebbian_at_this_edge

# γ-clamped EMA write (all three weight tensors updated same way)
W_intra[c, i, j] ← (1-γ_W) * W_intra[c, i, j] + γ_W * ΔW_intra[c, i, j]
W_inter[c, k]    ← (1-γ_W) * W_inter[c, k]    + γ_W * ΔW_inter[c, k]
W_long[c, k]     ← (1-γ_W) * W_long[c, k]     + γ_W * ΔW_long[c, k]
```

**Why this works.**
- Neuromod's ONE scalar per column (direction × magnitude) tells the
  whole column "how much to plasticize right now."
- Hebbian per edge tells "which edges have been active together."
- Product of the two gives per-edge scalar updates distinguishing
  edges that should change from those that shouldn't.

Classical three-factor learning: `Δsynapse = presynaptic × postsynaptic × neuromodulator`.
Here, presynaptic × postsynaptic → Hebbian trace; neuromodulator → η·dir.

At inference, this rule keeps firing. Only the fast weights move.
Everything else (neuromod, MLPs, readout, embedding) is frozen.

---

## 4. Event clocks (2 clocks)

| Clock | Period | What fires |
|---|---:|---|
| **FAST** | 1 tok | Intra + inter scalar mixing; LIF update; spike_MLP; Hebbian trace update; multi-horizon readout from motor cols; surprise scoring against `actual_t`. |
| **MOD** | 16 tok | Neuromod emits `(η_c, dir_c)` per column; three-factor gate; scalar plasticity writes to W_intra, W_inter, W_long. |

---

## 5. Module layout

```
src/cortex/
├── config.py              — (C, N, D_n, K_grid, K_long, sensory/motor,
│                             K_horizons, clock periods, γ's, σ)
├── grid.py                — compute grid_nbrs + long_nbrs indices at init
├── cortex.py              — main module; stacked per-col tensors;
│                             fast_tick + mod_tick + readout + surprise
├── neuromod.py            — shared trunk + head_η + head_dir + gate net
├── readout.py             — multi-horizon readout head + pred ring buffer
│                             + surprise scorer
├── train_phase1.py        — TBPTT multi-horizon CE; trains slow weights
│                             + fast-weight init templates
└── train_phase2.py        — GRPO rollouts with Gaussian noise on (η, dir);
                             trains neuromod + gate only
```

Two flat modules for the network (`cortex.py`, `neuromod.py`), one for
readout, one for precomputed grid indices. No `column.py` — columns are
indexed slices of stacked tensors. No `thalamus.py`. No `inter_column.py`
— grid and skip routing are both handled in `cortex.py` via one batched
gather.

---

## 6. Parameter count (dev tier, C=36, N=32, D_n=128, K_grid=8, K_long=4, K=8)

### Slow weights (trained by backprop, frozen at inference)

| Component | Shape / notes | Params |
|---|---|---:|
| Token embedding (tied to unembedding) | `[vocab=32K, D_e=512]` | 16.4 M |
| W_sense (input projection) | `[D_e, D_n]` | 66 K |
| state_MLP (shared across all neurons) | 2-layer `[D_n×2 → D_n]` | 65 K |
| spike_MLP (shared) | similar | 65 K |
| pool_head (neuron → column pooling) | `[N, D_n → D_n]` | 4 K |
| pool_motor (motor cols → D_e) | `[|motor|·N·D_n → D_e]` | 400 K |
| Horizon embedding | `[K=8, D_e]` | 4 K |
| Neuromod trunk | shared transformer/MLP, ~4-layer | ~3 M |
| Neuromod heads (η, dir) | `D_trunk → 1` each | ~20 K |
| Gate net | small MLP | ~10 K |
| cell_emb | `[C=36, 32]` | 1 K |

### Learned fast-weight init templates (trained by backprop, loaded at segment start)

| Component | Shape | Params |
|---|---|---:|
| W_intra_init | `[C, N, N]` scalar | 37 K |
| W_inter_init | `[C, K_grid]` scalar | 0.3 K |
| W_long_init | `[C, K_long]` scalar | 0.1 K |

### Fast weights (STATE, not params, zero optimizer entries)

Identical shapes to the init templates. Evolve via neuromod during
forward, reset to init at segment start.

### Hebbian traces (STATE, not params)

Identical shapes to W_intra / W_inter / W_long. Rolling EMA, reset per
segment.

### Totals

- **Slow weight params:** ~20 M
- **Fast-weight init template params:** ~37 K
- **Grand total:** ~20 M

That's ~5× smaller than the pretrained-LM path (100 M+ on Llama-1B)
and ~20× smaller than Llama-3B. The "model" is mostly the learning
rule (neuromod + MLPs); the "memory" lives in the fast-weight state.

---

## 7. Training protocol — two phases

### 7.1 Phase 1 — Differentiable plasticity with TBPTT

> **Reminder:** all resets in this section are training-time artifacts
> for keeping the backward graph bounded and keeping independent
> examples from contaminating each other. At inference these resets do
> NOT happen — see §3.4.

**Loss (multi-horizon NTP CE):**
```
L_NTP = Σ_t Σ_{k=1..K}  λ_k · CE(logits_t[k], actual_{t+k})
```

**TBPTT schedule (training-only):**
- Segment length `T = 256` tokens.
- `tbptt_block = 32` — DETACH the backward graph every 32 tokens
  (but values persist through the detach — this is a graph op, not a
  state op).
- Memory state reset ONLY at segment boundary, and ONLY between
  independent examples (passkey exampleN+1 shouldn't inherit exampleN's
  writes). For continuous-corpus training (e.g., streaming Wikipedia)
  we can skip even this reset and treat the full corpus as one stream.
- Fast weights LOAD from `W_*_init` values at the start of each
  independent example. Backprop updates the init templates, not the
  fast weights themselves.

**What trains:** all slow weights + the three fast-weight init
templates. The fast weights themselves are never in the optimizer.

**Gradient flow:**
```
CE → readout → h[motor] (post-plasticity)
     → W_intra at time t → mesa-opt update step → neuromod scalars at time t
     → neuromod weights; ALSO back through plasticity at t-16 → etc.
```

Two MOD events per 32-token TBPTT window.

**VRAM estimate (C=36, BS=8, tbptt=32):**
Activations: `~1057 neurons × 128 × 32 × 8 × 2 bytes ≈ 70 MB`. Plus
overhead, ~200 MB total. Easily fits on a 4090.

### 7.2 Phase 2 — GRPO long-horizon rollouts

> **Reminder:** same as §7.1, the rollout reset is a training-time
> artifact. Each of the K rollouts starts from the same init template
> so their log-probability trajectories can be compared. At inference
> there are no rollouts and no resets — the network just runs.

**Why phase 2.** Phase 1 backprop only reaches 32 tokens into the past.
It cannot learn "write something at token 10 that pays off at token 500."
Phase 2 fills the gap with policy-gradient over full rollouts.

**Stochasticity source:** Gaussian noise on the scalar neuromod actions.

```python
η_mean, dir_mean = neuromod(state, surprise)       # [C], [C]
η_sample = η_mean + σ_η * randn_like(η_mean)
dir_sample = dir_mean + σ_dir * randn_like(dir_mean)
log_pi = gaussian_log_pdf(η_sample, η_mean, σ_η) + \
          gaussian_log_pdf(dir_sample, dir_mean, σ_dir)
apply_plasticity(η_sample, dir_sample)             # uses sampled scalars
```

Very small action dimensionality (`36 × 2 = 72` scalars per MOD event)
means Gaussian noise is cheap and the `log_pi` is straightforward.

**Per step:**
```python
prefix = sample_prefix()                # [1, T_prefix]
reference = sample_continuation()       # [gen_length]

rollouts = []
for k in range(K_rollouts=8):
    reset_state(); reset_fast_weights_from_init()
    log_pi_sum = 0

    for t in prefix:
        fast_tick(t)
        if mod_clock_fires:
            η_sample, dir_sample = sample_noisy_actions()
            log_pi_sum += gaussian_log_pdf(...)
            apply_plasticity(η_sample, dir_sample)

    with torch.no_grad():
        generated = autoregressive_rollout(gen_length)
    reward = reward_fn(generated, reference)
    rollouts.append((log_pi_sum, reward))

# GRPO advantage + REINFORCE loss
advantages = (rewards - mean(rewards)) / max(std(rewards), adv_floor)
loss = -sum(log_pi * adv.detach() for (log_pi, _), adv in zip(rollouts, advantages))
loss.backward()
optimizer.step()   # only neuromod + gate update
```

**What trains:** neuromod trunk + heads + gate net. Everything else
frozen (readout, state/spike MLPs, fast-weight init templates,
embedding).

**Why phase 2 isn't SNR-degenerate anymore.** On the prior branch, the
frozen LM dominated CE and memory's contribution was ~10⁻⁴. Here
memory IS the model — K rollouts genuinely diverge because different
`(η_sample, dir_sample)` sequences produce different fast-weight
trajectories, different states, different logits, different samples.

### 7.3 Training schedule

```
  ┌────────────────────────────────────────────────────────┐
  │ PHASE 1 — multi-horizon CE with TBPTT                   │
  │   T=256, tbptt=32, trains slow weights + init templates │
  │   ~N_phase1 steps. Gate: passkey at T=256?              │
  └────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ PHASE 2 — GRPO long-horizon rollouts                    │
  │   K=8 rollouts/prompt, prefix=256, gen=256              │
  │   Gaussian σ on scalar actions; trains neuromod + gate  │
  │   ~N_phase2 steps. Gate: multi-needle at T=2K?          │
  └────────────────────────────────────────────────────────┘
```

Optionally cycle phase 1 ↔ phase 2 if phase-2 freezing drifts NTP
fluency.

### 7.4 Data (reuses `docs/training_plan.md` stack)

- **Stage 0** — passkey / K:V synthetic (`scripts/build_passphrase_data.py`
  on main). Primary gate for v0.
- **Stage 1** — conversational (deferred).
- **Stage 2** — real long-form (deferred).

---

## 8. Why v0 is fast

1. **LM removal.** Main's memory graph standalone ran 68K tok/s;
   bolted onto Llama-3B it dropped to ~90. We're back in the
   memory-graph-native regime.
2. **Scalar weights everywhere.** All routing is scalar-weighted sum,
   not D_n × D_n matrix multiply. Massive param and compute savings.
   4.7M→37K for intra, 4.7M→288 for grid, 2.4M→144 for skips.
3. **Fixed-shape batched ops.** Every op is a known-shape einsum or
   gather. GPU loves this. Triton-fusable per-tick.
4. **Fast weights cached in L2.** Total fast weights ~37K scalars =
   150 KB. Stays in L2 across all 16 FAST ticks of a MOD window.
5. **Constant memory in sequence length.** No KV cache.
6. **Small action space for neuromod.** 72 scalars per MOD event.
   Cheap forward + backward.

**Target throughput: 150–250 K tok/s at dev scale.** Exceeds current
main's standalone memory graph.

---

## 9. Fast-throughput analysis

### 9.1 Per-token FAST-clock workload

| Operation | Shape | FMA |
|---|---|---:|
| Intra-column scalar matmul | `[C=36, N, N] × [C, N, D_n]` | 4.7 M |
| Inter-column scalar (grid + skips) | `[C, 12] × [C, 12, D_n]` | 55 K |
| state_MLP (shared, batched 36×32=1152 neurons) | 2-layer `D_n×2 → D_n` | ~1 M |
| spike_MLP (same) | ~1 M |
| LIF update + elementwise | — | ~0.1 M |
| Hebbian trace updates (scalar per edge) | — | ~40 K |
| Multi-horizon readout (K=8 × `D_e → vocab`) | — | 131 M |
| Surprise scoring (softmax × K) | — | ~1 M |
| Input projection at sensory cols | `[D_e, D_n]` | ~67 K |
| **Total per token** | | **~139 M** |

The multi-horizon readout (131 M) dominates everything else (~8 M). All
the cortex-side work is cheap.

**RTX 4090** peak bf16 ≈ 80 TFLOPS, realistic 40-50 % MFU ≈ 35 TFLOPS.

**FLOP-bound throughput:** `35 × 10¹² / 139 × 10⁶ ≈ 252 K tok/s`.

### 9.2 Memory bandwidth

Per-token DRAM traffic (weights stay in L2 across FAST ticks):

| Item | Bytes |
|---|---:|
| msg_prev read (all columns' neurons) | 147 K × 2 = 295 KB |
| h read/write | 590 KB |
| received scratch | 295 KB |
| Hebbian trace scalar updates | ~40 KB |
| Multi-horizon readout output (K × vocab) | 8 × 32K × 2 = 512 KB |
| Prediction ring buffer | 512 KB |
| **Total** | **~2.3 MB** |

4090 HBM ≈ 1 TB/s; realistic 600 GB/s sustained.

**Bandwidth-bound throughput:** `600 × 10⁹ / 2.3 × 10⁶ ≈ 260 K tok/s`.

### 9.3 Weight reuse

Fast weights total: `37 K + 288 + 144 = 37.4 K` scalars = **~150 KB in
bf16**. Fits in L2 cache (48 MB on 4090) trivially. Stays resident
across all 16 FAST ticks of a MOD window, reloaded only on MOD update.

### 9.4 MOD clock overhead

Every 16 tokens: neuromod forward + plasticity write.

| Operation | Cost |
|---|---:|
| Neuromod trunk (batched across all 36 cols) | ~5 M FMA |
| Per-column heads (η, dir) | ~100 K FMA |
| Scalar plasticity step (all 37K edges) | ~150 K FMA |
| EMA write to fast weights | 150 KB DRAM write |
| **Amortized per token** | ~0.35 M FMA |

~0.25% overhead per FAST tick. Negligible.

### 9.5 Target throughput summary

| Regime | Throughput |
|---|---:|
| FLOP-bound | ~252 K tok/s |
| Bandwidth-bound | ~260 K tok/s |
| MOD amortization | −0.3 % |
| Realistic (Python loop, kernel launch overhead) | 150–200 K tok/s |

Compares favorably to current main memory graph standalone (68 K) and
crushes Llama-3B autoregressive (~90 tok/s).

### 9.6 What could break the fast-claim

1. **Python loop over FAST clock.** Same concern as current main.
   Fix: Triton-fuse the per-tick kernel (M5 in milestones).
2. **Multi-horizon readout GEMM.** 131 M of the 139 M total per token
   comes from the K=8 readouts. If this bottlenecks, reduce K or share
   more compute across horizons.
3. **Gather kernel for grid + skip indexing.** Static indices should be
   efficient, but worth benchmarking.

None are v0-scale dealbreakers.

---

## 10. Risks

### 10.1 Biggest risks

1. **Without the LM scaffold, does NTP even descend?** The LM was
   doing 99%+ of the lift. Kill it and we need to prove plastic cortex
   can learn language from scratch. Stage-0 passkey at T=256 is the gate.
2. **Scalar weights too restrictive.** Channel-preserving edge routing
   may not be enough for complex language. Fallback: upgrade to
   per-channel W_intra (matching current main), with cost `N²·D_n` per
   column. 4.7M params, still tractable.
3. **Wave propagation too slow for autoregressive LM.** At diameter 5,
   output lags input by ~5 ticks. Readout must predict based on
   5-token-old context. Mitigations: smaller grid (4×4 → diameter 3),
   fast-weight memory storing recent tokens, or rethinking the "no
   shortcuts" constraint.
4. **Scalar action neuromod can't learn useful policies.** If `(η, dir)`
   per column is too little information, the mesa-opt updates will be
   too blunt to steer plasticity. Fallback: add a small `y_c ∈ R^{D_n}`
   target back into the action space, at the cost of larger policy
   search.
5. **Unbounded plasticity drift at inference.** Because nothing resets
   at inference, bad plasticity decisions could compound over billions
   of tokens. Defenses: (a) γ_W-clamped EMA writes, (b) three-factor
   gate blocks noise-level updates, (c) Hebbian decay γ_e forgets
   stale eligibility. These should be sufficient (biology uses the
   same defenses), but unprecedented for this scale. Empirical
   question: does the network remain stable after 10⁶ tokens of
   uninterrupted inference? Hard to test until we have a deployed
   model, but can partially characterize via extended evaluation runs.

### 10.2 Open architectural questions

1. Grid size sweet spot — sweep `(C, dims)` once v0 works.
2. K_grid and K_long neighborhood sizes.
3. Learned vs random init for fast weights (v0 uses learned; v1 could
   test random).
4. Per-neuron vs per-column skip connections.

---

## 11. Milestones

**M0 — Scaffolding (this branch).**
Write `cortex.py`, `neuromod.py`, `readout.py`, `grid.py`, `config.py`
in pure PyTorch. No Triton. Shape + gradient sanity check on random data.

**M1 — Single-token forward + backward smoke.**
Confirm multi-horizon CE is finite, surprise ring buffer fills, past
predictions score against materialized tokens, gradient flows to all
slow weights + fast-weight init templates. Fast weights never in
optimizer.

**M2 — Stage-0 passkey training (phase 1 only).**
Train on `scripts/build_passphrase_data.py` at T=256, N=1 passphrase.
**Gate: does it solve passkey at T=256?**

**M3 — Phase 2 GRPO on long-horizon passkey.**
Add `train_phase2.py` with Gaussian-noise scalar-action rollouts. Test
at T=1024+. **Gate: does phase 2 extend retrieval beyond phase 1's
TBPTT reach?**

**M4 — Multi-needle + shape sweep.**
Grow N_phrases ∈ {2, 3, 5}. Sweep grid size. Ablate: scalar vs
per-channel W_intra (if needed), neuromod scalar-action vs
target-action.

**M5 — Triton fusion of the FAST kernel.**
Port per-tick hot path to Triton.

**M6 — Real-data training.**
Wikitext / PG-19 at T=512. Compare CE vs SmolLM2-135M at comparable
param count.

Everything beyond M6 — laminar split, per-level surprise (ERR units),
episodic buffer, replay, structural plasticity — is the full-vision
roadmap.

---

## 12. Relationship to prior work on our tree

- `main` — pretrained-LM pivot. Production path; CortexNet is a
  research bet.
- `src/model/memory.py` — current memory graph with per-channel W
  `[NC, N, N, D_n]`. CortexNet's scalar W_intra is a simplification;
  structure otherwise closely parallel.
- `abandoned/v8-*` — scalar weight branches that CortexNet's scalar
  design partially echoes.
- `abandoned/v9-backprop*` — differentiable plasticity on per-neuron W;
  primitives (phi-based plasticity, Hebbian traces) inspire v0's
  per-edge Hebbian approach.

---

## 13. References

Trimmed to papers directly load-bearing for v0.

**Three-factor plasticity / eligibility traces**

[1] [Eligibility Traces and Plasticity on Behavioral Time Scales](https://consensus.app/papers/details/1d1341ffc747562b85dd37015c1b7459/?utm_source=claude_code) (Gerstner et al., 2018, 299 citations) — foundational three-factor learning framework: `Δw = pre × post × modulator`. CortexNet's scalar plasticity rule is this, exactly.

[3] [A silent eligibility trace enables dopamine-dependent synaptic plasticity](https://consensus.app/papers/details/39e2eaa5a0ce50e8a55b075431e66c5e/?utm_source=claude_code) (Shindou et al., 2018, 80 citations) — direct experimental evidence for eligibility traces gating plasticity.

**Test-time training / mesa-optimization**

[7] [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://consensus.app/papers/details/ea4cdf995acb589f93c13cdfa76dbe7f/?utm_source=claude_code) (Sun et al., 2024, 161 citations) — TTT: hidden state is an ML model, updated by SGD each step. CortexNet's "fast weights as state" is this.

[28] [Test-Time Training Done Right](https://consensus.app/papers/details/fb2c8417ffb05dc0a9a0fd6426418ce4/?utm_source=claude_code) (Zhang et al., 2025, 17 citations) — LaCT: 14B params with fast weights, 56K context.

**Differentiable plasticity**

[32] [Differentiable plasticity: training plastic neural networks with backpropagation](https://consensus.app/papers/details/af2e3b8fc73053629c5c72276a207481/?utm_source=claude_code) (Miconi et al., 2018, 170 citations) — the template for CortexNet's phase-1 training: fast weights are not parameters, but their init templates are.

**Predictive coding / multi-horizon prediction**

[22] [A tutorial on the free-energy framework for modelling perception and learning](https://consensus.app/papers/details/f7210495d7fd5003885659f62e8e7efb/?utm_source=claude_code) (Bogacz, 2017, 286 citations) — predictive coding = prediction-error-driven learning; motivates the K-horizon surprise gate.

**Cortex topology**

[8] [The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence](https://consensus.app/papers/details/c46aa9f57b0854b696b1993c4a70008c/?utm_source=claude_code) (Clay et al., 2024, 4 citations) — repeated-column AI. CortexNet's grid is the flat-2D version of this.

[19] [A Resilient, Low-Frequency, Small-World Human Brain Functional Network](https://consensus.app/papers/details/a6b931c3a20958a78aff54ba9339237e/?utm_source=claude_code) (Achard et al., 2006, 2388 citations) — sparse small-world cortex with hub structure. Motivates v0's K_long random long-range edges over pure grid.

[21] [Scalable training of ANNs with adaptive sparse connectivity inspired by network science](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code) (Mocanu et al., 2017, 688 citations) — Sparse Evolutionary Training.

[Paper consensus search (20 full results per query)](https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code)
