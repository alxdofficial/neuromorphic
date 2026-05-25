# Exp 1 — `graph_baseline` design

Canonical reference for the graph-structured memory variant. Earlier
prototype iterations (with snap_gate / L_connectivity / learned saliency)
are kept in git history only; this doc describes the current
implementation.

## TL;DR

Bounded continuous-vector memory of K_max=68 edge slots. Per-window:
write candidates produced via cross-attention to the input, then assigned
to existing endpoints via **expert-choice routing** (geometric clustering
forces node reuse). Read path uses **directional R-GCN-style** transforms
plus cross-edge message passing. Saliency is **derived from observable
pick-affinity** (not a learned scalar) and gates readout + slot eviction.

No auxiliary loss. No learned routing scalars. The only knobs are
structural (K_max, decay rates, eviction cap) and downstream task loss.

## Slot state

Each of K_max=68 slots holds:

| field | shape | semantics |
|---|---|---|
| `src` | `[d_node=128]` | source endpoint vector |
| `dst` | `[d_node=128]` | destination endpoint vector |
| `state` | `[d_state=128]` | edge "label" / relational content |
| `u` | scalar | EMA of pick-affinity in [0, 1]; saliency |
| `age` | scalar | windows since last overwrite (cold-start guard) |

Total: 385 floats × 68 slots = 26,180 floats per example. Matches the
shared bottleneck convention used by the other variants.

At init: `src`, `dst`, `state` from learned `nn.Parameter` seeds (one set
per slot). `u = 1.0` (PER-style max-priority cold-start). `age = grace`
so initial slots are protected for one grace cycle then compete on merit.

## Per-window write protocol

The encoder consumes the chunk in fixed-size windows (default
`window_size=1024` → 4 windows per 4096-token chunk in tranche 1; v1h
training overrides to 256-token windows → 16 windows per chunk). For each
window:

### 1. Encode pins
```
pins = pin_encoder(llama_token_embeds) + sinusoidal_PE       # [B, N=256, d_updater=384]
```

### 2. Generate K_max=68 proposals (GraphUpdater)
Small 3-layer transformer. Encodes the K_max current slot states as
tokens (no `slot_pos` per **fix A** — slots distinguished by content
alone). 3× (cross-attention to pins → self-attention among slots → FFN).
Output head emits residual deltas:
```
proposed_src   = old_src   + Δsrc
proposed_dst   = old_dst   + Δdst
proposed_state = old_state + Δstate
```
No snap_gate, no keep_gate, no saliency_delta — just the three residuals.

### 3. Expert-choice routing (**fix B**)
```
endpoints           = concat(old_src, old_dst, dim=slot)        # [B, 2K, d_node]
proposed_endpoints  = concat(proposed_src, proposed_dst, dim=slot)
affinity            = cosine(endpoints, proposed_endpoints)     # [B, 2K endpoint, 2K proposal]

# Self-mask: endpoint i CANNOT pick proposal i (its own residual). Without
# this the diagonal would dominate and routing would never reach others.
affinity[diag] = -inf

picked_idx[i]       = argmax_j affinity[i, j]                   # each endpoint picks its top-1 NON-SELF proposal
pick_strength[i]    = affinity[i, picked_idx[i]]
margin              = sqrt(2·log(2K) / d) ≈ 0.28                # expected nearest-of-2K random cosine
α[i]                = sigmoid(8 · (pick_strength[i] - margin))  # near-zero unless pick is clearly above random

new_endpoint[i]     = (1 - α[i]) · old_endpoint[i] + α[i] · proposed_endpoints[picked_idx[i]]
```

Node reuse emerges geometrically (k-means attractor dynamics): endpoints
near "Alice region" all pick Alice-ish proposals, all update toward them,
all become closer to each other.

### 4. State update
Each endpoint pick is also a vote to merge with that proposal's slot
identity. State is gathered **from the slots that were picked**, not from
the slot's own proposal (a previous P1 audit bug). Combined via per-endpoint
α-weighting, so if both endpoints picked the same slot the state is
unambiguous; if they picked different slots, the higher-confidence
endpoint dominates.
```
slot_of_pick[i]    = picked_idx[i] % K
state_from_src[k]  = proposals.state[slot_of_pick[k]]
state_from_dst[k]  = proposals.state[slot_of_pick[K+k]]
picked_state[k]    = α-weighted average of (state_from_src[k], state_from_dst[k])
α_state[k]         = max(α[k], α[K+k])
new_state[k]       = (1 - α_state[k]) · old_state[k] + α_state[k] · picked_state[k]
```

### 5. Saliency update — popularity, not selectivity
```
pick_count[p]      = # endpoints whose argmax landed on proposal p   # column reduction
popularity[k]      = max(pick_count[k], pick_count[K+k]) / (max + 1) ∈ [0, 1)
u_new[k]           = 0.95 · u_old[k] + 0.05 · popularity[k]          # half-life ≈ 13 windows
```

`u` is the only saliency quantity. **Not** a learned scalar — derived
from observable routing assignments. Critically, it measures *being picked
by others* (popularity, column-reduction of the affinity argmax), not
*how confidently this slot picked others* (selectivity, row-reduction).
DNC-style usage tracking; previously was wired to α (selectivity) which
made u nearly uniform across slots.

### 6. Slot recycling (eviction + creation)
```
novelty[p]        = 1 / (1 + pick_count[p])                     # in (0, 1]; orphan proposal → 1.0
# Per-slot novelty = max of src/dst sides — admit if EITHER side is orphaned.

# Top-N candidates by rank
N                                   = max(1, ⌈0.05 × K_max⌉) = 3
top_novelty,  top_prop_idx          = novelty.topk(N)
bot_u,        victim_idx            = (-u_masked).topk(N)       # u masked to +∞ where age < grace

# Pure comparative admission — no thresholds.
# `>=` (not `>`): at cold start all slots have u=1.0 and orphan novelty=1.0;
# `>` would block cold-start admission on that exact tie. Tie defaults to admit.
admit                               = top_novelty >= bot_u

# Apply admitted pairs (variable count in [0, N])
for each admitted pair:
    slots[victim] ← (proposed.src, proposed.dst, proposed.state) at top_prop
    age[victim]   = 0
    u[victim]     = u.max()                                     # PER cold-start
```

Variable count per window in [0, N]. Quiet windows (no novel content OR
no dead slots) produce zero overwrites. Busy windows produce up to N=3.

**All-pad row protection.** Recycle runs after the per-row all-pad
restoration of the routing step; the protection is then *re-applied* to
`edges_new` and the `overwrite_mask` is zeroed for all-pad rows. Without
this, a batch row with no real tokens could have its substrate
destroyed even though its routing was correctly restored. Audit-found
bug, fixed.

### 7. Age
`age += 1` for every slot.

## Read protocol (`GraphReadout`, **fix E**)

```
# Directional encoding — W_src and W_dst are SEPARATE transforms
h = W_src(src) + W_dst(dst) + W_state(state)                    # [B, K, d_hidden=512]

# Cross-edge message passing (one self-attention layer over the K edges)
h ← h + MultiHeadAttention(h, h, h)
h ← h + FFN(h)

# Project to Llama
memory = LayerNorm(Linear(h))                                   # [B, K, d_llama=2048]

# Saliency gate (fades dying slots) — ABSOLUTE threshold, not median-centered.
# u ∈ [0, 1]. u=0 → gate ≈ 0.018; u=0.5 → gate = 0.5; u=1 → gate ≈ 0.98.
# Median-centering was tried earlier but a uniformly-dead bank still gets
# gate=0.5; absolute threshold actually fades dead substrate.
gate   = sigmoid(8 · (u - 0.5))
memory = memory * gate
```

`W_src ≠ W_dst` means "(Alice, mother-of, Bob)" and "(Bob, mother-of,
Alice)" produce different memory tokens by construction — directional
asymmetry is hard-wired, not learned-and-prayed.

The K_max=68 memory tokens are prepended to Llama's input. Llama
cross-attends to them during decoding.

## Three derived signals from one primitive

Everything reduces to operations on the per-window affinity matrix
`[B, 2K endpoints, K proposals]`:

| signal | per | derivation | role |
|---|---|---|---|
| `α` (update strength) | endpoint | sigmoid(8 · (row-max − margin)) over the self-masked affinity row | how strongly to overwrite this endpoint |
| `u` (saliency) | slot | column-sum (pick_count) → popularity = pc/(pc+1) → per-slot max → EMA | gates readout + ranks eviction candidates |
| `novelty` | proposal | 1 / (1 + column-sum) | candidate for admission to a new slot |

None is a learned scalar. The model can only influence them indirectly
by shaping (a) the proposals it produces and (b) the existing endpoints
those proposals are compared against — both of which train naturally
against task loss.

## Evolution from earlier prototypes

Earlier prototype iterations attempted to enforce graph structure via
learned gates and auxiliary losses:

| earlier component | current replacement | reason for change |
|---|---|---|
| `slot_pos` per-slot positional embedding | dropped | gave the decoder a content-free identity tag to memorize; forced slot specialization rather than content-driven reuse |
| `snap_gate_src`, `snap_gate_dst` | absorbed into expert-choice affinity | trivial-zero exploit: emit proposals far from anything, max_sim small, no penalty, no reuse pressure |
| `keep_gate` | absorbed into α (sigmoid of pick affinity) | combinatorial mess with snap_gate; redundant gating paths |
| `L_connectivity` aux loss | removed | only fired when proposals coincidentally landed near existing endpoints; structural top-1 picking removes the need |
| `L_adjust` aux loss | removed | gameable by lowering saliency to dodge the penalty |
| `saliency_logit` learned scalar | derived `u` from pick-affinity EMA | nothing externally pressured it to mean "informative slot"; drifted to whatever helped loss |
| `soft_snap` with self-mask and top-k attention | replaced by expert-choice top-1 | snap was decorative when the gates failed to fire |

A graph-structure probe on the prototype found these failure modes:
- 0% endpoint reuse (snap_gate had no real teeth)
- snap_gate saturated at 0.984 but firing on nothing useful
- saliency stuck near init across all slots
- 100% cross-window stagnation

All four are addressed at the design level in the current implementation
by removing gameable components and replacing them with structural
mechanics (geometric clustering via expert-choice, EMA-derived saliency,
percentile-based recycling).

## Config defaults (`ReprConfig`)

```python
graph_K_max: int = 68              # edge budget
graph_d_node: int = 128            # endpoint dim
graph_d_state: int = 128           # edge state dim
graph_updater_layers: int = 4
graph_d_updater: int = 384
graph_d_proj_hidden: int = 1024

# Expert-choice + saliency knobs
graph_u_decay: float = 0.95               # EMA decay for saliency (half-life ≈ 13 windows)
graph_grace_windows: int = 4              # cold-start immunity for newly-overwritten slots
graph_max_overwrites_fraction: float = 0.05  # ceiling: ≤ max(1, int(5% × K)) per window (= 3 for K=68)
graph_update_strength_scale: float = 4.0  # sigmoid steepness for α from cosine
graph_readout_n_heads: int = 4            # cross-edge attention heads
graph_readout_d_hidden: int = 512         # readout hidden dim
```

Trainable param count: ~16M (matches A/B/MT/Mamba's 12-15M band).

## Citations behind the design

- **Expert-choice routing**: Zhou et al. 2022, "Mixture-of-Experts with Expert Choice Routing" ([arXiv:2202.09368](https://arxiv.org/abs/2202.09368))
- **R-GCN directional transforms**: Schlichtkrull et al. 2017, "Modeling Relational Data with Graph Convolutional Networks" ([arXiv:1703.06103](https://arxiv.org/abs/1703.06103))
- **EMA usage tracking (`u` formulation)**: Graves et al. 2016, "Hybrid computing using a NN with dynamic external memory" (DNC, Nature)
- **PER-style max-priority cold-start**: Schaul et al. 2016, "Prioritized Experience Replay" ([arXiv:1511.05952](https://arxiv.org/abs/1511.05952))
- **S3-FIFO probationary period (grace window)**: Yang et al. SOSP 2023, "FIFO Queues are All You Need for Cache Eviction"
- **VQ-VAE dead-code revival pattern**: Razavi, van den Oord, Vinyals 2019, "VQ-VAE-2" ([arXiv:1906.00446](https://arxiv.org/abs/1906.00446))

## Status

- Implementation complete in `src/repr_learning/graph_substrate.py` + `encoder.py:GraphBaselineEncoder`/`GraphReadout`.
- Smoke tests pass. End-to-end with frozen Llama: all encoder params receive gradient.
- Structure probe (`scripts/repr_learning/probe_graph_v3.py`) confirms: self-pick masked (rate 0.0), gate response u=1 vs u=0 differs by 54×, u std across slots = 0.088 (not uniform).
- All-pad row protection verified post-recycle (row substrate preserved when no real tokens).

## Empirical results (v1h_t4k_v3, 2026-05-25)

After all P1 fixes + post-recycle all-pad protection:

```
val_recon (lower=better, trustworthy materialized-val protocol):

  recurrent_baseline (mamba)   2.674   ← winner
  continuous_baseline          2.692
  memorizing_baseline          2.703
  ───────────────────────────────────
  graph_baseline               3.257   ← 0.55 nat behind top
  flat_baseline                3.396
  vanilla_full_context (no-train)  3.448
  vanilla_llama (no-mem floor)     5.115
```

**Read.** Graph trains stably and beats the no-memory floor by 1.86 nat,
but is **0.55 nat behind** the simpler top-tier baselines and only 0.2
nat ahead of in-context Llama without training. The architecture is
correct (mechanics healthy, gradients flow) but not yet competitive at
this scale on this task mix.

**Caveat — likely under-trained.** Graph's val curve oscillates ±0.1 nat
between adjacent checkpoints in the "plateau" region; mamba oscillates
±0.025 nat. The patience early-stop (5 evals without 1e-4 improvement)
fired at step 10000 likely from val variance, not true convergence. Top1
accuracy at step 8500 (41.4%) was higher than at the best-val step 7500
(39.3%), inconsistent with a true plateau. The stochastic write
dynamics (recycling decisions vary with batch composition) plausibly
inflate weight-update variance between vals. A longer run with relaxed
patience is in flight.

**What the result tells us — irrespective of training duration:**
- The graph mechanics are not buggy; they're orthogonal to whatever
  signal mamba/continuous extract from this data mix.
- composite + hotpot + narrative is sequential-text-prediction; graph's
  relational inductive bias (edges as compositional facts) doesn't
  obviously help when the questions are mostly atomic-fact lookup or
  passage paraphrase.
- The architecture may shine on tasks where edges-as-relations carry
  load: kinship chains (CLUTRR), state tracking (Boxes), versioned
  facts. None of those are weighted heavily in this mix.
