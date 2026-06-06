# Representation Learning Baselines: Lineage & Citations

This doc maps each baseline in `src/repr_learning/encoder.py` to the
existing-work archetype it represents, the specific mechanisms it
borrows, and any modifications we made for our text-window
reconstruction setup. Cite this in any publication that uses
these baselines.

We currently compare **graph_v5 (our model) + 4 baselines + vanilla** in the trainer:
- **graph_v5** (`graph_v5_baseline`) — **OUR MODEL**: shared node bank + soft-pointer edges + message-passing readout (see `docs/exp1_graph_v5_design.md`)
- **A** (`vqvae_baseline`) — flat discrete codebook (VQ-VAE family) + dead-code revival, prepend reads
- **B** (`slot_attention_baseline`) — continuous slots via **canonical Slot Attention** (Locatello 2020: GRU + 3-iter refinement + stochastic shared-Gaussian init, no diversity loss), prepend reads
- **MT** (`memorizing_transformer_baseline`) — per-token KV bank with **per-position retrieval** (faithful Memorizing Transformers), prepend reads
- **Mamba** (`mamba_baseline`) — **canonical Mamba** SSM encoder (4-layer, RMSNorm pre-norm, official `selective_scan_cuda` kernel), prepend reads
- **vanilla** (`vanilla_llama`) — no-memory loss floor

> **Retired** (no longer in the active sweep): `plastic_baseline` (Hebbian
> fast-weights), `splat_baseline` (Gaussian mixture), `graph_baseline` (the
> pre-v5 edge-memory design). **The model moved from the codebook-based V2.1
> (described below for historical context) to the current `graph_v5` substrate**
> — treat "V2.1" sections as lineage, not the current candidate.
> The 2026-05-29 baseline-fidelity audit restored canonical Slot Attention (B),
> per-position retrieval (MT), and dead-code revival (A), and replaced the
> handicapped 2-layer-no-norm Mamba with the canonical 4-layer version.

All variants are matched at a comparable pre-projection bottleneck
width per window (~26,000 floats in current v1h sizing) — the precise
formulas differ by substrate. Per-architecture float-counts are
reported by `scripts/repr_learning/verify_v1h.py`.

- **Decoder**: frozen Llama-3.2-1B; only encoder-side params trainable
- Trainable param counts vary because each architecture's natural
  allocation differs (codebook size, slot mechanism, recurrent state,
  fast-weights matrix, edge bank, etc.)

## V2.1 (our model)

**Closest archetype**: [Yang et al. 2018/2020 — *Scene Graph Auto-Encoder
(SGAE)*](https://arxiv.org/abs/1812.02378) (788 citations) — uses a learned
dictionary D to reconstruct sentences via the pipeline
`S → G → D → S`, where G is a scene graph (object/attribute/relation nodes)
and D is a learned codebook. V2.1 generalizes this from image-captioning's
scene graphs (5-7 hand-coded node/edge labels) to arbitrary text with a
learned 4096-entry codebook and learned edges. The structural-graph-as-
bottleneck-for-reconstruction idea has clear precedent here.

Other components combine several prior ideas:

| Component | Source |
|---|---|
| Graph-as-learned-bottleneck for text reconstruction | [Yang et al. 2018 — *Scene Graph Auto-Encoder* (SGAE)](https://arxiv.org/abs/1812.02378) — closest archetype |
| Discrete codebook + Gumbel-STE picks | [van den Oord et al. 2017 — *Neural Discrete Representation Learning* (VQ-VAE)](https://arxiv.org/abs/1711.00937) |
| Switch-style load-balance auxiliary loss | [Fedus et al. 2022 — *Switch Transformers*](https://arxiv.org/abs/2101.03961) |
| Slot queries cross-attending to inputs | [Carion et al. 2020 — DETR](https://arxiv.org/abs/2005.12872) |
| Learned codebook with `N >> n_picks` | [Berges et al. 2024 — *Memory Layers at Scale*](https://arxiv.org/abs/2412.09764) |
| Per-edge (src, edge, dst) triple structure | Novel — closest in spirit to TPR / HRR symbolic binding |
| Modifier MLPs (concept + query → instance vector) | Novel — TPR-style role × filler decoupling |

**Differences from SGAE**: SGAE uses a fixed scene-graph parser on text
(hardcoded relation types); we learn the codebook end-to-end via Gumbel-STE.
SGAE evaluates on image-captioning ceiling; we evaluate on span-masked
text reconstruction against a frozen Llama decoder. SGAE's "dictionary" is
typed by scene-graph categories; ours is fully learned.

**Tests:** whether typed (src→edge→dst) discrete vocabulary beats
flat alternatives at compressing a text window into 96 memory tokens.

## Baseline A — Flat codebook (`VQVAEBaselineEncoder`)

**Archetype**: VQ-VAE / Memory Layers at Scale family — discrete codebook
indexed via softmax-gated picks.

| Component | Source |
|---|---|
| Discrete codebook + Gumbel-STE picks | [van den Oord et al. 2017 — VQ-VAE](https://arxiv.org/abs/1711.00937) |
| Switch load-balance auxiliary loss | [Fedus et al. 2022 — Switch Transformers](https://arxiv.org/abs/2101.03961) |
| Slot queries / DETR-style learned query bank | [Carion et al. 2020 — DETR](https://arxiv.org/abs/2005.12872) |
| Top-1 routing into a large codebook | [Lample et al. 2019 — *Product-Key Memory*](https://arxiv.org/abs/1907.05242) |
| `4096 codes × 725 dims` codebook scale | [Berges et al. 2024 — Memory Layers at Scale](https://arxiv.org/abs/2412.09764) |

**Differences vs V2.1**: no edge structure (96 independent slots, not 32
triples), no modifier MLPs, smaller codebook width (725 vs 1024).
Diversification comes from the load-balance loss alone — A's slot
queries are simple normal init (no orthogonal, no inverted attention)
because load-balance pressure is sufficient.

**Tests:** whether V2.1's (src→edge→dst) typed structure adds value
over a flat bag of 96 discrete picks from the same codebook.

## Baseline B — Continuous slots (`SlotAttentionBaselineEncoder`)

**Archetype**: Slot Attention / Perceiver IO family — fixed-size set of
continuous latent vectors that cross-attend to inputs.

| Component | Source |
|---|---|
| Cross-attention from learned slot queries to inputs | [Carion et al. 2020 — DETR](https://arxiv.org/abs/2005.12872), [Jaegle et al. 2021 — Perceiver IO](https://arxiv.org/abs/2107.14795) |
| Inverted attention (softmax-over-slots competition) | [Locatello et al. 2020 — *Object-Centric Learning with Slot Attention*](https://arxiv.org/abs/2006.15055) |
| Orthogonal slot-query initialization | [Saxe et al. 2014 — *Exact solutions to nonlinear dynamics*](https://arxiv.org/abs/1312.6120) |
| Diversity / pairwise-cosine regularizer | Common across the slot literature; cf. [Wu et al. 2024 — *AdaSlot*](https://arxiv.org/abs/2406.09196) on slot-redundancy reduction |

**Faithful Slot Attention (restored 2026-05-29 audit)**: B now implements
the canonical Locatello et al. 2020 recipe — `slot_iters=3` iterative
refinement with **shared weights**, a **GRU update** (input = attention
update, hidden = previous slots), a residual MLP, and **stochastic
shared-Gaussian slot init** (slots sampled from one learned N(μ, diag σ)
shared across slots). It uses **no diversity loss** (`b_diversity_scale=0`):
the stochastic init + GRU + iterative competition prevent collapse by
mechanism, as in the paper. Eval uses a fixed persistent noise vector for
determinism.

> Previously B used a single inverted-attention pass + deterministic
> orthogonal init + a pairwise-cosine diversity penalty. An external audit
> found the diversity loss is **not** standard in the slot literature and
> that dropping the GRU + iteration was a material handicap — both corrected.

**Tests:** whether graph_v5's discrete structure adds value over
unstructured continuous slots at matched bottleneck width.

## Baseline 4 — Memorizing Transformers (`MemorizingTransformerBaselineEncoder`)

> **MT is a different memory class from A / B / plastic / splat / graph.** Its raw
> memory bank is ~23,200 KB vs the bottleneck-matched variants' ~26 KB — about
> 900× larger. MT is included as a **published reference architecture**, not as
> an apples-to-apples competitor. **Expect MT to score well** because it has
> dramatically more memory; the interesting comparison is among the
> bottleneck-matched class (A, B, plastic, splat, graph) which all share the
> ~26 KB budget.

**Archetype**: kNN / retrieval-augmented memory family — store
verbatim, retrieve at decode.

| Component | Source |
|---|---|
| Verbatim per-token KV bank | [Wu et al. 2022 — *Memorizing Transformers*](https://arxiv.org/abs/2203.08913) |
| kNN retrieval from KV memory | [Khandelwal et al. 2020 — *kNN-LM*](https://arxiv.org/abs/1911.00172) |
| Single-query pooling from unmasked positions | Novel — adapted for span-masked reconstruction (the original MT paper retrieves per query position; we use a single pooled query for budget parity with V2.1's 96-token output) |
| Top-K hard retrieval + STE for gradient | [Locatello et al. 2020 — STE form for discrete picks](https://arxiv.org/abs/2006.15055) |

**Setup (per-position retrieval restored 2026-05-29 audit)**: every
question token produces its own query; bank keys are scored against all of
them and **max-pooled** (a key is kept if ANY decoding position wants it),
then hard top-K=96 with a soft STE gate for gradient. This restores MT's
core per-position mechanism while still emitting a fixed 96 memory tokens
for budget parity. No diversity loss (`mt_diversity_scale=0`) — canonical
MT has none, and per-position retrieval diversifies naturally.

> Previously the QA path pooled the question to a **single** query. An audit
> found this was the dominant handicap — MT scored *worst* on biographical
> fact-lookup (F1 0.04), exactly where a retrieval method with ~900× more
> memory should dominate. Corrected to per-position.

**Tests:** establishes where a retrieval-augmented architecture (with a
large verbatim bank) lands on this benchmark. Performance better than
the bottleneck-matched variants is expected and not the win condition
for our own designs — what matters is whether structured compression at
26 KB approaches the retrieval-augmented reference at 23,200 KB.

## Baseline 5 — Mamba SSM (`MambaBaselineEncoder`)

**Archetype**: State-space-model / recurrent sequence model family.

| Component | Source |
|---|---|
| Mamba selective state-space model | [Gu & Dao 2024 — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752) |
| Per-token bottleneck via linear projection | Standard transformer hidden-dim narrowing pattern |
| Adaptive pooling from T tokens to N memory tokens | [F.adaptive_avg_pool1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool1d.html) |

**Note on no slot collapse**: Mamba's per-token outputs are inherently
position-distinct (the state evolution gives each position a different
hidden state), so the pooled 96 memory tokens are diverse by
construction. No diversity regularizer needed.

**Tests:** whether parallel slot-attention compression matches
recurrent state-space compression at matched bottleneck.

## Vanilla Llama (`NullEncoder`)

**Archetype**: No encoder. Frozen Llama-3.2-1B with a trainable
mask_embed vector at masked positions. This is the **loss floor** —
what Llama achieves on the span-masked reconstruction task without any
side-car memory module.

| Component | Source |
|---|---|
| Llama-3.2-1B | [Touvron et al. 2023 — Llama 2; Meta 2024 — Llama 3](https://arxiv.org/abs/2307.09288) |
| Mask-embedding for masked LM | [Devlin et al. 2019 — BERT](https://arxiv.org/abs/1810.04805); [Raffel et al. 2020 — T5](https://arxiv.org/abs/1910.10683) |

**Tests:** sets the reference — any side-car encoder must beat this
floor to demonstrate that the memory module is contributing
non-redundant information.

## Summary table — what each baseline tests against V2.1

| Baseline | Removed from V2.1 | Question |
|---|---|---|
| Vanilla Llama | All structure | Is *any* memory module beneficial? |
| A (flat codebook) | Edge structure | Does (src→edge→dst) triple help over flat codes? |
| B (continuous) | Codebook + edges | Does discrete vocabulary help over continuous slots? |
| 4 (MT) | Compression | Does compression beat retrieval? |
| 5 (Mamba) | Parallel slots | Does parallel slot attention beat recurrent compression? |

## Per-baseline known issues from literature + our mitigation status

For each baseline we surveyed the foundational paper(s) + recent follow-ups
to catalog known failure modes. This section documents what we applied,
what we deliberately skipped, and why — so reviewers can see the
"full-capacity" version of each baseline was attempted.

### A (flat_baseline) — VQ-VAE family

**Canonical failure mode**: codebook collapse — "only a small subset of
codevectors receive gradients useful for their optimisation, whereas a
majority simply 'dies off' and is never updated"
([Zheng et al. 2023, CVQ-VAE](https://arxiv.org/abs/2307.15139)).

| Literature fix | Source | Status |
|---|---|---|
| Switch-style load-balance loss | Fedus et al. 2022 | ✅ Applied |
| Gumbel-STE for stable gradient | Jang et al. 2017 | ✅ Applied |
| Use actual STE picks in load-balance | (this work) | ✅ Applied (post-audit fix) |
| Dead-code revival | [CVQ-VAE Zheng 2023](https://arxiv.org/abs/2307.15139) | ✅ Applied (2026-05-29 audit — EMA usage tracking + reseed dead codes from heavy users; A had collapsed to ~341/4096) |
| EMA codebook update | [van den Oord 2017 §3.2](https://arxiv.org/abs/1711.00937) | ❌ Skipped — Gumbel-STE substitutes |
| Auxiliary-loss-free balancing (DeepSeek-style dynamic bias) | [Wang et al. 2024](https://arxiv.org/abs/2408.15664) | ❌ Skipped — load-balance + STE was sufficient |
| FSQ (Finite Scalar Quantization, no codebook) | [Mentzer et al. 2023](https://arxiv.org/abs/2309.15505) | ❌ Architectural alternative; not comparable to V2.1's discrete codebook |

### B (continuous_baseline) — Slot Attention / Perceiver IO family

**Canonical failure mode**: slot collapse — slots converge to identical
outputs when the loss has no per-slot supervision
([Locatello et al. 2020](https://arxiv.org/abs/2006.15055),
[DIAS / Zhao et al. 2025](https://arxiv.org/abs/2507.23755)).

| Literature fix | Source | Status |
|---|---|---|
| Inverted softmax-over-slots attention | Locatello 2020 | ✅ Applied (simplified, single-iteration) |
| GRU update + iterative refinement (T=3) | Locatello 2020 | ✅ Applied (2026-05-29 audit — shared-weight iteration + GRU restored) |
| Stochastic shared-Gaussian slot init | Locatello 2020 §3.2 | ✅ Applied (replaced deterministic orthogonal init) |
| Slot-diversity regularizer | — | ❌ Removed — non-canonical; collapse now prevented by init + GRU + iteration |
| Orthogonal slot init | Saxe et al. 2014 | ❌ Replaced by canonical stochastic shared-Gaussian init |
| Slot-contrastive loss | [Slot-BERT, Liao et al. 2025](https://arxiv.org/abs/2501.06481) | ❌ Subsumed by our diversity loss |
| Slot re-initialization / self-distillation | [DIAS Zhao 2025](https://arxiv.org/abs/2507.23755) | ❌ Skipped — collapse already prevented |

### Baseline 4 (memorizing_baseline) — Memorizing Transformers

**Canonical practice** from [Wu et al. 2022](https://arxiv.org/abs/2203.08913): per-layer kNN retrieval into
non-differentiable memory; learnable gate to blend local attention vs
memory attention; memory scales to 262K tokens with steady gains.

**Our setup differs significantly** — these are intentional simplifications
for budget parity (96 memory tokens to Llama like the other baselines):

| Original MT feature | Our variant | Why we simplified |
|---|---|---|
| Per-layer retrieval inside transformer | Single retrieval up-front, memory prepended to Llama | Frozen Llama; can't insert per-layer hooks without major surgery |
| Per-position queries (one per attention head per token) | Single pooled query from unmasked positions | Need fixed 96 output tokens; per-position would produce T × heads queries |
| Learnable gate to blend local + memory attention | No gate (just prepend) | Llama already does attention over [memory; text] |
| Memory size scales to 262K | Memory = 256-token KV bank from one window | Per-chunk task; multi-window persistence is Phase 2 |

**Known failure mode in our simplified variant**: collapse to a single
effective vector — the bi-transformer at random init produces similar
per-position values, top-K retrieval picks 96 similar vectors → memory
collapses (disp → 1.0). Same symptom as B's slot collapse, different cause.

| Fix | Status |
|---|---|
| Apply diversity loss (same recipe as B) | ⏳ Will apply if collapse persists through training |

### Baseline 5 (recurrent_baseline) — Mamba

**Known issues from literature**:
- HIPPO-based initialization for the A matrix is critical
  ([Solozabal et al. 2025](https://arxiv.org/abs/2505.18266))
- Careful `dt_init`, `dt_init_floor` for the dt parameter
  ([Liu et al. 2024](https://arxiv.org/abs/2411.19455))
- Training instability at large Mamba stacks
  ([GroupMamba Shaker 2024](https://arxiv.org/abs/2407.13772))

**Our setup (canonical, post-2026-05-29 audit)**:
- Official `mamba_ssm` library — inherits HIPPO init + proper dt init;
  fast `selective_scan_cuda` + `causal_conv1d` kernels confirmed engaged
- **4 Mamba layers** at d_model=1024 with **per-block pre-norm RMSNorm**
  (`h = h + mixer(RMSNorm(h))`) + a final RMSNorm — the canonical residual recipe
- fp32 master weights under bf16 autocast (standard mixed precision; not
  full-bf16, which would move the optimizer to bf16 and degrade quality)

> Previously 2 layers at d1792 with **no per-block norm** — a non-canonical,
> shallow config that handicapped Mamba. Replaced after the audit. (The
> single full-sequence sweep is Mamba's normal, efficient mode, not a cost.)

### V2.1 (our model)

Combines VQ-VAE issues (codebook collapse) + slot-attention issues
(edge_queries homogenization).

| Mitigation | Source / Status |
|---|---|
| Load-balance loss with actual STE picks | ✅ Applied (Switch Transformers + post-audit fix) |
| Modifier MLPs (instance-level deltas on top of codebook) | ✅ Applied (novel) |
| Separate proj_src / proj_dst / proj_edge | ✅ Applied — 3-way architectural symmetry breaking |
| Codebook learnable logit_scale | ✅ Applied (init at 1.0, learns to grow) |
| Dead-code revival | ❌ Skipped — could add for longer runs |
| Stochastic edge_query init | ❌ Skipped — load_balance + 3-way projections handle it |



**JEPA (Joint Embedding Predictive Architecture)** —
[Assran et al. 2023 (I-JEPA, 769 citations)](https://arxiv.org/abs/2301.08243).
Predicts target *representations* (in embedding space) from a context block,
using two encoders (context + EMA target) and no token-level loss. Excluded
because:

1. JEPA tests a fundamentally different protocol — embedding-space MSE
   prediction with no decoder — that doesn't compose with our frozen-Llama
   decoder setup. Any JEPA-style baseline would need its own evaluation
   harness, breaking apples-to-apples comparison.
2. There's no canonical pure-text JEPA in the literature. All current
   JEPAs are vision-based
   ([I-JEPA](https://arxiv.org/abs/2301.08243),
   [Point-JEPA](https://arxiv.org/abs/2404.16432),
   [3D-JEPA](https://arxiv.org/abs/2409.15803))
   or multimodal
   ([VL-JEPA](https://arxiv.org/abs/2502.07770),
   [TI-JEPA](https://arxiv.org/abs/2503.15534)). Building a text-JEPA from
   scratch and tuning it to convergence would be a research effort in itself.
3. It tests a different question. V2.1 is about "can structured discrete
   memory tokens compress a window into a form Llama can decode?" JEPA is
   about "can a model predict semantic content of masked regions in
   embedding space?" Both are interesting; they're not direct competitors.

**Hopfield / associative memory networks** — modern Hopfield (Schlag et al.,
Ramsauer et al.) and matrix-memory architectures could be a baseline but
operate at a different abstraction (associative key-value retrieval rather
than compressed bottleneck for reconstruction). Excluded for similar
protocol-mismatch reasons.

## Citations file

Bibtex entries available at the bottom of this document. (TODO: add
bibtex on first publication.)

---

## plastic_baseline — Hebbian fast weights

**Closest archetype**: [Schmidhuber 1992 — *Learning to control fast-weight memories*](https://www.semanticscholar.org/paper/Learning-to-Control-Fast-Weight-Memories%3A-An-to-Schmidhuber/3eedf48ed1a9f9bef0c46a91be5f33ae2887d77c) and the modern revival in [Schlag, Irie & Schmidhuber 2021 — *Linear Transformers Are Secretly Fast Weight Programmers*](https://arxiv.org/abs/2102.11174).

**Mechanism**: a per-layer plastic matrix `M ∈ R^{d_h × d_h}` is updated by Hebbian outer-products of (key, value) pairs derived from the input. Reads project a query to a key, retrieve M·k, project back. We keep `plastic_depth` parallel matrices for capacity. Update rule is the standard `M ← decay·M + η·(v ⊗ k)`.

**Differences from canonical Schlag-style**: per-position MemInject reads (decoder hidden state queries the fast weights), not per-token-prepend. This tests "fast weights as inline retrieval" rather than "fast weights as compressed context vector." Update writes happen once per chunk (not per token) to match streaming-window protocol of the other variants.

---

## splat_baseline — Gaussian mixture memory

See **[`docs/exp3_gaussian_splat_baseline.md`](exp3_gaussian_splat_baseline.md)** for the full design. Short version: memory is K signed Gaussians in d_latent space; reads emit ray probes from each decoder position and integrate the signed-density field along each ray. Influenced by 3D Gaussian Splatting (Kerbl et al. 2023) translated to text-memory.

---

## graph_baseline — bounded edge memory with expert-choice routing

See **[`docs/exp1_graph_baseline.md`](exp1_graph_baseline.md)** for the full design. Short version: 68 (src, dst, state) edge slots; per-window, a small transformer produces 68 candidate edge updates; expert-choice routing has each existing endpoint pick the most-similar new proposal (geometric clustering ⇒ node reuse without learned gates); saliency = EMA of pick affinity (derived, not learned); slot recycling via percentile-based admission (`novelty > u`). Directional R-GCN-style readout with cross-edge message passing.

---

## vanilla_llama — no-memory loss floor

The frozen Llama-3.2-1B decoder running with **zero memory tokens prepended**. Only trainable parameter is `mask_embed` (single 2048-dim vector). Establishes the reconstruction loss floor: any memory architecture that doesn't beat this number isn't doing anything useful.
