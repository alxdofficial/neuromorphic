# Representation Learning Baselines: Lineage & Citations

This doc maps each baseline in `src/repr_learning/encoder.py` to the
existing-work archetype it represents, the specific mechanisms it
borrows, and any modifications we made for our text-window
reconstruction setup. Cite this in any publication that uses
these baselines.

All five baselines (V2.1 + four) and `vanilla_llama` are matched at:
- **Pre-projection bottleneck width**: ~69,600 floats per chunk
- **Post-projection memory tokens to Llama**: 96 × 2048 = 196,608 floats
- **Decoder**: frozen Llama-3.2-1B; only encoder-side params trainable

Trainable parameter counts vary because each architecture's natural
allocation differs (codebook size, slot mechanism, recurrent state).

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

## Baseline A — Flat codebook (`FlatBaselineEncoder`)

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

## Baseline B — Continuous slots (`ContinuousBaselineEncoder`)

**Archetype**: Slot Attention / Perceiver IO family — fixed-size set of
continuous latent vectors that cross-attend to inputs.

| Component | Source |
|---|---|
| Cross-attention from learned slot queries to inputs | [Carion et al. 2020 — DETR](https://arxiv.org/abs/2005.12872), [Jaegle et al. 2021 — Perceiver IO](https://arxiv.org/abs/2107.14795) |
| Inverted attention (softmax-over-slots competition) | [Locatello et al. 2020 — *Object-Centric Learning with Slot Attention*](https://arxiv.org/abs/2006.15055) |
| Orthogonal slot-query initialization | [Saxe et al. 2014 — *Exact solutions to nonlinear dynamics*](https://arxiv.org/abs/1312.6120) |
| Diversity / pairwise-cosine regularizer | Common across the slot literature; cf. [Wu et al. 2024 — *AdaSlot*](https://arxiv.org/abs/2406.09196) on slot-redundancy reduction |

**Modifications vs original Slot Attention**:
- We use a **simplified** inverted-attention block (no GRU update, no
  iterative refinement) for two iterations. Sufficient for our setting
  and avoids the per-step overhead of the full Locatello recipe.
- We add an **explicit diversity loss** on memory tokens (squared
  pairwise cosine) scaled to compete with reconstruction CE. Without
  this, B reliably collapses all 96 slots to one effective vector
  because the reconstruction loss has no slot-level supervision (Llama
  treats memory tokens as a permutation-invariant set). The Slot
  Attention paper's stochastic-init + iterative-GRU mechanism could
  also work, but adding a direct diversity regularizer is the standard
  fallback in the slot-attention literature when collapse persists.
- We did NOT add the GRU update or stochastic slot sampling — those
  add training-time cost and our diversity loss already prevents
  collapse in this setting.

**Tests:** whether V2.1's discrete codebook structure adds value over
unstructured continuous slots at matched bottleneck width.

## Baseline 4 — Memorizing Transformers (`MemorizingBaselineEncoder`)

**Archetype**: kNN / retrieval-augmented memory family — store
verbatim, retrieve at decode.

| Component | Source |
|---|---|
| Verbatim per-token KV bank | [Wu et al. 2022 — *Memorizing Transformers*](https://arxiv.org/abs/2203.08913) |
| kNN retrieval from KV memory | [Khandelwal et al. 2020 — *kNN-LM*](https://arxiv.org/abs/1911.00172) |
| Single-query pooling from unmasked positions | Novel — adapted for span-masked reconstruction (the original MT paper retrieves per query position; we use a single pooled query for budget parity with V2.1's 96-token output) |
| Top-K hard retrieval + STE for gradient | [Locatello et al. 2020 — STE form for discrete picks](https://arxiv.org/abs/2006.15055) |

**Modifications**: hard top-K (rather than per-position kNN as in the
original MT paper) so the encoder outputs a fixed 96 memory tokens to
Llama. Added a soft STE gate so gradient flows back to the
key-producing weights despite the hard selection.

**Tests:** whether structured compression (V2.1, A, B) beats retrieval
from a verbatim per-token KV bank at matched memory-token budget.

## Baseline 5 — Mamba SSM (`RecurrentBaselineEncoder`)

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
| Dead-code revival | [CVQ-VAE Zheng 2023](https://arxiv.org/abs/2307.15139) | ❌ Skipped — A converged with 341/4096 codes used at 30K steps; not yet load-bearing |
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
| Slot-diversity regularizer | Slot-BERT, DIAS, general orthogonality regularizer literature | ✅ Applied (1000× scaled squared-cosine penalty) |
| Orthogonal slot init | Saxe et al. 2014 | ✅ Applied |
| Stochastic slot init (sample from learned distribution) | Locatello 2020 §3.2 | ❌ Skipped — diversity loss + orthogonal init covers the symptom; reduces non-determinism for eval |
| GRU update + iterative refinement (T=3) | Locatello 2020 | ❌ Skipped — simplified to single inverted-attention pass; principled but expensive |
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

**Our setup**:
- Uses the official `mamba_ssm` library — inherits HIPPO init + proper
  dt initialization automatically
- Only 2 Mamba layers — well below the instability regime
- LayerNorm + residual connections present

**No fixes applied**: we trust the library's defaults. Mamba should train
healthily in our setting; we'll confirm at health-check time.

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
