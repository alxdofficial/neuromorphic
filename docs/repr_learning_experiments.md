# Representation Learning Experiments

We are evaluating memory-augmented language model architectures by their
ability to compress text into a small memory bottleneck that a frozen
Llama-3.2-1B decoder then uses to reconstruct masked content. This
document covers the dataset, the two primary training protocols, the
baseline architectures we compare, the modifications we made for fair
comparison, and the current results.

V2.1 (our own architecture) is deferred from this writeup — it requires
per-sentence streaming design pending. The baselines below are A, B, MT,
Mamba, and Vanilla (loss floor).

---

## 1. Dataset

> **For per-source token statistics and worked examples of every task
> family**, see **[`docs/dataset_examples.md`](dataset_examples.md)** — it
> covers the 9 composite_v1 families plus HotpotQA, NarrativeQA, MuSiQue,
> BABILong with one fully-reconstructed example each plus a unified
> n_questions / token-budget census.

**FineWeb-edu** (HuggingFaceTB/fineweb-edu) — high-quality web text
filtered for educational content. Pretokenized with the Llama-3.2
tokenizer (`meta-llama/Llama-3.2-1B`, vocab=128,256).

- **Train**: 13,689 documents ≥ 4096 tokens (`data/wave1/fineweb_edu.train.parquet`)
- **Val**: 720 documents ≥ 4096 tokens (`data/wave1/fineweb_edu.val.parquet`)
- Per-document chunking depends on protocol:
  - HSM: 256-token chunks, paired (chunk₁, chunk₂) consecutively from the same doc
  - Sentence-recon: 4096-token chunks, split into sentences at terminator
    tokens (`.`, `!`, `?`)

The encoder ingests the chunk in its native form. Llama is **frozen** —
only its input embedding lookup and lm_head are reused; gradient flows
through Llama but no parameter update happens to it. The only Llama-side
trainable parameter is `decoder.mask_embed` (a single 2048-dim vector
inserted at masked positions).

---

## 2. Training Protocols

The two protocols test different aspects of "what does a memory
representation need to carry."

### 2.1 HSM — Hidden-State Matching (v1e, primary)

**Goal**: produce a memory that Llama can use as a substitute for the
verbatim previous chunk.

**Mechanism**:
- Take two consecutive 256-token chunks `(chunk₁, chunk₂)` from the same document
- Encoder compresses `chunk₁` into M memory tokens
- **Teacher forward**: frozen Llama on `[chunk₁_verbatim, chunk₂_masked]`
  → hidden states `h_teacher[chunk₂_positions]`
- **Student forward**: frozen Llama on `[memory_tokens, chunk₂_masked]` →
  hidden states `h_student[chunk₂_positions]`
- **Loss**: MSE(`h_student`, `h_teacher`) over all chunk₂ positions

The encoder has to compress chunk₁ such that Llama's downstream processing
of chunk₂ is preserved. This is a strong constraint: hidden states are
not just "predict-the-next-token" but encode all of Llama's intermediate
reasoning.

**Training**: 30k steps, batch=8. Memory bottleneck: ~11.6k floats per
chunk (16 slots × 725 floats at the pre-projection point).

Lineage: `scripts/repr_learning/train_repr_hsm.py`.

### 2.2 Sentence-recon — Sentence-level shuffled retrieval (v1g, primary)

**Goal**: produce a memory that supports random-access content prediction
across a long context, sentence by sentence.

**Mechanism**:
- Tokenize a 4096-token chunk; split into sentences at terminator tokens
- Encoder ingests the full chunk via **4 × 1024-token streaming writes**,
  carrying memory state across writes
- Per chunk, randomly pick **K=3 query sentences** (order shuffled)
- For each query sentence:
  - Randomly mask 80% of the sentence's tokens
  - Apply a **MaskGIT-style reveal curriculum**: a random fraction
    `r ∈ [0, 0.9]` of the masked positions get their GT exposed as
    "previously predicted" — the decoder sees these as visible
  - The remaining still-masked positions are CE-loss targets
- Decoder: frozen Llama with custom 4D attention mask:
  - Still-masked positions attend only to visible positions (unmasked +
    revealed) + memory + self
  - Visible/memory positions attend only to other visible/memory positions
  - Still-masked positions are **isolated from each other** — predictions
    are made in parallel given common visible + memory context, with no
    inter-prediction leak
- CE loss only at still-masked positions

**Training**: 500-step smokes so far (production runs to follow). Memory
bottleneck: 26,100 floats per query at the pre-projection point
(36 memory tokens × 725 floats, hitting a 20× compression target on the
4096-token input).

Lineage: `scripts/repr_learning/train_repr_sentence.py`.

---

## 3. Baselines

Each baseline is an existing memory architecture from the literature,
adapted to the protocols above. We picked the four memory variants to
span the architectural design space: discrete codebook, continuous slots,
retrieval-from-bank, and recurrent SSM. Vanilla is the no-memory floor.

### 3.1 Vanilla Llama (loss floor)

**Module**: `NullEncoder` — no encoder at all. Memory tokens count is 0.

The decoder runs Llama on the masked text directly (no memory prepended).
The only trainable parameter is `decoder.mask_embed`.

**Tests**: what can Llama do on this task without any side-car module?
If any of A/B/MT/Mamba can't beat this floor, their memory module isn't
contributing — the task is solvable from local context alone.

**Trainable params**: 2,048 (just the mask embedding).

### 3.2 A — Flat Codebook (`FlatBaselineEncoder`)

**Archetype**: VQ-VAE family — discrete codebook indexed via softmax-gated
picks.

**Mechanism**:
1. A small 2-layer bidirectional transformer encodes the input text
2. 36 learned slot queries cross-attend to the encoder hidden states
3. Each slot scores against a 4096-entry codebook; top-1 pick via
   Gumbel-STE (straight-through estimator on argmax, with Gumbel-softmax
   gradient surrogate)
4. Picked code embedding is projected to d_llama and used as a memory
   token

**Lineage**:
- VQ-VAE picks: van den Oord et al. 2017 ([arXiv:1711.00937](https://arxiv.org/abs/1711.00937))
- Switch-style load-balance auxiliary loss: Fedus et al. 2022 ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961))
- Top-1 routing into a large codebook: Lample et al. 2019, *Product-Key Memory* ([arXiv:1907.05242](https://arxiv.org/abs/1907.05242))
- Memory Layers at Scale (4096-code regime): Berges et al. 2024 ([arXiv:2412.09764](https://arxiv.org/abs/2412.09764))

**Modifications for fair comparison**:
- **Streaming slot state**: in v1g, slot queries persist across the 4
  streaming writes — they refine via cross-attention to each new window.
  Picks at the codebook only happen once at finalize.
- **Position-aware encoder**: the bi_transformer takes a `position_offset`
  argument so token@chunk_pos=1500 gets `pos_embed[1500]`, not the
  per-window-reset `pos_embed[476]` it would otherwise get (fixed in
  the v1g fairness audit pass).
- **Tests**: whether discrete code routing adds value over continuous
  alternatives at matched pre-projection budget.

**Trainable params**: ~14.9M (encoder + slot_attn + codebook + projections).

### 3.3 B — Continuous Slots (`ContinuousBaselineEncoder`)

**Archetype**: Slot Attention / Perceiver IO family — fixed-size set of
continuous latent vectors that compete for content via inverted attention.

**Mechanism**:
1. Same 2-layer bidirectional transformer encoder
2. 36 learned slot queries (orthogonally initialized)
3. **Inverted slot attention** (Locatello et al. 2020): softmax-over-slots
   rather than the usual softmax-over-keys, so each input token "votes"
   for which slot gets it — slots compete and specialize
4. Two rounds of inverted attention per write × 4 writes = 8 iterations
   total
5. Slot vectors are projected directly to d_llama (no quantization, no
   discrete pick) and used as memory tokens

**Lineage**:
- Slot queries + cross-attention: Carion et al. 2020, DETR ([arXiv:2005.12872](https://arxiv.org/abs/2005.12872))
- Perceiver IO latent decoupling: Jaegle et al. 2021 ([arXiv:2107.14795](https://arxiv.org/abs/2107.14795))
- Inverted attention competition: Locatello et al. 2020, *Object-Centric Learning with Slot Attention* ([arXiv:2006.15055](https://arxiv.org/abs/2006.15055))
- Diversity regularizer (cf. slot-redundancy reduction): Wu et al. 2024, AdaSlot ([arXiv:2406.09196](https://arxiv.org/abs/2406.09196))

**Modifications for fair comparison**:
- **Explicit diversity loss** on slot outputs (squared pairwise cosine
  averaged). Without this, B reliably collapses all 36 slots to one
  effective vector — Llama treats memory as a permutation-invariant set,
  giving no slot-level supervision otherwise.
- **Diversity scale retuned 1000 → 50**: at scale=1000 the diversity
  loss dominated total loss (62%) and starved recon of gradient. The
  diversity metric is structurally saturated at 1.0 regardless of scale,
  but the scale tuning lets recon learn properly.
- **Position-aware encoder**: same fix as A.
- **Tests**: whether discrete codebook structure (A) adds value over
  unstructured continuous slots at matched pre-projection width.

**Trainable params**: ~14.5M.

### 3.4 MT — Memorizing Transformer (`MemorizingBaselineEncoder`)

**Archetype**: per-token KV bank with top-K retrieval at query time
(Wu et al. 2022).

**Mechanism**:
1. Same bidirectional transformer encoder, applied per 1024-token window
2. Per-token KV head produces a (key, value) pair for every input position
3. After all 4 streaming writes, the bank holds 4096 (key, value) pairs
4. For each queried sentence (3 per chunk):
   - Pool a query vector from the sentence's visible (unmasked +
     revealed) positions
   - Score against all 4096 bank keys
   - Top-K=36 retrieved values are projected to d_llama and used as the
     memory for that sentence's reconstruction

**Lineage**:
- Memorizing Transformers (Wu et al. 2022, [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)) — the per-token bank + top-K retrieval recipe.

**Modifications for fair comparison**:
- **Streaming bi_transformer**: the encoder now runs once per 1024-token
  window (4 forwards per chunk) instead of once over the full 4096. This
  matches A/B's attention span. Before this fix MT had a 4× wider attn
  span and a corresponding ~4× FLOP advantage on the encoder.
- **Per-sentence retrieval, K=36 cap**: MT's bank stays uncapped at
  ~3M floats (4096 entries × 725), which is its defining feature, but
  the per-query *retrieval* is capped at K=36 — exactly matching the
  other baselines' M=36 memory-token count. Per-query memory exposure
  is therefore identical at 26,100 floats across all variants.
- **Raw-embed query pooling**: the query is now pooled from
  `bi_transformer.in_proj(raw_embeds)` — a pure linear projection of
  raw Llama embeddings without attention — rather than from the
  contextualized `text_h`. This eliminates a subtle info leak where
  bidirectional encoder attention could embed still-masked content into
  the visible positions used for query pooling.
- **Tests**: whether per-query retrieval from a large, uncompressed bank
  beats compression into a fixed-size slot set, at matched per-query
  memory exposure.

**Trainable params**: ~13.5M.

### 3.5 Mamba — Recurrent SSM (`RecurrentBaselineEncoder`)

**Archetype**: state-space model (Gu & Dao 2023) — linear-time recurrence
with selective hidden-state propagation.

**Mechanism**:
1. Input projection from d_llama down to `d_mamba`
2. Two Mamba blocks process the full 4096-token sequence
3. Per-token output narrowed to `d_recurrent=725`
4. Adaptive average pool to 36 memory tokens
5. Projected to d_llama

**Lineage**:
- Mamba: Gu & Dao 2023 ([arXiv:2312.00752](https://arxiv.org/abs/2312.00752))

**Modifications for fair comparison**:
- **`d_mamba`: 1024 → 768** to bring Mamba's trainable params into the
  same band as A/B/MT (~12.5M vs the original ~19M). At the default
  d_mamba=1024 Mamba had ~25% more params than the other memory variants.
- No bi_transformer to position-fix — Mamba's SSM uses no explicit
  positional embeddings, the recurrence naturally encodes position via
  the temporal scan.
- **Tests**: whether linear-time recurrent compression matches parallel
  slot-attention compression at matched bottleneck width.

**Trainable params**: ~12.5M.

---

## 4. Why This Comparison Is Fair

When comparing memory architectures, three things matter most: each
variant should have (a) roughly equal **trainable capacity**, (b) the
same **bottleneck budget** (the information the memory module is allowed
to pass to the decoder), and (c) roughly equal **encoder compute** so
that "memory module quality" is what's varying, not "amount of work."
The discussion below explains what we did for each, and how we measure
the bottleneck.

### 4.1 How we measure bottleneck width

The bottleneck is **measured at the pre-projection point in each
encoder** — i.e., the dimensionality of the memory state right before
each variant's final projection to Llama's hidden size (d_llama=2048).
We deliberately do *not* measure post-projection.

The reason: the projection from each variant's internal dim to d_llama
is essentially arbitrary (a learned linear layer). Two encoders that
hold the same amount of information internally can produce arbitrarily
different post-projection float counts depending on what we set d_llama
to. Measuring post-projection would penalize whichever architecture had
a narrower internal state, even though that state is the actual
information bottleneck.

Concretely, in our v1g configuration each variant's bottleneck is
**M memory tokens × per-slot dimensionality**, both measured pre-projection:

| variant | M | per-slot dim | pre-projection floats |
|---|---:|---:|---:|
| A | 36 codes | `d_concept_baseline=725` | **26,100** |
| B | 36 slots | `d_continuous=725` | **26,100** |
| MT | 36 retrieved values | `d_mt_value=725` | **26,100** |
| Mamba | 36 pooled outputs | `d_recurrent=725` | **26,100** |
| Vanilla | 0 | — | **0** |

All four memory variants are *exactly* matched at **26,100 floats per
query** at the pre-projection point. This is the core fairness invariant
we built the comparison around. The "20× compression" framing comes from
this number versus the input float count: 4096 input tokens × d=128
(`d_node_state`, the natural reference dim used by V2.1) gives 524,288
floats at the same dim → 524,288 / 26,100 ≈ 20× compression.

For HSM (v1e), the bottleneck is tighter: 16 memory tokens × 725 ≈
11,600 floats per chunk. Same matching principle — all five variants
are tied at that number.

### 4.2 Trainable parameter parity

Trainable params after the v1g fairness pass:

| variant | trainable params |
|---|---:|
| A | 14,875,350 (14.9M) |
| B | 14,532,309 (14.5M) |
| MT | 13,480,063 (13.5M) |
| Mamba | 12,522,453 (12.5M) |
| Vanilla | 2,048 (the mask embedding only) |

The four memory variants fit within a **2.4M-param band** (12.5M-14.9M),
roughly a 19% spread between heaviest (A) and lightest (Mamba). For
context, this is *much* tighter than the pre-fix state: Mamba was at
19M (+25% over A) until we dropped `d_mamba` from 1024 → 768.

This spread is small enough that none of the variants has a "capacity
buy" — any of them could in principle solve any of the others' tasks
with the params it has. Mamba being slightly lighter is now a
disadvantage in capacity rather than an advantage; if Mamba still
performs comparably, that's a genuine architectural finding rather than
"Mamba has more weights to throw at the problem."

Vanilla's 2k params is by design — it's the floor, the "what can Llama
do with mask_embed alone" reference. If memory variants don't beat it,
they're not contributing.

### 4.3 Encoder compute parity

Beyond parameters, compute matters: a variant that runs its encoder
4× could "see" more during ingestion even at matched params.

| variant | bi_transformer forwards/chunk | tokens per forward | max attention span | slot iterations |
|---|---:|---:|---:|---:|
| A | 4 | 1024 | 1024 | 4 |
| B | 4 | 1024 | 1024 | 8 |
| MT | 4 | 1024 | 1024 | 0 |
| Mamba | 0 (uses SSM) | — | 4096 (SSM-inherent) | 0 |
| Vanilla | 0 | — | — | 0 |

After the audit pass:
- A, B, and MT all run their bi_transformer the **same number of times**
  (4) over the **same attention span** (1024 tokens per forward). Slot
  iterations differ (B does 2 per write vs A's 1), but this is an
  architectural property of B (Slot Attention is designed iteratively),
  not a compute imbalance.
- Mamba's SSM is naturally O(T) — it processes the full 4096-token
  sequence with linear-time recurrence. We can't easily window it
  without breaking the architecture, so Mamba's "attention span" is
  inherently full-chunk. This is an architectural feature, not a
  fairness violation.

### 4.4 Accepted asymmetries

Two asymmetries we deliberately did not equalize because they would
amount to crippling each architecture's defining feature:

- **MT retrieves a different top-K per query**. The other three variants
  reuse the same 36 memory tokens across all K=3 query sentences.
  MT therefore has 3 × 26,100 = 78,300 unique floats exposed across the
  chunk vs the others' 26,100. We accept this because per-query selection
  *is* MT — it's the defining mechanism of memorizing transformers
  ([Wu et al. 2022](https://arxiv.org/abs/2203.08913)). The
  per-query budget is still matched at 26,100 floats.
- **MT's bank is uncapped at ingestion**. The full 4096 × 725 = ~3M
  floats of KV pairs are stored before retrieval. This is the architecture
  — without the bank MT has nothing to retrieve from. We cap at the
  *retrieval* step (K=36) so per-query exposure to the decoder is
  matched.

Both of these are noted limitations of the "per-query parity" framing
rather than bugs. They're documented openly so readers can interpret
results accordingly.

---

## 5. Results

### 5.1 HSM (v1e, 30k steps)

| variant | params | final val_loss_hsm |
|---|---:|---:|
| **continuous_baseline (B)** | ~15M | **0.2132** |
| recurrent_baseline (Mamba) | ~19M¹ | 0.3078 |
| memorizing_baseline (MT) | ~15M | 0.4621 |
| flat_baseline (A) | ~15M | 0.5071 |
| vanilla_llama | 2k | 0.5208 |

¹ HSM was trained with Mamba's pre-fix d_mamba=1024. A retrain at
d_mamba=768 is pending for clean apples-to-apples comparison.

### 5.2 Sentence-recon (v1g, 500 steps post-fix)

| variant | params | final val_loss_recon | top-1 acc¹ | top-5 acc¹ |
|---|---:|---:|---:|---:|
| **memorizing_baseline (MT)** | 13.5M | **6.20** | 15.4% | 30.8% |
| continuous_baseline (B) | 14.5M | 6.22 | 20.5% | 33.3% |
| recurrent_baseline (Mamba) | 12.5M | 6.34 | 17.9% | 23.1% |
| flat_baseline (A) | 14.9M | 6.99 | 17.9% | 30.8% |
| vanilla_llama | 2k | 8.67 | 2.6% | 10.3% |

¹ Top-1 / top-5 accuracy on still-masked positions, evaluated via
`scripts/repr_learning/inspect_v1g.py`. 500 steps is far from converged;
absolute numbers will shift with longer training.

### 5.3 Findings (current)

- **Memory variants beat vanilla floor** on both objectives. Sentence-recon
  zero-memory ablation shows +2.5–3.0 nat contribution from memory across
  A/B/MT/Mamba — the encoder is genuinely doing work.
- **B and MT alternate at the top** depending on objective. B wins HSM
  by a wide margin (0.21 vs Mamba's 0.31). MT wins sentence-recon (6.20)
  with B close behind (6.22).
- **Memory tokens collapse to ~1 effective vector** in every architecture
  at our v1g bottleneck (pairwise cosine ≈ 1.00 for all of A, B, MT,
  Mamba). The "36 slots" are architecturally fictional — we're running
  a 1-vector memory in all variants. This is consistent across designs
  and may be a fundamental limit of the bottleneck width rather than an
  architecture-specific issue.
- **500-step predictions are stopword-biased**: across all variants,
  70-97% of predictions land on a handful of common tokens (`.`, `,`,
  `the`, `The`). Whether content-aware prediction emerges with more
  training is an open question.
- **Vanilla floor matters**: at 500 steps vanilla gets 2.6% top-1 vs
  memory variants' 15–21%. The encoder provides a 7–8× lift on accuracy.

---

## 6. Open Items

1. **Longer sentence-recon runs** (10k+ steps) to see whether predictions
   become content-aware.
2. **Retrain Mamba on HSM** at `d_mamba=768` to land HSM rankings on the
   same param footing as v1g.
3. **V2.1 per-sentence streaming design** — our own architecture is not
   yet in the v1g comparison. Required for the full thesis test.
4. **Investigate the universal memory collapse** — is the bottleneck
   width fundamentally limiting? Per-slot supervision / identity loss /
   cross-window contrastive could be probes.

---

## File pointers

| Artifact | Path |
|---|---|
| HSM training | `scripts/repr_learning/train_repr_hsm.py` |
| Sentence-recon training | `scripts/repr_learning/train_repr_sentence.py` |
| Encoder definitions | `src/repr_learning/encoder.py` |
| Loss functions | `src/repr_learning/model.py` |
| Sentence data pipeline | `src/repr_learning/data_sentence.py` |
| v1g verification | `scripts/repr_learning/verify_v1g.py` |
| v1g prediction inspection | `scripts/repr_learning/inspect_v1g.py` |
| v1g fairness audit | `scripts/repr_learning/fairness_audit_v1g.py` |
| Cross-objective scoreboard | `docs/repr_learning_results.md` |
| Architectural lineage / citations | `docs/repr_learning_baselines.md` |

Last update: 2026-05-22, after v1g fairness audit fixes + 500-step re-sweep.
