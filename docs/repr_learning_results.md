# Representation Learning Results — Cross-Objective Summary

Centralized scoreboard for v1 representation-learning runs. Pairs with
`docs/repr_learning_baselines.md` (architectural lineage / citations) and
`docs/dataset_examples.md` (per-source token census).

> **Update 2026-05-25**: the val-sampling bug (task #614) is fixed —
> `materialize_val_set` pins a fixed val batch list so the same checkpoint
> evaluated twice now agrees to <0.01 nat. Numbers below the "v1h_t4k_v3"
> section use the corrected protocol; earlier sections still carry the
> streaming-val caveat.

**Objectives:**
- **HSM** — hidden-state matching, v1e. Frozen-Llama hidden-state MSE across
  cross-chunk pairs. Tests whether the encoder produces a representation that
  Llama can use as a substitute for the verbatim previous chunk.
- **Sentence-recon** — v1g. Sentence-level shuffled-retrieval reconstruction
  with restricted attention (MaskGIT-style). Tests whether the encoder produces
  a memory that supports random-access content prediction across a 4096-token
  context window.

All variants matched at the **pre-projection bottleneck (d_node_state level)**
within each objective; bottleneck size differs across objectives (see per-table
notes).

---

## 0. v1h_t4k_v3 — QA on composite_v1 + HotpotQA + NarrativeQA (2026-05-25, **current**)

The headline tranche: 5 trainable + 2 vanilla on the v1h QA loss
(per-token CE on the answer span). Protocol matches tranche-1-v2 for
direct comparison: chunk=4096, window=1024, mix [0.5, 0.25, 0.25]
(composite/hotpot/narrative). Max 20K steps with patience early-stop
(5 consecutive non-improving vals = 2500-step plateau detector).

This is the **first tranche with trustworthy val** (materialized fixed
val set, best.pt eval), and the **first with the graph_baseline P1
fixes** (u from pick_count popularity, state from picked, all-pad
post-recycle protection — see `docs/exp1_graph_baseline.md`).

| variant | val_recon | best_step | trained_to | notes |
|---|---:|---:|---:|---|
| **recurrent_baseline** (mamba) | **2.674** | 9500 | 10000 | tight val noise, clearly plateaued |
| continuous_baseline | 2.692 | 15500 | 18000 | longest-converging top variant |
| memorizing_baseline | 2.703 | 12000 | 14500 | top 3 within 0.03 nat |
| ─────────────────── | ─── | ─── | ─── | ─── |
| graph_baseline | 3.257 | 7500 | 10000 | high val variance — patience may have fired prematurely (see below) |
| flat_baseline | 3.396 | 6500 | 9000 | known underperformer at this scale |
| vanilla_full_context (no train) | 3.448 | 0 | — | in-context Llama ceiling reference |
| vanilla_llama (no train) | 5.115 | 0 | — | no-memory floor |

**Key reads:**
- Top tier (mamba, continuous, memorizing) tied within 0.03 nat. Mamba
  wins narrowly with the smallest param count.
- Graph (3.26) is **0.55 nat behind the top tier**, only 0.2 nat ahead
  of the no-train vanilla_full_context (3.45). The architecture trains
  cleanly but doesn't compete on this task at this scale.
- Vanilla floor (5.12) gives a 2.4-nat gap from the trained models —
  memory does meaningful work for everyone.

**Graph's val variance:**
Comparing oscillation amplitude over the last 5 vals:

```
GRAPH    7500-10000: 3.26 → 3.28 → 3.44 → 3.46 → 3.31 → 3.42  (±0.10 nat)
MAMBA    7500-10000: 2.67 → 2.69 → 2.72 → 2.68 → 2.67 → 2.68  (±0.025 nat)
```

Graph's noise is **4× larger** than mamba's. Suspected cause: stochastic
write dynamics (recycling picks vary with batch composition + popularity-
based admission). With patience threshold 1e-4 and patience=5, graph
plateaus in noise rather than in true convergence. The top1 accuracy at
step 8500 (41.4%) was actually higher than at the best-val step 7500
(39.3%) — consistent with val variance rather than true plateau.

→ Diagnostic in flight: longer graph run with disabled patience to test
if val continues to drop with more compute.

**What changed since tranche-1-v2 (pre-fix):**
- Graph P1 architecture fixes (u, state, recycle, all-pad protection)
- All-variant trustworthy eval (materialize_val_set + best.pt eval)
- Patience-based early stop (best.pt staleness criterion)

Lineage:
- Trainer: `scripts/repr_learning/train_repr_qa.py`
- Eval: `scripts/repr_learning/eval_best.py`
- Launch: `scripts/training/launch_t4k_v3_overnight.sh`
- Per-variant jsonl: `outputs/repr_learning/v1h_t4k_v3_<variant>/jsonl/`

---

## 1. Earlier scoreboard (v1e + v1g — pre-fix, streaming-val caveat applies)

Lower is better in every column. **Bold** = best memory variant per objective.

| variant | HSM val (v1e, 30k)¹ | Sentence-recon val (v1g, 500, post-fix)² |
|---|---:|---:|
| V2.1 (our model) | 0.3122 | *(streaming TBD)* |
| A (flat codebook) | 0.5071 | 6.99 |
| B (continuous slots) | 0.2132 | 6.22 (scale=50) |
| **MT (memorizing transformer)** | 0.4621 | **6.20** (K=36 per query) |
| Mamba (recurrent SSM) | 0.3078 | 6.34 |
| Vanilla (no memory, floor) | 0.5208 | 8.67 |

¹ Bottleneck ≈ 11.6k floats (16 slots × 725, or 30 fused edges × 384).
² Bottleneck ≈ 26.1k floats (36 slots × 725, or 68 fused edges × 384, target 20× compression of 4096 × 128).

**Observations:**
- **B wins HSM (0.21), MT wins sentence-recon (6.20)** — after 6 fairness
  fixes (see §3 "post-fix audit"), MT moved from 2nd to 1st on sentence-recon.
  B sits 0.02 behind MT — effectively a tie at 500 steps.
- **A trails on both objectives** — its discrete codebook routing seems to
  underperform continuous and retrieval-based alternatives at this scale.
- **V2.1 is third on HSM** among the discrete-routing models; its
  sentence-recon result is missing because per-sentence streaming isn't built
  yet.
- **Mamba is competitive on both** despite being a pure SSM with no
  slot/codebook structure (now also at the smallest param count: 12.5M).
- **Vanilla is the floor everywhere** — a ~2.4 nat gap on sentence-recon, 0.21 on HSM.

---

## 2. HSM — Hidden-State Matching (v1e)

**Setup.** 30k steps. Variants prepend their memory tokens to a masked chunk_2,
and the loss is MSE between frozen-Llama hidden states on `[chunk_1, chunk_2]`
(teacher: chunk_1 verbatim) vs `[encoder(chunk_1), chunk_2]` (student: memory).
The encoder has to compress chunk_1 well enough that Llama's downstream
processing of chunk_2 is preserved.

| variant | params | final val_loss_hsm |
|---|---:|---:|
| V2.1 | ~15M | 0.3122 |
| flat_baseline (A) | ~15M | 0.5071 |
| **continuous_baseline (B)** | ~15M | **0.2132** |
| memorizing_baseline (MT) | ~15M | 0.4621 |
| recurrent_baseline (Mamba) | ~19M | 0.3078 |
| vanilla_llama | 2k | 0.5208 |

**Findings:**
- B substantially best (0.21 vs next-best Mamba 0.31). Continuous-slot
  attention turns out to be well-suited to the cross-chunk teacher-forcing
  signal.
- V2.1 third (0.31). Comparable to Mamba on this objective despite its
  discrete-routing constraint.
- Vanilla and MT essentially tied at the bottom (0.52, 0.46). MT's
  per-token retrieval bank apparently doesn't help much for hidden-state
  matching at this scale.

Lineage: `scripts/repr_learning/train_repr_hsm.py`.
Plot: `outputs/repr_learning/v1e_plot.png`.

---

## 3. Sentence-recon — v1g (in progress)

**Setup.** 4096-token chunk split into sentences; encoder ingests via 4 ×
1024-token streaming writes; decoder reconstructs K=3 randomly-chosen
sentences from the chunk. 80% of each chosen sentence's tokens are masked;
a random fraction is "revealed" (GT exposed as if previously predicted) for
MaskGIT-style training. Restricted attention: still-masked positions can
see only visible + revealed + self (no inter-still-masked leak).

### Results @ 500 steps, post-fix (smoke; not converged)

| variant | params | final val_loss_recon |
|---|---:|---:|
| flat_baseline (A) | 14.9M | 6.99 |
| continuous_baseline (B), scale=50 | 14.5M | 6.22 |
| **memorizing_baseline (MT)**, K=36, scale=50 | 13.5M | **6.20** |
| recurrent_baseline (Mamba), d_mamba=768 | 12.5M | 6.34 |
| vanilla_llama (floor) | 2k | 8.67 |

V2.1 still excluded (needs per-sentence streaming design).

### Post-fix audit (six fixes applied 2026-05-22)

Before this point, several asymmetries were skewing the comparison. After
fixes the param/compute landscape is:

| variant | params | bi-xfwd | attn-span | M/query | pq_floats |
|---|---:|---:|---:|---:|---:|
| A | 14.9M | 4 | 1024 | 36 | 26,100 |
| B | 14.5M | 4 | 1024 | 36 | 26,100 |
| MT | 13.5M | 4 | 1024 | 36 | 26,100 |
| Mamba | 12.5M | 0 | 4096 (SSM-inherent) | 36 | 26,100 |
| Vanilla | 2k | 0 | — | 0 | 0 |

Fixes applied:
- **A/B/MT now use offset-aware positional encoding** (`SmallBiTransformer`
  takes `position_offset` so token@chunk_pos=1500 gets `pos_embed[1500]`
  not `pos_embed[476]`). A/B previously reset positions per window.
- **MT runs 4 × 1024-token bi_transformer forwards instead of 1 × 4096**
  (matches A/B's attention span; was previously 4× wider).
- **MT's query is pooled from `bi_transformer.in_proj(raw_embeds)`** — a
  pure linear projection of raw Llama embeds, no attention. Prevents
  bidirectional encoder attention from leaking still-masked content into
  the query.
- **Mamba `d_mamba`: 1024 → 768** to bring trainable params down from 19M
  to 12.5M (was +25% over A/B/MT).
- **`logits_to_keep=L_max`** on Llama forward (sentence positions only).
- **`--resume`** support in train_repr_sentence.py.

Effect on results: MT moved from 6.30 → 6.20, B from 6.10 → 6.22, A from
6.72 → 6.99, Mamba from 6.47 → 6.34. The biggest mover is A getting worse
— the position-reset bug was likely helping it overfit per-window patterns.
MT being best after fixes is notable: it lost its 4× attention span and
its query contamination, but won anyway.

**MT design notes.** MT's KV bank is naturally `4096 × 725 = 2.97M floats`
— ~113× larger than the other baselines' 26k-float memory budget. To
preserve per-query parity we cap MT's *retrieval* to K=36 tokens per
queried sentence (= the other baselines' memory token count), without
touching the bank. Each queried sentence pools its own query vector from
its visible (unmasked + revealed) positions, scores against the bank,
and retrieves top-36. Per-query memory exposure: 36 × 725 = 26,100 floats
= identical to the other variants. MT *does* retain one natural advantage:
it can retrieve a different 36 tokens per query (3 × 36 = 108 retrievals
across the chunk), while the other variants reuse the same memory for all
K=3 queries.

### Verification & inspection findings (after 500 steps)

Critical: we performed visual inspection (`scripts/repr_learning/inspect_v1g.py`)
and the picture is more nuanced than the loss table suggests.

**What's working:**
- Encoder IS being used. Zero-memory ablation: loss jumps 6.4 → 9.3 (Δ +2.9),
  top-1 accuracy drops 17–21% → 0% across all three memory variants.
- Memory variants beat vanilla on top-1 by 7–8× (17–21% vs 2.6%).
- Custom attention mask passes isolation checks; gradients flow correctly to
  encoder bottleneck params + decoder.mask_embed.

**What's concerning:**
- **Predictions are dominated by punctuation/stopwords**:

  | variant | top-5 most-predicted tokens (of 39 still-masked) | concentration |
  |---|---|---:|
  | A | `.`×19, `the`×9, `to`×3, `,`×3, `in`×2 | 92% on 5 tokens |
  | B | `,`×15, `The`×4, `.`×3, `I`×3, `under`×2 | 69% |
  | MT | `the`×15, `The`×8, `,`×3, `.`×2, `I`×2 | 77% |
  | Mamba | `,`×22, `the`×9, `to`×3, `.`×3, `of`×1 | 97% |

  The model has learned "predict the most common token" with mild memory-driven
  bias on which one. It is *not* predicting content words. Visual decode
  examples in `scripts/repr_learning/inspect_v1g.py` output.

- **Memory tokens collapse to ~1 effective vector** in every architecture:

  | variant | pairwise cosine of 36 memory tokens |
  |---|---:|
  | A | 1.000 |
  | B | 1.000 |
  | MT | 1.000 |
  | Mamba | 0.998 |

  Notably, MT's collapse is the most surprising — its 36 *retrieved* tokens
  come from a 4096-entry bank with per-query selection, yet they still
  collapse to one direction post-projection. This suggests `proj_value`
  homogenizes whatever differentiation existed in the raw retrieved values.

  The "36 slots" are architecturally fictional — we're effectively running a
  1-vector memory in all variants. This was previously known for B (we
  retuned the diversity scale and the slots stayed collapsed); confirming it
  for A and Mamba at v1g scale is new.

- **B's diversity loss is impotent at any reasonable scale.** Both
  `diversity_slots` and `diversity_mem` stay pinned at 1.000 throughout
  training. We retuned `b_diversity_scale` from 1000 → 50 because the loss
  was dominating (62% of total), freeing gradient budget for recon (Δ −0.46
  val improvement). The diversity values themselves did not move.

**Diagnosis.** Not a code bug — gradients flow, mask works, memory is used.
But the architecture is acting as "global topic vector + Llama" rather than
"36-slot memory bank + Llama." More training (10k+) may or may not change this.

### v1g verified pieces

- ✓ Data pipeline (`src/repr_learning/data_sentence.py`)
- ✓ Streaming write methods on A, B, Mamba, Vanilla
- ✓ MT per-sentence retrieval (`retrieve_per_sentence`) — bank built once
  in `finalize_memory`, then top-K=36 retrieval per queried sentence
- ✓ `compute_sentence_recon_loss` with restricted 4D attention mask +
  MT bank/retrieval branch
- ✓ Training script (`scripts/repr_learning/train_repr_sentence.py`)
- ✓ Verification suite (`scripts/repr_learning/verify_v1g.py`):
  attention construction, restricted-attention behavior, gradient flow,
  aux balance — all 5 variants pass
- ✓ Inspection script (`scripts/repr_learning/inspect_v1g.py`):
  top-1/5 accuracy, decoded predictions, zero-memory ablation, memory
  diagnostics (MT uses retrieve_per_sentence ablation path)
- ✓ Legacy v1e still works after encoder refactor

---

## 4. Cross-cutting findings

**Memory collapse is universal at this scale.** All four memory architectures
(A, B, MT, Mamba) collapse their N memory tokens to ~1 effective vector at
our v1g bottleneck (36 tokens × 725 floats). This is independent of:
- Discrete vs continuous quantization
- Slot-attention iterations
- Diversity penalty magnitude
- Architecture family (transformer slot attention, SSM, per-query retrieval
  from a 4096-entry bank)

Whatever is happening — Llama uses the memory as a soft prompt with effective
rank ≪ M — is consistent across designs. This may mean the bottleneck
*should* be conceptualized as "best 1-vector summary of the chunk" rather
than "best 36-slot index." If so, scaling M may waste capacity.

**Vanilla floor matters more than expected.** On sentence-recon, vanilla
(`mask_embed` only, 2k trainable params) gets 2.6% top-1 vs 17–21% for memory
variants. Memory provides a strong improvement on top of the vanilla floor,
but the absolute gap (8.67 → 6.10) is what the encoder is "buying."

**Rankings — B leads on both, MT helps on sentence-recon, A/MT swap.**
- HSM ranking: B ≫ Mamba > V2.1 ≫ MT > A ≈ vanilla
- Sentence-recon ranking (500 steps): B > MT > Mamba > A ≫ vanilla

B leads both objectives. MT moves from 4th on HSM to 2nd on sentence-recon
— consistent with MT's design (per-query retrieval is more useful when
the task is per-sentence). V2.1 still needs to be added to sentence-recon.

---

## 5. Open questions

1. **Will sentence-recon predictions become content-aware at 10k+ steps?**
   At 500 steps they're mostly stopwords with mild memory bias. Need to
   run longer and re-inspect.
2. **Is the universal memory collapse a fundamental limit, or fixable?**
   All four memory variants collapse to cos≈1 across their 36 tokens —
   including MT despite per-query retrieval. If fundamental, our "36
   slots/tokens" architectures are misnamed; if fixable, we need to
   identify the missing pressure (per-slot supervision? slot identity
   loss? cross-window contrastive?).
3. **V2.1 per-sentence streaming** — design + implementation pending. This
   is the actual architectural thesis test.
4. **Whether B's slot collapse undermines the B win.** B is best on both
   objectives but is functionally a 1-slot architecture. Is the win a real
   architectural finding or an artifact of having the *least* structural
   constraint?
5. **Why does MT collapse despite per-query retrieval?** MT's bank has 4096
   distinct positions and the query is per-sentence, yet the 36 retrieved
   tokens post-projection have cos≈1. Likely culprit: `proj_value`
   homogenizes them. Worth probing.

---

## File pointers

| Artifact | Path |
|---|---|
| Architectural lineage | `docs/repr_learning_baselines.md` |
| HSM training | `scripts/repr_learning/train_repr_hsm.py` |
| Sentence-recon training | `scripts/repr_learning/train_repr_sentence.py` |
| v1g verification | `scripts/repr_learning/verify_v1g.py` |
| v1g prediction inspection | `scripts/repr_learning/inspect_v1g.py` |
| v1e plot | `outputs/repr_learning/v1e_plot.png` |
| v1e jsonl | `outputs/repr_learning/v1e_<variant>/jsonl/` |
| v1g jsonl | `outputs/repr_learning/v1g_<variant>/jsonl/`, `v1g_bd50_continuous_baseline/jsonl/` |

Last update: 2026-05-22, after MT added to v1g with per-sentence retrieval cap.
