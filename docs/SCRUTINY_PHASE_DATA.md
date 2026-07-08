# Architecture-Scrutiny Phase: Datasets, Generation & Objectives

This is the reference for **how the training data works** during the architecture-scrutiny phase — the
sweep where we ask *which memory architecture best serves an always-on, lifelong, implicit memory
layer*. It documents every source, how episodes are generated, the objectives, and the calibrated
sweep regime. If you touch the data layer, read this first; if it disagrees with the code, the code
wins — fix the doc.

> **What we're measuring.** A **frozen** SmolLM2-135M decoder reads `M` memory tokens produced by a
> small **trainable** encoder (~7M params, capacity-matched across arms). We compress a long context
> into those `M` tokens and train end-to-end. The open problem is **loss-neutrality** (plain CE barely
> rewards the encoder for *binding* information). The sweep compares encoder architectures (icae / ccm /
> autocompressor / beacon / slotgraph* / vqicae / biomem) on a mixed objective that stresses
> **fidelity** (can memory store) and **addressing** (can memory be selectively read).

---

## 1. The 4-layer data model

Data is factored into four **orthogonal** concepts (key insight: **task ≠ objective** — they compose
as a matrix). Adding a dataset never touches the trainer.

| Layer | Question it answers | Where it lives |
|---|---|---|
| **Source** | *where do tokens come from?* | `src/memory/data/sources/<name>.py` → `SOURCE_REGISTRY` |
| **Task** | *what is presented + asked?* | `src/memory/data/tasks/<style>.py` → `TASK_STYLES` / `get_task` |
| **EpisodeSpec** | *how hard?* (difficulty knobs) | `src/memory/data/schedule.py` |
| **Objective** | *how is it scored?* | `src/memory/training/objectives.py` |

Orchestration: `src/memory/data/mixes.py` (`TASK_SPEC`) maps each mixed-task name to a `(Source, Task)`
pair + an addressing profile; `src/memory/training/data_mix.py` builds one dataloader per task from an
`EpisodeSpec` via the registries. The trainer (`loops.py`) round-robins over those per-task loaders.

### The 3-part item shape (fact / question / answer)

Every source yields items with the three parts **separated**, which is what lets us "write a bunch of
facts, then ask about a particular one":

- **Keyed items** (`KeyedItem`, from `bio`): `write = "key = value"`, `query = "key ="`, `answer = value`.
  The question is *derived from* the write (they share the key) — an explicit pointer to one fact.
- **QA items** (`QAItem`, from babi + the RC sources): `facts` (a passage), `question`, `answer`.
  The question is *independent* of the write; when several passages are packed, the others become
  distractors and the answer lives in exactly one.

---

## 2. Sources (where tokens come from)

Every source has one folder under `sources/` and is bounded (local-jsonl-first, HF fallback, robust
skip on unreachable). `kind` gates which tasks accept it.

### Corpus sources (`kind="corpus"` → mae / continuation)
| source | content | note |
|---|---|---|
| `fineweb` | web text (FineWeb) | backs **reconstruct** (mae task-style). `src_tokenizer_name="meta-llama/Llama-3.2-1B"` (gotcha) |
| `multicorpus` | fineweb + pile + redpajama + code | backs **continuation** (VARIETY); robust union, skips unreachable |
| `pile`, `redpajama`, `code` | Pile-10k / SlimPajama-6B / codeparrot-clean | reachable **only inside** `multicorpus` |

### Keyed source (`kind="keyed"` → reconstruction)
| source | content | note |
|---|---|---|
| `bio` | procedural biographical **world** (~410 entities, 7 types) | `build_scenario` at runtime; train/val name-firewall; `key = value` where the value packs `n_facts` attributes with per-span content marks |

### QA sources (`kind="qa"` → qa)
| source | content | note |
|---|---|---|
| `babi` | bAbI relational-reasoning stories | `pack_isolated` (single story per episode — see §4) |
| `qa_multi` | union of the 5 below (random sub-source per item) | backs **doc_qa**; `pack_n_queries=(1,2)` |
| `squad` | SQuAD 2.0 (unanswerable → "unanswerable") | |
| `triviaqa` | TriviaQA (answer-safe sentence merges) | carries answer aliases in `meta` |
| `hotpot_train` | HotpotQA **train** split | gold paragraphs first, capped ~900 tok; **firewalled** from the eval reader's validation split |
| `musique_train` | MuSiQue **train** split | multi-hop; capped ~900 tok; firewalled |
| `multiwoz` | MultiWOZ dialogues → slot-recall QA | from canonical GitHub JSON |
| `quality` | QuALITY (abstractive MC) | registered but **not** in the default mix (needs ctx≥4096) |

Unwired scaffolding (registered, intentionally inert): `mqar`, `ruler_overwrite` (+ `tasks/overwrite.py`),
`quality`. Every QA source tags `meta["dataset"]` so per-example provenance flows into `task_family`.

---

## 3. Tasks (what is presented + asked)

A Task turns source items into one training episode (a pre-collate sample dict consumed by
`collate_qa` → `compute_loss`). Four styles:

| task | source kind | shapes | objective it feeds |
|---|---|---|---|
| **mae** | corpus | contiguous `total_len` span; answer = the span; loss on all (masking done in the loss) | fidelity |
| **continuation** | corpus | compress a prefix, predict the next `predict_len`; **multi-horizon** (§5) | gist under accumulation |
| **qa** | qa | pack passages + distractors, ask questions (§4) | addressing / retrieval |
| **reconstruction** | keyed | pack `key = value` lines, ask keys, score value-spans (§4) | addressing / binding |

`qa` and `reconstruction` are ~15-line adapters over one shared packer (§4). `mae`/`continuation` are
their own small readers.

---

## 4. The shared streaming packer (`tasks/_pack.py`)

`qa` and `reconstruction` are both "pack facts, ask about some." That's factored into
`Unit(write, query, answer, answer_spans, answer_exclude, refs)` + `pack_streaming_episode(...)`:

- **keyed →** `write="key = value\n"`, `query="key ="`, `answer=value`, `answer_spans=value_subs`,
  `answer_exclude=(name, given)` → loss scores **only the un-guessable value fragments**, never the
  entity name or template scaffolding.
- **qa →** `write=<facts>`, `query=<question>`, `answer=<answer>`, whole (short) answer scored.

What the packer guarantees:

- **Causality (by construction).** Query facts are packed FIRST (so they always fit), the question is
  asked at the very END (after the whole context), so every queried fact is written before it's asked.
- **Un-guessability.** `filler_ok` rejects any distractor that contains a queried answer verbatim, so
  the answer lives in exactly one packed passage (real retrieve-among-noise).
- **Fill-to-budget.** Distractors fill to `total_len`; the fill loop breaks on 12 consecutive
  *size*-misses (answer-leak rejections don't count — otherwise small-vocab sources stall) and
  over-samples `nq + max(n_inputs, total_len//24)` candidates so small-context sources fill.
- **`query_lag` placement.** `"early"` (front / long retention lag), `"recent"` (back / recency),
  `"any"` (shuffled), `"vary"` (sampled per episode).
- **Multi-query.** `n_queries` sampled `1..max` per episode; extra queries appended with inline cues
  (cues not scored).
- **Efficiency.** Each write is tokenized once (cached) and the context is built by concatenating the
  cached ids — no re-tokenization of the joined string.

### bAbI is special: multi-segment entity-rename (`pack_rename`)

A bAbI item is tiny (~86 tokens), so one story in a 2048 context would be 96% padding and trivial
(memory bigger than input). Instead bAbI **co-packs ~24 stories to fill the budget** — but bAbI reuses
a tiny name pool (Mary/John/…), so two stories' "Mary" would collide ("which Mary?" → ambiguous
supervision). So `pack_rename = True`: each co-packed segment gets a **disjoint set of entities** —
`BabiSource.rename()` detects PEOPLE (subjects of action verbs, via a data-derived `people_vocab`) and
OBJECTS (`object_vocab`) and replaces them with fresh names popped from shuffled pools; **locations are
left untouched** (they're answers, not queried subjects, so a shared "kitchen" is fine). The same map
is applied to facts + question + answer, so a person/object answer stays consistent. Query 1–3 segments
(multi-query). Result: bAbI fills ~2048 at a real compression ratio and becomes a
**retrieve-the-right-segment + bind-within-it** task, not a trivial floor.

---

## 5. Multi-horizon streaming continuation

Continuation compresses a prefix and predicts the next `predict_len` block **at every streaming-window
boundary**, so one episode tests memory at growing compression horizons:

```
compress GT[0:256]   → predict GT[256:320]     (score CE)
compress GT[0:512]   → predict GT[512:576]     (score CE)      ← memory ACCUMULATES across windows
  ⋮                                             (8 boundaries at ctx=2048, window=256)
compress GT[0:2048]  → predict GT[2048:2112]   (score CE)      ← the classic single-shot horizon
loss = mean of the 8 CEs  →  ONE backward
```

Key points: input is always **ground truth** (teacher-forced window-to-window, no autoregressive
rollout); memory is **persistent/accumulating** (boundary `b` compresses all of `[0:b]`); intermediate
targets are sliced from `context_ids`, the final target is `answer_ids` (just past the compressed
span). Implemented in `model.compute_streaming_continuation_loss`; auto-active when `window_size <
total_len`; disable with `--no-continuation-multi-horizon`. `n_horizons` caps the deepest boundaries.

---

## 6. Objectives (how it's scored)

`src/memory/training/objectives.py` (a flat file; composes with any task):

- **plain** — teacher-forced CE on answer content-mask positions. The default.
- **behavioral_kl** — context distillation: teacher = frozen LM on `[full context ‖ query]`, student =
  frozen LM on `[memory ‖ query]`; forward-KL on answer spans (teacher stop-grad, differentiable, no
  RL). `E[KL] = I(context; answer)`, so gradient concentrates where memory matters — the published fix
  for loss-neutrality. `--objective-mode behavioral_kl --kl-coef 2 --kl-temp 2`.
- **contrastive** (InfoNCE) / **trajectory** (exact Plackett-Luce) — GradCache two-pass variants.

The **REAL / SHUF / OFF** binding gate (eval) measures whether memory is used *structurally*: REAL =
normal; SHUF = decode with another example's memory; OFF = no memory. REAL ≪ SHUF ≈ the goal;
REAL ≈ SHUF = memory ignored (the binding-failure signature).

---

## 7. The sweep regime (difficulty calibration)

The calibration principle: **the memory is slots-and-structure limited, not bits limited.** M=96 ×
d=576 ≈ 55K floats ≈ 442K bf16-bits ≫ the task's ~6K bits of real information — so we do *not* starve
`M` (that would make it an uninformative capacity test); difficulty comes from the **number of
distinct bindings + interference**, and `M` stays generous so "no room" is never the excuse. The
discriminator is SHUF−REAL / EM, not compression ratio. MAE is the exception — verbatim reconstruction
at 21:1 is the deliberate fidelity probe.

| knob | value | source of truth |
|---|---|---|
| frozen decoder | SmolLM2-135M (d=576) | `--backbone` |
| memory `M` | **96** (~55K floats; ~430× smaller than the full 2048-ctx KV cache) | `DEFAULT_MIXED_M` |
| context `total_len` | **2048** (21:1 compression) | `--mixed-ctx` |
| `window_size` | **256** (8 streaming windows, ~paragraph) | `--window-size` |
| default mix | reconstruct · babi · doc_qa · continuation · fact_recall | `DEFAULT_TRAIN_MIX` |

**Per-SOURCE addressing profile** (`Source.pack_n_queries`, sampled `1..max` per episode + feasibility-
capped by the packer — item size varies by source, so a single per-task constant can't be right for the
`doc_qa` union). Fill is always budget-driven, so the *segment count* falls out of item size.

| mix-task (source) | pack_n_queries | query_lag | why |
|---|---|---|---|
| fact_recall (bio) | 1..3 | vary | tiny unique facts → high addressing pressure |
| doc_qa (qa_multi) | 1..2 | vary | big contexts (~800–900 tok) → only ≤2 golds fit at 2048 |
| babi | 1..3 | vary | ~24 renamed segments packed → query several |
| continuation | — | — | uses multi-horizon, not n_queries |
| reconstruct (mae) | — | — | scales by length + `mask_ratio` only |

Old mix-task names (`mae`/`qa_rc`/`condrecon_bio`) still work via `TASK_ALIASES`.

Seeds are **decorrelated per task** (`train_seed + i·10007`) so tasks that share a doc pool
(mae=fineweb, continuation⊇fineweb) don't draw the same passage in lockstep.

---

## 8. Data flow (end to end)

```
Source.sample(rng, n) → items
   → Task.build(source, spec, tok, rng, pad) → per-sample dict {context_ids, context_mask,
       question_ids, answer_ids, answer_content_mask_list, task_family, question_type, answer_refs}
   → collate_qa (+ base._collate stamps k_slots / mask_ratio / n_horizons)
   → model.compute_loss(batch, window_size)  — routed by model.task_mode:
        masked_reconstruction → compute_masked_reconstruction_loss
        continuation (+ multi-horizon) → compute_streaming_continuation_loss
        else → the generic prepend-memory + teacher-forced CE path
```

The decoder **never sees raw context** — only the `M` memory tokens carry it. Every context tensor is
exactly `total_len` (fixed compression denominator), so there is zero inter-sample context padding;
batches are single-task (round-robin) so answer padding stays small.

---

## 9. Diagnostics & telemetry

| tool | what it does |
|---|---|
| `scripts/diagnostics/mixed/episode_peek.py` | pull episodes through the **real** harness loaders; decode a sample for eyeballing + per-(task×datasource) fill/pad/causality/frequency/`builds-per-episode` stats |
| `scripts/diagnostics/mixed/data_stats.py` | per-source input/answer length + padding + answer-in-context percentiles |
| `scripts/diagnostics/mixed/mixed_data_audit.py` | structure/invariant/firewall audit over the exact train+val loaders |
| `TaskDataset.n_builds / n_episodes` | resample telemetry — `builds/episode > 1` means the spec is too tight (heavy padding / infeasible n_queries) |

`causal` (in `episode_peek`) = fraction of scored answer-tokens present in context: ≈1 for qa/recon
(reconstructible), low for continuation (unseen future), n/a for mae. It is multi-query-robust (unlike
a naive substring check on the concatenated multi-answer).

---

## 10. Invariants (what must always hold)

1. **Causality** — a queried fact is always written before the query point (packer guarantees it).
2. **Un-guessability** — the answer requires the compressed context (distractors never contain it;
   the question alone + LM prior can't produce it).
3. **Value-span scoring (bio)** — loss falls only on the un-guessable value fragments, never the entity
   name or template.
4. **Train/val firewall** — hotpot/musique use TRAIN splits, disjoint from the eval readers'
   validation; bio val is a different world with a name-collision firewall.
5. **Fixed compression denominator** — every context is exactly `total_len`; `M` is uniform across arms
   (the capacity-matched fairness anchor).
