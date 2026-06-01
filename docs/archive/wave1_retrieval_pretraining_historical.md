# Wave 1 Retrieval-Pretraining Protocol (v4)

**Status:** designed 2026-05-13. Dataset v6 generated + train/val
split done (3,740 facts across 9 entity classes, see §3). Trainer
smoke-tested end-to-end. Ready for first full training run.
**Supersedes:** the v3 REALM-style MLM auxiliary-loss spec in `wave1_v3_protocol.md`
(which is now archived; we are not pursuing it).
**Author:** Alex Ding + Claude collaborative design session.

## TL;DR

Replace current Wave 1 (long-document NTP over FineWeb-edu) with a
write-then-retrieve QA protocol over a synthetic personal/biographical
universe. Each training step writes 8 facts to memory, then asks one
question about one of them; loss is **only on the answer tokens**.

The objective forces memory specialization: the question's answer is
not deducible from the question alone, so the gradient pressures the
read trajectory at question-time to land on the same concepts where
the relevant fact was written. This is the differentiable retrieval
signal that the current NTP setup doesn't provide — and the absence
of that signal is what produced our observed routing uniformity
(see `project_routing_uniformity` memory entry).

---

## 1. Motivation

Current Wave 1 trains memory on long-document next-token-prediction.
The empirical failure mode (Wave 1 v2 final run, step 8000):

- `r_uf ≈ 0`, `w_uf ≈ 0.003`: read and write routing are essentially
  uniform across the 4096 concepts.
- Answer-only EM at large distances regressed (8.86 at step 8000 vs
  3.76 at step 1500). Best-by-weighted-val checkpoint was not the
  best-by-answer-only.
- Memory module appears to contribute little above what Llama's
  in-context attention already provides.

Root cause: the NTP loss can be satisfied by **either** Llama's
in-context attention OR memory readout. Llama's attention is already
trained and effective; the optimizer has no pressure to make memory
load-bearing. Result: memory routes uniformly because no part of
the loss requires it to discriminate.

The retrieval protocol changes this: the question's answer can't be
deduced from the question text alone, so the loss **cannot** be
driven down without memory carrying the right information. This is
the same supervisory shape as REALM/RAG, but expressed through our
trajectory mechanism instead of dense embeddings — which is the
architectural bet.

---

## 2. Protocol

### Per training step

```
1. Reset manifold state.
2. Sample 8 facts from a pool of ~5200 atomic facts, enforcing
   that the 8 have distinct (entity_class, attribute) keys.
3. Pick one of the 8 at random as the target.
4. Format the target's question + answer using templates.
5. Write phase (8 sequential write windows, no loss):
     prev_state = manifold.reset_states(batch_size=BS)
     for fact_text in fact_texts:                       # 8 facts
         hidden = llama.forward(fact_text)              # frozen Llama, grad-tracked
         prev_state = write_module(hidden, prev_state)  # trainable, grad alive
6. Read phase (1 question encoding window, no loss):
     q_hidden = llama.forward(question_text, mem_inject=read_attn(prev_state))
7. Answer NTP (loss on answer tokens only):
     answer_logits = lm_head(answer_position_hiddens)
     loss = cross_entropy(answer_logits, answer_token_ids)
8. backward → updates read_module, read_attn, write_module,
   mem_inject.{W_in, W_out, scale}
9. State is reset for next batch element (no cross-step TBPTT).
```

### Per-chunk layout

| segment       | length         | role                             | loss-bearing |
|---------------|----------------|----------------------------------|--------------|
| Fact 1        | 256 tokens     | write window 1                   | no           |
| Fact 2        | 256 tokens     | write window 2                   | no           |
| ...           | ...            | ...                              | ...          |
| Fact 8        | 256 tokens     | write window 8                   | no           |
| Question      | ~256 tokens    | read+inject window               | no           |
| Answer        | ~50-150 tokens | NTP target                       | **yes**      |

Total chunk size: ~2400 tokens. Fits within `effective_lm_context=2048`
for the read+answer phase (the write phase resets manifold state and
KV cache between chunks, so the LM's attention only sees question +
answer at read time).

---

## 3. Data: synthetic personal/biographical universe

### Setting

"Alternate Earth" plausible-realism, **not** fantasy. Entities are
fictional but mundane — countries with constitutions, people with
careers, companies with products, events with dates. No magic, no
mythology. Closer to reading obscure Wikipedia stubs than to fantasy
novels.

This setting is preferred over fantasy because:
- Prose style matches downstream benchmarks (NarrativeQA, LongMemEval,
  MemoryAgentBench all are in mundane/realistic register).
- Cross-entity reference (Maria's mentor Hilde, Hilde's affiliation
  with X firm) creates an implicit graph the model can pick up as
  extra structure for free.

### Entity classes (heavily personal-biased)

Actual v6 dataset (`data/wave1_retrieval/facts_v6.jsonl`, generated
2026-05-13). Attributes shown are what's actually implemented in
`scripts/data/generate_wave1_retrieval.py`. Counts are tight because
each implemented attribute = 1 fact per entity (no oversampling).

| class | entities | implemented attributes | attrs/entity | facts |
|---|---:|---|---:|---:|
| **Private Individuals** | 250 (5 seed + 245 procedural) | occupation, hometown, signature_skill, recurring_habit, alma_mater, hobby, mentor_name, partner_relationship | 8 | 2000 |
| **Public Figures** | 80 | primary_field, signature_work, famous_award, birth_year | 4 | 320 |
| **Life Events** | 150 | event_type, event_year, event_location | 3 | 450 |
| **Organizations** | 50 | org_founding_year, org_founder, org_primary_activity | 3 | 150 |
| **Nations** | 40 | nation_founding_year, nation_capital, nation_head_of_government | 3 | 120 |
| **Historical Events** | 50 | he_event_year, he_event_location, he_outcome | 3 | 150 |
| **Cultural Works** | 50 | cw_creator, cw_year_released, cw_main_subject | 3 | 150 |
| **Personal Relationships** | 100 | pr_relationship_type, pr_meeting_year | 2 | 200 |
| **Personal Preferences** | 200 | pp_preference_value | 1 | 200 |
| **Total** | **970** | — | avg 3.85 | **3,740** |

Train/val split (entity-disjoint, 90/10): **3,367 train / 373 val facts.**

Actual passage-token totals: 715,436 tokens across 3,740 facts. Min 95
tokens, median 202, max 301, mean 191. Per-fact (question + answer)
adds ~30-40 tokens.

### Recombinatorial training

Training-step count target: ~80K. Per step: 8 facts sampled from the
pool (or from the train split, ~3,367 facts).

- Total fact-impressions across training: 80K × 8 = **640K**.
- Each atomic fact is seen ~190 times on average across training, but
  in a **different 7-distractor context every time**.
- C(3367, 8) ≈ 10²³ unique 8-fact draws from the train split —
  effectively infinite.

The repeated exposure forces the model to encode each fact's
**identity** rather than its co-occurrence pattern, because the
co-occurrence partners shuffle each step. This is the discriminative
pressure we want.

### Sampling constraint

At sample time, enforce:
- The 8 sampled facts must have **8 distinct `(entity_class, attribute)`
  keys**. (Same `(entity, attribute)` key would mean two facts state
  the same field-value about something, creating ambiguity in the
  question's answer.)

This single constraint subsumes both:
- "No multiple right answers" (only one fact in the batch answers any
  question)
- "No contradictions" (no conflicting values for the same field)

With ~6400 facts spanning ~970 entities × ~10 attributes, the
constraint is satisfied by almost every random draw — rejection
sampling is fast.

### Per-class generation patterns

Each entity class has its own procedural generator in
`scripts/data/wave1_worldspec.py` (Private_Individual, Public_Figure,
Life_Event) or `wave1_worldspec_extra.py` (the other 6). Generators
sample from class-specific value pools to ensure internal coherence —
e.g. a pediatric nurse doesn't train at the Yspara Naval Workshop.

Common conventions across all classes:
- **All names are drawn from `FIRST_NAMES_F` (female pool)** —
  matches what the prose templates assume ("she", "her"). 80 first
  names × 78 last names → ample uniqueness.
- **All entities live in Marlonia** (fictional alt-Earth nation) by
  default, except Nation entities which describe sibling-nations
  alongside Marlonia.
- **Entity keys are prefixed** by class (`pi_`, `pf_`, `le_`, `org_`,
  `nt_`, `he_`, `cw_`, `pr_`, `pp_`). The 5 seed Private_Individuals
  have no prefix.

| class | generation pattern |
|---|---|
| **Private_Individual** | Pick occupation → domain (conservation / healthcare / sciences / trades / hospitality / academic / legal). Workplace, alma mater, mentor institution all drawn from the **domain's** pool → coherent. Hometown + descriptor coupled (from a 46-town list with paired descriptors). Random partner name + occupation. ~25 slots per entity. |
| **Public_Figure** | Pick primary_field → corresponding signature_work pool (each field has 3 stock works). Higher birth-year range (1900–1960) and uses separate institution pool (`PF_INSTITUTIONS`, all academic — no naval workshops). Award is from `FAMOUS_AWARDS`; award_year clamped to lie between birth+30 and retirement. |
| **Life_Event** | Event_type from 20-item list (wedding, retirement dinner, etc.). Outcome pool is keyed by event_type → coherent ("wedding" → "the beginning of a long and quiet marriage"). Year 1980–2024. Actor name is procedurally generated and is *not* cross-referenced to any Private_Individual entity. |
| **Organization** | Name composed from `prefix + stem + suffix` ("The Drangsund Veterans' Council"). Founding year 1820–2010. Founder name + activity from 30-item pool. The activity is treated as text-only — no coherence constraint with org_type. |
| **Nation** | Names from a 45-item fictional pool (Vesterland, Sjørike, ...) paired 1:1 with capitals from a separate 45-item capital pool. Head of government stored **as name only** (`Ragnhild Falk`) plus separate `_title` slot (`First Minister`) and `_full` combined slot, to avoid templates rendering "First Minister First Minister Ragnhild Falk". |
| **Historical_Event** | Event name uses a template ("Treaty of {town}", "Founding of {institution}", "{town} Famine"). Year 1500–1950. Outcome from a 20-item pool. Primary figure is `{title} {first} {last}` (Chancellor, Bishop, Admiral, etc.). Adds a `nation_default = "Marlonia"` slot for use in HE V3 template ("The standard reference history of Marlonia covers..."). |
| **Cultural_Work** | Work type drawn from 7 pre-defined categories (novel, epic poem, film, orchestral work, song cycle, play, biography). Titles are hand-crafted **per type** (e.g. novels: "The Long Northern Winter", "Salt and Iron", ...) — 50–60 unique titles total across all types. Year 1900–2020. Genre + subject from generic pools (any genre×subject is plausible). |
| **Personal_Relationship** | Pair of fresh ad-hoc names (no cross-reference to other entities). Relationship type pool is **life-stage-neutral** (no "childhood best friends", "in-laws" — those would constrain meeting context). Meeting context pool is **adult-only** (no childhood meetings). Meeting year 1960–2015. Adds `nation_default = "Marlonia"` slot. |
| **Personal_Preference** | Single fresh ad-hoc name. Preference type from 10 categories (favorite season, favorite hot drink, favorite handcraft, …); each type has its own 5–7 value pool. Origin context is generic ("since her childhood years in a small coastal town"). 1 attribute per entity → 200 facts total. |

The pattern follows three principles:
1. **Domain-coherent composition** where the template might otherwise produce nonsense (occupation → workplace, event → outcome).
2. **No cross-entity references for the 6 new classes** — keeps generators simple and avoids accidental contradictions. Cross-entity references in v1 are limited to the 5 hand-crafted seed Private_Individuals (Maria trains under Hilde, etc.).
3. **Auto-capitalization helper** in `with_indefinites()`: any slot value starting with a lowercase letter gets a `{slot}_cap` variant ("the Drangsund Veterans' Council" → "The …"), which templates use at sentence starts.

### Heterogeneous classes per chunk

The 8 facts in a chunk can (and should) be drawn from a **mix** of
entity classes — e.g., 3 Private Individuals + 2 Life Events + 1
Public Figure + 1 Nation + 1 Organization. This exercises memory
across the full fact-type distribution every step rather than letting
the model learn class-specific routing shortcuts.

### Templates

Actual implementation uses **4 passage / 3 question / 2 answer**
variants per attribute (locked convention). Each variant exposes the
same slot set; at fact-generation time one is sampled at random.

- **Passage templates**: 15 attributes × 4 variants = ~60 distinct
  passage templates. Historical_Event and Personal_Relationship share
  their passage pool across multiple attributes (only the question /
  answer differ by attribute), so the unique total is closer to ~45.
  Templates live in `scripts/data/generate_wave1_retrieval.py` (3
  original classes) and `wave1_templates_extra.py` (6 new classes).
- **Question templates**: 15 × 3 = 45.
- **Answer templates**: 15 × 2 = 30.

All templates are hand-written in Python f-string format. **No LLM
generation cost.** Authoring took ~1 day for all 9 classes.

### Why multiple passage-template variants

Single template per attribute → every passage of that attribute would
have the same sentence structure with different nouns swapped in. The
model would learn the **template structure** rather than the fact.

Multiple variants force the surface form to vary while keeping the
target value embedded somewhere in the passage. The model can't game
positional/structural cues.

### Visual inspection step

Before scaling: generate a 50-100 fact pilot. Inspect:
- Are the facts genuinely distinct (not just paraphrases)?
- Is the target value actually load-bearing (other facts can't
  substitute)?
- Is the answer non-deducible from question alone?
- Is the natural-language quality good enough that it'd train a
  useful model (not robotic, not template-obvious)?

Only scale to full ~6400 facts if pilot passes.

---

## 4. Training mechanics

### Gradient flow

- **Llama params**: `requires_grad=False`. Never receive gradient
  updates. Frozen backbone.
- **Memory modules** (`read_module`, `read_attn`, `write_module`,
  `mem_inject.W_in`, `mem_inject.W_out`, `mem_inject.scale_raw`):
  `requires_grad=True`. Receive gradient.
- **Forward passes are NOT wrapped in `torch.no_grad()`**. Autograd
  graph builds through Llama's compute. Activations are kept so the
  backward can flow through.
- **Backward**: gradients propagate through Llama's saved activations
  to reach our trainable leaves. Llama params don't accumulate grads
  (because `requires_grad=False`), so no wasted `.grad` storage —
  autograd just treats Llama compute as a fixed differentiable
  function in the graph.

The mistake to avoid: wrapping fact-encoding or question-encoding in
`torch.no_grad()`. That would sever the gradient chain.

### Gradient chain at backward time

Starting from the loss at the answer tokens:

```
loss
 ↓
answer_logits = lm_head(answer_hiddens)      # lm_head frozen
 ↓
answer_hiddens ← Llama layers above inject point (frozen, grad-pass-through)
 ↓
mem_inject output at inject layer (W_in, W_out, scale trainable)
 ↓
readout = read_attn(h_mem, flat_traj)        # read_attn trainable
 ↓
flat_traj = read_module(prev_hiddens, prev_state)   # read_module trainable
 ↓
prev_state at end-of-write-phase
 ↓
write_module(hidden_8, write_module(hidden_7, ... write_module(hidden_1, init))) ...
   # write_module trainable; chained back through 8 fact encodings
 ↓
hidden_i ← Llama layers (frozen, grad-pass-through) ← fact_i_input_embed
```

The gradient chain traverses **9 Llama forwards** in total (8 fact
encodings + 1 question encoding).

### Activation checkpointing (required)

Without checkpointing, 9 stacked Llama-1B forwards at BS=8 would hold
~25-30 GB of activations alive during backward — OOM on 24 GB cards.

With gradient checkpointing on each Llama forward, activations are
discarded after the forward, recomputed during backward at the cost
of one extra forward per checkpointed region. At BS=8, expected
backward-phase VRAM ≈ 5-7 GB. Latency cost: ~33% more compute.

HF Llama supports `gradient_checkpointing_enable()`. Use it.

### Memory state

- **Per-batch-element**: each chunk owns one state trajectory through
  its 8 writes. State accumulates within the chunk.
- **Cross-chunk**: state resets between chunks. No TBPTT across chunks.

The current `TrajMemConfig.D` (TBPTT depth in windows) becomes
load-bearing within a chunk: D ≥ 9 to keep the full graph alive
through 8 writes + 1 read. **Set `D = 9` (or higher) for this
protocol** — the current default `D = 4` would truncate gradient
flow back through earlier writes.

### Batch construction

Per outer-batch element (chunk): 8 facts + 1 question + 1 answer.

A training step processes a batch of M chunks in parallel:
- Each chunk's 8 fact passages are length-bucketed to 256 tokens
  (pad/truncate as needed during template fill).
- The M chunks proceed through their write/read/answer phases in
  parallel as a BS=M batch.

Recommended initial `M = 8` to match current Wave 1 BS. VRAM headroom
permits going higher with checkpointing; let bench decide.

### Loss normalization

- Loss is per-token CE on the answer tokens only.
- Normalize by `(answer_token_count × M)` (not by total chunk token
  count). The write phase contributes zero loss.
- Skip padded tokens in the answer.

---

## 5. Why this addresses routing uniformity

Current Wave 1 NTP failure mode: the loss is satisfiable via Llama's
in-context attention alone, so memory has no necessary role.

Retrieval protocol pressure on each gradient component:

1. **Read trajectory at question time**: must steer the trajectory
   walk through concepts that hold the target fact's signature. If
   it walks through irrelevant concepts (uniform routing), the
   readout is meaningless and the answer can't be predicted.

2. **Write trajectory at fact time**: must encode the fact into
   concepts that the question's read trajectory will also visit. If
   the write goes elsewhere, the read can't find it.

3. **Read/write trajectory consistency**: the routing function must
   be consistent enough between write-time (with fact context) and
   read-time (with question context) that the same concepts get
   visited. The shared concept_ids and manifold geometry provide
   this consistency budget.

4. **mem_inject bridge**: must learn to extract the answer-relevant
   detail from the readout vector and inject it as a usable
   hidden-state perturbation at layer 8.

Every one of these signals is directly attributable to the answer-
token loss. The optimizer has nowhere else to "spend" the loss
reduction — Llama is frozen and unable to absorb it via its own
parameters.

---

## 6. Design decisions (locked)

| decision | choice | rationale |
|---|---|---|
| World setting | Alt-Earth plausible realism | Transfers to NarrativeQA/LongMemEval style; cross-entity references provide implicit graph |
| Entity-class mix | Personal-biased (~70% personal/biographical, ~30% encyclopedic) | Matches downstream agent/chatbot benchmark distribution |
| Fact passage length | ~256 tokens | One trajectory window per fact (matches J=4, K_read=K_write=8 architecture) |
| Question hop count | Single-hop (v1) | Lower-confound floor; multi-hop deferred to v2 |
| Target position | Uniform random across 1-8 | Forces position-invariant retrieval (matches benchmark distribution) |
| Replace vs augment current Wave 1 | Replace | Current NTP shown not to supervise memory |
| Llama params | Frozen (requires_grad=False) | Memory is what we're training |
| Llama forward | grad-enabled (not no_grad) | Need autograd graph for backward into memory modules |
| Activation checkpointing | Required, on Llama | 9 stacked forwards else OOM |
| Templates per attribute | **4** passage variants, **3** question, **2** answer (implemented) | Prevents surface-form overfitting while keeping authoring tractable |
| Sampling constraint | 8 distinct `(entity_class, attribute)` keys | Subsumes "no duplicate answers" + "no contradictions" |
| Heterogeneous chunks | Yes — mix entity classes within one 8-fact draw | Exercises memory across full type distribution per step |
| TBPTT depth `D` | Set to 9 (or higher) for this protocol | Keep gradient alive through full 8-write + 1-read chain |
| Unique-token target | ~2.0-2.6M | Recombinatorial expansion via 8-from-N sampling makes this sufficient for ~80K training steps |
| Generation method | Template-based, hand-written | Zero LLM cost; full control over fact content |

---

## 7. Implementation order

**Status (2026-05-13):** Steps 1–8 complete. Dataset v6 generated
(3,740 facts, 9 classes, 715K tokens). Smoke-tested end-to-end —
write_module gradient flows (`w_gn ≈ 0.2`, the canary). Ready for
step 9 (full ~80K-step training run).

1. **Worldspec data** (~1 day, includes naming pools + value tables)
   - Python dict literals for ~970 entities across 9 entity classes
   - Shared name pool (so cross-entity references reuse names
     consistently, e.g. mentor field of person X references person Y
     who exists in the pool)
   - Value pools (year ranges, occupation list, language list, etc.)

2. **Passage templates** (~1-2 days authoring)
   - 5-10 variants per `(entity_class, attribute)` combo
   - Total ~200-300 templates
   - Start with one attribute (e.g. `Private_Individual.occupation`)
     as a worked example; review before scaling

3. **Question + answer templates** (~half day)
   - 3-5 question variants + 2-3 answer variants per
     `(entity_class, attribute)` combo
   - Total ~75-150 templates

4. **Generator script** (`scripts/data/generate_wave1_retrieval.py`)
   - Loads worldspec + templates
   - For each `(entity, attribute)` pair: select random passage
     variant, fill slots, produce JSONL record
   - JSONL schema:
     ```
     {
       "fact_id": "private_individual_42.occupation",
       "entity_class": "private_individual",
       "entity_key": "private_individual_42",
       "attribute": "occupation",
       "target_value": "textile conservator",
       "passage": "<256-token natural-language passage>",
       "passage_token_ids": [...],
       "question_template_variants": [3-5 templates with slots],
       "answer_template_variants": [2-3 templates with slots],
       "slot_values": {<dict for filling templates>}
     }
     ```
   - Outputs `data/wave1_retrieval/facts.jsonl` (~6400 entries, ~30 MB)

5. **Pilot review** (50-100 facts)
   - Generate, format for review (one chunk per page), inspect by eye

6. **Sampler with distinctness constraint**
   - At training time: draw 8 from pool, reject if any two share
     `(entity_class, attribute)`. Almost always passes first try.
   - Random target index 0-7. Render question and answer from target's
     templates.

7. **`Phase1RetrievalTrainer`**
   - Reuse most of current Phase1Trainer plumbing
   - New `step_retrieval(facts, target_idx, question, answer)` method
   - Manages: state reset, 8 sequential `forward_window` writes (no
     loss), question encode (read+inject), answer NTP, backward
   - Honors activation checkpointing on Llama

8. **Smoke test** at small scale (1000 steps)
   - Loss curve sane, routing diversity rises (r_uf > 0.1 within a
     few hundred steps), no NaN/Inf

9. **Full Wave 1-retrieval run** (~80K steps, overnight)
   - Save ckpts at every val cycle (don't lose the answer-only-best
     mistake from v2)

10. **Benchmark suite run** (Track A retrieval + Track B lifelong)
    - Needle-in-haystack EM (built-in)
    - LongMemEval
    - MemoryAgentBench
    - Compare against Wave 1 v2 baseline numbers

---

## 8. Open questions / deferred to v2

- **Multi-hop questions**: questions that require composing 2+ facts
  (e.g. "What did the mentor of Maria's collaborator study?").
  Stronger test of trajectory-based reasoning but adds confounds.
  Defer to v2 once single-hop works.

- **Real-data mixing**: 30% obscure real-world facts (post-cutoff
  news, Wikipedia stubs <500 monthly views, recent arXiv abstracts)
  alongside synthetic for transfer testing. Defer until v1 baseline
  is set.

- **Format overfitting check**: hold out template variants entirely
  for benchmarks (test on never-seen templates) to confirm the model
  generalizes beyond template surface form.

- **Phase 2 GRPO compatibility**: confirm that retrieval-pretrained
  memory works as expected for Phase 2 GRPO on NarrativeQA. The
  protocols are structurally compatible (Phase 2 also has a
  prompt→answer flow that needs memory), but worth confirming.

- **D_concept = 1024 calibration**: with personal-biographical content
  (richer detail per fact), confirm D_concept=1024 is adequate.
  May need to bump to 2048 if facts contain too much information per
  256-token window.

- **`TrajMemConfig.D` setting**: D=9 vs D=4. If memory or compute
  becomes tight, can we get away with truncated backprop (D=4) at
  some loss of write-gradient signal for the earliest facts?

- **Curriculum**: start training with shorter passages (~128 tokens),
  ramp to 256? Or hold passages fixed and ramp distractor difficulty
  (start with 8 easy → 8 hard distractors)?

---

## 9. Benchmark plan (post-training)

The retrieval pretraining should manifestly improve:

| benchmark | metric | current (Wave 1 v2 baseline) | target |
|---|---|---|---|
| Needle EM at 2K-5K distance | answer-only CE | 5.78 | < 4.0 |
| Needle EM at 5K-10K distance | answer-only CE | regressed | improve |
| LongMemEval (when wired up) | retrieval F1 | not measured | establish baseline |
| MemoryAgentBench (when wired up) | task success | not measured | establish baseline |
| Routing uniformity `r_uf` at end of training | mean | ~0 (collapsed) | > 0.15 |
| Routing uniformity `w_uf` at end of training | mean | 0.003 | > 0.15 |

Track A (retrieval) and Track B (lifelong learning) tests from the
existing benchmark plan apply.

---

## 10. Relationship to other research

- **REALM (Guu et al. 2020)**: same supervisory shape (retrieval
  trained by MLM/QA loss), but with dense embeddings + index lookup.
  We use the trajectory walk as our differentiable lookup.
- **RAG (Lewis et al. 2020)**: retrieve-then-generate, with retrieval
  trained jointly with generation. Same pressure shape; different
  retrieval mechanism.
- **TEM (Whittington et al. 2020)**: memory trained by
  prediction-error of next-observation. Same principle: loss must
  depend on memory state, otherwise memory is unsupervised.
- **DNC / NTM (Graves et al.)**: external read/write memory trained
  on synthetic QA tasks. Direct ancestor of this protocol's
  controlled-fact design.
- **MemoryLLM, RetroLLM**: similar memory-augmented LM lineage;
  different training protocols.

The recombinatorial pool-of-N idea (small unique base, large
combinatorial training distribution) is the same trick used in
needle-in-haystack data construction.

---

## 11. Risks and mitigations

| risk | mitigation |
|---|---|
| Template overfitting (model learns template structure not facts) | 5-10 variants per attribute; hold out templates for benchmarks; mix surface forms |
| Synthetic-data-only doesn't transfer to real-world QA | v2 mixes 30% obscure real facts |
| 8-write chain blows up VRAM with grad | activation checkpointing required (already planned) |
| TBPTT D=9 is too long, training unstable | start with smaller D, verify; alternative: gradient detach between writes (would lose write-1 gradient signal but keep training stable) |
| Single-hop too easy, model just memorizes | dial up distractor density (replace random with topically-related); add multi-hop in v2 |
| 6400 unique facts is too few, model overfits | scale to 12K or 20K facts (still feasible authoring-wise); or auto-generate template fillers from a wider value pool |

---

## Cross-references

- Architecture: `docs/plan_trajectory_memory.md` §2 (read/write modules), §4 (write_module GRUCell)
- Failure mode being addressed: memory entry `project_routing_uniformity`
- Capacity concern background: memory entry `project_capacity_concern`
- Why we're not using LLM-as-judge or LLM generation: memory entry
  `feedback_rewards` (no LLM-as-judge) — same principle here, control
  the data fully via templates
- Phase 2 GRPO (downstream of this Wave 1): `src/trajectory_memory/training/phase2.py`
- v3 protocol (superseded): `docs/wave1_v3_protocol.md`

---

**Next step after reading this doc**: write the worldspec for ~30
entities + 5 passage templates for one attribute type + render 5
pilot fact passages. Then pause for human eyeball review before
scaling to the full ~970 entities × ~10 attributes × ~5 templates.
