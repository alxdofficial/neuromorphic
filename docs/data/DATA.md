# DATA — Sources, Tasks, Objectives & Current Config (the authoritative "what runs now")

This is the single reference for **how the training data works right now**: every source, how episodes are
generated, the objectives, the per-task objective dispatch, and the current sweep config. If you touch the
data layer, read this first; **if it disagrees with the code, the code wins — fix the doc.** For the
*future* full-corpus training + test-eval roadmap (Phase 1 / Phase 2, GRPO, benchmark panels) see
`docs/data/DATA_PHASES_PLAN.md` — that is a different scope (what's *planned*, not what runs now).

> **What we're measuring.** A **frozen** SmolLM2-135M decoder (d=576) reads `M=96` memory tokens produced by
> a small **trainable** encoder (~7M params — **parameter- and read-length-matched** across arms, though NOT
> persistent-state-matched: state floats vary ~194× across arms). We compress a 2048-token context
> into those 96 tokens and train end-to-end (default objective: **behavioral-KL**). The open problem is
> **loss-neutrality** (plain CE barely rewards *binding*). The sweep compares encoder architectures on a
> mixed objective stressing **fidelity** (can memory store) and **addressing** (can memory be selectively read).
>
> **Active trainable cohort (2026-07-10):** `icae` · `autocompressor` · `titans` · `gisting` · `memoryllm` ·
> `slotgraph` (ours). Eval-only references: `h2o` (training-free KV eviction), `vanilla_llama` (floor),
> `vanilla_full_context` (ceiling). Retired: slotgraph 1–4 (→ THE `slotgraph`), beacon, ccm, vqicae, biomem.

---

## 1. The 4-layer data model (task ≠ objective)

Data is factored into four **orthogonal** concepts. Key insight: **a task is how data is presented + what's
asked; an objective is the loss.** They compose as a matrix, not hardcoded pairs — so the same continuation
task runs on Pile *or* RedPajama, and behavioral-KL wraps *any* task. Adding a dataset never touches the trainer.

| Layer | Question it answers | Where it lives |
|---|---|---|
| **Source** | *where do tokens come from?* | `src/memory/data/sources/<name>.py` → `SOURCE_REGISTRY` |
| **Task** | *what is presented + asked?* | `src/memory/data/tasks/<style>.py` → `TASK_STYLES` / `get_task` |
| **EpisodeSpec** | *how hard?* (difficulty knobs) | `src/memory/data/schedule.py` |
| **Objective** | *how is it scored?* | `src/memory/training/objectives.py` |

Orchestration: `mixes.py` (`TASK_SPEC`) maps each mixed-task name → a `(Source, Task)` pair + addressing
profile; `training/data_mix.py` builds one dataloader per task from an `EpisodeSpec`; `loops.py` round-robins
over the per-task loaders (equal, one step per task in rotation).

**The 3-part item shape (fact / question / answer).** Every source yields items with the three parts
*separated* — that's what lets us "write many facts, then ask about a particular one":
- **Keyed items** (`KeyedItem`, from `bio`): `write="key = value"`, `query="key ="`, `answer=value`. The
  question is *derived from* the write (shared key) — an explicit pointer to one fact.
- **QA items** (`QAItem`, from babi + the RC sources): `facts` (a passage), `question`, `answer`. The
  question is *independent* of the write; packed neighbors become distractors and the answer lives in exactly one.

---

## 2. The 5 training tasks — what the model sees (worked examples)

The default mix (`DEFAULT_TRAIN_MIX`) is 5 tasks, equal round-robin. Each maps to a source and an objective
(§5). Context is always `total_len=2048`; memory is `M=96`.

### `reconstruct` — storage / fidelity (MAE) · source: **fineweb**
Compress a 2048-token fineweb passage, then reconstruct it with ~85% of decoder-input positions masked
(`mask_ratio=0.85`) so the model must recall from memory, not copy visible neighbors. The answer *is* the
input span. Pure "can memory store the bits" probe (the adversarial worst case).
```
Q:      "Reconstruct the text above."
CONTEXT: "…he explains existential analysis as an empirical science…"           (2048 tokens)
GOLDEN:  ← the same 2048 tokens (reconstruct all; ~85% masked in the decoder input)
```

### `continuation` — gist under accumulation (multi-horizon) · source: **multicorpus** (fineweb+pile+redpajama+code)
At **each streaming-window boundary** (256, 512, …, 2048) compress the prefix-so-far and predict the **next
`predict_len=64` tokens** from memory alone. See §4 for the precise encode→predict→advance loop. 8 horizons,
losses averaged into one backward. Input always ground-truth (teacher-forced); memory **accumulates**.
```
Q:      "Continue the passage."
CONTEXT: "…Vanderveer said 'I exonerate now and forever the American Legion…'"   (2048 tokens)
GOLDEN:  " March 1, 1920\n- 'Scene in I.W.W. Hall Prior to Shooting…'"      (next 64 tokens, ×8 horizons)
```

### `fact_recall` — key→value binding · source: **bio**
Pack ~30 `key = value` biographical facts (from a 410-entity procedural world) to fill 2048, then ask about
**1–3 keys**. Loss falls **only on the un-guessable value fragments** — never the entity name or template
scaffolding. The realistic binding anchor: bind each key to its attributes, read the right one back.
```
Q:      "the Suspension of the Egilstad Assembly ="                        (+ a 2nd key, multi-query)
CONTEXT: "Edvard Reinholdt = Note: … Halsten Sjoblom, born 1988 = …"       (~30 packed key=value facts)
GOLDEN:  " twenty-first  the end of a long dispute over inland water rights  …"   (value-spans ONLY)
```

### `babi` — relational binding (multi-segment, entity-renamed) · source: **babi**
bAbI stories are tiny (~86 tok), so co-pack **~24 stories to fill 2048** and query **1–3** of them. bAbI
reuses a tiny name pool, so each co-packed segment is **entity-renamed disjoint** (people = subjects of
action verbs, and objects, get fresh names; locations left alone — they're answers). Becomes *retrieve the
right segment, then reason within it* (verified: no cross-segment collision over 450 episodes).
```
Q:      "What is Ulyssa carrying?"
CONTEXT: "Ulyssa grabbed the phial there. … Ulyssa discarded the phial there. …"  (~24 renamed stories)
GOLDEN:  " nothing"     ← grabbed the phial then discarded it → carrying nothing
```

### `doc_qa` — retrieval / addressing (5 real RC sources) · source: **qa_multi**
Real reading-comprehension QA. Pack a gold passage + distractors (random mix of squad / triviaqa /
hotpot_train / musique_train / multiwoz), ask **1–2** questions; the answer lives in exactly one packed
passage (un-guessability enforced). Memory must *address* the right passage among noise.
```
squad:    "Who believe the Ming dynasty did not exercise direct control over Tibet?" → "Josef Kolmaš"
hotpot:   "Was Nam Woo-hyun or Eddie Vedder born first?" → "Eddie Vedder"          (compare 1991 vs 1964)
musique:  "The legal system of Au Kam San's country of birth?" → "Portuguese-based" (Au Kam San→Macau→PT)
multiwoz: "What party size did the user request for the hotel?" → "2"
triviaqa: "In which decade did Billboard first publish an American hit chart?" → "30s"
```

---

## 3. Sources & the shared packer

**Sources** (`kind` gates which tasks accept them; local-jsonl-first, HF fallback, robust skip):
- **corpus** (`→ mae / continuation`): `fineweb` (backs reconstruct; `src_tokenizer="meta-llama/Llama-3.2-1B"`
  — a gotcha), `multicorpus` = fineweb+pile+redpajama+code (backs continuation; `pile`/`redpajama`/`code`
  reachable ONLY inside multicorpus).
- **keyed** (`→ reconstruction`): `bio` — procedural biographical world (~410 entities, 7 types), built at
  runtime, train/val name-firewall, `key=value` with per-span content marks.
- **qa** (`→ qa`): `babi` (relational stories), `qa_multi` = the union of `squad` / `triviaqa` /
  `hotpot_train` / `musique_train` / `multiwoz` (random sub-source per item, gold capped ~900 tok, train
  splits firewalled from the eval readers' val splits). `quality` registered but needs ctx≥4096 (not in mix).
- **Registered but NOT in this mix:** the 12 Phase-1 full-corpus sources (`wildchat`, `lmsys_chat`, `msc`,
  `qasper`, `longcite`, `govreport`, `pg19`, `ruler_niah`, `babilong_train`, `wikibigedit`,
  `swe_trajectories`, `perltqa`) back the *next* phase (`docs/data/DATA_PHASES_PLAN.md`), plus inert scaffolding
  (`mqar`, `ruler_overwrite`).

**The shared streaming packer** (`tasks/_pack.py`) — `qa` and `reconstruction` are both "pack facts, ask
about some," factored into `Unit(write, query, answer, …)` + `pack_streaming_episode(...)`. Guarantees:
- **Causality** — query facts packed FIRST (always fit); question asked at the END (after the whole context)
  → every queried fact is written before it's asked.
- **Un-guessability** — `filler_ok` rejects any distractor containing a queried answer verbatim → the answer
  lives in exactly one packed passage. Plus a question-side filter: gold questions whose own text embeds the
  answer are demoted to distractors (else the readable question leaks it → SHUF passes for free).
- **Fill-to-budget** — distractors fill to `total_len`; the fill loop breaks on 12 consecutive *size*-misses
  (answer-leak rejections don't count). So the **segment count falls out of item size** — you set the ratio
  and pressure, not the count.
- **Multi-query** — `n_queries` sampled `1..max` per episode (extra queries appended with inline cues, cues
  not scored). `query_lag` placement: `early` / `recent` / `any` / `vary`.
- **bio value-span scoring** — loss falls only on the un-guessable value fragments, never the entity name or
  template (`answer_exclude`).

---

## 4. Multi-horizon streaming continuation (the encode→predict→advance loop)

Continuation compresses a prefix and predicts the next block **at every window boundary**. The precise
ordering: at boundary `b`, memory has **already encoded `[0:b]`** (including the most recent window); it then
predicts the **next, not-yet-seen block `[b : b+predict_len]`** (a *forward* prediction, never a
reconstruction of what was just added); then the next window is encoded and it repeats.
```
compress GT[0:256]   → predict GT[256:320]     (score CE)   ← predict the NEXT block from memory alone
compress GT[0:512]   → predict GT[512:576]     (score CE)   ← memory ACCUMULATES across windows
  ⋮                                            (8 boundaries at ctx=2048, window=256)
compress GT[0:2048]  → predict GT[2048:2112]   (score CE)   ← the classic single-shot horizon (answer_ids)
loss = mean of the 8 CEs  →  ONE backward
```
Input is always ground truth (teacher-forced window-to-window, no autoregressive rollout); memory is
persistent/accumulating; intermediate targets are sliced from `context_ids`, the final target is
`answer_ids`. Implemented in `model.compute_streaming_continuation_loss` (streams the encoder ONCE and
snapshots memory at each boundary — an efficiency fix; semantics identical to per-boundary re-encoding).
Auto-active when `window_size < total_len`; `--no-continuation-multi-horizon` disables; `n_horizons` caps
the deepest boundaries scored.

---

## 5. Objectives — and the per-task dispatch (which loss each task actually trains under)

The default objective is **behavioral-KL** (`--objective-mode behavioral_kl`, the CLI default). But the loss
is **dispatched per task at runtime** by `model.task_mode` inside `_behavioral_kl_step`
(`training/objectives.py`) — this is the part that was previously undocumented:

| task | `task_mode` | objective AT RUNTIME | why |
|---|---|---|---|
| **reconstruct** | `masked_reconstruction` | **plain CE** (teacher forward SKIPPED) | MAE target = the passage itself → teacher reconstructs itself → KL degenerate. Hard-wired to CE (`objectives.py:106`). |
| **babi** | `babi` | **behavioral-KL** (full: `kl_ce·CE + kl·KL`) | un-guessable answer; teacher genuinely uses context → KL bites |
| **doc_qa** | `qa` | **behavioral-KL** (full) | same |
| **fact_recall** | `conditioned_reconstruction_bio` | **behavioral-KL** (full) | same |
| **continuation** | `continuation` | **KL-ELIGIBLE, but CE-fallback** | goes through the QA path; if teacher/student answer logits don't align (common — the teacher can largely predict the next block itself), it falls back to plain CE (`objectives.py:140`) |

So: **MAE = always plain CE; babi/doc_qa/fact_recall = full KL; continuation = KL-eligible but frequently
CE-in-effect.** Coefficients (defaults): `kl_coef=2.0`, `kl_ce_coef=1.0`, `kl_temp=2.0`. The teacher is the
frozen LM over the FULL context with **`disable_lora()`** (load-bearing: a shared adapter would let the
optimizer satisfy KL by moving the teacher, breaking `E[KL]=I(context;answer)`).

**Why KL, why CE-for-MAE.** `E[KL(teacher‖student)] = I(context; answer)`, so gradient concentrates where
memory matters — the published fix for loss-neutrality (plain CE targets none of USE / ADDRESSING /
MEMBERSHIP). MAE is the legitimate CE exception: the target *is* the passage, so token-CE already forces
high-rank storage. Other objective modes exist (`plain`, `contrastive` InfoNCE) but behavioral-KL is the
active cohort objective. See `docs/design/OBJECTIVES.md` for the full ladder + the membership/addressing rungs.

**The REAL / SHUF / OFF binding gate** (eval) measures whether memory is used *structurally*: REAL = normal;
SHUF = decode with another example's memory; OFF = no memory. REAL ≪ SHUF is the goal; REAL ≈ SHUF = memory
ignored (the binding-failure signature).

---

## 6. Current config (the sweep regime)

Calibration principle: **the memory is slots-and-structure limited, not bits limited.** M=96 × d=576 ≈ 55K
floats ≫ the task's ~6K bits — so we do NOT starve `M` (that would make it a capacity test); difficulty comes
from the **number of distinct bindings + interference**. The discriminator is SHUF−REAL / EM, not compression
ratio. MAE is the exception (verbatim reconstruction at 21:1 = the deliberate fidelity probe).

| knob | value | source of truth |
|---|---|---|
| frozen decoder | **SmolLM2-135M** (d=576) | `--backbone` (default) |
| objective | **behavioral_kl** | `--objective-mode` (default) |
| memory `M` | **96** (~55K floats; ~430× smaller than the full 2048-ctx KV) | `DEFAULT_MIXED_M` |
| context `total_len` | **2048** (21:1 compression) | `--mixed-ctx` |
| `window_size` | **256** (8 streaming windows) | `--window-size` |
| `predict_len` | **64** (continuation block per horizon) | `--predict-len` |
| `mask_ratio` | **0.85** (MAE) | `mae_mask_ratio` |
| batch size / steps | **8 / 8000** (warmup 500, val 500, lr 1e-4, grad-clip 1.0) | `--batch-size` / `--steps` |
| default mix | reconstruct · babi · doc_qa · continuation · fact_recall | `DEFAULT_TRAIN_MIX` |

**Per-SOURCE addressing profile** (`Source.pack_n_queries`, sampled `1..max` per episode, feasibility-capped
— item size varies by source, so a per-task constant can't be right for the `doc_qa` union):

| mix-task (source) | pack_n_queries | query_lag | why |
|---|---|---|---|
| fact_recall (bio) | **1..3** | vary | tiny unique facts → high addressing pressure |
| babi | **1..3** | vary | ~24 renamed segments packed → query several |
| doc_qa (qa_multi) | **1..2** | vary | big contexts (~800–900 tok) → only ≤2 golds fit at 2048 |
| continuation | — | — | uses multi-horizon, not n_queries |
| reconstruct (mae) | — | — | scales by length + `mask_ratio` only |

Old mix-task names (`mae` / `qa_rc` / `condrecon_bio`) still work via `TASK_ALIASES`. Seeds are decorrelated
per task (`train_seed + i·10007`) so mae (fineweb) and continuation (⊇fineweb) don't draw the same passage.
**Note:** titans auto-disables the streaming activation-checkpoint (its inner `create_graph` conflicts) —
handled per-arm in the trainer.

---

## 7. Data flow (end to end)

```
Source.sample(rng, n) → items
   → Task.build(source, spec, tok, rng, pad) → per-sample dict {context_ids, context_mask,
       question_ids, answer_ids, answer_content_mask_list, task_family, question_type, answer_refs}
   → collate_qa (+ base._collate stamps k_slots / mask_ratio / n_horizons)
   → model.compute_loss(batch, window_size)  — routed by model.task_mode:
        masked_reconstruction → compute_masked_reconstruction_loss   (→ plain CE, §5)
        continuation (+ multi-horizon) → compute_streaming_continuation_loss
        else → the generic prepend-memory + teacher-forced path (→ behavioral-KL, §5)
```
The decoder **never sees raw context** — only the 96 memory tokens carry it. Every context tensor is exactly
`total_len` (fixed compression denominator, zero inter-sample context padding); batches are single-task
(round-robin) so answer padding stays small.

---

## 8. Diagnostics & invariants

**Diagnostics** (`scripts/diagnostics/mixed/`): `episode_peek.py` (pull episodes through the REAL loaders,
decode a sample + per-(task×datasource) fill/pad/causality/frequency stats), `data_stats.py` (length/padding
percentiles), `mixed_data_audit.py` (structure/firewall audit over the exact loaders). `causal` = fraction of
scored answer-tokens present in context (≈1 for qa/recon, low for continuation, n/a for mae).
`TaskDataset.n_builds/n_episodes`: `builds/episode > 1` means the spec is too tight.

**Invariants (must always hold):** (1) causality — a queried fact is written before the query point; (2)
un-guessability — the answer requires the compressed context (distractors never contain it; question + LM
prior can't produce it); (3) bio value-span scoring — loss only on un-guessable value fragments; (4)
train/val firewall — hotpot/musique use TRAIN splits disjoint from the eval readers' val; bio val is a
different world with a name-collision firewall; (5) fixed compression denominator — every context is exactly
`total_len`; `M` uniform across arms (the read-length fairness anchor — one of two matched axes with
trainable-param count; persistent state is deliberately NOT matched).
