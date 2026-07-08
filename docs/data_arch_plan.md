# Data architecture plan — Source / Task / Schedule / Objective (2026-07-07)

> **STATUS: phases 1-2, 4-7 COMPLETED.** The Source/Task/Schedule split and the mixes.py/data_mix.py
> rewire landed as designed. The one sub-phase **NOT executed**: Layer O (splitting
> `src/memory/training/objectives.py` into an `objectives/` package) — objectives stayed a flat
> file; see the Layer O section below for the current state.

Goal: make the data layer **comprehensive and modular** — many datasets, many task styles, a
configurable difficulty schedule, and objectives that compose — by decomposing the current
*fused* readers into four orthogonal concepts. Builds **on top of** the completed 3-layer
build/store/load reorg (`data_reorg_plan.md`) and the harness reorg (`harness_reorg_plan.md`);
it further decomposes the **Load** layer and adds a **Schedule** and an **Objective** package.

## The four orthogonal concepts (the core idea)

The key insight: **task ≠ objective.** A *task* is how data is presented and what's asked; an
*objective* is the loss. They compose as a matrix, not hardcoded pairs.

| Concept | Answers | Examples | Where |
|---|---|---|---|
| **Source** | *where tokens come from* | bio-world, babi, pile, redpajama, fineweb, mqar, code | `src/memory/data/sources/<name>.py` |
| **Task** | *what we present & ask* | reconstruction, qa, continuation, mae, multisession | `src/memory/data/tasks/<style>.py` |
| **EpisodeSpec** (schedule) | *how hard* | #inputs, input length, #distractors, #queries, query-lag | `src/memory/data/schedule.py` |
| **Objective** | *how we score* | ce, behavioral_kl, contrastive | `src/memory/training/objectives/<name>.py` |

Composition examples (source × task × objective — all free combinations):
- `(pile, continuation, behavioral_kl)` — bucket-1 natural-text KL
- `(bio, reconstruction, ce + contrastive)` — binding
- `(bio, qa, n_queries=4, contrastive)` — addressing
- `(fineweb, mae, ce)` — storage / compression
- `(ruler_overwrite, reconstruction, ce + behavioral_kl)` — overwrite / forced-forgetting

**Source knows nothing about windows/queries/distractors** — it yields *items*. **Task** takes
items + an `EpisodeSpec` and builds `(context windows, queries, answer spans, loss mask)`.
**Objective** scores the model's prediction on those spans. That separation is why the same
continuation task runs on Pile *or* RedPajama, and why behavioral-KL wraps *any* task.

## "One folder per dataset" — where each dataset lives (unchanged 3-layer + new Load split)

The user's "folder per dataset name" is honored at **build** and **store** (already per-name):
- **build** — `scripts/data_build/generate/<name>/` (procedural: bio, babi templates, mqar,
  ruler_overwrite) OR `scripts/data_build/ingest/<name>/` (download+preprocess: pile, redpajama,
  fineweb, code).
- **store** — `data/<name>/` (raw + preprocessed on disk).
- **load** — `src/memory/data/sources/<name>.py` (a module; a *folder* `sources/<name>/` only if
  it needs helpers, e.g. `sources/bio/` keeps `render.py`). One canonical `<name>` across all three.

---

## Layer L1 — Sources: `src/memory/data/sources/` (NEW)

> Snapshot note: the source list below is as originally planned. 9 more sources have since shipped
> under `src/memory/data/sources/`: `squad`, `triviaqa`, `hotpot_train`, `musique_train`,
> `multiwoz`, `quality`, `qa_multi` (the multi-source RC aggregator), `code`, `multicorpus` (the
> fineweb+pile+redpajama+code aggregator backing `continuation`). See `sources/__init__.py` for the
> live `SOURCE_REGISTRY`.

A **Source** yields raw *items* and (optionally) a noise pool. It does NO windowing, query
placement, or distractor insertion. Item kinds + the interface tasks consume:

| Source kind | Item interface | Sources |
|---|---|---|
| **keyed** | `.key_text`, `.value_text`, `.value_spans` | bio, mqar, ruler_overwrite |
| **corpus** | `.tokens` (backbone-tokenized array) | fineweb, pile, redpajama, code |
| **qa** | `.context_facts`, `.question`, `.answer`, `+ noise_pool` | babi |

Extraction from the current fused readers (the source-load half of each):
- `sources/bio.py` ← `bio.py`'s `build_scenario` wrapper + entity firewall + `bio_render` (moves to
  `sources/bio/render.py`); exposes keyed items (`render_key`/`render_value` → key/value/spans).
- `sources/babi.py` ← `babi.py`'s `_load_babi_rows` + `_split_sents`/`_caps_names` (the noise pool
  + entity-disjointness helpers); yields qa items + a distractor pool.
- `sources/fineweb.py` ← the **shared** `_decode_cache` + doc-loading loop (currently DUPLICATED in
  `mae.py` and `continuation.py`); yields corpus items (token arrays). **Dedup win.**
- `sources/pile.py`, `sources/redpajama.py` (NEW) — same corpus interface, different ingest.
- `sources/mqar.py`, `sources/ruler_overwrite.py` (NEW) — procedural keyed sources (survey recipe:
  fork RULER `variable_tracking`, reassign-same-key = overwrite; random UUIDs).
- `sources/__init__.py` — `SOURCE_REGISTRY: {name → Source factory}`.

## Layer L2 — Tasks: `src/memory/data/tasks/` (NEW)

A **Task** takes source items + an `EpisodeSpec` and emits the on-the-wire sample dict (the same
one `collate_qa` consumes today). One module per style; extraction from the `_gen` halves:
- `tasks/base.py` — `Task` ABC (`build(items, spec, rng) → sample dict`) + shared helpers
  (streaming-window placement, query-lag selection — lifted from `bio.py`'s `query_window` logic).
- `tasks/reconstruction.py` ← `bio.py`'s `_gen` (pack key=value lines, place queried pair per
  `query_lag`, value-span content mask, multi-query).
- `tasks/qa.py` ← `babi.py`'s `_gen` (distractor-pad to length per `n_distractors`, entity-disjoint
  noise, question→answer).
- `tasks/continuation.py` ← `continuation.py`'s `_gen` (compress-span → predict-span).
- `tasks/mae.py` ← `mae.py`'s `_gen` (contiguous span, context==answer, k_slots; masked_reconstruction path).
- `tasks/multisession.py` (NEW, later) — MS-TOD-style multi-session slot recall.
- `tasks/__init__.py` — `TASK_REGISTRY: {style → Task}`.

**On-the-wire format stays in `common.py` UNCHANGED** — `QABatch`, `collate_qa`, `_pack_context`.
Tasks emit the identical dict, so `compute_loss` + the REAL/SHUF/OFF gate are untouched. (The
composite `QADataset` eval reader in `common.py` — the separate `--task qa` path — also stays as-is.)

## Layer L3 — Schedule: `src/memory/data/schedule.py` (NEW)

```python
@dataclass(frozen=True)
class EpisodeSpec:
    source: str            # SOURCE_REGISTRY key
    task: str              # TASK_REGISTRY key
    n_inputs: int          # items packed into the context
    input_len: int         # tokens per item / streaming window size
    n_distractors: int     # filler items between write and query
    n_queries: int         # reads per episode (>1 → forces addressing)
    query_lag: str         # "recent" | "early" | window-index distribution
    window_size: int       # streaming write granularity (ties to encoder chunking)
    total_len: int         # target total context (derived or explicit)

@dataclass
class Curriculum:
    stages: list[tuple[int, EpisodeSpec]]   # (until_step, spec) — ratchets length/lag/distractors
    def spec_at(self, step) -> EpisodeSpec: ...
```

One shared controller; every field is a knob the user named. Same spec drives every task. Eval at
the hard setting throughout (curriculum can mask whether the hard regime is learned).

## Layer O — Objectives: `src/memory/training/objectives/` (package, was `objectives.py`)

> **SUPERSEDED / ABANDONED.** This package split never happened. `src/memory/training/objectives.py`
> is still a single flat file (it grew a `_behavioral_kl_step` alongside the CE/InfoNCE/coding-rate/
> GRPO ladder in place, no package). Treat everything below as the plan that was NOT taken; the
> `behavioral_kl` objective itself did ship, just not via this package structure.

Split the flat `objectives.py` into a package (keep the extracted fns byte-identical):
- `objectives/ce.py` — plain CE / masked_reconstruction (currently in `loops.py`'s compute path).
- `objectives/contrastive.py` ← `_infonce_logits_weights`, `_same_answer_valid_mask`, `_coding_rate`.
- `objectives/behavioral_kl.py` (NEW) — teacher = frozen LM on `[full passage ‖ query]`, student =
  frozen LM on `[memory ‖ query]`, forward `KL(teacher ‖ student)` on the answer/value-span
  positions, teacher stop-gradient, top-k logits, temperature≈2. Its own train step (2 forward
  passes) mirroring `_grad_cached_objective_step`. **The decisive loss-neutrality experiment.**
- `objectives/__init__.py` — `OBJECTIVE_REGISTRY` + the dispatch `_grad_cached_objective_step` uses.

## Orchestration — `mixes.py` + `training/data_mix.py`

- `mixes.py` evolves from `{name → adapter+task_mode}` to **named mixes = weighted
  `[(EpisodeSpec, objective, weight)]`** + a `Curriculum`. Single source of truth for "what trains".
- `training/data_mix.py` rewires from hardcoded per-task branches to **build a loader from a spec**
  (`SOURCE_REGISTRY[spec.source]` × `TASK_REGISTRY[spec.task]` × spec) — adding a source/task no
  longer edits `data_mix`. The trainer asks the mix for the next batch at the current step;
  the mix consults the Curriculum for the difficulty, picks a `(source, task, objective)`, builds it.

---

## Migration order (each phase ends with a verification gate)

**Core refactor (structure change, behavior IDENTICAL at the default spec):**
1. **Sources** — create `sources/`, extract the load-half of each reader, dedup the fineweb loader,
   `SOURCE_REGISTRY`. **Verify:** each source yields items; fineweb source matches the old docs.
2. **Tasks** — create `tasks/`, extract each `_gen` as a Task parametrized by `EpisodeSpec`,
   `TASK_REGISTRY`. **Verify:** each task rebuilds the *identical* sample dict the old `_gen`
   produced (byte-compare a fixed-seed batch).
3. **Schedule** — `schedule.py` (`EpisodeSpec` + `Curriculum`); fold `bio`'s `window_size`/
   `query_window` into it. **Verify:** specs drive the knobs (query-lag placement test as before).
4. **Rewire mixes + data_mix + back-compat** — `mixes.py` named mixes; `data_mix` builds from specs;
   keep `make_<name>_dataloader` thin shims (source+task+default spec) so diagnostics/REGISTRY stay
   green. **Verify:** a 2-step mixed streaming smoke trains **identically** to the pre-refactor
   default (same per-task val_loss within noise); data debug sweep PASS.

**Additive (new capability — after the core is green):**
5. **New sources** — `ingest/{pile,redpajama}` + `sources/{pile,redpajama}.py`; `generate/mqar` +
   `generate/ruler_overwrite` + their sources. **Verify:** load + sample smoke per source.
6. **behavioral_kl objective** — `objectives/` package + the KL step; wire an `objective_mode=kl`.
   **Verify:** a KL step runs (teacher/student through frozen SmolLM2 on value-span-masked bio),
   finite loss/grad; A/B vs plain CE on SHUF−REAL is the experiment (separate run).
7. **Docs** — update `DATASETS.md` (source/task/schedule columns), `HARNESS.md` (objective package),
   write the data-arch section; remove stale reader-layout references.

## Doc reconciliation (no contradictory leftovers)
- `data_reorg_plan.md` — **completed**; still correct for build/store and the 3-layer split. Add a
  header: "Load-layer reader structure superseded by `data_arch_plan.md` (Source/Task split)."
- `harness_reorg_plan.md` — **completed**; complementary. The objectives-package split here extends
  it; add a one-line cross-ref.
- `DATASETS.md` / `HARNESS.md` — **living indexes**; updated in Phase 7 to the new structure (not
  contradictory, just kept in sync).
- Unrelated docs (slotgraph*, furlgraph, biomem, mamba) — untouched.

## Back-compat
User accepts churn. Keep `make_<name>_dataloader` + `REGISTRY` as thin shims through Phase 4 so the
~10 diagnostics and the trainer keep importing one symbol; update call sites and remove shims in
Phase 7 once the spec-based path is the only one.
