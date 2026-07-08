# Datasets — the index

**Data is 4 orthogonal layers (2026-07-07 refactor — see `docs/data_arch_plan.md`):**
- **Source** (*where tokens come from*) — `src/memory/data/sources/<name>.py` + `SOURCE_REGISTRY`.
  Yields raw items (`CorpusItem`/`KeyedItem`/`QAItem`); knows nothing about windows/queries.
- **Task** (*what's presented/asked*) — `src/memory/data/tasks/<style>.py` + `get_task`. Shapes
  items → the on-the-wire sample. Styles: `reconstruction`, `qa`, `continuation`, `mae`, `overwrite`.
- **Schedule** (*how hard*) — `src/memory/data/schedule.py` `EpisodeSpec` + `Curriculum`.
- **Objective** (*how scored*) — `src/memory/training/objectives.py`: `plain` CE / `contrastive` /
  `behavioral_kl` (context distillation) — composes with any task. **task ≠ objective.**

Build/store still per-name: **build** `scripts/data_build/{generate,ingest}/<name>`; **store**
`data/<name>/`. Training mixes = `(source, task, spec)` in `src/memory/data/mixes.py::TASK_SPEC`,
composed by `src/memory/training/data_mix.py`. `REGISTRY` now holds only the **eval** flat readers.
See `HARNESS.md` for the training wiring.

| name | what | kind | role | source (`data/sources/`) × task | build | gotchas |
|---|---|---|---|---|---|---|
| `bio` | biographical conditioned-reconstruction (trusted anchor) | keyed | train | `bio.py` × `reconstruction` (+ `bio_render.py`) | `generate/bio/` | loss on **fact-value spans only** (`value_subs` → task's `_value_ids_content`), excl. name + template, so SHUF−REAL charges for binding not boilerplate. `_name_derivable` drops key-derivable `short_name`/`org_type`/`work_type`. query-lag placement via `EpisodeSpec.query_lag`. |
| `babi` | bAbI relational QA (story→1-word answer) | qa | train | `babi.py` × `qa` | `ingest/babi_10k.py` (TODO) | HF **1k** dump; 10k needed for 80k budgets. Offline fallback = task-1 only (raises otherwise). |
| `fineweb` | fineweb-edu corpus (backs mae + continuation) | corpus | train | `fineweb.py` × `mae` / `continuation` | `ingest/fineweb.py` (TODO) | ~1.25% train/val leak (rebuild document-disjoint). **src_tokenizer = `meta-llama/Llama-3.2-1B`**, cache `.meta`-fingerprinted. |
| `mqar` | random-token multi-query associative recall | keyed | train | `mqar.py` × `reconstruction` | runtime-procedural | un-guessable binding kill-switch (Zoology); multi-query = addressing. |
| `ruler_overwrite` | same-key reassignment (v1→v2), query returns latest | keyed | train | `ruler_overwrite.py` × `overwrite` | runtime-procedural | T2 forced-forgetting probe (RULER-fork). answer = latest binding, stale = distractor. |
| `pile` | The Pile natural text (bucket-1) | corpus | train | `pile.py` × `continuation`/`mae` | `ingest/pile/download.py` | `NeelNanda/pile-10k` stream; local-jsonl → HF-stream → "run ingest" error. |
| `redpajama` | SlimPajama (dedup RedPajama, bucket-1) | corpus | train | `redpajama.py` × `continuation`/`mae` | `ingest/redpajama/download.py` | `DKYoon/SlimPajama-6B` (classic RedPajama-1T loader unsupported in datasets 4.5). |
| `babilong` | bAbI-in-haystack (long-context) | qa | **eval** | flat `babilong.py` (REGISTRY) | HF-auto | only **qa1-qa10** at configs ≥1k (WARNs). |
| `hotpot` | HotpotQA multi-hop (distractor) | HF `hotpot_qa` | eval | `hotpot.py` | HF-auto (reader) | HF cache | active | ~570MB first download; supporting paragraphs guaranteed in-context |
| `musique` | MuSiQue-Ans 2-4 hop | HF `dgslibisey/MuSiQue` | eval | `musique.py` | HF-auto (reader) | HF cache | active | answerable-only subset (filtered) |
| `narrativeqa` | NarrativeQA (summaries-only) | HF `narrativeqa` | eval | `narrativeqa.py` | HF-auto (reader) | HF cache | active | summaries-only setting; abstractive answers → headline metric is the LLM judge, EM/containment secondary |
| `ruler` | RULER multi-key needle-in-a-haystack | synthetic (runtime) | eval | `ruler.py` | runtime-synth (no build) | — (none) | active | single-needle default; 4-needle/64:1 is unwinnable for compressive memory |
| `locomo` | LoCoMo very-long-term dialogue | static JSON (snap-research/locomo) | eval | `locomo.py` | download-once (reader) | `data/eval/locomo10.json` | active (dirty) | **dirty — cross-ref only**; convs up to ~24k tokens → run at a large `--chunk-size` or late-session evidence is truncated |
| `mqar` | multi-query associative recall (procedural binding) | procedural | train | `mqar.py` (TODO) | `generate/mqar.py` (TODO) | — | planned (Tier-B) | not yet wired |
| `fineweb_edu` | corpus backing `mae`/`continuation` | corpus | (source) | — (via `mae`/`continuation`) | `ingest/fineweb.py` (TODO) | `data/fineweb_edu/` | active | parquet provenance currently UNSCRIPTED; rebuild document-disjoint to fix the leak |

## Composite `bio` siblings (generators only, not yet standalone readers)

The composite `data/bio/` set is produced from nine structure-necessary task families.
`bio` (biographical) is the one wired as a standalone reader; the other eight remain
generators under `scripts/data_build/generate/<name>/` and enter training only through the
composite mix:

`boxes` · `revisions` · `calendar` · `knights` · `preferences` · `theory_of_mind` ·
`passphrase` · `triage`

Each can graduate to its own `src/memory/data/<name>.py` reader when wired.

## Eval-only static assets

`data/eval/` holds eval-only static sets: `locomo10.json` (LoCoMo) and the pre-generated
`needle*.parquet` (needle-in-a-haystack) sets.

## Phase-1 & Phase-2 sources (2026-07-08) — see `docs/DATA_PHASES_PLAN.md`

**Phase-1 training sources** (`SOURCE_REGISTRY`, `sources/`): long real conversations — `wildchat`
(allenai/WildChat-1M), `lmsys_chat` (lmsys/lmsys-chat-1m — **GATED**, needs HF access), `msc`
(Multi-Session Chat); long docs — `qasper`, `longcite`, `govreport`, `pg19` (books, ~1M tok);
procedural — `ruler_niah` (NIAH/multikey/vartrack generator + overwrite-fork base), `babilong_train`
(bAbI-in-PG19, TRAIN — distinct from the eval `babilong`); streaming/agentic — `wikibigedit`
(sequential fact-update / forced-forgetting), `swe_trajectories` (nebius, reward-labeled), `perltqa`
(personalization). STAGED locally: babilong_train, wikibigedit, swe_trajectories, perltqa. Need a
one-time ingest run (`scripts/data_build/ingest/<name>/download.py`): wildchat, msc, qasper, longcite,
govreport, pg19. Gated: lmsys_chat.

**Phase-2 eval readers** (`REGISTRY`, `data/`): `longmemeval` (HEADLINE — multi-session chat memory),
`longbench` (v1 21-subtask + v2 MCQs), `infinitebench` (Retrieve.* + En.QA), `niah` (procedural).
Already had: babilong, hotpot, musique, narrativeqa, ruler, locomo.

datasets 4.x dropped loading-script support → qasper uses `revision="refs/convert/parquet"`, pg19 uses
`emozilla/pg19`, longbench pulls `data.zip` directly.
