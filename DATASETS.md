# Datasets — the index

> **Papers & source links for every dataset AND baseline: [`docs/REFERENCES.md`](docs/REFERENCES.md)** — the authoritative reference index so we never re-search a citation.

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
| `babi` | bAbI relational QA (story→1-word answer) | qa | train | `babi.py` × `qa` | `ingest/babi_10k.py` (TODO) | HF **1k** dump; 10k needed for 80k budgets. Offline fallback = task-1 only (raises otherwise). Multi-segment entity-renamed co-packing (`pack_rename`), not a single story — see `SCRUTINY_PHASE_DATA.md` §4. |
| `fineweb` | fineweb-edu corpus | corpus | train | `fineweb.py` × `mae` (backs the `reconstruct` mix-task directly) | `ingest/fineweb.py` (TODO) | ~1.25% train/val leak (rebuild document-disjoint). **src_tokenizer = `meta-llama/Llama-3.2-1B`**, cache `.meta`-fingerprinted. Also one of `multicorpus`'s 4 pools (which backs `continuation`, not `fineweb` directly). |
| `pile` | The Pile natural text (bucket-1) | corpus | train | `pile.py` × `continuation`/`mae` | `ingest/pile/download.py` | `NeelNanda/pile-10k` stream; local-jsonl → HF-stream → "run ingest" error. Reachable only inside `multicorpus`. |
| `redpajama` | SlimPajama (dedup RedPajama, bucket-1) | corpus | train | `redpajama.py` × `continuation`/`mae` | `ingest/redpajama/download.py` | `DKYoon/SlimPajama-6B` (classic RedPajama-1T loader unsupported in datasets 4.5). Reachable only inside `multicorpus`. |
| `code` | source code (codeparrot-clean, Python) | corpus | train | `code.py` × `continuation`/`mae` | `ingest/code/download.py` | un-guessable **exact-token** binding (`def foo(...)` / `const KEY=` referenced far later) — the "hard middle" fineweb/bio under-supply. Reachable only inside `multicorpus`. |
| `multicorpus` | corpus VARIETY union (fineweb + pile + redpajama + code) | corpus (composite) | train | `multicorpus.py` × `continuation` | (composes the 4 above; no own ingest) | backs the `continuation` mix-task; **corpus-uniform** sampling (each loaded corpus gets equal weight, not doc-uniform, so fineweb's larger doc count doesn't drown the others); robust to any pool being unreachable/not-yet-ingested (skip + warn, fineweb is the always-local anchor). |
| `squad` | SQuAD 2.0 extractive QA (short paragraph; v2 unanswerable → `"unanswerable"`) | qa | train | `squad.py` × `qa` (via `qa_multi`) | `ingest/squad/download.py` | short (~150 tok) contexts, padded to budget via `distractor_pool`. |
| `triviaqa` | TriviaQA open-domain factoid + evidence | qa | train | `triviaqa.py` × `qa` (via `qa_multi`) | `ingest/triviaqa/download.py` | answer-centred evidence window (sentence-split); carries answer aliases in `meta` for EM; examples whose evidence never contains the answer are skipped at load. |
| `hotpot_train` | HotpotQA multi-hop, **TRAIN split** | qa | train | `hotpot_train.py` × `qa` (via `qa_multi`) | `ingest/hotpot_train/download.py` | gold paragraphs FIRST, capped ~900 tok; **firewalled** from the eval `hotpot` reader's validation split. |
| `musique_train` | MuSiQue-Ans 2-4 hop, **TRAIN split** | qa | train | `musique_train.py` × `qa` (via `qa_multi`) | `ingest/musique_train/download.py` | supporting paragraphs FIRST, capped ~900 tok; **firewalled** from the eval `musique` reader's validation split. |
| `multiwoz` | MultiWOZ 2.2 dialogue → slot-recall QA | qa | train | `multiwoz.py` × `qa` (via `qa_multi`) | `ingest/multiwoz/download.py` | canonical GitHub JSON; slot value stated verbatim earlier in the dialogue = un-guessable exact recall. |
| `quality` | QuALITY long-document multiple-choice RC | qa | train (registered, unwired) | `quality.py` × `qa` | `ingest/quality/download.py` | needs `total_len>=4096` (story ~4.7k-7.6k tok) — excluded from `qa_multi`'s default pool at the current `total_len=2048`; usable standalone. |
| `qa_multi` | QA VARIETY union (squad + triviaqa + hotpot_train + musique_train + multiwoz) | qa (composite) | train | `qa_multi.py` × `qa` | (composes the 5 above; no own ingest) | backs the `doc_qa` mix-task; `pack_n_queries=(1,2)` (big RC contexts → only ~2 golds fit at 2048); robust to any sub-source being unreachable/not-yet-ingested (skip + warn). bAbI is intentionally NOT unioned in (needs `pack_rename`, which only applies to the top-level source). |
| `mqar` | random-token multi-query associative recall | keyed | train | `mqar.py` × `reconstruction` | runtime-procedural | un-guessable binding kill-switch (Zoology); multi-query = addressing. |
| `ruler_overwrite` | same-key reassignment (v1→v2), query returns latest | keyed | train | `ruler_overwrite.py` × `overwrite` | runtime-procedural | T2 forced-forgetting probe (RULER-fork). answer = latest binding, stale = distractor. |
| `babilong` | bAbI-in-haystack (long-context) | qa | **eval** | flat `babilong.py` (REGISTRY) | HF-auto | only **qa1-qa10** at configs ≥1k (WARNs). |
| `hotpot` | HotpotQA multi-hop (distractor) | HF `hotpot_qa` | eval | `hotpot.py` | HF-auto (reader) | HF cache | active | ~570MB first download; supporting paragraphs guaranteed in-context |
| `musique` | MuSiQue-Ans 2-4 hop | HF `dgslibisey/MuSiQue` | eval | `musique.py` | HF-auto (reader) | HF cache | active | answerable-only subset (filtered) |
| `narrativeqa` | NarrativeQA (summaries-only) | HF `narrativeqa` | eval | `narrativeqa.py` | HF-auto (reader) | HF cache | active | summaries-only setting; abstractive answers → headline metric is the LLM judge, EM/containment secondary |
| `ruler` | RULER multi-key needle-in-a-haystack | synthetic (runtime) | eval | `ruler.py` | runtime-synth (no build) | — (none) | active | single-needle default; 4-needle/64:1 is unwinnable for compressive memory |
| `locomo` | LoCoMo very-long-term dialogue | static JSON (snap-research/locomo) | eval | `locomo.py` | download-once (reader) | `data/eval/locomo10.json` | active (dirty) | **dirty — cross-ref only**; convs up to ~24k tokens → run at a large `--chunk-size` or late-session evidence is truncated |

## `bio` internal builders (NOT separate composite task families — that design was removed)

An earlier plan (`docs/history/data_reorg_plan.md`, historical) had `bio` alongside eight sibling
*task-family* generators (`boxes`/`revisions`/`calendar`/`knights`/`preferences`/`theory_of_mind`/
`passphrase`/`triage`) meant to graduate into their own readers. **That composite-sibling design was
removed** — those eight generator directories no longer exist under `scripts/data_build/generate/`.
The single `bio` world generator now composes its facts from internal **builders** under
`scripts/data_build/generate/bio/builders/`: `people`, `orgs`, `nations`, `places`, `events`, `works`,
`edges` (relationship wiring) — these feed `build_scenario()` directly, they are not standalone
mix-task entries and never will be (no `src/memory/data/<name>.py` reader is planned for them).

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
