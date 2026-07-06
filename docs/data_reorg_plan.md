# Data code + storage reorganization plan (2026-07-06)

Goal: make everything about a dataset findable in one obvious place, with **one consistent `<name>`
across all three layers** and a clear generate → store → load separation. Higher-risk churn is
acceptable; the plan is ordered with a verification step after each move.

## The model — three layers, one name

Every dataset lives in exactly three places, all keyed by the same canonical `<name>`:

| Layer | Where | Contains |
|---|---|---|
| **Build** (offline, runs once → writes `data/`) | `scripts/data_build/{generate,ingest}/<name>` | **generate** = synthesize procedurally (templates/rules); **ingest** = download from HF/internet + clean/dedup/**pretokenize→parquet**/split |
| **Store** (on disk) | `data/<name>/{raw,processed,cache}` | the *output* of the build layer — raw downloads and/or pretokenized parquet (NOT only raw) |
| **Load** (runtime, per batch) | `src/memory/data/<name>.py` | the `Dataset`/`make_<name>_dataloader` + all **runtime** preprocessing (pack, mask, collate, teacher-force align, tokenize-if-not-cached) |

The offline↔runtime line: `ingest` does the expensive one-time work (download, pretokenize→parquet,
split); the `Dataset` does the light per-batch shaping (pack, mask, collate). fineweb does both —
`ingest` pretokenizes to parquet, the `Dataset` re-tokenizes with SmolLM2 + packs + masks.

Rule of thumb: pick a name → build is `scripts/data_build/{generate,ingest}/<name>`, data is
`data/<name>/`, reader is `src/memory/data/<name>.py`. `DATASETS.md` ties the three together.

## Canonical dataset names

| `<name>` | what | source | role | current location(s) → target |
|---|---|---|---|---|
| `bio` | biographical conditioned-reconstruction (the trusted anchor) | procedural | train | `data_conditioned_reconstruction_bio.py` + `..._templates.py` + `scripts/data_gen/tasks/biographical/` + `data/wave1/composite_v1/` |
| `babi` | bAbI relational QA | HF `Muennighoff/babi` → local 10k | train | `data_babi.py` |
| `babilong` | bAbI-in-haystack (long-context) | HF `RMT-team/babilong` | **eval** | `data_qa.py::BABILongDataset` |
| `mae` | masked reconstruction (gist sentinel) | fineweb corpus | train | `data_masked_reconstruction.py` |
| `continuation` | gist continuation (sentinel) | fineweb corpus | train | `data_continuation.py` |
| `mqar` | multi-query associative recall (procedural binding) | procedural | train | **NEW** (Tier-B) |
| `hotpot` | HotpotQA multi-hop | HF | eval | `data_qa.py::HotpotQADataset` |
| `musique` | MuSiQue multi-hop | HF | eval | `data_qa.py::MuSiQueDataset` |
| `narrativeqa` | NarrativeQA | HF | eval | `data_qa.py::NarrativeQADataset` |
| `ruler` | RULER NIAH / variable-tracking | HF/synthetic | eval | `data_qa.py::RULERNIAHDataset` |
| `locomo` | LoCoMo dialogue (dirty — cross-ref only) | static json | eval | `data_qa.py::LoCoMoQADataset` + `data/eval/locomo10.json` |
| `fineweb_edu` | corpus backing mae/continuation | corpus | (source) | `data/fineweb_edu/` |

The composite-family *siblings* (`boxes`, `revisions`, `calendar`, `knights`, `preferences`,
`theory_of_mind`, `passphrase`, `triage`) are generators today; each can graduate to its own reader
`src/memory/data/<name>.py` when wired. They stay under `scripts/data_gen/tasks/<name>/` for now.

---

## Layer 1 — readers: `src/memory/data/` package

Split the `data_qa.py` grab-bag and fold the flat `data_*.py` into one package. Exact mapping:

```
src/memory/data/
  __init__.py        # REGISTRY {name: make_*_dataloader} + re-export QABatch, collate_qa, make_mixed_qa_dataloader
  common.py          # ← data_qa.py: QABatch, _pack_context, QADataset(base), collate_qa, make_qa_dataloader
  babi.py            # ← data_babi.py  (BabiDataset, make_babi_dataloader, _load_babi_rows, DEFAULT_TASKS)
  babilong.py        # ← data_qa.py::BABILongDataset (+ a make_babilong_dataloader)      [EVAL]
  bio.py             # ← data_conditioned_reconstruction_bio.py  (reader)
  bio_render.py      # ← conditioned_reconstruction_bio_templates.py  (runtime render templates)
  mae.py             # ← data_masked_reconstruction.py
  continuation.py    # ← data_continuation.py
  hotpot.py          # ← data_qa.py::HotpotQADataset                                     [EVAL]
  musique.py         # ← data_qa.py::MuSiQueDataset                                      [EVAL]
  narrativeqa.py     # ← data_qa.py::NarrativeQADataset                                  [EVAL]
  ruler.py           # ← data_qa.py::RULERNIAHDataset                                    [EVAL]
  locomo.py          # ← data_qa.py::LoCoMoQADataset                                     [EVAL, dirty]
  mqar.py            # NEW procedural associative-recall                                 [Tier-B, train]
  mixed.py           # ← data_qa.py::MixedQADataset, make_mixed_qa_dataloader (the QA-mix)
```

- `common.py` is the ONLY place the base class / collate / QABatch live; every reader imports from it.
- `__init__.py` exposes `REGISTRY: dict[str, Callable]` (name → `make_*_dataloader`) so `train.py`
  and diagnostics import one symbol, not eleven module paths. Also re-export the hot names
  (`QABatch`, `collate_qa`, `make_mixed_qa_dataloader`) for back-compat.
- `bio.py` keeps its runtime render code in `bio_render.py` (was the `..._templates.py`); the
  *generator* (`build_scenario`) stays in Layer 2 and `bio.py` imports it from there.
- Update import sites (≈10): `scripts/train/train.py` lines 32-36, `scripts/diagnostics/mixed_data_audit.py`,
  `scripts/diagnostics/smoke_slotgraph_mpread.py`, and any others `grep -rl "src.memory.data"` finds.
- `train.py::make_mixed_train_dataloaders` stays in train.py (it's trainer wiring) but imports the
  per-task loaders from `src.memory.data`.

## Layer 2 — build: `scripts/data_build/` (rename from `data_gen`)

The build layer has TWO kinds of producers — **generate** (synthesize) and **ingest** (download +
pretokenize). Rename `data_gen` → `data_build` (it's build, not only generate) and split accordingly:

```
scripts/data_build/          # ← RENAME from scripts/data_gen/
  README.md                  # NEW — how build (generate + ingest) works
  common/                    # keep (composite, driver, sampler, schema, text, drafts)
  generate/                  # ← RENAME tasks/  — PROCEDURAL synthesizers
    bio/                     # ← RENAME tasks/biographical/  (matches data/bio + data/bio.py)
    boxes/ calendar/ knights/ passphrase/ preferences/ revisions/ theory_of_mind/ triage/
    mqar/                    # NEW procedural associative-recall generator  [Tier-B]
  ingest/                    # NEW — download + clean/dedup + pretokenize→parquet + split
    fineweb.py               # build data/fineweb_edu/*.parquet (currently UNSCRIPTED provenance)
    babi_10k.py              # download Facebook tasks_1-20_v1-2 en-10k → data/babi/  [Tier-B]
    README.md                # notes: babilong/hotpot/musique are HF-auto-downloaded by their readers
  orchestration/             # NEW — the wave/build drivers (currently loose at top level)
    wave1_worldspec.py   wave1_worldspec_extra.py   split.py
```

- `generate/` = procedural synthesizers (no external source); `ingest/` = fetch external + pretokenize.
- Rename `tasks/biographical` → `generate/bio`; update `bio.py`'s import
  (`scripts.data_gen.tasks.biographical.state` → `scripts.data_build.generate.bio.state`) and the
  `wave1_worldspec` refs.
- Move `wave1_worldspec*.py` + `split.py` into `orchestration/`.
- `ingest/` gives every non-procedural dataset a reproducible "how to get it," and the fineweb parquet
  stops being unauditable.

## Layer 3 — storage: `data/` by name

```
data/
  README.md                     # NEW — what each folder is + how it was produced
  bio/                          # ← git mv data/wave1/composite_v1/  (raw/ raw_val/ train/ val/)
  fineweb_edu/
    processed/                  # train.parquet val.parquet (pretokenized — the ingest OUTPUT)
    cache/                      # ← RENAME text_cache/ ; DELETE the corrupted val.HuggingFaceTB*.jsonl
  babi/                         # local 10k dump (Tier-B), if adopted
  eval/                         # eval-only static sets
    locomo10.json               # keep
    needle.parquet …            # ← git mv data/wave1/needle*.parquet here
  # HF-cached (Muennighoff/babi, RMT-team/babilong, hotpot, musique…) stay in ~/.cache/huggingface
```

Per-dataset `raw/` (untouched download) · `processed/` (pretokenized parquet / split) · `cache/`
(runtime-tokenizer caches) as needed — store holds the build layer's output, not only raw text.

- Replace the opaque `data/wave1/composite_v1/` with `data/bio/` (if versioning is needed, use
  `data/bio/v1/` — a name+version, not "wave1/composite_v1").
- Update `train.py` `COMPOSITE_TRAIN_P/Q`, `COMPOSITE_VAL_P/Q` (lines 42-45) to `data/bio/...`.
- Delete the corrupted cache file the audit found; add the tokenizer-fingerprint guard to
  `mae.py::_decode_cache` so a mismatched cache regenerates instead of silently reusing.

## The index — `DATASETS.md` (repo root)

One table, the single source of truth for a new developer:
`name | what | source | role (train/eval) | reader | generator/download | data | status | gotchas`.
Gotchas column carries the audit findings (BABILong only qa1-10 at ≥1k; condrecon value-span-mask
fix; fineweb ~1.25% leak; babi 10k needed for 80k).

---

## Migration order (each step ends with a verification)

1. **Reader package** — create `src/memory/data/`, move + split per the Layer-1 map, add
   `__init__.py` registry, update all import sites. **Verify:** `python -c "import src.memory.data"`
   + rerun the data debug sweep (babi/bio/babilong load a batch).
2. **Storage** — `git mv` composite_v1→`data/bio`, needle→`data/eval`, rename `text_cache`→`cache`,
   delete corrupted file, update `train.py` COMPOSITE_* + `_decode_cache` fingerprint. **Verify:**
   mixed-QA loader + mae loader find their data.
3. **Generators** — rename `tasks/biographical`→`tasks/bio`, create `orchestration/` + `download/`,
   fix the `bio.py`→generator import. **Verify:** `build_scenario` import + condrecon reader batch.
4. **Docs** — write `DATASETS.md`, `data/README.md`, `scripts/data_gen/README.md`, cross-ref docstrings.
5. **Fold in the data-correctness fixes** (we're already in the code): condrecon value-span content
   mask + the two generator one-liners; BABILong eval-only + loaded==requested assertion; babi
   offline-fallback fix + split-name whitelist. Then Tier-B: `download/babi_10k.py`, `mqar.py`,
   `download/fineweb.py` rebuild (document-disjoint), WikiBigEdit reader.

## Back-compat
The user accepts churn, so no permanent shims — update all import sites in step 1. Optional: a
one-release `data_qa.py`/`data_babi.py` that just `from .data.X import *` + `DeprecationWarning`, so
any stray external caller fails loudly rather than silently. Remove after one green run.
