# Datasets — the index

One canonical `<name>` per dataset, tied across all three layers:

- **build** (offline) — `scripts/data_build/{generate,ingest}/<name>` — *generate* =
  synthesize procedurally; *ingest* = download + clean + pretokenize→parquet.
- **store** (on disk) — `data/<name>/` — the build layer's output (gitignored).
- **load** (runtime) — `src/memory/data/<name>.py` — the `Dataset` + `make_<name>_dataloader`
  (registered in `src/memory/data/__init__.py::REGISTRY`).

See `scripts/data_build/README.md` and `data/README.md` for the layer mechanics.

| name | what | source | role | reader (`src/memory/data/`) | build (`scripts/data_build/`) | data (`data/` or cache) | status | gotchas |
|---|---|---|---|---|---|---|---|---|
| `bio` | biographical conditioned-reconstruction (trusted anchor) | procedural | train | `bio.py` (+ `bio_render.py`) | `generate/bio/` (`build_scenario`) | `data/bio/{raw,raw_val,train,val}` | active | condrecon value-span content-mask fix is hand-applied separately (do not touch here) |
| `babi` | bAbI relational QA (story→question→1-word answer) | HF `Muennighoff/babi` (1k) | train | `babi.py` | `ingest/babi_10k.py` (TODO) | HF cache (`data/babi/` = Tier-B 10k) | active | reader uses the HF **1k** dump; the **10k** dump is needed for 80k-token budgets. Offline fallback only synthesizes task-1 (raises otherwise). Unknown split names raise. |
| `babilong` | bAbI-in-haystack (long-context state-tracking) | HF `RMT-team/babilong` | **eval** | `babilong.py` | HF-auto (reader) | HF cache | active | only **qa1-qa10** exist at configs ≥1k — requesting qa11-qa20 there loads fewer tasks (now WARNs, not silent) |
| `mae` | masked reconstruction (gist sentinel) | fineweb-edu corpus | train | `mae.py` | `ingest/fineweb.py` (TODO) | `data/fineweb_edu/{train,val}.parquet` + `cache/` | active | fineweb split has a ~1.25% train/val leak (rebuild document-disjoint). Text cache is now tokenizer-fingerprinted (`.meta`) so a mismatched-tokenizer cache regenerates. |
| `continuation` | gist continuation (compress prefix → predict next span) | fineweb-edu corpus | train | `continuation.py` | `ingest/fineweb.py` (TODO) | `data/fineweb_edu/` (shares `cache/`) | active | same fineweb ~1.25% leak as `mae` |
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
