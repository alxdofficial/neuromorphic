# `data/` — the Store layer

On-disk output of the build layer (`scripts/data_build/`), keyed by canonical
dataset `<name>`. **Everything under `data/` is gitignored** (large binaries kept
locally, not tracked) — regenerate from the build layer or re-download.

```
data/
  bio/                     # biographical conditioned-reconstruction world set
    raw/  raw_val/         #   per-family raw generator output (train / val)
    train/  val/           #   packed passages.jsonl + questions.jsonl (the reader input)
  fineweb_edu/             # corpus backing mae + continuation
    train.parquet  val.parquet   # pretokenized (Llama-3 ids) — the ingest OUTPUT
    cache/                 # runtime decoded-text caches (SmolLM2 re-tokenization),
                           #   tokenizer-fingerprinted via sidecar .meta
  code/  hotpot_train/  multiwoz/  musique_train/  pile/  quality/  redpajama/
  squad/  triviaqa/        # training sources ingested via scripts/data_build/ingest/<name>/
                           #   download.py; each holds train.jsonl + val.jsonl
  babilong_train/  perltqa/  swe_trajectories/  wikibigedit/
                           # Phase-1 full-corpus sources (shipped 2026-07-08, staged locally;
                           #   NOT yet in the scrutiny-phase mix — see docs/DATA_PHASES_PLAN.md).
                           #   msc/qasper/longcite/govreport/pg19/wildchat/lmsys_chat are the
                           #   remaining Phase-1 sources — same ingest pattern, need a one-time
                           #   `scripts/data_build/ingest/<name>/download.py` run to stage locally.
  eval/                    # eval-only static sets
    locomo10.json          #   LoCoMo dialogues
    needle*.parquet        #   needle-in-a-haystack sets
  # HF-cached datasets (Muennighoff/babi, RMT-team/babilong, hotpot_qa,
  # dgslibisey/MuSiQue, narrativeqa, longmemeval, longbench, infinitebench) live
  # in ~/.cache/huggingface, not here.
```

## How to (re)produce

- **`data/bio/`** — run the generator/orchestration under
  `scripts/data_build/generate/bio/` (world builder `build_scenario`). The
  trainer reads `data/bio/{train,val}/{passages,questions}.jsonl`.
- **`data/fineweb_edu/*.parquet`** — provenance currently UNSCRIPTED; the
  reproducible builder is `scripts/data_build/ingest/fineweb.py` (TODO). The
  `cache/` dir is auto-populated by the `mae`/`continuation` readers on first use.
- **`data/{code,hotpot_train,multiwoz,musique_train,pile,quality,redpajama,squad,triviaqa}/`** —
  each built by `scripts/data_build/ingest/<name>/download.py`.
- **`data/{babilong_train,perltqa,swe_trajectories,wikibigedit}/`** — same pattern, Phase-1
  full-corpus sources; built by `scripts/data_build/ingest/<name>/download.py`.
- **`data/eval/locomo10.json`** — auto-downloaded once by `src/memory/data/locomo.py`.
- HF datasets (`babi`, `babilong`, `hotpot`, `musique`, `narrativeqa`, `longmemeval`, `longbench`,
  `infinitebench`) are auto-downloaded by their readers to the HF cache — nothing stored here.

See `DATASETS.md` at the repo root for the full per-dataset index.
