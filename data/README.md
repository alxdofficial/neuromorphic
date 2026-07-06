# `data/` — the Store layer

On-disk output of the build layer (`scripts/data_build/`), keyed by canonical
dataset `<name>`. **Everything under `data/` is gitignored** (large binaries kept
locally, not tracked) — regenerate from the build layer or re-download.

```
data/
  bio/                     # composite biographical conditioned-reconstruction set
    raw/  raw_val/         #   per-family raw generator output (train / val)
    train/  val/           #   packed passages.jsonl + questions.jsonl (the reader input)
  fineweb_edu/             # corpus backing mae + continuation
    train.parquet  val.parquet   # pretokenized (Llama-3 ids) — the ingest OUTPUT
    cache/                 # runtime decoded-text caches (SmolLM2 re-tokenization),
                           #   tokenizer-fingerprinted via sidecar .meta
  eval/                    # eval-only static sets
    locomo10.json          #   LoCoMo dialogues
    needle*.parquet        #   needle-in-a-haystack sets
  # HF-cached datasets (Muennighoff/babi, RMT-team/babilong, hotpot_qa,
  # dgslibisey/MuSiQue, narrativeqa) live in ~/.cache/huggingface, not here.
```

## How to (re)produce

- **`data/bio/`** — run the composite generator/orchestration under
  `scripts/data_build/generate/bio/` + `scripts/data_build/common/composite.py`
  (world builder `build_scenario`). The trainer reads
  `data/bio/{train,val}/{passages,questions}.jsonl`.
- **`data/fineweb_edu/*.parquet`** — provenance currently UNSCRIPTED; the
  reproducible builder is `scripts/data_build/ingest/fineweb.py` (TODO). The
  `cache/` dir is auto-populated by the `mae`/`continuation` readers on first use.
- **`data/eval/locomo10.json`** — auto-downloaded once by `src/memory/data/locomo.py`.
- HF datasets (`babi`, `babilong`, `hotpot`, `musique`, `narrativeqa`) are
  auto-downloaded by their readers to the HF cache — nothing stored here.

See `DATASETS.md` at the repo root for the full per-dataset index.
