# `ingest/` — download external datasets + clean/dedup + pretokenize → parquet

The **ingest** half of the build layer (the **generate** half is procedural
synthesizers under `../generate/`). Ingest fetches non-procedural data from
HuggingFace / the internet, cleans it, and where useful pretokenizes it to
parquet under `data/<name>/`.

## What is auto-downloaded (no script needed)
The eval readers pull their data from HuggingFace on first use — no ingest
script, and nothing is stored under `data/` for them:

- `babilong`  → HF `RMT-team/babilong`   (reader: `src/memory/data/babilong.py`)
- `hotpot`    → HF `hotpot_qa` distractor (reader: `src/memory/data/hotpot.py`)
- `musique`   → HF `dgslibisey/MuSiQue`   (reader: `src/memory/data/musique.py`)
- `narrativeqa` → HF `narrativeqa`        (reader: `src/memory/data/narrativeqa.py`)

`ruler` is fully synthesized at runtime (no download). `locomo` is a static
JSON auto-downloaded once to `data/eval/locomo10.json`.

## TODO — scripts to add (Tier-B)
- `fineweb.py`  — build `data/fineweb_edu/{train,val}.parquet` (the mae/
  continuation corpus). Currently the parquet provenance is UNSCRIPTED; this
  should rebuild it document-disjoint (the current split has a ~1.25% leak).
- `babi_10k.py` — download Facebook `tasks_1-20_v1-2` en-10k → `data/babi/`
  (the 10k dump needed to reach the 80k-token BABILong-style budgets; the
  current `babi` reader uses the HF `Muennighoff/babi` 1k dump).

See `DATASETS.md` at the repo root for the full per-dataset index.
