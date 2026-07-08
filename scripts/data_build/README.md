# `scripts/data_build/` — the Build layer

Offline producers of `data/`, keyed by canonical dataset `<name>`. Two kinds:

```
scripts/data_build/
  common/          # shared generator machinery (driver, text, drafts)
  generate/        # PROCEDURAL synthesizers (no external source)
    bio/           #   biographical world builder (build_scenario) → the `bio` set
    mqar/          #   runtime-procedural (no offline build step; see generate/mqar/README.md)
    ruler_overwrite/  # runtime-procedural (no offline build step; see its README.md)
  ingest/          # DOWNLOAD + clean/dedup + pretokenize→parquet (see ingest/README.md)
    code/  hotpot_train/  multiwoz/  musique_train/  pile/  quality/  redpajama/  squad/  triviaqa/
    #   fineweb.py, babi_10k.py  (still TODO Tier-B)
  orchestration/   # wave/build drivers: wave1_worldspec{,_extra}.py
```

- **generate** = synthesize from templates/rules; output lands in `data/<name>/`. `mqar/` and
  `ruler_overwrite/` are runtime-procedural sources (nothing to build offline; their `README.md`
  explains why) rather than the old composite bio-sibling families (boxes/calendar/knights/
  passphrase/preferences/revisions/theory_of_mind/triage), which have been removed.
- **ingest** = fetch external data (HF/internet), clean, pretokenize to parquet.
  Most eval datasets are HF-auto-downloaded by their readers (no script) — see
  `ingest/README.md`.
- **orchestration** = the multi-family wave drivers.

## How to regenerate

- **`bio` set** (`data/bio/`) — the biographical world builder is
  `generate/bio/state.py::build_scenario`, driven via `generate/bio/generate.py`.
- **fineweb parquet** (`data/fineweb_edu/`) — `ingest/fineweb.py` (TODO;
  provenance currently unscripted).

Runtime readers live in `src/memory/data/<name>.py`. See `DATASETS.md` at the
repo root for the full per-dataset index tying build ↔ store ↔ load.
