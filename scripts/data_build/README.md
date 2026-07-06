# `scripts/data_build/` — the Build layer

Offline producers of `data/`, keyed by canonical dataset `<name>`. Two kinds:

```
scripts/data_build/
  common/          # shared generator machinery (composite, driver, sampler, schema, text, drafts)
  generate/        # PROCEDURAL synthesizers (no external source)
    bio/           #   biographical world builder (build_scenario) → the composite `bio` set
    boxes/  calendar/  knights/  passphrase/  preferences/  revisions/  theory_of_mind/  triage/
    #   mqar/      #   (TODO Tier-B) procedural associative-recall generator
  ingest/          # DOWNLOAD + clean/dedup + pretokenize→parquet + split (see ingest/README.md)
    #   fineweb.py, babi_10k.py  (TODO Tier-B)
  orchestration/   # wave/build drivers: wave1_worldspec{,_extra}.py, split.py
```

- **generate** = synthesize from templates/rules; output lands in `data/<name>/`.
- **ingest** = fetch external data (HF/internet), clean, pretokenize to parquet.
  Most eval datasets are HF-auto-downloaded by their readers (no script) — see
  `ingest/README.md`.
- **orchestration** = the multi-family wave drivers + train/val split logic.

## How to regenerate

- **composite `bio` set** (`data/bio/`) — drive the per-family generators under
  `generate/<family>/` and combine via `common/composite.py`; the biographical
  world comes from `generate/bio/state.py::build_scenario`. Split with
  `orchestration/split.py`.
- **fineweb parquet** (`data/fineweb_edu/`) — `ingest/fineweb.py` (TODO;
  provenance currently unscripted).

Runtime readers live in `src/memory/data/<name>.py`. See `DATASETS.md` at the
repo root for the full per-dataset index tying build ↔ store ↔ load.
