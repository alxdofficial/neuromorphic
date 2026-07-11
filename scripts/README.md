# scripts/

Entry-point scripts, grouped by purpose. Run from the repo root with the venv
active (imports use `src.memory.*`):

```bash
source .venv/bin/activate
python scripts/<subdir>/<script>.py [args]
```

## Layout

| Subdir         | What's inside |
| -------------- | ------------- |
| `train/`       | `train.py` (thin CLI entrypoint) + `cli.py` (`build_parser`/`args_to_config`) — the single training harness for every variant + task (default `--task mixed`). Reusable trainer/objective/eval logic lives in `src/memory/training/` (a library, not here). |
| `diagnostics/` | Grouped by subject: `mixed/`, `cohort/`, `slotgraph/`. Cohort evaluation, band+gate eval, dashboards, episode/data audits, slotgraph gradient-flow. |
| `data_build/`  | Data build layer (generate + ingest), not `data_gen/` (renamed 2026-07-06). Synthetic generators (`generate/bio/`, `generate/mqar/`, `generate/ruler_overwrite/`) + HF-download ingest scripts (`ingest/<name>/download.py`). Loaders live in `src/memory/data/` (the `data_qa.py`/`data_conditioned_reconstruction_bio.py`/`conditioned_reconstruction_bio_templates.py` flat files this row used to name are gone — superseded by the `src/memory/data/{sources,tasks}/` package; see `DATASETS.md`). |

## diagnostics/ quick reference

```bash
python scripts/diagnostics/cohort/cohort_results.py          # build a cohort head-to-head from run JSONLs + checkpoints
python scripts/diagnostics/cohort/debug_sweep_new_models.py
python scripts/diagnostics/cohort/analyze_sentence_lengths.py
python scripts/diagnostics/mixed/mixed_band_gate_eval.py     # REAL/SHUF/OFF band + binding gate over the cohort
python scripts/diagnostics/mixed/mixed_dashboard.py          # per-task training/val dashboard from run JSONLs
python scripts/diagnostics/mixed/mixed_data_audit.py         # structure/firewall audit over the exact loaders
python scripts/diagnostics/mixed/data_stats.py
python scripts/diagnostics/mixed/episode_peek.py             # decode real episodes + per-(task×source) stats
python scripts/diagnostics/slotgraph/slotgraph_gradflow.py   # per-module grad/param ratios (slotgraph canary)
```

(The slotgraph1/2/3 and biomem/objective probe scripts were removed 2026-07-11 with their arms; the
`docs/history/slotgraph_*` docs they regenerated are now frozen snapshots.)

## train/

```bash
.venv/bin/python scripts/train/train.py --task mixed \
    --variants icae_baseline autocompressor_baseline titans_baseline \
               gisting_baseline memoryllm_baseline slotgraph_baseline \
    --objective-mode behavioral_kl --batch-size 4 --steps 8000 --warmup 500 --val-every 500
```
(These are the active-cohort defaults — running `train.py --task mixed` with no `--variants` already
selects them + the eval-only h2o/vanilla refs. Titans auto-disables the streaming checkpoint per-arm.)

## Where things live

- Models (one self-contained folder each): `src/memory/models/`
- Results and design notes: `docs/` (see `docs/README.md`)
