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
| `diagnostics/` | Grouped by subject: `slotgraph3/`, `slotgraph/`, `biomem/`, `objective/`, `mixed/`, `cohort/`. Cohort evaluation, slotgraph probes, band+gate eval, dashboards. |
| `data_build/`  | Data build layer (generate + ingest), not `data_gen/` (renamed 2026-07-06). Synthetic generators (`generate/bio/`, `generate/mqar/`, `generate/ruler_overwrite/`) + HF-download ingest scripts (`ingest/<name>/download.py`). Loaders live in `src/memory/data/` (the `data_qa.py`/`data_conditioned_reconstruction_bio.py`/`conditioned_reconstruction_bio_templates.py` flat files this row used to name are gone — superseded by the `src/memory/data/{sources,tasks}/` package; see `DATASETS.md`). |

## diagnostics/ quick reference

```bash
python scripts/diagnostics/cohort/cohort_results.py          # build docs/history/cohort_results.md (old slotgraph cohort) from run JSONLs + checkpoints
python scripts/diagnostics/cohort/debug_sweep_new_models.py
python scripts/diagnostics/cohort/analyze_sentence_lengths.py
python scripts/diagnostics/slotgraph/slotgraph_diag.py
python scripts/diagnostics/slotgraph/slotgraph_gradflow.py
python scripts/diagnostics/slotgraph/smoke_slotgraph.py
python scripts/diagnostics/slotgraph/smoke_slotgraph_mpread.py
python scripts/diagnostics/slotgraph3/slotgraph3_diag.py
python scripts/diagnostics/slotgraph3/slotgraph3_tokgeom_sweep.py
python scripts/diagnostics/slotgraph3/slotgraph3_validity_battery.py
python scripts/diagnostics/mixed/mixed_band_gate_eval.py     # REAL/SHUF/OFF band + binding gate over the cohort
python scripts/diagnostics/mixed/mixed_dashboard.py          # per-task training/val dashboard from run JSONLs
python scripts/diagnostics/mixed/mixed_data_audit.py
python scripts/diagnostics/mixed/data_stats.py
python scripts/diagnostics/mixed/episode_peek.py
python scripts/diagnostics/objective/objective_debug.py
python scripts/diagnostics/objective/objective_profile.py
python scripts/diagnostics/biomem/biomem_stage0_probe.py
python scripts/diagnostics/biomem/smoke_biomem.py
```

(`slotgraph_attribution.py`/`slotgraph_metrics.py`, which used to regenerate
`docs/history/slotgraph_attribution.md`/`docs/history/slotgraph_metrics.md`, and the
`slotgraph_ablation_probe.py`/`slotgraph_rank_probe.py`/`slotgraph_topology_probe.py` probes, no longer
exist — those two docs are now frozen snapshots.)

## train/

```bash
.venv/bin/python scripts/train/train.py --task mixed \
    --variants slotgraph_baseline biomem_baseline icae_baseline ccm_baseline \
               autocompressor_baseline beacon_baseline \
    --steps 8000 --warmup 500 --val-every 500 --batch-size 8
```

## Where things live

- Models (one self-contained folder each): `src/memory/models/`
- Results and design notes: `docs/` (see `docs/README.md`)
