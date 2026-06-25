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
| `train/`       | `train.py` — the single training harness for every variant + task (default `--task mixed`). |
| `diagnostics/` | Cohort evaluation, slotgraph attribution/metrics/probes, band+gate eval, dashboards. |
| `data_gen/`    | Synthetic data generation for the composite-QA + conditioned-reconstruction-bio universe (biographical worldspec, task generators). Imported by `src/memory/data_qa.py`, `data_conditioned_reconstruction_bio.py`, `conditioned_reconstruction_bio_templates.py`. |

## diagnostics/ quick reference

```bash
python scripts/diagnostics/cohort_results.py          # build docs/cohort_results.md from run JSONLs + checkpoints
python scripts/diagnostics/slotgraph_metrics.py       # standing instrument panel → docs/slotgraph_metrics.md
python scripts/diagnostics/slotgraph_attribution.py   # 2×2 attribution (MP vs id-tags) → docs/slotgraph_attribution.md
python scripts/diagnostics/slotgraph_ablation_probe.py
python scripts/diagnostics/slotgraph_diag.py
python scripts/diagnostics/slotgraph_gradflow.py
python scripts/diagnostics/slotgraph_rank_probe.py
python scripts/diagnostics/slotgraph_topology_probe.py
python scripts/diagnostics/mixed_band_gate_eval.py    # REAL/SHUF/OFF band + binding gate over the cohort
python scripts/diagnostics/mixed_dashboard.py         # per-task training/val dashboard from run JSONLs
python scripts/diagnostics/biomem_stage0_probe.py
python scripts/diagnostics/smoke_biomem.py
python scripts/diagnostics/smoke_slotgraph.py
python scripts/diagnostics/smoke_slotgraph_mpread.py
python scripts/diagnostics/debug_sweep_new_models.py
python scripts/diagnostics/analyze_sentence_lengths.py
```

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
