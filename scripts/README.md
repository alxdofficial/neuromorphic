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
| `train/`       | `train.py` — the single training harness for every variant + task (the active task is `--task masked_reconstruction`). |
| `diagnostics/` | Smoke tests, the floor/ceiling band scan, and the hierarchical-learned-vocab debug sweeps. |
| `data_gen/`    | Synthetic data generation for the (dormant, kept) composite-QA + conditioned-reconstruction-bio universe (biographical worldspec, task generators). Imported by `src/memory/data_qa.py`, `data_conditioned_reconstruction_bio.py`, `conditioned_reconstruction_bio_templates.py`. |

## diagnostics/ quick reference (the active compression line)
```bash
python scripts/diagnostics/mae_smoke.py            # pre-flight: all variants construct/train/grad-flow on masked_reconstruction
python scripts/diagnostics/mae_band_scan.py        # the floor(no-mem) / ceiling(full-ctx) band
python scripts/diagnostics/hlvocab_diag.py         # hlvocab: load a ckpt, gradient/routing/collapse sweep
python scripts/diagnostics/hlvocab_graph_diag.py   # hlvocab v2 graph: fresh-init gradient + selection health sweep
python scripts/diagnostics/hlvocab_eff_rank.py     # hlvocab: per-slot value / memory effective-rank probe
python scripts/diagnostics/analyze_sentence_lengths.py
```

## train/
```bash
python scripts/train/train.py --task masked_reconstruction --backbone HuggingFaceTB/SmolLM2-135M \
    --variants hlvocab_baseline icae_baseline ccm_baseline autocompressor_baseline beacon_baseline \
    --steps 4000 --warmup 200 --val-every 500 --batch-size 16 --out-tag <tag>
```

## Where things live
- Current model design: [`docs/compression_model_design.md`](../docs/compression_model_design.md)
- Objective + baseline band + results: [`docs/compression_objective.md`](../docs/compression_objective.md)
- Models (one self-contained folder each): `src/memory/models/`
- Retired lineages (graph_v5–v8, operator-v9, v2.1, QA/EMAT eras): `docs/archive/`
