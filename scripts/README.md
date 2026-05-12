# scripts/

Entry-point scripts for the trajectory-memory project. Grouped by purpose
so devs can find things by what they do rather than scanning a flat list.

All scripts are designed to run from the repo root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python scripts/<subdir>/<script>.py [args]
```

## Layout

| Subdir            | What's inside                                                                   |
| ----------------- | ------------------------------------------------------------------------------- |
| `training/`       | Wave 1–4 training entry points (`train_wave{1,2,3,4}.py`)                       |
| `data/`           | Dataset download, preprocessing, train/val split, synthetic data generation    |
| `bench/`          | Benchmarking + perf experiments. `_bench_common.py` holds shared helpers       |
| `diagnostics/`    | Training-health checks + manifold inspection + interpretability viz. See its own README |

## Quick reference

### Training
```bash
PYTHONPATH=. python scripts/training/train_wave1.py [...]
PYTHONPATH=. python scripts/training/train_wave2.py [...]
```

### Data
```bash
PYTHONPATH=. python scripts/data/download_data.py
PYTHONPATH=. python scripts/data/preprocess_longdoc.py
PYTHONPATH=. python scripts/data/synthesize_needle.py
PYTHONPATH=. python scripts/data/split_train_val.py
```

### Benchmarks
```bash
PYTHONPATH=. python scripts/bench/bench_trajmem.py     # default trajectory-memory bench
PYTHONPATH=. python scripts/bench/bench_compare.py     # cross-config comparison
PYTHONPATH=. python scripts/bench/deep_profile_trajmem.py
PYTHONPATH=. python scripts/bench/experiment_memory_cost.py
PYTHONPATH=. python scripts/bench/experiment_lm_context.py
PYTHONPATH=. python scripts/bench/experiment_compile_dynamic.py
```

### Diagnostics
```bash
PYTHONPATH=. python scripts/diagnostics/diagnose.py           # training-health check
PYTHONPATH=. python scripts/diagnostics/viz_manifold.py       # Tier 1 manifold viz
# Tier 2-4 are stubs — see scripts/diagnostics/README.md
```

## Conventions

- **Scripts in the same subdir can share code via relative imports.**
  `scripts/bench/_bench_common.py` is imported by other `bench/*` scripts
  via a `sys.path.insert(0, str(Path(__file__).parent))` trick. Same
  pattern works for any subdir.
- **All scripts work from the repo root** with `PYTHONPATH=.` prepended.
  Don't `cd scripts/<subdir>` first; output paths and configs assume
  cwd=repo-root.
- **Diagnostics are CPU-only by default** and safe to run during active
  training. GPU-using scripts (Tier 2+ viz, benches) say so prominently
  in their docstrings.

## Where things live (cross-reference)

- Architecture spec: [`docs/plan_trajectory_memory.md`](../docs/plan_trajectory_memory.md)
- Eval plan: [`docs/eval_plan.md`](../docs/eval_plan.md)
- Research backlog (post-Wave-1+2 work): [`docs/research_backlog.md`](../docs/research_backlog.md)
- Diagnostics tier roadmap: [`scripts/diagnostics/README.md`](diagnostics/README.md)
- Perf analysis: [`docs/profile_analysis.md`](../docs/profile_analysis.md)
