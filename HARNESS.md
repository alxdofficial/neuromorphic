# Training harness — the index

How training is organized, mirroring the data layer's split (see `DATASETS.md`):
**importable logic lives in `src/memory/`; executable entrypoints live in `scripts/`.**
Design rationale + history: `docs/harness_reorg_plan.md`.

## The three questions this answers

| Question | Answer |
|---|---|
| What dataset **adapters** exist? | `src/memory/data/__init__.py` — `REGISTRY` (name → loader) |
| Which datasets **train**, and how does each route? | `src/memory/data/mixes.py` — `TASK_SPEC` + `DEFAULT_TRAIN_MIX` |
| Where is the **trainer / objective / eval** code? | `src/memory/training/` (library) |
| Where is the **CLI**? | `scripts/train/{train.py, cli.py}` |

## `src/memory/training/` — the reusable harness (importable)

```
src/memory/training/
  __init__.py     # public surface (diagnostics + the CLI import from here)
  loops.py        # train_mixed_variant (the mixed benchmark path), train_one_variant
                  #   (single-task path), probe_bs. main() calls both.
  objectives.py   # the objective ladder: _infonce_logits_weights, _same_answer_valid_mask,
                  #   _coding_rate, _grad_cached_objective_step  (CE / InfoNCE / coding-rate / GRPO)
  eval.py         # run_val, run_mixed_val, _continuation_early_loss, CONT_EARLY_TOKENS
  checkpoint.py   # save_checkpoint, _ckpt_metadata, _grad_group_norm
  data_mix.py     # make_mixed_train_dataloaders, make_mixed_val_sets (build loaders from the spec)
  utils.py        # lr_at_step, to_device, materialize_val_set
```

Imported **explicitly** (`from src.memory.training import …`), never eagerly from
`src.memory.__init__`, so its torch/model imports stay off the light `src.memory` import path.

## `src/memory/data/mixes.py` — the training-mix spec

The single source of truth for *what trains and how it routes*, co-located with the adapter
`REGISTRY`:

- `TaskSpec(adapter, task_mode, role)` — `adapter` is the `REGISTRY` key backing the task;
  `task_mode` sets `model.task_mode` (routes `compute_loss`; `"masked_reconstruction"` = the MAE
  infill path, everything else = the generic QA/CE path); `role` ∈ {`train`, `eval`}.
- `TASK_SPEC` — mix-task name → spec. A mix name may differ from its adapter (a *framing* over an
  adapter): `condrecon_bio` is the conditioned-reconstruction framing over the `bio` reader.
- `DEFAULT_TRAIN_MIX = ("mae", "babi", "continuation", "condrecon_bio")`.
- `TASK_MODE` — flat `{name → task_mode}` convenience dict. `CONDRECON_BIO_N_PAIRS/N_FACTS` — bio
  construction constants.
- Import-time assert: every `TaskSpec.adapter` ∈ `REGISTRY` (catches spec/registry drift).

## `scripts/train/` — the entrypoint (executable)

- `train.py` — thin `main()`: parse args → build tokenizer + frozen LM → call
  `train_mixed_variant` / `train_one_variant` / `probe_bs` → write summary. ~150 lines.
- `cli.py` — `build_parser()` (all 91 flags) + `args_to_config(args, ap) → (cfg, …)`.

Run: `python -m scripts.train.train --task mixed --variants slotgraph3_baseline …`
(`--backbone HuggingFaceTB/SmolLM2-135M` for the param-matched 135M setup).

## `scripts/diagnostics/` — grouped by subject

```
scripts/diagnostics/
  slotgraph3/  slotgraph/  biomem/     # per-arm smokes / probes / batteries / sweeps
  objective/                           # objective-ladder probes (objective_debug, objective_profile)
  mixed/                               # the mixed-campaign dashboard / band-gate eval / data audit
  cohort/                              # cross-arm results, new-arm sweeps, dataset analyses
```

Run as scripts (`python scripts/diagnostics/<subject>/<file>.py`); they self-insert the repo root
on `sys.path` and import the harness from `src.memory.training` + the spec from
`src.memory.data.mixes`.

## How to…

- **Add a training task**: register the reader in `src/memory/data/__init__.py::REGISTRY`, add a
  `TaskSpec` in `mixes.py` (and to `DEFAULT_TRAIN_MIX` if it should train by default), and add its
  loader-construction branch in `training/data_mix.py` (the makers have heterogeneous signatures).
- **Add a CLI flag**: `scripts/train/cli.py` (`build_parser` + map it in `args_to_config`).
- **Add a diagnostic**: drop it in the matching `scripts/diagnostics/<subject>/` folder; use
  `Path(__file__).resolve().parents[3]` for the repo root.

Deliberately **not** restructured (high blast radius, unrelated to the harness): `ReprConfig`
(`config.py`) and the core `model.py` — see `docs/harness_reorg_plan.md`.
