# Training harness — the index

How training is organized, mirroring the data layer's split (see `DATASETS.md`):
**importable logic lives in `src/memory/`; executable entrypoints live in `scripts/`.**
Design rationale + history: `docs/history/harness_reorg_plan.md`.

## The three questions this answers

| Question | Answer |
|---|---|
| What **sources** (datasets) exist? | `src/memory/data/sources/` — `SOURCE_REGISTRY`. (`REGISTRY` = eval-only flat readers.) |
| What **task styles** exist? | `src/memory/data/tasks/` — `get_task` / `TASK_STYLES` (reconstruction/qa/continuation/mae/overwrite) |
| Which datasets **train**, and how does each route? | `src/memory/data/mixes.py` — `TASK_SPEC` (`source × task_style × task_mode`) + `DEFAULT_TRAIN_MIX` |
| Where is the **trainer / objective / eval** code? | `src/memory/training/` (library) |
| Where is the **CLI**? | `scripts/train/{train.py, cli.py}` |

**Data is 4 orthogonal layers** (`docs/data_arch_plan.md`, `DATASETS.md`): **Source** (where tokens
come from) × **Task** (what's asked) × **EpisodeSpec** (`schedule.py` — how hard) × **Objective**
(how scored). task ≠ objective — they compose as a matrix.

## `src/memory/training/` — the reusable harness (importable)

```
src/memory/training/
  __init__.py     # public surface (diagnostics + the CLI import from here)
  loops.py        # train_mixed_variant (the mixed benchmark path), probe_bs.
  objectives.py   # the objective ladder: _infonce_logits_weights, _same_answer_valid_mask,
                  #   _coding_rate, _grad_cached_objective_step (CE / InfoNCE / coding-rate / GRPO),
                  #   _behavioral_kl_step (context distillation: KL(teacher=full-ctx ‖ student=memory))
  eval.py         # run_val, run_mixed_val, _continuation_early_loss, CONT_EARLY_TOKENS
  checkpoint.py   # save_checkpoint, _ckpt_metadata, _grad_group_norm
  data_mix.py     # make_mixed_train_dataloaders, make_mixed_val_sets (build loaders from the spec)
  utils.py        # lr_at_step, to_device, materialize_val_set
```

Imported **explicitly** (`from src.memory.training import …`), never eagerly from
`src.memory.__init__`, so its torch/model imports stay off the light `src.memory` import path.

## `src/memory/data/mixes.py` — the training-mix spec

The single source of truth for *what trains and how it routes*:
- `TaskSpec(source, task_style, task_mode, role)` — `source` ∈ `SOURCE_REGISTRY`, `task_style` ∈
  `TASK_STYLES`; `task_mode` sets `model.task_mode` (`"masked_reconstruction"` = MAE infill,
  else generic QA/CE path).
- `TASK_SPEC` — mix-task name → spec. The name may differ from the source (a *framing*):
  `fact_recall` = `reconstruction` over the `bio` source; `reconstruct` = `mae` over `fineweb`;
  `continuation` = `continuation` over `multicorpus` (fineweb+pile+redpajama+code); `doc_qa` = `qa`
  over `qa_multi` (squad+triviaqa+hotpot_train+musique_train+multiwoz). Mix-task names were renamed
  2026-07-08 (`mae`→`reconstruct`, `qa_rc`→`doc_qa`, `condrecon_bio`→`fact_recall`); old names still
  resolve via `TASK_ALIASES`.
- `DEFAULT_TRAIN_MIX = ("reconstruct", "babi", "doc_qa", "continuation", "fact_recall")`. `TASK_MODE` flat dict.
- `training/data_mix.py` composes `SOURCE_REGISTRY[spec.source] × get_task(spec.task_style) ×
  EpisodeSpec` — adding a source/task never edits `data_mix`. Import-time assert: every
  `TaskSpec.source` ∈ `SOURCE_REGISTRY`, `task_style` ∈ `TASK_STYLES`.

## `scripts/train/` — the entrypoint (executable)

- `train.py` — thin `main()`: parse args → build tokenizer + frozen LM → call
  `train_mixed_variant` / `probe_bs` → write summary. ~150 lines.
- `cli.py` — `build_parser()` (the mixed-training flags) + `args_to_config(args, ap) → (cfg, …)`.

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

- **Add a dataset (source)**: `src/memory/data/sources/<name>.py` (yield items) + register in
  `sources/__init__.py::_SOURCES`. Reuse an existing task, or add one in `tasks/<style>.py`.
- **Add a training task (mix entry)**: add a `TaskSpec(source, task_style, task_mode)` in
  `mixes.py::TASK_SPEC` (+ `DEFAULT_TRAIN_MIX` if default). Per-source construction kwargs (only if
  novel) go in `training/data_mix.py::_build_source`; the rest is automatic.
- **Add an objective**: a step fn in `training/objectives.py` + a dispatch branch in `loops.py` +
  an `--objective-mode` choice in `cli.py` (e.g. `behavioral_kl`: `--kl-coef`/`--kl-temp`).
- **Add a CLI flag**: `scripts/train/cli.py` (`build_parser` + map it in `args_to_config`).
- **Add a diagnostic**: drop it in the matching `scripts/diagnostics/<subject>/` folder; use
  `Path(__file__).resolve().parents[3]` for the repo root.

Deliberately **not** restructured (high blast radius, unrelated to the harness): `ReprConfig`
(`config.py`) and the core `model.py` — see `docs/history/harness_reorg_plan.md`.
