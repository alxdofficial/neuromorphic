# Training-harness + diagnostics reorganization plan (2026-07-06)

> **STATUS: COMPLETED.** Complementary to `docs/data_arch_plan.md` (2026-07-07), which further
> splits `src/memory/training/objectives.py` into an `objectives/` package and adds a
> `behavioral_kl` objective — an extension of the harness structure established here, not a change to it.

Goal: one clear, principled structure for the **training harness** and **diagnostics**, matching
the library/executable split the data reorg already established (`src/memory/data/` library vs
`scripts/data_build/` executables). Full fix in one arc, not incremental patches.

## The governing principle (why this shape)

**Importable logic lives in `src/memory/`; executable entrypoints live in `scripts/`.**

The audit proved `scripts/train/train.py` (2485 lines) is *already a de-facto library*: **10
diagnostics import reusable functions from it** — `make_mixed_val_sets`, `to_device`,
`MIXED_TASK_MODE`, and the objective-ladder internals `_grad_cached_objective_step` /
`_infonce_logits_weights` / `_same_answer_valid_mask` — three of them via fragile
`from train import …` sys.path hacks. A script directory should not be an import target. So the
reusable harness moves into the library; `scripts/train/` keeps only a thin CLI entrypoint.

Same logic answers "where is the training mix defined": today the *menu* (`REGISTRY`) is in the
data package but the *order + routing* (`MIXED_TASKS_DEFAULT`, `MIXED_TASK_MODE`) is buried at
`train.py:91`. Co-locate the spec with the adapters.

## Current smells (from the audit)

1. `scripts/train/train.py` — 2485-line monolith, ~7 concerns, **two live trainer loops**
   (`train_mixed_variant` 340 ln + `train_one_variant` 559 ln; `main()` calls both), 91-flag
   `main()` (~900 ln), the objective ladder, eval, checkpointing, data-mix wiring.
2. Training-mix spec (`MIXED_TASKS_DEFAULT`, `MIXED_TASK_MODE`) divorced from `REGISTRY`; and a
   **name mismatch** — the mix uses `condrecon_bio` but the reader is registered as `bio`.
3. `scripts/diagnostics/` — 18-file flat pile spanning ~7 kinds (smoke / battery / probe /
   objective / sweep / eval / audit), model-specific and cross-cutting jumbled; several reach into
   `train.py` internals.

## Already clean (leave alone) — and deliberately out of scope

- **Clean, untouched:** `src/memory/data/` (adapters + REGISTRY), `src/memory/models/<arm>/`,
  `scripts/data_build/` (3-layer), `src/memory/{decoder,selection,common,chat_template}.py`.
- **Out of scope, on purpose:**
  - `src/memory/config.py` — one `ReprConfig`, but fields are already cleanly bannered by concern,
    and splitting `ModelConfig`/`TrainConfig` ripples through `model.py` + every encoder that reads
    cfg. Low payoff, high blast radius. Only the CLI→cfg *mapping* moves; the dataclass stays.
  - `src/memory/model.py` — 1246 ln but the genuine core (frozen-LM wrap, injection hooks,
    `compute_loss`, Set-LLM attention). A separate effort, unrelated to harness/diagnostics.

---

## Target structure

### 1. Mix spec → co-located with the adapters (`src/memory/data/`)

New `src/memory/data/mixes.py` — the single declarative source of "what trains + how it routes":

```python
@dataclass(frozen=True)
class TaskSpec:
    adapter: str       # REGISTRY key (the reader that backs this task)
    task_mode: str     # model.task_mode → routes compute_loss ("masked_reconstruction" = MAE path)
    role: str = "train"

# mix-task name → spec. The mix name may differ from the adapter (a task framing over an adapter):
# "condrecon_bio" is the conditioned-reconstruction framing over the "bio" reader.
TASK_SPEC: dict[str, TaskSpec] = {
    "mae":           TaskSpec("mae",          "masked_reconstruction"),
    "babi":          TaskSpec("babi",         "babi"),
    "continuation":  TaskSpec("continuation", "continuation"),
    "condrecon_bio": TaskSpec("bio",          "conditioned_reconstruction_bio"),
}
DEFAULT_TRAIN_MIX = ("mae", "babi", "continuation", "condrecon_bio")
CONDRECON_BIO_N_PAIRS, CONDRECON_BIO_N_FACTS = 24, 3   # bio task-construction constants
```

- **No risky global rename.** Instead of renaming the `condrecon_bio` mix-label to `bio` (18 sites,
  breaks `--mixed-tasks` CLI compat + telemetry), make the label→adapter link **explicit** via
  `TaskSpec.adapter`. A mix task-name (a framing) and an adapter-name (a reader) are genuinely
  different concepts; this documents the mapping as first-class instead of an implicit branch.
- Consistency assert (in `mixes.py`): every `TaskSpec.adapter` ∈ `REGISTRY`.
- Now "what adapters exist / which train / how each routes" is **one place**, next to the menu.
- This *replaces* `train.py`'s `MIXED_TASKS_DEFAULT` / `MIXED_TASK_MODE` / bio constants; `data_mix.py`
  reads the spec and picks the loader via `REGISTRY[spec.adapter]`.

### 2. Harness → `src/memory/training/` (NEW importable package)

Pure mechanical extraction from `train.py` — no logic changes:

```
src/memory/training/
  __init__.py     # public surface (what diagnostics + the CLI import)
  loops.py        # train_mixed_variant, train_one_variant, probe_bs
  objectives.py   # _infonce_logits_weights, _same_answer_valid_mask, _coding_rate,
                  #   _grad_cached_objective_step   (the CE / InfoNCE / coding-rate / GRPO ladder)
  eval.py         # run_val, run_mixed_val, _continuation_early_loss, materialize_val_set
  checkpoint.py   # save_checkpoint, _ckpt_metadata, _grad_group_norm
  data_mix.py     # make_mixed_train_dataloaders, make_mixed_val_sets (consume mixes.py + cfg)
  utils.py        # lr_at_step, to_device
```

- `data_mix.py` reads the spec from `src/memory/data/mixes.py` and builds loaders via `REGISTRY`.
- `__init__.py` re-exports the names diagnostics currently pull, so their imports become
  `from src.memory.training import make_mixed_val_sets, to_device` (clean library imports).
- No heavy import at `src/memory/__init__.py` — `training` is imported explicitly, not eagerly.

### 3. Entrypoint → thin `scripts/train/`

```
scripts/train/
  train.py   # main(): orchestration only — parse → build model → call training.loops → save
  cli.py     # build_argparser() + args→ReprConfig mapping  (the ~900-line, 91-flag block)
```

`train.py` shrinks from 2485 → ~150 ln; it imports trainers/eval/checkpoint from
`src.memory.training` and the mix from `src.memory.data.mixes`.

### 4. Diagnostics → `scripts/diagnostics/` grouped by SUBJECT/model

```
scripts/diagnostics/
  slotgraph3/  slotgraph3_diag.py  slotgraph3_tokgeom_sweep.py  slotgraph3_validity_battery.py
  slotgraph/   smoke_slotgraph.py  smoke_slotgraph_mpread.py  slotgraph_diag.py  slotgraph_gradflow.py
  biomem/      smoke_biomem.py  biomem_stage0_probe.py
  objective/   objective_debug.py  objective_profile.py
  mixed/       mixed_dashboard.py  mixed_band_gate_eval.py  mixed_data_audit.py
  cohort/      cohort_results.py  debug_sweep_new_models.py  analyze_sentence_lengths.py
```

- **By subject/model** — per-arm folders for `slotgraph3`/`slotgraph`/`biomem`; `objective` and
  `mixed` for the cross-arm campaigns; `cohort` for cross-arm results/sweeps/analyses that don't
  belong to one arm. No singleton folders.
- Repoint every `from scripts.train.train import …` → `from src.memory.training import …` /
  `from src.memory.data.mixes import TASK_SPEC`; kill the fragile bare `from train import …`.

---

## Execution — phased, verification gate after each

1. **Mix spec** — add `src/memory/data/mixes.py`; reconcile `condrecon_bio`→`bio`; consistency
   assert. **Verify:** `import src.memory.data`; `TASK_SPEC` keys == `REGISTRY` keys; data debug
   sweep still loads bio/babi/babilong.
2. **Extract `src/memory/training/`** — move the 7 modules' worth of functions out of `train.py`;
   `train.py` imports them back so it still runs unchanged. **Verify:** `import src.memory.training`;
   `python -m scripts.train.train --help`.
3. **Thin the entrypoint** — split `cli.py` out; `train.py` becomes orchestration-only, importing
   from `training` + `data.mixes`. **Verify:** `--help`, then a **1-step real training smoke** (the
   hot path must still train — this is the one non-trivial gate).
4. **Repoint diagnostics imports** — all 10 files → library imports; fix the sys.path hacks.
   **Verify:** import every diagnostic module.
5. **Restructure diagnostics into by-kind dirs** — `git mv` + update any cross-refs/`__main__`
   path assumptions. **Verify:** run one smoke (`smoke/smoke_slotgraph.py`) + one battery.
6. **Docs** — `HARNESS.md` (or a training/diagnostics section in an existing index): the training
   package map, the mix spec, how to add a task, how to run a diagnostic. Cross-ref from
   `DATASETS.md`. Remove stale references to `scripts/train/train.py` internals.

## Back-compat

User accepts churn → no permanent shims. Update all import sites in the phase that moves each
symbol. Optional one-release guard: a `scripts/train/train.py` that still re-exports the moved
names with a `DeprecationWarning`, removed after one green run.
