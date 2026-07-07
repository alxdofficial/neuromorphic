# `generate/ruler_overwrite/` — RULER-overwrite (forced-forgetting)

**Runtime-procedural — nothing to build here.** The overwrite source synthesizes random variables
on the fly per draw, so there is no offline generation step and nothing is stored under
`data/ruler_overwrite/`.

Recipe (from the survey): fork RULER `variable_tracking` by **reassigning the SAME key** a new
value — a correct memory must return the LATEST binding and forget the stale one.

- **Source (load):** `src/memory/data/sources/ruler_overwrite.py` (`RulerOverwriteSource`,
  `kind="keyed"`) — supplies distinct random variables (random key + random value each).
- **Task:** the dedicated `overwrite` task (`src/memory/data/tasks/overwrite.py`) composes the
  reassignment sequence `KEY = v1 … distractors … KEY = v2` and queries `KEY` for `v2`
  (EM-scorable; loss on the `v2` value span only).
- Random alnum keys/values (shared `rand_alnum` helper with `mqar`), non-parametric so the answer
  can't be language-guessed.

See `DATASETS.md` and `docs/data_arch_plan.md` (Layer L1/L2).
