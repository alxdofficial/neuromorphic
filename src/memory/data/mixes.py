"""Training-mix spec — the single source of truth for WHICH tasks train and HOW each routes.

Co-located with the adapter ``REGISTRY`` (this package): ``REGISTRY`` says what adapters *exist*;
this says which are composed into a training run and each one's decoder routing
(``model.task_mode``, which selects the ``compute_loss`` path). Consumed by
``src/memory/training/data_mix.py``; imported by the diagnostics that build mixed val sets.

Replaces the old ``scripts/train/train.py`` constants ``MIXED_TASKS_DEFAULT`` / ``MIXED_TASK_MODE``
/ ``MIXED_CONDRECON_BIO_*``. See ``docs/harness_reorg_plan.md`` and ``DATASETS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import REGISTRY   # safe: __init__ does not import this module → no import cycle


@dataclass(frozen=True)
class TaskSpec:
    """One mixed-training task: which reader backs it and how the decoder routes it."""
    adapter: str        # REGISTRY key (the reader that backs this task)
    task_mode: str      # model.task_mode → routes compute_loss ("masked_reconstruction" = MAE infill)
    role: str = "train"


# mix-task name → spec. The mix name MAY differ from the adapter (a task *framing* over an adapter):
# "condrecon_bio" is the conditioned-reconstruction framing over the "bio" reader.
TASK_SPEC: dict[str, TaskSpec] = {
    "mae":           TaskSpec("mae",          "masked_reconstruction"),
    "babi":          TaskSpec("babi",         "babi"),
    "continuation":  TaskSpec("continuation", "continuation"),
    "condrecon_bio": TaskSpec("bio",          "conditioned_reconstruction_bio"),
}

# the default 4-task training mix (mae = compression control; babi = relational binding sanity;
# continuation = gist-LM; condrecon_bio = the realistic key→value binding anchor).
DEFAULT_TRAIN_MIX: tuple[str, ...] = ("mae", "babi", "continuation", "condrecon_bio")

# default uniform memory budget M (slots/edges) for the mixed regimen — the SINGLE source of truth.
# The CLI --mixed-M default and the mixed diagnostics import this so they never drift (raised 32→64
# for the streaming-write regime: binding headroom, forgetting pressure from distractor load).
DEFAULT_MIXED_M: int = 64

# bio conditioned-reconstruction construction constants (fill-to-budget packs key=value lines to
# ~ctx_len; n_pairs = MAX entity pool — the world supports ~32 after the train/val name firewall;
# n_facts = attributes packed per value sentence).
CONDRECON_BIO_N_PAIRS = 24
CONDRECON_BIO_N_FACTS = 3


def task_mode(name: str) -> str:
    """model.task_mode for a mixed-task name (raises KeyError on an unknown task)."""
    return TASK_SPEC[name].task_mode


# {task name → model.task_mode}. The flat form the diagnostics want (drop-in for the old
# scripts/train/train.py ``MIXED_TASK_MODE``).
TASK_MODE: dict[str, str] = {n: s.task_mode for n, s in TASK_SPEC.items()}


# every task must be backed by a real adapter — catch a spec/registry drift at import time.
_missing = {n: s.adapter for n, s in TASK_SPEC.items() if s.adapter not in REGISTRY}
assert not _missing, f"TASK_SPEC adapters absent from REGISTRY: {_missing} (have {sorted(REGISTRY)})"
del _missing
