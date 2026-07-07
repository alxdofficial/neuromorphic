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
from .sources import SOURCE_REGISTRY
from .tasks import TASK_STYLES


@dataclass(frozen=True)
class TaskSpec:
    """One mixed-training task: the data (Source × Task) that backs it and the decoder routing.

    ``source``/``task_style`` are the 4-layer keys (``SOURCE_REGISTRY`` × ``TASK_STYLES``) that
    ``data_mix`` composes into an ``EpisodeSpec``. ``adapter`` is the legacy flat-reader ``REGISTRY``
    key kept for back-compat (diagnostics). ``task_mode`` is the model's ``compute_loss`` routing."""
    adapter: str        # legacy flat-reader REGISTRY key (back-compat)
    task_mode: str      # model.task_mode → routes compute_loss ("masked_reconstruction" = MAE infill)
    source: str = ""    # SOURCE_REGISTRY key (where items come from)
    task_style: str = ""  # TASK_REGISTRY key (how they're presented/asked)
    role: str = "train"


# mix-task name → spec. The mix name MAY differ from source/adapter (a task *framing* over a source):
# "condrecon_bio" = the reconstruction task over the "bio" source; "mae"/"continuation" both over "fineweb".
TASK_SPEC: dict[str, TaskSpec] = {
    "mae":           TaskSpec("mae",          "masked_reconstruction",          source="fineweb", task_style="mae"),
    "babi":          TaskSpec("babi",         "babi",                           source="babi",    task_style="qa"),
    "continuation":  TaskSpec("continuation", "continuation",                   source="fineweb", task_style="continuation"),
    "condrecon_bio": TaskSpec("bio",          "conditioned_reconstruction_bio", source="bio",     task_style="reconstruction"),
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


# every task must be backed by a real adapter AND a real (source, task) pair — catch drift at import.
_bad_adapter = {n: s.adapter for n, s in TASK_SPEC.items() if s.adapter not in REGISTRY}
assert not _bad_adapter, f"TASK_SPEC adapters absent from REGISTRY: {_bad_adapter} (have {sorted(REGISTRY)})"
_bad_src = {n: s.source for n, s in TASK_SPEC.items() if s.source not in SOURCE_REGISTRY}
assert not _bad_src, f"TASK_SPEC sources absent from SOURCE_REGISTRY: {_bad_src} (have {sorted(SOURCE_REGISTRY)})"
_bad_task = {n: s.task_style for n, s in TASK_SPEC.items() if s.task_style not in TASK_STYLES}
assert not _bad_task, f"TASK_SPEC task_styles absent from TASK_REGISTRY: {_bad_task} (have {sorted(TASK_STYLES)})"
del _bad_adapter, _bad_src, _bad_task
