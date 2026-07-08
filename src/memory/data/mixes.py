"""Training-mix spec — the single source of truth for WHICH tasks train and HOW each routes.

Each mixed task is backed by a (Source × Task) pair from the 4-layer data packages
(``SOURCE_REGISTRY`` × ``TASK_STYLES``) plus a decoder routing (``model.task_mode``, which selects
the ``compute_loss`` path). Consumed by ``src/memory/training/data_mix.py``; imported by the
diagnostics that build mixed val sets.

Replaces the old ``scripts/train/train.py`` constants ``MIXED_TASKS_DEFAULT`` / ``MIXED_TASK_MODE``
/ ``MIXED_CONDRECON_BIO_*``. See ``docs/data_arch_plan.md`` and ``DATASETS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .sources import SOURCE_REGISTRY
from .tasks import TASK_STYLES


@dataclass(frozen=True)
class TaskSpec:
    """One mixed-training task: the data (Source × Task) that backs it and the decoder routing.

    ``source``/``task_style`` are the 4-layer keys (``SOURCE_REGISTRY`` × ``TASK_STYLES``) that
    ``data_mix`` composes into an ``EpisodeSpec``. ``task_mode`` is the model's ``compute_loss`` routing."""
    source: str         # SOURCE_REGISTRY key (where items come from)
    task_style: str     # TASK_STYLES key (how they're presented/asked)
    task_mode: str      # model.task_mode → routes compute_loss ("masked_reconstruction" = MAE infill)
    role: str = "train"
    vary_lag: bool = False  # sample query_lag ∈ {early, recent, any} per episode (retention-lag robustness).
                            # NB: n_queries is now a per-SOURCE property (Source.pack_n_queries) — item size
                            # drives how many to ask, so it can't be one constant across the qa_multi union.


# mix-task name → spec. Names are the human-facing keys (renamed 2026-07-08 for clarity); the SAME task
# can be reached by its old name via TASK_ALIASES. The mix name differs from the source: "fact_recall" =
# the reconstruction task over "bio"; "reconstruct" = mae over "fineweb"; "doc_qa" = qa over the
# "qa_multi" union; "continuation" = multi-horizon over "multicorpus".
TASK_SPEC: dict[str, TaskSpec] = {
    "reconstruct":   TaskSpec("fineweb",     "mae",            "masked_reconstruction"),          # storage/fidelity (was "mae")
    "babi":          TaskSpec("babi",        "qa",             "babi",                            vary_lag=True),  # relational binding; multi-segment renamed
    "doc_qa":        TaskSpec("qa_multi",    "qa",             "qa",                              vary_lag=True),  # real RC QA (was "qa_rc"): squad+triviaqa+hotpot+musique+multiwoz
    "continuation":  TaskSpec("multicorpus", "continuation",   "continuation"),                   # gist-LM, multi-horizon; fineweb+pile+redpajama+code
    "fact_recall":   TaskSpec("bio",         "reconstruction", "conditioned_reconstruction_bio", vary_lag=True),  # key→value binding anchor (was "condrecon_bio")
}

# back-compat: old mix-task names → current, so existing scripts/checkpoints/diagnostics keep working.
TASK_ALIASES: dict[str, str] = {"mae": "reconstruct", "qa_rc": "doc_qa", "condrecon_bio": "fact_recall"}


def resolve_task(name: str) -> str:
    """Map an old mix-task alias to its current name (identity for current names)."""
    return TASK_ALIASES.get(name, name)


# the default 5-task training mix: reconstruct = compression/fidelity control; babi = relational binding;
# doc_qa = real multi-source RC QA; continuation = gist-LM (multi-horizon); fact_recall = key→value anchor.
DEFAULT_TRAIN_MIX: tuple[str, ...] = ("reconstruct", "babi", "doc_qa", "continuation", "fact_recall")

# default uniform memory budget M (slots/edges) for the mixed regimen — the SINGLE source of truth.
# The CLI --mixed-M default and the mixed diagnostics import this so they never drift. 32→64→96: 96
# deliberately keeps CAPACITY off the table (96 slots ≫ ~30 packed bindings) so the sweep measures
# ADDRESSING/structure, not raw bits; 96×d_llama floats is the persistent-state budget per arm.
DEFAULT_MIXED_M: int = 96

# bio conditioned-reconstruction construction constants (fill-to-budget packs key=value lines to
# ~ctx_len; n_pairs = entity pool sampled per episode — the world has ≈410 entities so this is a
# diversity-safe fill knob, raised 24→40 to fill ctx=2048; n_facts = attributes per value sentence).
# NB: data_mix._build_loader also reuses N_PAIRS as the generic EpisodeSpec.n_inputs for EVERY task
# (harmless — qa treats it as a distractor-pool floor; continuation/mae ignore n_inputs).
CONDRECON_BIO_N_PAIRS = 40
CONDRECON_BIO_N_FACTS = 3


def task_mode(name: str) -> str:
    """model.task_mode for a mixed-task name (accepts old aliases; raises KeyError on unknown)."""
    return TASK_SPEC[resolve_task(name)].task_mode


# {task name → model.task_mode} incl. old aliases. Flat form the diagnostics want (drop-in for the old
# scripts/train/train.py ``MIXED_TASK_MODE``).
TASK_MODE: dict[str, str] = {**{n: s.task_mode for n, s in TASK_SPEC.items()},
                             **{old: TASK_SPEC[new].task_mode for old, new in TASK_ALIASES.items()}}


# every task must be backed by a real (source, task) pair — catch drift at import.
_bad_src = {n: s.source for n, s in TASK_SPEC.items() if s.source not in SOURCE_REGISTRY}
assert not _bad_src, f"TASK_SPEC sources absent from SOURCE_REGISTRY: {_bad_src} (have {sorted(SOURCE_REGISTRY)})"
_bad_task = {n: s.task_style for n, s in TASK_SPEC.items() if s.task_style not in TASK_STYLES}
assert not _bad_task, f"TASK_SPEC task_styles absent from TASK_STYLES: {_bad_task} (have {sorted(TASK_STYLES)})"
del _bad_src, _bad_task
