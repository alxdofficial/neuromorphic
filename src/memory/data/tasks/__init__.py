"""Task registry — style → Task. Tasks shape source items into the on-the-wire sample dict.

See ``docs/data_arch_plan.md`` (Layer L2).
"""
from __future__ import annotations

import importlib
from typing import Callable

from .base import Task, TaskDataset, make_task_dataloader

# style → (submodule, class-name)
_TASKS: dict[str, tuple[str, str]] = {
    "qa": ("qa", "QATask"),
    "reconstruction": ("reconstruction", "ReconstructionTask"),
    "continuation": ("continuation", "ContinuationTask"),
    "mae": ("mae", "MaeTask"),
    "overwrite": ("overwrite", "OverwriteTask"),
    # added incrementally: multisession
}


def _lazy(module_name: str, cls_name: str) -> Task:
    mod = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(mod, cls_name)()


def get_task(style: str) -> Task:
    if style not in _TASKS:
        raise KeyError(f"unknown task style {style!r} (have {sorted(_TASKS)})")
    return _lazy(*_TASKS[style])


TASK_STYLES = tuple(_TASKS)

__all__ = ["Task", "TaskDataset", "make_task_dataloader", "get_task", "TASK_STYLES"]
