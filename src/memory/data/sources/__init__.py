"""Source registry — name → Source factory. Sources yield raw items; Tasks shape them.

Lazy so importing the package never pulls heavy deps (datasets/pyarrow) until a source is built.
See ``docs/data_arch_plan.md`` (Layer L1).
"""
from __future__ import annotations

import importlib
from typing import Callable

from .base import Source, CorpusItem, KeyedItem, QAItem

# name → (submodule, class-name)
_SOURCES: dict[str, tuple[str, str]] = {
    "babi": ("babi", "BabiSource"),
    # added incrementally as sources are extracted: bio, fineweb, pile, redpajama, mqar, ...
}


def _lazy(module_name: str, cls_name: str) -> Callable:
    def _factory(*args, **kwargs):
        mod = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(mod, cls_name)(*args, **kwargs)
    _factory.__name__ = cls_name
    return _factory


SOURCE_REGISTRY: dict[str, Callable] = {
    name: _lazy(mod, cls) for name, (mod, cls) in _SOURCES.items()
}

__all__ = ["SOURCE_REGISTRY", "Source", "CorpusItem", "KeyedItem", "QAItem"]
