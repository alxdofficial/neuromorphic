"""Reader (Load) layer — one module per dataset `<name>`.

Every dataset has a matching triplet keyed by the same canonical `<name>`:
  build  → `scripts/data_build/{generate,ingest}/<name>`
  store  → `data/<name>/`
  load   → `src/memory/data/<name>.py`  (this package)
See `DATASETS.md` at the repo root for the full index.

- `common.py` holds the shared `QABatch` / `collate_qa` / `_pack_context`
  contract; every reader imports from it.
- `REGISTRY` maps `<name>` → `make_*_dataloader` (lazy: importing this package
  never pulls `datasets`/heavy deps until a loader is actually built).
- The hot names (`QABatch`, `collate_qa`, `make_mixed_qa_dataloader`) are
  re-exported here for back-compat.
"""
from __future__ import annotations

import importlib
from typing import Callable

# ── back-compat re-exports (names imported directly by train.py / diagnostics) ──
from .common import QABatch, collate_qa, _pack_context
from .mixed import MixedQADataset, make_mixed_qa_dataloader

# name → (submodule, factory-function-name). Kept lazy via _lazy_maker below.
# TRAIN readers (babi/bio/mae/continuation) are GONE — superseded by the sources/ + tasks/ packages
# (SOURCE_REGISTRY × get_task, built by training/data_mix.py). Only EVAL readers + mixed remain.
_MAKERS: dict[str, tuple[str, str]] = {
    "babilong":    ("babilong",     "make_babilong_dataloader"),
    "hotpot":      ("hotpot",       "make_hotpot_dataloader"),
    "musique":     ("musique",      "make_musique_dataloader"),
    "narrativeqa": ("narrativeqa",  "make_narrativeqa_dataloader"),
    "ruler":       ("ruler",        "make_ruler_dataloader"),
    "locomo":      ("locomo",       "make_locomo_dataloader"),
    # ── PHASE-2 (test-eval) benchmark readers ────────────────────────────────────────────────────
    "longmemeval":  ("longmemeval",  "make_longmemeval_dataloader"),   # HEADLINE: multi-session chat memory
    "longbench":    ("longbench",    "make_longbench_dataloader"),     # v1 (21 subtasks) + v2 MCQs
    "infinitebench": ("infinitebench", "make_infinitebench_dataloader"),  # ∞Bench Retrieve.* + En.QA
    "niah":         ("niah",         "make_niah_dataloader"),          # procedural needle-in-haystack
    "mixed":       ("mixed",        "make_mixed_qa_dataloader"),
}


def _lazy_maker(module_name: str, func_name: str) -> Callable:
    """Return a callable that imports `<pkg>.<module_name>` on first call and
    forwards to its `func_name`. Keeps package import cheap (no `datasets` etc.)."""
    def _factory(*args, **kwargs):
        mod = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(mod, func_name)(*args, **kwargs)
    _factory.__name__ = func_name
    _factory.__qualname__ = func_name
    _factory.__doc__ = f"Lazy loader factory → {module_name}.{func_name}"
    return _factory


REGISTRY: dict[str, Callable] = {
    name: _lazy_maker(mod, fn) for name, (mod, fn) in _MAKERS.items()
}

__all__ = [
    "REGISTRY",
    "QABatch", "collate_qa", "_pack_context",
    "MixedQADataset", "make_mixed_qa_dataloader",
]
