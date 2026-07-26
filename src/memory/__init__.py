"""Memory module (compression + binding for a frozen LM decoder).

Public model classes are lazy so importing lightweight utilities such as ``src.memory.eval.results`` does not
initialize PyTorch, Transformers, and every baseline encoder. ``from src.memory import ReprConfig`` retains the
same public API and resolves the underlying class on first access.
"""
from importlib import import_module

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "ICAEBaselineEncoder",
    "AutoCompressorBaselineEncoder",
    "NullEncoder",
    "FullContextEncoder",
]

_LAZY_IMPORTS = {
    "ReprConfig": (".config", "ReprConfig"),
    "ReprLearningModel": (".model", "ReprLearningModel"),
    "ICAEBaselineEncoder": (".models.icae", "ICAEBaselineEncoder"),
    "AutoCompressorBaselineEncoder": (".models.autocompressor", "AutoCompressorBaselineEncoder"),
    "NullEncoder": (".models.vanilla", "NullEncoder"),
    "FullContextEncoder": (".models.vanilla", "FullContextEncoder"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
