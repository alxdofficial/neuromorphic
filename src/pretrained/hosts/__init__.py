"""Host adapters — one per HF model family.

Usage:
    from transformers import AutoModelForCausalLM
    from src.pretrained.hosts import build_host

    hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    host = build_host(hf_model)
    host.layer_list()[14] = MemInjectLayer(...)

Add support for a new family by dropping a `<family>.py` next to `llama.py`,
subclassing `HostAdapter`, and registering it in `_REGISTRY` below.
"""

from __future__ import annotations

import torch.nn as nn

from src.pretrained.hosts.base import HostAdapter
from src.pretrained.hosts.llama import LlamaHost


# Maps HF `config.model_type` → adapter class.
# Llama covers: Llama 3.x / TinyLlama / SmolLM2 (all register as "llama").
# Mistral and Qwen2 also use the same attribute layout; adding them is
# just new entries here once we confirm on a real checkpoint.
_REGISTRY: dict[str, type[HostAdapter]] = {
    "llama": LlamaHost,
    # "mistral": LlamaHost,   # same layout; enable after smoke-verifying
    # "qwen2":   LlamaHost,
}


def build_host(hf_model: nn.Module) -> HostAdapter:
    """Dispatch to the HostAdapter matching `hf_model.config.model_type`."""
    model_type = getattr(hf_model.config, "model_type", None)
    if model_type is None:
        raise ValueError(
            "hf_model.config has no model_type attribute; cannot pick a HostAdapter.")
    try:
        cls = _REGISTRY[model_type]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"No HostAdapter registered for model_type={model_type!r}. "
            f"Known: {{{known}}}. Add an adapter in "
            f"src/pretrained/hosts/<family>.py and register it in "
            f"src/pretrained/hosts/__init__.py._REGISTRY.") from None
    return cls(hf_model)


__all__ = ["HostAdapter", "LlamaHost", "build_host"]
