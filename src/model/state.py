"""
StateManager — utility for coordinating state across the full model.

Provides functions to walk the model tree and perform bulk state operations
(detach, reset, save, load) on all StateMixin instances.
"""

import torch
from torch import Tensor
from typing import Dict, Any

from .utils import StateMixin


def _walk_state_mixins(module, prefix=""):
    """Yield (path, mixin) for all StateMixin instances in the module tree."""
    if isinstance(module, StateMixin):
        yield prefix, module
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        yield from _walk_state_mixins(child, child_prefix)


def detach_all(model):
    """Walk model tree, call detach_states() on all StateMixin instances."""
    for _, mixin in _walk_state_mixins(model):
        mixin.detach_states()


def reset_all(model, mask: Tensor):
    """Walk model tree, call reset_states(mask) on all StateMixin instances."""
    for _, mixin in _walk_state_mixins(model):
        mixin.reset_states(mask)


def save_runtime_state(model) -> Dict[str, Any]:
    """Serialize all non-parameter runtime state for checkpointing.

    Keys are stable module paths (e.g. 'blocks.0.layers.1.pm') so
    checkpoints remain valid after unrelated model changes.
    """
    state = {}
    for path, mixin in _walk_state_mixins(model):
        state[path] = mixin.state_dict_runtime()
    return state


def load_runtime_state(model, state: Dict[str, Any]):
    """Load runtime state from checkpoint.

    Supports both path-based keys (new) and legacy index-based keys.
    """
    # Build path -> mixin mapping
    mixins_by_path = {path: mixin for path, mixin in _walk_state_mixins(model)}

    # Try path-based keys first
    matched = 0
    for path, mixin in mixins_by_path.items():
        if path in state:
            mixin.load_state_runtime(state[path])
            matched += 1

    # Fallback: legacy index-based keys (mixin_0_ClassName)
    if matched == 0:
        for i, (path, mixin) in enumerate(_walk_state_mixins(model)):
            legacy_key = f"mixin_{i}_{type(mixin).__name__}"
            if legacy_key in state:
                mixin.load_state_runtime(state[legacy_key])
