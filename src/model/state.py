"""
StateManager — utility for coordinating state across the full model.

Provides functions to walk the model tree and perform bulk state operations
(detach, reset, save, load) on all StateMixin instances.
"""

import torch
from torch import Tensor
from typing import Dict, Any

from .utils import StateMixin


def _walk_state_mixins(module):
    """Yield all StateMixin instances in the module tree."""
    if isinstance(module, StateMixin):
        yield module
    for child in module.children():
        yield from _walk_state_mixins(child)


def detach_all(model):
    """Walk model tree, call detach_states() on all StateMixin instances."""
    for mixin in _walk_state_mixins(model):
        mixin.detach_states()


def reset_all(model, mask: Tensor):
    """Walk model tree, call reset_states(mask) on all StateMixin instances."""
    for mixin in _walk_state_mixins(model):
        mixin.reset_states(mask)


def save_runtime_state(model) -> Dict[str, Any]:
    """Serialize all non-parameter runtime state for checkpointing."""
    state = {}
    for i, mixin in enumerate(_walk_state_mixins(model)):
        key = f"mixin_{i}_{type(mixin).__name__}"
        state[key] = mixin.state_dict_runtime()
    return state


def load_runtime_state(model, state: Dict[str, Any]):
    """Load runtime state from checkpoint."""
    for i, mixin in enumerate(_walk_state_mixins(model)):
        key = f"mixin_{i}_{type(mixin).__name__}"
        if key in state:
            mixin.load_state_runtime(state[key])
