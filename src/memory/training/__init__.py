"""Reusable training harness (extracted from ``scripts/train/train.py``).

Importable library surface for the trainer loops, objectives, eval, checkpointing, and mixed
data wiring. ``scripts/train/`` keeps only the thin CLI entrypoint; diagnostics import from here.
Imported explicitly (NOT eagerly from ``src.memory.__init__``) so this package's torch/model
imports stay off the light-weight ``src.memory`` import path.
"""
from __future__ import annotations

from .utils import lr_at_step, to_device, materialize_val_set
from .eval import run_val, run_mixed_val, _continuation_early_loss, CONT_EARLY_TOKENS
from .data_mix import make_mixed_train_dataloaders, make_mixed_val_sets
from .checkpoint import save_checkpoint, _ckpt_metadata, _grad_group_norm
from .objectives import (
    _infonce_logits_weights, _same_answer_valid_mask, _coding_rate, _grad_cached_objective_step,
)
from .loops import train_one_variant, train_mixed_variant, probe_bs

__all__ = [
    "lr_at_step", "to_device", "materialize_val_set",
    "run_val", "run_mixed_val", "_continuation_early_loss", "CONT_EARLY_TOKENS",
    "make_mixed_train_dataloaders", "make_mixed_val_sets",
    "save_checkpoint", "_ckpt_metadata", "_grad_group_norm",
    "_infonce_logits_weights", "_same_answer_valid_mask", "_coding_rate",
    "_grad_cached_objective_step",
    "train_one_variant", "train_mixed_variant", "probe_bs",
]
