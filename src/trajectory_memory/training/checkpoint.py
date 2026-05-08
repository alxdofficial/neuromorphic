"""Checkpoint save/load.

A checkpoint captures everything needed to resume training:
- model state_dict
- optimizer state_dict
- scheduler state_dict (optional)
- step counter
- RNG state (optional, for determinism on resume)
- arbitrary extra metadata (config, loss history, etc.)

Saving uses `torch.save`. For multi-GPU, save only on rank 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    scheduler: Any = None,
    rng_state: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path:      output path (parent dirs created if needed).
        model:     model to snapshot.
        optimizer: optimizer to snapshot.
        step:      global step counter.
        scheduler: optional scheduler with state_dict() / load_state_dict().
        rng_state: optional dict of RNG states (torch + python random) —
                   use `capture_rng_state()` to build.
        extra:     arbitrary metadata to store under "extra" key.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if rng_state is not None:
        state["rng_state"] = rng_state
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> dict:
    """Load a checkpoint and (optionally) restore model/optimizer/scheduler.

    Args:
        path:         input path.
        model:        if provided, `model.load_state_dict(state["model_state_dict"])`.
        optimizer:    if provided, restore optimizer state.
        scheduler:    if provided, restore scheduler state.
        map_location: torch.load map_location.
        strict:       passed to `model.load_state_dict`.

    Returns:
        the full loaded state dict (so caller can read step / extra).
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def capture_rng_state() -> dict:
    """Capture torch + Python random RNG state for deterministic resume."""
    import random
    state: dict = {
        "torch_cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNG state from `capture_rng_state()`."""
    import random
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
