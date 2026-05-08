"""Learning-rate schedules.

`warmup_then_cosine`: linear warmup from 0 → lr_max over `warmup_steps`,
then cosine decay from lr_max → lr_min over the remaining
`total_steps - warmup_steps` steps. Standard for transformer-style training.
"""

from __future__ import annotations

import math


def warmup_then_cosine(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
) -> float:
    """LR multiplier or absolute LR at `step`.

    Args:
        step:         current global step (0-indexed).
        warmup_steps: linear warmup from 0 to lr_max over this many steps.
        total_steps:  total training steps (warmup + decay).
        lr_max:       peak LR.
        lr_min:       floor LR after full decay.

    Returns:
        LR for this step.
    """
    if step < 0:
        return lr_min
    if warmup_steps > 0 and step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return lr_max
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


class WarmupCosineScheduler:
    """Per-param-group warmup + cosine LR scheduler.

    Compatible with `torch.optim.Optimizer` — call `.step()` after each
    optimizer step (NOT after each forward, despite torch's default
    convention; we go simple).

    Each param group can have its own `lr_max` (set via `base_lrs`); the
    schedule shape is shared across groups.
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        lr_min_ratio: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min_ratio = lr_min_ratio
        # Snapshot per-group base LR.
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0
        # Apply step-0 LR.
        self._apply()

    def _apply(self) -> None:
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = warmup_then_cosine(
                self._step,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps,
                lr_max=base,
                lr_min=base * self.lr_min_ratio,
            )

    def step(self) -> None:
        self._step += 1
        self._apply()

    def state_dict(self) -> dict:
        return {
            "step": self._step,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.lr_min_ratio = state["lr_min_ratio"]
        self.base_lrs = state["base_lrs"]
        self._apply()

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def current_lrs(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]
