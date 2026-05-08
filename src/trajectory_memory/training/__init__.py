"""Training scaffolding for trajectory-memory.

- `phase1.py`     — `Phase1Trainer` for Wave 1 (long-doc) + Wave 2 (long-chat) TF NTP.
- `phase2.py`     — `Phase2Trainer` for Wave 3 (verifiable-reward) + Wave 4 (long-session) GRPO.
- `loaders.py`    — parquet readers + collators for each wave's format.
- `rewards.py`    — exact_match + BERT cosine + verifiable rewards.
- `lr_schedule.py` — `WarmupCosineScheduler` for warmup + cosine decay.
- `checkpoint.py` — `save_checkpoint` / `load_checkpoint` / RNG capture.
"""

from src.trajectory_memory.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from src.trajectory_memory.training.loaders import (
    LongDocDataset,
    PromptResponseDataset,
    TurnPairDataset,
)
from src.trajectory_memory.training.lr_schedule import (
    WarmupCosineScheduler,
    warmup_then_cosine,
)
from src.trajectory_memory.training.optim import build_optimizer
from src.trajectory_memory.training.phase1 import Phase1Trainer
from src.trajectory_memory.training.phase2 import (
    Phase2Trainer,
    compute_grpo_advantages,
)

__all__ = [
    "LongDocDataset",
    "TurnPairDataset",
    "PromptResponseDataset",
    "Phase1Trainer",
    "Phase2Trainer",
    "compute_grpo_advantages",
    "WarmupCosineScheduler",
    "warmup_then_cosine",
    "save_checkpoint",
    "load_checkpoint",
    "capture_rng_state",
    "restore_rng_state",
    "build_optimizer",
]
