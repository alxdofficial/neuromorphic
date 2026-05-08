"""Training scaffolding for trajectory-memory.

- `phase1.py` — TF NTP trainer (Wave 1 long-doc, Wave 2 long-chat).
- `phase2.py` — GRPO trainer (Wave 3 verifiable-reward, Wave 4 long-session).
- `loaders.py` — parquet readers + collators for each wave's format.
- `rewards.py` — exact_match + BERT cosine + verifiable rewards
                 (no LLM-as-judge per project policy).
"""

from src.trajectory_memory.training.loaders import (
    LongDocDataset,
    TurnPairDataset,
    PromptResponseDataset,
)

__all__ = [
    "LongDocDataset",
    "TurnPairDataset",
    "PromptResponseDataset",
]
