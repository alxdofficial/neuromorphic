"""Pretrained-LM + graph_walker integration.

Hooks the `GraphWalkerMemory` into a frozen HuggingFace causal LM (Llama /
TinyLlama / SmolLM2) via a mid-stack `MemInjectLayer` from the existing
`src/pretrained/` scaffolding.

Public entry points:
- `PretrainedGWConfig`                — config (factories for Llama-1B/3B/etc).
- `GraphWalkerPretrainedLM`           — the wrapper nn.Module.
- `phase1_pretrained_step`            — parallel teacher-forced phase-1.
- `grpo_step`                         — phase-2 GRPO on routing decisions.
- `autoregressive_rollout`            — stand-alone inference rollout primitive.
- `StatsCollector`                    — telemetry (see `src/graph_walker/telemetry.py`).

The AR-unrolled-teacher-forced phase-1 path (formerly Wave 3) was retired
in scope-B cleanup (2026-05-06). Memory exercise now happens through
phase-2 GRPO on chat data (Waves 3+4), where the LM samples its own
tokens and the walker is the only continuity carrier across the gen
window.
"""

from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rollout import (
    RolloutOutput,
    autoregressive_rollout,
)
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    Phase1Stats,
    phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import GRPOStats, grpo_step

__all__ = [
    "PretrainedGWConfig",
    "GraphWalkerPretrainedLM",
    # Phase 1 parallel
    "Phase1Batch",
    "Phase1Stats",
    "phase1_pretrained_step",
    # Phase 2 GRPO
    "GRPOStats",
    "grpo_step",
    "autoregressive_rollout",
    "RolloutOutput",
]
