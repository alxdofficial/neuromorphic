"""Pretrained-LM + graph_walker integration.

Hooks the `GraphWalkerMemory` into a frozen HuggingFace causal LM (Llama /
TinyLlama / SmolLM2) via a mid-stack `MemInjectLayer` from the existing
`src/pretrained/` scaffolding.

Public entry points:
- `PretrainedGWConfig`                — config (factories for Llama-1B/3B/etc).
- `GraphWalkerPretrainedLM`           — the wrapper nn.Module.
- `phase1_pretrained_step`            — parallel teacher-forced phase-1.
- `run_phase1_ar`                     — autoregressive phase-1 unroll.
- `grpo_step`                         — phase-2 GRPO on routing decisions.
- `autoregressive_rollout`            — stand-alone inference rollout primitive.
- `run_cycle_loop` / `CycleConfig`    — bootstrap → p1-AR → p2 orchestrator.
- `StatsCollector`                    — telemetry (see `src/graph_walker/telemetry.py`).
"""

from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rollout import (
    RolloutOutput,
    autoregressive_rollout,
)
from src.graph_walker.pretrained.train_loop import CycleConfig, run_cycle_loop
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    Phase1Stats,
    phase1_pretrained_step,
)
from src.graph_walker.pretrained.train_phase1_ar import (
    Phase1ARBatch,
    Phase1ARStats,
    phase1_ar_pretrained_step,
)
from src.graph_walker.pretrained.train_phase2 import GRPOStats, grpo_step

__all__ = [
    "PretrainedGWConfig",
    "GraphWalkerPretrainedLM",
    # Phase 1 parallel
    "Phase1Batch",
    "Phase1Stats",
    "phase1_pretrained_step",
    # Phase 1 AR
    "Phase1ARBatch",
    "Phase1ARStats",
    "phase1_ar_pretrained_step",
    # Phase 2 GRPO
    "GRPOStats",
    "grpo_step",
    "autoregressive_rollout",
    "RolloutOutput",
    # Cycle orchestrator
    "CycleConfig",
    "run_cycle_loop",
]
