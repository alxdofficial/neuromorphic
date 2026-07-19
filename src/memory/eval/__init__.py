"""Evaluation utilities for the test-eval phase (Phase 2).

Deterministic, policy-compliant scoring of free-form generated answers against
benchmark gold answers — NO LLM-as-judge (see project policy). The first
inhabitant is the LongMemEval scorer; baseline runners import from here so every
panel (matched-decoder + native-scale) is scored by the same code.
"""

from .longmemeval_score import (
    LongMemEvalScorer,
    score_longmemeval,
    normalize_answer,
)
from .memoryagentbench_score import (
    MemoryAgentBenchScorer,
    score_memoryagentbench,
)

__all__ = ["LongMemEvalScorer", "score_longmemeval", "normalize_answer",
           "MemoryAgentBenchScorer", "score_memoryagentbench"]
