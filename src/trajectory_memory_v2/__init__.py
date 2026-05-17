"""Vocabulary-Trajectory memory architecture (v2).

Replaces the fixed-N mutable-cell-state architecture in
`src/trajectory_memory/` with a sparse-edge-state design:

- Nodes are abstract vocabulary embeddings (learnable throughout training).
- Memory content lives on edges, not nodes.
- Edge buffer is sparse with bounded fan-out (K_max per source).
- Edge updates are EMA-driven (deterministic plasticity).
- Eviction is rule-based with specificity protection.
- Reads are generative autocomplete (walk the graph, no allocation).

Design doc: docs/design_vocabulary_trajectory.md
"""

from src.trajectory_memory_v2.config import TrajMemV2Config

__all__ = ["TrajMemV2Config"]
