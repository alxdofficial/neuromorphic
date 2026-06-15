"""graph — VQ-codebook graph memory + TokenGT controller + custom inject reader.

The current primary model (2026-06-15), superseding the abandoned hlvocab/
soft_pointer_graph. Edge endpoints are discrete VQ codes (distinct addresses);
a TokenGT writer cross-attends the observation + self-attends the graph; a custom
reader cross-attends the graph + causal-self-attends decode positions and injects
(RMS-matched, gated) into the frozen LLM. Design: docs/graph_model.md.
"""
from .substrate import GraphConfig, GraphWriter, GraphReader

__all__ = ["GraphConfig", "GraphWriter", "GraphReader"]
