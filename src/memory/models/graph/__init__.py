"""graph — relational-parser graph memory over a learnable node bank + inject reader.

The current model (2026-06-16), superseding the abandoned hlvocab/soft_pointer_graph
AND the earlier VQ-codebook graph. A learnable node bank is the vocabulary; a TokenGT
parser selects edges by pointing into the bank; a custom reader binds each edge and
injects (RMS-matched, gated) into the frozen LLM. Design: docs/graph_model.md.
"""
from .substrate import GraphConfig, GraphParser, GraphReader
from .encoder import GraphEncoder

__all__ = ["GraphConfig", "GraphParser", "GraphReader", "GraphEncoder"]
