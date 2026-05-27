"""V2.1 representation learning module.

Learns a compressed internal language for text via classification-style node
selection (N=4096 nodes × D_concept=1024) + continuous typed edges (32 edges
per window, D_edge=128), trained by span-masked reconstruction with a frozen
Llama-3.2-1B decoder.

Design doc: docs/v2.1_repr_learning.md
"""
from .config import ReprConfig
from .model import ReprLearningModel
from .encoder import (
    ContinuousBaselineEncoder,
    FlatBaselineEncoder,
    GraphBaselineEncoder,
    GraphV5BaselineEncoder,
    MemorizingBaselineEncoder,
    NullEncoder,
    PlasticBaselineEncoder,
    RecurrentBaselineEncoder,
    SplatBaselineEncoder,
    V21Encoder,
)

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "V21Encoder",
    "FlatBaselineEncoder",
    "ContinuousBaselineEncoder",
    "MemorizingBaselineEncoder",
    "RecurrentBaselineEncoder",
    "PlasticBaselineEncoder",
    "SplatBaselineEncoder",
    "GraphBaselineEncoder",
    "GraphV5BaselineEncoder",
    "NullEncoder",
]
