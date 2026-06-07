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
    BeaconBaselineEncoder,
    CCMBaselineEncoder,
    ContinuousBaselineEncoder,
    FlatBaselineEncoder,
    FullContextEncoder,
    GraphV6BaselineEncoder,
    GraphV7BaselineEncoder,
    ICAEBaselineEncoder,
    MemorizingBaselineEncoder,
    NullEncoder,
    RecurrentBaselineEncoder,
)

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "FlatBaselineEncoder",
    "ContinuousBaselineEncoder",
    "MemorizingBaselineEncoder",
    "RecurrentBaselineEncoder",
    "GraphV6BaselineEncoder",
    "ICAEBaselineEncoder",
    "CCMBaselineEncoder",
    "BeaconBaselineEncoder",
    "NullEncoder",
    "FullContextEncoder",
]
