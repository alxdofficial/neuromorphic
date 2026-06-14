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
    AutoCompressorBaselineEncoder,
    BeaconBaselineEncoder,
    CCMBaselineEncoder,
    ContinuousBaselineEncoder,
    FlatBaselineEncoder,
    FullContextEncoder,
    HLVocabEncoder,
    ICAEBaselineEncoder,
    MemorizingBaselineEncoder,
    NullEncoder,
    RecurrentBaselineEncoder,
    SoftPointerGraphEncoder,
)

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "FlatBaselineEncoder",
    "ContinuousBaselineEncoder",
    "MemorizingBaselineEncoder",
    "RecurrentBaselineEncoder",
    "SoftPointerGraphEncoder",
    "HLVocabEncoder",
    "ICAEBaselineEncoder",
    "CCMBaselineEncoder",
    "BeaconBaselineEncoder",
    "NullEncoder",
    "FullContextEncoder",
]
