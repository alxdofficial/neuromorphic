"""V2.1 representation learning module.

Learns a compressed internal language for text via classification-style node
selection (N=4096 nodes × D_concept=1024) + continuous typed edges (32 edges
per window, D_edge=128), trained by span-masked reconstruction with a frozen
Llama-3.2-1B decoder.

Design doc: docs/v2.1_repr_learning.md
"""
from .config import ReprConfig
from .model import ReprLearningModel
from .models.autocompressor import AutoCompressorBaselineEncoder
from .models.beacon import BeaconBaselineEncoder
from .models.ccm import CCMBaselineEncoder
from .models.hierarchical_learned_vocab import HLVocabEncoder
from .models.icae import ICAEBaselineEncoder
from .models.soft_pointer_graph import SoftPointerGraphEncoder
from .models.vanilla import FullContextEncoder, NullEncoder

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "SoftPointerGraphEncoder",
    "HLVocabEncoder",
    "ICAEBaselineEncoder",
    "CCMBaselineEncoder",
    "BeaconBaselineEncoder",
    "AutoCompressorBaselineEncoder",
    "NullEncoder",
    "FullContextEncoder",
]
