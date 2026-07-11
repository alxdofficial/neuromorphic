"""Memory module (compression + binding for a frozen LM decoder).

Active arms: the published compressor baselines (icae / autocompressor / gisting /
memoryllm / titans), the `slotgraph` (emergent-topology slots), the training-free
`h2o` KV-eviction reference, and the vanilla floor/ceiling. Trained by span-masked
reconstruction (+ the mixed multi-task harness) on a frozen backbone.
"""
from .config import ReprConfig
from .model import ReprLearningModel
from .models.autocompressor import AutoCompressorBaselineEncoder
from .models.icae import ICAEBaselineEncoder
from .models.vanilla import FullContextEncoder, NullEncoder

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "ICAEBaselineEncoder",
    "AutoCompressorBaselineEncoder",
    "NullEncoder",
    "FullContextEncoder",
]
