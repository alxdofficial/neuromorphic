"""Memory module (compression + binding for a frozen LM decoder).

Active arms: the four published compressor baselines (icae / ccm / autocompressor / beacon),
the relational `graph` parser, `biomem` (fast-Hebbian), `slotgraph` (emergent-topology slots),
`vqicae` (icae + VQ-discretized slots), and the vanilla floor/ceiling. Trained by span-masked
reconstruction (+ the mixed multi-task harness) on a frozen backbone.
"""
from .config import ReprConfig
from .model import ReprLearningModel
from .models.autocompressor import AutoCompressorBaselineEncoder
from .models.beacon import BeaconBaselineEncoder
from .models.ccm import CCMBaselineEncoder
from .models.icae import ICAEBaselineEncoder
from .models.vanilla import FullContextEncoder, NullEncoder

__all__ = [
    "ReprConfig",
    "ReprLearningModel",
    "ICAEBaselineEncoder",
    "CCMBaselineEncoder",
    "BeaconBaselineEncoder",
    "AutoCompressorBaselineEncoder",
    "NullEncoder",
    "FullContextEncoder",
]
