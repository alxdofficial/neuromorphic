"""soft_pointer_graph — soft-pointer graph memory (encoder + substrate).

ABANDONED 2026-06-15 (was graph_v6, "free-endpoint" graph). Kept loadable for
reproducing prior results; not in the active suite. Hit the rank-1 read /
membership wall (REAL≈SHUF; pool-then-address). Superseded by the VQ-VAE→graph
+TokenGT model. See memory: project_mae_4k_collapse_result.
"""
from .encoder import SoftPointerGraphEncoder
from .substrate import (
    SoftPointer,
    SoftPointerGraphUpdater,
    SoftPointerGraphGate,
    SoftPointerGraphFactBuilder,
    init_soft_pointer_graph_state,
)

__all__ = [
    "SoftPointerGraphEncoder",
    "SoftPointer",
    "SoftPointerGraphUpdater",
    "SoftPointerGraphGate",
    "SoftPointerGraphFactBuilder",
    "init_soft_pointer_graph_state",
]
