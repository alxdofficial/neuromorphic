"""ReadModule — non-mutating trajectory walk for memory retrieval.

Thin wrapper around TrajectoryWalker with write_mode=False. The output
trajectory's embeddings get aggregated and injected into Llama via
mem_inject (handled by IntegratedLM).

In Wave 1, the caller provides question hiddens (from a zero-memory
forward over the question text).
In streaming mode (Wave 2+), the caller provides prev-window hiddens.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.trajectory_memory_v2._shared import EntryProjector
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold
from src.trajectory_memory_v2.walker import TrajectoryWalker, WalkerResult


class ReadModule(nn.Module):
    """Generates J parallel read trajectories. Does not modify memory."""

    def __init__(
        self,
        cfg: TrajMemV2Config,
        *,
        entry_proj: EntryProjector | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        # Shared entry projector with the paired WriteModule (Hopfield-tied).
        # If not provided, instantiates its own (standalone testing).
        self.entry_proj = entry_proj if entry_proj is not None else EntryProjector(cfg)
        self.walker = TrajectoryWalker(cfg, K=cfg.K_read)

    def forward(
        self,
        prev_window_hiddens: Tensor,    # [BS, T, d_lm] — the conditioning
        manifold: VocabularyManifold,
        *,
        window_mask: "Tensor | None" = None,  # [BS, T] bool — True=real
        hard: bool = True,
    ) -> WalkerResult:
        """Walk a read trajectory over `prev_window_hiddens`.

        For Wave 1: `prev_window_hiddens` is the question's zero-memory
        Llama hiddens.
        For streaming: `prev_window_hiddens` is the actual previous
        window's hiddens.

        Does NOT modify the manifold's edge state (write_mode=False).
        """
        return self.walker.forward(
            window_hiddens=prev_window_hiddens,
            entry_proj=self.entry_proj,
            manifold=manifold,
            window_mask=window_mask,
            write_mode=False,
            hard_routing=hard,
        )
