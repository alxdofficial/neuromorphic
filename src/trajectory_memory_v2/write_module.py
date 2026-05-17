"""WriteModule — trajectory walk + edge state updates.

Same machinery as ReadModule but with write_mode=True. After each
traversal from current → next, the edge (current, next) gets:
  - allocated if it doesn't exist (possibly evicting another edge)
  - EMA-updated with signature=step_query

The edge update is under @torch.no_grad() (see manifold.py for the
rationale and gradient-flow implications).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.trajectory_memory_v2._shared import EntryProjector
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold
from src.trajectory_memory_v2.walker import TrajectoryWalker, WalkerResult


class WriteModule(nn.Module):
    """Generates J parallel write trajectories. Updates edge state."""

    def __init__(
        self,
        cfg: TrajMemV2Config,
        *,
        entry_proj: EntryProjector | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        # Shared with ReadModule's entry_proj (Hopfield-tied).
        self.entry_proj = entry_proj if entry_proj is not None else EntryProjector(cfg)
        self.walker = TrajectoryWalker(cfg, K=cfg.K_write)

    def forward(
        self,
        current_window_hiddens: Tensor,    # [BS, T, d_lm]
        manifold: VocabularyManifold,
        *,
        hard: bool = True,
    ) -> WalkerResult:
        """Walk a write trajectory over the current window's hiddens.
        Updates manifold's edge state at each hop's (src, dst) edge.

        The trainer is expected to call `manifold.advance_step()` once
        per training step to bump the step counter (used by eviction
        protection's MIN_AGE check).
        """
        return self.walker.forward(
            window_hiddens=current_window_hiddens,
            entry_proj=self.entry_proj,
            manifold=manifold,
            write_mode=True,
            hard_routing=hard,
        )
