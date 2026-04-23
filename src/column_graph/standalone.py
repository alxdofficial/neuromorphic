"""StandaloneLM — wraps ColumnGraphMemory with token embedding and tied unembed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from src.column_graph.column_graph import ColumnGraphMemory, MemoryReadout
from src.column_graph.config import ColumnGraphConfig


@dataclass
class StepOutputs:
    logits_per_tick: list[torch.Tensor]        # T entries of [B, K_horizons, V]
    last_surprise: torch.Tensor                # [B, K_horizons]


class StandaloneLM(nn.Module):
    """Token embedding → ColumnGraphMemory → tied unembed.

    Usage per training step:
        lm.memory.begin_segment(B, device)
        for t in range(T_seq):
            r = lm.step(tokens[:, t])
        # then TBPTT: lm.memory.detach_state() at block boundaries
    """

    def __init__(self, cfg: ColumnGraphConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.D_s)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        self.memory = ColumnGraphMemory(cfg, tied_token_emb=self.token_emb)

    def forward_segment(
        self, tokens: torch.Tensor, tbptt_block: int
    ) -> StepOutputs:
        """Run one segment of shape [B, T_seq], producing per-tick logits
        with TBPTT detach every `tbptt_block` tokens.

        This is the main training loop primitive.
        """
        B, T = tokens.shape
        device = tokens.device
        self.memory.begin_segment(B, device)
        logits_per_tick: list[torch.Tensor] = []
        last: MemoryReadout | None = None
        for t in range(T):
            r = self.memory.step(tokens[:, t])
            logits_per_tick.append(r.logits)
            last = r
            if (t + 1) % tbptt_block == 0 and (t + 1) < T:
                # Detach state between blocks. Emit the logits we've collected
                # up to here as one TBPTT chunk; caller sums losses across chunks.
                # In this simple forward_segment we just detach and continue —
                # the caller should pair this with a per-block loss/backward
                # pattern (see train_phase1.py).
                self.memory.detach_state()
        assert last is not None
        return StepOutputs(logits_per_tick=logits_per_tick, last_surprise=last.surprise_ema)

    def tick(self, token_id: torch.Tensor) -> MemoryReadout:
        """Single-token step. Caller manages begin_segment / detach_state."""
        return self.memory.step(token_id)
