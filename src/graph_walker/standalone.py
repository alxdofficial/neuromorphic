"""StandaloneLM wrapper — token_emb + GraphWalkerMemory + tied unembed."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.graph_walker import GraphWalkerMemory, WalkerReadout


class StandaloneLM(nn.Module):
    def __init__(self, cfg: GraphWalkerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.D_s)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        self.memory = GraphWalkerMemory(cfg, tied_token_emb=self.token_emb)

    def set_training_step(self, step: int) -> None:
        """Sets the step counter used for Gumbel temperature / ε annealing."""
        self.memory.training_step = step

    def tick(self, token_id: torch.Tensor) -> WalkerReadout:
        return self.memory.step(token_id)
