"""Language Model — split-scan with mid-scan memory injection.

Lower scan → memory graph → upper scan → head.
The LM head is shared as the memory-prediction head via weight tying.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .scan import ScanLayer, RMSNorm
from .config import Config


class LM(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        config.validate()
        self.config = config

        D = config.D
        D_embed = config.D_embed

        self.embedding = nn.Embedding(config.vocab_size, D_embed)
        if D_embed != D:
            self.proj_up = nn.Linear(D_embed, D)
            self.proj_down = nn.Linear(D, D_embed)
        else:
            self.proj_up = None
            self.proj_down = None
        self.pos_embed = nn.Parameter(torch.randn(config.T, D) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(config.L_total):
            self.layers.append(
                ScanLayer(D, config.d_inner, config.dropout,
                          n_layers=config.L_total, glu_output=config.glu_output))

        self.mem_scale = nn.Parameter(
            torch.full((D,), math.sqrt(config.alpha)))

        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._carries = [None] * config.L_total

    def forward_scan_lower(self, input_ids: Tensor,
                           reset_mask: Tensor | None = None) -> Tensor:
        """Lower scan. Returns: H_mid [BS, T, D]"""
        BS, T = input_ids.shape
        split = self.config.scan_split_at

        x = self.embedding(input_ids)
        if self.proj_up is not None:
            x = self.proj_up(x)
        x = x + self.pos_embed[:T]

        H = x
        for i in range(split):
            H, h_last = self.layers[i](H, self._carries[i], reset_mask=reset_mask)
            self._carries[i] = h_last

        return H

    def mem_head_target_logit(self, readout: Tensor, target: Tensor) -> Tensor:
        """Unnormalized log-prob at target under the memory head.

        Shares proj_down → ln_final → lm_head with the main LM head (weight-tied).
        readout: [BS, D] → target_logit: [BS] (scalar per sample).
        Used as the cheap live surprise signal inside the memory loop.
        """
        dt = self.lm_head.weight.dtype
        x = readout.to(dt)
        if self.proj_down is not None:
            x = self.proj_down(x)
        x = self.ln_final(x)  # [BS, D_embed]
        target_emb = self.lm_head.weight[target]  # [BS, D_embed]
        return (x * target_emb).sum(dim=-1)  # [BS]

    def mem_head_logits(self, readouts: Tensor) -> Tensor:
        """Full memory-head logits over a sequence of readouts.

        readouts: [BS, T, D] → logits: [BS, T, V].
        Used at segment end to compute mem_pred_loss (proper CE).
        """
        return self.forward_output(readouts.to(self.lm_head.weight.dtype))

    def forward_scan_upper(self, H_enriched: Tensor,
                           reset_mask: Tensor | None = None) -> Tensor:
        split = self.config.scan_split_at
        H = H_enriched
        for i in range(split, self.config.L_total):
            H, h_last = self.layers[i](H, self._carries[i], reset_mask=reset_mask)
            self._carries[i] = h_last
        return H

    def forward_output(self, H: Tensor) -> Tensor:
        out = H
        if self.proj_down is not None:
            out = self.proj_down(out)
        out = self.ln_final(out)
        return self.lm_head(out)

    def detach_carries(self):
        for i, h in enumerate(self._carries):
            if h is not None:
                self._carries[i] = h.detach()

    def runtime_state_dict(self) -> dict:
        return {
            "carries": [
                h.clone() if h is not None else None
                for h in self._carries
            ],
        }

    def load_runtime_state(self, state: dict):
        carries = state.get("carries")
        if carries is None:
            return
        self._carries = [
            h.to(self.pos_embed.device) if h is not None else None
            for h in carries
        ]

    def reset_carries(self, mask: Tensor):
        """mask: [BS] bool — True for elements to reset."""
        for i, h in enumerate(self._carries):
            if h is not None:
                keep = (~mask).to(h.dtype).unsqueeze(-1)
                self._carries[i] = h * keep

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
