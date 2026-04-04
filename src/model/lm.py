"""Language Model — split-scan with mid-scan memory injection + PCM.

Lower scan layers → PCM surprise → augment → memory graph → upper scan → head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .scan import ScanLayer, RMSNorm
from .pcm import BatchedPCM
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

        if config.pcm_enabled:
            self.pcm = BatchedPCM(config.C, config.D_cc, hidden=config.pcm_hidden)
            self.split_mlp = nn.Sequential(
                nn.Linear(2 * D, config.d_inner),
                nn.SiLU(),
                nn.Linear(config.d_inner, D),
            )
            with torch.no_grad():
                self.split_mlp[2].weight.mul_(1.0 / math.sqrt(2 * config.L_total))
        else:
            self.pcm = None
            self.split_mlp = None

        self.mem_scale = nn.Parameter(
            torch.full((D,), math.sqrt(config.alpha)))

        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._carries = [None] * config.L_total

    def forward_scan_lower(self, input_ids: Tensor,
                           reset_mask: Tensor | None = None,
                           ) -> tuple[Tensor, Tensor, Tensor]:
        """Lower scan + PCM.

        Returns: H_mid [BS,T,D], surprise [BS,T,D], aux_loss scalar.
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_cc = self.config.D_cc
        split = self.config.scan_split_at

        x = self.embedding(input_ids)
        if self.proj_up is not None:
            x = self.proj_up(x)
        x = x + self.pos_embed[:T]

        H = x
        for i in range(split):
            H, h_last = self.layers[i](H, self._carries[i], reset_mask=reset_mask)
            self._carries[i] = h_last

        H_mid = H

        aux_loss = torch.tensor(0.0, device=H.device)
        if self.pcm is not None:
            H_cols = H_mid.view(BS, T, C, D_cc)
            surprise_cols, _, pred_loss, _ = self.pcm(H_cols)
            surprise = surprise_cols.reshape(BS, T, self.config.D)
            aux_loss = pred_loss * self.config.pcm_pred_weight
        else:
            surprise = torch.zeros_like(H_mid)

        return H_mid, surprise, aux_loss

    def augment(self, H_mid: Tensor, surprise: Tensor) -> Tensor:
        """Combine H_mid and surprise into H_aug."""
        if self.split_mlp is not None:
            surp_rms = surprise.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            surprise_normed = surprise * surp_rms
            H_aug = H_mid + self.split_mlp(
                torch.cat([H_mid, surprise_normed], dim=-1))
        else:
            H_aug = H_mid
        return H_aug

    def inject_memory(self, H_aug: Tensor, mem_out: Tensor) -> Tensor:
        return H_aug + self.mem_scale * mem_out.to(H_aug.dtype)

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

    def reset_carries(self, mask: Tensor):
        """mask: [BS] bool — True for elements to reset."""
        for i, h in enumerate(self._carries):
            if h is not None:
                keep = (~mask).to(h.dtype).unsqueeze(-1)
                self._carries[i] = h * keep

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
