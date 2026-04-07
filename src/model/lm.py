"""Language Model — split-scan with mid-scan memory injection + PCM.

Lower scan layers → PCM surprise → augment → memory graph → upper scan → head.
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

        # split_mlp: combines H_mid with surprise (called per-token from memory loop)
        split_h = config.split_mlp_hidden
        self.split_mlp = nn.Sequential(
            nn.Linear(2 * D, split_h),
            nn.SiLU(),
            nn.Linear(split_h, D),
        )
        with torch.no_grad():
            self.split_mlp[2].weight.mul_(1.0 / math.sqrt(2 * config.L_total))

        self.mem_scale = nn.Parameter(
            torch.full((D,), math.sqrt(config.alpha)))

        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._carries = [None] * config.L_total

    def forward_scan_lower(self, input_ids: Tensor,
                           reset_mask: Tensor | None = None) -> Tensor:
        """Lower scan only. PCM is now in the memory loop.

        Returns: H_mid [BS, T, D]
        """
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

    def augment_single(self, H_mid_t: Tensor, surprise_t: Tensor) -> Tensor:
        """Per-token augmentation for interleaved PCM. H_mid_t/surprise_t: [BS, D]."""
        surp_rms = surprise_t.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        surprise_normed = surprise_t * surp_rms
        mlp_input = torch.cat([H_mid_t, surprise_normed], dim=-1)
        # Cast to f32 for split_mlp (f32 params), then back
        out = self.split_mlp(mlp_input.float()).to(H_mid_t.dtype)
        return H_mid_t + out

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
