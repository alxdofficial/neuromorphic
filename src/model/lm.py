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
        # No positional embedding: the scan layers are recurrent and their
        # hidden carry encodes sequence order implicitly. Same design as
        # Mamba / RWKV / S4 / RetNet. Explicit position would be redundant.

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

        H = x
        for i in range(split):
            H, h_last = self.layers[i](H, self._carries[i], reset_mask=reset_mask)
            self._carries[i] = h_last

        return H

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
        # Pick any parameter to get the target device — pos_embed may be None
        # when disabled, so fall back to the embedding weight.
        device = self.embedding.weight.device
        self._carries = [
            h.to(device) if h is not None else None
            for h in carries
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compute_mem_scale_stats(self) -> dict:
        """Diagnostic for mem_scale drift. See audit #9.

        mem_scale is a learnable per-dim scale on the memory readout in
        H_enriched = H_mid + mem_scale * readout. If it drifts toward 0
        the memory contribution collapses silently; if it explodes the
        memory overwhelms H_mid. Neither has any regularizer, so we need
        to watch it in the metrics stream.

        Init is sqrt(alpha) = 2.0. Sane range for abs_mean is ~[0.1, 10.0]
        — drift outside raises a one-time stdout warning and sets a flag in
        the returned dict so the plotter can surface it.
        """
        ms = self.mem_scale.detach().float()
        abs_mean = ms.abs().mean().item()
        # One-time drift warning: flag the first step where abs_mean leaves
        # the sane range in either direction. Avoids log-spam by caching
        # whether we've already warned per direction.
        MS_LO, MS_HI = 0.1, 10.0
        alert = ""
        if abs_mean < MS_LO:
            alert = f"below {MS_LO:.2f} — memory contribution collapsing"
            if not getattr(self, "_mem_scale_low_warned", False):
                print(f"[WARN] mem_scale abs_mean={abs_mean:.4f} {alert}")
                self._mem_scale_low_warned = True
        elif abs_mean > MS_HI:
            alert = f"above {MS_HI:.2f} — memory overwhelming H_mid"
            if not getattr(self, "_mem_scale_high_warned", False):
                print(f"[WARN] mem_scale abs_mean={abs_mean:.4f} {alert}")
                self._mem_scale_high_warned = True
        return {
            "mem_scale_mean": ms.mean().item(),
            "mem_scale_std": ms.std().item(),
            "mem_scale_abs_mean": abs_mean,
            "mem_scale_abs_max": ms.abs().max().item(),
            "mem_scale_abs_min": ms.abs().min().item(),
            "mem_scale_drift_alert": alert,
        }
