"""V8 Language Model — single-pass scan stack + per-CC PCM + end-injection.

All scan layers run once over T tokens (parallel). PCM computes surprise.
Memory signals are injected as a position-wise additive residual at the end,
right before the output head. No two-pass split.

CC→memory: raw H_slice[t] (D_cc=128) — the scan's causal representation
Memory→CC: raw mem_signal[t] (D_mem=D_cc=128) — gated, added to H

The scan is shared across CCs. Memory signals come from the memory graph
which runs per-token sequentially (outside autograd).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..model.scan import ScanLayer, RMSNorm
from .pcm import SingleColumnPCM
from .config import V8Config


class V8LM(nn.Module):
    """Language model: single-pass scan + PCM + end-injection of memory."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config

        D = config.D
        D_embed = config.D_embed
        D_cc = config.D_cc
        C = config.C

        # Embedding + positional encoding (sized to T)
        self.embedding = nn.Embedding(config.vocab_size, D_embed)
        if D_embed != D:
            self.proj_up = nn.Linear(D_embed, D)
            self.proj_down = nn.Linear(D, D_embed)
        else:
            self.proj_up = None
            self.proj_down = None
        self.pos_embed = nn.Parameter(torch.randn(config.T, D) * 0.02)

        # Full-D scan layers — ALL layers run in one pass
        self.layers = nn.ModuleList([
            ScanLayer(D, config.d_inner, config.dropout,
                      n_layers=config.L_total, glu_output=config.glu_output)
            for _ in range(config.L_total)
        ])

        # Per-CC PCM (independent weights per column)
        if config.pcm_enabled:
            self.pcm_modules = nn.ModuleList([
                SingleColumnPCM(D_cc, hidden=config.pcm_hidden) for _ in range(C)
            ])
        else:
            self.pcm_modules = None

        # Memory gate per CC (sigmoid(0) = 0.5 at init)
        self.mem_gate = nn.Parameter(torch.zeros(C))

        # Output head
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Scan carries
        self._carries = [None] * config.L_total

    def forward_scan(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full scan + PCM. Runs once, shared across memory samples.

        Args:
            input_ids: [BS, T]

        Returns:
            H: [BS, T, D] — hidden states after ALL scan layers
            x: [BS, T, D] — embedded input (for PCM reference)
            surprise: [BS, T, C, D_cc] — per-CC surprise
            aux_loss: scalar — PCM prediction loss
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_cc = self.config.D_cc
        D = self.config.D

        # Embed
        x = self.embedding(input_ids)
        if self.proj_up is not None:
            x = self.proj_up(x)
        x = x + self.pos_embed[:T]

        # ALL scan layers in one pass
        H = x
        for i in range(self.config.L_total):
            carry = self._carries[i]
            if self.config.gradient_checkpointing and self.training:
                H, h_last = grad_checkpoint(self.layers[i], H, carry,
                                            use_reentrant=False)
            else:
                H, h_last = self.layers[i](H, carry)
            self._carries[i] = h_last

        # Per-CC PCM (after scan — surprise reflects full scan context)
        aux_loss = torch.tensor(0.0, device=H.device)
        if self.pcm_modules is not None:
            H_cols = H.view(BS, T, C, D_cc)
            x_cols = x.view(BS, T, C, D_cc)
            surprise_list = []
            gain_modulated = []
            for c in range(C):
                h_c = H_cols[:, :, c]
                x_c = x_cols[:, :, c]
                surp, z_hat, z = self.pcm_modules[c].compute_surprise(h_c, x_c)
                h_c = self.pcm_modules[c].apply_gain(h_c, surp)
                aux_loss = aux_loss + self.pcm_modules[c].prediction_loss(z_hat, z)
                surprise_list.append(surp)
                gain_modulated.append(h_c)
            H = torch.stack(gain_modulated, dim=2).reshape(BS, T, D)
            surprise = torch.stack(surprise_list, dim=2)
            aux_loss = aux_loss / C * self.config.pcm_pred_weight
        else:
            surprise = torch.zeros(BS, T, C, D_cc, device=H.device, dtype=H.dtype)

        return H, x, surprise, aux_loss

    def forward_output(self, H: Tensor, mem_signals: Tensor | None = None) -> Tensor:
        """End-injection of memory + output head. Cheap, per-sample.

        Args:
            H: [BS, T, D] — scan output (shared across samples)
            mem_signals: [BS, T, C, D_cc] or None — memory signals to inject

        Returns:
            logits: [BS, T, vocab]
        """
        BS, T, D = H.shape

        if mem_signals is not None:
            gate = torch.sigmoid(self.mem_gate)  # [C]
            mem_contribution = gate[None, None, :, None] * mem_signals
            H = H + mem_contribution.reshape(BS, T, D)

        out = H
        if self.proj_down is not None:
            out = self.proj_down(out)
        out = self.ln_final(out)
        logits = self.lm_head(out) * (self.config.D_embed ** -0.5)

        return logits

    def initialize_carries(self):
        self._carries = [None] * self.config.L_total

    def detach_carries(self):
        self._carries = [
            h.detach() if h is not None else None for h in self._carries
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
