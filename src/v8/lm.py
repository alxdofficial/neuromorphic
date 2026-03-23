"""V8 Language Model — split-scan with mid-scan memory injection.

Lower scan layers (0..split-1) produce H_mid. Memory graph reads H_mid
per-token and produces mem_signals. Memory is injected into H_mid, then
upper scan layers (split..L-1) process the memory-enriched representation.

CC→memory: H_mid sliced per CC (D_cc=128) — lower scan's representation
Memory→CC: port neuron messages (D_mem=D_cc=128) — gated, added to H_mid
Upper scan layers see memory-enriched input and learn to use it.
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
    """Language model: split-scan + mid-scan memory injection + PCM."""

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

        # Scan layers — split into lower and upper
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

    def forward_scan_lower(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Lower scan layers (0..split-1). Produces H_mid for memory graph.

        Args:
            input_ids: [BS, T]

        Returns:
            H_mid: [BS, T, D] — hidden states after lower scan layers (WITH autograd)
            x: [BS, T, D] — embedded input (for PCM reference later)
        """
        BS, T = input_ids.shape
        split = self.config.scan_split_at

        # Embed
        x = self.embedding(input_ids)
        if self.proj_up is not None:
            x = self.proj_up(x)
        x = x + self.pos_embed[:T]

        # Lower scan layers
        H = x
        for i in range(split):
            carry = self._carries[i]
            if self.config.gradient_checkpointing and self.training:
                H, h_last = grad_checkpoint(self.layers[i], H, carry,
                                            use_reentrant=False)
            else:
                H, h_last = self.layers[i](H, carry)
            self._carries[i] = h_last

        return H, x

    def forward_scan_upper(self, H_enriched: Tensor, x: Tensor
                           ) -> tuple[Tensor, Tensor, Tensor]:
        """Upper scan layers (split..L-1) + PCM on memory-enriched input.

        Args:
            H_enriched: [BS, T, D] — H_mid + gated memory signals
            x: [BS, T, D] — embedded input (for PCM)

        Returns:
            H: [BS, T, D] — hidden states after all upper scan layers + PCM
            surprise: [BS, T, C, D_cc]
            aux_loss: scalar
        """
        BS, T, D = H_enriched.shape
        C = self.config.C
        D_cc = self.config.D_cc
        split = self.config.scan_split_at

        # Upper scan layers
        H = H_enriched
        for i in range(split, self.config.L_total):
            carry = self._carries[i]
            if self.config.gradient_checkpointing and self.training:
                H, h_last = grad_checkpoint(self.layers[i], H, carry,
                                            use_reentrant=False)
            else:
                H, h_last = self.layers[i](H, carry)
            self._carries[i] = h_last

        # Per-CC PCM (after upper scan — surprise reflects full context + memory)
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

        return H, surprise, aux_loss

    def inject_memory(self, H_mid: Tensor, mem_signals: Tensor) -> Tensor:
        """Add gated memory signals to H_mid.

        Args:
            H_mid: [BS, T, D] — lower scan output (with autograd)
            mem_signals: [BS, T, C, D_cc] — port neuron messages (detached)

        Returns:
            H_enriched: [BS, T, D] — memory-enriched hidden states
        """
        gate = torch.sigmoid(self.mem_gate)  # [C]
        mem_contribution = gate[None, None, :, None] * mem_signals
        return H_mid + mem_contribution.reshape(H_mid.shape)

    def forward_output(self, H: Tensor) -> Tensor:
        """Output head only (memory injection happens mid-scan now).

        Args:
            H: [BS, T, D] — final hidden states after upper scan + PCM

        Returns:
            logits: [BS, T, vocab]
        """
        out = H
        if self.proj_down is not None:
            out = self.proj_down(out)
        out = self.ln_final(out)
        logits = self.lm_head(out) * (self.config.D_embed ** -0.5)
        return logits

    # --- Legacy: full scan for no-memory path ---
    def forward_scan(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full scan + PCM (no memory). Used only for no-memory baseline."""
        H_mid, x = self.forward_scan_lower(input_ids)
        H, surprise, aux_loss = self.forward_scan_upper(H_mid, x)
        return H, x, surprise, aux_loss

    def initialize_carries(self):
        self._carries = [None] * self.config.L_total

    def detach_carries(self):
        self._carries = [
            h.detach() if h is not None else None for h in self._carries
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
