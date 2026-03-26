"""V8 Language Model — split-scan with mid-scan memory injection.

Lower scan layers (0..split-1) produce H_mid. Memory graph reads H_mid
per-token and produces mem_signals. Memory is injected into H_mid, then
upper scan layers (split..L-1) process the memory-enriched representation.

PCM computes surprise at the split point. Surprise is passed as a side
input to the first upper scan layer, influencing both its gate and input
additively. No multiplicative gain modulation.

CC->memory: H_mid sliced per CC (D_cc=128) -- lower scan's representation
Memory->CC: port neuron messages (D_mem=D_cc=128) -- gated, added to H_mid
Upper scan layers see memory-enriched input and learn to use it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..model.scan import ScanLayer, RMSNorm
from .pcm import BatchedPCM
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
        split = config.scan_split_at
        self.layers = nn.ModuleList()
        for i in range(config.L_total):
            # First upper scan layer gets side_dim=D for PCM surprise input
            side_dim = D if (i == split and config.pcm_enabled) else 0
            self.layers.append(
                ScanLayer(D, config.d_inner, config.dropout,
                          n_layers=config.L_total, glu_output=config.glu_output,
                          side_dim=side_dim)
            )

        # Batched PCM across all columns
        if config.pcm_enabled:
            self.pcm = BatchedPCM(C, D_cc, hidden=config.pcm_hidden)
        else:
            self.pcm = None

        # Memory gate per CC (sigmoid(0) = 0.5 at init)
        self.mem_gate = nn.Parameter(torch.zeros(C))

        # Output head
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Scan carries
        self._carries = [None] * config.L_total

    def forward_scan_lower(self, input_ids: Tensor
                           ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Lower scan layers (0..split-1) + PCM. Produces H_mid for memory graph.

        PCM computes surprise at the split point. H_mid passes through
        unchanged (no gain modulation). Surprise is returned separately
        for the upper scan to use as a side input.

        Args:
            input_ids: [BS, T]

        Returns:
            H_mid: [BS, T, D] — hidden states after lower scan (WITH autograd)
            surprise: [BS, T, D] — per-CC surprise stacked to full D (WITH autograd)
            x: [BS, T, D] — embedded input
            aux_loss: scalar — PCM prediction loss
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_cc = self.config.D_cc
        D = self.config.D
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

        # PCM at the split point — compute surprise, no gain modulation
        aux_loss = torch.tensor(0.0, device=H.device)
        if self.pcm is not None:
            H_cols = H.view(BS, T, C, D_cc)
            x_cols = x.view(BS, T, C, D_cc)
            surprise, z_hat, z, pred_loss, per_cc_pred_loss = self.pcm(H_cols, x_cols)
            aux_loss = pred_loss * self.config.pcm_pred_weight

            # Cache lightweight PCM stats for diagnostics (no tensors retained)
            surp_norms = surprise.detach().norm(dim=-1)  # [BS, T, C]
            self._pcm_stats = {
                "surprise_mean": surp_norms.mean().item(),
                "surprise_std": surp_norms.std().item(),
                "surprise_max": surp_norms.max().item(),
                "surprise_per_cc": surp_norms.mean(dim=(0, 1)).tolist(),  # [C]
                "pred_loss_per_cc": per_cc_pred_loss,  # [C]
            }

            # Reshape to [BS, T, D] for upper scan side input
            surprise_flat = surprise.reshape(BS, T, D)
        else:
            surprise_flat = torch.zeros(BS, T, D, device=H.device, dtype=H.dtype)
            self._pcm_stats = None

        # H passes through unchanged — no gain modulation
        return H, surprise_flat, x, aux_loss

    def forward_scan_upper(self, H_enriched: Tensor,
                           surprise: Tensor | None = None) -> Tensor:
        """Upper scan layers (split..L-1) on memory-enriched input.

        The first upper layer receives surprise as a side input, which
        additively influences its gate and scan input. Subsequent layers
        see the result through normal residual connections.

        Args:
            H_enriched: [BS, T, D] — H_mid + gated memory signals
            surprise: [BS, T, D] or None — PCM surprise (side input to first upper layer)

        Returns:
            H: [BS, T, D] — hidden states after upper scan layers
        """
        split = self.config.scan_split_at

        H = H_enriched
        for i in range(split, self.config.L_total):
            carry = self._carries[i]
            # Only the first upper layer gets surprise as side input
            side = surprise if i == split else None
            if self.config.gradient_checkpointing and self.training:
                H, h_last = grad_checkpoint(self.layers[i], H, carry, side,
                                            use_reentrant=False)
            else:
                H, h_last = self.layers[i](H, carry, side_input=side)
            self._carries[i] = h_last

        return H

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
        H_mid, surprise, x, aux_loss = self.forward_scan_lower(input_ids)
        H = self.forward_scan_upper(H_mid, surprise=surprise)
        return H, surprise, x, aux_loss

    def initialize_carries(self):
        self._carries = [None] * self.config.L_total

    def detach_carries(self):
        self._carries = [
            h.detach() if h is not None else None for h in self._carries
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
