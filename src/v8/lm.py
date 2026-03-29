"""V8 Language Model — split-scan with mid-scan memory injection.

Lower scan layers (0..split-1) produce H_mid. Memory graph reads H_mid
per-token and produces mem_signals. Memory is injected into H_mid, then
upper scan layers (split..L-1) process the memory-enriched representation.

PCM computes surprise at the split point by predicting the next scan
hidden state. A split-point MLP combines H_mid and surprise into a
unified representation before the upper scan.

CC->memory: H_mid replicated per neuron (D_neuron=256, C_mem = D // D_neuron = 2048 // 256 = 8 slices)
Memory->CC: neuron messages averaged over replicas, combined via MLP, added to H_mid
Upper scan layers see memory-enriched + surprise-modulated input.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
            self.layers.append(
                ScanLayer(D, config.d_inner, config.dropout,
                          n_layers=config.L_total, glu_output=config.glu_output)
            )

        # Batched PCM across all columns
        if config.pcm_enabled:
            self.pcm = BatchedPCM(C, D_cc, hidden=config.pcm_hidden)
            # Split-point MLP: combines H_mid + surprise before upper scan
            self.split_mlp = nn.Sequential(
                nn.Linear(2 * D, config.d_inner),
                nn.SiLU(),
                nn.Linear(config.d_inner, D),
            )
            # Depth-scaled init: same pattern as scan layer proj_out.
            # Kaiming (default) × 1/sqrt(2*L) — small residual at start,
            # but nonzero so gradients flow to w1 and to surprise/PCM.
            with torch.no_grad():
                self.split_mlp[2].weight.mul_(1.0 / math.sqrt(2 * config.L_total))
        else:
            self.pcm = None
            self.split_mlp = None

        # Memory injection: learnable per-dim scale, no MLP.
        # Direct addition avoids backward attenuation from MLP weights.
        # Scale initialized to balance magnitudes: readout ~0.078/elem,
        # H_mid ~0.46/elem, so start at ~6 to match.
        C_mem = D // config.D_neuron
        N_per_slice = config.N_mem_neurons // C_mem
        init_scale = N_per_slice ** 0.5  # undo the 1/sqrt(N) in readout
        self.mem_scale = nn.Parameter(torch.full((D,), init_scale))

        # Output head
        self.ln_final = nn.LayerNorm(D_embed)
        self.lm_head = nn.Linear(D_embed, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Scan carries
        self._carries = [None] * config.L_total

    def forward_scan_lower(self, input_ids: Tensor,
                           reset_mask: Tensor | None = None,
                           ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Lower scan layers (0..split-1) + PCM. Produces H_mid for memory graph.

        PCM computes surprise at the split point. H_mid passes through
        unchanged (no gain modulation). Surprise is returned separately
        for the upper scan to use as a side input.

        Args:
            input_ids: [BS, T]
            reset_mask: [BS, T] bool or None — True at positions where
                the recurrent state should be reset (internal document
                boundaries).

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
            H, h_last = self.layers[i](H, carry, reset_mask=reset_mask)
            self._carries[i] = h_last

        # PCM at the split point — predict next hidden state, compute surprise
        aux_loss = torch.tensor(0.0, device=H.device)
        if self.pcm is not None:
            H_cols = H.view(BS, T, C, D_cc)
            surprise, H_hat, pred_loss, per_cc_pred_loss = self.pcm(H_cols)
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

            # Reshape to [BS, T, D] for split-point MLP
            surprise_flat = surprise.reshape(BS, T, D)
        else:
            surprise_flat = torch.zeros(BS, T, D, device=H.device, dtype=H.dtype)
            self._pcm_stats = None

        # H passes through unchanged — no gain modulation
        return H, surprise_flat, x, aux_loss

    def forward_scan_upper(self, H_enriched: Tensor,
                           surprise: Tensor | None = None,
                           reset_mask: Tensor | None = None) -> Tensor:
        """Upper scan layers (split..L-1) on memory-enriched input.

        Surprise is mixed into the representation via split_mlp before
        the upper scan layers process it. No side_input mechanism.

        Args:
            H_enriched: [BS, T, D] — H_mid + gated memory signals
            surprise: [BS, T, D] or None — PCM surprise
            reset_mask: [BS, T] bool or None — True at positions where
                the recurrent state should be reset (internal document
                boundaries).

        Returns:
            H: [BS, T, D] — hidden states after upper scan layers
        """
        split = self.config.scan_split_at

        H = H_enriched

        # Mix surprise at split point via MLP (residual, depth-scaled init)
        # RMSNorm on surprise prevents unbounded growth — the PCM prediction
        # error grows as H_mid evolves faster than the PCM can track.
        if surprise is not None and self.split_mlp is not None:
            surp_rms = surprise.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
            surprise_normed = surprise * surp_rms
            H = H + self.split_mlp(torch.cat([H, surprise_normed], dim=-1))

        for i in range(split, self.config.L_total):
            carry = self._carries[i]
            H, h_last = self.layers[i](H, carry, reset_mask=reset_mask)
            self._carries[i] = h_last

        return H

    def inject_memory(self, H_mid: Tensor, mem_signals: Tensor) -> Tensor:
        """Add scaled memory readout to H_mid.

        Direct addition with learnable per-dim scale — no MLP in the
        gradient path. Gradient flows to memory with no attenuation
        beyond the scale factor itself.

        Args:
            H_mid: [BS, T, D] — lower scan output (with autograd)
            mem_signals: [BS, T, D] — memory readout (carry gradients)

        Returns:
            H_enriched: [BS, T, D] — memory-enriched hidden states
        """
        return H_mid + self.mem_scale * mem_signals.to(H_mid.dtype)

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
    def forward_scan(self, input_ids: Tensor,
                     reset_mask: Tensor | None = None,
                     ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full scan + PCM (no memory). Used only for no-memory baseline."""
        H_mid, surprise, x, aux_loss = self.forward_scan_lower(
            input_ids, reset_mask=reset_mask)
        H = self.forward_scan_upper(H_mid, surprise=surprise,
                                    reset_mask=reset_mask)
        return H, surprise, x, aux_loss

    def initialize_carries(self):
        self._carries = [None] * self.config.L_total

    def detach_carries(self):
        self._carries = [
            h.detach() if h is not None else None for h in self._carries
        ]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
