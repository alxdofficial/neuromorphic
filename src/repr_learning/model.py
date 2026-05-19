"""Top-level ReprLearningModel — wires encoder + decoder together.

The model takes an encoder (V21Encoder, FlatBaselineEncoder, or
ContinuousBaselineEncoder) and the frozen Llama decoder, and produces
the reconstruction loss for training.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import ReprConfig
from .encoder import (
    ContinuousBaselineEncoder,
    FlatBaselineEncoder,
    V21Encoder,
)
from .decoder import FrozenLlamaDecoder


class ReprLearningModel(nn.Module):
    """Encoder + frozen Llama decoder, end-to-end.

    Args:
        cfg: ReprConfig instance
        variant: one of "v21", "flat_baseline", "continuous_baseline"
        llama_model: optional pre-loaded Llama (for sharing across models)
    """

    VARIANTS = {
        "v21": V21Encoder,
        "flat_baseline": FlatBaselineEncoder,
        "continuous_baseline": ContinuousBaselineEncoder,
    }

    def __init__(
        self,
        cfg: ReprConfig,
        variant: str = "v21",
        llama_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.variant = variant

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Must be one of {list(self.VARIANTS)}."
            )

        self.encoder = self.VARIANTS[variant](cfg)
        self.decoder = FrozenLlamaDecoder(cfg, llama_model=llama_model)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        mask_positions: Tensor,
    ) -> dict:
        """
        Args:
            input_ids: [B, T] Llama token ids
            attention_mask: [B, T] bool — True where real (not padding)
            mask_positions: [B, T] bool — True at positions to mask + predict

        Returns:
            dict with:
                loss        : total loss (recon + aux)
                loss_recon  : reconstruction CE on masked positions
                loss_aux    : auxiliary loss (e.g., load-balance)
                memory      : [B, M, d_llama] memory tokens (for inspection)
                **aux       : any extras from the encoder
        """
        # 1. Get Llama's frozen token embeddings as encoder input
        with torch.no_grad():
            embed_layer = self.decoder.llama.get_input_embeddings()
            token_embeds = embed_layer(input_ids)             # [B, T, d_llama]

        # 2. Encoder → memory tokens
        memory, aux = self.encoder(token_embeds, attention_mask)

        # 3. Decoder → loss
        _, loss_recon = self.decoder(input_ids, mask_positions, memory)

        # 4. Combine losses
        loss_aux = aux.get(
            "load_balance_loss",
            torch.zeros((), device=loss_recon.device, dtype=loss_recon.dtype),
        )
        loss = loss_recon + self.cfg.load_balance_coef * loss_aux

        out = {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_aux": loss_aux.detach() if isinstance(loss_aux, Tensor) else loss_aux,
            "memory_shape": tuple(memory.shape),
        }
        # Forward any extra encoder outputs (non-tensor or already detached)
        for k, v in aux.items():
            if k != "load_balance_loss":
                out[k] = v
        return out

    def trainable_parameters(self):
        """Yield only the trainable parameters (excludes frozen Llama)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
