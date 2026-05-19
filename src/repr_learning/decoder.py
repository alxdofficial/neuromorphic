"""Frozen Llama decoder + span-masked reconstruction.

Llama is loaded from HuggingFace, parameters frozen. The decoder:
1. Replaces masked token embeddings with a learned mask_embed
2. Prepends the memory tokens (from an encoder) to the input embeddings
3. Runs Llama forward (with gradient flow but no parameter updates)
4. Computes cross-entropy loss on the masked positions only

The mask_embed and the encoder are the only trainable components.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ReprConfig


def load_frozen_llama(model_name: str, dtype: torch.dtype = torch.bfloat16):
    """Load Llama, freeze all params, return the model + tokenizer.

    Llama is put in inference mode (no dropout) by `.train(False)`.
    All parameters get `requires_grad = False`. Gradient still flows
    through Llama during backward — Llama just doesn't update.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    for p in model.parameters():
        p.requires_grad = False
    model.train(False)
    return model, tok


class FrozenLlamaDecoder(nn.Module):
    """Frozen Llama with prepended memory tokens + span masking.

    Forward signature mirrors a standard LM:
        decoder(input_ids, mask_positions, memory_tokens) -> logits, loss
    """

    def __init__(self, cfg: ReprConfig, llama_model: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg

        if llama_model is None:
            llama_model, _ = load_frozen_llama(cfg.llama_model)
        self.llama = llama_model

        # Learnable mask embedding — initialized from mean of Llama's
        # input embedding table. Centered near Llama's typical token region.
        embed_layer = self.llama.get_input_embeddings()
        with torch.no_grad():
            init_vec = embed_layer.weight.mean(dim=0).clone()
        self.mask_embed = nn.Parameter(init_vec.to(dtype=torch.float32))

    def forward(
        self,
        input_ids: Tensor,            # [B, T] long
        mask_positions: Tensor,        # [B, T] bool
        memory_tokens: Tensor,         # [B, M, d_llama]
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            logits: [B, M+T, vocab] full logits over all positions
            loss:   scalar — cross-entropy on masked positions only
        """
        B, T = input_ids.shape
        M = memory_tokens.shape[1]
        d_llama = self.cfg.d_llama

        # 1. Llama's token embeddings (frozen)
        with torch.no_grad():
            text_embed = self.llama.get_input_embeddings()(input_ids)

        # 2. Replace masked positions with learnable mask_embed
        mask_vec = self.mask_embed.to(dtype=text_embed.dtype)
        text_embed = torch.where(
            mask_positions.unsqueeze(-1),
            mask_vec.view(1, 1, d_llama).expand(B, T, d_llama),
            text_embed,
        )

        # 3. Concat memory + text
        all_embed = torch.cat([
            memory_tokens.to(dtype=text_embed.dtype),
            text_embed,
        ], dim=1)                                              # [B, M+T, d_llama]

        # 4. Run Llama forward
        out = self.llama(inputs_embeds=all_embed)
        logits = out.logits                                     # [B, M+T, vocab]

        # 5. Slice text logits, compute CE on masked positions
        text_logits = logits[:, M:, :]                          # [B, T, vocab]

        # Causal LM: logits[t-1] predicts token[t].
        pred_logits = text_logits[:, :-1, :]                    # [B, T-1, V]
        targets = input_ids[:, 1:]                              # [B, T-1]
        target_mask = mask_positions[:, 1:]                     # [B, T-1]

        if target_mask.any():
            loss = F.cross_entropy(
                pred_logits[target_mask].float(),
                targets[target_mask],
                reduction="mean",
            )
        else:
            loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)

        return logits, loss
