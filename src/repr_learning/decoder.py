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


class LoRALinear(nn.Module):
    """Manual LoRA wrapper around a frozen Linear.

    Forward computes `base(x) + (x @ A^T @ B^T) * scale`, where A is a
    small-init [rank, d_in] random matrix and B is a zero-init [d_out, rank]
    matrix. At init, B=0 ⇒ output exactly matches base — LoRA is a no-op
    until trained. Only A and B receive gradients.

    Reference: Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language
    Models" (arXiv:2106.09685).
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")
        d_out, d_in = base_linear.weight.shape
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        # Train A, B in float32 even when base is bf16 — keeps the low-rank
        # update numerically stable. Cast on the fly in forward.
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / rank
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        # Compute update in float32 for stability, cast back to match base
        x32 = x.to(self.lora_A.dtype)
        update = (x32 @ self.lora_A.T) @ self.lora_B.T            # [..., d_out]
        update = update.to(base_out.dtype) * self.scale
        return base_out + update


def apply_lora_to_llama(
    llama: nn.Module,
    rank: int,
    alpha: float,
    target_names: tuple,
) -> int:
    """Wrap target Linear submodules in Llama with LoRALinear.

    Walks the module tree and replaces children whose name is in
    `target_names` (e.g., 'q_proj', 'v_proj') with a LoRALinear wrapper.
    Returns the count of replacements.
    """
    n_wrapped = 0
    for module in llama.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, wrapped)
                n_wrapped += 1
    return n_wrapped


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
        # Always freeze the base LM — whether self-loaded or externally supplied
        # (don't trust callers to have frozen it). LoRA below re-enables only the
        # adapter params. Prevents an accidental-unfrozen-Llama footgun.
        for p in self.llama.parameters():
            p.requires_grad_(False)

        # Optional LoRA — wraps Llama's q_proj+v_proj (or configured targets)
        # with trainable low-rank updates. Lets the frozen LM adapt to the
        # memory token distribution that the encoder produces.
        if cfg.use_llama_lora:
            n_wrapped = apply_lora_to_llama(
                self.llama,
                rank=cfg.llama_lora_rank,
                alpha=cfg.llama_lora_alpha,
                target_names=tuple(cfg.llama_lora_target_names),
            )
            print(f"[decoder] LoRA wrapped {n_wrapped} Linear layers "
                  f"(rank={cfg.llama_lora_rank}, alpha={cfg.llama_lora_alpha})")

        # Learnable mask embedding — initialized from mean of Llama's
        # input embedding table. Centered near Llama's typical token region.
        embed_layer = self.llama.get_input_embeddings()
        with torch.no_grad():
            init_vec = embed_layer.weight.mean(dim=0).clone()
        self.mask_embed = nn.Parameter(init_vec.to(dtype=torch.float32))

    def train(self, mode: bool = True):
        """Override to keep frozen Llama in inference mode regardless of
        the parent wrapper's train/eval state."""
        super().train(mode)
        self.llama.train(False)
        return self

    def forward(
        self,
        input_ids: Tensor,                       # [B, T] long
        mask_positions: Tensor,                  # [B, T] bool
        memory_tokens: Tensor,                   # [B, M, d_llama]
        attention_mask: Optional[Tensor] = None, # [B, T] bool, True=real token
        token_embeds: Optional[Tensor] = None,   # [B, T, d_llama] pre-computed embeds
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            logits: [B, M+T, vocab] full logits over all positions
            loss:   scalar — cross-entropy on masked positions only
        """
        B, T = input_ids.shape
        M = memory_tokens.shape[1]
        d_llama = self.cfg.d_llama

        # 1. Llama's token embeddings (reuse if caller already computed them
        # for the encoder; avoids a second embed lookup per step).
        if token_embeds is None:
            with torch.no_grad():
                text_embed = self.llama.get_input_embeddings()(input_ids)
        else:
            text_embed = token_embeds

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

        # 3b. Build full attention mask: memory tokens always attended,
        # text tokens follow caller-provided mask. Llama uses HF convention
        # (1=attend, 0=ignore). Pass None when no padding to let Llama
        # use its causal-mask default.
        llama_attn_mask = None
        if attention_mask is not None:
            mem_mask = torch.ones(B, M, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            full_mask = torch.cat([mem_mask, attention_mask], dim=1)
            llama_attn_mask = full_mask.to(dtype=torch.long)

        # 4. Run Llama forward
        out = self.llama(inputs_embeds=all_embed, attention_mask=llama_attn_mask)
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
            # No masked targets in this batch. Return a zero loss that's
            # still connected to trainable inputs so backward() succeeds.
            # Without this, variants with zero aux loss (B, MT, Mamba)
            # would hit "tensor does not require grad" on backward.
            loss = (memory_tokens.float().sum() * 0.0
                    + self.mask_embed.float().sum() * 0.0)

        return logits, loss
