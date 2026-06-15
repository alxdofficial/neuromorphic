"""AutoCompressors / RMT-style recurrent soft-prompt summary compressor."""
from __future__ import annotations

import torch
import torch.nn as nn

from ...common import _NormMatch
from ...config import ReprConfig


class AutoCompressorBaselineEncoder(nn.Module):
    """AutoCompressors (Chevalier et al. EMNLP'23) / RMT-style RECURRENT soft-prompt
    summary compressor on a FROZEN Llama-1B + encoder-LoRA — the recurrent baseline.

    Per window: cat([carried summary (read-memory), window embeds, M write slots]),
    run the LoRA-Llama, read the write-slot hiddens as the NEW summary (= next window's
    read-memory). finalize returns the final M summary vectors as prepend memory. The
    summary is the differentiable cross-window carry (BPTT'd by the trainer). Adapted to
    a frozen backbone + LoRA (the papers fine-tune the whole LM) — same fair-comparison
    adaptation as ICAE/CCM/Beacon. M = n_flat_codes (= mem_tokens). Port skeleton: RMT
    MemoryCell (booydar/recurrent-memory-transformer)."""

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        base, _ = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        n_wrapped = apply_lora_to_llama(
            base,
            rank=int(getattr(cfg, "autocompressor_lora_rank", cfg.icae_lora_rank)),
            alpha=int(getattr(cfg, "autocompressor_lora_alpha", cfg.icae_lora_alpha)),
            target_names=tuple(getattr(cfg, "autocompressor_lora_targets",
                                       cfg.llama_lora_target_names)),
        )
        base.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        self.M = int(getattr(cfg, "autocompressor_n_slots", 0) or cfg.n_flat_codes)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(dim=0)
            emb_std = embed.weight.float().std().item()
        slot_init = (mean_vec.view(1, cfg.d_llama).repeat(self.M, 1)
                     + emb_std * torch.randn(self.M, cfg.d_llama))
        self.slots = nn.Parameter(slot_init)                  # M write-memory slots
        self.summary0 = nn.Parameter(slot_init.clone())       # initial read-memory (window 1)
        self.norm = _NormMatch(cfg.d_llama)
        with torch.no_grad():   # seed norm-match scale to the backbone embed norm
            self.norm.scale.data.fill_(   # (0.9 default is ~3x too quiet on SmolLM2; match hlvocab)
                base.get_input_embeddings().weight.float().norm(dim=-1).mean().item())
        print(f"[AutoCompressor] encoder-LoRA wrapped {n_wrapped} layers; M={self.M} (recurrent)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def init_streaming_state(self, batch_size: int, device, dtype):
        return {"summary": self.summary0.to(dtype).unsqueeze(0)
                .expand(batch_size, self.M, self.cfg.d_llama).contiguous()}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        B, W, d = token_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        summ = state["summary"]                               # [B, M, d] read-memory
        slots = self.slots.to(token_embeds.dtype).unsqueeze(0).expand(B, self.M, d)
        inp = torch.cat([summ, token_embeds, slots], dim=1)   # [B, M+W+M, d]
        _ones = lambda n: torch.ones(B, n, device=attention_mask.device, dtype=torch.long)
        attn = torch.cat([_ones(self.M), attention_mask.long(), _ones(self.M)], dim=1)
        h = self.base.model(inputs_embeds=inp, attention_mask=attn).last_hidden_state
        new_summary = h[:, -self.M:, :].to(token_embeds.dtype)   # write slots -> next read-memory
        # all-pad window: carry the summary unchanged (no real tokens to compress)
        has_real = attention_mask.bool().any(dim=1).view(-1, 1, 1)
        new_summary = torch.where(has_real, new_summary, summ)
        return {"summary": new_summary}, {}

    def finalize_memory(self, state):
        return self.norm(state["summary"].float()), {}       # [B, M, d_llama]

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        state = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        return self.finalize_memory(state)
