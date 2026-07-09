"""Vanilla references: NullEncoder (loss floor) + FullContextEncoder (ceiling)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ...config import ReprConfig


class NullEncoder(nn.Module):
    """Baseline 6: vanilla Llama — no encoder, no memory tokens.

    Returns an empty memory tensor of shape [B, 0, d_llama]. The decoder's
    forward path then runs Llama purely on the masked text input, with
    only mask_embed as a trainable parameter. This is the LOSS FLOOR for
    "what can Llama do on this task without any side-car module?"

    If V2.1 / A / B / MT / Mamba can't beat this floor, the memory module
    isn't contributing — the task is solvable from local context alone.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

    def init_streaming_state(self, batch_size: int, device, dtype):
        # NullEncoder has no slot state; we stash (B, device, dtype) in a
        # zero-width tensor so finalize_memory can recover B and the device.
        return torch.zeros(batch_size, 0, self.cfg.d_llama, device=device, dtype=dtype)

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra  # Null encoder has no encoder modules (ignores surprise etc.)
        return state, {}

    def finalize_memory(self, state: Tensor) -> tuple[Tensor, dict]:
        aux = {"load_balance_loss": torch.zeros((), device=state.device, dtype=torch.float32)}
        return state, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        B = token_embeds.shape[0]
        memory = torch.zeros(
            B, 0, self.cfg.d_llama,
            device=token_embeds.device,
            dtype=token_embeds.dtype,
        )
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=token_embeds.device, dtype=torch.float32,
            ),
        }
        return memory, aux


class FullContextEncoder(nn.Module):
    """Vanilla full-context — the CEILING reference.

    Passes the raw token embeddings through unchanged as "memory tokens."
    No encoding, no compression — Llama literally sees the full context
    prepended to the question. Establishes the upper bound: what frozen
    Llama achieves when the entire source is visible.

    Companion to NullEncoder (the floor with NO memory). Together they
    bracket every compressed memory variant:
        NullEncoder (floor)  ≤  any memory variant  ≤  FullContextEncoder (ceiling)

    Any variant that doesn't beat NullEncoder has useless memory; any that
    approaches FullContextEncoder has near-perfect compression. The variants
    in between tell us the bits-per-information cost of each architecture.

    Caveat: this variant has zero compression — its "memory" is the raw
    text (M = chunk_size = 4096). Reads via the same prepend pathway as
    other prepend variants. Llama-3.2-1B has a 128k context window so
    prepending 4k + question + answer fits comfortably.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg

    def init_streaming_state(self, batch_size: int, device, dtype):
        # Start with an empty memory; streaming_write accumulates token embeds
        # across windows so that finalize_memory returns the full context.
        return {
            "context_embeds": None,
            "context_mask": None,
            "_B": batch_size,
            "_device": device,
            "_dtype": dtype,
        }

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        # Accumulate raw embeddings across the streaming windows. attention_mask
        # is honored: padded positions still occupy slots but are zero-vectored
        # so the decoder's positional encoding tracks correctly.
        del chunk_offset, extra  # full-context carries raw embeds; ignores surprise etc.
        if attention_mask is not None:
            mask = attention_mask.to(token_embeds.dtype).unsqueeze(-1)
            token_embeds = token_embeds * mask
        prev = state.get("context_embeds")
        prev_mask = state.get("context_mask")
        if prev is None:
            new = token_embeds
            new_mask = (
                attention_mask if attention_mask is not None
                else torch.ones(token_embeds.shape[0], token_embeds.shape[1],
                                dtype=torch.bool, device=token_embeds.device)
            )
        else:
            new = torch.cat([prev, token_embeds], dim=1)
            window_mask = (
                attention_mask if attention_mask is not None
                else torch.ones(token_embeds.shape[0], token_embeds.shape[1],
                                dtype=torch.bool, device=token_embeds.device)
            )
            new_mask = torch.cat([prev_mask, window_mask], dim=1)
        new_state = dict(state)
        new_state["context_embeds"] = new
        new_state["context_mask"] = new_mask
        return new_state, {}

    def finalize_memory(self, state) -> tuple[Tensor, dict]:
        ctx = state.get("context_embeds")
        mem_mask = state.get("context_mask")
        if ctx is None:
            ctx = torch.zeros(
                state["_B"], 0, self.cfg.d_llama,
                device=state["_device"], dtype=state["_dtype"],
            )
            mem_mask = torch.zeros(state["_B"], 0, dtype=torch.bool,
                                    device=state["_device"])
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=ctx.device, dtype=ctx.dtype,
            ),
            # v5.5: surface the real-token mask so model.py can mask out the
            # padded context positions in Llama's attention mask. Previously
            # padded slots were zero-vectored but still attended-to, letting
            # Llama use them as causal scratch space and contaminating the
            # vanilla_full_context "ceiling" reference.
            "memory_mask": mem_mask,
        }
        return ctx, aux

    def forward(
        self,
        token_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        mask_positions: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        del mask_positions
        if attention_mask is not None:
            mask = attention_mask.to(token_embeds.dtype).unsqueeze(-1)
            token_embeds = token_embeds * mask
        aux = {
            "load_balance_loss": torch.zeros(
                (), device=token_embeds.device, dtype=torch.float32,
            ),
        }
        return token_embeds, aux
