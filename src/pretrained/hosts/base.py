"""HostAdapter — abstract interface between a pretrained HF causal LM and
the memory-integration plumbing.

The memory graph needs to plug into a pretrained LM at three points:
  1. Swap one decoder layer (mid-stack) with `MemInjectLayer`.
  2. Reuse the LM's final norm + lm_head to compute the auxiliary
     memory-CE loss (memory's own NTP side-objective).
  3. Freeze or unfreeze the backbone.

Each HF model family stores these at different attribute paths:
    Llama / TinyLlama / SmolLM2 / Mistral / Qwen2:
        model.model.layers, model.model.norm, model.lm_head
    GPT-NeoX (Pythia):
        model.gpt_neox.layers, model.gpt_neox.final_layer_norm, model.embed_out
    GPT-2 / OPT:
        model.transformer.h, model.transformer.ln_f, model.lm_head

Subclasses override the abstract methods. Non-abstract methods provide
sane defaults shared across families (freeze/unfreeze, vocab/hidden size
from config).

No state; every method reaches into `self.hf_model`. Keeping the adapter
stateless means `PretrainedLMWithMemory` can serialize / move to device
via the HF model alone, and the adapter is a thin view over it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class HostAdapter(ABC):
    """Abstract accessor for a HuggingFace causal-LM's memory-integration points."""

    def __init__(self, hf_model: nn.Module):
        self.hf_model = hf_model

    # ------------------------------------------------------------------
    # Abstract — one implementation per HF model family.
    # ------------------------------------------------------------------

    @abstractmethod
    def layer_list(self) -> nn.ModuleList:
        """The ModuleList of decoder layers. Mutating the list (e.g.,
        replacing an entry) rewires the HF model in-place."""
        ...

    @abstractmethod
    def lm_head(self) -> nn.Module:
        """Final Linear that maps hidden states → vocab logits. Typically
        the `lm_head` attribute on Llama-like models, `embed_out` on
        GPT-NeoX, `lm_head` on GPT-2."""
        ...

    @abstractmethod
    def final_norm(self) -> nn.Module:
        """The norm applied to the last hidden state before lm_head.
        RMSNorm for Llama-family, LayerNorm for GPT-2-family."""
        ...

    @abstractmethod
    def norm_eps(self) -> float:
        """eps used by `final_norm`. Memory re-implements the norm inline
        for its aux-loss head, so it needs this value."""
        ...

    def use_rmsnorm(self) -> bool:
        """Whether final_norm is RMSNorm (no bias, no mean-centering).

        Default True — Llama, Mistral, Qwen, TinyLlama, SmolLM2 all use
        RMSNorm. Override to False in GPT-2-family hosts."""
        return True

    # ------------------------------------------------------------------
    # Shared defaults — read off the HF config.
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.hf_model.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.hf_model.config.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.hf_model.config.vocab_size

    # ------------------------------------------------------------------
    # Mutation helpers — used by PretrainedLMWithMemory.
    # ------------------------------------------------------------------

    def replace_layer(self, idx: int, new_layer: nn.Module):
        """Swap layer `idx` for `new_layer` in-place."""
        self.layer_list()[idx] = new_layer

    def freeze_backbone(self):
        """Set requires_grad=False on every parameter of the hosted model.
        MemInjectLayer params added later are not part of `hf_model`."""
        for p in self.hf_model.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.hf_model.parameters():
            p.requires_grad = True
