"""LlamaHost — covers Llama 3.x, TinyLlama, SmolLM2, Mistral, Qwen2.

All of these use the Llama architecture in HuggingFace (`LlamaForCausalLM`
or very close: Mistral / Qwen2 follow the same attribute layout). HF sets
`config.model_type` to `"llama"`, `"mistral"`, `"qwen2"`, or similar; the
`hosts/__init__.py` factory maps all of them to this adapter.

The `model.model.layers` path is consistent across the family. So is
`model.lm_head` and `model.model.norm`. Memory integration is identical
for all of them — only `inject_layer` and `d_mem` differ per model in the
`PretrainedConfig` factory methods.
"""

from __future__ import annotations

import torch.nn as nn

from src.pretrained.hosts.base import HostAdapter


class LlamaHost(HostAdapter):
    """Adapter for HF models that follow the Llama attribute layout."""

    def layer_list(self) -> nn.ModuleList:
        return self.hf_model.model.layers

    def lm_head(self) -> nn.Module:
        return self.hf_model.lm_head

    def final_norm(self) -> nn.Module:
        return self.hf_model.model.norm

    def norm_eps(self) -> float:
        # Llama, Mistral, Qwen2 all expose `rms_norm_eps` on the HF config.
        # SmolLM2 uses the same name. Fallback to a safe default if absent.
        return getattr(self.hf_model.config, "rms_norm_eps", 1e-5)

    def use_rmsnorm(self) -> bool:
        return True
