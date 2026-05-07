"""MemAdapter — duck-types an HF host as the `lm` object that
`MemoryGraph.walk_segment` expects.

`walk_segment` reads five fields / one method off `lm`:
  - `lm.lm_head`           — Linear(d_lm, vocab)
  - `lm.proj_down`         — Linear(d_mem, d_lm) or None
  - `lm.ln_final`          — object with `.weight` (+ `.bias`, may be None)
  - `lm.mem_head_logits(x) → [BS, T, vocab]` — maps readouts through
    proj_down → norm → lm_head
  - plus `lm.lm_head.weight`, `lm.ln_final.weight/.bias` accessed inline
    for the per-token surprise signal.

For any HF host:
  - `lm.lm_head`   = host.lm_head()
  - `lm.proj_down` = MemInjectLayer.W_out (d_mem → d_lm) so readouts enter
     LM space via the same projection used for the injected residual.
  - `lm.ln_final`  = wrapper that exposes `.weight` / `.bias` from host's
     final norm (RMSNorm for Llama-family, LayerNorm for GPT-2-family).
     `_run_block` takes `use_rmsnorm=host.use_rmsnorm()` so the inline
     norm math matches.

This replaces the old `LlamaMemAdapter`. The old class name is aliased at
the bottom of this file for backward compatibility with any import not
yet updated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.pretrained.hosts.base import HostAdapter


@dataclass
class _LnFinalShim:
    """Tiny object exposing `.weight` and `.bias` in the shape memory.py expects."""
    weight: Tensor
    bias: Tensor | None = None


class MemAdapter(nn.Module):
    """Adapts any HostAdapter to the `lm` interface memory.walk_segment uses."""

    def __init__(self, host: HostAdapter, W_out: nn.Linear):
        super().__init__()
        # List-wrap so the HostAdapter (and its hf_model) is NOT registered
        # as a submodule — the backbone is owned by PretrainedLMWithMemory.
        self._host = [host]
        # W_out is owned by MemInjectLayer; we reference it for the aux-loss
        # projection but don't re-register as a parameter here.
        self._W_out = W_out
        self.rms_eps = host.norm_eps()

        # Duck-typed fields consumed by memory.walk_segment.
        self.lm_head = host.lm_head()
        self.proj_down = W_out
        fn = host.final_norm()
        # RMSNorm has no bias; LayerNorm does. Grab whatever's there.
        bias = getattr(fn, "bias", None)
        self.ln_final = _LnFinalShim(weight=fn.weight, bias=bias)

    @property
    def host(self) -> HostAdapter:
        return self._host[0]

    def mem_head_logits(self, readouts: Tensor) -> Tensor:
        """readouts [BS, T, d_mem] → logits [BS, T, vocab_lm].

        Dtype plumbing: `proj_down` (W_out) is fp32 by design for stable
        Adam updates; `readouts` arrive in the host's dtype (bf16 in
        production). Cast up to fp32 for the projection, then down to
        host dtype for norm + lm_head (both fp32-safe but the host's
        native dtype avoids an extra round trip).
        """
        proj_dt = self.proj_down.weight.dtype
        if readouts.dtype != proj_dt:
            readouts = readouts.to(proj_dt)
        x = self.proj_down(readouts)                       # [BS, T, d_lm]
        fn = self.host.final_norm()
        host_dt = fn.weight.dtype
        if x.dtype != host_dt:
            x = x.to(host_dt)
        x = fn(x)
        return self.lm_head(x)                             # [BS, T, vocab]


# Back-compat alias. Any code still importing LlamaMemAdapter keeps working.
LlamaMemAdapter = MemAdapter
