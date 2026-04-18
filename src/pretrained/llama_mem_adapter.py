"""Adapter that duck-types a Llama model as the `lm` object expected by
`MemoryGraph.forward_segment`.

`forward_segment` reads five fields / one method off `lm`:
  - `lm.lm_head`           — Linear(d_lm, vocab)
  - `lm.proj_down`         — Linear(d_mem, d_lm) or None
  - `lm.ln_final`          — object with `.weight` (+ `.bias`, may be None)
  - `lm.mem_head_logits(x) → [BS, T, vocab]` — maps readouts through
    proj_down → norm → lm_head
  - plus `lm.lm_head.weight`, `lm.ln_final.weight/.bias` accessed inline
    for the per-token surprise signal.

For the Llama integration:
  - `lm.lm_head`  = Llama's lm_head
  - `lm.proj_down` = MemInjectLayer.W_out (d_mem → d_lm) so readouts enter
     LM space via the same projection used for the injected residual.
  - `lm.ln_final` = wrapper that exposes `.weight` from Llama's RMSNorm and
     `.bias = None`. The `_run_block` code path is told to use RMSNorm via
     `forward_segment(..., use_rmsnorm=True, rms_eps=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class _LnFinalShim:
    """Tiny object exposing `.weight` and `.bias` in a way memory.py expects."""
    weight: Tensor
    bias: Tensor | None = None


class LlamaMemAdapter(nn.Module):
    def __init__(self, llama: nn.Module, W_out: nn.Linear, rms_eps: float):
        super().__init__()
        self._llama = [llama]               # list-wrap to hide from Module registration
        self._W_out = W_out                 # not registered here; owned by MemInjectLayer
        self.rms_eps = rms_eps

        # Duck-typed fields consumed by memory.forward_segment.
        self.lm_head = llama.lm_head
        self.proj_down = W_out              # d_mem → d_lm; shares weights with MemInjectLayer
        self.ln_final = _LnFinalShim(
            weight=llama.model.norm.weight, bias=None)

    @property
    def llama(self) -> nn.Module:
        return self._llama[0]

    def mem_head_logits(self, readouts: Tensor) -> Tensor:
        """readouts [BS, T, d_mem] → logits [BS, T, vocab_lm].

        `proj_down` is W_out (fp32 by design, for stable Adam updates) but
        readouts arrive in Llama's dtype (bf16 in production). Cast to
        match the projection weight, then to Llama dtype for the rest of
        the head so the norm + lm_head run in their native dtypes.
        """
        proj_dt = self.proj_down.weight.dtype
        if readouts.dtype != proj_dt:
            readouts = readouts.to(proj_dt)
        x = self.proj_down(readouts)                       # [BS, T, d_lm]
        llama_dt = self.llama.model.norm.weight.dtype
        if x.dtype != llama_dt:
            x = x.to(llama_dt)
        x = self.llama.model.norm(x)
        return self.lm_head(x)                             # [BS, T, vocab]
