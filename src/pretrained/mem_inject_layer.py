"""MemInjectLayer — wraps one LlamaDecoderLayer with a memory read/write side-channel.

Forward:
    h_in  = hidden_states from previous layer (shape [BS, T, d_lm])
    h_mem = W_in(h_in)                          # [BS, T, d_mem]
    m     = memory.readout(h_mem, ...)          # [BS, T, d_mem]
    h_inj = h_in + scale * W_out(m)             # [BS, T, d_lm]
    return orig_layer(h_inj, **kw)

`scale` is a per-dim trainable vector (init sqrt(alpha)). When `memory_fn`
is None (smoke/identity mode) this layer is a no-op wrapper around
`orig_layer` — used to verify the replacement doesn't change Llama's output.

Memory is called with the full [BS, T, d_mem] segment. The memory graph
internally handles the per-token LIF, per-4 msg update, per-16 modulator
event clock. Writes to W/decay/hebbian persist on the `memory` module
across forward() calls (carry state — `memory.detach_states()` between
segments).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class MemInjectLayer(nn.Module):
    def __init__(
        self,
        orig_layer: nn.Module,
        d_lm: int,
        d_mem: int,
        scale_init: float,
        memory_fn: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.orig_layer = orig_layer
        self.d_lm = d_lm
        self.d_mem = d_mem
        self.memory_fn = memory_fn

        # Projections into and out of memory space. Trainable. When
        # d_lm == d_mem, init as identity + small noise so the layer starts
        # near a no-op; otherwise Xavier.
        self.W_in = nn.Linear(d_lm, d_mem, bias=False)
        self.W_out = nn.Linear(d_mem, d_lm, bias=False)
        if d_lm == d_mem:
            with torch.no_grad():
                self.W_in.weight.copy_(torch.eye(d_lm))
                self.W_out.weight.copy_(torch.eye(d_lm))
        else:
            nn.init.xavier_uniform_(self.W_in.weight)
            nn.init.xavier_uniform_(self.W_out.weight)

        # Per-dim scale gate. Init sqrt(alpha) = 2.0 by default; tiny init
        # is fine for smoke testing because we verify scale=0 reproduces
        # vanilla bit-for-bit.
        self.scale = nn.Parameter(torch.full((d_lm,), scale_init))

    def set_memory_fn(self, memory_fn: Callable[[Tensor], Tensor] | None):
        """Wire or unwire the memory read/write callback at runtime."""
        self.memory_fn = memory_fn

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        if self.memory_fn is None:
            # Transparent bypass only when the inject residual is provably
            # zero — i.e. scale is all-zero. Any nonzero scale without a
            # memory_fn silently drops the inject and produces outputs the
            # training loop can't distinguish from a correctly-wired run.
            # That's a footgun; error out instead.
            if not (self.scale == 0).all():
                raise RuntimeError(
                    "MemInjectLayer called without memory_fn but scale is "
                    "not all-zero. Either wire memory_fn via "
                    "PretrainedLMWithMemory.forward (which installs a "
                    "closure per call) or pin scale to zero. Silent "
                    "bypass here would produce incorrect training output.")
            return self.orig_layer(hidden_states, *args, **kwargs)

        h_mem = self.W_in(hidden_states)              # [BS, T, d_mem]
        readout = self.memory_fn(h_mem)               # [BS, T, d_mem]
        injected = hidden_states + self.scale * self.W_out(readout)  # [BS, T, d_lm]
        return self.orig_layer(injected, *args, **kwargs)
