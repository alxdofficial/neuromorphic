"""V8/v9.1 Model — split-scan LM + differentiable memory graph.

Training: single backward pass, single optimizer.
  1. Lower scan + PCM → H_mid, surprise  (backprop)
  2. Memory graph: per-token dynamics     (backprop, TBPTT at segment boundaries)
  3. Inject + upper scan → logits         (backprop)

Memory graph params get gradients through mem_out → inject_memory → loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config
from .lm import V8LM
from .memory_graph import MemoryGraph


class V8Model(nn.Module):
    """Top-level model: LM (backprop) + Memory Graph (backprop)."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config
        self.lm = V8LM(config)
        self.memory = MemoryGraph(
            config, device=torch.device('cpu'), dtype=torch.bfloat16)
        self._states_initialized = False

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        use_memory: bool = True,
        has_reset: bool = False,
    ) -> dict:
        """Process one chunk. Both LM and memory trained by backprop."""
        BS, T = input_ids.shape
        C = self.config.C
        D_cc = self.config.D_cc
        D_mem = self.config.D_mem
        action_every = self.config.action_every
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        if has_reset and reset_mask is not None:
            self._reset_carries(reset_mask)

        # Lower scan + PCM (with grad)
        H_mid, surprise, x, aux_loss = self.lm.forward_scan_lower(input_ids)

        if not use_memory:
            H = self.lm.forward_scan_upper(H_mid, surprise=surprise)
            logits = self.lm.forward_output(H)
            return {"logits": logits, "aux_loss": aux_loss,
                    "surprise": surprise.detach()}

        # Memory graph (differentiable — params on compute graph)
        if not self._states_initialized:
            self.memory.initialize_states(BS)
            self._states_initialized = True

        # Reshape H_mid into memory channels: [BS, T, D] → [BS, T, C_mem, D_mem]
        # C_mem = D // D_mem (e.g., 2048 // 16 = 128 channels of 16 dims)
        C_mem = self.memory.C_mem
        cc_all = H_mid.detach().view(BS, T, C_mem, D_mem)
        n_segments = T // action_every
        cc_segments = cc_all.view(BS, n_segments, action_every, C_mem, D_mem)

        # Process segments — memory output IS on the compute graph
        mem_segments = []
        for seg in range(n_segments):
            seg_out = self.memory.forward_segment(cc_segments[:, seg])
            mem_segments.append(seg_out)

        mem_out = torch.cat(mem_segments, dim=1)  # [BS, T, C_mem, D_mem]

        # Reshape back and inject: mem_out → [BS, T, C, D_cc] for LM compatibility
        mem_out_lm = mem_out.view(BS, T, C, D_cc)
        H_enriched = self.lm.inject_memory(H_mid, mem_out_lm)
        H = self.lm.forward_scan_upper(H_enriched, surprise=surprise)
        logits = self.lm.forward_output(H)

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
        }

    # ================================================================
    # Utilities
    # ================================================================

    def _reset_carries(self, mask: Tensor):
        if hasattr(self.lm, '_carries'):
            for i, h in enumerate(self.lm._carries):
                if h is not None:
                    mask_f = (~mask).to(dtype=h.dtype).unsqueeze(-1)
                    self.lm._carries[i] = h * mask_f

    def initialize_states(self, BS: int, device: torch.device = None):
        self.lm.initialize_carries()
        self.memory.initialize_states(BS)
        self._states_initialized = True

    def detach_states(self):
        self.lm.detach_carries()
        self.memory.detach_states()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return self.lm.param_count()

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())
