"""V8/v9 Model — split-scan LM + differentiable memory graph.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0..split-1) + PCM → H_mid, surprise
  2. Memory graph: per-segment differentiable dynamics (TBPTT)
     Per-neuron modulators compute effective params from internal state.
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers split..L-1) → H
  5. Output head → logits

Everything is end-to-end differentiable (single backward pass).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config
from .lm import V8LM
from .memory_graph import MemoryGraph


class V8Model(nn.Module):
    """Top-level model: LM + differentiable Memory Graph."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config

        self.lm = V8LM(config)
        self.memory = MemoryGraph(
            config,
            device=torch.device('cpu'),  # moved to GPU in train.py
            dtype=torch.float32,          # cast in train.py
        )
        self._states_initialized = False

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        use_memory: bool = True,
        has_reset: bool = False,
    ) -> dict:
        """Process a full T-token chunk with split-scan + differentiable memory.

        1. Lower scan → H_mid (parallel over T)
        2. Memory graph: per-segment dynamics (sequential, differentiable)
        3. Inject: H_enriched = H_mid + gate * mem_signals
        4. Upper scan → H (parallel over T)
        5. Output head → logits
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_mem = self.config.D_mem
        action_every = self.config.action_every
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # Reset scan carries at doc boundaries (LM only, not memory graph)
        if has_reset and reset_mask is not None:
            self._reset_carries(reset_mask)

        # ==========================================
        # Lower scan (layers 0..split-1) + PCM
        # ==========================================
        H_mid, surprise, x, aux_loss = self.lm.forward_scan_lower(input_ids)

        # --- No-memory fast path ---
        if not use_memory:
            H = self.lm.forward_scan_upper(H_mid, surprise=surprise)
            logits = self.lm.forward_output(H)
            return {
                "logits": logits,
                "aux_loss": aux_loss,
                "surprise": surprise.detach(),
            }

        # ==========================================
        # Memory graph: differentiable per-segment dynamics
        # ==========================================
        if not self._states_initialized:
            self.memory.initialize_states(BS)
            self._states_initialized = True

        # CC signals (detached — memory graph doesn't backprop into lower scan)
        cc_signals_all = H_mid.detach().view(BS, T, C, D_mem)

        n_segments = T // action_every
        cc_segments = cc_signals_all.view(BS, n_segments, action_every, C, D_mem)

        mem_out_list = []
        h = self.memory.h  # persistent state

        for seg in range(n_segments):
            h_detached = h.detach()  # TBPTT cut
            seg_cc = cc_segments[:, seg]  # [BS, action_every, C, D_mem]

            seg_out, h = self.memory.forward_segment(seg_cc, h_detached)
            mem_out_list.append(seg_out)

        # Store h for next chunk (detached)
        self.memory.h = h.detach()

        # Concatenate segment outputs
        mem_out = torch.cat(mem_out_list, dim=1)  # [BS, T, C, D_mem]

        # ==========================================
        # Upper scan + output
        # ==========================================
        H_enriched = self.lm.inject_memory(H_mid, mem_out)
        H = self.lm.forward_scan_upper(H_enriched, surprise=surprise)
        logits = self.lm.forward_output(H)

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
        }

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lm_param_count(self) -> int:
        return self.lm.param_count()

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters()
                   if p.requires_grad)
