"""V8Model Phase 1 — split-scan + backprop-trained memory graph.

No neuromodulator. No RL. Memory graph parameters (primitives, conn_weights,
decay) are nn.Parameters trained by backprop through K-step gradient windows.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0-3) + PCM (parallel over T)
  2. Memory graph: per-token dynamics with K-step gradient windows (sequential)
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers 4-6) over memory-enriched H (parallel)
  5. Output head → logits
  6. Loss → backprop through upper scan → gate → memory graph → H_mid → lower scan
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config
from .lm import V8LM
from .memory_graph_backprop import MemoryGraphBackprop


class V8ModelBackprop(nn.Module):
    """Phase 1: LM + backprop-trained memory graph. No neuromodulator."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config

        self.lm = V8LM(config)
        self.mem_graph = None  # created in initialize_states

        # Plasticity step counter
        self._step_counter = 0

    def _ensure_memory(self, BS: int, device: torch.device,
                       dtype: torch.dtype):
        if (self.mem_graph is not None
                and self.mem_graph.is_initialized()
                and self.mem_graph.h.shape[0] == BS):
            return
        self.mem_graph = MemoryGraphBackprop(
            self.config, device, dtype).to(device)
        self.mem_graph.initialize(BS)

    @property
    def memory(self):
        return self.mem_graph

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        use_memory: bool = True,
        has_reset: bool = False,
    ) -> dict:
        """Process a full T-token chunk.

        With memory: gradients flow through the memory graph via K-step windows.
        Without memory: standard LM forward (no-memory baseline).
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_mem = self.config.D_mem
        eot_id = self.config.eot_id
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # Reset scan carries at doc boundaries
        if has_reset and reset_mask is not None:
            self._reset_carries(reset_mask)

        # Lower scan + PCM
        H_mid, x, surprise, aux_loss = self.lm.forward_scan_lower(input_ids)

        # No-memory fast path
        if not use_memory:
            H = self.lm.forward_scan_upper(H_mid)
            logits = self.lm.forward_output(H)
            return {
                "logits": logits,
                "aux_loss": aux_loss,
                "surprise": surprise.detach(),
            }

        # Memory graph
        self._ensure_memory(BS, device, dtype)

        if not self.config.lifelong_mode and has_reset and reset_mask is not None:
            self.mem_graph.reset_streams(reset_mask)

        # CC signals from lower scan — NOT detached (gradient flows through!)
        cc_signals = H_mid.view(BS, T, C, D_mem)

        # EOT mask for state reset within chunk
        eot_mask = None
        if not self.config.lifelong_mode:
            eot_at = (input_ids == eot_id)
            eot_mask = torch.zeros(BS, T, dtype=torch.bool, device=device)
            eot_mask[:, 1:] = eot_at[:, :-1]

        # Structural plasticity cadence
        sp_every = self.config.structural_plasticity_every
        self._step_counter += 1
        needs_phi = (sp_every > 0
                     and self._step_counter % sp_every == 0)

        # Run memory graph (backprop-compatible, K-step gradient windows)
        mem_signals = self.mem_graph.forward_chunk(
            cc_signals, eot_mask=eot_mask,
            update_co_activation=needs_phi)

        if needs_phi:
            self.mem_graph.structural_plasticity()

        # Inject memory + upper scan
        H_enriched = self.lm.inject_memory(H_mid, mem_signals)
        H = self.lm.forward_scan_upper(H_enriched)
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

    def initialize_states(self, BS: int, device: torch.device):
        self.lm.initialize_carries()
        lm_dtype = next(self.lm.parameters()).dtype
        self._ensure_memory(BS, device, lm_dtype)

    def detach_states(self):
        self.lm.detach_carries()
        if self.mem_graph is not None:
            self.mem_graph.detach_state()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def mem_param_count(self) -> int:
        if self.mem_graph is None:
            return 0
        return sum(p.numel() for p in self.mem_graph.parameters()
                   if p.requires_grad)
