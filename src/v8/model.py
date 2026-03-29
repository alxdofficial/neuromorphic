"""v9-backprop Model — split-scan LM + differentiable memory graph.

Training flow per chunk:
  1. Lower scan + PCM → H_mid, surprise  (backprop)
  2. Memory graph: per-token dynamics     (backprop through modulator + dendrites)
     - Modulator predicts w_conn, decay, primitives (once per segment)
     - Token loop: gather → weight → dendritic tree → inject → integrate → message
     - Readout: average neuron replicas → D_lm
  3. Inject + upper scan → logits         (backprop)

Single optimizer trains LM + memory graph jointly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Process one chunk. Everything trained by backprop."""
        BS, T = input_ids.shape
        D = self.config.D
        action_every = self.config.action_every
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        if has_reset and reset_mask is not None:
            self._reset_carries(reset_mask)

        # Build per-position reset mask for internal document boundaries.
        # If input_ids[b, t] == eot_id, the NEXT position (t+1) should
        # start with a fresh recurrent state.  Position 0 is covered by
        # the chunk-boundary reset above.
        eot_id = self.config.eot_id
        eos_positions = (input_ids == eot_id)  # [BS, T]
        has_internal_eos = eos_positions.any().item()
        if has_internal_eos:
            # Shift right: reset at t+1 when input_ids[t] == eot_id.
            # Position 0 gets False (already handled by chunk-boundary reset).
            internal_reset = torch.zeros_like(eos_positions)
            internal_reset[:, 1:] = eos_positions[:, :-1]
            # Also include chunk-boundary resets at position 0
            if has_reset and reset_mask is not None:
                internal_reset[:, 0] = internal_reset[:, 0] | reset_mask
        else:
            internal_reset = None

        # Lower scan + PCM (with grad)
        H_mid, surprise, x, aux_loss = self.lm.forward_scan_lower(
            input_ids, reset_mask=internal_reset)

        if not use_memory:
            H = self.lm.forward_scan_upper(
                H_mid, surprise=surprise, reset_mask=internal_reset)
            logits = self.lm.forward_output(H)
            return {"logits": logits, "aux_loss": aux_loss,
                    "surprise": surprise.detach()}

        # Memory graph (differentiable through modulator + dendrites)
        if not self._states_initialized:
            self.memory.initialize_states(BS)
            self._states_initialized = True

        # Reset memory state for batch elements that hit a document
        # boundary anywhere in the chunk.  Memory is long-range state,
        # but carrying it across documents would leak context.
        if has_internal_eos:
            batch_has_eos = eos_positions.any(dim=1)  # [BS]
            if batch_has_eos.any():
                self._reset_memory_states(batch_has_eos)

        # Detach H_mid for memory graph input — lower scan gets its own
        # gradient path through the CE loss, memory graph provides a second
        # gradient channel through inject_memory
        cc_all = H_mid.detach()  # [BS, T, D]
        n_segments = T // action_every

        mem_out_segs = []
        for seg in range(n_segments):
            t0 = seg * action_every
            t1 = t0 + action_every
            seg_cc = cc_all[:, t0:t1]  # [BS, T_seg, D]

            # Memory forward (on compute graph through modulator/dendrites)
            seg_out = self.memory.forward_segment(seg_cc)  # [BS, T_seg, D]
            mem_out_segs.append(seg_out)

        mem_out = torch.cat(mem_out_segs, dim=1)  # [BS, T, D]

        # Upper scan + output (with grad)
        H_enriched = self.lm.inject_memory(H_mid, mem_out)
        H = self.lm.forward_scan_upper(
            H_enriched, surprise=surprise, reset_mask=internal_reset)
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

    def _reset_memory_states(self, mask: Tensor):
        """Zero memory graph runtime state for batch elements where mask is True.

        Args:
            mask: [BS] bool — True for batch elements to reset.
        """
        mg = self.memory
        if not mg._initialized:
            return
        # mask_f: [BS, 1, 1] keep-mask (1 = keep, 0 = reset)
        keep = (~mask).to(dtype=mg.h.dtype)
        keep_3d = keep.unsqueeze(-1).unsqueeze(-1)  # [BS, 1, 1]
        keep_2d = keep.unsqueeze(-1)  # [BS, 1]
        with torch.no_grad():
            mg.h = mg.h * keep_3d
            mg.prev_messages = mg.prev_messages * keep_3d
            mg.w_conn = mg.w_conn * keep_3d
            mg.primitives_state = mg.primitives_state * keep_3d
            mg.decay_logit = mg.decay_logit * keep_2d
            mg.hebbian_traces = mg.hebbian_traces * keep_3d

    def initialize_states(self, BS: int, device: torch.device = None):
        self.lm.initialize_carries()
        self.memory.initialize_states(BS)
        self._states_initialized = True

    def detach_states(self):
        self.lm.detach_carries()
        self.memory.detach_states()
        # Structural plasticity runs between chunks (after backward)
        if hasattr(self.memory, 'rewire_connections'):
            self.memory.rewire_connections()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return self.lm.param_count()

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())
