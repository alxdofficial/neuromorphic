"""Top-level Model: LM + MemoryGraph with interleaved PCM.

Training flow per chunk:
  1. Lower scan → H_mid
  2. Memory graph (with interleaved PCM): forward_segment(H_mid) → mem_out, pcm_loss
  3. H_enriched = H_mid + mem_scale * mem_out (note: surprise augmentation happens inside memory)
  4. Upper scan → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import Config
from .lm import LM
from .memory import MemoryGraph


class Model(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        config.validate()
        self.config = config
        self.lm = LM(config)
        self.memory = MemoryGraph(config)
        self._initialized = False

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        use_memory: bool = True,
    ) -> dict:
        BS, T = input_ids.shape
        device = input_ids.device

        if not self._initialized:
            self.memory.initialize_states(BS, device)
            self._initialized = True

        # Build LM reset mask: reset scan carry at positions after EOT tokens.
        # Memory does NOT reset (lifelong). Only the LM scan resets.
        eos_positions = (input_ids == self.config.eot_id)  # [BS, T]
        internal_reset = torch.zeros_like(eos_positions)
        internal_reset[:, 1:] = eos_positions[:, :-1]  # reset at t+1 after EOT
        reset_mask = internal_reset if internal_reset.any() else None

        # 1. Lower scan with EOT reset
        H_mid = self.lm.forward_scan_lower(input_ids, reset_mask=reset_mask)

        # 2. Memory graph with interleaved PCM (no reset — lifelong memory)
        if use_memory:
            augment_fn = self.lm.augment_single

            mem_out, pcm_loss = self.memory.forward_segment(
                H_mid.detach(), augment_fn=augment_fn)
            H_enriched = H_mid + self.lm.mem_scale * mem_out.to(H_mid.dtype)
            aux_loss = self.config.pcm_pred_weight * pcm_loss
        else:
            H_enriched = H_mid
            aux_loss = torch.tensor(0.0, device=device)

        # 3. Upper scan with EOT reset
        H = self.lm.forward_scan_upper(H_enriched, reset_mask=reset_mask)

        # 4. Output
        logits = self.lm.forward_output(H)

        result = {"logits": logits, "aux_loss": aux_loss}

        if target_ids is not None:
            ce_per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape(BS, T)
            valid_mask = (input_ids != self.config.eot_id).float()
            valid_count = valid_mask.sum().clamp(min=1.0)
            ce_loss = (ce_per_token * valid_mask).sum() / valid_count
            result["ce_loss"] = ce_loss
            result["loss"] = ce_loss + aux_loss

        return result

    def detach_states(self):
        """Call between chunks for TBPTT."""
        self.lm.detach_carries()
        self.memory.detach_states()

    def runtime_state_dict(self) -> dict:
        return {
            "initialized": self._initialized,
            "lm": self.lm.runtime_state_dict(),
            "memory": self.memory.runtime_state_dict(),
        }

    def load_runtime_state(self, state: dict):
        if not state:
            return
        self.lm.load_runtime_state(state.get("lm", {}))
        self.memory.load_runtime_state(state.get("memory", {}))
        self._initialized = state.get("initialized", self.memory.is_initialized)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return sum(p.numel() for p in self.lm.parameters())

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())
