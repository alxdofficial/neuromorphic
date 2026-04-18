"""Top-level Model: LM + MemoryGraph with weight-tied memory-prediction head.

Training flow per chunk:
  1. Lower scan → H_mid
  2. Memory graph: forward_segment(H_mid, input_ids, lm) → readouts, mem_pred_loss
     (memory head = lm_head(readout), weight-tied; trains memory to carry
     information useful for predicting tokens; its per-token CE is also the
     live surprise signal fed to the modulator.)
  3. H_enriched = H_mid + mem_scale * readouts
  4. Upper scan → logits
  5. total_loss = ce_loss + mem_pred_weight * mem_pred_loss
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
        prev_token: Tensor | None = None,
    ) -> dict:
        BS, T = input_ids.shape
        device = input_ids.device

        # Re-initialize memory state if not yet initialized, OR if the loaded
        # runtime state has a different batch size than the current batch.
        # The latter happens when resuming phase 1 from a phase-2 checkpoint
        # (phase 2 typically runs at smaller BS than phase 1) or vice versa.
        needs_init = (
            not self._initialized
            or not hasattr(self.memory, "h")
            or self.memory.h.shape[0] != BS
        )
        if needs_init:
            self.memory.initialize_states(BS, device)
            self._initialized = True
            # LM scan carries are also BS-shaped — drop them so the next lower
            # scan starts fresh at the correct BS.
            self.lm._carries = [None] * self.config.L_total

        # Build LM reset mask: reset scan carry at positions following EOT tokens.
        # Memory does NOT reset (lifelong). Only the LM scan resets.
        # prev_token is the last input token from the previous chunk — if it was
        # EOT, position 0 of this chunk must be reset.
        eos_positions = (input_ids == self.config.eot_id)  # [BS, T]
        reset_mask = torch.zeros_like(eos_positions)
        reset_mask[:, 1:] = eos_positions[:, :-1]  # reset at t+1 after in-chunk EOT
        if prev_token is not None:
            prev_token = prev_token.to(device)
            reset_mask[:, 0] = (prev_token == self.config.eot_id)
        if not reset_mask.any():
            reset_mask = None

        # 1. Lower scan with EOT reset
        H_mid = self.lm.forward_scan_lower(input_ids, reset_mask=reset_mask)

        # 2. Memory graph (no reset — lifelong memory). The memory head is
        #    weight-tied to lm_head; its CE loss trains memory to carry info
        #    useful for predicting tokens, and its per-token value is the
        #    live surprise signal the modulator uses.
        if use_memory:
            readouts, mem_pred_loss = self.memory.forward_segment(
                H_mid.detach(), input_ids, self.lm, prev_token=prev_token)
            H_enriched = H_mid + self.lm.mem_scale * readouts.to(H_mid.dtype)
            aux_loss = self.config.mem_pred_weight * mem_pred_loss
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
            # Mask positions where the TARGET is EOT — i.e. don't train the
            # model to predict EOT as an output. Masking on input_ids would
            # zero out the wrong position (the one after EOT), defeating the
            # intent.
            valid_mask = (target_ids != self.config.eot_id).float()
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
