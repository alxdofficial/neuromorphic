"""Top-level Model: LM + MemoryGraph.

Training flow per chunk:
  1. Lower scan + PCM → H_mid, surprise
  2. Augment: split_mlp(H_mid, surprise) → H_aug
  3. Memory graph: forward_segment(H_aug.detach()) → mem_out
  4. Inject: H_aug + mem_scale * mem_out → H_enriched
  5. Upper scan → logits
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
        self._tokens_since_rewire = 0

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        use_memory: bool = True,
    ) -> dict:
        BS, T = input_ids.shape
        device = input_ids.device

        # Initialize memory on first call (MUST come before resets)
        if not self._initialized:
            self.memory.initialize_states(BS, device)
            self._initialized = True

        # 1. Lower scan + PCM
        H_mid, surprise, aux_loss = self.lm.forward_scan_lower(
            input_ids, reset_mask=None)

        # 2. Augment
        H_aug = self.lm.augment(H_mid, surprise)

        # 3. Memory graph
        if use_memory:
            mem_out = self.memory.forward_segment(H_aug.detach())
            H_enriched = self.lm.inject_memory(H_aug, mem_out)
        else:
            H_enriched = H_aug

        # 4. Upper scan
        H = self.lm.forward_scan_upper(H_enriched, reset_mask=None)

        # 5. Output
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

        self._tokens_since_rewire += self.config.T
        if (self.config.structural_plasticity
                and self._tokens_since_rewire >= self.config.plasticity_interval):
            self.memory.rewire_connections()
            self._tokens_since_rewire = 0

    def runtime_state_dict(self) -> dict:
        return {
            "initialized": self._initialized,
            "tokens_since_rewire": self._tokens_since_rewire,
            "lm": self.lm.runtime_state_dict(),
            "memory": self.memory.runtime_state_dict(),
        }

    def load_runtime_state(self, state: dict):
        if not state:
            return
        self.lm.load_runtime_state(state.get("lm", {}))
        self.memory.load_runtime_state(state.get("memory", {}))
        self._initialized = state.get("initialized", self.memory.is_initialized)
        self._tokens_since_rewire = state.get("tokens_since_rewire", 0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return sum(p.numel() for p in self.lm.parameters())

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())
