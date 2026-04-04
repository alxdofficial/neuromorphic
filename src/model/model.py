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

        # EOS-based resets
        eos_positions = (input_ids == self.config.eot_id)
        internal_reset = torch.zeros_like(eos_positions)
        internal_reset[:, 1:] = eos_positions[:, :-1]

        batch_has_eos = eos_positions.any(dim=1)
        if batch_has_eos.any():
            self.memory.reset_states(batch_has_eos)
            self.lm.reset_carries(batch_has_eos)

        reset_mask = internal_reset if internal_reset.any() else None

        # 1. Lower scan + PCM
        H_mid, surprise, aux_loss = self.lm.forward_scan_lower(
            input_ids, reset_mask=reset_mask)

        # 2. Augment
        H_aug = self.lm.augment(H_mid, surprise)

        # 3. Memory graph
        if use_memory:
            mem_out = self.memory.forward_segment(H_aug.detach())
            H_enriched = self.lm.inject_memory(H_aug, mem_out)
        else:
            H_enriched = H_aug

        # 4. Upper scan
        H = self.lm.forward_scan_upper(H_enriched, reset_mask=reset_mask)

        # 5. Output
        logits = self.lm.forward_output(H)

        result = {"logits": logits, "aux_loss": aux_loss}

        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            result["loss"] = loss + aux_loss

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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return sum(p.numel() for p in self.lm.parameters())

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())
