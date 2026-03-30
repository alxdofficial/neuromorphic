"""v10-gnn Model — Lower Scan + GNN Memory Graph + Decoder.

Forward flow per chunk:
  1. Lower scan + PCM → H_inject [BS, T, D_scan]
  2. Memory graph: sequential simulation (1 step/token)
     → word_states [BS, T, num_words, D_scan]
  3. Decoder: cross-attend to word_states → logits [BS, T, vocab]

Single optimizer trains everything jointly.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .lm import LowerScan
from .decoder import SlidingWindowDecoder


class V10Model(nn.Module):
    """Top-level model: Lower Scan + Memory Graph + Decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lm = LowerScan(config)
        self.decoder = SlidingWindowDecoder(
            D_dec=config.D_dec,
            D_scan=config.D_scan,
            n_heads=config.n_heads_dec,
            n_layers=config.L_dec,
            d_ff=config.d_ff_dec,
            W_sliding=config.W_sliding,
            vocab_size=config.vocab_size,
            D_embed=config.D_embed,
            dropout=config.dropout,
        )

        # Tie embedding weights between lower scan and decoder lm_head
        if config.tie_embeddings:
            self.decoder.lm_head.weight = self.lm.embedding.weight

        # Memory graph imported lazily (may not be created yet by agent)
        self._memory = None
        self._states_initialized = False

    @property
    def memory(self):
        if self._memory is None:
            from .memory_graph import MemoryGraph
            self._memory = MemoryGraph(
                self.config,
                device=next(self.parameters()).device,
            )
            # Move to same device/dtype as model
            device = next(self.parameters()).device
            self._memory = self._memory.to(device)
        return self._memory

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        use_memory: bool = True,
    ) -> dict:
        """Process one chunk through the full pipeline.

        Args:
            input_ids: [BS, T]
            target_ids: [BS, T] (unused here, for API compat)
            reset_mask: [BS, T] bool — document boundaries
            use_memory: if False, skip memory graph (baseline mode)

        Returns:
            dict with 'logits', 'aux_loss'
        """
        BS, T = input_ids.shape

        # Detect internal EOS for reset
        if reset_mask is None:
            eos_positions = (input_ids == self.config.eot_id)
            internal_reset = torch.zeros_like(eos_positions)
            internal_reset[:, 1:] = eos_positions[:, :-1]
            if eos_positions.any():
                reset_mask = internal_reset

        # 1. Lower scan + PCM → H_inject
        H_inject, aux_loss = self.lm(input_ids, reset_mask=reset_mask)

        if not use_memory:
            # Baseline: decoder gets H_inject directly as word_states
            # Reshape to [BS, T, num_words, D_scan] by repeating
            word_states = H_inject.unsqueeze(2).expand(
                -1, -1, self.config.num_words, -1)
            logits = self.decoder(word_states)
            return {"logits": logits, "aux_loss": aux_loss}

        # 2. Memory graph simulation
        mg = self.memory
        if not self._states_initialized:
            mg.initialize_states(BS)
            self._states_initialized = True

        word_states = mg.forward_segment(H_inject)  # [BS, T, num_words, D_scan]

        # 3. Decoder → logits
        logits = self.decoder(word_states)

        return {"logits": logits, "aux_loss": aux_loss}

    def initialize_states(self, BS: int):
        """Initialize all runtime state."""
        self.lm.initialize_carries()
        self.memory.initialize_states(BS)
        self._states_initialized = True

    def detach_states(self):
        """Detach runtime state at chunk boundaries."""
        self.lm.detach_carries()
        if self._states_initialized:
            self.memory.detach_states()
            if self.config.structural_plasticity:
                self.memory.rewire_connections()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return sum(p.numel() for p in self.lm.parameters())

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())

    def decoder_param_count(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters())
