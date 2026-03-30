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

        # Create memory graph eagerly (not lazily) so .to() works properly
        from .memory_graph import MemoryGraph
        self.memory = MemoryGraph(config, device=torch.device('cpu'))

        self._states_initialized = False

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
            reset_mask: [BS] or [BS, T] bool — document boundaries
            use_memory: if False, skip memory graph (baseline mode)

        Returns:
            dict with 'logits', 'aux_loss'
        """
        BS, T = input_ids.shape
        device = input_ids.device
        eot_id = self.config.eot_id

        # Build internal reset mask: [BS, T] bool
        # Merge chunk-boundary reset (from caller) with internal EOS detection
        eos_positions = (input_ids == eot_id)
        internal_reset = torch.zeros(BS, T, dtype=torch.bool, device=device)
        internal_reset[:, 1:] = eos_positions[:, :-1]

        # Merge with caller-provided reset_mask
        if reset_mask is not None:
            if reset_mask.dim() == 1:
                # [BS] → mark position 0 for reset
                internal_reset[:, 0] |= reset_mask
            else:
                # [BS, T] → merge
                internal_reset |= reset_mask

        scan_reset = internal_reset if internal_reset.any() else None

        # 1. Lower scan + PCM → H_inject
        H_inject, aux_loss = self.lm(input_ids, reset_mask=scan_reset)

        if not use_memory:
            # Baseline: decoder gets H_inject directly as word_states
            word_states = H_inject.unsqueeze(2).expand(
                -1, -1, self.config.num_words, -1)
            logits = self.decoder(word_states)
            return {"logits": logits, "aux_loss": aux_loss}

        # 2. Memory graph simulation
        if not self._states_initialized:
            self.memory.initialize_states(BS)
            self._states_initialized = True

        # Reset memory state for batch elements with EOS
        if eos_positions.any():
            self._reset_memory_for_eos(eos_positions)

        word_states = self.memory.forward_segment(H_inject)

        # 3. Decoder → logits
        # Build decoder reset mask for document boundaries
        # The decoder's self-attention should not attend across EOS
        logits = self.decoder(word_states, reset_mask=internal_reset)

        return {"logits": logits, "aux_loss": aux_loss}

    def _reset_memory_for_eos(self, eos_positions: Tensor):
        """Reset memory state for batch elements containing EOS.

        Args:
            eos_positions: [BS, T] bool — True where input_ids == eot_id
        """
        # Any batch element with an EOS gets its memory reset
        has_eos = eos_positions.any(dim=1)  # [BS]
        if not has_eos.any():
            return

        mg = self.memory
        mask = has_eos.to(dtype=mg.h.dtype).unsqueeze(-1)  # [BS, 1]

        with torch.no_grad():
            # Zero out h and messages for batch elements with EOS
            mg.h = mg.h * (1 - mask.unsqueeze(-1))
            mg.messages = mg.messages * (1 - mask.unsqueeze(-1))

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
