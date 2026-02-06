"""
NeuromorphicLM — top-level model module.

Combines embedding, working memory, B parallel blocks, and LM head.
Processes one token at a time (online, no [BS, T, vocab] materialization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .working_memory import WorkingMemory
from .block import Block
from .utils import StateMixin


class NeuromorphicLM(nn.Module, StateMixin):
    _state_tensor_names = ["surprise"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.D)
        self.wm = WorkingMemory(config)
        self.blocks = nn.ModuleList([
            Block(config, b) for b in range(config.B)
        ])
        self.lm_head = nn.Linear(config.D, config.vocab_size, bias=False)

        # Input projection: D -> D (split across blocks after)
        self.W_in = nn.Linear(config.D, config.D, bias=False)

        # Surprise signal (lazily initialized)
        self.surprise: Tensor = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward_one_token(self, input_id: Tensor, reset_mask: Tensor,
                          collect: bool = False):
        """Process one token through the full model.

        Args:
            input_id: [BS] — single token IDs
            reset_mask: [BS] bool — True for streams at doc boundary
            collect: bool — if True, return (logits, x_emb, y_wm, stats_dict)

        Returns:
            logits: [BS, vocab]
            x_emb: [BS, D] — token embedding
            y_wm: [BS, D] — working memory output
            stats: dict (only when collect=True) — per-block, per-layer gate stats
        """
        BS = input_id.shape[0]
        device = input_id.device

        # Initialize surprise on first call
        if self.surprise is None:
            self.surprise = torch.zeros(BS, 1, device=device)

        # Reset states for masked streams
        if reset_mask.any():
            self.reset_at_doc_boundary(reset_mask)

        # Embed token
        x = self.embedding(input_id)  # [BS, D]

        # Working memory
        y_wm = self.wm.step(x, reset_mask)  # [BS, D]

        # Split input across blocks
        x_proj = self.W_in(x)  # [BS, D]
        x_blocks = x_proj.view(BS, self.config.B, self.config.D_h)  # [BS, B, D_h]

        # Carry mask: 0 at doc boundaries, 1 otherwise
        carry = (~reset_mask).float().unsqueeze(-1)  # [BS, 1]

        # Process each block
        h_blocks = []
        block_stats = {}
        for b, block in enumerate(self.blocks):
            result = block.step(x_blocks[:, b], y_wm, x, self.surprise, carry,
                                collect=collect)
            if collect:
                h_b, bstats = result
                block_stats[b] = bstats
            else:
                h_b = result
            h_blocks.append(h_b)

        # Merge block outputs
        h_final = torch.cat(h_blocks, dim=-1)  # [BS, D]

        # LM head
        logits = self.lm_head(h_final)  # [BS, vocab]

        if collect:
            return logits, x, y_wm, block_stats
        return logits, x, y_wm

    def update_surprise(self, logits: Tensor, target: Tensor):
        """Update surprise signal from teacher-forced target.

        Args:
            logits: [BS, vocab] — model output
            target: [BS] — target token ids
        """
        with torch.no_grad():
            logp = F.log_softmax(logits, dim=-1)
            self.surprise = -logp.gather(-1, target.unsqueeze(-1))  # [BS, 1]

    def commit_at_boundary(self, force_mode: str = "normal"):
        """Called every P tokens. Triggers PM commits + EM writes.

        Args:
            force_mode: "normal" — use controller decisions
                        "force_on" — commit all streams
                        "force_off" — skip all commits
        """
        for block in self.blocks:
            if self.config.pm_enabled:
                block.commit_pm(force_mode=force_mode)
            # EM writes are handled by the trainer (needs candidate buffers)

    def reset_at_doc_boundary(self, mask: Tensor):
        """Per-stream reset of all memory states.

        Args:
            mask: [BS] bool — True for streams to reset
        """
        # Reset surprise for masked streams
        if self.surprise is not None:
            self.surprise = self.surprise * (~mask).float().unsqueeze(-1)

        # Reset all blocks
        for block in self.blocks:
            block.reset_states(mask)

        # WM resets are handled internally by wm.step()

    def detach_states(self):
        """TBPTT boundary: detach all recurrent states."""
        if self.surprise is not None:
            self.surprise = self.surprise.detach()

        self.wm.detach_states()
        for block in self.blocks:
            block.detach_states()

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
