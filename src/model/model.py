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
from .decoder import SpatialDecoder
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

        # Spatial decoder (hierarchical aggregation + deep cross-attention)
        self.spatial_decoder = SpatialDecoder(config) if config.snapshot_enabled else None

        # Surprise signal (lazily initialized)
        self.surprise: Tensor = None

        # Initialize weights
        self._init_weights()

        # Small-init the decoder output_proj after global init so it starts
        # near-identity (h_final + small_noise) while still allowing gradient flow.
        # Zero-init would kill ALL gradients through the decoder (chain rule: grad × 0 = 0).
        if self.spatial_decoder is not None:
            nn.init.normal_(self.spatial_decoder.output_proj.weight, std=0.01)

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

        # Respect reset_on_doc_boundary across all reset-related paths.
        if self.config.reset_on_doc_boundary:
            effective_reset_mask = reset_mask
        else:
            effective_reset_mask = torch.zeros_like(reset_mask)

        # Reset states for masked streams when doc-boundary reset is enabled.
        if effective_reset_mask.any():
            self.reset_at_doc_boundary(effective_reset_mask)

        # Embed token
        x = self.embedding(input_id)  # [BS, D]

        # Working memory
        if self.config.wm_enabled:
            y_wm = self.wm.step(x, effective_reset_mask)  # [BS, D]
        else:
            y_wm = torch.zeros_like(x)

        # Split input across blocks
        x_proj = self.W_in(x)  # [BS, D]
        x_blocks = x_proj.view(BS, self.config.B, self.config.D_h)  # [BS, B, D_h]

        # Carry mask: 0 at doc boundaries, 1 otherwise
        carry = (~effective_reset_mask).float().unsqueeze(-1)  # [BS, 1]

        # Process each block
        snapshot = self.config.snapshot_enabled
        h_blocks = []
        block_layer_outputs = [] if snapshot else None
        block_stats = {}
        for b, block in enumerate(self.blocks):
            result = block.step(x_blocks[:, b], y_wm, x, self.surprise, carry,
                                collect=collect, return_layers=snapshot)
            if collect and snapshot:
                h_b, bstats, layers_b = result
                block_stats[b] = bstats
                block_layer_outputs.append(layers_b)
            elif collect:
                h_b, bstats = result
                block_stats[b] = bstats
            elif snapshot:
                h_b, layers_b = result
                block_layer_outputs.append(layers_b)
            else:
                h_b = result
            h_blocks.append(h_b)

        # Merge block outputs
        h_final = torch.cat(h_blocks, dim=-1)  # [BS, D]

        # Spatial decoder or direct LM head
        if snapshot:
            pm_summary = self._compute_pm_summary(BS, device)
            em_summary = self._compute_em_summary(BS, device)
            h_decoded = self.spatial_decoder(
                block_layer_outputs, pm_summary, em_summary, y_wm, h_final,
            )
            logits = self.lm_head(h_decoded)  # [BS, vocab]
        else:
            logits = self.lm_head(h_final)  # [BS, vocab]

        if collect:
            return logits, x, y_wm, block_stats
        return logits, x, y_wm

    def _compute_pm_summary(self, BS: int, device: torch.device) -> Tensor:
        """Strength-weighted readout of PM slots, averaged across all instances.

        Returns: [BS, D_h] — zero vector if PM is disabled or uninitialized.
        """
        if not self.config.pm_enabled:
            return torch.zeros(BS, self.config.D_h, device=device)
        readouts = []
        for block in self.blocks:
            for layer in block.layers:
                pm = layer.pm
                if pm.pm_a is not None and pm.pm_V is not None:
                    weights = pm.pm_a.unsqueeze(-1)  # [BS, r, 1]
                    denom = pm.pm_a.sum(dim=1, keepdim=True) + 1e-8  # [BS, 1]
                    readout = (weights * pm.pm_V).sum(dim=1) / denom  # [BS, D_h]
                    readouts.append(readout)
        if readouts:
            return torch.stack(readouts, dim=0).mean(dim=0)  # [BS, D_h]
        return torch.zeros(BS, self.config.D_h, device=device)

    def _compute_em_summary(self, BS: int, device: torch.device) -> Tensor:
        """Strength-weighted readout of EM slots, averaged across all instances.

        Returns: [BS, D_em] — zero vector if EM is disabled or uninitialized.
        """
        if not self.config.em_enabled:
            return torch.zeros(BS, self.config.D_em, device=device)
        readouts = []
        for block in self.blocks:
            em = block.em
            if em.em_S is not None and em.em_V is not None:
                weights = em.em_S.unsqueeze(-1)  # [BS, M, 1]
                denom = em.em_S.sum(dim=1, keepdim=True) + 1e-8  # [BS, 1]
                readout = (weights * em.em_V).sum(dim=1) / denom  # [BS, D_em]
                readouts.append(readout)
        if readouts:
            return torch.stack(readouts, dim=0).mean(dim=0)  # [BS, D_em]
        return torch.zeros(BS, self.config.D_em, device=device)

    def update_surprise(self, logits: Tensor, target: Tensor, mask: Tensor = None):
        """Update surprise signal from teacher-forced target.

        Args:
            logits: [BS, vocab] — model output
            target: [BS] — target token ids
            mask: [BS] bool — optional update mask; masked-out streams get 0
        """
        with torch.no_grad():
            logp = F.log_softmax(logits, dim=-1)
            next_surprise = -logp.gather(-1, target.unsqueeze(-1))  # [BS, 1]
            if mask is not None:
                if mask.dtype is not torch.bool:
                    mask = mask.bool()
                next_surprise = next_surprise * mask.unsqueeze(-1).float()
            self.surprise = next_surprise

    def commit_at_boundary(self, force_mode: str = "normal",
                           span_surprise: Tensor = None):
        """Called every P tokens. Triggers PM commits + EM writes.

        Args:
            force_mode: "normal" — use controller decisions
                        "force_on" — commit all streams
                        "force_off" — skip all commits
            span_surprise: [BS] — mean surprise over span (for PM controller)
        """
        commit_info = {}
        for b_idx, block in enumerate(self.blocks):
            if self.config.pm_enabled:
                commit_info[b_idx] = block.commit_pm(
                    force_mode=force_mode, span_surprise=span_surprise
                )
            # EM writes are handled by the trainer (needs candidate buffers)
        return commit_info

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

    def rl_parameters(self):
        """Yield neuromodulator MLP parameters (for separate RL optimizer)."""
        for name, param in self.named_parameters():
            if "neuromodulator" in name:
                yield param

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
