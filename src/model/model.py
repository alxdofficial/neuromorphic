"""
NeuromorphicLM (v4) — iterative refinement with cortical columns.

Processes N-token segments through R iterative passes. Each pass, all
B_blocks * C columns process all N tokens in parallel, then PM/EM update.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .block import ColumnBlock
from .utils import runtime_state_dtype


class NeuromorphicLM(nn.Module):
    """v4: Iterative refinement with cortical columns."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding + position
        self.embedding = nn.Embedding(config.vocab_size, config.D)
        self.pos_embedding = nn.Embedding(config.N, config.D)

        # Fan-out: D -> B_blocks * C * D_col
        total_col_dim = config.B_blocks * config.C * config.D_col
        self.fan_out = nn.Linear(config.D, total_col_dim)

        # Blocks
        self.blocks = nn.ModuleList([
            ColumnBlock(i, config) for i in range(config.B_blocks)
        ])

        # Fan-in: B_blocks * C * D_col -> D
        self.fan_in = nn.Linear(total_col_dim, config.D)
        self.ln_final = nn.LayerNorm(config.D)

        # LM head (optionally tied to embedding)
        self.lm_head = nn.Linear(config.D, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Damped mixing parameter (sigmoid -> lambda_mix init)
        # logit(0.5) = 0.0
        init_logit = torch.log(torch.tensor(config.lambda_mix / (1 - config.lambda_mix + 1e-8)))
        self.lambda_logit = nn.Parameter(torch.tensor(float(init_logit)))

    def forward_segment(self, input_ids: Tensor, reset_mask: Tensor | None = None):
        """Process N tokens through R refinement passes.

        input_ids: [BS, N]
        reset_mask: [BS] bool — streams to reset PM/EM before processing

        Returns: (logits [BS, N, vocab], aux_loss scalar)
        """
        if reset_mask is not None and reset_mask.any():
            self._reset_memory(reset_mask)

        BS, N = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)  # [BS, N, D]
        positions = torch.arange(N, device=device)
        x = x + self.pos_embedding(positions)

        # Fan-out to column space
        x_flat = self.fan_out(x)  # [BS, N, B*C*D_col]
        x_blocks = x_flat.view(
            BS, N, self.config.B_blocks, self.config.C, self.config.D_col
        )

        z_hat_prev = [None] * self.config.B_blocks
        aux_loss = torch.tensor(0.0, device=device)
        lam = torch.sigmoid(self.lambda_logit)

        for r in range(self.config.R):
            block_outputs = []
            z_hat_new = []

            for b, block in enumerate(self.blocks):
                x_b = x_blocks[:, :, b]  # [BS, N, C, D_col]
                x_out, z, z_hat, pcm_loss, elig, em_cands = \
                    block.forward_pass(x_b, z_hat_prev[b])

                aux_loss = aux_loss + pcm_loss * self.config.pcm_pred_weight

                # PM/EM update between passes
                block.commit_and_write(elig, em_cands)

                block_outputs.append(x_out)
                z_hat_new.append(z_hat)

            x_new = torch.stack(block_outputs, dim=2)  # [BS, N, B, C, D_col]

            # Damped mixing
            if r > 0:
                x_blocks = (1 - lam) * x_blocks + lam * x_new
            else:
                x_blocks = x_new

            z_hat_prev = z_hat_new

        # Fan-in
        x = x_blocks.reshape(BS, N, -1)  # [BS, N, B*C*D_col]
        x = self.fan_in(x)               # [BS, N, D]
        x = self.ln_final(x)
        logits = self.lm_head(x)         # [BS, N, vocab]

        return logits, aux_loss

    def _reset_memory(self, mask: Tensor):
        """Reset PM/EM for masked streams at doc boundary."""
        for block in self.blocks:
            if not self.config.lifelong_mode:
                block.pm.reset_content(mask)
                block.em.reset_states(mask)

    def initialize_states(self, BS: int, device: torch.device):
        """Pre-allocate runtime state tensors for all blocks."""
        dtype = runtime_state_dtype(device)
        for block in self.blocks:
            block.initialize_states(BS, device, dtype)

    def detach_states(self):
        """TBPTT boundary: detach all PM/EM state."""
        for block in self.blocks:
            block.pm.detach_states()
            block.em.detach_states()

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        # Uses nn.Module's built-in eval mode toggle
        return self.train(False)
