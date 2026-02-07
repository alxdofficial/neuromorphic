"""
Layer — scan-friendly recurrence with procedural memory.

One Layer per (block, layer). Contains the affine recurrence
h_t = a_t * (carry * h_{t-1}) + b_t  and a PM instance.

Gate inputs: concat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])
giving input_dim = 4*D_h + 1.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .procedural_memory import ProceduralMemory, PMController
from .utils import StateMixin


class Layer(nn.Module, StateMixin):
    _state_tensor_names = ["h"]

    def __init__(self, config: ModelConfig, block_idx: int, layer_idx: int):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.layer_idx = layer_idx
        D_h = config.D_h

        # Procedural memory for this layer
        self.pm = ProceduralMemory(config)
        self.pm_controller = PMController(config)

        # Gate input: x_block + y_pm + y_wm_proj + y_em_proj + surprise
        input_dim = 4 * D_h + 1

        # Scan-friendly gates (no dependence on h_{t-1})
        self.gate_a = nn.Linear(input_dim, D_h)  # sigmoid -> retention
        self.gate_b = nn.Linear(input_dim, D_h)  # tanh -> update

        # Per-layer output projection (spec §7.4)
        self.W_o = nn.Linear(D_h, D_h)

        self.norm = nn.LayerNorm(D_h)

        # Recurrent hidden state (lazily initialized)
        self.h: Tensor = None

    def _lazy_init(self, BS: int, device: torch.device):
        self.h = torch.zeros(BS, self.config.D_h, device=device)

    def step(self, x_block: Tensor, y_pm: Tensor, y_wm_proj: Tensor,
             y_em_proj: Tensor, surprise: Tensor, carry: Tensor,
             collect: bool = False):
        """Process one token through this layer.

        Args:
            x_block: [BS, D_h] — block input
            y_pm: [BS, D_h] — PM output (zeros if disabled)
            y_wm_proj: [BS, D_h] — WM output projected to D_h
            y_em_proj: [BS, D_h] — EM output projected to D_h (zeros if disabled)
            surprise: [BS, 1] — surprise signal
            carry: [BS, 1] — 0 at doc boundaries, 1 otherwise
            collect: bool — if True, return (output, stats_dict)

        Returns:
            output: [BS, D_h] — layer output
            stats: dict (only when collect=True) — gate_a, gate_b, h_norm
        """
        BS = x_block.shape[0]
        device = x_block.device

        if self.h is None:
            self._lazy_init(BS, device)

        # Fuse inputs
        u = torch.cat([x_block, y_pm, y_wm_proj, y_em_proj, surprise], dim=-1)

        # Compute gates (no dependence on h_prev -> scan-friendly)
        a = torch.sigmoid(self.gate_a(u))  # [BS, D_h]
        b = torch.tanh(self.gate_b(u))     # [BS, D_h]

        # Affine recurrence with carry mask for doc boundaries
        self.h = a * (carry * self.h) + b

        # Output projection + residual + LayerNorm (spec §7.4)
        output = self.norm(self.W_o(self.h) + x_block)

        if collect:
            stats = {
                "gate_a": a.detach(),
                "gate_b": b.detach(),
                "h_norm": self.h.detach().norm(dim=-1).mean().item(),
            }
            return output, stats
        return output
