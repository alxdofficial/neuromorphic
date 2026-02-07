"""
Block — parallel processing unit with L sequential layers + 1 EM bank.

One Block per parallel track (B total). Each block processes D_h dimensions,
contains L Layers (each with their own PM), and one shared EpisodicMemory.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .layer import Layer
from .episodic_memory import EpisodicMemory, EMController
from .utils import StateMixin


class Block(nn.Module, StateMixin):
    _state_tensor_names = []  # state lives in child modules

    def __init__(self, config: ModelConfig, block_idx: int):
        super().__init__()
        self.config = config
        self.block_idx = block_idx

        # L sequential layers
        self.layers = nn.ModuleList([
            Layer(config, block_idx, l) for l in range(config.L)
        ])

        # Episodic memory (1 per block)
        self.em = EpisodicMemory(config)
        self.em_controller = EMController(config)

        # Projections from shared D to per-block D_h
        self.W_wm_proj = nn.Linear(config.D, config.D_h, bias=False)
        self.W_em_proj = nn.Linear(config.D, config.D_h, bias=False)

    def step(self, x_block: Tensor, y_wm: Tensor, x_emb: Tensor,
             surprise: Tensor, carry: Tensor, collect: bool = False):
        """Process one token through this block.

        Args:
            x_block: [BS, D_h] — input slice for this block
            y_wm: [BS, D] — shared WM output
            x_emb: [BS, D] — token embedding (for EM retrieval query)
            surprise: [BS, 1] — surprise signal
            carry: [BS, 1] — 0 at doc boundaries, 1 otherwise
            collect: bool — if True, return (h_out, stats_dict)

        Returns:
            h_out: [BS, D_h] — final layer output
            stats: dict (only when collect=True) — per-layer gate stats
        """
        # EM retrieval (if enabled)
        if self.config.em_enabled:
            y_em = self.em.retrieve(x_emb, y_wm)  # [BS, D]
        else:
            y_em = torch.zeros_like(y_wm)

        # Project to D_h
        y_wm_proj = self.W_wm_proj(y_wm)      # [BS, D_h]
        y_em_proj = self.W_em_proj(y_em)        # [BS, D_h]

        # Sequential layers
        x = x_block
        layer_stats = {}
        for l_idx, layer in enumerate(self.layers):
            # PM read (if enabled)
            if self.config.pm_enabled:
                y_pm = layer.pm.apply(x)
            else:
                y_pm = torch.zeros_like(x)

            # Forward through layer
            result = layer.step(x, y_pm, y_wm_proj, y_em_proj, surprise, carry,
                                collect=collect)

            if collect:
                h, lstats = result
                layer_stats[l_idx] = lstats
            else:
                h = result

            # Update eligibility traces (if PM enabled)
            if self.config.pm_enabled:
                layer.pm.update_eligibility(x, h)

            x = h  # next layer's input

        if collect:
            return x, layer_stats
        return x  # final layer output [BS, D_h]

    def commit_pm(self, force_mode: str = "normal"):
        """Trigger PM commits for all layers in this block.

        Args:
            force_mode: "normal" — use controller decisions
                        "force_on" — commit all streams
                        "force_off" — skip all commits
        """
        if force_mode == "force_off":
            return

        for layer in self.layers:
            pm = layer.pm

            # Compute eligibility norm for controller
            if pm.elig_K is not None:
                elig_norm = pm.elig_K.norm(dim=-1).mean(dim=-1)  # [BS]
                pm_usage = pm.pm_a.sum(dim=-1)  # [BS]
            else:
                return

            if force_mode == "force_on":
                BS = elig_norm.shape[0]
                commit_mask = torch.ones(BS, dtype=torch.bool,
                                         device=elig_norm.device)
                lambda_vals = torch.full((BS,), pm.decay,
                                         device=elig_norm.device)
                g = torch.full((BS,), 0.5, device=elig_norm.device)
                pm.commit(commit_mask, lambda_vals, g, None)
            else:
                # Controller decides commit mask and parameters
                # (span_surprise not available here in MVP — use elig_norm proxy)
                commit_mask, lambda_vals, g, slot_logits = \
                    layer.pm_controller.forward(
                        elig_norm, pm_usage, elig_norm  # surprise proxy
                    )
                pm.commit(commit_mask, lambda_vals, g, slot_logits)

    def detach_states(self):
        """Detach all states in child modules."""
        for layer in self.layers:
            layer.detach_states()
            layer.pm.detach_states()
        self.em.detach_states()

    def reset_states(self, mask: Tensor):
        """Reset states for masked streams.

        In lifelong mode (Phase E), only transient state resets:
        h and eligibility traces. PM committed state and EM persist.
        """
        for layer in self.layers:
            layer.reset_states(mask)  # always zeros h
            if self.config.lifelong_mode:
                layer.pm.reset_eligibility(mask)  # only elig_K, elig_V
            else:
                layer.pm.reset_states(mask)  # zeros all PM state

        if not self.config.lifelong_mode:
            self.em.reset_states(mask)  # zeros em_S
        # In lifelong mode: EM fully persists
