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
from .episodic_memory import EpisodicMemory, EMNeuromodulator
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
        self.em_neuromodulator = EMNeuromodulator(config)

        # Projections from shared D to per-block D_h
        self.W_wm_proj = nn.Linear(config.D, config.D_h, bias=False)
        self.W_em_proj = nn.Linear(config.D, config.D_h, bias=False)

    def step(self, x_block: Tensor, y_wm: Tensor, x_emb: Tensor,
             surprise: Tensor, carry: Tensor, collect: bool = False,
             return_layers: bool = False):
        """Process one token through this block.

        Args:
            x_block: [BS, D_h] — input slice for this block
            y_wm: [BS, D] — shared WM output
            x_emb: [BS, D] — token embedding (for EM retrieval query)
            surprise: [BS, 1] — surprise signal
            carry: [BS, 1] — 0 at doc boundaries, 1 otherwise
            collect: bool — if True, include per-layer gate stats
            return_layers: bool — if True, include stacked layer outputs

        Returns:
            h_out: [BS, D_h] — final layer output
            stats: dict (only when collect=True) — per-layer gate stats
            layer_outputs: [BS, L, D_h] (only when return_layers=True)
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
        layer_outs = [] if return_layers else None
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

            if return_layers:
                layer_outs.append(h)

            # Update eligibility traces (if PM enabled)
            if self.config.pm_enabled:
                layer.pm.update_eligibility(x, h, surprise)

            x = h  # next layer's input

        # Build stacked layer outputs [BS, L, D_h]
        layer_stack = torch.stack(layer_outs, dim=1) if return_layers else None

        # Return based on flags
        if collect and return_layers:
            return x, layer_stats, layer_stack
        if collect:
            return x, layer_stats
        if return_layers:
            return x, layer_stack
        return x  # final layer output [BS, D_h]

    def forward_span(self, x_block_all: Tensor, y_wm_all: Tensor,
                     x_emb_all: Tensor, surprise_span: Tensor,
                     carry_all: Tensor,
                     collect: bool = False) -> Tensor:
        """Process P tokens in parallel through this block.

        Note: PM eligibility is NOT updated here — it's deferred to the
        trainer's post-forward step (Step 8 in the refactor plan).

        Args:
            x_block_all: [BS, P, D_h] — block input for all tokens
            y_wm_all: [BS, P, D] — shared WM output
            x_emb_all: [BS, P, D] — token embeddings
            surprise_span: [BS, 1] — frozen surprise for this span
            carry_all: [BS, P, 1] — 0 at doc boundaries, 1 otherwise
            collect: if True, return (h_out, layer_stats) tuple

        Returns:
            h_out_all: [BS, P, D_h] — final layer output for all tokens
            (if collect: also returns {layer_idx: gate_stats} dict)
        """
        # EM retrieval (if enabled)
        if self.config.em_enabled:
            y_em_all = self.em.retrieve_batch(x_emb_all, y_wm_all)
        else:
            y_em_all = torch.zeros_like(y_wm_all)

        # Project to D_h: [BS, P, D_h]
        y_wm_proj_all = self.W_wm_proj(y_wm_all)
        y_em_proj_all = self.W_em_proj(y_em_all)

        # Sequential layers (each with batched forward)
        x = x_block_all
        layer_stats = {} if collect else None
        for l_idx, layer in enumerate(self.layers):
            if self.config.pm_enabled:
                y_pm_all = layer.pm.apply_batch(x)
            else:
                y_pm_all = torch.zeros_like(x)

            result = layer.forward_span(x, y_pm_all, y_wm_proj_all, y_em_proj_all,
                                        surprise_span, carry_all, collect=collect)
            if collect:
                x, lstats = result
                layer_stats[l_idx] = lstats
            else:
                x = result

        # Collect per-layer outputs for spatial decoder.
        # Each layer's _last_h_all is [BS, P, D_h], stored during forward_span.
        self._last_layer_stack = torch.stack(
            [layer._last_h_all for layer in self.layers], dim=2
        )  # [BS, P, L, D_h]

        if collect:
            return x, layer_stats
        return x

    def commit_pm(self, force_mode: str = "normal",
                  span_surprise: Tensor = None) -> dict:
        """Trigger PM commits for all layers in this block.

        Args:
            force_mode: "normal" — use controller decisions
                        "force_on" — commit all streams
                        "force_off" — skip all commits
            span_surprise: [BS] — mean surprise over span (for controller)
        """
        commit_info = {}
        if force_mode == "force_off":
            return commit_info

        for l_idx, layer in enumerate(self.layers):
            pm = layer.pm

            # Compute eligibility norm for controller
            if pm.elig_K is not None:
                elig_norm = pm.elig_K.norm(dim=-1).mean(dim=-1)  # [BS]
                pm_usage = pm.pm_a.sum(dim=-1)  # [BS]
            else:
                continue

            if force_mode == "force_on":
                BS = elig_norm.shape[0]
                p_commit = torch.ones(BS, device=elig_norm.device)
                lambda_vals = torch.full((BS,), pm.decay,
                                         device=elig_norm.device)
                g = torch.full((BS,), self.config.g_pm_default, device=elig_norm.device)
                tau = torch.full((BS,), self.config.tau_pm, device=elig_norm.device)
                pm.commit(p_commit, lambda_vals, g, None, tau)
                commit_info[l_idx] = p_commit.detach()
            else:
                # Content embedding: mean eligibility key for content-aware neuromod
                content_emb = pm.elig_K.mean(dim=1)  # [BS, D_h]

                # Neuromodulator decides all commit parameters (fully differentiable)
                surprise_input = span_surprise if span_surprise is not None else elig_norm
                p_commit, lambda_vals, g, slot_logits, tau = \
                    layer.pm_neuromodulator.forward(
                        elig_norm,
                        pm_usage / self.config.budget_pm,
                        surprise_input,
                        content_emb=content_emb,
                    )
                pm.commit(p_commit, lambda_vals, g, slot_logits, tau)
                commit_info[l_idx] = p_commit.detach()

        return commit_info

    def detach_states(self):
        """Detach all states in child modules."""
        for layer in self.layers:
            layer.detach_states()
            layer.pm.detach_states()
        self.em.detach_states()

    def reset_states(self, mask: Tensor):
        """Reset states for masked streams.

        In lifelong mode (Phase D), only transient state resets:
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



