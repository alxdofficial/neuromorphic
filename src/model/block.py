"""
Block — parallel processing unit with L sequential layers + 1 EM bank.

One Block per parallel track (B total). Each block processes D_h dimensions,
contains L Layers (each with their own PM), and one shared EpisodicMemory.

When multi-timescale is enabled (block_scales), each block can operate at a
different temporal resolution via TemporalPooler (causal conv downsampling).

When PCM is enabled, each block has a PredictiveCodingModule that computes
vector surprise (δ) from evidence vs hypothesis, replacing scalar surprise.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .layer import Layer
from .episodic_memory import EpisodicMemory, EMNeuromodulator
from .temporal_pool import TemporalPooler, carry_min_pool
from .predictive_coding import PredictiveCodingModule
from .utils import StateMixin


class Block(nn.Module, StateMixin):
    _state_tensor_names = []  # state lives in child modules

    def __init__(self, config: ModelConfig, block_idx: int):
        super().__init__()
        self.config = config
        self.block_idx = block_idx

        # Multi-timescale: per-block temporal scale factor
        if config.block_scales is not None:
            self.scale = config.block_scales[block_idx]
        else:
            self.scale = 1
        self.pooler = TemporalPooler(config.D_h, self.scale)

        # Predictive Coding Module (per block)
        if config.pcm_enabled:
            self.pcm = PredictiveCodingModule(config)
        else:
            self.pcm = None

        # Pre-LayerNorm for Layer 0 input: ensures L0 receives normalized
        # input (like L1+ which get LayerNorm'd output from previous layer),
        # fixing gradient asymmetry between L0 and deeper layers.
        self.input_norm = nn.LayerNorm(config.D_h)

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

        # Runtime caches (set during forward_span, consumed at boundary)
        self._last_z: Tensor | None = None
        self._last_L_recon: Tensor | None = None
        self._last_token_surprise: Tensor | None = None

        # Multi-timescale step() mode state
        # Slow blocks only process every `scale` tokens; on hold steps
        # they return the cached output from the last update step.
        self._step_counter: int = 0
        self._last_step_output: Tensor | None = None
        self._last_step_layer_stack: Tensor | None = None

    def _step_hold_result(self, collect: bool, return_layers: bool):
        """Return cached output for multi-timescale hold steps."""
        h = self._last_step_output
        if collect and return_layers:
            return h, {}, self._last_step_layer_stack
        if collect:
            return h, {}
        if return_layers:
            return h, self._last_step_layer_stack
        return h

    def step(self, x_block: Tensor, y_wm: Tensor, x_proj: Tensor,
             surprise: Tensor, carry: Tensor, collect: bool = False,
             return_layers: bool = False):
        """Process one token through this block.

        For multi-timescale blocks (scale > 1), only processes every s-th
        token and holds the last output on intermediate steps. Doc boundaries
        (carry=0) force immediate processing and counter reset.

        Args:
            x_block: [BS, D_h] — input slice for this block
            y_wm: [BS, D] — shared WM output
            x_proj: [BS, D] — W_in projection (for EM retrieval query)
            surprise: [BS, 1] — scalar surprise signal
            carry: [BS, 1] — 0 at doc boundaries, 1 otherwise
            collect: bool — if True, include per-layer gate stats
            return_layers: bool — if True, include stacked layer outputs

        Returns:
            h_out: [BS, D_h] — final layer output
            stats: dict (only when collect=True) — per-layer gate stats
            layer_outputs: [BS, L, D_h] (only when return_layers=True)
        """
        s = self.scale
        if s > 1:
            # Doc boundary forces immediate processing and counter reset
            has_boundary = (carry < 0.5).any()
            if has_boundary:
                self._step_counter = 0

            # Hold steps: return cached output, skip all computation
            if self._step_counter > 0 and self._last_step_output is not None:
                self._step_counter = (self._step_counter + 1) % s
                return self._step_hold_result(collect, return_layers)

        # EM retrieval (if enabled)
        if self.config.em_enabled:
            y_em = self.em.retrieve(x_proj, y_wm)  # [BS, D]
        else:
            y_em = torch.zeros_like(y_wm)

        # Project to D_h
        y_wm_proj = self.W_wm_proj(y_wm)      # [BS, D_h]
        y_em_proj = self.W_em_proj(y_em)        # [BS, D_h]

        # PCM: compute vector surprise (δ) and FFN gain
        if self.pcm is not None:
            z = self.pcm.encode(x_block.unsqueeze(1))   # [BS, 1, D_pc]
            delta = self.pcm.compute_surprise(z)         # [BS, 1, D_pc]
            gate_surprise = delta.squeeze(1)             # [BS, D_pc]
            ffn_gain = self.pcm.compute_ffn_gain(delta).squeeze(1)  # [BS, D_h]
            # Scalar surprise from PCM for PM eligibility: ‖δ‖
            pm_surprise = delta.norm(dim=-1)             # [BS, 1]
        else:
            gate_surprise = surprise                     # [BS, 1]
            ffn_gain = None
            pm_surprise = surprise                       # [BS, 1]

        # Normalize input for Layer 0 (matches the LayerNorm'd output L1+ receive)
        x = self.input_norm(x_block)
        layer_stats = {}
        layer_outs = [] if return_layers else None
        for l_idx, layer in enumerate(self.layers):
            # PM read (if enabled)
            if self.config.pm_enabled:
                y_pm = layer.pm.apply(x)
            else:
                y_pm = torch.zeros_like(x)

            # Forward through layer
            result = layer.step(x, y_pm, y_wm_proj, y_em_proj, gate_surprise,
                                carry, ffn_gain=ffn_gain, collect=collect)

            if collect:
                h, lstats = result
                layer_stats[l_idx] = lstats
            else:
                h = result

            if return_layers:
                layer_outs.append(h)

            # Update eligibility traces (if PM enabled)
            # Uses ‖δ‖ from PCM when enabled, else model's scalar surprise
            if self.config.pm_enabled:
                layer.pm.update_eligibility(x, h, pm_surprise)

            x = h  # next layer's input

        # Build stacked layer outputs [BS, L, D_h]
        layer_stack = torch.stack(layer_outs, dim=1) if return_layers else None

        # Cache output for multi-timescale hold steps
        if s > 1:
            self._last_step_output = x
            self._last_step_layer_stack = layer_stack
            self._step_counter = (self._step_counter + 1) % s

        # Return based on flags
        if collect and return_layers:
            return x, layer_stats, layer_stack
        if collect:
            return x, layer_stats
        if return_layers:
            return x, layer_stack
        return x  # final layer output [BS, D_h]

    def forward_span(self, x_block_all: Tensor, y_wm_all: Tensor,
                     x_proj_all: Tensor, surprise_span: Tensor,
                     carry_all: Tensor,
                     collect: bool = False) -> Tensor:
        """Process P tokens in parallel through this block.

        When multi-timescale is enabled (scale > 1), input is downsampled
        before processing and upsampled after. PCM operates at the pooled
        resolution.

        Note: PM eligibility is NOT updated here — it's deferred to the
        trainer's post-forward step (Step 8 in the refactor plan).

        Args:
            x_block_all: [BS, P, D_h] — block input for all tokens
            y_wm_all: [BS, P, D] — shared WM output
            x_proj_all: [BS, P, D] — W_in projections (for EM retrieval)
            surprise_span: [BS, 1] — frozen surprise for this span
            carry_all: [BS, P, 1] — 0 at doc boundaries, 1 otherwise
            collect: if True, return (h_out, layer_stats) tuple

        Returns:
            h_out_all: [BS, P, D_h] — final layer output for all tokens
            (if collect: also returns {layer_idx: gate_stats} dict)
        """
        BS, P, D_h = x_block_all.shape
        s = self.scale

        # --- Temporal downsampling (multi-timescale) ---
        # Learned weighted aggregation within non-overlapping windows.
        # carry_all is passed so tokens from previous docs within a window
        # are masked out before aggregation (boundary-aware pooling).
        x_block_b = self.pooler.downsample(x_block_all, carry_all)  # [BS, P_b, D_h]
        P_b = x_block_b.shape[1]

        y_wm_b = self.pooler.downsample(y_wm_all, carry_all)       # [BS, P_b, D]
        x_proj_b = self.pooler.downsample(x_proj_all, carry_all)    # [BS, P_b, D]
        # carry uses min-pool: any boundary in window forces reset
        carry_b = carry_min_pool(carry_all, s)           # [BS, P_b, 1]

        # EM retrieval at pooled resolution (if enabled)
        if self.config.em_enabled:
            y_em_b = self.em.retrieve_batch(x_proj_b, y_wm_b)
        else:
            y_em_b = torch.zeros_like(y_wm_b)

        # Project to D_h: [BS, P_b, D_h]
        y_wm_proj_b = self.W_wm_proj(y_wm_b)
        y_em_proj_b = self.W_em_proj(y_em_b)

        # --- PCM: encode evidence, compute surprise, FFN gain ---
        if self.pcm is not None:
            z = self.pcm.encode(x_block_b)                  # [BS, P_b, D_pc]
            delta = self.pcm.compute_surprise(z)             # [BS, P_b, D_pc]
            surprise_b = delta                               # [BS, P_b, D_pc]
            ffn_gain_b = self.pcm.compute_ffn_gain(delta)    # [BS, P_b, D_h]

            # Per-token scalar surprise from ‖δ‖ (for PM eligibility + EM candidates)
            pcm_surprise_b = delta.norm(dim=-1, keepdim=True)  # [BS, P_b, 1]
            self._last_token_surprise = self.pooler.upsample(
                pcm_surprise_b, P
            )  # [BS, P, 1]

            # Cache for boundary ops (called after forward pass)
            self._last_z = z
            self._last_L_recon = self.pcm.compute_recon_loss(z, x_block_b)
        else:
            # Scalar surprise: expand [BS, 1] → [BS, P_b, 1]
            surprise_b = surprise_span.unsqueeze(1).expand(BS, P_b, 1)
            ffn_gain_b = None
            self._last_token_surprise = None

        # Normalize input for Layer 0 (matches the LayerNorm'd output L1+ receive)
        x = self.input_norm(x_block_b)
        layer_stats = {} if collect else None
        for l_idx, layer in enumerate(self.layers):
            if self.config.pm_enabled:
                y_pm_b = layer.pm.apply_batch(x)
            else:
                y_pm_b = torch.zeros_like(x)

            result = layer.forward_span(x, y_pm_b, y_wm_proj_b, y_em_proj_b,
                                        surprise_b, carry_b,
                                        ffn_gain_all=ffn_gain_b,
                                        collect=collect)
            if collect:
                x, lstats = result
                layer_stats[l_idx] = lstats
            else:
                x = result

        # --- Temporal upsampling back to P ---
        x_up = self.pooler.upsample(x, P)  # [BS, P, D_h]

        # Collect per-layer outputs for spatial decoder (at original resolution).
        if self.config.snapshot_enabled:
            self._last_layer_stack = torch.stack(
                [self.pooler.upsample(layer._last_h_all, P)
                 for layer in self.layers], dim=2
            )  # [BS, P, L, D_h]

        if collect:
            return x_up, layer_stats
        return x_up

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
        if self.pcm is not None:
            self.pcm.detach_states()

    def reset_states(self, mask: Tensor):
        """Reset states for masked streams.

        In lifelong mode (Phase C), only transient state resets:
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

        if self.pcm is not None:
            self.pcm.reset_states(mask)

        # Reset step-mode state for multi-timescale blocks
        if self.scale > 1 and mask.any():
            self._step_counter = 0
            self._last_step_output = None
            self._last_step_layer_stack = None
