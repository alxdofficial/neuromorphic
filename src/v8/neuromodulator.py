"""Neuromodulator — policy network for memory graph plasticity.

Shared MLP across all neurons. Observes neuron state + plasticity metrics,
outputs new primitives, routing keys, and decay delta.
Trained by REINFORCE with counterfactual baseline (no learned value function).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config

_LOG_2PI = math.log(2 * math.pi)


class Neuromodulator(nn.Module):
    """Policy network for memory graph plasticity control.

    Shared across all neurons — each neuron's observation is processed
    independently through the same network (parameter sharing).
    Counterfactual baseline (no learned value function).
    """

    def __init__(self, config: V8Config, obs_dim: int):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        D_mem = config.D_mem
        hidden = config.neuromod_hidden

        # Policy backbone (per-neuron)
        layers = []
        in_dim = obs_dim
        for _ in range(config.neuromod_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)

        # Action heads
        self.prim_head = nn.Linear(hidden, D_mem)
        self.key_head = nn.Linear(hidden, D_mem)
        self.decay_head = nn.Linear(hidden, 1)

        # Per-group log_std (state-independent)
        logstd_init = config.neuromod_logstd_init
        self.prim_logstd = nn.Parameter(torch.full((1, D_mem), logstd_init))
        self.key_logstd = nn.Parameter(torch.full((1, D_mem), logstd_init))
        self.decay_logstd = nn.Parameter(torch.full((1, 1), logstd_init))

        self.max_action = config.max_action_magnitude

        # Heads use default Kaiming init — nonzero from step 1 so
        # backbone gets gradients immediately (no zero-init trap)

    @property
    def act_dim(self) -> int:
        return self.config.D_mem * 2 + 1  # prim + key + decay

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        """Sample action (or evaluate given action).

        Args:
            obs:    [*, obs_dim] — per-neuron observations
            action: [*, act_dim] or None — if None, sample from policy

        Returns:
            action:   [*, act_dim]
            log_prob: [*]
            entropy:  [*]
            value:    None (no learned value function)
        """
        h = self.backbone(obs)
        prim_mean = self.prim_head(h)
        key_mean = self.key_head(h)
        decay_mean = self.decay_head(h)

        mean = torch.cat([prim_mean, key_mean, decay_mean], dim=-1)
        logstd = torch.cat([
            self.prim_logstd.expand_as(prim_mean),
            self.key_logstd.expand_as(key_mean),
            self.decay_logstd.expand_as(decay_mean),
        ], dim=-1)
        std = logstd.exp().clamp(min=1e-6)
        mean = mean.nan_to_num(0.0)
        std = std.nan_to_num(1e-2)

        if action is None:
            action = mean + std * torch.randn_like(mean)

        # Direct Gaussian log-prob and entropy (avoids Normal object overhead)
        var = std * std
        log_prob = (-0.5 * ((action - mean) ** 2 / var + _LOG_2PI) - logstd).sum(dim=-1)
        entropy = (0.5 * (1.0 + _LOG_2PI) + logstd).sum(dim=-1)

        return action, log_prob, entropy, None

