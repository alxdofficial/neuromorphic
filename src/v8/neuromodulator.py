"""Neuromodulator — policy network for memory graph plasticity.

Shared MLP across all neurons. Observes neuron state + context, outputs
modifications to primitives, thresholds, temperature, and decay.
Trained by sampling-based RL (no critic needed).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from .config import V8Config


class Neuromodulator(nn.Module):
    """Policy network for memory graph plasticity control.

    Shared across all neurons — each neuron's observation is processed
    independently through the same network (parameter sharing).
    No critic — advantage is computed by comparing K sampled trajectories.
    """

    def __init__(self, config: V8Config, obs_dim: int):
        super().__init__()
        self.config = config
        D_mem = config.D_mem
        max_conn = config.max_connections
        hidden = config.neuromod_hidden

        # Policy backbone
        layers = []
        in_dim = obs_dim
        for _ in range(config.neuromod_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)

        # Action heads
        self.prim_head = nn.Linear(hidden, D_mem)
        self.thresh_head = nn.Linear(hidden, max_conn)
        self.temp_head = nn.Linear(hidden, 1)
        self.decay_head = nn.Linear(hidden, 1)

        # Per-group log_std (state-independent)
        # Init std ≈ 0.05 so clamp at ±0.1 is rarely triggered
        self.prim_logstd = nn.Parameter(torch.full((1, D_mem), -3.0))
        self.thresh_logstd = nn.Parameter(torch.full((1, max_conn), -3.0))
        self.temp_logstd = nn.Parameter(torch.full((1, 1), -3.0))
        self.decay_logstd = nn.Parameter(torch.full((1, 1), -3.0))

        self.max_action = config.max_action_magnitude

        # Zero-init heads for stable start
        for head in [self.prim_head, self.thresh_head, self.temp_head, self.decay_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def act_dim(self) -> int:
        return self.config.D_mem + self.config.max_connections + 2

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        """Sample action (or evaluate given action).

        Args:
            obs:    [*, obs_dim]
            action: [*, act_dim] or None — if None, sample from policy

        Returns:
            action:   [*, act_dim]
            log_prob: [*]
            entropy:  [*]
            value:    None (no critic)
        """
        h = self.backbone(obs)
        prim_mean = self.prim_head(h)
        thresh_mean = self.thresh_head(h)
        temp_mean = self.temp_head(h)
        decay_mean = self.decay_head(h)

        mean = torch.cat([prim_mean, thresh_mean, temp_mean, decay_mean], dim=-1)
        logstd = torch.cat([
            self.prim_logstd.expand_as(prim_mean),
            self.thresh_logstd.expand_as(thresh_mean),
            self.temp_logstd.expand_as(temp_mean),
            self.decay_logstd.expand_as(decay_mean),
        ], dim=-1)
        std = logstd.exp()

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, None
