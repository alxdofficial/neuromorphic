"""Neuromodulator — policy network for memory graph plasticity.

Shared MLP across all neurons. Observes neuron state + plasticity metrics,
outputs modifications to primitives, connection weights, temperature, and decay.
Trained by sampling-based RL (REINFORCE / GRPO, no critic needed).
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
        K_conn = config.K_connections
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
        self.conn_weight_head = nn.Linear(hidden, K_conn)
        self.decay_head = nn.Linear(hidden, 1)

        # Per-group log_std (state-independent)
        # Init at -2.0 → std ≈ 0.135, enough exploration for 8 actions/chunk
        self.prim_logstd = nn.Parameter(torch.full((1, D_mem), -2.0))
        self.conn_weight_logstd = nn.Parameter(torch.full((1, K_conn), -2.0))
        self.decay_logstd = nn.Parameter(torch.full((1, 1), -2.0))

        self.max_action = config.max_action_magnitude

        # Zero-init heads for stable start
        for head in [self.prim_head, self.conn_weight_head, self.decay_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def act_dim(self) -> int:
        return self.config.D_mem + self.config.K_connections + 1

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
        conn_mean = self.conn_weight_head(h)
        decay_mean = self.decay_head(h)

        mean = torch.cat([prim_mean, conn_mean, decay_mean], dim=-1)
        logstd = torch.cat([
            self.prim_logstd.expand_as(prim_mean),
            self.conn_weight_logstd.expand_as(conn_mean),
            self.decay_logstd.expand_as(decay_mean),
        ], dim=-1)
        std = logstd.exp().clamp(min=1e-6)
        mean = mean.nan_to_num(0.0)
        std = std.nan_to_num(1e-2)

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, None
