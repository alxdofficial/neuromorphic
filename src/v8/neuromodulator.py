"""Neuromodulator — PPO policy + value network for memory graph plasticity.

Shared MLP across all neurons. Observes neuron state + context, outputs
modifications to primitives and energy thresholds. Trained by PPO.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from .config import V8Config


class Neuromodulator(nn.Module):
    """PPO actor-critic for memory graph plasticity control.

    Shared across all neurons — each neuron's observation is processed
    independently through the same network (parameter sharing).
    """

    def __init__(self, config: V8Config, obs_dim: int):
        super().__init__()
        self.config = config
        D_mem = config.D_mem
        max_conn = config.max_connections
        hidden = config.neuromod_hidden

        # Actor: backbone → two heads (primitive deltas + threshold deltas)
        # Build N-layer backbone
        actor_layers = []
        in_dim = obs_dim
        for _ in range(config.neuromod_layers):
            actor_layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        self.actor_backbone = nn.Sequential(*actor_layers)
        self.prim_head = nn.Linear(hidden, D_mem)
        self.thresh_head = nn.Linear(hidden, max_conn)

        # Separate log_std per action group (state-independent)
        self.prim_logstd = nn.Parameter(torch.full((1, D_mem), -1.0))
        self.thresh_logstd = nn.Parameter(torch.full((1, max_conn), -1.0))

        # Critic: separate network (same depth as actor)
        critic_layers = []
        in_dim = obs_dim
        for _ in range(config.neuromod_layers):
            critic_layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        critic_layers.append(nn.Linear(hidden, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.max_action = config.max_action_magnitude

        # Zero-init output heads for stable start
        nn.init.zeros_(self.prim_head.weight)
        nn.init.zeros_(self.prim_head.bias)
        nn.init.zeros_(self.thresh_head.weight)
        nn.init.zeros_(self.thresh_head.bias)

    @property
    def act_dim(self) -> int:
        return self.config.D_mem + self.config.max_connections

    def get_value(self, obs: Tensor) -> Tensor:
        """Compute value estimate.

        Args:
            obs: [*, obs_dim]
        Returns:
            value: [*]
        """
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample action (or evaluate given action) and compute value.

        Args:
            obs:    [*, obs_dim] — neuron observations
            action: [*, act_dim] or None — if None, sample from policy

        Returns:
            action:   [*, act_dim]
            log_prob: [*]
            entropy:  [*]
            value:    [*]
        """
        h = self.actor_backbone(obs)
        prim_mean = self.prim_head(h)
        thresh_mean = self.thresh_head(h)

        mean = torch.cat([prim_mean, thresh_mean], dim=-1)
        logstd = torch.cat([
            self.prim_logstd.expand_as(prim_mean),
            self.thresh_logstd.expand_as(thresh_mean),
        ], dim=-1)
        std = logstd.exp()

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()

        # Clip actions to max magnitude
        action = action.clamp(-self.max_action, self.max_action)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value
