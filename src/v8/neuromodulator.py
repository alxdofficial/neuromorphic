"""Neuromodulator — policy network for memory graph plasticity.

Shared MLP across all neurons. Observes neuron state + plasticity metrics,
outputs modifications to primitives, connection weights, and decay.
Trained by REINFORCE with learned value baseline.
Collects across multiple chunks for longer reward horizon.
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
    Learned value function on pooled global state for baseline.
    """

    def __init__(self, config: V8Config, obs_dim: int):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        D_mem = config.D_mem
        K_conn = config.K_connections
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
        self.conn_weight_head = nn.Linear(hidden, K_conn)
        self.decay_head = nn.Linear(hidden, 1)

        # Per-group log_std (state-independent)
        logstd_init = config.neuromod_logstd_init
        self.prim_logstd = nn.Parameter(torch.full((1, D_mem), logstd_init))
        self.conn_weight_logstd = nn.Parameter(torch.full((1, K_conn), logstd_init))
        self.decay_logstd = nn.Parameter(torch.full((1, 1), logstd_init))

        self.max_action = config.max_action_magnitude

        # Zero-init heads for stable start
        for head in [self.prim_head, self.conn_weight_head, self.decay_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Value function: pooled global obs → scalar
        # Separate small MLP (not shared with policy backbone)
        v_hidden = config.rl_value_hidden
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, v_hidden),
            nn.Tanh(),
            nn.Linear(v_hidden, v_hidden),
            nn.Tanh(),
            nn.Linear(v_hidden, 1),
        )

    @property
    def act_dim(self) -> int:
        return self.config.D_mem + self.config.K_connections + 1

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
            value:    None (use get_value for value estimates)
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

    def get_value(self, global_obs: Tensor) -> Tensor:
        """Predict expected return from pooled global memory state.

        Args:
            global_obs: [BS, obs_dim] — mean-pooled across all neurons

        Returns:
            value: [BS] — predicted return
        """
        return self.value_net(global_obs).squeeze(-1)
