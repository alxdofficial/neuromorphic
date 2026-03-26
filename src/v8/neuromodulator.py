"""Neuromodulator — gates Hebbian eligibility traces + controls decay.

Three-factor learning (Fremaux & Gerstner 2016):
  1. Eligibility traces (Hebbian, local): propose updates to primitives/key
  2. Neuromodulator (RL-trained): gates whether to apply those updates
  3. Result: primitives/key change only when local evidence + global reward align

The neuromod observes each neuron's activity and trace confidence, then outputs:
  - gate ∈ [-1, 1]: how much of the Hebbian update to apply (+consolidate, -explore)
  - decay_target: target for the neuron's decay_logit (temporal persistence)

Trained by GRPO trajectory scoring (no learned value function).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config

_LOG_2PI = math.log(2 * math.pi)


class Neuromodulator(nn.Module):
    """Policy network for gated Hebbian plasticity + decay control.

    Shared across all neurons — each neuron's observation is processed
    independently through the same network (parameter sharing).
    """

    def __init__(self, config: V8Config, obs_dim: int):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        hidden = config.neuromod_hidden

        # Backbone (smaller: 2 layers of 512 — action space is just 2 dims)
        layers = []
        in_dim = obs_dim
        for _ in range(config.neuromod_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)

        # Gate head: controls Hebbian update application
        # Zero-init so initial gate ≈ 0 (no plasticity at start)
        self.gate_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        # Decay head: target for decay_logit
        # Zero-init so initial target = 0 (matching decay_logit init)
        self.decay_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.decay_head.weight)
        nn.init.zeros_(self.decay_head.bias)

        # Per-action-dim log_std (state-independent)
        logstd_init = config.neuromod_logstd_init
        self.gate_logstd = nn.Parameter(torch.full((1, 1), logstd_init))
        self.decay_logstd = nn.Parameter(torch.full((1, 1), logstd_init))

    @property
    def act_dim(self) -> int:
        return 2  # gate + decay_target

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        """Sample action (or evaluate given action).

        Args:
            obs:    [*, obs_dim] — per-neuron observations
            action: [*, 2] or None — if None, sample from policy

        Returns:
            action:   [*, 2] — (gate, decay_target)
            log_prob: [*]
            entropy:  [*]
            value:    None (no learned value function)
        """
        h = self.backbone(obs)
        gate_mean = self.gate_head(h)    # [*, 1]
        decay_mean = self.decay_head(h)  # [*, 1]

        mean = torch.cat([gate_mean, decay_mean], dim=-1)  # [*, 2]
        logstd = torch.cat([
            self.gate_logstd.expand_as(gate_mean),
            self.decay_logstd.expand_as(decay_mean),
        ], dim=-1)  # [*, 2]
        std = logstd.exp().clamp(min=1e-6)
        mean = mean.nan_to_num(0.0)
        std = std.nan_to_num(1e-2)

        if action is None:
            action = mean + std * torch.randn_like(mean)

        # Direct Gaussian log-prob and entropy
        var = std * std
        log_prob = (-0.5 * ((action - mean) ** 2 / var + _LOG_2PI) - logstd).sum(dim=-1)
        entropy = (0.5 * (1.0 + _LOG_2PI) + logstd).sum(dim=-1)

        return action, log_prob, entropy, None
