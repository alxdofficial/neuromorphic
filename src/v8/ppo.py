"""PPO training for the neuromodulator.

Rollout buffer stores (obs, action, logprob, value, reward) per neuron per
action step. GAE computes advantages. PPO clipped surrogate updates the
neuromodulator policy.

Each "environment" is one neuron in one batch stream. With BS=16 and
N_neurons=1024, there are 16384 parallel environments.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config
from .neuromodulator import Neuromodulator


class RunningMeanStd:
    """Running mean/std for reward normalization."""

    def __init__(self, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x: Tensor):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.numel()
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return torch.sqrt(self.var).clamp(min=1e-8)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class PPORolloutBuffer:
    """Stores one rollout of T_actions steps across N_envs parallel neurons.

    Shape: [T_actions, N_envs, ...] where N_envs = BS * N_neurons.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int,
                 act_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.logprobs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        self.advantages = None
        self.returns = None
        self.step = 0

    def add(self, obs: Tensor, action: Tensor, logprob: Tensor,
            reward: Tensor, value: Tensor):
        """Add one timestep of experience.

        All tensors: [N_envs, ...] or [N_envs].
        """
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.step += 1

    def compute_returns(self, next_value: Tensor, gamma: float = 0.99,
                        gae_lambda: float = 0.95):
        """Compute GAE advantages and returns.

        Args:
            next_value: [N_envs] — bootstrap value for the last step
        """
        T = self.step  # actual steps filled (may be < num_steps)
        advantages = torch.zeros(T, self.num_envs, device=self.device)
        lastgaelam = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                nextvalue = next_value
            else:
                nextvalue = self.values[t + 1]

            delta = self.rewards[t] + gamma * nextvalue - self.values[t]
            advantages[t] = delta + gamma * gae_lambda * lastgaelam
            lastgaelam = advantages[t]

        self.advantages = advantages[:T]
        self.returns = self.advantages + self.values[:T]

    def get_minibatches(self, minibatch_size: int):
        """Flatten [T, N_envs] → [T*N_envs] and yield random minibatches."""
        assert self.advantages is not None, \
            "compute_returns() must be called before get_minibatches()"
        T = self.step
        if T == 0:
            return  # empty buffer, yield nothing
        batch_size = T * self.num_envs
        indices = torch.randperm(batch_size, device=self.device)

        def flat(x):
            return x[:T].reshape(batch_size, *x.shape[2:])

        obs = flat(self.obs)
        actions = flat(self.actions)
        logprobs = flat(self.logprobs)
        advantages = flat(self.advantages)
        returns = flat(self.returns)
        values = flat(self.values)

        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            idx = indices[start:end]
            yield (obs[idx], actions[idx], logprobs[idx],
                   advantages[idx], returns[idx], values[idx])

    def reset(self):
        self.step = 0
        self.advantages = None
        self.returns = None


class PPOTrainer:
    """Trains the neuromodulator via PPO."""

    def __init__(self, neuromod: Neuromodulator, config: V8Config,
                 device: torch.device):
        self.neuromod = neuromod
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            neuromod.parameters(), lr=config.ppo_lr, eps=1e-5,
        )
        self.reward_rms = RunningMeanStd(device=device)

    def normalize_rewards(self, rewards: Tensor) -> Tensor:
        """Normalize rewards using running statistics."""
        self.reward_rms.update(rewards)
        return rewards / self.reward_rms.std

    def update(self, buffer: PPORolloutBuffer) -> dict:
        """Run PPO epochs on collected rollout.

        Returns dict of training metrics.
        """
        config = self.config
        total_metrics = {
            "ppo_policy_loss": 0.0,
            "ppo_value_loss": 0.0,
            "ppo_entropy": 0.0,
            "ppo_approx_kl": 0.0,
            "ppo_clip_frac": 0.0,
        }
        n_updates = 0

        for epoch in range(config.ppo_epochs):
            for batch in buffer.get_minibatches(config.ppo_minibatch):
                obs, actions, old_logprobs, advantages, returns, old_values = batch

                # Normalize advantages per minibatch
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward through neuromodulator
                _, new_logprobs, entropy, new_values = \
                    self.neuromod.get_action_and_value(obs, actions)

                # Policy loss (clipped surrogate)
                log_ratio = new_logprobs - old_logprobs
                ratio = log_ratio.exp()

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - config.ppo_clip,
                                    1.0 + config.ppo_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -config.ppo_clip, config.ppo_clip
                )
                vf_loss1 = (new_values - returns) ** 2
                vf_loss2 = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total
                loss = (policy_loss
                        + config.ppo_vf_coef * value_loss
                        + config.ppo_ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.neuromod.parameters(), 0.5)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > config.ppo_clip).float().mean().item()

                total_metrics["ppo_policy_loss"] += policy_loss.item()
                total_metrics["ppo_value_loss"] += value_loss.item()
                total_metrics["ppo_entropy"] += -entropy_loss.item()
                total_metrics["ppo_approx_kl"] += approx_kl
                total_metrics["ppo_clip_frac"] += clip_frac
                n_updates += 1

        # Average metrics
        if n_updates > 0:
            for k in total_metrics:
                total_metrics[k] /= n_updates

        return total_metrics
