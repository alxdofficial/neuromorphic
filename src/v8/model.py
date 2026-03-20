"""V8Model — top-level model wiring CCs + Memory Graph + Neuromodulator.

Training flow per chunk (T=2048 tokens):
  Pass 1: Pre-memory scan + PCM over all T tokens (parallel, full D)
  Memory loop: step memory graph per token, neuromod acts every action_every
  Pass 2: Post-memory scan over all T tokens with injected memory signals
  LM loss backward through CCs only
  PPO update on neuromodulator using collected experience
"""

import torch
import torch.nn as nn
from torch import Tensor

from .config import V8Config
from .lm import V8LM
from .memory_graph import MemoryGraph
from .neuromodulator import Neuromodulator
from .ppo import PPORolloutBuffer


class V8Model(nn.Module):
    """Top-level v8 model: LM + Memory Graph + Neuromodulator."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config

        # Language model (in autograd graph)
        self.lm = V8LM(config)

        # Neuromodulator (trained by PPO, has its own optimizer)
        self._mem_graph = None  # initialized lazily per device/dtype
        self._neuromod_obs_dim = None

    def _ensure_memory(self, BS: int, device: torch.device,
                       dtype: torch.dtype = torch.float32):
        """Lazily initialize memory graph and neuromodulator."""
        if self._mem_graph is not None and self._mem_graph.is_initialized():
            return
        self._mem_graph = MemoryGraph(self.config, device, dtype)
        self._mem_graph.initialize(BS)
        self._neuromod_obs_dim = self._mem_graph.obs_dim
        if not hasattr(self, 'neuromod') or self.neuromod is None:
            self.neuromod = Neuromodulator(
                self.config, self._neuromod_obs_dim
            ).to(device)

    @property
    def memory(self) -> MemoryGraph:
        return self._mem_graph

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        collect_ppo: bool = True,
        use_memory: bool = True,
    ) -> dict:
        """Process a full T-token chunk with per-token memory access.

        Args:
            input_ids:  [BS, T]
            target_ids: [BS, T] or None — for per-token reward computation
            reset_mask: [BS] bool — reset memory for these streams
            collect_ppo: whether to collect PPO experience
            use_memory: if False, skip memory graph entirely (LM-only baseline)

        Returns:
            dict with keys:
                logits:     [BS, T, vocab]
                aux_loss:   scalar (PCM)
                ppo_buffer: PPORolloutBuffer or None
                surprise:   [BS, T, C, D_cc] (detached)
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_mem = self.config.D_mem
        D_cc = self.config.D_cc
        action_every = self.config.action_every
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # ==========================================
        # Pass 1: Pre-memory scan + PCM (parallel)
        # ==========================================
        H, x, surprise, aux_loss = self.lm.forward_pre_memory(input_ids)
        # H: [BS, T, D], surprise: [BS, T, C, D_cc]

        # --- No-memory fast path: skip graph entirely ---
        if not use_memory:
            mem_signals = torch.zeros(BS, T, C, D_mem, device=device, dtype=dtype)
            logits = self.lm.forward_post_memory(H, mem_signals)
            return {
                "logits": logits,
                "aux_loss": aux_loss,
                "ppo_buffer": None,
                "surprise": surprise.detach(),
            }

        # --- Memory path ---
        self._ensure_memory(BS, device, torch.float32)

        if reset_mask is not None and reset_mask.any():
            self._mem_graph.reset_streams(reset_mask)

        # Build per-token CC→memory signals
        cc_signals_all = self.lm.build_cc_signals(H, surprise)
        # [BS, T, C, D_mem] — detach from LM graph (memory is environment)
        cc_signals_all = cc_signals_all.detach().float()

        # ==========================================
        # Memory loop: step graph per token
        # ==========================================
        mem_signals = torch.zeros(BS, T, C, D_mem, device=device)
        N_neurons = self.config.N_neurons

        # PPO experience buffer
        ppo_buffer = None
        if collect_ppo:
            n_envs = BS * N_neurons
            # Neuromod fires at t=0, action_every, 2*action_every, ...
            # Use ceiling to handle T not divisible by action_every
            n_actions = (T + action_every - 1) // action_every
            ppo_buffer = PPORolloutBuffer(
                n_actions, n_envs,
                self._neuromod_obs_dim, self.neuromod.act_dim,
                device,
            )

        # Previous surprise for neuromod obs (start with zeros)
        prev_cc_surprise = torch.zeros(BS, C, D_cc, device=device)
        action_step = 0

        for t in range(T):
            # Doc boundary: reset BEFORE stepping so EOT token gets fresh memory
            if not self.config.lifelong_mode and t > 0:
                eot_mask = (input_ids[:, t - 1] == self.config.eot_id)
                if eot_mask.any():
                    self._mem_graph.reset_streams(eot_mask)

            # Step memory graph with CC signal for this token
            # Disable autocast: memory graph operates in float32
            with torch.autocast(device_type=device.type, enabled=False):
                mem_out = self._mem_graph.step(cc_signals_all[:, t])  # [BS, C, D_mem]
            mem_signals[:, t] = mem_out

            # Neuromodulator acts every action_every tokens
            if t % action_every == 0 and collect_ppo:
                # Mean surprise across the last action_every tokens for obs
                if t > 0:
                    recent_surp = surprise[:, max(0, t - action_every):t].detach()
                    mean_surp = recent_surp.mean(dim=1)  # [BS, C, D_cc]
                else:
                    mean_surp = prev_cc_surprise

                # Neuron observations
                obs = self._mem_graph.get_neuron_obs(
                    cc_surprise=mean_surp.float()
                )  # [BS, N_neurons, obs_dim]
                obs_flat = obs.reshape(BS * N_neurons, -1)

                # Policy forward (no_grad for action sampling during collection)
                with torch.no_grad():
                    action, logprob, _, value = self.neuromod.get_action_and_value(
                        obs_flat
                    )

                # Clamp actions before applying to memory graph
                max_act = self.config.max_action_magnitude
                clamped_action = action.clamp(-max_act, max_act)
                D_mem_act = self.config.D_mem
                max_conn = self.config.max_connections
                delta_prim = clamped_action[:, :D_mem_act].reshape(BS, N_neurons, D_mem_act)
                delta_thresh = clamped_action[:, D_mem_act:].reshape(BS, N_neurons, max_conn)
                self._mem_graph.apply_actions(delta_prim, delta_thresh)

                # Store experience (reward filled in later)
                if ppo_buffer is not None:
                    ppo_buffer.add(
                        obs=obs_flat,
                        action=action,
                        logprob=logprob,
                        reward=torch.zeros(BS * N_neurons, device=device),
                        value=value,
                    )
                    action_step += 1

                prev_cc_surprise = mean_surp

            # (doc boundary reset moved to top of loop — fires before step)

        # ==========================================
        # Pass 2: Post-memory scan (parallel)
        # ==========================================
        # Convert mem_signals to LM dtype for injection
        logits = self.lm.forward_post_memory(H, mem_signals.to(dtype))

        # ==========================================
        # Compute per-token rewards for PPO
        # ==========================================
        if target_ids is not None and ppo_buffer is not None and action_step > 0:
            with torch.no_grad():
                # Per-token CE loss [BS, T]
                per_token_ce = torch.nn.functional.cross_entropy(
                    logits.detach().reshape(-1, self.config.vocab_size),
                    target_ids.reshape(-1),
                    reduction='none',
                ).reshape(BS, T)

                # Reward = negative loss (lower CE = higher reward)
                rewards = -per_token_ce  # [BS, T]

                # Aggregate rewards per action step (mean over action_every tokens)
                # and per block (each block's reward = mean of its CC's tokens)
                for step_idx in range(action_step):
                    t_start = step_idx * action_every
                    t_end = min(t_start + action_every, T)
                    step_reward = rewards[:, t_start:t_end].mean(dim=1)  # [BS]

                    # Expand to per-neuron: each neuron in a block gets block reward
                    M = self.config.M_per_block
                    block_rewards = step_reward.unsqueeze(1).expand(BS, C)  # [BS, C]
                    neuron_rewards = block_rewards.unsqueeze(2).expand(
                        BS, C, M
                    ).reshape(BS * N_neurons)  # [BS * N_neurons]

                    ppo_buffer.rewards[step_idx] = neuron_rewards

                # Bootstrap value for GAE
                final_obs = self._mem_graph.get_neuron_obs(
                    cc_surprise=prev_cc_surprise.float()
                ).reshape(BS * N_neurons, -1)
                with torch.no_grad():
                    next_value = self.neuromod.get_value(final_obs)
                ppo_buffer.compute_returns(
                    next_value,
                    gamma=self.config.ppo_gamma,
                    gae_lambda=self.config.ppo_lambda,
                )

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "ppo_buffer": ppo_buffer,
            "surprise": surprise.detach(),
        }

    def initialize_states(self, BS: int, device: torch.device):
        """Initialize all state (carries + memory)."""
        self.lm.initialize_carries()
        dtype = next(self.lm.parameters()).dtype
        self._ensure_memory(BS, device, torch.float32)

    def detach_states(self):
        """TBPTT boundary: detach scan carries. Memory persists without detach."""
        self.lm.detach_carries()

    def param_count(self) -> int:
        """Total trained parameters (LM + neuromodulator)."""
        total = self.lm.param_count()
        if hasattr(self, 'neuromod') and self.neuromod is not None:
            total += sum(p.numel() for p in self.neuromod.parameters()
                         if p.requires_grad)
        return total
