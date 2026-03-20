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
        # Created eagerly so it's in the module tree for .parameters(), .to(), etc.
        self._mem_graph = None  # initialized lazily per BS/device
        # obs_dim: D_mem*3 (prim+mean_in+mean_out) + 4 (usage+temp+decay+entropy) + D_cc (surprise)
        # Since D_mem = D_cc: obs_dim = D_cc * 4 + 4
        obs_dim = config.D_cc * 4 + 4
        self.neuromod = Neuromodulator(config, obs_dim)

    def _ensure_memory(self, BS: int, device: torch.device,
                       dtype: torch.dtype = torch.float32):
        """Lazily initialize memory graph."""
        if (self._mem_graph is not None
                and self._mem_graph.is_initialized()
                and self._mem_graph.primitives.shape[0] == BS):
            return
        self._mem_graph = MemoryGraph(self.config, device, dtype)
        self._mem_graph.initialize(BS)

    @property
    def memory(self) -> MemoryGraph:
        return self._mem_graph

    def get_ppo_params(self):
        """All parameters trained by PPO (neuromodulator only).

        W_mod (memory graph modulation MLP) is a plain tensor, not yet
        included in any optimizer — flagged as open issue.
        """
        return list(self.neuromod.parameters())

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
            reset_mask: [BS] bool — reset memory + scan carries for these streams
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
        eot_id = self.config.eot_id
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # Reset scan carries at doc boundaries (applies to both memory and no-memory)
        if reset_mask is not None and reset_mask.any():
            self._reset_carries(reset_mask)

        # ==========================================
        # Pass 1: Pre-memory scan + PCM (parallel)
        # ==========================================
        H, x, surprise, aux_loss = self.lm.forward_pre_memory(input_ids)

        # --- No-memory fast path ---
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

        # CC→memory signals: raw H sliced into per-CC columns (D_mem = D_cc)
        # No projection needed. Detach since memory graph is the environment.
        cc_signals_all = H.detach().view(BS, T, C, D_mem).float()

        # ==========================================
        # Memory loop: step graph per token
        # ==========================================
        mem_signals = torch.zeros(BS, T, C, D_mem, device=device)
        N_neurons = self.config.N_neurons

        # PPO experience buffer
        ppo_buffer = None
        if collect_ppo:
            n_envs = BS * N_neurons
            n_actions = (T + action_every - 1) // action_every
            ppo_buffer = PPORolloutBuffer(
                n_actions, n_envs,
                self._mem_graph.obs_dim, self.neuromod.act_dim,
                device,
            )

        prev_cc_surprise = torch.zeros(BS, C, D_cc, device=device)
        action_step = 0

        # Precompute per-token EOT mask for reward masking and doc resets
        eot_at = (input_ids == eot_id)  # [BS, T]

        # Track which action steps had a doc boundary reset (for GAE dones)
        action_dones = []  # list of [BS * N_neurons] tensors

        for t in range(T):
            # Doc boundary: reset if PREVIOUS token was EOT
            did_reset = None
            if not self.config.lifelong_mode and t > 0:
                eot_prev = eot_at[:, t - 1]  # [BS]
                if eot_prev.any():
                    self._mem_graph.reset_streams(eot_prev)
                    self._reset_carries(eot_prev)
                    did_reset = eot_prev

            # Step memory graph
            with torch.autocast(device_type=device.type, enabled=False):
                mem_out = self._mem_graph.step(cc_signals_all[:, t])
            mem_signals[:, t] = mem_out

            # Neuromodulator acts every action_every tokens
            if t % action_every == 0 and collect_ppo:
                # Surprise context for observation
                if t > 0:
                    recent_surp = surprise[:, max(0, t - action_every):t].detach()
                    mean_surp = recent_surp.mean(dim=1)
                else:
                    mean_surp = prev_cc_surprise

                obs = self._mem_graph.get_neuron_obs(
                    cc_surprise=mean_surp.float()
                )
                obs_flat = obs.reshape(BS * N_neurons, -1)

                with torch.no_grad():
                    action, logprob, _, value = self.neuromod.get_action_and_value(
                        obs_flat
                    )

                # Clamp and apply actions to memory graph
                max_act = self.config.max_action_magnitude
                clamped = action.clamp(-max_act, max_act)
                D_mem_act = self.config.D_mem
                max_conn = self.config.max_connections
                # Parse action: [D_mem | max_conn | 1 (temp) | 1 (decay)]
                idx = 0
                d_prim = clamped[:, idx:idx + D_mem_act].reshape(BS, N_neurons, D_mem_act)
                idx += D_mem_act
                d_thresh = clamped[:, idx:idx + max_conn].reshape(BS, N_neurons, max_conn)
                idx += max_conn
                d_temp = clamped[:, idx].reshape(BS, N_neurons)
                idx += 1
                d_decay = clamped[:, idx].reshape(BS, N_neurons)
                self._mem_graph.apply_actions(d_prim, d_thresh, d_temp, d_decay)

                # Record done flag: did any stream reset during this action window?
                if did_reset is not None:
                    M = self.config.M_per_block
                    done_per_neuron = did_reset.unsqueeze(1).expand(BS, N_neurons).reshape(
                        BS * N_neurons
                    ).float()
                else:
                    done_per_neuron = torch.zeros(BS * N_neurons, device=device)
                action_dones.append(done_per_neuron)

                ppo_buffer.add(
                    obs=obs_flat,
                    action=action,
                    logprob=logprob,
                    reward=torch.zeros(BS * N_neurons, device=device),
                    value=value,
                    done=done_per_neuron,
                )
                action_step += 1
                prev_cc_surprise = mean_surp

        # ==========================================
        # Pass 2: Post-memory scan (parallel)
        # ==========================================
        logits = self.lm.forward_post_memory(H, mem_signals.to(dtype))

        # ==========================================
        # Compute per-token rewards for PPO
        # ==========================================
        if target_ids is not None and ppo_buffer is not None and action_step > 0:
            with torch.no_grad():
                per_token_ce = torch.nn.functional.cross_entropy(
                    logits.detach().reshape(-1, self.config.vocab_size),
                    target_ids.reshape(-1),
                    reduction='none',
                ).reshape(BS, T)

                # Mask EOT positions: don't reward/penalize cross-doc predictions
                reward_mask = (~eot_at).float()  # [BS, T]
                rewards = -per_token_ce * reward_mask  # [BS, T]

                # Aggregate: per action step, per CC (block-level credit)
                # Action at step_idx acted at token t_act = step_idx * action_every
                # Reward window = [t_act + action_every, t_act + 2*action_every)
                # (shifted by action_every: action can only influence NEXT window)
                M = self.config.M_per_block
                for step_idx in range(action_step):
                    # Reward from the NEXT window (action at t can't influence t)
                    r_start = (step_idx + 1) * action_every
                    r_end = min(r_start + action_every, T)
                    if r_start >= T:
                        # Last action has no future tokens in this chunk
                        ppo_buffer.rewards[step_idx] = torch.zeros(
                            BS * N_neurons, device=device
                        )
                        continue

                    window_rewards = rewards[:, r_start:r_end]  # [BS, r_end-r_start]
                    window_mask = reward_mask[:, r_start:r_end]
                    valid = window_mask.sum(dim=1).clamp(min=1.0)

                    # Per-stream base reward
                    per_stream_reward = window_rewards.sum(dim=1) / valid  # [BS]

                    # Per-CC surprise → per-block weighting
                    # Average CC surprises within each block
                    window_surp = surprise[:, r_start:r_end].detach()  # [BS, win, C, D_cc]
                    cc_surp_mag = window_surp.norm(dim=-1).mean(dim=1)  # [BS, C]
                    N_blocks = self.config.N_blocks
                    ccs_pb = self.config.CCs_per_block
                    # [BS, C] → [BS, N_blocks, ccs_pb] → mean → [BS, N_blocks]
                    block_surp = cc_surp_mag.view(BS, N_blocks, ccs_pb).mean(dim=2)
                    block_weights = block_surp / (block_surp.mean(dim=1, keepdim=True) + 1e-8)

                    # Per-block reward
                    block_rewards = per_stream_reward.unsqueeze(1) * block_weights  # [BS, N_blocks]

                    # Expand to per-neuron: each neuron in block gets its block's reward
                    M = self.config.M_per_block
                    neuron_rewards = block_rewards.unsqueeze(2).expand(
                        BS, N_blocks, M
                    ).reshape(BS * N_neurons)

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

    def _reset_carries(self, mask: Tensor):
        """Reset scan carries for masked streams."""
        if hasattr(self.lm, '_carries'):
            mask_f = (~mask).to(dtype=torch.float32).unsqueeze(-1)  # [BS, 1]
            for i, h in enumerate(self.lm._carries):
                if h is not None:
                    self.lm._carries[i] = h * mask_f

    def initialize_states(self, BS: int, device: torch.device):
        """Initialize all state (carries + memory)."""
        self.lm.initialize_carries()
        self._ensure_memory(BS, device, torch.float32)

    def detach_states(self):
        """TBPTT boundary: detach scan carries. Memory persists without detach."""
        self.lm.detach_carries()

    def param_count(self) -> int:
        """Total trained parameters (LM + neuromodulator)."""
        total = self.lm.param_count()
        total += sum(p.numel() for p in self.neuromod.parameters()
                     if p.requires_grad)
        return total
