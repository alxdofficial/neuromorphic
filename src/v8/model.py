"""V8Model — single-pass scan + memory graph + sampling-based RL.

Training flow per chunk (T=2048 tokens):
  1. Full scan + PCM over all T tokens (parallel, once)           ← expensive, shared
  2. Sample K neuromodulator action trajectories
  3. For each sample k: run memory loop + cheap output head       ← cheap per sample
  4. Compare losses across samples → advantage → policy gradient
  No critic, no value function, no GAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config
from .lm import V8LM
from .memory_graph import MemoryGraph
from .neuromodulator import Neuromodulator


class V8Model(nn.Module):
    """Top-level v8 model: LM + Memory Graph + Neuromodulator."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config

        # Language model (in autograd graph)
        self.lm = V8LM(config)

        # Neuromodulator (actor only — no critic needed for sampling-based RL)
        self._mem_graph = None
        obs_dim = config.D_cc * 4 + 4  # D_mem*3 + usage + temp + decay + entropy + D_cc surprise
        self.neuromod = Neuromodulator(config, obs_dim)

    def _ensure_memory(self, BS: int, device: torch.device,
                       dtype: torch.dtype = torch.float32):
        if (self._mem_graph is not None
                and self._mem_graph.is_initialized()
                and self._mem_graph.primitives.shape[0] == BS):
            return
        self._mem_graph = MemoryGraph(self.config, device, dtype)
        self._mem_graph.initialize(BS)

    @property
    def memory(self) -> MemoryGraph:
        return self._mem_graph

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        use_memory: bool = True,
        n_samples: int = 1,
    ) -> dict:
        """Process a full T-token chunk.

        Single scan pass (shared), then K memory+output samples for RL.

        Args:
            input_ids:  [BS, T]
            target_ids: [BS, T] or None
            reset_mask: [BS] bool
            use_memory: if False, skip memory (LM-only baseline)
            n_samples:  number of action trajectories to sample for RL

        Returns:
            dict with:
                logits:     [BS, T, vocab] — from best sample (or no-memory)
                aux_loss:   scalar (PCM)
                surprise:   [BS, T, C, D_cc] (detached)
                rl_data:    dict with sample losses + actions for policy update, or None
        """
        BS, T = input_ids.shape
        C = self.config.C
        D_mem = self.config.D_mem
        D_cc = self.config.D_cc
        action_every = self.config.action_every
        eot_id = self.config.eot_id
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # Reset scan carries at doc boundaries
        if reset_mask is not None and reset_mask.any():
            self._reset_carries(reset_mask)

        # ==========================================
        # Full scan + PCM (once, shared across all samples)
        # ==========================================
        H, x, surprise, aux_loss = self.lm.forward_scan(input_ids)
        # H: [BS, T, D], surprise: [BS, T, C, D_cc]

        # --- No-memory fast path ---
        if not use_memory:
            logits = self.lm.forward_output(H, mem_signals=None)
            return {
                "logits": logits,
                "aux_loss": aux_loss,
                "surprise": surprise.detach(),
                "rl_data": None,
            }

        # --- Memory path ---
        self._ensure_memory(BS, device, torch.float32)

        if reset_mask is not None and reset_mask.any():
            self._mem_graph.reset_streams(reset_mask)

        # CC→memory signals: raw H sliced into per-CC columns
        cc_signals_all = H.detach().view(BS, T, C, D_mem).float()

        # Precompute EOT positions
        eot_at = (input_ids == eot_id)

        # ==========================================
        # Sample K action trajectories
        # ==========================================
        # Save memory graph state so we can restore between samples
        saved_state = self._save_mem_state()

        sample_logits = []
        sample_losses = []
        sample_log_probs = []  # mean of log_probs across all actions in trajectory
        sample_end_states = []  # per-sample memory end state for best restoration

        for k in range(n_samples):
            # Restore memory to start-of-chunk state for each sample
            if k > 0:
                self._restore_mem_state(saved_state)

            # Run memory loop for this sample
            mem_signals, traj_log_prob = self._run_memory_trajectory(
                cc_signals_all, surprise, eot_at, BS, T, C, D_mem, D_cc,
                action_every, device,
            )

            # Save this sample's end state
            sample_end_states.append(self._save_mem_state())

            # Cheap output: end-injection + output head
            logits_k = self.lm.forward_output(H, mem_signals.to(dtype))
            sample_logits.append(logits_k)
            sample_log_probs.append(traj_log_prob)

            # Compute loss for this sample if targets available
            if target_ids is not None:
                with torch.no_grad():
                    per_token_ce = F.cross_entropy(
                        logits_k.detach().reshape(-1, self.config.vocab_size),
                        target_ids.reshape(-1),
                        reduction='none',
                    ).reshape(BS, T)
                    reward_mask = (~eot_at).float()
                    masked_loss = (per_token_ce * reward_mask).sum() / reward_mask.sum().clamp(min=1)
                    sample_losses.append(masked_loss)

        # ==========================================
        # Compute RL data for policy update
        # ==========================================
        rl_data = None
        if target_ids is not None and n_samples > 1:
            # Advantage: sample loss vs mean loss (lower loss = higher advantage)
            losses = torch.stack(sample_losses)  # [K]
            mean_loss = losses.mean()
            # Advantage = negative (loss - mean_loss) → better-than-average samples get positive advantage
            advantages = -(losses - mean_loss)  # [K]
            # Normalize
            adv_std = advantages.std().clamp(min=1e-8)
            advantages = advantages / adv_std

            rl_data = {
                "log_probs": sample_log_probs,     # list of K tensors
                "advantages": advantages,           # [K]
                "losses": losses.detach(),           # [K]
                "mean_loss": mean_loss.item(),
            }

        # Use the best sample's logits for LM loss and restore its memory state
        if sample_losses:
            best_idx = torch.stack(sample_losses).argmin().item()
            best_logits = sample_logits[best_idx]
            self._restore_mem_state(sample_end_states[best_idx])
        else:
            best_logits = sample_logits[0]
            if sample_end_states:
                self._restore_mem_state(sample_end_states[0])

        return {
            "logits": best_logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    def _run_memory_trajectory(
        self, cc_signals_all, surprise, eot_at,
        BS, T, C, D_mem, D_cc, action_every, device,
    ) -> tuple[Tensor, Tensor]:
        """Run one memory trajectory with sampled neuromod actions.

        Returns:
            mem_signals: [BS, T, C, D_mem]
            traj_log_prob: scalar — mean of log_probs for all actions in trajectory
        """
        N_neurons = self.config.N_neurons
        mem_signals = torch.zeros(BS, T, C, D_mem, device=device)
        traj_log_prob = torch.tensor(0.0, device=device)
        prev_cc_surprise = torch.zeros(BS, C, D_cc, device=device)

        for t in range(T):
            # Doc boundary: reset memory graph if previous token was EOT
            # (scan carries are NOT reset here — scan already completed)
            if not self.config.lifelong_mode and t > 0:
                eot_prev = eot_at[:, t - 1]
                if eot_prev.any():
                    self._mem_graph.reset_streams(eot_prev)

            # Step memory graph
            with torch.autocast(device_type=device.type, enabled=False):
                mem_out = self._mem_graph.step(cc_signals_all[:, t])
            mem_signals[:, t] = mem_out

            # Neuromodulator acts every action_every tokens
            if t % action_every == 0:
                if t > 0:
                    recent_surp = surprise[:, max(0, t - action_every):t].detach()
                    mean_surp = recent_surp.mean(dim=1)
                else:
                    mean_surp = prev_cc_surprise

                obs = self._mem_graph.get_neuron_obs(
                    cc_surprise=mean_surp.float()
                )
                obs_flat = obs.reshape(BS * N_neurons, -1)

                # Sample action from policy (use sample(), not rsample() — REINFORCE)
                action, logprob, _, _ = self.neuromod.get_action_and_value(obs_flat)
                traj_log_prob = traj_log_prob + logprob.mean()

                # Clamp and apply
                max_act = self.config.max_action_magnitude
                clamped = action.clamp(-max_act, max_act)
                idx = 0
                d_prim = clamped[:, idx:idx + D_mem].reshape(BS, N_neurons, D_mem)
                idx += D_mem
                max_conn = self.config.max_connections
                d_thresh = clamped[:, idx:idx + max_conn].reshape(BS, N_neurons, max_conn)
                idx += max_conn
                d_temp = clamped[:, idx].reshape(BS, N_neurons)
                idx += 1
                d_decay = clamped[:, idx].reshape(BS, N_neurons)
                self._mem_graph.apply_actions(d_prim, d_thresh, d_temp, d_decay)

                prev_cc_surprise = mean_surp

        return mem_signals, traj_log_prob

    def _save_mem_state(self) -> dict:
        """Save memory graph state for restoring between samples."""
        mg = self._mem_graph
        return {
            'primitives': mg.primitives.clone(),
            'thresholds': mg.thresholds.clone(),
            'temperature': mg.temperature.clone(),
            'decay': mg.decay.clone(),
            'activations': mg.activations.clone(),
            'prev_output': mg.prev_output.clone(),
            'mean_input': mg.mean_input.clone(),
            'mean_output': mg.mean_output.clone(),
            'usage_count': mg.usage_count.clone(),
        }

    def _restore_mem_state(self, state: dict):
        """Restore memory graph state from saved snapshot."""
        mg = self._mem_graph
        mg.primitives = state['primitives'].clone()
        mg.thresholds = state['thresholds'].clone()
        mg.temperature = state['temperature'].clone()
        mg.decay = state['decay'].clone()
        mg.activations = state['activations'].clone()
        mg.prev_output = state['prev_output'].clone()
        mg.mean_input = state['mean_input'].clone()
        mg.mean_output = state['mean_output'].clone()
        mg.usage_count = state['usage_count'].clone()

    def _reset_carries(self, mask: Tensor):
        """Reset scan carries for masked streams."""
        if hasattr(self.lm, '_carries'):
            mask_f = (~mask).to(dtype=torch.float32).unsqueeze(-1)
            for i, h in enumerate(self.lm._carries):
                if h is not None:
                    self.lm._carries[i] = h * mask_f

    def initialize_states(self, BS: int, device: torch.device):
        self.lm.initialize_carries()
        self._ensure_memory(BS, device, torch.float32)

    def detach_states(self):
        self.lm.detach_carries()

    def param_count(self) -> int:
        total = self.lm.param_count()
        total += sum(p.numel() for p in self.neuromod.parameters()
                     if p.requires_grad)
        return total
