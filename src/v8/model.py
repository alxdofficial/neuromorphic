"""V8Model — split-scan + per-token memory graph + per-segment RL.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0-3) + PCM (parallel over T)
  2. Memory graph: per-token receive → integrate → message (sequential)
     Neuromod acts every 256 tokens (8 segments)
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers 4-6) over memory-enriched H (parallel)
  5. Per-segment CE losses, discounted returns, batch-mean baseline
  6. Replay: log_prob with entropy bonus, per-step advantages
  Memory is injected MID-SCAN — upper layers see memory-enriched input.
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

        self.lm = V8LM(config)

        self._mem_graph = None
        obs_dim = config.D_mem * 3 + 3  # prim+mean_in+mean_out + firing_rate+decay+entropy
        self.neuromod = Neuromodulator(config, obs_dim)

        # RL discount factor for per-segment returns
        self._rl_gamma = 0.99

        # Structural plasticity counter (segments processed)
        self._segment_counter = 0

    def _ensure_memory(self, BS: int, device: torch.device,
                       dtype: torch.dtype = torch.bfloat16):
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
    ) -> dict:
        """Process a full T-token chunk with split-scan memory injection.

        1. Lower scan (layers 0..split-1) → H_mid (parallel over T)
        2. Memory graph: per-token receive → integrate → message (sequential)
        3. Inject: H_enriched = H_mid + gate * mem_signals
        4. Upper scan (layers split..L-1) + PCM (parallel over T)
        5. Output head → logits

        The upper scan layers see memory-enriched representations.
        """
        BS, T = input_ids.shape
        C = self.config.C
        D = self.config.D
        D_mem = self.config.D_mem
        action_every = self.config.action_every
        eot_id = self.config.eot_id
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # Reset scan carries at doc boundaries
        if reset_mask is not None and reset_mask.any():
            self._reset_carries(reset_mask)

        # ==========================================
        # Lower scan (layers 0..split-1) + PCM
        # ==========================================
        H_mid, x, surprise, aux_loss = self.lm.forward_scan_lower(input_ids)

        # --- No-memory fast path ---
        if not use_memory:
            H = self.lm.forward_scan_upper(H_mid)
            logits = self.lm.forward_output(H)
            return {
                "logits": logits,
                "aux_loss": aux_loss,
                "surprise": surprise.detach(),
                "rl_data": None,
            }

        # ==========================================
        # Memory graph: per-token processing
        # ==========================================
        self._ensure_memory(BS, device, dtype)

        if not self.config.lifelong_mode and reset_mask is not None and reset_mask.any():
            self._mem_graph.reset_streams(reset_mask)

        # CC signals from lower scan (detached for memory graph input)
        cc_signals_all = H_mid.detach().view(BS, T, C, D_mem)
        eot_at = (input_ids == eot_id)

        n_segments = T // action_every
        N_neurons = self.config.N_neurons

        # Pre-slice CC signals and pre-compute EOT masks for all segments
        cc_segments = cc_signals_all.view(BS, n_segments, action_every, C, D_mem)
        eot_masks = None
        if not self.config.lifelong_mode:
            eot_shifted = torch.zeros(BS, T, dtype=torch.bool, device=device)
            eot_shifted[:, 1:] = eot_at[:, :-1]
            eot_masks = eot_shifted.view(BS, n_segments, action_every)

        mem_out = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)
        actions = []
        obs_list = []

        for seg in range(n_segments):
            # 1. Neuromod observes and acts
            obs = self._mem_graph.get_neuron_obs()
            obs_flat = obs.reshape(BS * N_neurons, -1)

            with torch.no_grad():
                action, _, _, _ = self.neuromod.get_action_and_value(obs_flat)

            obs_list.append(obs_flat.detach())
            actions.append(action.detach())
            self._apply_neuromod_action(action, BS)

            # 2. Run memory graph for this segment
            seg_cc = cc_segments[:, seg]
            eot_mask = eot_masks[:, seg] if eot_masks is not None else None
            seg_out = self._mem_graph.forward_segment(seg_cc, eot_mask=eot_mask)
            t0 = seg * action_every
            mem_out[:, t0:t0 + action_every] = seg_out

            # 3. Structural plasticity at configured cadence
            self._segment_counter += 1
            sp_every = self.config.structural_plasticity_every
            if sp_every > 0 and self._segment_counter % sp_every == 0:
                self._mem_graph.structural_plasticity()

        mem_signals = mem_out

        # ==========================================
        # Inject memory into H_mid, then upper scan
        # ==========================================
        H_enriched = self.lm.inject_memory(H_mid, mem_signals)
        H = self.lm.forward_scan_upper(H_enriched)

        # ==========================================
        # Output head
        # ==========================================
        logits = self.lm.forward_output(H)

        # ==========================================
        # Per-segment RL: dense rewards + discounted returns
        # ==========================================
        rl_data = None
        if target_ids is not None:
            reward_mask = (~eot_at).to(dtype=dtype)
            with torch.no_grad():
                per_token_ce = F.cross_entropy(
                    logits.detach().reshape(-1, self.config.vocab_size),
                    target_ids.reshape(-1),
                    reduction='none',
                ).reshape(BS, T)

                # Per-segment CE losses: [BS, n_segments]
                seg_ce = per_token_ce.view(BS, n_segments, action_every)
                seg_mask = reward_mask.view(BS, n_segments, action_every)
                seg_losses = (seg_ce * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1).clamp(min=1)

                # Per-segment rewards (negative loss = reward)
                seg_rewards = -seg_losses  # [BS, n_segments]

                # Discounted returns: G_t = r_t + gamma*r_{t+1} + ... + gamma^(H-1-t)*r_{H-1}
                gamma = self._rl_gamma
                returns = torch.zeros_like(seg_rewards)  # [BS, n_segments]
                returns[:, -1] = seg_rewards[:, -1]
                for t in range(n_segments - 2, -1, -1):
                    returns[:, t] = seg_rewards[:, t] + gamma * returns[:, t + 1]

                # Batch-mean baseline per step: [n_segments]
                baseline = returns.mean(dim=0)

                # Per-step advantage: [BS, n_segments]
                advantages = returns - baseline.unsqueeze(0)

                # Chunk-level loss for logging
                chunk_loss = (per_token_ce * reward_mask).sum() / reward_mask.sum().clamp(min=1)
                loss_val = chunk_loss.item()

            rl_data = {
                "obs": obs_list,              # list of n_segments [BS*N, obs_dim] tensors
                "actions": actions,           # list of n_segments [BS*N, act_dim] tensors
                "advantages": advantages,     # [BS, n_segments]
                "seg_losses": seg_losses.detach(),  # [BS, n_segments] for logging
                "loss": loss_val,             # scalar chunk loss for logging
            }

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    def replay_for_neuromod_grads(
        self, rl_data: dict,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> float:
        """Compute policy gradient via replay with per-segment advantages.

        Uses stored per-segment obs for accurate log_prob evaluation.
        Each segment's actions are weighted by its own advantage.

        Returns:
            policy_loss_val: scalar loss value for logging
        """
        obs_list = rl_data["obs"]          # list of n_seg [BS*N, obs_dim]
        actions = rl_data["actions"]       # list of n_seg [BS*N, act_dim]
        advantages = rl_data["advantages"] # [BS, n_segments]

        N_neurons = self.config.N_neurons
        BS = actions[0].shape[0] // N_neurons
        device = actions[0].device
        n_segments = len(actions)

        # Batch all segments into single tensors for one forward pass
        all_obs = torch.stack(obs_list)        # [n_seg, BS*N, obs_dim]
        all_actions = torch.stack(actions)      # [n_seg, BS*N, act_dim]
        flat_obs = all_obs.reshape(-1, all_obs.shape[-1])
        flat_actions = all_actions.reshape(-1, all_actions.shape[-1])

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
            _, log_prob, entropy, _ = self.neuromod.get_action_and_value(
                flat_obs, action=flat_actions,
            )

        # log_prob: [n_seg * BS * N] — reshape to [n_seg, BS, N]
        log_prob = log_prob.reshape(n_segments, BS, N_neurons)
        entropy = entropy.reshape(n_segments, BS, N_neurons)

        # Per-segment advantage: [BS, n_seg] → [n_seg, BS, 1] for broadcasting
        adv = advantages.T.unsqueeze(-1)  # [n_seg, BS, 1]

        # Weighted policy loss + entropy bonus to prevent std collapse
        policy_loss = -(adv * log_prob).mean()
        entropy_bonus = -0.01 * entropy.mean()  # encourage exploration
        total_loss = policy_loss + entropy_bonus
        total_loss.backward()

        return policy_loss.item()

    def _apply_neuromod_action(self, action: Tensor, BS: int):
        """Clamp and apply a neuromod action to the memory graph."""
        N = self.config.N_neurons
        D_mem = self.config.D_mem
        K_conn = self.config.K_connections
        max_act = self.config.max_action_magnitude

        clamped = action.clamp(-max_act, max_act).reshape(BS, N, -1)
        idx = 0
        d_prim = clamped[:, :, idx:idx + D_mem]
        idx += D_mem
        d_conn = clamped[:, :, idx:idx + K_conn]
        idx += K_conn
        d_decay = clamped[:, :, idx]
        self._mem_graph.apply_actions(d_prim, d_conn, d_decay)

    def _save_mem_state(self) -> dict:
        mg = self._mem_graph
        state = mg.state_dict()
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()}

    def _restore_mem_state(self, state: dict):
        mg = self._mem_graph
        restored = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in state.items()}
        mg.load_state_dict(restored)

    def _reset_carries(self, mask: Tensor):
        if hasattr(self.lm, '_carries'):
            for i, h in enumerate(self.lm._carries):
                if h is not None:
                    mask_f = (~mask).to(dtype=h.dtype).unsqueeze(-1)
                    self.lm._carries[i] = h * mask_f

    def initialize_states(self, BS: int, device: torch.device):
        self.lm.initialize_carries()
        lm_dtype = next(self.lm.parameters()).dtype
        self._ensure_memory(BS, device, lm_dtype)

    def detach_states(self):
        self.lm.detach_carries()

    def param_count(self) -> int:
        total = self.lm.param_count()
        total += sum(p.numel() for p in self.neuromod.parameters()
                     if p.requires_grad)
        return total
