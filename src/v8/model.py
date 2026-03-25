"""V8Model — split-scan + per-token memory graph + RL neuromodulator.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0-3) + PCM (parallel over T)
  2. Memory graph: per-token dynamics with neuromod actions per segment
     Real trajectory + counterfactual (K neurons reverted) per segment
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers 4-6) for both real and counterfactual paths
  5. Per-segment CE losses for real and counterfactual

RL update (every rl_collect_chunks chunks):
  6. Counterfactual advantage for K neurons: real_reward - cf_reward
  7. GAE with batch-mean baseline for remaining neurons
  8. Policy gradient (no value function)
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
        # obs_dim derived from config (matches MemoryGraph.obs_dim property)
        obs_dim = config.D_mem * 4 + 2  # prim + key + mean_in + mean_out + firing_rate + decay
        self.neuromod = Neuromodulator(config, obs_dim)
        self._obs_dim = obs_dim

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
        has_reset: bool = False,
        use_neuromod: bool = True,
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

        # Reset scan carries at doc boundaries (has_reset pre-computed on CPU)
        if has_reset and reset_mask is not None:
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

        if not self.config.lifelong_mode and has_reset and reset_mask is not None:
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
        mg = self._mem_graph

        for seg in range(n_segments):
            seg_cc = cc_segments[:, seg]
            eot_mask = eot_masks[:, seg] if eot_masks is not None else None
            t0 = seg * action_every

            self._segment_counter += 1
            sp_every = self.config.structural_plasticity_every
            needs_phi = (sp_every > 0
                         and self._segment_counter % sp_every == 0)

            # Neuromod observes and acts (if enabled)
            if use_neuromod:
                obs = mg.get_neuron_obs()
                obs_flat = obs.reshape(BS * N_neurons, -1)
                with torch.no_grad():
                    action, _, _, _ = self.neuromod.get_action_and_value(obs_flat)
                obs_list.append(obs_flat.detach())
                actions.append(action.detach())
                self._apply_neuromod_action(action, BS)

            # Run memory graph
            seg_out = mg.forward_segment(
                seg_cc, eot_mask=eot_mask,
                update_co_activation=needs_phi)
            mem_out[:, t0:t0 + action_every] = seg_out

            if needs_phi:
                mg.structural_plasticity()

        # ==========================================
        # Upper scan + output
        # ==========================================
        H_enriched = self.lm.inject_memory(H_mid, mem_out)
        H = self.lm.forward_scan_upper(H_enriched)
        logits = self.lm.forward_output(H)

        # ==========================================
        # Collect RL data (scoring happens at RL update time, not here)
        # ==========================================
        rl_data = None
        if use_memory and use_neuromod:
            rl_data = {
                "obs": obs_list,
                "actions": actions,
                "eot_at": eot_at,
                "n_segments": n_segments,
                "action_every": action_every,
                "cc_segments": cc_segments.detach(),
                "eot_masks": eot_masks,
                "H_mid": H_mid.detach(),
            }

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    @torch.no_grad()
    def score_trajectories(self, rl_data: dict, target_ids: Tensor) -> dict:
        """GRPO-style trajectory scoring for neuromod credit assignment.

        Picks K neurons, samples N trajectories with different actions for
        those neurons (same text, same other neurons). Scores each by CE loss.
        Returns per-neuron advantages based on ranking.

        Args:
            rl_data: from the most recent chunk's forward_chunk
            target_ids: [BS, T] target token IDs for CE computation

        Returns:
            dict with k_neurons, trajectory_losses, trajectory_actions, obs
        """
        BS = target_ids.shape[0]
        T = self.config.T
        N = self.config.N_neurons
        C = self.config.C
        D_mem = self.config.D_mem
        K = self.config.rl_counterfactual_k
        N_traj = self.config.rl_counterfactual_n
        action_every = rl_data["action_every"]
        n_segments = rl_data["n_segments"]
        cc_segments = rl_data["cc_segments"]  # [BS, n_seg, action_every, C, D_mem]
        eot_masks = rl_data["eot_masks"]
        H_mid = rl_data["H_mid"]  # [BS, T, D]
        device = target_ids.device
        dtype = H_mid.dtype
        mg = self._mem_graph

        # Pick K neurons for this scoring round (same across all trajectories)
        K = min(K, N)  # handle tiny configs where N < K
        k_neurons = torch.randperm(N, device=device)[:K]

        # Save current memory graph state (to restore after scoring)
        mg_state = mg.state_dict()

        # Save upper scan carries
        split = self.config.scan_split_at
        L = self.config.L_total
        saved_carries = [self.lm._carries[i].clone()
                         if self.lm._carries[i] is not None else None
                         for i in range(split, L)]

        # Get observation for the K neurons (from start of last segment)
        obs = rl_data["obs"][-1]  # [BS*N, obs_dim] from last segment

        # Cast obs to neuromod dtype
        nm_dtype = next(self.neuromod.parameters()).dtype
        obs = obs.to(nm_dtype)

        # Sample N trajectories with different actions for K neurons
        trajectory_losses = []  # [N_traj] list of scalar losses
        trajectory_actions = []  # [N_traj] list of [BS*N, act_dim]

        eot_id = self.config.eot_id
        eot_at = rl_data["eot_at"]
        reward_mask = (~eot_at).to(dtype=dtype)
        seg_mask = reward_mask.view(BS, n_segments, action_every)

        for traj_i in range(N_traj):
            # Restore memory graph to pre-scoring state
            mg.load_state_dict({k: v.clone() for k, v in mg_state.items()})

            # Sample fresh actions for ALL neurons
            action, _, _, _ = self.neuromod.get_action_and_value(obs)
            trajectory_actions.append(action.detach())

            # Run full chunk with these actions
            mem_out = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)
            for seg in range(n_segments):
                # Apply actions (only K neurons get varied actions;
                # other neurons keep the same action across trajectories
                # since the policy is deterministic given obs, but the
                # stochastic sampling gives different actions each time)
                self._apply_neuromod_action(action, BS)

                seg_cc = cc_segments[:, seg]
                eot_mask = eot_masks[:, seg] if eot_masks is not None else None
                seg_out = mg.forward_segment(seg_cc, eot_mask=eot_mask,
                                             update_co_activation=False)
                t0 = seg * action_every
                mem_out[:, t0:t0 + action_every] = seg_out

            # Upper scan + CE (restore carries each time)
            for i, c in enumerate(saved_carries):
                self.lm._carries[split + i] = c.clone() if c is not None else None

            H_enriched = self.lm.inject_memory(H_mid, mem_out)
            H = self.lm.forward_scan_upper(H_enriched)
            logits = self.lm.forward_output(H)

            # Per-segment losses
            ce = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                target_ids.reshape(-1),
                reduction='none',
            ).reshape(BS, T)
            seg_ce = ce.view(BS, n_segments, action_every)
            seg_losses = (seg_ce * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
            trajectory_losses.append(seg_losses.mean().item())  # scalar loss

            del mem_out, logits, H, H_enriched, ce

        # Restore memory graph and carries to real state
        mg.load_state_dict({k: v.clone() for k, v in mg_state.items()})
        for i, c in enumerate(saved_carries):
            self.lm._carries[split + i] = c

        # Rank trajectories — lower loss = better
        losses_t = torch.tensor(trajectory_losses, device=device)
        # GRPO advantage: z-score normalization across trajectories
        adv_per_traj = -(losses_t - losses_t.mean()) / losses_t.std().clamp(min=1e-8)
        # Positive = better than average trajectory

        return {
            "k_neurons": k_neurons,                # [K] — which neurons were varied
            "trajectory_advantages": adv_per_traj,  # [N_traj] — per-trajectory advantage
            "trajectory_actions": trajectory_actions,  # [N_traj] list of [BS*N, act_dim]
            "obs": obs,                             # [BS*N, obs_dim]
        }

    def compute_grpo_advantages(self, scoring_result: dict,
                                collected_rl_data: list[dict]) -> dict:
        """Combine GRPO scoring with GAE for non-scored neurons.

        Args:
            scoring_result: from score_trajectories
            collected_rl_data: list of per-chunk rl_data dicts (for GAE)

        Returns:
            Combined rl_data with per-neuron advantages for replay.
        """
        all_obs = []
        all_actions = []
        all_seg_rewards = []

        N = self.config.N_neurons

        for chunk_data in collected_rl_data:
            all_obs.extend(chunk_data["obs"])
            all_actions.extend(chunk_data["actions"])
            all_seg_rewards.append(chunk_data["seg_rewards"])

        seg_rewards = torch.cat(all_seg_rewards, dim=1)
        BS = seg_rewards.shape[0]
        total_segments = seg_rewards.shape[1]
        device = seg_rewards.device
        gamma = self.config.rl_gamma
        lam = self.config.rl_gae_lambda

        with torch.no_grad():
            # GAE for all neurons (batch-mean baseline)
            baseline = seg_rewards.mean(dim=0, keepdim=True)
            deltas = seg_rewards - baseline
            gae_advantages = torch.zeros_like(deltas)
            gae = torch.zeros(BS, device=device, dtype=deltas.dtype)
            for t in range(total_segments - 1, -1, -1):
                gae = deltas[:, t] + gamma * lam * gae
                gae_advantages[:, t] = gae

            # Per-neuron advantages [BS, total_seg, N]
            advantages = gae_advantages.unsqueeze(-1).expand(
                BS, total_segments, N).clone()

        return {
            "obs": all_obs,
            "actions": all_actions,
            "advantages": advantages,
            "scoring": scoring_result,  # passed through for GRPO replay
        }

    def replay_for_neuromod_grads(
        self, rl_data: dict,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Compute policy gradient from GAE + GRPO scoring.

        Two gradient sources:
        1. GAE replay: standard REINFORCE on collected trajectories
        2. GRPO replay: encourage actions from best-scoring trajectories

        Returns:
            dict with policy_loss, grpo_loss, entropy for logging
        """
        obs_list = rl_data["obs"]
        actions = rl_data["actions"]
        advantages = rl_data["advantages"]  # [BS, n_segments, N]
        scoring = rl_data.get("scoring")    # from score_trajectories

        N_neurons = self.config.N_neurons
        BS = actions[0].shape[0] // N_neurons
        device = actions[0].device
        n_segments = len(actions)

        # --- GAE replay (per-segment mini-batches) ---
        adv = advantages.permute(1, 0, 2)  # [n_seg, BS, N]
        total_policy_loss = 0.0
        total_entropy = 0.0

        for seg_i in range(n_segments):
            seg_obs = obs_list[seg_i]
            seg_act = actions[seg_i]
            seg_adv = adv[seg_i]

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                _, log_prob, entropy, _ = self.neuromod.get_action_and_value(
                    seg_obs, action=seg_act)

            log_prob = log_prob.reshape(BS, N_neurons)
            entropy = entropy.reshape(BS, N_neurons)

            seg_loss = -(seg_adv * log_prob).mean()
            seg_entropy = entropy.mean()
            seg_total = seg_loss - self.config.rl_entropy_coef * seg_entropy
            (seg_total / n_segments).backward()

            total_policy_loss += seg_loss.item()
            total_entropy += seg_entropy.item()

        # --- GRPO replay: encourage best trajectories' actions ---
        grpo_loss_val = 0.0
        if scoring is not None:
            traj_advs = scoring["trajectory_advantages"]  # [N_traj]
            traj_actions = scoring["trajectory_actions"]   # list of [BS*N, act_dim]
            obs = scoring["obs"]                           # [BS*N, obs_dim]

            # Replay each trajectory weighted by its advantage
            for traj_i, traj_adv in enumerate(traj_advs):
                if traj_adv.item() <= 0:
                    continue  # only encourage better-than-average trajectories

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                    _, log_prob, _, _ = self.neuromod.get_action_and_value(
                        obs, action=traj_actions[traj_i])

                grpo_loss = -(traj_adv * log_prob.mean())
                (grpo_loss / max(1, (traj_advs > 0).sum().item())).backward()
                grpo_loss_val += grpo_loss.item()

        return {
            "policy_loss": total_policy_loss / max(n_segments, 1),
            "grpo_loss": grpo_loss_val,
            "entropy": total_entropy / max(n_segments, 1),
        }

    def _apply_neuromod_action(self, action: Tensor, BS: int):
        """Clamp and apply additive neuromod action to the memory graph."""
        N = self.config.N_neurons
        D_mem = self.config.D_mem
        max_act = self.config.max_action_magnitude

        clamped = action.clamp(-max_act, max_act).reshape(BS, N, -1)
        idx = 0
        d_prim = clamped[:, :, idx:idx + D_mem]
        idx += D_mem
        d_key = clamped[:, :, idx:idx + D_mem]
        idx += D_mem
        d_decay = clamped[:, :, idx]
        self._mem_graph.apply_actions(d_prim, d_key, d_decay)

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
