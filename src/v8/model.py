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

        mem_out_A = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)
        mem_out_B = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype) if use_neuromod else None
        actions = []
        obs_list = []
        k_indices_list = []
        K = self.config.rl_counterfactual_k
        mg = self._mem_graph

        for seg in range(n_segments):
            seg_cc = cc_segments[:, seg]
            eot_mask = eot_masks[:, seg] if eot_masks is not None else None
            t0 = seg * action_every

            self._segment_counter += 1
            sp_every = self.config.structural_plasticity_every
            needs_phi = (sp_every > 0
                         and self._segment_counter % sp_every == 0)

            if not use_neuromod:
                # Phase 1: no neuromod, just run memory graph
                seg_out = mg.forward_segment(
                    seg_cc, eot_mask=eot_mask,
                    update_co_activation=needs_phi)
                mem_out_A[:, t0:t0 + action_every] = seg_out
                if needs_phi:
                    mg.structural_plasticity()
                continue

            # --- Phase 2: Neuromod + counterfactual ---

            # 1. Observe and sample action
            obs = mg.get_neuron_obs()
            obs_flat = obs.reshape(BS * N_neurons, -1)
            with torch.no_grad():
                action, _, _, _ = self.neuromod.get_action_and_value(obs_flat)
            obs_list.append(obs_flat.detach())
            actions.append(action.detach())

            # 2. Pick K random neurons for counterfactual evaluation
            k_idx = torch.randperm(N_neurons, device=device)[:K]
            k_indices_list.append(k_idx)

            # 3. Save pre-action state for K neurons
            saved_prim_k = mg.primitives[:, k_idx].clone()
            saved_key_k = mg.key[:, k_idx].clone()
            saved_decay_k = mg.decay_logit[:, k_idx].clone()

            # 4. Apply ALL neuromod actions
            self._apply_neuromod_action(action, BS)

            # 5. Save pre-segment dynamic state
            pre_h = mg.h.clone()
            pre_msg = mg.prev_messages.clone()

            # Save actioned params (to restore after counterfactual)
            actioned_prim = mg.primitives.clone()
            actioned_key = mg.key.clone()
            actioned_decay = mg.decay_logit.clone()

            # --- REAL trajectory ---
            seg_out_A = mg.forward_segment(
                seg_cc, eot_mask=eot_mask,
                update_co_activation=needs_phi)
            mem_out_A[:, t0:t0 + action_every] = seg_out_A

            # Save post-segment state
            post_h = mg.h.clone()
            post_msg = mg.prev_messages.clone()
            post_firing = mg.firing_rate.clone()
            post_mean_in = mg.mean_input.clone()
            post_mean_out = mg.mean_output.clone()

            # --- COUNTERFACTUAL trajectory ---
            # Restore pre-segment state
            mg.h = pre_h
            mg.prev_messages = pre_msg

            # Revert K neurons to pre-action values
            mg.primitives[:, k_idx] = saved_prim_k
            mg.key[:, k_idx] = saved_key_k
            mg.decay_logit[:, k_idx] = saved_decay_k

            seg_out_B = mg.forward_segment(
                seg_cc, eot_mask=eot_mask,
                update_co_activation=False)  # no plasticity stats for counterfactual
            mem_out_B[:, t0:t0 + action_every] = seg_out_B

            # --- Restore real trajectory state ---
            mg.h = post_h
            mg.prev_messages = post_msg
            mg.primitives = actioned_prim
            mg.key = actioned_key
            mg.decay_logit = actioned_decay
            mg.firing_rate = post_firing
            mg.mean_input = post_mean_in
            mg.mean_output = post_mean_out

            # Structural plasticity (real trajectory only)
            if needs_phi:
                mg.structural_plasticity()

        # ==========================================
        # Real + counterfactual upper scan paths
        # ==========================================
        split = self.config.scan_split_at
        L = self.config.L_total

        # Save upper-scan carries BEFORE the real scan (for counterfactual)
        pre_upper_carries = None
        if use_neuromod:
            pre_upper_carries = [self.lm._carries[i].clone()
                                 if self.lm._carries[i] is not None else None
                                 for i in range(split, L)]

        # Real path
        H_enriched_A = self.lm.inject_memory(H_mid, mem_out_A)
        H_A = self.lm.forward_scan_upper(H_enriched_A)
        logits_A = self.lm.forward_output(H_A)

        # Counterfactual path (Phase 2 only)
        logits_B = None
        if use_neuromod:
            # Save post-real carries (to restore after counterfactual)
            post_real_carries = [self.lm._carries[i].clone()
                                 if self.lm._carries[i] is not None else None
                                 for i in range(split, L)]

            # Restore pre-upper carries for counterfactual scan
            for i, c in enumerate(pre_upper_carries):
                self.lm._carries[split + i] = c

            with torch.no_grad():
                H_enriched_B = self.lm.inject_memory(H_mid.detach(), mem_out_B)
                H_B = self.lm.forward_scan_upper(H_enriched_B)
                logits_B = self.lm.forward_output(H_B)

            # Restore real carries for continuation
            for i, c in enumerate(post_real_carries):
                self.lm._carries[split + i] = c

        # ==========================================
        # Collect RL data
        # ==========================================
        rl_data = None
        if use_memory and use_neuromod:
            rl_data = {
                "obs": obs_list,
                "actions": actions,
                "eot_at": eot_at,
                "n_segments": n_segments,
                "action_every": action_every,
                "k_indices": k_indices_list,
                "logits_B": logits_B,
            }

        return {
            "logits": logits_A,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    def compute_counterfactual_advantages(
        self, collected_rl_data: list[dict],
    ) -> dict:
        """Compute per-neuron advantages using counterfactual baseline + GAE.

        For K counterfactual neurons per segment: advantage = loss_cf - loss_real
        (positive means their actions helped — reverting them hurts).
        For other neurons: GAE with batch-mean baseline.

        Args:
            collected_rl_data: list of per-chunk rl_data dicts

        Returns:
            Combined rl_data dict with obs, actions, and per-neuron advantages.
        """
        all_obs = []
        all_actions = []
        all_seg_rewards = []
        all_cf_rewards = []
        all_k_indices = []

        N = self.config.N_neurons

        for chunk_data in collected_rl_data:
            all_obs.extend(chunk_data["obs"])
            all_actions.extend(chunk_data["actions"])
            all_seg_rewards.append(chunk_data["seg_rewards"])
            all_cf_rewards.append(chunk_data["seg_rewards_cf"])
            all_k_indices.extend(chunk_data["k_indices"])

        seg_rewards = torch.cat(all_seg_rewards, dim=1)  # [BS, total_seg]
        cf_rewards = torch.cat(all_cf_rewards, dim=1)    # [BS, total_seg]
        BS = seg_rewards.shape[0]
        total_segments = seg_rewards.shape[1]
        device = seg_rewards.device
        gamma = self.config.rl_gamma
        lam = self.config.rl_gae_lambda

        with torch.no_grad():
            # --- GAE for batch-mean baseline (all neurons) ---
            baseline = seg_rewards.mean(dim=0, keepdim=True)  # [1, total_seg]
            deltas = seg_rewards - baseline  # [BS, total_seg]

            gae_advantages = torch.zeros_like(deltas)
            gae = torch.zeros(BS, device=device, dtype=deltas.dtype)
            for t in range(total_segments - 1, -1, -1):
                gae = deltas[:, t] + gamma * lam * gae
                gae_advantages[:, t] = gae

            # --- Per-neuron advantages [BS, total_seg, N] ---
            advantages = gae_advantages.unsqueeze(-1).expand(
                BS, total_segments, N).clone()  # [BS, total_seg, N]

            # Override counterfactual neurons with their direct advantage
            # Normalize CF advantages to match GAE scale
            gae_std = gae_advantages.std().clamp(min=1e-8)
            for seg_idx, k_idx in enumerate(all_k_indices):
                # cf_advantage: positive = real better than counterfactual = actions helped
                cf_adv = seg_rewards[:, seg_idx] - cf_rewards[:, seg_idx]  # [BS]
                # Scale CF advantages to have similar magnitude as GAE advantages
                cf_adv_normalized = cf_adv / cf_adv.abs().mean().clamp(min=1e-8) * gae_std
                advantages[:, seg_idx, k_idx] = cf_adv_normalized.unsqueeze(-1).expand(
                    BS, len(k_idx))

        return {
            "obs": all_obs,
            "actions": all_actions,
            "advantages": advantages,  # [BS, total_segments, N]
        }

    def replay_for_neuromod_grads(
        self, rl_data: dict,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Compute policy gradient via replay with per-neuron advantages.

        Uses counterfactual advantages for evaluated neurons and GAE
        batch-mean advantages for others. No learned value function.

        Returns:
            dict with policy_loss and entropy for logging
        """
        obs_list = rl_data["obs"]          # list of n_seg [BS*N, obs_dim]
        actions = rl_data["actions"]       # list of n_seg [BS*N, act_dim]
        advantages = rl_data["advantages"] # [BS, n_segments, N]

        N_neurons = self.config.N_neurons
        BS = actions[0].shape[0] // N_neurons
        device = actions[0].device
        n_segments = len(actions)

        # Batch all segments for one forward pass
        all_obs = torch.stack(obs_list)        # [n_seg, BS*N, obs_dim]
        all_actions = torch.stack(actions)      # [n_seg, BS*N, act_dim]
        flat_obs = all_obs.reshape(-1, all_obs.shape[-1])
        flat_actions = all_actions.reshape(-1, all_actions.shape[-1])

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
            _, log_prob, entropy, _ = self.neuromod.get_action_and_value(
                flat_obs, action=flat_actions,
            )

        # [n_seg * BS * N] → [n_seg, BS, N]
        log_prob = log_prob.reshape(n_segments, BS, N_neurons)
        entropy = entropy.reshape(n_segments, BS, N_neurons)

        # Per-neuron advantage: [BS, n_seg, N] → [n_seg, BS, N]
        adv = advantages.permute(1, 0, 2)  # [n_seg, BS, N]

        # Weighted policy loss + entropy bonus
        policy_loss = -(adv * log_prob).mean()
        entropy_bonus = -self.config.rl_entropy_coef * entropy.mean()

        total_loss = policy_loss + entropy_bonus
        total_loss.backward()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.mean().item(),
        }

    def _apply_neuromod_action(self, action: Tensor, BS: int):
        """Apply neuromod action to the memory graph.

        Primitives and key are direct predictions (not deltas) — the neuromod
        outputs what the new values should be. Decay is still additive delta.
        """
        N = self.config.N_neurons
        D_mem = self.config.D_mem
        max_act = self.config.max_action_magnitude

        raw = action.reshape(BS, N, -1)
        idx = 0
        # Primitives and key: direct prediction (no clamp — normalization handles scale)
        new_prim = raw[:, :, idx:idx + D_mem]
        idx += D_mem
        new_key = raw[:, :, idx:idx + D_mem]
        idx += D_mem
        # Decay: additive delta (clamped)
        d_decay = raw[:, :, idx].clamp(-max_act, max_act)
        self._mem_graph.apply_actions(new_prim, new_key, d_decay)

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
