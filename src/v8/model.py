"""V8Model — split-scan + per-token memory graph + RL neuromodulator.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0-3) + PCM (parallel over T)
  2. Memory graph: per-token dynamics with neuromod actions per segment
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers 4-6) + output head → logits

RL update (every rl_collect_chunks chunks):
  5. GRPO scoring: replay last chunk with N sampled trajectories, rank by CE
  6. Policy gradient: encourage best trajectories' actions (scored neurons only)
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

        # Save upper scan carries BEFORE this chunk's forward (for GRPO scoring)
        split = self.config.scan_split_at
        L = self.config.L_total
        pre_upper_carries = [
            self.lm._carries[i].clone() if self.lm._carries[i] is not None else None
            for i in range(split, L)
        ] if use_neuromod else None

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
                "pre_upper_carries": pre_upper_carries,
            }

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    @torch.no_grad()
    def score_trajectories(self, rl_data: dict, target_ids: Tensor) -> dict:
        """GRPO trajectory scoring with spatial specificity.

        Picks K neurons. Only those K neurons get varied across trajectories;
        the other N-K neurons use identical (baseline) actions in every trajectory.
        This isolates the K neurons' contribution to the loss difference.

        Args:
            rl_data: from the most recent chunk's forward_chunk
            target_ids: [BS, T] target token IDs for CE computation

        Returns:
            dict with k_neurons, trajectory advantages, and per-segment
            actions/obs for the K neurons only (for replay).
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
        cc_segments = rl_data["cc_segments"]
        eot_masks = rl_data["eot_masks"]
        H_mid = rl_data["H_mid"]
        device = target_ids.device
        dtype = H_mid.dtype
        mg = self._mem_graph

        K = min(K, N)
        k_neurons = torch.randperm(N, device=device)[:K]

        # Save memory graph state (to restore after scoring)
        mg_state = mg.state_dict()

        # Use pre-forward carries (saved before this chunk's upper scan)
        split = self.config.scan_split_at
        L = self.config.L_total
        pre_carries = rl_data.get("pre_upper_carries")
        if pre_carries is None:
            # Fallback: use current carries (less accurate but won't crash)
            pre_carries = [self.lm._carries[i].clone()
                           if self.lm._carries[i] is not None else None
                           for i in range(split, L)]

        nm_dtype = next(self.neuromod.parameters()).dtype
        act_dim = self.neuromod.act_dim

        trajectory_losses = torch.empty(N_traj, device=device)
        # Only store K neurons' actions/obs for replay (not all N)
        trajectory_k_actions = []  # [N_traj][n_seg] of [BS*K, act_dim]
        trajectory_k_obs = []      # [N_traj][n_seg] of [BS*K, obs_dim]

        eot_at = rl_data["eot_at"]
        reward_mask = (~eot_at).to(dtype=dtype)
        seg_mask = reward_mask.view(BS, n_segments, action_every)

        # k_neurons indices for extracting K neurons from [BS*N] flat tensors
        # k_idx_flat[b*K + k] = b*N + k_neurons[k]
        batch_offsets = torch.arange(BS, device=device).unsqueeze(1) * N  # [BS, 1]
        k_idx_flat = (batch_offsets + k_neurons.unsqueeze(0)).reshape(-1)  # [BS*K]

        # Trajectory 0 establishes baseline actions for non-K neurons
        baseline_actions = []  # [n_seg] of [BS*N, act_dim] — fixed for all trajectories

        for traj_i in range(N_traj):
            mg.load_state_dict({k: v.clone() for k, v in mg_state.items()})

            mem_out = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)
            traj_k_actions = []
            traj_k_obs = []

            for seg in range(n_segments):
                seg_obs = mg.get_neuron_obs().reshape(BS * N, -1).to(nm_dtype)
                action, _, _, _ = self.neuromod.get_action_and_value(seg_obs)

                if traj_i == 0:
                    # First trajectory: save full action as baseline
                    baseline_actions.append(action.detach().clone())
                else:
                    # Subsequent trajectories: only K neurons get new actions,
                    # non-K neurons use baseline from trajectory 0
                    base = baseline_actions[seg].clone()
                    base[k_idx_flat] = action[k_idx_flat]
                    action = base

                # Store only K neurons' obs and actions for replay
                traj_k_obs.append(seg_obs[k_idx_flat].detach())
                traj_k_actions.append(action[k_idx_flat].detach())

                self._apply_neuromod_action(action, BS)

                seg_cc = cc_segments[:, seg]
                eot_mask = eot_masks[:, seg] if eot_masks is not None else None
                seg_out = mg.forward_segment(seg_cc, eot_mask=eot_mask,
                                             update_co_activation=False)
                t0 = seg * action_every
                mem_out[:, t0:t0 + action_every] = seg_out

            trajectory_k_actions.append(traj_k_actions)
            trajectory_k_obs.append(traj_k_obs)

            # Upper scan + CE (restore pre-forward carries each time)
            for i, c in enumerate(pre_carries):
                self.lm._carries[split + i] = c.clone() if c is not None else None

            H_enriched = self.lm.inject_memory(H_mid, mem_out)
            H = self.lm.forward_scan_upper(H_enriched)
            logits = self.lm.forward_output(H)

            # Float32 CE for precision
            ce = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                target_ids.reshape(-1),
                reduction='none',
            ).reshape(BS, T)
            seg_ce = ce.view(BS, n_segments, action_every)
            seg_losses = (seg_ce * seg_mask).sum(-1) / seg_mask.sum(-1).clamp(min=1)
            trajectory_losses[traj_i] = seg_losses.mean()

            del mem_out, logits, H, H_enriched, ce

        # Restore real state
        mg.load_state_dict({k: v.clone() for k, v in mg_state.items()})
        for i, c in enumerate(pre_carries):
            self.lm._carries[split + i] = c

        # Z-score normalize trajectory losses
        tl_std = trajectory_losses.std()
        if tl_std < 1e-6:
            import logging
            logging.getLogger(__name__).warning(
                f"GRPO: all trajectories tied (std={tl_std.item():.2e}, "
                f"losses={trajectory_losses.tolist()})")
        adv_per_traj = -(trajectory_losses - trajectory_losses.mean()) / tl_std.clamp(min=1e-8)

        return {
            "k_neurons": k_neurons,                    # [K]
            "trajectory_advantages": adv_per_traj,      # [N_traj]
            "trajectory_k_actions": trajectory_k_actions,  # [N_traj][n_seg] of [BS*K, act_dim]
            "trajectory_k_obs": trajectory_k_obs,          # [N_traj][n_seg] of [BS*K, obs_dim]
        }

    def prepare_grpo_replay(self, scoring_result: dict) -> dict:
        """Prepare GRPO-only replay data. No GAE — only scored neurons get gradient.

        Args:
            scoring_result: from score_trajectories

        Returns:
            Replay data with per-trajectory advantages (already z-score normalized).
        """
        return {"scoring": scoring_result}

    def replay_for_neuromod_grads(
        self, rl_data: dict,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """GRPO-only policy gradient for K scored neurons.

        Replays each above-average trajectory's K-neuron actions through the
        policy, weighted by z-score advantage. Entropy bonus always applied.

        Returns:
            dict with grpo_loss, entropy for logging
        """
        scoring = rl_data["scoring"]
        traj_advs = scoring["trajectory_advantages"]       # [N_traj]
        traj_k_actions = scoring["trajectory_k_actions"]   # [N_traj][n_seg] of [BS*K, act_dim]
        traj_k_obs = scoring["trajectory_k_obs"]           # [N_traj][n_seg] of [BS*K, obs_dim]

        device = traj_advs.device
        n_positive = max(1, (traj_advs > 0).sum().item())

        grpo_loss_val = 0.0
        total_entropy = 0.0
        n_replayed = 0

        for traj_i, traj_adv in enumerate(traj_advs):
            if traj_adv.item() <= 0:
                continue

            # Batch all segments' K-neuron obs and actions
            batch_obs = torch.cat(traj_k_obs[traj_i], dim=0)
            batch_act = torch.cat(traj_k_actions[traj_i], dim=0)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                _, log_prob, entropy, _ = self.neuromod.get_action_and_value(
                    batch_obs, action=batch_act)

            grpo_loss = -(traj_adv * log_prob.mean())
            ent_bonus = -self.config.rl_entropy_coef * entropy.mean()
            ((grpo_loss + ent_bonus) / n_positive).backward()

            grpo_loss_val += grpo_loss.item()
            total_entropy += entropy.mean().item()
            n_replayed += 1

        # If all trajectories tied, still apply entropy bonus for exploration
        if n_replayed == 0 and traj_k_obs and traj_k_obs[0]:
            batch_obs = torch.cat(traj_k_obs[0], dim=0)
            batch_act = torch.cat(traj_k_actions[0], dim=0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                _, _, entropy, _ = self.neuromod.get_action_and_value(
                    batch_obs, action=batch_act)
            ent_loss = -self.config.rl_entropy_coef * entropy.mean()
            ent_loss.backward()
            total_entropy = entropy.mean().item()

        return {
            "grpo_loss": grpo_loss_val / max(n_replayed, 1),
            "entropy": total_entropy / max(n_replayed, 1),
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
