"""V8Model — split-scan + per-token memory graph + RL neuromodulator.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0-3) + PCM (parallel over T)
  2. Memory graph: per-token dynamics with neuromod actions per segment
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers 4-6) + output head → logits

RL update (every rl_collect_chunks chunks):
  5. GRPO scoring: replay ALL collected chunks with N sampled trajectories
     Only K neurons get actions (non-K get zero delta). Rank by CE.
  6. Policy gradient: encourage best trajectories' actions (K neurons only)
  7. Best trajectory's final state persists for the next forward pass
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
        obs_dim = config.D_mem * 4 + 4  # prim + key + mean_in + mean_out + msg_mag + decay + trace_norms
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
           Memory graph is persistent — no resets at doc boundaries.
        3. Inject: H_enriched = H_mid + gate * mem_signals
        4. Upper scan (layers split..L-1) (parallel over T)
        5. Output head → logits
        """
        BS, T = input_ids.shape
        C = self.config.C
        D = self.config.D
        D_mem = self.config.D_mem
        action_every = self.config.action_every
        eot_id = self.config.eot_id
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        # Reset scan carries at doc boundaries (LM only, not memory graph)
        if has_reset and reset_mask is not None:
            self._reset_carries(reset_mask)

        # ==========================================
        # Lower scan (layers 0..split-1) + PCM
        # ==========================================
        H_mid, surprise, x, aux_loss = self.lm.forward_scan_lower(input_ids)

        # --- No-memory fast path ---
        if not use_memory:
            H = self.lm.forward_scan_upper(H_mid, surprise=surprise)
            logits = self.lm.forward_output(H)
            return {
                "logits": logits,
                "aux_loss": aux_loss,
                "surprise": surprise.detach(),
                "rl_data": None,
            }

        # ==========================================
        # Memory graph: per-token processing
        # Memory graph is persistent — no doc boundary resets.
        # It learns to handle transitions through CC signal changes.
        # ==========================================
        self._ensure_memory(BS, device, dtype)

        # CC signals from lower scan (detached for memory graph input)
        cc_signals_all = H_mid.detach().view(BS, T, C, D_mem)
        eot_at = (input_ids == eot_id)

        n_segments = T // action_every
        N_neurons = self.config.N_neurons

        cc_segments = cc_signals_all.view(BS, n_segments, action_every, C, D_mem)

        mem_out = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)
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
            t0 = seg * action_every

            self._segment_counter += 1
            sp_every = self.config.structural_plasticity_every
            needs_phi = (sp_every > 0
                         and self._segment_counter % sp_every == 0)

            # Run memory graph (no eot_mask — memory persists across docs)
            seg_out = mg.forward_segment(
                seg_cc, eot_mask=None,
                update_co_activation=needs_phi)
            mem_out[:, t0:t0 + action_every] = seg_out

            # Neuromod: compute traces from this segment, then gate them
            if use_neuromod:
                mg.compute_eligibility_traces()
                obs = mg.get_neuron_obs()
                obs_flat = obs.reshape(BS * N_neurons, -1)
                with torch.no_grad():
                    action, _, _, _ = self.neuromod.get_action_and_value(obs_flat)
                self._apply_neuromod_action(action, BS)

            if needs_phi:
                mg.structural_plasticity()

        # ==========================================
        # Upper scan + output
        # ==========================================
        H_enriched = self.lm.inject_memory(H_mid, mem_out)
        H = self.lm.forward_scan_upper(H_enriched, surprise=surprise)
        logits = self.lm.forward_output(H)

        # ==========================================
        # Collect RL data (scoring happens at RL update time, not here)
        # ==========================================
        rl_data = None
        if use_memory and use_neuromod:
            rl_data = {
                "eot_at": eot_at,
                "n_segments": n_segments,
                "action_every": action_every,
                "cc_segments": cc_segments.detach(),
                "H_mid": H_mid.detach(),
                "surprise": surprise.detach(),
                "pre_upper_carries": pre_upper_carries,
            }

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    @torch.no_grad()
    def score_trajectories(
        self,
        rl_buffer: list[dict],
        pre_mg_state: dict,
    ) -> dict:
        """GRPO trajectory scoring across multiple chunks.

        Replays ALL collected chunks for N trajectories. Only K neurons
        get stochastic actions; non-K neurons get zero delta. This gives
        clean credit assignment to the K neurons.

        After scoring, memory graph and upper scan carries are set to the
        best trajectory's final state (best-of-N selection).

        Args:
            rl_buffer: list of rl_data dicts from collected chunks
            pre_mg_state: memory graph state from before the first
                          collected chunk (starting point for all trajectories)

        Returns:
            dict with k_neurons, trajectory advantages, and per-segment
            actions/obs for the K neurons only (for replay).
        """
        N = self.config.N_neurons
        C = self.config.C
        D_mem = self.config.D_mem
        T = self.config.T
        K = min(self.config.rl_counterfactual_k, N)
        N_traj = self.config.rl_counterfactual_n
        split = self.config.scan_split_at
        L = self.config.L_total
        mg = self._mem_graph

        first = rl_buffer[0]
        BS = first["H_mid"].shape[0]
        device = first["H_mid"].device
        dtype = first["H_mid"].dtype
        n_chunks = len(rl_buffer)
        n_segments = first["n_segments"]
        action_every = first["action_every"]

        nm_dtype = next(self.neuromod.parameters()).dtype

        # Choose K neurons (fixed for all trajectories and all chunks)
        k_neurons = torch.randperm(N, device=device)[:K]
        batch_offsets = torch.arange(BS, device=device).unsqueeze(1) * N
        k_idx_flat = (batch_offsets + k_neurons.unsqueeze(0)).reshape(-1)  # [BS*K]

        # Pre-forward upper carries from chunk 0 (starting point for all trajectories)
        pre_upper_carries = first.get("pre_upper_carries")

        trajectory_losses = torch.empty(N_traj, device=device)
        trajectory_k_actions = []  # [N_traj] of list[Tensor]
        trajectory_k_obs = []      # [N_traj] of list[Tensor]

        # Track best trajectory for state selection (issue 10)
        best_traj_idx = 0
        best_traj_loss = float('inf')
        best_mg_state = None
        best_upper_carries = None

        for traj_i in range(N_traj):
            # Restore memory graph to pre-collection state
            mg.load_state_dict({k: v.clone() for k, v in pre_mg_state.items()})

            # Restore upper scan carries to chunk 0's pre-forward state
            if pre_upper_carries is not None:
                for i, c in enumerate(pre_upper_carries):
                    self.lm._carries[split + i] = c.clone() if c is not None else None

            traj_total_loss = 0.0
            traj_k_acts = []
            traj_k_obs_list = []

            for chunk_idx in range(n_chunks):
                chunk = rl_buffer[chunk_idx]
                cc_segments = chunk["cc_segments"]
                H_mid = chunk["H_mid"]
                chunk_surprise = chunk["surprise"]
                target_ids = chunk["target_ids"]
                eot_at = chunk["eot_at"]

                mem_out = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)

                for seg in range(n_segments):
                    seg_cc = cc_segments[:, seg]
                    seg_out = mg.forward_segment(
                        seg_cc, eot_mask=None, update_co_activation=False)
                    t0 = seg * action_every
                    mem_out[:, t0:t0 + action_every] = seg_out

                    # Compute traces from this segment, then gate
                    mg.compute_eligibility_traces()
                    seg_obs = mg.get_neuron_obs().reshape(BS * N, -1).to(nm_dtype)
                    action, _, _, _ = self.neuromod.get_action_and_value(seg_obs)

                    # Only K neurons get actions, non-K get zero gate + current decay
                    # (zero gate = no plasticity; current decay = no drift)
                    full_action = torch.zeros_like(action)
                    full_action[:, 1] = mg.decay_logit.reshape(-1)  # preserve decay for non-K
                    full_action[k_idx_flat] = action[k_idx_flat]

                    # Store K neurons' obs and actions for replay
                    traj_k_obs_list.append(seg_obs[k_idx_flat].detach())
                    traj_k_acts.append(full_action[k_idx_flat].detach())

                    self._apply_neuromod_action(full_action, BS)

                # Upper scan + CE for this chunk
                # (upper scan carries evolve naturally across chunks within a trajectory)
                H_enriched = self.lm.inject_memory(H_mid, mem_out)
                H = self.lm.forward_scan_upper(H_enriched, surprise=chunk_surprise)
                logits = self.lm.forward_output(H)

                # Float32 CE, masked by EOT positions
                reward_mask = (~eot_at).to(dtype=dtype)
                ce = F.cross_entropy(
                    logits.float().reshape(-1, self.config.vocab_size),
                    target_ids.reshape(-1),
                    reduction='none',
                ).reshape(BS, T)
                chunk_loss = (ce * reward_mask).sum() / reward_mask.sum().clamp(min=1)
                traj_total_loss += chunk_loss.item()

                del mem_out, logits, H, H_enriched, ce

            avg_loss = traj_total_loss / n_chunks
            trajectory_losses[traj_i] = avg_loss
            trajectory_k_actions.append(traj_k_acts)
            trajectory_k_obs.append(traj_k_obs_list)

            # Track best trajectory (lowest loss)
            if traj_total_loss < best_traj_loss:
                best_traj_loss = traj_total_loss
                best_traj_idx = traj_i
                best_mg_state = {k: v.clone() for k, v in mg.state_dict().items()}
                best_upper_carries = [
                    self.lm._carries[split + i].clone()
                    if self.lm._carries[split + i] is not None else None
                    for i in range(L - split)
                ]

        # Apply best trajectory's final state — next chunk starts from the
        # best memory configuration instead of the original
        mg.load_state_dict(best_mg_state)
        for i, c in enumerate(best_upper_carries):
            self.lm._carries[split + i] = c

        # Z-score normalize trajectory losses → advantages
        tl_std = trajectory_losses.std()
        if tl_std < 1e-6:
            import logging
            logging.getLogger(__name__).warning(
                f"GRPO: all trajectories tied (std={tl_std.item():.2e}, "
                f"losses={trajectory_losses.tolist()})")
        adv_per_traj = -(trajectory_losses - trajectory_losses.mean()) / tl_std.clamp(min=1e-8)

        return {
            "k_neurons": k_neurons,                      # [K]
            "trajectory_advantages": adv_per_traj,        # [N_traj]
            "trajectory_k_actions": trajectory_k_actions,  # [N_traj][n_total_segs] of [BS*K, act_dim]
            "trajectory_k_obs": trajectory_k_obs,          # [N_traj][n_total_segs] of [BS*K, obs_dim]
            "best_traj_idx": best_traj_idx,
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
        traj_k_actions = scoring["trajectory_k_actions"]   # [N_traj][n_segs] of [BS*K, act_dim]
        traj_k_obs = scoring["trajectory_k_obs"]           # [N_traj][n_segs] of [BS*K, obs_dim]

        device = traj_advs.device
        n_positive = max(1, (traj_advs > 0).sum().item())

        grpo_loss_val = 0.0
        total_entropy = 0.0
        n_replayed = 0

        for traj_i, traj_adv in enumerate(traj_advs):
            if traj_adv.item() <= 0:
                continue

            # Batch all segments' K-neuron obs and actions (across all chunks)
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
        """Extract gate and decay_target from neuromod output, apply gated plasticity."""
        N = self.config.N_neurons
        action = action.reshape(BS, N, 2)
        gate = action[:, :, 0].tanh()   # [BS, N] — clamp to [-1, 1]
        decay_target = action[:, :, 1]  # [BS, N] — unbounded (sigmoid in memory graph)
        self._mem_graph.apply_gated_plasticity(gate, decay_target)

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
