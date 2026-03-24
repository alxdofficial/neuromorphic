"""V8Model — split-scan + per-token memory graph + RL neuromodulator.

Training flow per chunk (T=2048 tokens):
  1. Lower scan (layers 0-3) + PCM (parallel over T)
  2. Memory graph: per-token receive → integrate → message (sequential)
     Neuromod acts every 256 tokens (8 segments)
  3. Inject: H_enriched = H_mid + gate * mem_signals
  4. Upper scan (layers 4-6) over memory-enriched H (parallel)
  5. Per-segment CE losses → rewards collected

RL update (every rl_collect_chunks chunks):
  6. Concatenate segment data across collected chunks
  7. Discounted returns with value bootstrap at end
  8. Advantages = returns - V(global_state) (learned value baseline)
  9. Policy gradient + value function MSE loss
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
        obs_dim = config.D_mem * 3 + 3  # prim + mean_in + mean_out + firing_rate + decay + entropy
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

        for seg in range(n_segments):
            # 1. Neuromod observes and acts (skip if disabled)
            if use_neuromod:
                obs = self._mem_graph.get_neuron_obs()
                obs_flat = obs.reshape(BS * N_neurons, -1)

                with torch.no_grad():
                    action, _, _, _ = self.neuromod.get_action_and_value(obs_flat)

                obs_list.append(obs_flat.detach())
                actions.append(action.detach())
                self._apply_neuromod_action(action, BS)

            # 2. Run memory graph for this segment
            # Only compute co-activation phi when plasticity will run next
            self._segment_counter += 1
            sp_every = self.config.structural_plasticity_every
            needs_phi = (sp_every > 0
                         and self._segment_counter % sp_every == 0)

            seg_cc = cc_segments[:, seg]
            eot_mask = eot_masks[:, seg] if eot_masks is not None else None
            seg_out = self._mem_graph.forward_segment(
                seg_cc, eot_mask=eot_mask,
                update_co_activation=needs_phi)
            t0 = seg * action_every
            mem_out[:, t0:t0 + action_every] = seg_out

            # 3. Structural plasticity at configured cadence
            if needs_phi:
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
        # Collect RL data (CE computed once in trainer, passed back here)
        # ==========================================
        rl_data = None
        if use_memory and use_neuromod:
            rl_data = {
                "obs": obs_list,              # list of n_segments [BS*N, obs_dim] tensors
                "actions": actions,           # list of n_segments [BS*N, act_dim] tensors
                "eot_at": eot_at,             # [BS, T] for reward masking
                "n_segments": n_segments,
                "action_every": action_every,
            }

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "rl_data": rl_data,
        }

    def compute_rl_advantages(
        self, collected_rl_data: list[dict],
    ) -> dict:
        """Compute returns and advantages across collected chunks.

        Concatenates segment data from multiple chunks, computes discounted
        returns with value function bootstrap at the end, and uses the learned
        value function as baseline for advantages.

        Args:
            collected_rl_data: list of per-chunk rl_data dicts

        Returns:
            Combined rl_data dict with obs, actions, advantages, and global_obs
            ready for replay_for_neuromod_grads.
        """
        # Concatenate across chunks
        all_obs = []
        all_actions = []
        all_seg_rewards = []
        all_global_obs = []

        N_neurons = self.config.N_neurons

        for chunk_data in collected_rl_data:
            all_obs.extend(chunk_data["obs"])         # each is [BS*N, obs_dim]
            all_actions.extend(chunk_data["actions"])  # each is [BS*N, act_dim]
            all_seg_rewards.append(chunk_data["seg_rewards"])  # [BS, n_seg]

            # Compute global obs (mean-pooled across neurons) for each segment
            for obs_t in chunk_data["obs"]:
                BS = obs_t.shape[0] // N_neurons
                global_t = obs_t.reshape(BS, N_neurons, -1).mean(dim=1)  # [BS, obs_dim]
                all_global_obs.append(global_t)

        # seg_rewards: [BS, total_segments]
        seg_rewards = torch.cat(all_seg_rewards, dim=1)
        BS = seg_rewards.shape[0]
        total_segments = seg_rewards.shape[1]
        device = seg_rewards.device
        gamma = self.config.rl_gamma

        # Global obs for value function: [total_segments, BS, obs_dim]
        global_obs_stack = torch.stack(all_global_obs)  # [total_seg, BS, obs_dim]

        with torch.no_grad():
            # Value predictions at each segment: [total_seg, BS]
            global_obs_flat = global_obs_stack.reshape(-1, global_obs_stack.shape[-1])
            # Cast to model dtype (obs may be bf16/f32 depending on autocast)
            v_dtype = next(self.neuromod.value_net.parameters()).dtype
            values_flat = self.neuromod.get_value(global_obs_flat.to(v_dtype))
            values = values_flat.reshape(total_segments, BS).T  # [BS, total_seg]

            # Bootstrap: value of state after the last segment
            # Use the value at the last segment as an approximation
            # (ideally we'd have the next state, but the last value is close)
            v_bootstrap = values[:, -1]  # [BS]

            # Discounted returns with value bootstrap at the end
            returns = torch.zeros_like(seg_rewards)  # [BS, total_segments]
            returns[:, -1] = seg_rewards[:, -1] + gamma * v_bootstrap
            for t in range(total_segments - 2, -1, -1):
                returns[:, t] = seg_rewards[:, t] + gamma * returns[:, t + 1]

            # Advantages: return - value baseline
            advantages = returns - values

        return {
            "obs": all_obs,                    # list of total_seg [BS*N, obs_dim]
            "actions": all_actions,            # list of total_seg [BS*N, act_dim]
            "advantages": advantages,          # [BS, total_segments]
            "returns": returns,                # [BS, total_segments] for value training
            "global_obs": global_obs_stack,    # [total_seg, BS, obs_dim] for value training
        }

    def replay_for_neuromod_grads(
        self, rl_data: dict,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Compute policy gradient + value loss via replay.

        Uses stored per-segment obs for accurate log_prob evaluation.
        Each segment's actions are weighted by its own advantage.
        Also trains the value function on observed returns.

        Returns:
            dict with policy_loss, value_loss, entropy for logging
        """
        obs_list = rl_data["obs"]          # list of n_seg [BS*N, obs_dim]
        actions = rl_data["actions"]       # list of n_seg [BS*N, act_dim]
        advantages = rl_data["advantages"] # [BS, n_segments]
        returns = rl_data["returns"]       # [BS, n_segments]
        global_obs = rl_data["global_obs"] # [n_seg, BS, obs_dim]

        N_neurons = self.config.N_neurons
        BS = actions[0].shape[0] // N_neurons
        device = actions[0].device
        n_segments = len(actions)

        # ---- Policy gradient ----
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

        # Weighted policy loss + entropy bonus
        policy_loss = -(adv * log_prob).mean()
        entropy_bonus = -self.config.rl_entropy_coef * entropy.mean()

        # ---- Value function loss ----
        global_obs_flat = global_obs.reshape(-1, global_obs.shape[-1])  # [n_seg*BS, obs_dim]
        v_dtype = next(self.neuromod.value_net.parameters()).dtype
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
            v_pred = self.neuromod.get_value(global_obs_flat.to(v_dtype))  # [n_seg*BS]
        v_pred = v_pred.reshape(n_segments, BS).T  # [BS, n_seg]
        value_loss = 0.5 * (v_pred - returns).pow(2).mean()

        total_loss = policy_loss + entropy_bonus + value_loss
        total_loss.backward()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
        }

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
