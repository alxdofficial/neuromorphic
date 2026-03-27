"""V8/v9 Model — split-scan LM + memory graph (ES-trained).

Training flow per chunk:
  1. Lower scan + PCM → H_mid, surprise  (backprop)
  2. Memory graph: per-token dynamics     (no grad, fast)
  3. Inject + upper scan → logits         (backprop)

ES update (every es_collect_chunks chunks):
  - Perturb K neurons' params with Gaussian noise
  - Replay chunks with N trajectories, score by CE loss
  - Update params toward good perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import V8Config
from .lm import V8LM
from .memory_graph import MemoryGraph


class V8Model(nn.Module):
    """Top-level model: LM (backprop) + Memory Graph (ES)."""

    def __init__(self, config: V8Config):
        super().__init__()
        config.validate()
        self.config = config
        self.lm = V8LM(config)
        self.memory = MemoryGraph(
            config, device=torch.device('cpu'), dtype=torch.bfloat16)
        self._states_initialized = False

    def forward_chunk(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        reset_mask: Tensor | None = None,
        use_memory: bool = True,
        has_reset: bool = False,
    ) -> dict:
        """Process one chunk. LM by backprop, memory graph no-grad."""
        BS, T = input_ids.shape
        C = self.config.C
        D_mem = self.config.D_mem
        action_every = self.config.action_every
        device = input_ids.device
        dtype = next(self.lm.parameters()).dtype

        if has_reset and reset_mask is not None:
            self._reset_carries(reset_mask)

        # Lower scan + PCM (with grad)
        H_mid, surprise, x, aux_loss = self.lm.forward_scan_lower(input_ids)

        if not use_memory:
            H = self.lm.forward_scan_upper(H_mid, surprise=surprise)
            logits = self.lm.forward_output(H)
            return {"logits": logits, "aux_loss": aux_loss,
                    "surprise": surprise.detach()}

        # Memory graph (no grad)
        if not self._states_initialized:
            self.memory.initialize_states(BS)
            self._states_initialized = True

        cc_all = H_mid.detach().view(BS, T, C, D_mem)
        n_segments = T // action_every
        cc_segments = cc_all.view(BS, n_segments, action_every, C, D_mem)
        mem_out = torch.empty(BS, T, C, D_mem, device=device, dtype=dtype)

        for seg in range(n_segments):
            seg_out = self.memory.forward_segment(cc_segments[:, seg])
            t0 = seg * action_every
            mem_out[:, t0:t0 + action_every] = seg_out

        # Upper scan + output (with grad)
        H_enriched = self.lm.inject_memory(H_mid, mem_out)
        H = self.lm.forward_scan_upper(H_enriched, surprise=surprise)
        logits = self.lm.forward_output(H)

        return {
            "logits": logits,
            "aux_loss": aux_loss,
            "surprise": surprise.detach(),
            "cc_segments": cc_segments.detach(),
            "H_mid": H_mid.detach(),
        }

    # ================================================================
    # ES trajectory scoring
    # ================================================================

    @torch.no_grad()
    def score_es_trajectories(
        self,
        es_buffer: list[dict],
        pre_mg_state: dict,
        pre_mg_params: dict,
    ) -> dict:
        """Score N trajectories with perturbed K neurons.

        For each trajectory: perturb K neurons' params → replay all chunks
        → score by CE loss → z-score normalize → advantages.

        Args:
            es_buffer: list of {cc_segments, H_mid, surprise, target_ids, eot_at,
                                pre_upper_carries} from collected chunks
            pre_mg_state: runtime state before collection window
            pre_mg_params: ES params before collection window

        Returns:
            dict with k_neurons, advantages, noise vectors, best state
        """
        N = self.config.N_neurons
        K = min(self.config.es_k_neurons, N)
        N_traj = self.config.es_n_trajectories
        sigma = self.config.es_sigma
        split = self.config.scan_split_at
        L = self.config.L_total
        mg = self.memory

        first = es_buffer[0]
        BS = first["H_mid"].shape[0]
        device = first["H_mid"].device
        dtype = first["H_mid"].dtype
        n_chunks = len(es_buffer)
        n_segments = first["cc_segments"].shape[1]
        action_every = self.config.action_every

        # Choose K neurons
        k_neurons = torch.randperm(N, device=device)[:K]

        # Get current K-neuron param shapes for noise generation
        k_params = mg.get_neuron_es_params(k_neurons)

        # Generate noise for all trajectories (antithetic: +ε and -ε)
        all_noise = []
        for traj_i in range(N_traj):
            noise = {}
            if traj_i % 2 == 0:
                # Fresh noise
                for name, param in k_params.items():
                    noise[name] = torch.randn_like(param)
            else:
                # Antithetic (negate previous)
                prev_noise = all_noise[-1]
                noise = {name: -eps for name, eps in prev_noise.items()}
            all_noise.append(noise)

        # Score each trajectory
        trajectory_losses = torch.empty(N_traj, device=device)
        pre_upper_carries = first.get("pre_upper_carries")

        best_traj_idx = 0
        best_traj_loss = float('inf')
        best_mg_state = None
        best_upper_carries = None

        for traj_i in range(N_traj):
            # Restore params + state
            for name, val in pre_mg_params.items():
                getattr(mg, name).data.copy_(val)
            mg.load_runtime_state({k: v.clone() for k, v in pre_mg_state.items()})

            if pre_upper_carries is not None:
                for i, c in enumerate(pre_upper_carries):
                    self.lm._carries[split + i] = c.clone() if c is not None else None

            # Apply perturbation
            mg.apply_es_perturbation(k_neurons, all_noise[traj_i], sigma)

            # Replay all chunks
            traj_total_loss = 0.0
            for chunk in es_buffer:
                cc_segments = chunk["cc_segments"]
                H_mid = chunk["H_mid"]
                surprise = chunk["surprise"]
                target_ids = chunk["target_ids"]
                eot_at = chunk["eot_at"]

                # Memory forward
                mem_out = torch.empty(BS, self.config.T, self.config.C, self.config.D_mem,
                                      device=device, dtype=dtype)
                for seg in range(n_segments):
                    seg_out = mg.forward_segment(cc_segments[:, seg])
                    t0 = seg * action_every
                    mem_out[:, t0:t0 + action_every] = seg_out

                # Upper scan + CE
                H_enriched = self.lm.inject_memory(H_mid, mem_out)
                H = self.lm.forward_scan_upper(H_enriched, surprise=surprise)
                logits = self.lm.forward_output(H)

                reward_mask = (~eot_at).to(dtype=dtype)
                ce = F.cross_entropy(
                    logits.float().reshape(-1, self.config.vocab_size),
                    target_ids.reshape(-1), reduction='none'
                ).reshape(BS, self.config.T)
                chunk_loss = (ce * reward_mask).sum() / reward_mask.sum().clamp(min=1)
                traj_total_loss += chunk_loss.item()

            avg_loss = traj_total_loss / n_chunks
            trajectory_losses[traj_i] = avg_loss

            if traj_total_loss < best_traj_loss:
                best_traj_loss = traj_total_loss
                best_traj_idx = traj_i
                best_mg_state = mg.runtime_state_dict()
                best_mg_state = {k: v.clone() for k, v in best_mg_state.items()}
                best_mg_params = {name: getattr(mg, name).data.clone()
                                  for name in pre_mg_params}
                best_upper_carries = [
                    self.lm._carries[split + i].clone()
                    if self.lm._carries[split + i] is not None else None
                    for i in range(L - split)]

        # Z-score advantages (lower loss = higher advantage)
        tl_std = trajectory_losses.std().clamp(min=1e-8)
        advantages = -(trajectory_losses - trajectory_losses.mean()) / tl_std

        # Restore best trajectory state
        for name, val in best_mg_params.items():
            getattr(mg, name).data.copy_(val)
        mg.load_runtime_state(best_mg_state)
        for i, c in enumerate(best_upper_carries):
            self.lm._carries[split + i] = c

        return {
            "k_neurons": k_neurons,
            "advantages": advantages,
            "noise": all_noise,
            "trajectory_losses": trajectory_losses,
            "best_traj_idx": best_traj_idx,
        }

    def apply_es_gradient(self, scoring: dict):
        """Apply ES update: move params toward good perturbations.

        θ += lr / (N * σ) * Σ_i advantage_i * ε_i
        """
        advantages = scoring["advantages"]  # [N_traj]
        noise_list = scoring["noise"]       # [N_traj] of {name: Tensor}
        k_neurons = scoring["k_neurons"]    # [K]
        N_traj = len(noise_list)
        sigma = self.config.es_sigma
        lr = self.config.es_lr

        # Compute weighted noise: Σ advantage_i * ε_i
        weighted_noise = {}
        for traj_i, noise in enumerate(noise_list):
            adv = advantages[traj_i].item()
            for name, eps in noise.items():
                if name not in weighted_noise:
                    weighted_noise[name] = torch.zeros_like(eps)
                weighted_noise[name] += adv * eps

        # Scale by 1 / (N_traj * σ)
        scale = lr / (N_traj * sigma)
        for name in weighted_noise:
            weighted_noise[name] *= scale

        self.memory.apply_es_update(k_neurons, weighted_noise, lr=1.0)

    # ================================================================
    # Utilities
    # ================================================================

    def _reset_carries(self, mask: Tensor):
        if hasattr(self.lm, '_carries'):
            for i, h in enumerate(self.lm._carries):
                if h is not None:
                    mask_f = (~mask).to(dtype=h.dtype).unsqueeze(-1)
                    self.lm._carries[i] = h * mask_f

    def initialize_states(self, BS: int, device: torch.device = None):
        self.lm.initialize_carries()
        self.memory.initialize_states(BS)
        self._states_initialized = True

    def detach_states(self):
        self.lm.detach_carries()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self) -> int:
        return self.lm.param_count()

    def memory_param_count(self) -> int:
        return sum(p.numel() for p in self.memory.parameters())
