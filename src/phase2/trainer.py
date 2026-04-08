"""Phase 2 trainer: GRPO over discrete RVQ codes for the neuromodulator.

Everything except the modulator's mod_w1/b1/w2/b2 is frozen. The modulator's
continuous output is encoded via a frozen RVQ-VAE, sampled into discrete codes,
and the quantized reconstruction is applied to memory state. The training
signal is a group-relative policy gradient on windowed mem_pred_loss reward.

See docs/training_strategy.md for the full design.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ..model.config import Config
from ..model.model import Model
from ..codebook import ActionVQVAE


@dataclass
class CurriculumStage:
    """One stage of the phase 2 reward-window curriculum."""
    reward_window: int   # W: number of tokens in the reward window
    token_budget: int    # how many tokens to train this stage for


class Phase2Trainer:
    """GRPO trainer for the modulator with frozen everything else.

    Usage:
        trainer = Phase2Trainer(model, vqvae, dataloader, config, device, ...)
        trainer.run_curriculum([
            CurriculumStage(reward_window=512,  token_budget=25_000_000),
            CurriculumStage(reward_window=2048, token_budget=15_000_000),
            CurriculumStage(reward_window=4096, token_budget=10_000_000),
        ])
    """

    def __init__(
        self,
        model: Model,
        vqvae: ActionVQVAE,
        dataloader,
        config: Config,
        device: torch.device,
        group_size: int = 8,
        lr: float = 1e-4,
        tau: float = 1.0,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
    ):
        self.model = model
        self.vqvae = vqvae
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.group_size = group_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # Freeze everything except the modulator's MLP params
        for p in model.parameters():
            p.requires_grad = False
        self.trainable_params = [
            model.memory.mod_w1, model.memory.mod_b1,
            model.memory.mod_w2, model.memory.mod_b2,
        ]
        for p in self.trainable_params:
            p.requires_grad = True
        for p in vqvae.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, betas=(0.9, 0.95))
        self.global_step = 0

        # Current curriculum stage (set by run_curriculum)
        self.reward_window: int = 0
        self.segment_length: int = 0

    # ------------------------------------------------------------------
    # Rollout + reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_lower_scan(self, input_ids: Tensor) -> Tensor:
        """Run the frozen LM lower scan to produce H_mid."""
        return self.model.lm.forward_scan_lower(input_ids)

    @torch.no_grad()
    def _compute_per_token_reward(
        self, readouts: Tensor, prev_readout_at_start: Tensor, input_ids: Tensor,
    ) -> Tensor:
        """Compute negative per-token mem_pred_loss to use as dense reward.

        Returns: [BS, T] float — per-token reward (higher = better prediction).
        """
        BS, T, D = readouts.shape
        # Shift: readout[t-1] predicts token at position t.
        shifted = torch.cat([
            prev_readout_at_start.unsqueeze(1).to(readouts.dtype),
            readouts[:, :-1],
        ], dim=1)  # [BS, T, D]

        # Chunk along time to bound VRAM
        rewards = torch.empty(BS, T, device=readouts.device, dtype=torch.float32)
        chunk = 64
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_readout = shifted[:, s:e]
            sub_target = input_ids[:, s:e]
            sub_logits = self.model.lm.mem_head_logits(sub_readout)   # [BS, chunk, V]
            ce = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                sub_target.reshape(-1),
                reduction="none",
            ).reshape(BS, e - s)
            rewards[:, s:e] = -ce
        return rewards

    @torch.no_grad()
    def _windowed_reward(
        self, per_token_reward: Tensor, call_positions: Tensor, window: int,
    ) -> Tensor:
        """For each modulator call at token t, compute mean reward over [t, t+W].

        Args:
            per_token_reward: [BS, T]
            call_positions: [n_calls] — absolute token index of each call
            window: reward window W

        Returns:
            rewards: [n_calls, BS] — per-action reward averaged over window
        """
        BS, T = per_token_reward.shape
        n_calls = call_positions.shape[0]
        out = torch.empty(n_calls, BS, device=per_token_reward.device, dtype=torch.float32)
        for i, t in enumerate(call_positions.tolist()):
            end = min(t + window, T)
            if end <= t:
                out[i] = 0.0
            else:
                out[i] = per_token_reward[:, t:end].mean(dim=1)
        return out

    @torch.no_grad()
    def _save_mem_state(self) -> dict:
        """Snapshot the memory runtime state so we can reset between rollouts."""
        mem = self.model.memory
        return {
            "h": mem.h.clone(), "msg": mem.msg.clone(), "W": mem.W.clone(),
            "decay_logit": mem.decay_logit.clone(),
            "s_mem_live": mem.s_mem_live.clone(),
            "s_mem_ema_fast": mem.s_mem_ema_fast.clone(),
            "prev_readout": mem.prev_readout.clone(),
            "prev_readout_cell": mem.prev_readout_cell.clone(),
            "readout_drift": mem.readout_drift.clone(),
        }

    @torch.no_grad()
    def _restore_mem_state(self, snapshot: dict):
        mem = self.model.memory
        mem.h = snapshot["h"].clone()
        mem.msg = snapshot["msg"].clone()
        mem.W = snapshot["W"].clone()
        mem.decay_logit = snapshot["decay_logit"].clone()
        mem.s_mem_live = snapshot["s_mem_live"].clone()
        mem.s_mem_ema_fast = snapshot["s_mem_ema_fast"].clone()
        mem.prev_readout = snapshot["prev_readout"].clone()
        mem.prev_readout_cell = snapshot["prev_readout_cell"].clone()
        mem.readout_drift = snapshot["readout_drift"].clone()

    @torch.no_grad()
    def rollout(self, batch) -> dict:
        """Run K trajectories on the same batch, collect (state, codes, reward).

        Each trajectory resets the memory to the pre-rollout snapshot, runs a
        sampled forward via `forward_segment_phase2`, and computes per-action
        windowed rewards. After all K trajectories, the memory is restored to
        the snapshot from whichever trajectory we pick to persist (default:
        the highest-reward trajectory) so that phase-2 rollouts still carry
        lifelong memory from one batch to the next.

        Returns dict:
            mod_inputs: [K, n_calls, BS, NC, mod_in]
            codes:      [K, n_calls, BS, NC, num_levels]
            rewards:    [K, n_calls, BS]  per-action reward
        """
        device = self.device
        input_ids = batch.input_ids.to(device, non_blocking=True)
        BS, T = input_ids.shape

        # Run lower scan once (LM is frozen and deterministic per batch)
        H_mid = self._run_lower_scan(input_ids)

        # Snapshot initial memory state
        init_snapshot = self._save_mem_state()
        init_prev_readout = self.model.memory.prev_readout.clone()

        all_mod_inputs = []
        all_codes = []
        all_rewards = []
        best_trajectory_snapshot = None
        best_mean_reward = -float("inf")

        for k in range(self.group_size):
            # Reset memory to the start of the batch
            self._restore_mem_state(init_snapshot)

            result = self.model.memory.forward_segment_phase2(
                H_mid, input_ids, self.model.lm, self.vqvae,
                tau=self.tau, sample=True)

            readouts = result["readouts"]
            call_positions = result["call_positions"].to(device)
            mod_inputs = result["mod_inputs"]         # cpu
            codes = result["codes"]                    # cpu

            per_token_reward = self._compute_per_token_reward(
                readouts, init_prev_readout, input_ids)
            windowed_r = self._windowed_reward(
                per_token_reward, call_positions, self.reward_window)   # [n_calls, BS]

            all_mod_inputs.append(mod_inputs)
            all_codes.append(codes)
            all_rewards.append(windowed_r.cpu())

            mean_r = windowed_r.mean().item()
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                best_trajectory_snapshot = self._save_mem_state()

        # Persist best trajectory's end-state so lifelong memory continues forward
        if best_trajectory_snapshot is not None:
            self._restore_mem_state(best_trajectory_snapshot)

        return {
            "mod_inputs": torch.stack(all_mod_inputs, dim=0),   # [K, n_calls, BS, NC, mod_in]
            "codes": torch.stack(all_codes, dim=0),             # [K, n_calls, BS, NC, L]
            "rewards": torch.stack(all_rewards, dim=0),         # [K, n_calls, BS]
            "mean_reward": best_mean_reward,
            "T": T,
            "BS": BS,
        }

    # ------------------------------------------------------------------
    # GRPO gradient step
    # ------------------------------------------------------------------

    def _compute_advantages(self, rewards: Tensor) -> Tensor:
        """Group-relative advantages.

        rewards: [K, n_calls, BS] — per-action reward per trajectory
        Returns: [K, n_calls, BS] — normalized advantages
        """
        # Per-action baseline: mean over K trajectories
        baseline = rewards.mean(dim=0, keepdim=True)         # [1, n_calls, BS]
        advantages = rewards - baseline                      # [K, n_calls, BS]
        # Per-batch normalization
        mean = advantages.mean()
        std = advantages.std().clamp(min=1e-8)
        return (advantages - mean) / std

    def grpo_step(self, rollout_result: dict) -> dict:
        """Compute GRPO loss from rollout records and update modulator.

        The gradient pass:
          1. Flatten all (mod_input, codes) records across K x n_calls x BS x NC
             into a single batch of modulator inputs.
          2. Run modulator forward -> raw action -> vqvae encoder -> latent z.
             This is the only on-graph computation.
          3. Compute log pi(codes | z) via distance-based categorical over each
             RVQ level.
          4. Loss = -(advantage * log_pi).mean(), backward, step.

        Memory dynamics, LM, mem_head, decoder — all frozen and off the graph.
        """
        device = self.device
        K, n_calls, BS, NC, mod_in_dim = rollout_result["mod_inputs"].shape
        num_levels = rollout_result["codes"].shape[-1]

        # Move to device, flatten
        mod_inputs = rollout_result["mod_inputs"].to(device)   # [K, n_calls, BS, NC, mod_in]
        codes = rollout_result["codes"].to(device)             # [K, n_calls, BS, NC, L]
        rewards = rollout_result["rewards"].to(device)         # [K, n_calls, BS]

        # Advantages — broadcast over NC since actions are per-cell
        advantages = self._compute_advantages(rewards)                 # [K, n_calls, BS]
        advantages = advantages.unsqueeze(-1).expand(-1, -1, -1, NC)   # [K, n_calls, BS, NC]

        # Flatten all records into [M, ...] where M = K * n_calls * BS * NC
        mod_inputs_flat = mod_inputs.reshape(-1, NC, mod_in_dim)       # [K*n_calls*BS, NC, mod_in]
        codes_flat = codes.reshape(-1, NC, num_levels)                 # same
        advantages_flat = advantages.reshape(-1, NC)                   # same
        # Each per-cell action is its own "record"; reshape [M, mod_in] by folding NC
        # But the modulator forward is per-cell (einsum "bni,nih->bnh") and needs
        # the NC dim. So we'll process [M, NC, ...] as a big batch.

        # Chunk the gradient pass to bound VRAM
        M = mod_inputs_flat.shape[0]
        chunk_size = 4096
        total_loss = 0.0
        total_log_pi = 0.0
        n_chunks = 0

        self.optimizer.zero_grad(set_to_none=True)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            mi_chunk = mod_inputs_flat[start:end]          # [C, NC, mod_in]
            c_chunk = codes_flat[start:end]                # [C, NC, L]
            a_chunk = advantages_flat[start:end]           # [C, NC]
            C = mi_chunk.shape[0]

            # Modulator forward (on-graph)
            raw_action = self.model.memory._modulator_forward(
                mi_chunk.to(self.model.memory.mod_w1.dtype))  # [C, NC, mod_out]

            # Flatten [C, NC] -> [C*NC] for the VQ encoder
            action_flat = raw_action.reshape(C * NC, -1).float()
            action_norm = self.vqvae.normalize(action_flat)
            z = self.vqvae.encoder(action_norm)                # [C*NC, latent]

            # log pi(codes | z) summed across RVQ levels (on-graph through z)
            codes_flat_flat = c_chunk.reshape(C * NC, num_levels)
            log_pi = self.vqvae.rvq.log_prob(z, codes_flat_flat, tau=self.tau)  # [C*NC]
            log_pi = log_pi.reshape(C, NC)

            # GRPO loss for this chunk (normalized by full M, so sum across chunks)
            chunk_loss = -(a_chunk * log_pi).sum() / (M * NC)
            chunk_loss.backward()

            total_loss += chunk_loss.item() * (M * NC)
            total_log_pi += log_pi.detach().sum().item()
            n_chunks += 1

        grad_norm = nn.utils.clip_grad_norm_(
            self.trainable_params, self.max_grad_norm).item()
        self.optimizer.step()

        return {
            "loss": total_loss / (M * NC),
            "log_pi_mean": total_log_pi / (M * NC),
            "mean_reward": rollout_result["mean_reward"],
            "mod_grad_norm": grad_norm,
            "n_chunks": n_chunks,
            "M": M,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run_curriculum(self, stages: list[CurriculumStage]):
        """Run the full phase-2 curriculum, advancing through stages sequentially."""
        for stage_idx, stage in enumerate(stages):
            self.reward_window = stage.reward_window
            self.segment_length = stage.reward_window
            print(f"\n=== Phase 2 stage {stage_idx+1}/{len(stages)}: "
                  f"W={stage.reward_window}, budget={stage.token_budget:,} tokens ===")
            self._run_stage(stage)

    def _run_stage(self, stage: CurriculumStage):
        """Run one curriculum stage until token_budget is reached."""
        tokens_seen = 0
        t_stage_start = time.time()
        for step_idx, batch in enumerate(self.dataloader):
            # Assume batch.input_ids.shape == [BS, T]; token budget is on total
            # tokens seen across all steps.
            BS, T = batch.input_ids.shape
            step_tokens = BS * T
            if tokens_seen >= stage.token_budget:
                break

            t0 = time.time()
            rollout_result = self.rollout(batch)
            t_rollout = time.time() - t0

            t0 = time.time()
            step_metrics = self.grpo_step(rollout_result)
            t_grad = time.time() - t0

            tokens_seen += step_tokens
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                print(f"[p2 step {self.global_step}] "
                      f"W={stage.reward_window} "
                      f"loss={step_metrics['loss']:.4f} "
                      f"r={step_metrics['mean_reward']:+.3f} "
                      f"log_pi={step_metrics['log_pi_mean']:+.2f} "
                      f"mod_gn={step_metrics['mod_grad_norm']:.3f} "
                      f"records={step_metrics['M']} "
                      f"roll={t_rollout:.1f}s grad={t_grad:.1f}s "
                      f"tokens={tokens_seen:,}/{stage.token_budget:,}")

        elapsed = time.time() - t_stage_start
        print(f"  stage complete: {tokens_seen:,} tokens in {elapsed:.0f}s")
