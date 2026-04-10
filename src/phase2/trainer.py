"""Phase 2 trainer: GRPO over discrete RVQ codes for the neuromodulator.

Everything except the modulator's mod_w1/b1/w2/b2 is frozen. The modulator's
continuous output is encoded via a frozen RVQ-VAE, sampled into discrete codes,
and the quantized reconstruction is applied to memory state. The training
signal is a group-relative policy gradient on windowed mem_pred_loss reward.

See docs/training_strategy.md for the full design.
"""

from __future__ import annotations

import json
import math
import os
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
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        eval_loader_factory=None,
        eval_batches: int = 4,
        metrics_path: str | None = None,
    ):
        self.model = model
        self.vqvae = vqvae
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.group_size = group_size
        self.tau = tau
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_loader_factory = eval_loader_factory
        self.eval_batches = eval_batches

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

        # JSONL metrics log
        self.metrics_path = metrics_path
        if metrics_path is not None:
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    def _append_metrics(self, metrics: dict):
        if self.metrics_path is None:
            return
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    @torch.no_grad()
    def evaluate(self, eval_loader, n_batches: int = 4) -> dict:
        """Held-out eval pass using the CONTINUOUS modulator path.

        Phase 2 trains the modulator via GRPO over discretized VQ actions, but
        the gradient updates the underlying continuous head. Evaluating with
        the continuous path (via Model.forward_chunk, which does not use VQ)
        gives us a stable measurement of the "mean" policy that GRPO is pushing
        toward. Sampling from the VQ codebook during eval would add noise to
        the signal we're trying to read.

        Uses a fresh memory state (snapshot + restore around the call) so eval
        is independent of the current phase-2 rollout memory carry.
        """
        model = self.model
        memory = model.memory

        train_mem_state = (
            memory.runtime_state_dict() if memory._initialized else None
        )
        train_lm_carries = [
            h.clone() if h is not None else None for h in model.lm._carries
        ]
        train_initialized = model._initialized
        was_training = model.training

        memory._initialized = False
        model._initialized = False
        model.lm._carries = [None] * self.config.L_total
        model.train(False)

        total_ce = 0.0
        total_aux = 0.0
        count = 0
        prev_token = None
        try:
            for i, batch in enumerate(eval_loader):
                if i >= n_batches:
                    break
                input_ids = batch.input_ids.to(self.device, non_blocking=True)
                target_ids = batch.target_ids.to(self.device, non_blocking=True)
                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(self.device, non_blocking=True)
                result = model.forward_chunk(
                    input_ids, target_ids=target_ids,
                    use_memory=True,
                    prev_token=batch_prev if batch_prev is not None else prev_token,
                )
                total_ce += result["ce_loss"].item()
                total_aux += result["aux_loss"].item()
                count += 1
                prev_token = input_ids[:, -1]
        finally:
            if train_mem_state is not None:
                memory.load_runtime_state(train_mem_state)
            else:
                memory._initialized = False
            model.lm._carries = train_lm_carries
            model._initialized = train_initialized
            if was_training:
                model.train(True)

        if count == 0:
            return {"eval_ce_loss": 0.0, "eval_aux_loss": 0.0,
                    "eval_ppl": 0.0, "eval_batches": 0}
        avg_ce = total_ce / count
        avg_aux = total_aux / count
        return {
            "eval_ce_loss": avg_ce,
            "eval_aux_loss": avg_aux,
            "eval_ppl": min(math.exp(avg_ce), 1e6),
            "eval_batches": count,
        }

    # ------------------------------------------------------------------
    # Rollout + reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_lower_scan(
        self, input_ids: Tensor, prev_token: Tensor | None = None,
    ) -> Tensor:
        """Run the frozen LM lower scan to produce H_mid.

        Chunks the input into segments of config.T so pos_embed stays in range.
        Mirrors the EOT reset logic in `Model.forward_chunk`: builds a
        reset_mask from in-batch EOT positions and the optional `prev_token`
        (last token of previous batch).
        """
        eot_id = self.config.eot_id
        BS, T_total = input_ids.shape
        seg_T = self.config.T
        chunks = []
        for start in range(0, T_total, seg_T):
            end = min(start + seg_T, T_total)
            seg_ids = input_ids[:, start:end]
            eos_positions = (seg_ids == eot_id)
            reset_mask = torch.zeros_like(eos_positions)
            reset_mask[:, 1:] = eos_positions[:, :-1]
            if start == 0 and prev_token is not None:
                reset_mask[:, 0] = (prev_token.to(seg_ids.device) == eot_id)
            elif start > 0:
                reset_mask[:, 0] = (input_ids[:, start - 1] == eot_id)
            if not reset_mask.any():
                reset_mask = None
            chunks.append(self.model.lm.forward_scan_lower(seg_ids, reset_mask=reset_mask))
        return torch.cat(chunks, dim=1)

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

        # Chunk along time to bound VRAM — scale inversely with BS.
        # At K*BS=64, chunk=64 → [64,64,32K] = 0.5GB per chunk (fine).
        rewards = torch.empty(BS, T, device=readouts.device, dtype=torch.float32)
        chunk = max(32, 4096 // max(BS, 1))
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
        """For each modulator call at token t, compute mean reward over [t+1, t+1+W].

        The +1 shift is because `per_token_reward[t]` is computed from
        `readout[t-1]`, which is independent of the action taken at modulator
        call t. The action at t first influences `per_token_reward[t+1]`
        (via `readout[t]`).

        Args:
            per_token_reward: [*, BS, T] on-device (leading dims broadcast, e.g. [K, BS, T])
            call_positions: [n_calls] cpu long tensor — absolute token index of each call
            window: reward window W

        Returns:
            rewards: [*, n_calls, BS] on-device — per-action reward averaged over window
        """
        T = per_token_reward.shape[-1]
        positions = call_positions.to(per_token_reward.device)

        # Build [n_calls, W] index tensor for vectorized gather
        starts = positions + 1                                        # [n_calls]
        offsets = torch.arange(window, device=per_token_reward.device)  # [W]
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)          # [n_calls, W]
        valid = indices < T                                           # [n_calls, W]
        indices = indices.clamp(max=T - 1)
        valid_count = valid.float().sum(dim=1).clamp(min=1)           # [n_calls]

        # Gather: per_token_reward[..., indices] → [..., n_calls, W]
        gathered = per_token_reward[..., indices]                     # [*, BS, n_calls, W]
        gathered = gathered * valid.float()                            # zero out invalid
        out = gathered.sum(dim=-1) / valid_count                      # [*, BS, n_calls]
        # Move n_calls before BS: [..., n_calls, BS]
        return out.transpose(-1, -2)

    @torch.no_grad()
    def _save_mem_state(self) -> dict:
        """Snapshot the memory runtime state so we can reset between rollouts."""
        mem = self.model.memory
        return {
            "h": mem.h.clone(), "msg": mem.msg.clone(), "W": mem.W.clone(),
            "decay": mem.decay.clone(), "hebbian": mem.hebbian.clone(),
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
        mem.decay = snapshot["decay"].clone()
        mem.hebbian = snapshot["hebbian"].clone()
        mem.s_mem_live = snapshot["s_mem_live"].clone()
        mem.s_mem_ema_fast = snapshot["s_mem_ema_fast"].clone()
        mem.prev_readout = snapshot["prev_readout"].clone()
        mem.prev_readout_cell = snapshot["prev_readout_cell"].clone()
        mem.readout_drift = snapshot["readout_drift"].clone()

    @torch.no_grad()
    def rollout(self, batch) -> dict:
        """Run K trajectories on the same batch in parallel via batch-expanded memory.

        Instead of looping K times sequentially, we expand the memory state's
        batch dimension from BS to K*BS and run a single forward_segment_phase2.
        Each of the K "sub-batches" gets the same H_mid and input_ids but
        independent stochastic VQ sampling, producing K divergent trajectories
        in one pass.

        Returns dict:
            mod_inputs: [K, n_calls, BS, NC, mod_in]
            codes:      [K, n_calls, BS, NC, num_levels]
            rewards:    [K, n_calls, BS]  per-action reward
        """
        device = self.device
        K = self.group_size
        input_ids = batch.input_ids.to(device, non_blocking=True)
        prev_token = getattr(batch, "prev_token", None)
        BS, T = input_ids.shape

        self.model.lm.detach_carries()

        # Run lower scan once (LM is frozen and deterministic per batch).
        H_mid = self._run_lower_scan(input_ids, prev_token=prev_token)

        # Snapshot initial memory state at BS
        init_snapshot = self._save_mem_state()

        # Expand memory state: replicate each tensor K times along batch dim.
        # After this, memory operates on K*BS "samples" simultaneously.
        mem = self.model.memory
        for key, val in init_snapshot.items():
            expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
            setattr(mem, key, expanded.clone())

        # Expand H_mid and input_ids: [BS, T, D] -> [K*BS, T, D]
        H_mid_exp = H_mid.unsqueeze(0).expand(K, *H_mid.shape).reshape(K * BS, *H_mid.shape[1:])
        ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])

        # Also expand identity matrix cache

        # Single batched rollout — all K trajectories run in parallel
        result = mem.forward_segment_phase2(
            H_mid_exp, ids_exp, self.model.lm, self.vqvae,
            tau=self.tau, sample=True)

        # Unpack: split K*BS back to (K, BS) using .view() so any
        # non-contiguous layout bugs raise immediately instead of silently
        # scrambling trajectory assignments.
        readouts = result["readouts"].view(K, BS, T, -1)              # [K, BS, T, D]
        call_positions = result["call_positions"]                     # [n_calls] cpu
        n_calls = result["mod_inputs"].shape[0]
        mod_inputs = result["mod_inputs"].view(
            n_calls, K, BS, *result["mod_inputs"].shape[2:]).transpose(0, 1)  # [K, n_calls, BS, NC, mod_in]
        codes = result["codes"].view(
            n_calls, K, BS, *result["codes"].shape[2:]).transpose(0, 1)       # [K, n_calls, BS, NC, L]

        # Compute per-trajectory reward
        init_prev_readout = init_snapshot["prev_readout"]  # [BS, D]
        init_prev_exp = init_prev_readout.unsqueeze(0).expand(K, *init_prev_readout.shape).reshape(K * BS, -1)
        per_token_reward = self._compute_per_token_reward(
            result["readouts"], init_prev_exp, ids_exp)     # [K*BS, T]
        per_token_reward = per_token_reward.reshape(K, BS, T)

        # Windowed rewards — vectorized across K trajectories
        rewards = self._windowed_reward(
            per_token_reward, call_positions, self.reward_window)  # [K, n_calls, BS]

        # Pick best trajectory and restore its end-state
        mean_per_k = rewards.mean(dim=(1, 2))  # [K]
        best_k = mean_per_k.argmax().item()
        best_mean_reward = mean_per_k[best_k].item()

        # Extract best trajectory's end-of-rollout memory state
        for key in init_snapshot:
            full = getattr(mem, key)  # [K*BS, ...]
            best_slice = full[best_k * BS : (best_k + 1) * BS]
            setattr(mem, key, best_slice.clone())

        return {
            "mod_inputs": mod_inputs,
            "codes": codes,
            "rewards": rewards,
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
        K, n_calls, BS, NC, mod_in_dim = rollout_result["mod_inputs"].shape
        num_levels = rollout_result["codes"].shape[-1]

        # Records already live on device from forward_segment_phase2
        mod_inputs = rollout_result["mod_inputs"]    # [K, n_calls, BS, NC, mod_in] f32 gpu
        codes = rollout_result["codes"]              # [K, n_calls, BS, NC, L] long gpu
        rewards = rollout_result["rewards"]          # [K, n_calls, BS] f32 gpu

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
        # Scale chunk size with codes_per_level — more codes = bigger distance
        # tensors in log_prob and entropy ([C*NC, codes, latent_dim])
        codes_per_level = self.vqvae.rvq.codes_per_level
        chunk_size = max(1024, 16384 // max(codes_per_level // 16, 1))
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

            # Modulator forward (on-graph). mi_chunk is already f32 from the rollout.
            raw_action = self.model.memory._modulator_forward(mi_chunk)  # [C, NC, mod_out]

            # Flatten [C, NC] -> [C*NC] for the VQ encoder
            action_flat = raw_action.reshape(C * NC, -1)
            action_norm = self.vqvae.normalize(action_flat)
            z = self.vqvae.encoder(action_norm)                # [C*NC, latent]

            # log pi(codes | z) summed across RVQ levels (on-graph through z)
            codes_flat_flat = c_chunk.reshape(C * NC, num_levels)
            log_pi = self.vqvae.rvq.log_prob(z, codes_flat_flat, tau=self.tau)  # [C*NC]
            log_pi = log_pi.reshape(C, NC)

            # Entropy bonus: encourage policy diversity to prevent code collapse
            entropy = self.vqvae.rvq.entropy(z, tau=self.tau).reshape(C, NC)

            # GRPO loss - entropy bonus (normalized by full M)
            chunk_loss = (-(a_chunk * log_pi) - self.entropy_coeff * entropy).sum() / (M * NC)
            chunk_loss.backward()

            total_loss += chunk_loss.item() * (M * NC)
            total_log_pi += log_pi.detach().sum().item()
            n_chunks += 1

        grad_norm = nn.utils.clip_grad_norm_(
            self.trainable_params, self.max_grad_norm).item()
        self.optimizer.step()

        # Reward distribution stats over the flattened records
        rewards_flat = rollout_result["rewards"].float()
        reward_mean = rewards_flat.mean().item()
        reward_std = rewards_flat.std().item()
        reward_min = rewards_flat.min().item()
        reward_max = rewards_flat.max().item()

        # Codebook usage over the sampled codes (fraction of unique tuples)
        codes_flat_all = rollout_result["codes"].reshape(-1, num_levels)
        n_unique = torch.unique(codes_flat_all, dim=0).shape[0]

        return {
            "loss": total_loss / (M * NC),
            "log_pi_mean": total_log_pi / (M * NC),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "mod_grad_norm": grad_norm,
            "n_chunks": n_chunks,
            "M": M,
            "n_unique_codes": n_unique,
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

            # Persistent jsonl logging for plotting
            log_row = {
                "step": self.global_step,
                "stage_window": stage.reward_window,
                "tokens_seen": tokens_seen,
                "rollout_time": t_rollout,
                "grad_time": t_grad,
                **step_metrics,
            }
            self._append_metrics(log_row)

            if self.global_step % self.log_interval == 0:
                print(f"[p2 step {self.global_step}] "
                      f"W={stage.reward_window} "
                      f"loss={step_metrics['loss']:.4f} "
                      f"r={step_metrics['reward_mean']:+.3f}"
                      f"±{step_metrics['reward_std']:.2f} "
                      f"log_pi={step_metrics['log_pi_mean']:+.2f} "
                      f"mod_gn={step_metrics['mod_grad_norm']:.3f} "
                      f"codes={step_metrics['n_unique_codes']} "
                      f"records={step_metrics['M']} "
                      f"roll={t_rollout:.1f}s grad={t_grad:.1f}s "
                      f"tokens={tokens_seen:,}/{stage.token_budget:,}")

            if (
                self.eval_loader_factory is not None
                and self.eval_interval > 0
                and self.global_step > 0
                and self.global_step % self.eval_interval == 0
            ):
                eval_metrics = self.evaluate(
                    self.eval_loader_factory(), n_batches=self.eval_batches)
                print(f"[p2 eval {self.global_step}] "
                      f"ce={eval_metrics['eval_ce_loss']:.3f} "
                      f"ppl={eval_metrics['eval_ppl']:.1f} "
                      f"mem_pred={eval_metrics['eval_aux_loss']:.3f} "
                      f"({eval_metrics['eval_batches']} batches)")
                self._append_metrics({
                    "step": self.global_step,
                    "stage_window": stage.reward_window,
                    "event": "eval",
                    **eval_metrics,
                })

        elapsed = time.time() - t_stage_start
        print(f"  stage complete: {tokens_seen:,} tokens in {elapsed:.0f}s")

        # Auto-regenerate plots at end of each stage
        if self.metrics_path is not None:
            try:
                from scripts.plot_training import load_metrics, plot_phase2_grpo
                plots_dir = os.path.join(os.path.dirname(self.metrics_path), "plots")
                os.makedirs(plots_dir, exist_ok=True)
                records = load_metrics(self.metrics_path)
                plot_phase2_grpo(records,
                    os.path.join(plots_dir, "phase2_grpo.png"))
            except Exception as e:
                print(f"  (plot regen failed: {e})")
