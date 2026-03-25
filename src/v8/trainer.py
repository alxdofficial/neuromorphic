"""V8 Trainer — joint LM training + RL for neuromodulator.

Each step:
1. Full scan over T tokens (lower scan shared, upper scan for real + counterfactual)
2. Segmented memory processing: neuromod acts + counterfactual (K neurons reverted)
3. Per-segment CE loss for real and counterfactual paths
4. LM backward on logits (every chunk)
5. Every rl_collect_chunks chunks: counterfactual + GAE advantages, neuromod backward
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .config import V8Config
from .model import V8Model


class V8Trainer:
    """Training loop for v8 model."""

    def __init__(
        self,
        model: V8Model,
        lm_optimizer: torch.optim.Optimizer,
        neuromod_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        dataloader,
        config: V8Config,
        device: torch.device,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        collector=None,
        use_memory: bool = True,
        use_neuromod: bool = True,
        neuromod_scheduler=None,
    ):
        self.model = model
        self.lm_optimizer = lm_optimizer
        self.neuromod_optimizer = neuromod_optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.collector = collector
        self.global_step = 0
        self.use_memory = use_memory
        self.use_neuromod = use_neuromod
        self.neuromod_scheduler = neuromod_scheduler

        self._states_initialized = False
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # RL collection buffer: accumulate across chunks before updating neuromod
        self.rl_collect_chunks = config.rl_collect_chunks
        self._rl_buffer: list[dict] = []

    def train_chunk(self, batch) -> dict:
        """Process one T-token chunk.

        Returns dict with training metrics.
        """
        self.model.train()
        BS = batch.input_ids.shape[0]
        T = batch.input_ids.shape[1]

        if not self._states_initialized:
            self.model.initialize_states(BS, self.device)
            self._states_initialized = True

        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)

        # Compute reset_mask on CPU first (avoids GPU-CPU sync from .any())
        eot_id = self.config.eot_id
        has_reset = (batch.prev_token == eot_id).any().item()  # CPU check, no sync
        reset_mask = batch.prev_token.to(self.device, non_blocking=True) == eot_id

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

        # ==========================================
        # Forward: single scan + segmented memory
        # ==========================================
        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                reset_mask=reset_mask,
                use_memory=self.use_memory,
                has_reset=has_reset,
                use_neuromod=self.use_neuromod,
            )

        logits = result["logits"]
        aux_loss = result["aux_loss"]
        rl_data = result["rl_data"]

        # ==========================================
        # CE loss (computed ONCE, used for both LM backward and RL rewards)
        # ==========================================
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
            reduction='none',
        ).reshape(BS, T)

        is_eot = (input_ids == eot_id)
        valid_mask = (~is_eot).float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = (ce_per_token * valid_mask).sum() / valid_count

        lm_loss = ce_loss + aux_loss

        # LM backward + optimizer step (skip when LM is fully frozen)
        grad_norm = 0.0
        if lm_loss.requires_grad:
            self.lm_optimizer.zero_grad(set_to_none=True)
            lm_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.lm.parameters(), self.max_grad_norm
            ).item()
            self.lm_optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        # ==========================================
        # Neuromodulator: derive RL rewards from the same CE, accumulate
        # ==========================================
        rl_metrics = {}
        if rl_data is not None:
            # Compute per-segment rewards from real and N_cf counterfactual CEs
            with torch.no_grad():
                n_segments = rl_data["n_segments"]
                action_every = rl_data["action_every"]
                eot_at = rl_data["eot_at"]
                reward_mask = (~eot_at).to(dtype=ce_per_token.dtype)

                seg_ce = ce_per_token.detach().view(BS, n_segments, action_every)
                seg_mask = reward_mask.view(BS, n_segments, action_every)
                seg_losses = (seg_ce * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1).clamp(min=1)
                seg_rewards = -seg_losses

                # N_cf counterfactual CEs → per-trajectory seg_rewards
                all_seg_rewards_cf = []
                for logits_B in rl_data["logits_Bs"]:
                    ce_cf = F.cross_entropy(
                        logits_B.reshape(-1, self.config.vocab_size),
                        target_ids.reshape(-1),
                        reduction='none',
                    ).reshape(BS, T)
                    seg_ce_cf = ce_cf.view(BS, n_segments, action_every)
                    seg_losses_cf = (seg_ce_cf * seg_mask).sum(dim=-1) / seg_mask.sum(dim=-1).clamp(min=1)
                    all_seg_rewards_cf.append(-seg_losses_cf)

            rl_data["seg_rewards"] = seg_rewards
            rl_data["all_seg_rewards_cf"] = all_seg_rewards_cf  # list of N_cf [BS, n_seg]
            rl_data["loss"] = ce_loss.item()
            # Drop logits to free memory
            del rl_data["logits_Bs"]
            self._rl_buffer.append(rl_data)

            # Per-chunk logging (average CF advantage across trajectories)
            avg_cf_adv = sum(
                ((-r) - seg_losses).mean().item() for r in all_seg_rewards_cf
            ) / len(all_seg_rewards_cf)
            cf_adv = avg_cf_adv
            rl_metrics = {
                "rl_loss": rl_data["loss"],
                "rl_seg_loss_first": seg_losses[:, 0].mean().item(),
                "rl_seg_loss_last": seg_losses[:, -1].mean().item(),
                "rl_cf_adv": cf_adv,
            }

            # Update neuromod when buffer is full
            if len(self._rl_buffer) >= self.rl_collect_chunks:
                self.neuromod_optimizer.zero_grad(set_to_none=True)

                combined = self.model.compute_counterfactual_advantages(
                    self._rl_buffer)

                rl_losses = self.model.replay_for_neuromod_grads(
                    combined,
                    amp_enabled=self.use_amp,
                    amp_dtype=self.amp_dtype,
                )

                nm_grad_norm = nn.utils.clip_grad_norm_(
                    self.model.neuromod.parameters(), self.max_grad_norm
                ).item()
                self.neuromod_optimizer.step()

                adv = combined["advantages"]
                rl_metrics.update({
                    "rl_policy_loss": rl_losses["policy_loss"],
                    "rl_entropy": rl_losses["entropy"],
                    "rl_adv_mean": adv.mean().item(),
                    "rl_adv_std": adv.std().item(),
                    "rl_nm_grad_norm": nm_grad_norm,
                })

                # Clear buffer
                self._rl_buffer = []

            # Step neuromod scheduler every chunk (not just on RL updates)
            # so the LR schedule tracks global training progress
            if self.neuromod_scheduler is not None:
                self.neuromod_scheduler.step()

        # ==========================================
        # TBPTT boundary
        # ==========================================
        self.model.detach_states()

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed

        loss_val = ce_loss.item()
        ppl = min(math.exp(loss_val), 1e6)  # compute on CPU, avoid extra GPU kernel
        lr = self.lm_optimizer.param_groups[0]["lr"]

        nm_lr = (self.neuromod_optimizer.param_groups[0]["lr"]
                 if self.use_memory else 0.0)
        metrics = {
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": lr,
            "nm_lr": nm_lr,
            "tok_s": tok_per_s,
            "grad_norm": grad_norm,
            "elapsed": elapsed,
            **rl_metrics,
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, max_steps: int, step_callback=None) -> list[dict]:
        all_metrics = []
        pbar = tqdm(total=max_steps, desc="v8 train", unit="step")

        for step_idx, batch in enumerate(self.dataloader):
            if step_idx >= max_steps:
                break

            metrics = self.train_chunk(batch)
            all_metrics.append(metrics)

            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                ppl=f"{metrics['ppl']:.1f}",
                toks=f"{metrics['tok_s']/1e3:.1f}K",
            )
            pbar.update(1)

            if step_callback is not None:
                step_callback(metrics)

        pbar.close()
        return all_metrics
