"""V8 Trainer — joint LM training + PPO neuromodulator training.

Each step:
1. Forward full T-token chunk through V8Model (LM loss + PPO experience)
2. Backward LM loss → update CC params (Adam)
3. PPO update on collected experience → update neuromodulator (PPO-Adam)
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .config import V8Config
from .model import V8Model
from .ppo import PPOTrainer


class V8Trainer:
    """Training loop for v8 model."""

    def __init__(
        self,
        model: V8Model,
        lm_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        dataloader,
        config: V8Config,
        device: torch.device,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        collector=None,
    ):
        self.model = model
        self.lm_optimizer = lm_optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.collector = collector
        self.global_step = 0

        self._states_initialized = False
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # PPO trainer for neuromodulator
        self.ppo_trainer = None  # initialized after model.initialize_states

    def _ensure_ppo(self):
        if self.ppo_trainer is None and self.model.neuromod is not None:
            self.ppo_trainer = PPOTrainer(
                self.model.neuromod, self.config, self.device,
            )

    def train_chunk(self, batch) -> dict:
        """Process one T-token chunk: LM forward + backward + PPO update.

        Args:
            batch: StreamBatch with input_ids [BS, T], target_ids [BS, T],
                   prev_token [BS]

        Returns:
            dict with training metrics
        """
        self.model.train()
        BS = batch.input_ids.shape[0]
        T = batch.input_ids.shape[1]

        if not self._states_initialized:
            self.model.initialize_states(BS, self.device)
            self._states_initialized = True
            self._ensure_ppo()

        input_ids = batch.input_ids.to(self.device)
        target_ids = batch.target_ids.to(self.device)
        prev_token = batch.prev_token.to(self.device)

        eot_id = self.config.eot_id
        reset_mask = (prev_token == eot_id)

        t_start = time.time()

        # ==========================================
        # Forward: full T-token chunk
        # ==========================================
        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp,
        )
        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                reset_mask=reset_mask, collect_ppo=True,
            )

        logits = result["logits"]
        aux_loss = result["aux_loss"]
        ppo_buffer = result["ppo_buffer"]

        # ==========================================
        # LM loss + backward
        # ==========================================
        # Mask EOT tokens from loss
        is_eot = (input_ids == eot_id)
        loss_mask = ~is_eot

        # Per-token CE
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
            reduction='none',
        ).reshape(BS, T)

        valid_mask = loss_mask.float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = (ce_per_token * valid_mask).sum() / valid_count

        total_loss = ce_loss + aux_loss

        self.lm_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.lm.parameters(), self.max_grad_norm
        ).item()
        self.lm_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # ==========================================
        # PPO update (neuromodulator)
        # ==========================================
        ppo_metrics = {}
        if self.ppo_trainer is not None and ppo_buffer is not None and ppo_buffer.step > 0:
            ppo_metrics = self.ppo_trainer.update(ppo_buffer)

        # ==========================================
        # TBPTT boundary
        # ==========================================
        self.model.detach_states()

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed

        # Build metrics
        loss_val = ce_loss.item()
        ppl = min(torch.exp(ce_loss.detach()).item(), 1e6)
        lr = self.lm_optimizer.param_groups[0]["lr"]

        metrics = {
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": lr,
            "tok_s": tok_per_s,
            "grad_norm": grad_norm,
            "elapsed": elapsed,
            **ppo_metrics,
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, max_steps: int, step_callback=None) -> list[dict]:
        """Train for max_steps, yielding metrics per step."""
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
