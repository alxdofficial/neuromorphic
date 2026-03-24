"""Phase 1 Trainer — joint backprop for LM + memory graph.

No RL, no neuromodulator. Memory graph parameters trained by backprop
through K-step gradient windows. Normalizations projected after optimizer step.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import V8Config
from .model_backprop import V8ModelBackprop


class V8TrainerBackprop:
    """Phase 1 training loop: LM + backprop memory graph."""

    def __init__(
        self,
        model: V8ModelBackprop,
        optimizer: torch.optim.Optimizer,
        scheduler,
        dataloader,
        config: V8Config,
        device: torch.device,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        use_memory: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.global_step = 0
        self.use_memory = use_memory

        self._states_initialized = False
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

    def train_chunk(self, batch) -> dict:
        self.model.train()
        BS = batch.input_ids.shape[0]
        T = batch.input_ids.shape[1]

        if not self._states_initialized:
            self.model.initialize_states(BS, self.device)
            self._states_initialized = True

        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)

        eot_id = self.config.eot_id
        has_reset = (batch.prev_token == eot_id).any().item()
        reset_mask = batch.prev_token.to(self.device, non_blocking=True) == eot_id

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

        # Forward
        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                reset_mask=reset_mask,
                use_memory=self.use_memory,
                has_reset=has_reset,
            )

        logits = result["logits"]
        aux_loss = result["aux_loss"]

        # CE loss
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
            reduction='none',
        ).reshape(BS, T)

        is_eot = (input_ids == eot_id)
        valid_mask = (~is_eot).float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = (ce_per_token * valid_mask).sum() / valid_count

        total_loss = ce_loss + aux_loss

        # Backward + step
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        ).item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Project memory graph params back to constraints
        if self.use_memory and self.model.mem_graph is not None:
            self.model.mem_graph.project_params()

        # TBPTT boundary
        self.model.detach_states()

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed

        loss_val = ce_loss.item()
        ppl = min(math.exp(loss_val), 1e6)
        lr = self.optimizer.param_groups[0]["lr"]

        metrics = {
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": lr,
            "tok_s": tok_per_s,
            "grad_norm": grad_norm,
            "elapsed": elapsed,
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, max_steps: int, step_callback=None) -> list[dict]:
        all_metrics = []
        pbar = tqdm(total=max_steps, desc="v8-bp train", unit="step")

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
