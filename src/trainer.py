"""Trainer — single optimizer, joint LM + memory graph training."""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model.config import Config
from .model.model import Model


class Trainer:

    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        dataloader,
        config: Config,
        device: torch.device,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        use_memory: bool = True,
        freeze_modulator: bool = False,
        collect_actions: bool = False,
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
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # Cycle-1+: freeze modulator so phase 1 TBPTT doesn't undo phase 2 GRPO.
        self.freeze_modulator = freeze_modulator
        if freeze_modulator:
            for p in (model.memory.mod_w1, model.memory.mod_b1,
                      model.memory.mod_w2, model.memory.mod_b2):
                p.requires_grad = False

        # Action collection: snapshot one modulator output per training step.
        self.collect_actions = collect_actions
        self.action_buffer: list[torch.Tensor] = []

    def train_chunk(self, batch) -> dict:
        self.model.train()
        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)
        prev_token = getattr(batch, "prev_token", None)
        if prev_token is not None:
            prev_token = prev_token.to(self.device, non_blocking=True)
        BS, T = input_ids.shape

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp)

        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                use_memory=self.use_memory,
                prev_token=prev_token)

        logits = result["logits"]
        aux_loss = result["aux_loss"]
        ce_loss = result["ce_loss"]
        total_loss = result["loss"]

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        lm_grad_norm = nn.utils.clip_grad_norm_(
            self.model.lm.parameters(), self.max_grad_norm).item()
        mem_grad_norm = nn.utils.clip_grad_norm_(
            self.model.memory.parameters(), self.max_grad_norm).item()

        # Phase-1 telemetry: per-cell modulator grad norm (read after clip,
        # before optimizer.step()).
        mod_grad_norm = self.model.memory.compute_mod_grad_norm()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.model.detach_states()

        # Phase-1 telemetry: snapshot modulator action stats after the step.
        mod_stats = self.model.memory.compute_modulator_stats()

        # Optional action collection for codebook fitting.
        if self.collect_actions:
            action = self.model.memory.collect_modulator_action()
            if action is not None:
                self.action_buffer.append(action)

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed
        loss_val = ce_loss.item()
        ppl = min(math.exp(loss_val), 1e6)

        metrics = {
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "tok_s": tok_per_s,
            "lm_grad_norm": lm_grad_norm,
            "mem_grad_norm": mem_grad_norm,
            "mod_grad_norm": mod_grad_norm,
            "mod_action_norm": mod_stats["mod_action_norm"],
            "mod_action_var": mod_stats["mod_action_var"],
            "elapsed": elapsed,
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, max_steps: int, step_callback=None) -> list[dict]:
        all_metrics = []
        pbar = tqdm(total=max_steps, desc="train", unit="step")

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

    def flush_action_database(self, path: str) -> int:
        """Save collected modulator actions to disk.

        Actions were accumulated per step as [BS, NC, mod_out] float32 cpu tensors.
        Concatenate along batch, reshape to [N * NC, mod_out] so each cell's
        per-call output is one sample, and save. Returns the number of samples.
        """
        if not self.action_buffer:
            return 0
        stacked = torch.cat(self.action_buffer, dim=0)   # [N_steps*BS, NC, mod_out]
        samples = stacked.reshape(-1, stacked.shape[-1])  # [N*NC, mod_out]
        torch.save(samples, path)
        n = samples.shape[0]
        self.action_buffer.clear()
        return n
