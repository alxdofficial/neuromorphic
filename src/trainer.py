"""Trainer — single optimizer, joint LM + memory graph training."""

import json
import math
import os
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
        metrics_path: str | None = None,
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

        # Cache param groups for split grad clipping.
        #   - LM pool: everything in model.lm
        #   - Dynamics pool: memory params that are NOT the modulator
        #   - Modulator pool: the 4 modulator tensors
        # Each pool gets clipped independently so a spike in the dynamics pool
        # doesn't starve the modulator (or vice versa) via shared clipping.
        mod_param_set = {id(p) for p in (
            model.memory.mod_w1, model.memory.mod_b1,
            model.memory.mod_w2, model.memory.mod_b2,
        )}
        self._mod_params = [
            model.memory.mod_w1, model.memory.mod_b1,
            model.memory.mod_w2, model.memory.mod_b2,
        ]
        self._dyn_params = [
            p for p in model.memory.parameters()
            if id(p) not in mod_param_set
        ]

        # Action collection: snapshot one modulator output per training step.
        self.collect_actions = collect_actions
        self.action_buffer: list[torch.Tensor] = []

        # JSONL metrics log. Appended per step. One dict per line.
        self.metrics_path = metrics_path
        if metrics_path is not None:
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    def _append_metrics(self, metrics: dict):
        if self.metrics_path is None:
            return
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

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

        # Component grad norms (read BEFORE clip to see the raw signal).
        component_grads = self.model.memory.compute_component_grad_norms()

        # Three independent clip pools so a spike in one pool doesn't drown
        # the others. Each gets the same max_grad_norm budget.
        lm_grad_norm = nn.utils.clip_grad_norm_(
            self.model.lm.parameters(), self.max_grad_norm).item()
        dyn_grad_norm = nn.utils.clip_grad_norm_(
            self._dyn_params, self.max_grad_norm).item()
        if self.freeze_modulator:
            # Modulator params have no grad in this mode; no clip to do.
            mod_clip_norm = 0.0
        else:
            mod_clip_norm = nn.utils.clip_grad_norm_(
                self._mod_params, self.max_grad_norm).item()
        # Backward-compat aggregate (sqrt of sum of squares).
        mem_grad_norm = math.sqrt(dyn_grad_norm ** 2 + mod_clip_norm ** 2)

        # Phase-1 telemetry: per-cell modulator grad norm (POST clip of the
        # modulator pool — isolated from dynamics spikes now).
        mod_grad_norm = self.model.memory.compute_mod_grad_norm()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.model.detach_states()

        # Phase-1 telemetry: snapshot modulator action stats + memory health.
        mod_stats = self.model.memory.compute_modulator_stats()
        mem_health = self.model.memory.compute_memory_health()
        param_norms = self.model.memory.compute_param_norms()

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
            "step": self.global_step + 1,
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "tok_s": tok_per_s,
            "lm_grad_norm": lm_grad_norm,
            "mem_grad_norm": mem_grad_norm,
            "dyn_grad_norm": dyn_grad_norm,
            "mod_clip_norm": mod_clip_norm,
            "mod_grad_norm": mod_grad_norm,
            "mod_action_norm": mod_stats["mod_action_norm"],
            "mod_action_var": mod_stats["mod_action_var"],
            "elapsed": elapsed,
            "frozen_modulator": self.freeze_modulator,
            **mem_health,
            **param_norms,
            **component_grads,
        }

        self.global_step += 1
        self._append_metrics(metrics)
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

    @torch.no_grad()
    def evaluate(self, eval_loader, n_batches: int = 8) -> dict:
        """Run a held-out forward-only evaluation pass.

        Uses a fresh memory state (snapshots and restores the current training
        state around the call) so eval results are independent of the current
        training memory carry. Model is placed in eval mode so dropout is off.
        """
        model = self.model
        memory = model.memory

        # Snapshot training state so we can restore after eval
        train_mem_state = (
            memory.runtime_state_dict() if memory._initialized else None
        )
        train_lm_carries = [
            h.clone() if h is not None else None for h in model.lm._carries
        ]
        train_initialized = model._initialized
        was_training = model.training

        # Reset to fresh memory/LM state for a clean eval run
        memory._initialized = False
        model._initialized = False
        model.lm._carries = [None] * self.config.L_total
        model.train(False)

        total_ce = 0.0
        total_aux = 0.0
        count = 0
        prev_token = None
        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp)
        try:
            for i, batch in enumerate(eval_loader):
                if i >= n_batches:
                    break
                input_ids = batch.input_ids.to(self.device, non_blocking=True)
                target_ids = batch.target_ids.to(self.device, non_blocking=True)
                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(self.device, non_blocking=True)
                with amp_ctx:
                    result = model.forward_chunk(
                        input_ids, target_ids=target_ids,
                        use_memory=self.use_memory,
                        prev_token=batch_prev if batch_prev is not None else prev_token,
                    )
                total_ce += result["ce_loss"].item()
                total_aux += result["aux_loss"].item()
                count += 1
                # Track last-in-batch token for cross-chunk EOT reset
                prev_token = input_ids[:, -1]
        finally:
            # Restore training state regardless of whether eval succeeded
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
