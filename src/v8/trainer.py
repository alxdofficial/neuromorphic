"""V8/v9 Trainer — LM backprop + ES for memory graph.

Each step:
1. Full scan over T tokens (lower scan + PCM + memory graph + upper scan)
2. LM backward on CE + aux_loss
3. Every es_collect_chunks: ES scoring + parameter update for memory graph
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
    """Training loop: LM by backprop, memory graph by ES."""

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
        use_memory: bool = True,
    ):
        self.model = model
        self.lm_optimizer = lm_optimizer
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

        # ES collection buffer
        self.es_collect_chunks = config.es_collect_chunks
        self._es_buffer: list[dict] = []
        self._es_pre_mg_state: dict | None = None
        self._es_pre_mg_params: dict | None = None

    def train_chunk(self, batch) -> dict:
        """Process one chunk. LM by backprop, ES data collected."""
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

        # Save memory graph state before ES collection window
        if self.use_memory and len(self._es_buffer) == 0:
            mg = self.model.memory
            if mg is not None and mg.is_initialized():
                self._es_pre_mg_state = {
                    k: v.clone() for k, v in mg.runtime_state_dict().items()}
                self._es_pre_mg_params = {
                    k: v.clone() for k, v in mg.get_es_params().items()}

        # Snapshot upper carries BEFORE forward (for ES replay)
        pre_upper_carries = None
        if self.use_memory:
            split = self.config.scan_split_at
            L = self.config.L_total
            pre_upper_carries = [
                self.model.lm._carries[split + i].clone()
                if self.model.lm._carries[split + i] is not None else None
                for i in range(L - split)]

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp)

        # Forward
        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                reset_mask=reset_mask,
                use_memory=self.use_memory,
                has_reset=has_reset)

        logits = result["logits"]
        aux_loss = result["aux_loss"]

        # CE loss
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1), reduction='none'
        ).reshape(BS, T)

        is_eot = (input_ids == eot_id)
        valid_mask = (~is_eot).float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = (ce_per_token * valid_mask).sum() / valid_count
        lm_loss = ce_loss + aux_loss

        # LM backward
        grad_norm = 0.0
        self.lm_optimizer.zero_grad(set_to_none=True)
        lm_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.lm.parameters(), self.max_grad_norm).item()
        self.lm_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Collect ES data
        es_metrics = {}
        if self.use_memory:
            eot_at = (input_ids == eot_id)

            self._es_buffer.append({
                "cc_segments": result["cc_segments"],
                "H_mid": result["H_mid"],
                "surprise": result["surprise"],
                "target_ids": target_ids.detach(),
                "eot_at": eot_at.detach(),
                "pre_upper_carries": pre_upper_carries,
            })

            # ES update when buffer is full
            if len(self._es_buffer) >= self.es_collect_chunks:
                scoring = self.model.score_es_trajectories(
                    self._es_buffer,
                    self._es_pre_mg_state,
                    self._es_pre_mg_params)

                self.model.apply_es_gradient(scoring)

                es_metrics = {
                    "es_adv_std": scoring["advantages"].std().item(),
                    "es_loss_best": scoring["trajectory_losses"].min().item(),
                    "es_loss_worst": scoring["trajectory_losses"].max().item(),
                    "es_loss_mean": scoring["trajectory_losses"].mean().item(),
                    "es_best_traj": scoring["best_traj_idx"],
                }

                self._es_buffer = []
                self._es_pre_mg_state = None
                self._es_pre_mg_params = None

        # Detach states
        self.model.detach_states()

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed
        loss_val = ce_loss.item()
        ppl = min(math.exp(loss_val), 1e6)

        metrics = {
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": self.lm_optimizer.param_groups[0]["lr"],
            "tok_s": tok_per_s,
            "grad_norm": grad_norm,
            "elapsed": elapsed,
            **es_metrics,
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
