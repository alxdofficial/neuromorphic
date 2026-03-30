"""v10-gnn Trainer -- single optimizer, joint LM + memory graph + decoder training.

Each step:
1. Forward: lower scan + PCM -> memory graph -> decoder -> logits
2. CE loss (with valid mask for non-padding/EOS tokens) + aux_loss
3. Single backward through entire model
4. Separate grad clip for lm, memory, decoder
5. Single optimizer step
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import V10Config
from .model import V10Model


class V10Trainer:
    """Training loop for v10-gnn: everything by backprop."""

    def __init__(
        self,
        model: V10Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        dataloader,
        config: V10Config,
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
        """Process one chunk. Single forward + backward."""
        self.model.train()
        BS = batch.input_ids.shape[0]
        T = batch.input_ids.shape[1]

        if not self._states_initialized:
            if not self.model._states_initialized:
                self.model.initialize_states(BS)
            self._states_initialized = True

        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)

        eot_id = self.config.eot_id
        # Chunk-boundary reset: [BS] bool — True if previous chunk ended with EOS
        chunk_reset = (batch.prev_token.to(self.device, non_blocking=True) == eot_id)

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp)

        # Forward
        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                reset_mask=chunk_reset,
                use_memory=self.use_memory)

        logits = result["logits"]
        aux_loss = result["aux_loss"]

        # CE loss with valid mask (mask out EOS tokens)
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1), reduction='none'
        ).reshape(BS, T)

        is_eot = (input_ids == eot_id)
        valid_mask = (~is_eot).float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = (ce_per_token * valid_mask).sum() / valid_count
        total_loss = ce_loss + aux_loss

        # Single backward + optimizer step
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # Separate grad norms for monitoring: lm, memory, decoder
        lm_grad_norm = nn.utils.clip_grad_norm_(
            self.model.lm.parameters(), self.max_grad_norm).item()
        mem_grad_norm = nn.utils.clip_grad_norm_(
            self.model.memory.parameters(), self.max_grad_norm).item()
        dec_grad_norm = nn.utils.clip_grad_norm_(
            self.model.decoder.parameters(), self.max_grad_norm).item()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # TBPTT boundary: detach states for next chunk
        self.model.detach_states()

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
            "dec_grad_norm": dec_grad_norm,
            "elapsed": elapsed,
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, max_steps: int, step_callback=None) -> list[dict]:
        all_metrics = []
        pbar = tqdm(total=max_steps, desc="v10 train", unit="step")

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
