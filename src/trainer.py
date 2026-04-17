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
        self.optimizer_step = 0
        self.use_memory = use_memory
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # Freeze modulator (conv encoder + logit head) during cycle phase 1
        # when phase 2 GRPO comes back. requires_grad gates are belt-and-
        # suspenders for direct-Trainer callers; the optimizer step itself
        # just skips zero-grad params.
        self.freeze_modulator = freeze_modulator
        # "Modulator" pool = conv encoder + logit head (what phase 2 GRPO
        # trains). Codebook + decoder are in the DYNAMICS pool — they train
        # during phase 1, get frozen explicitly under --freeze-codebook-decoder.
        mod_params = list(model.memory.modulator.parameters())
        if freeze_modulator:
            for p in mod_params:
                p.requires_grad = False

        # Cache param groups for split grad clipping.
        #   - LM pool: everything in model.lm
        #   - Dynamics pool: memory params other than the modulator encoder
        #     (state/msg/inject/neuron_id MLPs, plasticity logits, codebook,
        #      conv-transpose decoder)
        #   - Modulator pool: just the conv encoder + logit head
        mod_param_set = {id(p) for p in mod_params}
        self._mod_params = mod_params
        self._dyn_params = [
            p for p in model.memory.parameters()
            if id(p) not in mod_param_set
        ]
        self._lm_params = list(model.lm.parameters())

        # Per-pool clip budgets scaled by sqrt(param_count). Under a flat
        # max_grad_norm=1.0 budget, a 67M-param LM pool and a 35M-param
        # modulator pool and a 2.5M dynamics pool all get the same ceiling,
        # which means very different per-parameter step sizes between
        # pools. Scaling by sqrt(pool_params) / sqrt(reference_pool_params)
        # gives each pool a budget proportional to its "natural" gradient
        # magnitude (under the assumption that well-conditioned grads scale
        # as sqrt(N)). The reference pool is the LM (it's the biggest and
        # the best-studied). See audit #7.
        def _count(pool):
            return sum(p.numel() for p in pool)
        lm_count = max(_count(self._lm_params), 1)
        dyn_count = max(_count(self._dyn_params), 1)
        mod_count = max(_count(self._mod_params), 1)
        import math as _math
        self._lm_clip_scale = 1.0
        self._dyn_clip_scale = _math.sqrt(dyn_count / lm_count)
        self._mod_clip_scale = _math.sqrt(mod_count / lm_count)

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
        self.last_consumed_prev_tokens = input_ids[:, -1].cpu()

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp)

        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                use_memory=self.use_memory,
                prev_token=prev_token)

        aux_loss = result["aux_loss"]
        ce_loss = result["ce_loss"]
        total_loss = result["loss"]

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # Whether this step should emit full telemetry. All of the per-
        # parameter reductions below (grad norms, param norms, mem health
        # stats) require CUDA sync and were a real wall-clock tax when run
        # every step. Gate them on the log cadence; non-log steps skip them.
        is_log_step = (
            self.global_step == 0
            or (self.global_step + 1) % self.log_interval == 0
        )

        component_grads = (self.model.memory.compute_component_grad_norms()
                            if is_log_step else {})

        # Three independent clip pools so a spike in one pool doesn't drown
        # the others. Budgets scaled by sqrt(pool_params / lm_params) so
        # pools get comparable per-parameter clipping pressure.
        lm_grad_norm = nn.utils.clip_grad_norm_(
            self._lm_params,
            self.max_grad_norm * self._lm_clip_scale).item()
        dyn_grad_norm = nn.utils.clip_grad_norm_(
            self._dyn_params,
            self.max_grad_norm * self._dyn_clip_scale).item()
        if self.freeze_modulator:
            mod_clip_norm = 0.0
        else:
            mod_clip_norm = nn.utils.clip_grad_norm_(
                self._mod_params,
                self.max_grad_norm * self._mod_clip_scale).item()

        mod_grad_norm = (self.model.memory.compute_mod_grad_norm()
                         if is_log_step else 0.0)

        self.optimizer.step()
        self.optimizer_step += 1
        if self.scheduler is not None:
            self.scheduler.step()

        self.model.detach_states()

        # Telemetry reductions (diagnostic; gated on log cadence).
        lane_stats = {}
        plasticity_rates = {}
        mem_health = {}
        param_norms = {}
        mem_scale_stats = {}
        if is_log_step:
            if self.global_step > 0:
                lane_stats = self.model.memory.compute_lane_divergence()
            plasticity_rates = self.model.memory.compute_plasticity_rates()
            mem_health = self.model.memory.compute_memory_health()
            param_norms = self.model.memory.compute_param_norms()
            mem_scale_stats = self.model.lm.compute_mem_scale_stats()

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed
        loss_val = ce_loss.item()
        ppl = min(math.exp(loss_val), 1e6)

        metrics = {
            "step": self.global_step + 1,
            "optimizer_step": self.optimizer_step,
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "aux_ce_ratio": aux_loss.item() / max(loss_val, 1e-6),
            "lr": self.optimizer.param_groups[0]["lr"],
            "tok_s": tok_per_s,
            "lm_grad_norm": lm_grad_norm,
            "dyn_grad_norm": dyn_grad_norm,
            "mod_clip_norm": mod_clip_norm,
            "mod_grad_norm": mod_grad_norm,
            "elapsed": elapsed,
            "frozen_modulator": self.freeze_modulator,
            **plasticity_rates,
            **mem_health,
            **param_norms,
            **mem_scale_stats,
            **component_grads,
            **lane_stats,
        }

        # Data-stream health — exhaustion/restart counters from the streaming
        # dataloader so we can spot stream collapse during long runs.
        for attr in ("stream_restarts_total",
                     "stream_restarts_last_batch",
                     "streams_exhausted_last_batch"):
            if hasattr(self.dataloader, attr):
                metrics[attr] = getattr(self.dataloader, attr)

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
    def _eval_one_pass(self, eval_loader, n_batches: int, use_memory: bool,
                       warmup_batches: int = 0) -> dict:
        """Run one eval pass with memory on or off. Saves/restores train state.

        The first `warmup_batches` batches run forward without contributing
        to the averaged CE — they're used purely to warm up the memory state
        so later batches measure steady-state performance rather than
        cold-start-then-warming behavior.
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
        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp)
        total_batches = warmup_batches + n_batches
        try:
            for i, batch in enumerate(eval_loader):
                if i >= total_batches:
                    break
                input_ids = batch.input_ids.to(self.device, non_blocking=True)
                target_ids = batch.target_ids.to(self.device, non_blocking=True)
                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(self.device, non_blocking=True)
                with amp_ctx:
                    result = model.forward_chunk(
                        input_ids, target_ids=target_ids,
                        use_memory=use_memory,
                        prev_token=batch_prev if batch_prev is not None else prev_token,
                    )
                prev_token = input_ids[:, -1]
                # Warmup batches: run forward to warm memory state, don't score.
                if i < warmup_batches:
                    continue
                total_ce += result["ce_loss"].item()
                total_aux += result["aux_loss"].item()
                count += 1
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
            return {"ce": 0.0, "aux": 0.0, "ppl": 0.0, "count": 0}
        return {
            "ce": total_ce / count,
            "aux": total_aux / count,
            "ppl": min(math.exp(total_ce / count), 1e6),
            "count": count,
        }

    def evaluate(self, eval_loader, n_batches: int = 8,
                 eval_loader_off=None, warmup_batches: int = 0) -> dict:
        """Run held-out eval: memory-on and memory-off (for leverage gap).

        The memory-off path disables the memory read — it's the upper
        baseline you'd get from the LM alone. The gap (ce_off - ce_on) tells
        you how much the memory path is actually contributing.

        warmup_batches: number of batches to run forward before starting
        to average CE. Lets the memory state warm up so we measure
        steady-state performance rather than cold-start bias.

        If eval_loader_off is None, the memory-off pass is skipped.
        """
        on = self._eval_one_pass(
            eval_loader, n_batches, use_memory=self.use_memory,
            warmup_batches=warmup_batches)
        out = {
            "eval_ce_loss": on["ce"],
            "eval_aux_loss": on["aux"],
            "eval_ppl": on["ppl"],
            "eval_batches": on["count"],
        }
        if eval_loader_off is not None and self.use_memory:
            off = self._eval_one_pass(
                eval_loader_off, n_batches, use_memory=False,
                warmup_batches=warmup_batches)
            out["eval_ce_loss_no_mem"] = off["ce"]
            out["eval_ppl_no_mem"] = off["ppl"]
            # Memory leverage: positive = memory helps, negative = hurts.
            out["mem_leverage_ce"] = off["ce"] - on["ce"]
        return out

