"""
TBPTTTrainer (v4) — main training loop for neuromorphic LM.

Processes TBPTT chunks as K segments of N tokens. PM/EM updates happen
inside model.forward_segment(), so no external span_ops needed.
"""

import math
import time

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterator, Optional, Callable, TYPE_CHECKING

from ..data.streaming import StreamBatch
from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from .loss import batched_cross_entropy, compute_loss_and_surprise, compute_regularizers

if TYPE_CHECKING:
    from ..debug.collector import MetricsCollector


class TBPTTTrainer:
    def __init__(
        self,
        model: NeuromorphicLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        dataloader: Iterator[StreamBatch],
        config: ModelConfig,
        device: torch.device = torch.device("cpu"),
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        collector: Optional["MetricsCollector"] = None,
        fail_fast: bool = True,
        max_consecutive_zero_valid: int = 3,
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
        self.collector = collector
        self.fail_fast = fail_fast
        self.max_consecutive_zero_valid = max_consecutive_zero_valid
        self._consecutive_zero_valid = 0
        self.override_prev_token: Optional[Tensor] = None

        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        self._states_initialized = False
        self._compile_requested = config.use_compile and device.type == "cuda"
        if self._compile_requested:
            self.model.forward_segment = torch.compile(
                self.model.forward_segment, mode="default"
            )

    def _memory_budget_utils(self) -> tuple:
        """Global PM/EM budget utilization fractions in [0, 1]."""
        pm_util = None
        em_util = None

        if self.config.pm_enabled:
            pm = self.model.pm
            if pm.pm_a is not None:
                total = pm.pm_a.detach().sum().item()
                cap = pm.budget * pm.pm_a.shape[0]
                if cap > 0:
                    pm_util = total / cap

        if self.config.em_enabled:
            em = self.model.em
            if em.em_S is not None:
                total = em.em_S.detach().sum().item()
                cap = em.budget * em.em_S.shape[0]
                if cap > 0:
                    em_util = total / cap

        return pm_util, em_util

    def train_chunk(self, batch: StreamBatch) -> dict:
        """Process one TBPTT chunk (K_segments * N tokens).

        Args:
            batch: StreamBatch with input_ids [BS, T], target_ids [BS, T],
                   prev_token [BS]

        Returns:
            dict with loss, perplexity, tokens_per_sec, etc.
        """
        self.model.train()
        N = self.config.N
        T = batch.input_ids.shape[1]
        BS = batch.input_ids.shape[0]
        eot_id = self.config.eot_id

        # Pre-allocate states (once)
        if not self._states_initialized:
            self.model.initialize_states(BS, self.device)
            self._states_initialized = True

        input_ids = batch.input_ids.to(self.device)
        target_ids = batch.target_ids.to(self.device)

        if self.override_prev_token is not None:
            prev_token = self.override_prev_token.to(self.device)
            self.override_prev_token = None
        else:
            prev_token = batch.prev_token.to(self.device)

        chunk_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_count = torch.tensor(0, device=self.device, dtype=torch.long)
        total_tokens = BS * T
        eot_inputs = 0
        reset_events = 0
        seg_count = 0

        t_start = time.time()
        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        )

        for seg_start in range(0, T, N):
            seg_end = min(seg_start + N, T)
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_targets = target_ids[:, seg_start:seg_end]

            # Doc boundary detection
            if seg_start == 0:
                reset_mask = (prev_token == eot_id)
            else:
                reset_mask = (input_ids[:, seg_start - 1] == eot_id)

            reset_events += int(reset_mask.sum().item())
            is_eot = (seg_ids == eot_id)
            eot_inputs += int(is_eot.sum().item())

            # Forward (R passes + PM/EM updates all inside)
            with amp_ctx:
                logits, aux_loss = self.model.forward_segment(seg_ids, reset_mask)

            # Loss masking: skip EOT positions
            if self.config.reset_on_doc_boundary:
                loss_mask = ~is_eot
            else:
                loss_mask = torch.ones_like(is_eot)

            # Compute loss
            ce_loss, seg_valid = batched_cross_entropy(logits, seg_targets, loss_mask)

            if self.fail_fast and not torch.isfinite(ce_loss):
                raise RuntimeError(
                    f"Non-finite loss at step {self.global_step}, "
                    f"segment [{seg_start}, {seg_end})."
                )

            # aux_loss is mean-reduced (per-token), ce_loss is sum-reduced.
            # Scale aux to sum-reduction so both are divided by valid_count equally.
            chunk_loss = chunk_loss + ce_loss + aux_loss * seg_valid
            valid_count = valid_count + seg_valid
            seg_count += 1

        # Backward + clip + optimizer step
        avg_loss, reg, grad_norm = self._backward_and_step(chunk_loss, valid_count)

        # TBPTT boundary
        self.model.detach_states()

        # Track last token for checkpoint resume
        self._last_prev_token = input_ids[:, -1].detach()

        elapsed = time.time() - t_start

        return self._build_step_metrics(
            avg_loss, reg, valid_count, total_tokens, eot_inputs,
            reset_events, grad_norm, elapsed, seg_count,
        )

    def _backward_and_step(self, chunk_loss, valid_count):
        """Backward, gradient clip, optimizer step."""
        if torch.is_tensor(valid_count):
            avg_loss = chunk_loss / valid_count.float().clamp(min=1.0)
        else:
            avg_loss = chunk_loss / max(valid_count, 1)

        reg = compute_regularizers(self.model)
        total_loss = avg_loss + reg

        if self.fail_fast and not torch.isfinite(total_loss.detach()):
            raise RuntimeError(
                f"Non-finite total loss at step {self.global_step}: "
                f"avg_loss={avg_loss.item()}, reg={reg.item()}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        ).item()

        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            self._nan_steps = getattr(self, '_nan_steps', 0) + 1
            tqdm.write(
                f"WARNING: Non-finite gradient norm at step {self.global_step} "
                f"(skipped, total nan steps: {self._nan_steps})"
            )
        else:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return avg_loss, reg, grad_norm

    def _build_step_metrics(
        self, avg_loss, reg, valid_count, total_tokens, eot_inputs,
        reset_events, grad_norm, elapsed, seg_count,
    ) -> dict:
        """Assemble step metrics dict."""
        valid_count = int(valid_count.item()) if torch.is_tensor(valid_count) else int(valid_count)

        if valid_count == 0:
            self._consecutive_zero_valid += 1
        else:
            self._consecutive_zero_valid = 0

        if (self.fail_fast
                and self._consecutive_zero_valid >= self.max_consecutive_zero_valid):
            raise RuntimeError(
                f"Zero valid tokens for {self._consecutive_zero_valid} consecutive chunks "
                f"(step {self.global_step})."
            )

        valid_fraction = valid_count / max(total_tokens, 1)
        eot_input_fraction = eot_inputs / max(total_tokens, 1)
        reset_fraction = reset_events / max(total_tokens, 1)

        if (self.global_step + 1) % self.log_interval == 0 or self.global_step == 0:
            pm_budget_util, em_budget_util = self._memory_budget_utils()
        else:
            pm_budget_util, em_budget_util = None, None

        _scalars = torch.stack([avg_loss.detach(), reg.detach(),
                                torch.exp(avg_loss.detach())]).cpu()
        _loss_f = _scalars[0].item()
        _reg_f = _scalars[1].item()
        _ppl_f = min(_scalars[2].item(), 1e6)

        step_metrics = {
            "loss": _loss_f,
            "reg": _reg_f,
            "ppl": _ppl_f,
            "tokens_per_sec": total_tokens / max(elapsed, 1e-6),
            "valid_tokens": valid_count,
            "step": self.global_step,
            "grad_norm": grad_norm,
            "valid_fraction": valid_fraction,
            "eot_input_fraction": eot_input_fraction,
            "reset_fraction": reset_fraction,
            "pm_budget_util_global": pm_budget_util,
            "em_budget_util_global": em_budget_util,
        }

        if self.collector is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            do_full = self.collector.should_collect_full(self.global_step)
            extras = {
                "mode": "train",
                "valid_tokens": valid_count,
                "valid_fraction": valid_fraction,
                "eot_input_fraction": eot_input_fraction,
                "reset_fraction": reset_fraction,
                "pm_budget_util_global": pm_budget_util,
                "em_budget_util_global": em_budget_util,
            }
            if do_full:
                basic = {
                    "loss": _loss_f, "ppl": _ppl_f,
                    "lr": lr, "tok_s": step_metrics["tokens_per_sec"],
                    "grad_norm": grad_norm, "reg": _reg_f,
                    "elapsed": elapsed,
                    "nan_grad_steps": getattr(self, '_nan_steps', 0),
                }
                self.collector.log_full(
                    self.global_step, {}, basic, extras=extras, mode="train",
                )
            else:
                self.collector.log_basic(
                    self.global_step, _loss_f, _ppl_f, lr,
                    step_metrics["tokens_per_sec"], grad_norm, _reg_f,
                    elapsed, extras=extras, mode="train",
                )

        return step_metrics

    def train_epoch(
        self,
        num_steps: int,
        step_callback: Optional[Callable[[dict], None]] = None,
    ) -> list:
        """Train for num_steps TBPTT chunks."""
        metrics = []
        pbar = tqdm(
            range(num_steps),
            desc=f"Training (step {self.global_step})",
            unit="step",
            dynamic_ncols=True,
        )

        for step_idx in pbar:
            try:
                batch = next(self.dataloader)
            except StopIteration:
                print(f"Dataloader exhausted at step {step_idx}")
                break

            step_metrics = self.train_chunk(batch)
            self.global_step += 1
            metrics.append(step_metrics)

            m = step_metrics
            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix_str(
                f"loss={m['loss']:.3f} ppl={m['ppl']:.0f} tok/s={m['tokens_per_sec']:.0f} lr={lr:.1e}",
                refresh=False,
            )
            pbar.set_description(f"Step {self.global_step}", refresh=False)

            if self.global_step % self.log_interval == 0:
                tqdm.write(
                    f"step {m['step']:5d} | "
                    f"loss {m['loss']:.4f} | "
                    f"ppl {m['ppl']:.1f} | "
                    f"tok/s {m['tokens_per_sec']:.0f} | "
                    f"lr {lr:.2e}"
                )

            if step_callback is not None:
                step_callback(step_metrics)

        pbar.close()
        return metrics
