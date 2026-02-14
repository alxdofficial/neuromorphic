"""
TBPTTTrainer — main training loop for neuromorphic LM.

Processes TBPTT chunks with plasticity spans. Handles doc boundary
resets, online loss accumulation, PM commits, and EM writes.
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
from .loss import batched_cross_entropy, compute_regularizers
from . import span_ops

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
        # Tracks last chunk's final tokens for checkpoint resume.
        # When set, overrides the dataloader's prev_token on the next batch
        # so that doc-boundary resets don't wipe restored memory state.
        self.override_prev_token: Optional[Tensor] = None

        # bf16 mixed precision (spec §3): forward/backward in bf16,
        # optimizer + state tensors stay fp32. No GradScaler needed for bf16.
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # torch.compile critical paths for CUDA training
        if config.use_compile and device.type == "cuda":
            model.compile_for_training()

    def _memory_budget_utils(self) -> tuple:
        """Global PM/EM budget utilization fractions in [0, 1]."""
        pm_util = None
        em_util = None

        if self.config.pm_enabled:
            total = 0.0
            cap = 0.0
            for block in self.model.blocks:
                for layer in block.layers:
                    pm = layer.pm
                    if pm.pm_a is None:
                        continue
                    total += pm.pm_a.detach().sum().item()
                    cap += pm.budget * pm.pm_a.shape[0]
            if cap > 0:
                pm_util = total / cap

        if self.config.em_enabled:
            total = 0.0
            cap = 0.0
            for block in self.model.blocks:
                em = block.em
                if em.em_S is None:
                    continue
                total += em.em_S.detach().sum().item()
                cap += em.budget * em.em_S.shape[0]
            if cap > 0:
                em_util = total / cap

        return pm_util, em_util

    def train_chunk(self, batch: StreamBatch) -> dict:
        """Process one TBPTT chunk (T tokens).

        Args:
            batch: StreamBatch with input_ids [BS, T], target_ids [BS, T],
                   prev_token [BS]

        Returns:
            dict with loss, perplexity, tokens_per_sec, etc.
        """
        self.model.train()
        T = batch.input_ids.shape[1]
        P = self.config.P
        eot_id = self.config.eot_id

        input_ids = batch.input_ids.to(self.device)
        target_ids = batch.target_ids.to(self.device)
        # Use override if set (first batch after checkpoint resume)
        if self.override_prev_token is not None:
            prev_token = self.override_prev_token.to(self.device)
            self.override_prev_token = None
        else:
            prev_token = batch.prev_token.to(self.device)

        chunk_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_count = 0
        BS = input_ids.shape[0]
        total_tokens = BS * T
        eot_inputs = 0
        reset_events = 0
        span_valid_mean_accum = 0.0
        span_count = 0
        # Determine if this step should do full collection
        do_full = (self.collector is not None
                   and self.collector.should_collect_full(self.global_step))

        t_start = time.time()
        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        )

        accum = span_ops.SpanAccumulator.create(BS, self.config.B, self.device)
        last_gate_stats = None

        for span_start in range(0, T, P):
            span_end = min(span_start + P, T)

            accum.reset_span()

            span_ids = input_ids[:, span_start:span_end]
            span_targets = target_ids[:, span_start:span_end]

            # Reset mask for first token of this span
            if span_start == 0:
                reset_first = (prev_token == eot_id)
            else:
                reset_first = (input_ids[:, span_start - 1] == eot_id)

            # Forward pass + loss + surprise + PM/EM accumulation
            # Collect gate stats on the last span of full-collection steps
            is_last_span = (span_start + P >= T)
            collect_gates = do_full and is_last_span
            fwd = self._forward_span_and_loss(
                span_ids, span_targets, reset_first, amp_ctx, accum,
                span_start, span_end, collect=collect_gates,
            )
            chunk_loss = chunk_loss + fwd["span_loss"]
            valid_count += fwd["span_valid"]
            eot_inputs += fwd["eot_count"]
            reset_events += fwd["reset_count"]
            if collect_gates:
                last_gate_stats = fwd["gate_stats"]

            # Finalize span: compute surprise mean and stack EM candidates
            result = accum.finalize(self.device, self.config)
            span_surprise_mean = result.surprise_mean
            span_valid_mean_accum += float(accum.valid_tokens.mean().item())
            span_count += 1

            # Update model surprise to span mean (frozen for next span's gates).
            self.model.surprise = span_surprise_mean.unsqueeze(-1)  # [BS, 1]

            # Span boundary: PM commit + EM write
            self._apply_boundary_updates(result, span_surprise_mean)

        # Backward + clip + optimizer step
        avg_loss, reg, grad_norm = self._backward_and_step(
            chunk_loss, valid_count,
        )

        # TBPTT boundary: detach all states
        self.model.detach_states()

        # Track last token for checkpoint resume (prevents false doc-boundary reset)
        self._last_prev_token = input_ids[:, -1].detach().cpu()

        elapsed = time.time() - t_start

        return self._build_step_metrics(
            avg_loss, reg, valid_count, total_tokens, eot_inputs,
            reset_events, grad_norm, elapsed,
            span_valid_mean_accum, span_count, do_full,
            gate_stats=last_gate_stats,
        )

    # ------------------------------------------------------------------
    # train_chunk sub-methods
    # ------------------------------------------------------------------

    def _forward_span_and_loss(
        self, span_ids, span_targets, reset_first, amp_ctx, accum,
        span_start, span_end, collect: bool = False,
    ) -> dict:
        """Forward pass + loss + surprise + PM/EM accumulation for one span.

        Returns dict with span_loss, span_valid, eot_count, reset_count,
        and optionally gate_stats (when collect=True).
        """
        with amp_ctx:
            fwd_result = self.model.forward_span(
                span_ids, reset_first, collect=collect,
            )
            if collect:
                logits_all, x_emb_all, y_wm_all, gate_stats = fwd_result
            else:
                logits_all, x_emb_all, y_wm_all = fwd_result
                gate_stats = None

            if self.fail_fast and not torch.isfinite(logits_all).all():
                raise RuntimeError(
                    f"Non-finite logits at global step {self.global_step}, "
                    f"span [{span_start}, {span_end})."
                )

            # Loss masking: skip EOT input positions [BS, span_P]
            is_eot_all, loss_mask_all = span_ops.compute_loss_mask(
                span_ids, self.config.eot_id, self.config.reset_on_doc_boundary
            )

            # Batched loss
            span_loss, span_valid = batched_cross_entropy(
                logits_all, span_targets, loss_mask_all
            )

            # Per-token surprise (no grad — used for PM/EM gating only)
            with torch.no_grad():
                logp = F.log_softmax(logits_all, dim=-1)
                token_surprise = -logp.gather(-1, span_targets.unsqueeze(-1))
                token_surprise = token_surprise * loss_mask_all.unsqueeze(-1).float()

            # Reset masks for span accumulators
            reset_mask_all = span_ops.compute_reset_mask(
                self.model, span_ids, reset_first,
                self.config.reset_on_doc_boundary,
            )

            # Accumulate surprise per-stream
            span_ops.accumulate_span_surprise(
                token_surprise, loss_mask_all, reset_mask_all,
                self.config.reset_on_doc_boundary,
                accum.surprise_accum, accum.valid_tokens, accum.last_reset,
            )

            # PM eligibility accumulation
            if self.config.pm_enabled:
                span_ops.apply_pm_eligibility_batch(
                    self.model, x_emb_all, token_surprise,
                    reset_mask_all, self.config,
                )

            # EM candidate proposal (stacking deferred to accum.finalize)
            if self.config.em_enabled:
                span_ops.propose_em_candidates(
                    self.model, x_emb_all, y_wm_all, token_surprise,
                    loss_mask_all, accum.em_cand_K, accum.em_cand_V,
                    accum.em_cand_score, accum.em_cand_valid,
                )

        return {
            "span_loss": span_loss,
            "span_valid": span_valid,
            "eot_count": int(is_eot_all.sum().item()),
            "reset_count": int(reset_mask_all.sum().item()),
            "gate_stats": gate_stats,
        }

    def _apply_boundary_updates(
        self, result: span_ops.SpanResult, span_surprise_mean: Tensor,
    ) -> None:
        """PM base_decay + commit, EM neuromod + write at span boundary."""
        if self.config.pm_enabled:
            commit_info = span_ops.apply_pm_boundary(
                self.model, span_surprise_mean,
            )
            if self.collector is not None:
                for b_idx, layer_dict in commit_info.items():
                    for l_idx, p_commit in layer_dict.items():
                        self.collector.record_pm_commit(
                            b_idx, l_idx, p_commit
                        )

        if self.config.em_enabled:
            write_info = span_ops.apply_em_boundary(
                self.model, result.em_stacked, span_surprise_mean, self.config,
            )
            if self.collector is not None:
                for b_idx, novelty_mean, g_em_mean in write_info:
                    self.collector.record_em_write(
                        b_idx, novelty_mean, g_em_mean,
                    )

    def _backward_and_step(
        self, chunk_loss, valid_count,
    ) -> tuple:
        """Backward, gradient clip, optimizer step.

        Returns (avg_loss, reg, grad_norm).
        """
        # Finalize loss
        if valid_count > 0:
            avg_loss = chunk_loss / valid_count
        else:
            avg_loss = chunk_loss

        reg = compute_regularizers(self.model)
        total_loss = avg_loss + reg

        if self.fail_fast and not torch.isfinite(total_loss.detach()):
            raise RuntimeError(
                f"Non-finite total loss at global step {self.global_step}: "
                f"avg_loss={avg_loss.item()}, reg={reg.item()}"
            )

        # Backward + clip + step
        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        ).item()

        if not math.isfinite(grad_norm):
            # Skip this step: zero out nan/inf grads, don't update weights.
            # Scheduler still steps to keep LR schedule consistent.
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

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _build_step_metrics(
        self, avg_loss, reg, valid_count, total_tokens, eot_inputs,
        reset_events, grad_norm, elapsed,
        span_valid_mean_accum, span_count, do_full,
        gate_stats=None,
    ) -> dict:
        """Assemble step metrics dict and log to collector. Pure bookkeeping."""
        tokens = total_tokens

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

        valid_fraction = valid_count / max(tokens, 1)
        eot_input_fraction = eot_inputs / max(tokens, 1)
        reset_fraction = reset_events / max(tokens, 1)
        mean_span_valid_tokens = span_valid_mean_accum / max(span_count, 1)
        pm_budget_util, em_budget_util = self._memory_budget_utils()
        dataloader_stats = {}
        if hasattr(self.dataloader, "monitor_stats"):
            dataloader_stats = self.dataloader.monitor_stats()

        warn_high_grad_norm = float(grad_norm >= 0.95 * self.max_grad_norm)
        warn_low_valid_fraction = float(valid_fraction < 0.2)
        warn_memory_saturation = float(
            (pm_budget_util is not None and pm_budget_util > 0.98)
            or (em_budget_util is not None and em_budget_util > 0.98)
        )

        step_metrics = {
            "loss": avg_loss.item(),
            "reg": reg.item(),
            "ppl": min(torch.exp(avg_loss).item(), 1e6),
            "tokens_per_sec": tokens / max(elapsed, 1e-6),
            "valid_tokens": valid_count,
            "step": self.global_step,
            "grad_norm": grad_norm,
            "valid_fraction": valid_fraction,
            "eot_input_fraction": eot_input_fraction,
            "reset_fraction": reset_fraction,
            "mean_span_valid_tokens": mean_span_valid_tokens,
            "pm_budget_util_global": pm_budget_util,
            "em_budget_util_global": em_budget_util,
            "warn_high_grad_norm": warn_high_grad_norm,
            "warn_low_valid_fraction": warn_low_valid_fraction,
            "warn_memory_saturation": warn_memory_saturation,
        }
        step_metrics.update(dataloader_stats)

        # Log to collector
        if self.collector is not None:
            lr = self.optimizer.param_groups[0]["lr"]
            extras = {
                "mode": "train",
                "valid_tokens": valid_count,
                "valid_fraction": valid_fraction,
                "eot_input_fraction": eot_input_fraction,
                "reset_fraction": reset_fraction,
                "mean_span_valid_tokens": mean_span_valid_tokens,
                "pm_budget_util_global": pm_budget_util,
                "em_budget_util_global": em_budget_util,
                "warn_high_grad_norm": warn_high_grad_norm,
                "warn_low_valid_fraction": warn_low_valid_fraction,
                "warn_memory_saturation": warn_memory_saturation,
            }
            extras.update(dataloader_stats)
            if do_full:
                basic = {
                    "loss": step_metrics["loss"],
                    "ppl": step_metrics["ppl"],
                    "lr": lr,
                    "tok_s": step_metrics["tokens_per_sec"],
                    "grad_norm": grad_norm,
                    "reg": step_metrics["reg"],
                    "elapsed": elapsed,
                    "nan_grad_steps": getattr(self, '_nan_steps', 0),
                }
                self.collector.log_full(
                    self.global_step,
                    gate_stats if gate_stats else {},
                    basic,
                    extras=extras, mode="train",
                )
            else:
                self.collector.log_basic(
                    self.global_step,
                    step_metrics["loss"],
                    step_metrics["ppl"],
                    lr,
                    step_metrics["tokens_per_sec"],
                    grad_norm,
                    step_metrics["reg"],
                    elapsed,
                    extras=extras,
                    mode="train",
                )

        return step_metrics

    def train_epoch(
        self,
        num_steps: int,
        step_callback: Optional[Callable[[dict], None]] = None,
    ) -> list:
        """Train for num_steps TBPTT chunks.

        Args:
            num_steps: number of chunks to process

        Returns:
            list of per-step metric dicts
        """
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

            # Update progress bar postfix
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
