"""
TBPTTTrainer — main training loop for neuromorphic LM.

Processes TBPTT chunks with plasticity spans. Handles doc boundary
resets, online loss accumulation, PM commits, and EM writes.
"""

import math
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterator, Optional, Callable, TYPE_CHECKING

from ..data.streaming import StreamBatch
from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from .loss import online_cross_entropy, compute_regularizers

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
        span_surprise_accum = torch.zeros(BS, device=self.device)
        span_valid_tokens = torch.zeros(BS, device=self.device)  # per-stream count

        # Determine if this step should do full collection
        do_full = (self.collector is not None
                   and self.collector.should_collect_full(self.global_step))
        gate_stats = None

        t_start = time.time()
        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        )

        for span_start in range(0, T, P):
            span_end = min(span_start + P, T)
            is_last_span = (span_end >= T)

            # EM candidate buffers (per block)
            B = self.config.B
            cand_K = [[] for _ in range(B)]
            cand_V = [[] for _ in range(B)]
            cand_score = [[] for _ in range(B)]
            cand_token_valid = [[] for _ in range(B)]
            # Per-stream last reset position within this span.
            # Used to mask out pre-reset candidates without in-place edits.
            span_last_reset = torch.zeros(BS, dtype=torch.long, device=self.device)

            span_surprise_accum.zero_()
            span_valid_tokens.zero_()

            with amp_ctx:
                for t in range(span_start, span_end):
                    # Doc boundary reset mask
                    if t == 0:
                        reset_mask = (prev_token == eot_id)
                    else:
                        reset_mask = (input_ids[:, t - 1] == eot_id)

                    # Clear span-level accumulators at doc boundary for streams
                    # that reset in this span (only when reset is enabled).
                    if reset_mask.any() and self.config.reset_on_doc_boundary:
                        local_t = t - span_start
                        span_last_reset[reset_mask] = local_t
                        span_surprise_accum[reset_mask] = 0
                        span_valid_tokens[reset_mask] = 0

                    # Collect on last token of last span for full-collection steps
                    collect_this = (do_full and is_last_span and t == span_end - 1)

                    # Forward one token
                    result = self.model.forward_one_token(
                        input_ids[:, t], reset_mask, collect=collect_this
                    )
                    if collect_this:
                        logits, x_emb, y_wm, gate_stats = result
                    else:
                        logits, x_emb, y_wm = result

                    if self.fail_fast and not torch.isfinite(logits).all():
                        raise RuntimeError(
                            f"Non-finite logits at global step {self.global_step}, "
                            f"span [{span_start}, {span_end}), token index {t}."
                        )

                    # Loss masking: skip EOT input positions
                    is_eot = (input_ids[:, t] == eot_id)
                    if self.config.reset_on_doc_boundary:
                        loss_mask = ~is_eot
                    else:
                        loss_mask = torch.ones_like(is_eot)
                    eot_inputs += int(is_eot.sum().item())
                    reset_events += int(reset_mask.sum().item())

                    # Accumulate loss
                    token_loss, count = online_cross_entropy(
                        logits, target_ids[:, t], loss_mask
                    )
                    chunk_loss = chunk_loss + token_loss
                    valid_count += count

                    # Update surprise (mask out excluded EOT positions so they
                    # do not drive PM/EM controller statistics).
                    self.model.update_surprise(logits, target_ids[:, t], mask=loss_mask)

                    # Accumulate surprise for controller decisions
                    if self.model.surprise is not None:
                        span_surprise_accum = (
                            span_surprise_accum
                            + self.model.surprise.squeeze(-1) * loss_mask.float()
                        )
                    span_valid_tokens = span_valid_tokens + loss_mask.float()

                    # Buffer EM candidates (if enabled)
                    if self.config.em_enabled:
                        for b, block in enumerate(self.model.blocks):
                            # Get final layer hidden state for this block
                            h_final = block.layers[-1].h
                            if h_final is not None:
                                k_c, v_c, nov = block.em.propose_candidate(
                                    x_emb, y_wm, h_final,
                                    self.model.surprise
                                )
                                cand_K[b].append(k_c)
                                cand_V[b].append(v_c)
                                cand_score[b].append(nov)
                                cand_token_valid[b].append(loss_mask)

            # Per-stream surprise mean for this span (used by both PM and EM)
            span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)
            span_valid_mean_accum += float(span_valid_tokens.mean().item())
            span_count += 1

            # Span boundary: PM base decay + commits
            if self.config.pm_enabled:
                # Base decay: all streams lose pm_a each span (FIX-3)
                for block in self.model.blocks:
                    for layer in block.layers:
                        layer.pm.base_decay()

                commit_info = self.model.commit_at_boundary(
                    span_surprise=span_surprise_mean.detach()
                )

                # Record commit rates for collector
                if self.collector is not None:
                    for b_idx, layer_dict in commit_info.items():
                        for l_idx, commit_mask in layer_dict.items():
                            self.collector.record_pm_commit(
                                b_idx, l_idx, commit_mask
                            )

            # Span boundary: EM writes
            if self.config.em_enabled:
                for b, block in enumerate(self.model.blocks):
                    if len(cand_K[b]) > 0:
                        stacked_K = torch.stack(cand_K[b], dim=1)      # [BS, P, D_em]
                        stacked_V = torch.stack(cand_V[b], dim=1)      # [BS, P, D_em]
                        stacked_score = torch.stack(cand_score[b], dim=1)  # [BS, P]
                        stacked_token_valid = torch.stack(cand_token_valid[b], dim=1)  # [BS, P]

                        # Valid candidates are positions at/after last reset for
                        # each stream in this span.
                        S = stacked_score.shape[1]
                        pos = torch.arange(S, device=self.device).unsqueeze(0)  # [1, S]
                        cand_valid = (
                            pos >= span_last_reset.unsqueeze(1)
                        ) & stacked_token_valid.bool()                            # [BS, S]

                        # Controller decides write mask and strength
                        em_usage = block.em.em_S.sum(dim=-1) if block.em.em_S is not None else torch.zeros_like(span_surprise_mean)
                        # Per-stream novelty mean over valid candidate positions.
                        cand_valid_f = cand_valid.float()
                        cand_count = cand_valid_f.sum(dim=-1).clamp(min=1)  # [BS]
                        cand_novelty_mean = (stacked_score * cand_valid_f).sum(dim=-1) / cand_count

                        write_mask, g_em = block.em_controller.forward(
                            span_surprise_mean, em_usage, cand_novelty_mean
                        )

                        # Record write rates for collector
                        if self.collector is not None:
                            self.collector.record_em_write(
                                b, write_mask,
                                cand_novelty_mean.mean().item()
                            )

                        block.em.write_at_boundary(
                            stacked_K, stacked_V, stacked_score,
                            write_mask, g_em, cand_valid=cand_valid
                        )

        # Finalize loss
        if valid_count > 0:
            avg_loss = chunk_loss / valid_count
        else:
            avg_loss = chunk_loss

        # Add regularizers
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

        if self.fail_fast and not math.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite gradient norm at global step {self.global_step}."
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # TBPTT boundary: detach all states
        self.model.detach_states()

        # Track last token for checkpoint resume (prevents false doc-boundary reset)
        self._last_prev_token = input_ids[:, -1].detach().cpu()

        elapsed = time.time() - t_start
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
            if do_full and gate_stats is not None:
                basic = {
                    "loss": step_metrics["loss"],
                    "ppl": step_metrics["ppl"],
                    "lr": lr,
                    "tok_s": step_metrics["tokens_per_sec"],
                    "grad_norm": grad_norm,
                    "reg": step_metrics["reg"],
                    "elapsed": elapsed,
                }
                self.collector.log_full(
                    self.global_step, gate_stats, basic, extras=extras, mode="train"
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

        for step_idx in range(num_steps):
            try:
                batch = next(self.dataloader)
            except StopIteration:
                print(f"Dataloader exhausted at step {step_idx}")
                break

            step_metrics = self.train_chunk(batch)
            self.global_step += 1
            metrics.append(step_metrics)

            if self.global_step % self.log_interval == 0:
                m = step_metrics
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"step {m['step']:5d} | "
                    f"loss {m['loss']:.4f} | "
                    f"ppl {m['ppl']:.1f} | "
                    f"tok/s {m['tokens_per_sec']:.0f} | "
                    f"lr {lr:.2e}"
                )

            if step_callback is not None:
                step_callback(step_metrics)

        return metrics
