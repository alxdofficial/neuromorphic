"""
TBPTTTrainer — main training loop for neuromorphic LM.

Processes TBPTT chunks with plasticity spans. Handles doc boundary
resets, online loss accumulation, PM commits, and EM writes.
"""

import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterator, Optional, TYPE_CHECKING

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
        prev_token = batch.prev_token.to(self.device)

        chunk_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_count = 0
        span_surprise_accum = torch.zeros(input_ids.shape[0], device=self.device)
        span_tokens = 0

        # Determine if this step should do full collection
        do_full = (self.collector is not None
                   and self.collector.should_collect_full(self.global_step))
        gate_stats = None

        t_start = time.time()

        for span_start in range(0, T, P):
            span_end = min(span_start + P, T)
            is_last_span = (span_end >= T)

            # EM candidate buffers (per block)
            B = self.config.B
            cand_K = [[] for _ in range(B)]
            cand_V = [[] for _ in range(B)]
            cand_score = [[] for _ in range(B)]

            span_surprise_accum.zero_()
            span_tokens = 0

            for t in range(span_start, span_end):
                # Doc boundary reset mask
                if t == 0:
                    reset_mask = (prev_token == eot_id)
                else:
                    reset_mask = (input_ids[:, t - 1] == eot_id)

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

                # Loss masking: skip EOT input positions
                is_eot = (input_ids[:, t] == eot_id)
                if self.config.reset_on_doc_boundary:
                    loss_mask = ~is_eot
                else:
                    loss_mask = torch.ones_like(is_eot)

                # Accumulate loss
                token_loss, count = online_cross_entropy(
                    logits, target_ids[:, t], loss_mask
                )
                chunk_loss = chunk_loss + token_loss
                valid_count += count

                # Update surprise
                self.model.update_surprise(logits, target_ids[:, t])

                # Accumulate surprise for controller decisions
                if self.model.surprise is not None:
                    span_surprise_accum = span_surprise_accum + self.model.surprise.squeeze(-1)
                span_tokens += 1

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

            # Span boundary: PM base decay + commits
            if self.config.pm_enabled:
                # Base decay: all streams lose pm_a each span (FIX-3)
                for block in self.model.blocks:
                    for layer in block.layers:
                        layer.pm.base_decay()

                self.model.commit_at_boundary()

                # Record commit rates for collector
                if self.collector is not None:
                    for b_idx, block in enumerate(self.model.blocks):
                        for l_idx, layer in enumerate(block.layers):
                            pm = layer.pm
                            if pm.elig_K is not None:
                                elig_norm = pm.elig_K.norm(dim=-1).mean(dim=-1)
                                commit_mask, _, _, _ = layer.pm_controller.forward(
                                    elig_norm,
                                    pm.pm_a.sum(dim=-1),
                                    elig_norm,
                                )
                                self.collector.record_pm_commit(
                                    b_idx, l_idx, commit_mask
                                )

            # Span boundary: EM writes
            if self.config.em_enabled:
                span_surprise_mean = span_surprise_accum / max(span_tokens, 1)
                for b, block in enumerate(self.model.blocks):
                    if len(cand_K[b]) > 0:
                        stacked_K = torch.stack(cand_K[b], dim=1)      # [BS, P, D_em]
                        stacked_V = torch.stack(cand_V[b], dim=1)      # [BS, P, D_em]
                        stacked_score = torch.stack(cand_score[b], dim=1)  # [BS, P]

                        # Controller decides write mask and strength
                        em_usage = block.em.em_S.sum(dim=-1) if block.em.em_S is not None else torch.zeros_like(span_surprise_mean)
                        cand_novelty_mean = stacked_score.mean(dim=-1)

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
                            write_mask, g_em
                        )

        # Finalize loss
        if valid_count > 0:
            avg_loss = chunk_loss / valid_count
        else:
            avg_loss = chunk_loss

        # Add regularizers
        reg = compute_regularizers(self.model)
        total_loss = avg_loss + reg

        # Backward + clip + step
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        ).item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # TBPTT boundary: detach all states
        self.model.detach_states()

        elapsed = time.time() - t_start
        tokens = T * input_ids.shape[0]

        step_metrics = {
            "loss": avg_loss.item(),
            "reg": reg.item(),
            "ppl": min(torch.exp(avg_loss).item(), 1e6),
            "tokens_per_sec": tokens / max(elapsed, 1e-6),
            "valid_tokens": valid_count,
            "step": self.global_step,
            "grad_norm": grad_norm,
        }

        # Log to collector
        if self.collector is not None:
            lr = self.optimizer.param_groups[0]["lr"]
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
                self.collector.log_full(self.global_step, gate_stats, basic)
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
                )

        return step_metrics

    def train_epoch(self, num_steps: int) -> list:
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

        return metrics
