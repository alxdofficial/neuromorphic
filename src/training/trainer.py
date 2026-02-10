"""
TBPTTTrainer — main training loop for neuromorphic LM.

Processes TBPTT chunks with plasticity spans. Handles doc boundary
resets, online loss accumulation, PM commits, and EM writes.
"""

import copy
import math
import time
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterator, Optional, Callable, TYPE_CHECKING

from ..data.streaming import StreamBatch
from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from ..model.state import save_runtime_state, load_runtime_state
from .loss import online_cross_entropy, batched_cross_entropy, compute_regularizers
from ..model.utils import unit_normalize

if TYPE_CHECKING:
    from ..debug.collector import MetricsCollector


@dataclass
class BoundarySnapshot:
    """State at a span boundary for counterfactual rollouts.

    Captured BEFORE PM commit + EM write so the rollout can apply
    force_on/off and then measure its effect on the next span.
    """
    runtime_state: dict
    span_start: int              # position in chunk where rollout's NEXT span starts
    input_ids: Tensor            # full chunk input (reference, not copy)
    target_ids: Tensor           # full chunk targets (reference, not copy)
    span_surprise_mean: Tensor   # [BS] surprise for this span
    pm_elig_norms: dict = field(default_factory=dict)   # {(b_idx, l_idx): [BS]}
    pm_usages: dict = field(default_factory=dict)        # {(b_idx, l_idx): [BS]}
    em_novelties: dict = field(default_factory=dict)     # {b_idx: [BS]}
    em_usages: dict = field(default_factory=dict)         # {b_idx: [BS]}
    # EM candidate buffers for forced write in rollout
    em_cand_K: list = field(default_factory=list)         # [B] of [BS, P, D_em]
    em_cand_V: list = field(default_factory=list)         # [B] of [BS, P, D_em]
    em_cand_score: list = field(default_factory=list)     # [B] of [BS, P]
    em_cand_valid: list = field(default_factory=list)     # [B] of [BS, P]
    em_g_em_chosen: dict = field(default_factory=dict)    # {b_idx: [BS]}


def _detached_runtime_state(model) -> dict:
    """Save runtime state with all tensors detached+cloned for safe deepcopy."""
    state = save_runtime_state(model)
    detached = {}
    for path, sub in state.items():
        detached[path] = {}
        for name, val in sub.items():
            if val is not None and isinstance(val, Tensor):
                detached[path][name] = val.detach().clone()
            else:
                detached[path][name] = val
    return detached


def _select_rl_spans(num_spans: int, rl_events: int) -> list[int]:
    """Select evenly-spaced span indices for RL rollouts."""
    if rl_events <= 0 or num_spans <= 0:
        return []
    rl_events = min(rl_events, num_spans)
    step = num_spans / rl_events
    return [int(i * step + step / 2) for i in range(rl_events)]


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
        rl_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.rl_optimizer = rl_optimizer
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
        # RL optimizer warmup: when > 0, linearly ramp RL LR over this many
        # steps from 0 to config.rl_lr. Used after phase transitions where
        # neuromod params move to a fresh optimizer with cold Adam state.
        self._rl_warmup_steps: int = 0
        self._rl_warmup_step: int = 0
        self._rl_base_lr: float = config.rl_lr if config.rl_enabled else 0.0
        # Tracks last chunk's final tokens for checkpoint resume.
        # When set, overrides the dataloader's prev_token on the next batch
        # so that doc-boundary resets don't wipe restored memory state.
        self.override_prev_token: Optional[Tensor] = None

        # Cache RL param ids for excluding from main grad clip
        self._rl_param_ids: set = set()
        if config.rl_enabled:
            self._rl_param_ids = {id(p) for p in model.rl_parameters()}

        # bf16 mixed precision (spec §3): forward/backward in bf16,
        # optimizer + state tensors stay fp32. No GradScaler needed for bf16.
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

    def set_rl_warmup(self, warmup_steps: int):
        """Enable RL optimizer LR warmup for the next N steps."""
        self._rl_warmup_steps = warmup_steps
        self._rl_warmup_step = 0
        if self.rl_optimizer is not None and warmup_steps > 0:
            for pg in self.rl_optimizer.param_groups:
                pg["lr"] = 0.0

    def _tick_rl_warmup(self):
        """Advance RL warmup by one step, scaling LR linearly."""
        if self._rl_warmup_steps <= 0 or self.rl_optimizer is None:
            return
        self._rl_warmup_step += 1
        if self._rl_warmup_step >= self._rl_warmup_steps:
            # Warmup complete — restore full LR
            for pg in self.rl_optimizer.param_groups:
                pg["lr"] = self._rl_base_lr
            self._rl_warmup_steps = 0
        else:
            scale = self._rl_warmup_step / self._rl_warmup_steps
            for pg in self.rl_optimizer.param_groups:
                pg["lr"] = self._rl_base_lr * scale

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

        # RL rollout setup
        if self.config.rl_enabled:
            num_spans = T // P
            rl_span_indices = set(_select_rl_spans(num_spans, self.config.rl_events_per_chunk))
            snapshots = []
        else:
            rl_span_indices = set()
            snapshots = []

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

            span_P = span_end - span_start
            span_ids = input_ids[:, span_start:span_end]          # [BS, span_P]
            span_targets = target_ids[:, span_start:span_end]     # [BS, span_P]

            # Reset mask for first token of this span
            if span_start == 0:
                reset_first = (prev_token == eot_id)
            else:
                reset_first = (input_ids[:, span_start - 1] == eot_id)

            with amp_ctx:
                # --- Parallel forward pass ---
                logits_all, x_emb_all, y_wm_all = self.model.forward_span(
                    span_ids, reset_first
                )

                if self.fail_fast and not torch.isfinite(logits_all).all():
                    raise RuntimeError(
                        f"Non-finite logits at global step {self.global_step}, "
                        f"span [{span_start}, {span_end})."
                    )

                # Loss masking: skip EOT input positions [BS, span_P]
                is_eot_all = (span_ids == eot_id)
                if self.config.reset_on_doc_boundary:
                    loss_mask_all = ~is_eot_all
                else:
                    loss_mask_all = torch.ones_like(is_eot_all)

                # Batched loss
                span_loss, span_valid = batched_cross_entropy(
                    logits_all, span_targets, loss_mask_all
                )
                chunk_loss = chunk_loss + span_loss
                valid_count += span_valid

                # --- Post-forward: compute per-token surprise ---
                with torch.no_grad():
                    logp = F.log_softmax(logits_all, dim=-1)          # [BS, span_P, V]
                    token_surprise = -logp.gather(
                        -1, span_targets.unsqueeze(-1)
                    )  # [BS, span_P, 1]
                    # Mask out EOT positions
                    token_surprise = token_surprise * loss_mask_all.unsqueeze(-1).float()

                    # Note: model.surprise is updated below (after span_surprise_mean
                    # is computed) to use the span mean rather than a single noisy sample.

                # Compute reset masks for span (needed for accumulators)
                reset_mask_all = self.model._compute_reset_masks(span_ids, reset_first)
                if not self.config.reset_on_doc_boundary:
                    reset_mask_all = torch.zeros_like(reset_mask_all)

                # Track span_last_reset and accumulate surprise per-stream
                for t_local in range(span_P):
                    reset_t = reset_mask_all[:, t_local]
                    if reset_t.any() and self.config.reset_on_doc_boundary:
                        span_last_reset[reset_t] = t_local
                        span_surprise_accum[reset_t] = 0
                        span_valid_tokens[reset_t] = 0

                    lm = loss_mask_all[:, t_local]
                    span_surprise_accum = (
                        span_surprise_accum
                        + token_surprise[:, t_local, 0] * lm.float()
                    )
                    span_valid_tokens = span_valid_tokens + lm.float()

                eot_inputs += int(is_eot_all.sum().item())
                reset_events += int(reset_mask_all.sum().item())

                # --- Post-forward: PM eligibility accumulation ---
                if self.config.pm_enabled:
                    # Compute block inputs once (shared across all blocks)
                    x_proj_all = self.model.W_in(x_emb_all)  # [BS, span_P, D]
                    x_blocks_all = x_proj_all.view(
                        BS, span_P, self.config.B, self.config.D_h
                    )  # [BS, span_P, B, D_h]

                    for b, block in enumerate(self.model.blocks):
                        for layer in block.layers:
                            if layer._last_h_all is None:
                                continue

                            # Layer input: block input (layer 0) or previous
                            # layer's output (_last_h_all).
                            l_idx = layer.layer_idx
                            if l_idx == 0:
                                x_in = x_blocks_all[:, :, b]  # [BS, span_P, D_h]
                            else:
                                x_in = block.layers[l_idx - 1]._last_h_all

                            h_out = layer._last_h_all  # [BS, span_P, D_h]

                            # Batched eligibility: projections + affine scan
                            layer.pm.update_eligibility_batch(
                                x_in, h_out, token_surprise, reset_mask_all,
                            )

                # --- Post-forward: EM candidate proposal ---
                if self.config.em_enabled:
                    for b, block in enumerate(self.model.blocks):
                        h_final_all = block.layers[-1]._last_h_all
                        if h_final_all is not None:
                            k_c, v_c, nov = block.em.propose_candidate_batch(
                                x_emb_all, y_wm_all, h_final_all,
                                token_surprise,
                            )
                            # Store as stacked tensors directly
                            cand_K[b].append(k_c)         # [BS, span_P, D_em]
                            cand_V[b].append(v_c)          # [BS, span_P, D_em]
                            cand_score[b].append(nov)      # [BS, span_P]
                            cand_token_valid[b].append(loss_mask_all)  # [BS, span_P]

            # Per-stream surprise mean for this span (used by boundary controllers
            # AND as the frozen surprise for the next span's layer gates).
            span_surprise_mean = span_surprise_accum / span_valid_tokens.clamp(min=1)
            span_valid_mean_accum += float(span_valid_tokens.mean().item())
            span_count += 1

            # Update model surprise to span mean (frozen for next span's gates).
            # Using the mean is more stable than the last token's value, which
            # is a single noisy sample (and would be 0 if last token was EOT).
            self.model.surprise = span_surprise_mean.unsqueeze(-1)  # [BS, 1]

            # Pre-compute EM candidate stacks and novelty (needed for both
            # the RL snapshot and the actual EM write below).
            em_stacked = {}  # {b: (K, V, score, valid, novelty_mean)}
            if self.config.em_enabled:
                for b, block in enumerate(self.model.blocks):
                    if len(cand_K[b]) > 0:
                        # Each entry is [BS, span_P, ...] from batch proposal
                        sK = torch.cat(cand_K[b], dim=1)             # [BS, span_P, D_em]
                        sV = torch.cat(cand_V[b], dim=1)             # [BS, span_P, D_em]
                        sScore = torch.cat(cand_score[b], dim=1)     # [BS, span_P]
                        sTokValid = torch.cat(cand_token_valid[b], dim=1)  # [BS, span_P]
                        S = sScore.shape[1]
                        pos = torch.arange(S, device=self.device).unsqueeze(0)
                        sValid = (
                            pos >= span_last_reset.unsqueeze(1)
                        ) & sTokValid.bool()
                        cvf = sValid.float()
                        cc = cvf.sum(dim=-1).clamp(min=1)
                        novelty = (sScore * cvf).sum(dim=-1) / cc
                        em_stacked[b] = (sK, sV, sScore, sValid, novelty)

            # RL snapshot: capture state BEFORE PM commit + EM write so the
            # rollout can apply force_on/off and measure the effect.
            span_idx = span_start // P
            if self.config.rl_enabled and span_idx in rl_span_indices and span_end < T:
                # Build EM candidate tensors for the snapshot
                snap_cK = [None] * B
                snap_cV = [None] * B
                snap_cS = [None] * B
                snap_cValid = [None] * B
                snap_novelties = {}
                for b, tup in em_stacked.items():
                    sK, sV, sScore, sValid, novelty = tup
                    snap_cK[b] = sK.detach().clone()
                    snap_cV[b] = sV.detach().clone()
                    snap_cS[b] = sScore.detach().clone()
                    snap_cValid[b] = sValid.detach().clone()
                    snap_novelties[b] = novelty.detach().clone()

                snap_g_em = {}
                if self.config.rl_enabled:
                    for b in snap_novelties:
                        block = self.model.blocks[b]
                        em_usage = (
                            block.em.em_S.sum(dim=-1)
                            if block.em.em_S is not None
                            else torch.zeros(BS, device=self.device)
                        )
                        _, g_em_val, _ = block.em_neuromodulator.forward(
                            span_surprise_mean,
                            em_usage / self.config.budget_em,
                            snap_novelties[b],
                        )
                        snap_g_em[b] = g_em_val.detach().clone()

                snap = BoundarySnapshot(
                    runtime_state=_detached_runtime_state(self.model),
                    span_start=span_end,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    span_surprise_mean=span_surprise_mean.detach().clone(),
                    pm_elig_norms=self._collect_elig_norms(),
                    pm_usages=self._collect_pm_usages(),
                    em_novelties=snap_novelties,
                    em_usages=self._collect_em_usages(),
                    em_cand_K=snap_cK,
                    em_cand_V=snap_cV,
                    em_cand_score=snap_cS,
                    em_cand_valid=snap_cValid,
                    em_g_em_chosen=snap_g_em,
                )
                snapshots.append(snap)

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

            # Span boundary: EM writes (using pre-computed stacks)
            if self.config.em_enabled:
                for b, block in enumerate(self.model.blocks):
                    if b not in em_stacked:
                        continue
                    sK, sV, sScore, sValid, cand_novelty_mean = em_stacked[b]

                    em_usage = block.em.em_S.sum(dim=-1) if block.em.em_S is not None else torch.zeros_like(span_surprise_mean)

                    write_mask, g_em, _p_write = block.em_neuromodulator.forward(
                        span_surprise_mean,
                        em_usage / self.config.budget_em,
                        cand_novelty_mean,
                    )

                    # Record write rates for collector
                    if self.collector is not None:
                        self.collector.record_em_write(
                            b, write_mask,
                            cand_novelty_mean.mean().item(),
                            g_em_mean=g_em.mean().item(),
                        )

                    block.em.write_at_boundary(
                        sK, sV, sScore,
                        write_mask, g_em, cand_valid=sValid
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
        if self.rl_optimizer is not None:
            self.rl_optimizer.zero_grad()
        total_loss.backward()

        # Clip only main-model params; neuromodulator continuous-head grads
        # must survive for the RL optimizer to combine with gate grads.
        if self._rl_param_ids:
            main_params = [p for p in self.model.parameters()
                           if id(p) not in self._rl_param_ids]
        else:
            main_params = list(self.model.parameters())
        grad_norm = nn.utils.clip_grad_norm_(
            main_params, self.max_grad_norm
        ).item()

        if self.fail_fast and not math.isfinite(grad_norm):
            raise RuntimeError(
                f"Non-finite gradient norm at global step {self.global_step}."
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # RL: counterfactual rollouts ADD gate grads on top of continuous grads,
        # then rl_optimizer steps with the combined gradient.
        rl_metrics = {}
        if self.config.rl_enabled and snapshots:
            self._final_runtime_state = _detached_runtime_state(self.model)
            rl_metrics = self._rl_step(snapshots)
        elif self.rl_optimizer is not None:
            # No rollout events this chunk — still step for continuous grads.
            # Capture neuromod grad norms before zero_grad clears them.
            for b_idx, block in enumerate(self.model.blocks):
                nm_em = block.em_neuromodulator
                total_sq = 0.0
                for p in nm_em.parameters():
                    if p.grad is not None:
                        total_sq += p.grad.detach().norm().item() ** 2
                if total_sq > 0:
                    rl_metrics[f"gnorm_b{b_idx}_em_neuromod"] = math.sqrt(total_sq)
                for l_idx, layer in enumerate(block.layers):
                    nm_pm = layer.pm_neuromodulator
                    total_sq = 0.0
                    for p in nm_pm.parameters():
                        if p.grad is not None:
                            total_sq += p.grad.detach().norm().item() ** 2
                    if total_sq > 0:
                        rl_metrics[f"gnorm_b{b_idx}_l{l_idx}_pm_neuromod"] = math.sqrt(total_sq)
            self._tick_rl_warmup()
            self.rl_optimizer.step()
            self.rl_optimizer.zero_grad()

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
        if rl_metrics:
            step_metrics.update(rl_metrics)

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
            if rl_metrics:
                extras.update(rl_metrics)
            if do_full:
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
                    self.global_step, gate_stats or {}, basic,
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

    # ------------------------------------------------------------------
    # RL helpers: snapshot collection, rollouts, neuromodulator updates
    # ------------------------------------------------------------------

    def _collect_elig_norms(self) -> dict:
        """Collect eligibility norms for all PM instances."""
        norms = {}
        for b_idx, block in enumerate(self.model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                if pm.elig_K is not None:
                    norms[(b_idx, l_idx)] = pm.elig_K.detach().norm(dim=-1).mean(dim=-1)
        return norms

    def _collect_pm_usages(self) -> dict:
        """Collect normalized PM usage for all instances."""
        usages = {}
        for b_idx, block in enumerate(self.model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                if pm.pm_a is not None:
                    usages[(b_idx, l_idx)] = (
                        pm.pm_a.detach().sum(dim=-1) / self.config.budget_pm
                    )
        return usages

    def _collect_em_novelties(self) -> dict:
        """Placeholder — novelty is now pre-computed and stored directly in the
        snapshot. This method exists for backward compatibility only."""
        return {}

    def _collect_em_usages(self) -> dict:
        """Collect normalized EM usage for all blocks."""
        usages = {}
        for b_idx, block in enumerate(self.model.blocks):
            em = block.em
            if em.em_S is not None:
                usages[b_idx] = em.em_S.detach().sum(dim=-1) / self.config.budget_em
        return usages

    def _select_pm_targets(self, snap: "BoundarySnapshot") -> list[tuple[int, int]]:
        """Pick salient PM controllers for deconfounded rollouts."""
        if not snap.pm_elig_norms:
            return []
        k = int(getattr(self.config, "rl_pm_targets_per_event", 1))
        if k <= 0:
            return []
        ranked = sorted(
            snap.pm_elig_norms.items(),
            key=lambda kv: float(kv[1].mean().item()),
            reverse=True,
        )
        return [key for key, _ in ranked[:k]]

    def _select_em_targets(self, snap: "BoundarySnapshot") -> list[int]:
        """Pick salient EM controllers for deconfounded rollouts."""
        if not snap.em_novelties:
            return []
        k = int(getattr(self.config, "rl_em_targets_per_event", 1))
        if k <= 0:
            return []
        ranked = sorted(
            snap.em_novelties.items(),
            key=lambda kv: float(kv[1].mean().item()),
            reverse=True,
        )
        return [key for key, _ in ranked[:k]]

    def _commit_pm_rollout_boundary(
        self,
        span_surprise: Tensor,
        force_mode: str = "normal",
        target: Optional[tuple[int, int]] = None,
    ):
        """PM commit helper for rollouts.

        If `target` is set, only that PM instance uses force_on/force_off.
        All other PM instances run in normal policy mode.
        """
        for b_idx, block in enumerate(self.model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                if pm.elig_K is None:
                    continue

                local_force = force_mode
                if target is not None and (b_idx, l_idx) != target:
                    local_force = "normal"

                if local_force == "force_off":
                    continue

                elig_norm = pm.elig_K.norm(dim=-1).mean(dim=-1)
                pm_usage = pm.pm_a.sum(dim=-1)

                if local_force == "force_on":
                    bs = elig_norm.shape[0]
                    commit_mask = torch.ones(bs, dtype=torch.bool, device=elig_norm.device)
                    lambda_vals = torch.full((bs,), pm.decay, device=elig_norm.device)
                    g = torch.full((bs,), 0.5, device=elig_norm.device)
                    pm.commit(commit_mask, lambda_vals, g, None)
                    continue

                surprise_input = span_surprise if span_surprise is not None else elig_norm
                commit_mask, lambda_vals, g, slot_logits, _ = layer.pm_neuromodulator.forward(
                    elig_norm,
                    pm_usage / self.config.budget_pm,
                    surprise_input,
                )
                pm.commit(commit_mask, lambda_vals, g, slot_logits)

    def _rollout_span(self, snap: "BoundarySnapshot",
                      pm_force: str, em_force: str,
                      pm_target: Optional[tuple[int, int]] = None,
                      em_target: Optional[int] = None) -> Tensor:
        """Run counterfactual rollout: apply forced commit/write, then measure
        loss over the next span of tokens.

        The snapshot was taken BEFORE the real commit/write. This method:
        1. Applies forced PM commit + EM write (using snapshot candidate buffers)
        2. Runs the next span's tokens forward-only
        3. Returns per-stream loss [BS]
        """
        P = self.config.P
        input_ids = snap.input_ids
        target_ids = snap.target_ids
        span_start = snap.span_start
        span_end = min(span_start + P, input_ids.shape[1])
        BS = input_ids.shape[0]
        eot_id = self.config.eot_id

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

        with torch.no_grad(), amp_ctx:
            # Step 1: Apply forced PM commit + EM write BEFORE measuring tokens.
            # This is the operation being counterfactually tested.
            if self.config.pm_enabled:
                for block in self.model.blocks:
                    for layer in block.layers:
                        layer.pm.base_decay()
                self._commit_pm_rollout_boundary(
                    span_surprise=snap.span_surprise_mean,
                    force_mode=pm_force,
                    target=pm_target,
                )

            if self.config.em_enabled:
                for b, block in enumerate(self.model.blocks):
                    sK = snap.em_cand_K[b]
                    if sK is None:
                        continue
                    sV = snap.em_cand_V[b]
                    sScore = snap.em_cand_score[b]
                    sValid = snap.em_cand_valid[b]

                    local_mode = em_force
                    if em_target is not None and b != em_target:
                        local_mode = "normal"

                    default_g = getattr(block.em_neuromodulator, "default_g", 0.3)
                    if local_mode == "baseline":
                        write_mask = torch.ones(BS, dtype=torch.bool, device=self.device)
                        g_em = torch.full((BS,), default_g, device=self.device)
                    elif local_mode == "chosen":
                        write_mask = torch.ones(BS, dtype=torch.bool, device=self.device)
                        g_em = snap.em_g_em_chosen.get(
                            b, torch.full((BS,), default_g, device=self.device)
                        )
                    else:  # "normal"
                        em_usage = (
                            block.em.em_S.sum(dim=-1)
                            if block.em.em_S is not None
                            else torch.zeros(BS, device=self.device)
                        )
                        cvf = sValid.float()
                        cc = cvf.sum(dim=-1).clamp(min=1)
                        novelty = (sScore * cvf).sum(dim=-1) / cc
                        write_mask, g_em, _ = block.em_neuromodulator.forward(
                            snap.span_surprise_mean,
                            em_usage / self.config.budget_em,
                            novelty,
                        )

                    block.em.write_at_boundary(
                        sK, sV, sScore,
                        write_mask, g_em, cand_valid=sValid,
                    )

            # Step 2: Run the next span's tokens and measure loss.
            stream_loss = torch.zeros(BS, device=self.device)
            stream_count = torch.zeros(BS, device=self.device)

            for t in range(span_start, span_end):
                if t == span_start:
                    if span_start == 0:
                        reset_mask = torch.zeros(BS, dtype=torch.bool, device=self.device)
                    else:
                        reset_mask = (input_ids[:, span_start - 1] == eot_id)
                else:
                    reset_mask = (input_ids[:, t - 1] == eot_id)

                logits, x_emb, y_wm = self.model.forward_one_token(
                    input_ids[:, t], reset_mask
                )
                is_eot = (input_ids[:, t] == eot_id)
                loss_mask = ~is_eot if self.config.reset_on_doc_boundary else torch.ones_like(is_eot)
                self.model.update_surprise(logits, target_ids[:, t], mask=loss_mask)

                per_token = F.cross_entropy(logits, target_ids[:, t], reduction="none")
                stream_loss = stream_loss + per_token * loss_mask.float()
                stream_count = stream_count + loss_mask.float()

        return stream_loss / stream_count.clamp(min=1)

    def _update_pm_neuromodulators(self, snap: BoundarySnapshot, reward: Tensor,
                                   targets: Optional[set[tuple[int, int]]] = None) -> float:
        """Weighted BCE update for each PM neuromodulator's gate head.

        Returns total gate loss (float) for metrics.
        """
        total_loss_val = 0.0
        for (b_idx, l_idx), elig_norm in snap.pm_elig_norms.items():
            if targets is not None and (b_idx, l_idx) not in targets:
                continue
            neuromod = self.model.blocks[b_idx].layers[l_idx].pm_neuromodulator
            if not neuromod.rl_enabled:
                continue
            pm_usage_norm = snap.pm_usages.get((b_idx, l_idx))
            if pm_usage_norm is None:
                continue

            _, _, _, _, p_commit = neuromod(
                elig_norm, pm_usage_norm, snap.span_surprise_mean,
            )

            label = (reward > 0).float()
            credit = elig_norm / elig_norm.clamp(min=1e-6).max()
            weight = reward.abs() * credit

            loss = F.binary_cross_entropy(p_commit, label, weight=weight)
            total_loss_val += loss.item()
            loss.backward()
        return total_loss_val

    def _update_em_neuromodulators(self, snap: BoundarySnapshot, reward: Tensor,
                                   targets: Optional[set[int]] = None) -> float:
        """Continuous EM objective for g_em using deconfounded rewards.

        Positive reward -> move toward chosen g_em.
        Negative reward -> move toward baseline g_em.
        """
        total_loss_val = 0.0
        for b_idx in range(self.config.B):
            if targets is not None and b_idx not in targets:
                continue
            neuromod = self.model.blocks[b_idx].em_neuromodulator
            if not neuromod.rl_enabled:
                continue
            em_usage_norm = snap.em_usages.get(b_idx)
            if em_usage_norm is None:
                continue
            novelty = snap.em_novelties.get(b_idx)
            if novelty is None:
                continue
            chosen_g = snap.em_g_em_chosen.get(b_idx)
            if chosen_g is None:
                continue

            # Re-forward to get g_em with grad
            _, g_em, _ = neuromod(
                snap.span_surprise_mean, em_usage_norm, novelty,
            )

            baseline_g = torch.full_like(chosen_g, neuromod.default_g)
            target_g = torch.where(reward > 0, chosen_g, baseline_g)

            scale = max(neuromod.g_em_ceil - neuromod.g_em_floor, 1e-6)
            g_em_normalized = ((g_em - neuromod.g_em_floor) / scale).clamp(0.0, 1.0)
            target_normalized = ((target_g - neuromod.g_em_floor) / scale).clamp(0.0, 1.0)

            credit = novelty / novelty.clamp(min=1e-6).max()
            weight = reward.abs() * credit

            sq_err = (g_em_normalized - target_normalized).pow(2)
            loss = (sq_err * weight).sum() / weight.sum().clamp(min=1e-6)
            total_loss_val += loss.item()
            loss.backward()
        return total_loss_val

    def _rl_step(self, snapshots: list) -> dict:
        """Run counterfactual rollouts and update neuromodulators."""
        if not snapshots:
            return {}

        rl_metrics = {
            "rl_pm_reward_mean": 0.0,
            "rl_em_reward_mean": 0.0,
            "rl_pm_gate_loss": 0.0,
            "rl_em_g_loss": 0.0,
            "rl_events": len(snapshots),
        }
        pm_reward_sum = 0.0
        em_reward_sum = 0.0
        pm_gate_loss_sum = 0.0
        em_gate_loss_sum = 0.0
        pm_count = 0
        em_count = 0

        for snap in snapshots:
            # -- PM rollout --
            if self.config.pm_enabled and self.config.rl_enabled:
                pm_targets = self._select_pm_targets(snap)
                for target in pm_targets:
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_off = self._rollout_span(
                        snap, pm_force="force_off", em_force="normal",
                        pm_target=target,
                    )
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_on = self._rollout_span(
                        snap, pm_force="force_on", em_force="normal",
                        pm_target=target,
                    )
                    reward = loss_off - loss_on
                    pm_reward_sum += reward.mean().item()
                    pm_count += 1
                    pm_gate_loss_sum += self._update_pm_neuromodulators(
                        snap, reward, targets={target}
                    )

            # -- EM rollout --
            if self.config.em_enabled and self.config.rl_enabled:
                em_targets = self._select_em_targets(snap)
                for b_idx in em_targets:
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_baseline = self._rollout_span(
                        snap, pm_force="normal", em_force="baseline",
                        em_target=b_idx,
                    )
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_chosen = self._rollout_span(
                        snap, pm_force="normal", em_force="chosen",
                        em_target=b_idx,
                    )
                    reward = loss_baseline - loss_chosen
                    em_reward_sum += reward.mean().item()
                    em_count += 1
                    em_gate_loss_sum += self._update_em_neuromodulators(
                        snap, reward, targets={b_idx}
                    )

        # Restore final real state
        load_runtime_state(self.model, self._final_runtime_state)

        # Capture neuromodulator grad norms BEFORE optimizer zeros them
        # (collector's _collect_grad_norms runs too late — after zero_grad)
        for b_idx, block in enumerate(self.model.blocks):
            nm_em = block.em_neuromodulator
            total_sq = 0.0
            for p in nm_em.parameters():
                if p.grad is not None:
                    total_sq += p.grad.detach().norm().item() ** 2
            if total_sq > 0:
                rl_metrics[f"gnorm_b{b_idx}_em_neuromod"] = math.sqrt(total_sq)
            for l_idx, layer in enumerate(block.layers):
                nm_pm = layer.pm_neuromodulator
                total_sq = 0.0
                for p in nm_pm.parameters():
                    if p.grad is not None:
                        total_sq += p.grad.detach().norm().item() ** 2
                if total_sq > 0:
                    rl_metrics[f"gnorm_b{b_idx}_l{l_idx}_pm_neuromod"] = math.sqrt(total_sq)

        # Single optimizer step for all neuromodulators
        if self.rl_optimizer is not None:
            self._tick_rl_warmup()
            self.rl_optimizer.step()
            self.rl_optimizer.zero_grad()

        # Collect RL-specific metrics
        if pm_count > 0:
            rl_metrics["rl_pm_reward_mean"] = pm_reward_sum / pm_count
            rl_metrics["rl_pm_gate_loss"] = pm_gate_loss_sum / pm_count
        if em_count > 0:
            rl_metrics["rl_em_reward_mean"] = em_reward_sum / em_count
            rl_metrics["rl_em_g_loss"] = em_gate_loss_sum / em_count

        # Neuromodulator output stats — use last snapshot's real inputs
        last_snap = snapshots[-1]
        commit_rates = []
        lambda_vals = []
        g_vals = []
        g_em_vals = []

        with torch.no_grad():
            for b_idx, block in enumerate(self.model.blocks):
                for l_idx, layer in enumerate(block.layers):
                    nm = layer.pm_neuromodulator
                    if nm.rl_enabled and hasattr(nm, "gate_head"):
                        elig_norm = last_snap.pm_elig_norms.get((b_idx, l_idx))
                        pm_usage = last_snap.pm_usages.get((b_idx, l_idx))
                        if elig_norm is not None and pm_usage is not None:
                            _, lv, gv, _, pc = nm(
                                elig_norm, pm_usage, last_snap.span_surprise_mean,
                            )
                            if pc is not None:
                                commit_rates.append((pc > 0.5).float().mean().item())
                                lambda_vals.append(lv.mean().item())
                                g_vals.append(gv.mean().item())

                nm_em = block.em_neuromodulator
                if nm_em.rl_enabled and hasattr(nm_em, "g_head"):
                    em_usage = last_snap.em_usages.get(b_idx)
                    em_nov = last_snap.em_novelties.get(b_idx)
                    if em_usage is not None and em_nov is not None:
                        _, gem, _ = nm_em(
                            last_snap.span_surprise_mean, em_usage, em_nov,
                        )
                        g_em_vals.append(gem.mean().item())

        if commit_rates:
            rl_metrics["rl_pm_commit_rate"] = sum(commit_rates) / len(commit_rates)
        if g_em_vals:
            mean_g = sum(g_em_vals) / len(g_em_vals)
            rl_metrics["rl_em_write_rate"] = mean_g  # effective write strength
            rl_metrics["rl_em_g_mean"] = mean_g
        if lambda_vals:
            rl_metrics["rl_pm_lambda_mean"] = sum(lambda_vals) / len(lambda_vals)
        if g_vals:
            rl_metrics["rl_pm_g_mean"] = sum(g_vals) / len(g_vals)

        return rl_metrics

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
