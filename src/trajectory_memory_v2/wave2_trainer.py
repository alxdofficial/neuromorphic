"""Wave2TrainerV2 — chat SFT trainer for v2 architecture (streaming mode).

Streaming mode: read at window d+1 conditions on prev_window_hiddens from
window d. Same passage hiddens are used by both read (at d+1) and write
(at d), so contrastive losses can be off (alignment by construction).

Protocol:
  For each batch of (prior + response) TurnPairs:
    - Split prior into K windows of T_window tokens
    - For each window: forward_window(write_mode="passage"), passing
      prev_window_hiddens from the previous window (None at window 0).
      No loss on these windows.
    - Process the response window: same forward, but compute NTP loss on
      response tokens.

Loss = answer_loss + load_balance + z_loss
  (No contrastive losses — streaming mode aligns reads/writes by construction.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2


@dataclass
class V2Wave2Metrics:
    loss: float = 0.0
    answer_loss: float = 0.0
    answer_token_count: int = 0
    aux_load_balance: float = 0.0
    aux_z_loss: float = 0.0
    grad_norm: float = 0.0
    n_active_edges: int = 0
    edge_active_fraction: float = 0.0
    mean_fan_out: float = 0.0
    mean_edge_state_norm: float = 0.0
    mean_edge_specificity: float = 0.0
    mean_visit_count: float = 0.0
    mean_edge_age: float = 0.0


class Wave2TrainerV2:
    """SFT trainer for chat data, streaming mode."""

    def __init__(
        self,
        model: IntegratedLMV2,
        optimizer: Optimizer,
        *,
        pad_token_id: int,
        scheduler: Any | None = None,
        grad_clip: float | None = 1.0,
        load_balance_coef: float = 1e-4,
        z_loss_coef: float = 1e-4,
        max_prior_windows: int = 4,  # cap prior context (saves memory)
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.pad_token_id = pad_token_id
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        self.max_prior_windows = max_prior_windows
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

    def _forward(self, batch, *, train: bool):
        """Shared forward used by both `step` and `eval_step`."""
        cfg = self.model.cfg
        device = next(self.model.parameters()).device
        T = cfg.T_window
        BS = batch.prior_ids.shape[0]

        prior_ids = batch.prior_ids.to(device)
        response_ids = batch.response_ids.to(device)
        prior_mask = batch.prior_mask.to(device)

        # NOTE: zero_grad + manifold.advance_step belong in `step()`, not
        # here — _forward is shared by step() and eval_step() and must be
        # pure computation. Calling them here would silently re-zero
        # gradients during eval.
        aux_lb_acc = torch.zeros((), device=device)
        aux_z_acc = torch.zeros((), device=device)
        n_walker_calls = 0

        T_prior = prior_ids.shape[1]
        max_prior_len = self.max_prior_windows * T
        if T_prior > max_prior_len:
            prior_ids = prior_ids[:, -max_prior_len:]
            prior_mask = prior_mask[:, -max_prior_len:]
            T_prior = max_prior_len

        prev_hiddens: Optional[Tensor] = None
        prev_mask: Optional[Tensor] = None
        n_prior_windows = max(1, (T_prior + T - 1) // T)
        for w in range(n_prior_windows):
            start = w * T
            end = min(start + T, T_prior)
            window_ids = prior_ids[:, start:end]
            window_mask_w = prior_mask[:, start:end]
            if window_ids.shape[1] < T:
                pad = torch.full(
                    (BS, T - window_ids.shape[1]),
                    self.pad_token_id, dtype=torch.long, device=device,
                )
                window_ids = torch.cat([window_ids, pad], dim=1)
                mask_pad = torch.zeros(BS, T - window_mask_w.shape[1], dtype=torch.bool, device=device)
                window_mask_w = torch.cat([window_mask_w, mask_pad], dim=1)
            out = self.model.forward_window(
                lm_input_ids=window_ids,
                prev_window_hiddens=prev_hiddens,
                attention_mask=window_mask_w,
                prev_attention_mask=prev_mask,
                hard_routing=True,
                write_mode="passage",
            )
            # Detach to prevent unbounded BPTT through the prior-window Llama
            # passes. prev_hiddens is only used as read conditioning at the
            # next window — read_module operates on it but gradient through
            # the prior-window Llama would unroll ~max_prior_windows×T tokens
            # of backprop through 1B params and OOM.
            prev_hiddens = out["current_hiddens"].detach()
            prev_mask = window_mask_w
            aux_lb_acc = aux_lb_acc + out["aux_load_balance"]
            aux_z_acc = aux_z_acc + out["aux_z_loss"]
            n_walker_calls += 1

        T_response = response_ids.shape[1]
        if T_response > T:
            response_window = response_ids[:, :T]
            response_mask_w = batch.response_mask.to(device)[:, :T]
        else:
            pad = torch.full(
                (BS, T - T_response),
                self.pad_token_id, dtype=torch.long, device=device,
            )
            response_window = torch.cat([response_ids, pad], dim=1)
            mask_pad = torch.zeros(BS, T - T_response, dtype=torch.bool, device=device)
            response_mask_w = torch.cat([batch.response_mask.to(device), mask_pad], dim=1)

        out_resp = self.model.forward_window(
            lm_input_ids=response_window,
            prev_window_hiddens=prev_hiddens,
            attention_mask=response_mask_w,
            prev_attention_mask=prev_mask,
            hard_routing=True,
            write_mode="passage",
        )
        aux_lb_acc = aux_lb_acc + out_resp["aux_load_balance"]
        aux_z_acc = aux_z_acc + out_resp["aux_z_loss"]
        n_walker_calls += 1

        aux_lb_acc = aux_lb_acc / max(n_walker_calls, 1)
        aux_z_acc = aux_z_acc / max(n_walker_calls, 1)

        logits = out_resp["logits"]
        V = logits.shape[-1]
        shift_logits = logits[:, :-1, :]
        shift_targets = response_window[:, 1:]
        shift_mask = response_mask_w[:, 1:]
        per_tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_targets.reshape(-1),
            reduction="none",
        ).reshape(BS, T - 1)
        n_answer = shift_mask.float().sum().clamp_min(1.0)
        answer_loss = (per_tok_ce * shift_mask.float()).sum() / n_answer

        total_loss = (
            answer_loss
            + self.load_balance_coef * aux_lb_acc
            + self.z_loss_coef * aux_z_acc
        )
        return total_loss, answer_loss, aux_lb_acc, aux_z_acc, n_answer

    def step(self, batch) -> V2Wave2Metrics:
        """One gradient update over a TurnPairBatch.

        batch.prior_ids:    [BS, T_prior]
        batch.response_ids: [BS, T_response]
        batch.prior_mask:   [BS, T_prior]  bool — True for real tokens (not pad)
        batch.response_mask: [BS, T_response] bool
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.manifold.advance_step()
        total_loss, answer_loss, aux_lb_acc, aux_z_acc, n_answer = self._forward(
            batch, train=True,
        )

        total_loss.backward()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._step_count += 1

        edge_stats = self.model.manifold.edge_stats()

        return V2Wave2Metrics(
            loss=float(total_loss.detach()),
            answer_loss=float(answer_loss.detach()),
            answer_token_count=int(n_answer.item()),
            aux_load_balance=float(aux_lb_acc.detach()),
            aux_z_loss=float(aux_z_acc.detach()),
            grad_norm=float(grad_norm.detach()) if isinstance(grad_norm, Tensor) else float(grad_norm),
            n_active_edges=edge_stats["n_active_edges"],
            edge_active_fraction=edge_stats["active_fraction"],
            mean_fan_out=edge_stats["mean_fan_out"],
            mean_edge_state_norm=edge_stats["mean_state_norm"],
            mean_edge_specificity=edge_stats["mean_specificity"],
            mean_visit_count=edge_stats["mean_visit_count"],
            mean_edge_age=edge_stats["mean_age"],
        )

    @torch.no_grad()
    def eval_step(self, batch) -> V2Wave2Metrics:
        """Forward-only validation. No backward, no optimizer.step, no manifold advance.

        Writes still happen inside forward_window (writes are gradient-free
        anyway), but EMA on val data is harmless — the manifold gets a slight
        contribution from val passages, which is fine for a streaming model.
        If you need a frozen-manifold val, snapshot+restore edge buffers
        around the call site.
        """
        self.model.eval()
        total_loss, answer_loss, aux_lb_acc, aux_z_acc, n_answer = self._forward(
            batch, train=False,
        )
        edge_stats = self.model.manifold.edge_stats()
        return V2Wave2Metrics(
            loss=float(total_loss.detach()),
            answer_loss=float(answer_loss.detach()),
            answer_token_count=int(n_answer.item()),
            aux_load_balance=float(aux_lb_acc.detach()),
            aux_z_loss=float(aux_z_acc.detach()),
            grad_norm=0.0,
            n_active_edges=edge_stats["n_active_edges"],
            edge_active_fraction=edge_stats["active_fraction"],
            mean_fan_out=edge_stats["mean_fan_out"],
            mean_edge_state_norm=edge_stats["mean_state_norm"],
            mean_edge_specificity=edge_stats["mean_specificity"],
            mean_visit_count=edge_stats["mean_visit_count"],
            mean_edge_age=edge_stats["mean_age"],
        )
