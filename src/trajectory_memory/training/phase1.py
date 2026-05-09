"""Phase 1 (TF NTP) trainer for Wave 1 (long-doc) and Wave 2 (long-chat).

Per plan §4.7:
- Wave 1: standard TF NTP on long documents, surprise on all tokens.
- Wave 2: TF NTP on TurnPair (prior, response), surprise only on response.

Both use cross-window TBPTT (plan §4.2): each "training sequence" is a
chunk of `D * T_window` tokens; backward fires per chunk; manifold state
detached at chunk boundary.

The `Phase1Trainer` class is the proper training harness:
- gradient clipping
- LR scheduler hookup
- per-step metrics dict (loss, grad_norm, lr)
- checkpoint integration via `state_dict()` / `load_state_dict()`

The legacy free functions `phase1_wave1_step` / `phase1_wave2_step` are
preserved for backward-compatibility with the old train_wave*.py scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.tbptt import run_chunk
from src.trajectory_memory.training.loaders import TurnPairBatch


@dataclass
class Phase1Metrics:
    """Per-step metrics returned by Phase1Trainer.step_*."""
    loss: float
    grad_norm: float
    lr: list[float]                # per param group
    surprise_history: Tensor | None = None
    final_states: Tensor | None = None
    final_hiddens: Tensor | None = None
    final_lm_context: Tensor | None = None


class Phase1Trainer:
    """TF NTP trainer for Wave 1 + Wave 2.

    Usage:
        trainer = Phase1Trainer(model, optimizer, scheduler=scheduler, grad_clip=1.0)
        for step in range(num_steps):
            metrics = trainer.step_wave1(chunk)        # Wave 1
            # OR
            metrics = trainer.step_wave2(batch)        # Wave 2
            log(metrics)
    """

    def __init__(
        self,
        model: IntegratedLM,
        optimizer: Optimizer,
        *,
        scheduler: object | None = None,    # WarmupCosineScheduler-like
        grad_clip: float | None = 1.0,
        pad_token_id: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        # pad_token_id is used in step_wave2 / eval_wave2 for chunk padding.
        # Default 0 (legacy) only for backward compat with toy tests; real
        # training MUST pass the tokenizer's pad_token_id (128001 for
        # Llama-3) so Llama doesn't see synthetic `!` tokens in its context.
        self.pad_token_id = pad_token_id
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        """Trainer state (for checkpointing). Does NOT include model /
        optimizer / scheduler state — caller saves those separately via
        `save_checkpoint`."""
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

    # ── Wave 1: long-doc TF NTP ───────────────────────────────────────

    def step_wave1(
        self,
        chunk: Tensor,                              # [BS, D*T_window]
        *,
        prev_states: Tensor | None = None,
        prev_window_hiddens: Tensor | None = None,
        prev_lm_context: Tensor | None = None,
    ) -> Phase1Metrics:
        """One Wave 1 step (long-doc TF NTP)."""
        cfg = self.model.cfg
        BS, T_total = chunk.shape
        assert T_total == cfg.D * cfg.T_window, (
            f"chunk length {T_total} != D*T_window {cfg.D * cfg.T_window}"
        )

        if prev_states is None:
            prev_states = self.model.manifold.reset_states(batch_size=BS)

        windows = chunk.view(BS, cfg.D, cfg.T_window)

        self.optimizer.zero_grad()
        out = run_chunk(
            self.model,
            windows,
            prev_states=prev_states,
            prev_window_hiddens=prev_window_hiddens,
            prev_lm_context=prev_lm_context,
            target_mask=None,
            hard_routing=True,
        )
        loss = out["aggregate_loss"]
        loss.backward()

        grad_norm = self._clip_and_step()
        self._step_count += 1

        return Phase1Metrics(
            loss=float(loss.detach()),
            grad_norm=float(grad_norm),
            lr=self._current_lrs(),
            surprise_history=out["surprise_history"].detach(),
            final_states=out["final_states"].detach(),
            final_hiddens=out["final_hiddens"].detach(),
            final_lm_context=out["final_lm_context"],
        )

    # ── Wave 2: long-chat TF NTP (TurnPair) ───────────────────────────

    def step_wave2(self, batch: TurnPairBatch) -> Phase1Metrics:
        """One Wave 2 step (long-chat TF NTP).

        Concatenates prior + response per example, chunks into TBPTT
        windows, applies surprise-on-response mask, detaches state at
        chunk boundary, accumulates loss across chunks, single backward
        per example.
        """
        cfg = self.model.cfg
        BS = batch.prior_ids.shape[0]
        device = batch.prior_ids.device
        pad_token = self.pad_token_id

        full_ids = torch.cat([batch.prior_ids, batch.response_ids], dim=1)
        full_mask = torch.cat(
            [torch.zeros_like(batch.prior_mask), batch.response_mask], dim=1,
        )
        T_full = full_ids.shape[1]
        chunk_len = cfg.D * cfg.T_window

        # Pad to multiple of chunk_len.
        if T_full % chunk_len != 0:
            pad_n = chunk_len - (T_full % chunk_len)
            full_ids = torch.cat([
                full_ids,
                torch.full((BS, pad_n), pad_token, dtype=full_ids.dtype, device=device),
            ], dim=1)
            full_mask = torch.cat([
                full_mask,
                torch.zeros((BS, pad_n), dtype=torch.bool, device=device),
            ], dim=1)
            T_full = full_ids.shape[1]
        n_chunks = T_full // chunk_len

        prev_states = self.model.manifold.reset_states(batch_size=BS)
        prev_window_hiddens: Tensor | None = None
        prev_lm_context: Tensor | None = None

        self.optimizer.zero_grad()
        total_loss = torch.zeros((), device=device)
        all_surprise: list[Tensor] = []

        for c in range(n_chunks):
            ids = full_ids[:, c * chunk_len : (c + 1) * chunk_len]
            mask = full_mask[:, c * chunk_len : (c + 1) * chunk_len]
            windows = ids.view(BS, cfg.D, cfg.T_window)
            win_mask = mask.view(BS, cfg.D, cfg.T_window)

            out = run_chunk(
                self.model, windows,
                prev_states=prev_states,
                prev_window_hiddens=prev_window_hiddens,
                prev_lm_context=prev_lm_context,
                target_mask=win_mask,
                hard_routing=True,
            )
            total_loss = total_loss + out["aggregate_loss"]
            all_surprise.append(out["surprise_history"])

            # Detach for cross-chunk TBPTT cut.
            prev_states = out["final_states"].detach()
            prev_window_hiddens = out["final_hiddens"].detach()
            prev_lm_context = out["final_lm_context"]

        total_loss.backward()
        grad_norm = self._clip_and_step()
        self._step_count += 1

        return Phase1Metrics(
            loss=float(total_loss.detach()),
            grad_norm=float(grad_norm),
            lr=self._current_lrs(),
            surprise_history=torch.stack(all_surprise, dim=1).detach() if all_surprise else None,
        )

    # ── Validation (no-grad) ──────────────────────────────────────────

    @torch.no_grad()
    def eval_wave1(self, chunk: Tensor) -> float:
        """Forward-only Wave 1 chunk; returns NTP loss. No grad, no opt step."""
        cfg = self.model.cfg
        BS, T_total = chunk.shape
        assert T_total == cfg.D * cfg.T_window
        prev_states = self.model.manifold.reset_states(batch_size=BS)
        windows = chunk.view(BS, cfg.D, cfg.T_window)
        out = run_chunk(
            self.model, windows,
            prev_states=prev_states,
            prev_window_hiddens=None,
            prev_lm_context=None,
            target_mask=None,
            hard_routing=True,
        )
        return float(out["aggregate_loss"].detach())

    @torch.no_grad()
    def eval_wave2(self, batch: TurnPairBatch) -> float:
        """Forward-only Wave 2 TurnPair; returns response-masked NTP loss."""
        cfg = self.model.cfg
        BS = batch.prior_ids.shape[0]
        device = batch.prior_ids.device
        pad_token = self.pad_token_id

        full_ids = torch.cat([batch.prior_ids, batch.response_ids], dim=1)
        full_mask = torch.cat(
            [torch.zeros_like(batch.prior_mask), batch.response_mask], dim=1,
        )
        T_full = full_ids.shape[1]
        chunk_len = cfg.D * cfg.T_window
        if T_full % chunk_len != 0:
            pad_n = chunk_len - (T_full % chunk_len)
            full_ids = torch.cat([
                full_ids,
                torch.full((BS, pad_n), pad_token, dtype=full_ids.dtype, device=device),
            ], dim=1)
            full_mask = torch.cat([
                full_mask,
                torch.zeros((BS, pad_n), dtype=torch.bool, device=device),
            ], dim=1)
            T_full = full_ids.shape[1]
        n_chunks = T_full // chunk_len

        prev_states = self.model.manifold.reset_states(batch_size=BS)
        prev_window_hiddens: Tensor | None = None
        prev_lm_context: Tensor | None = None
        total = torch.zeros((), device=device)
        for c in range(n_chunks):
            ids = full_ids[:, c * chunk_len : (c + 1) * chunk_len]
            mask = full_mask[:, c * chunk_len : (c + 1) * chunk_len]
            out = run_chunk(
                self.model, ids.view(BS, cfg.D, cfg.T_window),
                prev_states=prev_states,
                prev_window_hiddens=prev_window_hiddens,
                prev_lm_context=prev_lm_context,
                target_mask=mask.view(BS, cfg.D, cfg.T_window),
                hard_routing=True,
            )
            total = total + out["aggregate_loss"]
            prev_states = out["final_states"]
            prev_window_hiddens = out["final_hiddens"]
            prev_lm_context = out["final_lm_context"]
        return float(total.detach())

    # ── helpers ───────────────────────────────────────────────────────

    def _clip_and_step(self) -> float:
        """Clip gradients, step optimizer + scheduler, return grad_norm."""
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                max_norm=self.grad_clip,
            )
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(grad_norm)

    def _current_lrs(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


# ── Legacy free functions (kept for backward-compat with older callers) ──


def phase1_wave1_step(
    model: IntegratedLM,
    chunk: Tensor,
    *,
    optimizer: Optimizer,
    prev_states: Tensor | None = None,
    prev_window_hiddens: Tensor | None = None,
    prev_lm_context: Tensor | None = None,
    grad_clip: float | None = 1.0,
) -> dict:
    """Single-call Phase 1 / Wave 1 step. Prefer `Phase1Trainer.step_wave1`."""
    trainer = Phase1Trainer(model, optimizer, grad_clip=grad_clip)
    m = trainer.step_wave1(
        chunk,
        prev_states=prev_states,
        prev_window_hiddens=prev_window_hiddens,
        prev_lm_context=prev_lm_context,
    )
    return {
        "loss": m.loss,
        "grad_norm": m.grad_norm,
        "surprise_history": m.surprise_history,
        "final_states": m.final_states,
        "final_hiddens": m.final_hiddens,
        "final_lm_context": m.final_lm_context,
    }


def phase1_wave2_step(
    model: IntegratedLM,
    batch: TurnPairBatch,
    *,
    optimizer: Optimizer,
    grad_clip: float | None = 1.0,
) -> dict:
    """Single-call Phase 1 / Wave 2 step. Prefer `Phase1Trainer.step_wave2`."""
    trainer = Phase1Trainer(model, optimizer, grad_clip=grad_clip)
    m = trainer.step_wave2(batch)
    return {"loss": m.loss, "grad_norm": m.grad_norm,
            "surprise_history": m.surprise_history}
