"""Phase 1 (TF NTP) trainer for Wave 1 (long-doc) and Wave 2 (long-chat).

Per plan §4.7:
- Wave 1: standard TF NTP on long documents, surprise on all tokens.
- Wave 2: TF NTP on TurnPair (prior, response), surprise only on response.

Both use cross-window TBPTT (plan §4.2): each "training sequence" is a
chunk of `D * T_window` tokens; backward fires per chunk; manifold state
detached at chunk boundary.

This module exposes one trainer entry: `phase1_step(model, batch, ...)`
which runs forward + backward + optimizer step for one training step.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.tbptt import run_chunk
from src.trajectory_memory.training.loaders import TurnPairBatch


def phase1_wave1_step(
    model: IntegratedLM,
    chunk: Tensor,                  # [BS, D*T_window]
    *,
    optimizer: Optimizer,
    prev_states: Tensor | None = None,
    prev_window_hiddens: Tensor | None = None,
    prev_lm_context: Tensor | None = None,
) -> dict:
    """One Phase 1 / Wave 1 training step (long-doc TF NTP).

    Args:
        model:               IntegratedLM with attached Llama.
        chunk:               [BS, D*T_window] token IDs.
        optimizer:           torch.optim.Optimizer over trainable params.
        prev_states:         carry from previous chunk (None at sequence start).
        prev_window_hiddens: carry from previous chunk's last window.
        prev_lm_context:     carry from previous chunk's LM context buffer.

    Returns:
        dict with: loss (float), surprise_history (Tensor [BS, D]),
                   final_states / final_hiddens / final_lm_context (carries).
    """
    cfg = model.cfg
    BS, T_total = chunk.shape
    assert T_total == cfg.D * cfg.T_window, (
        f"chunk length {T_total} != D*T_window {cfg.D * cfg.T_window}"
    )

    if prev_states is None:
        prev_states = model.manifold.reset_states(batch_size=BS)

    windows = chunk.view(BS, cfg.D, cfg.T_window)

    optimizer.zero_grad()
    out = run_chunk(
        model,
        windows,
        prev_states=prev_states,
        prev_window_hiddens=prev_window_hiddens,
        prev_lm_context=prev_lm_context,
        target_mask=None,                        # all tokens are NTP targets in Wave 1
        hard_routing=True,
    )
    loss = out["aggregate_loss"]
    loss.backward()
    optimizer.step()

    # Detach the carries for the next chunk (TBPTT boundary).
    return {
        "loss": float(loss.detach()),
        "surprise_history": out["surprise_history"].detach(),
        "final_states": out["final_states"].detach(),
        "final_hiddens": out["final_hiddens"].detach(),
        "final_lm_context": out["final_lm_context"],   # already detached internally
    }


def phase1_wave2_step(
    model: IntegratedLM,
    batch: TurnPairBatch,
    *,
    optimizer: Optimizer,
) -> dict:
    """One Phase 1 / Wave 2 training step (long-chat TurnPair TF NTP).

    Each TurnPair example is processed independently — the manifold is
    reset between examples (plan §4.8 "TurnPair flattening"). Within
    one example, the prior + response are concatenated into a single
    sequence and run through TBPTT chunks, with the assistant-token mask
    applied for surprise.

    Args:
        model:     IntegratedLM with attached Llama.
        batch:     TurnPairBatch (length-bucketed).
        optimizer: torch.optim.Optimizer.

    Returns:
        dict with loss + per-example surprise history.
    """
    cfg = model.cfg
    BS = batch.prior_ids.shape[0]

    # Concatenate prior + response into one sequence per example.
    # Handle variable lengths: we already have padded prior/response, so
    # cat them along dim=1.
    full_ids = torch.cat([batch.prior_ids, batch.response_ids], dim=1)  # [BS, T_full]
    full_mask = torch.cat(
        [torch.zeros_like(batch.prior_mask), batch.response_mask], dim=1,
    )                                                                    # [BS, T_full] — True only on response

    T_full = full_ids.shape[1]
    chunk_len = cfg.D * cfg.T_window

    # Pad to multiple of chunk_len.
    if T_full % chunk_len != 0:
        pad_n = chunk_len - (T_full % chunk_len)
        pad_ids = torch.full((BS, pad_n), 0, dtype=full_ids.dtype, device=full_ids.device)
        pad_mask = torch.zeros((BS, pad_n), dtype=torch.bool, device=full_ids.device)
        full_ids = torch.cat([full_ids, pad_ids], dim=1)
        full_mask = torch.cat([full_mask, pad_mask], dim=1)
        T_full = full_ids.shape[1]
    n_chunks = T_full // chunk_len

    # Reset manifold per example (per-TurnPair reset, plan §4.8).
    prev_states = model.manifold.reset_states(batch_size=BS)
    prev_window_hiddens: Tensor | None = None
    prev_lm_context: Tensor | None = None

    optimizer.zero_grad()
    total_loss = torch.zeros((), device=full_ids.device)
    all_surprise = []

    for c in range(n_chunks):
        chunk = full_ids[:, c * chunk_len : (c + 1) * chunk_len]
        chunk_mask = full_mask[:, c * chunk_len : (c + 1) * chunk_len]
        windows = chunk.view(BS, cfg.D, cfg.T_window)
        win_mask = chunk_mask.view(BS, cfg.D, cfg.T_window)

        out = run_chunk(
            model, windows, prev_states=prev_states,
            prev_window_hiddens=prev_window_hiddens,
            prev_lm_context=prev_lm_context,
            target_mask=win_mask,
            hard_routing=True,
        )
        total_loss = total_loss + out["aggregate_loss"]
        all_surprise.append(out["surprise_history"])

        # Detach for cross-chunk TBPTT boundary.
        prev_states = out["final_states"].detach()
        prev_window_hiddens = out["final_hiddens"].detach()
        prev_lm_context = out["final_lm_context"]

    total_loss.backward()
    optimizer.step()

    return {
        "loss": float(total_loss.detach()),
        "surprise_history": torch.stack(all_surprise, dim=1).detach() if all_surprise else None,
    }
