"""TBPTT scaffolding — chain D forward_window calls with cross-window
gradient flow.

Linear-in-D implementation: every window's autograd graph stays alive
until backward fires at the end of the chunk. With per-block Llama
checkpointing (already on inside the LM forward), memory cost is roughly
1 window of activations + a slim per-window write graph skeleton.

Constant-in-D (custom autograd streaming) is a v2 optimization — see
plan §4.3.

Usage:
    chunks = TBPTTChunker(cfg).split(input_ids)   # [num_chunks, D, T_window]
    prev_states, prev_hiddens, prev_lm_ctx = None, None, None
    for chunk in chunks:
        result = run_chunk(
            integrated_lm, chunk, prev_states, prev_hiddens, prev_lm_ctx,
        )
        loss = result["aggregate_loss"]
        loss.backward()
        prev_states = result["final_states"].detach()              # cut grad
        prev_hiddens = result["final_hiddens"].detach()
        prev_lm_ctx = result["final_lm_context"]                   # already detached
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM


class TBPTTChunker:
    """Splits a long sequence into D-window chunks for TBPTT.

    Given input_ids of shape [BS, total_T], where total_T = num_chunks *
    D * T_window, return iterable of [BS, D, T_window] chunks.

    For training, each chunk's backward fires before the next chunk's
    forward, with `final_states` and `final_hiddens` detached at the
    boundary.
    """

    def __init__(self, cfg: TrajMemConfig):
        self.cfg = cfg

    def split(self, input_ids: Tensor) -> list[Tensor]:
        cfg = self.cfg
        BS, T = input_ids.shape
        chunk_size = cfg.D * cfg.T_window
        if T % chunk_size != 0:
            # Drop the trailing partial chunk; data loader is expected to pad.
            usable = (T // chunk_size) * chunk_size
            input_ids = input_ids[:, :usable]
            T = usable
        num_chunks = T // chunk_size
        windows_per_chunk = cfg.D
        return [
            input_ids[
                :,
                c * chunk_size : (c + 1) * chunk_size,
            ].view(BS, windows_per_chunk, cfg.T_window)
            for c in range(num_chunks)
        ]


def run_chunk(
    model: IntegratedLM,
    windows: Tensor,
    prev_states: Tensor,
    prev_window_hiddens: Tensor | None,
    prev_lm_context: Tensor | None,
    *,
    target_mask: Tensor | None = None,
    hard_routing: bool = True,
) -> dict:
    """Run D consecutive windows with autograd kept alive across the chunk.

    Maintains a rolling LM-context buffer of up to `cfg.effective_lm_context`
    tokens. At each window, Llama sees the rolling buffer + the new window's
    tokens (truncated to the cap); we only compute surprise on the new
    window's positions (see plan §4.1).

    Args:
        model:               IntegratedLM
        windows:             [BS, D, T_window] token IDs
        prev_states:         [BS, N, D_concept] state at start of chunk
        prev_window_hiddens: [BS, T_window, d_lm] from window before this
                             chunk, or None if chunk starts a new sequence.
        prev_lm_context:     [BS, L] tokens from before this chunk, used as
                             rolling LM context (no gradient — Llama is
                             frozen and gradient cuts at chunk boundary
                             anyway). None at sequence start.
        target_mask:         [BS, D, T_window] or None — per-window mask
                             over the CURRENT window's tokens for surprise.
        hard_routing:        Gumbel-STE if True

    Returns:
        dict with keys:
            window_outputs:    list of D forward_window outputs
            aggregate_loss:    scalar loss summed over windows (NTP CE)
            final_states:      [BS, N, D_concept] state after last window
            final_hiddens:     [BS, T_window, d_lm] hiddens of last window
            final_lm_context:  [BS, L] rolling buffer for next chunk's
                               first-window LM context
            surprise_history:  [BS, D] surprise per window
    """
    BS, D, T = windows.shape
    assert D == model.cfg.D, f"chunk has D={D}, expected {model.cfg.D}"
    cfg = model.cfg
    cap = cfg.effective_lm_context

    states = prev_states
    cur_prev_hiddens = prev_window_hiddens
    # Rolling LM-context buffer. Tokens here are not in the autograd graph.
    if prev_lm_context is None:
        lm_buffer = torch.empty(BS, 0, dtype=windows.dtype, device=windows.device)
    else:
        lm_buffer = prev_lm_context.detach()                       # ensure no grad

    outputs: list[dict] = []
    losses: list[Tensor] = []
    surprises: list[Tensor] = []

    for d in range(D):
        win_input = windows[:, d, :]                               # [BS, T_window]
        win_mask = target_mask[:, d, :] if target_mask is not None else None

        # Build the full LM input: rolling buffer + current window, capped.
        lm_input_ids = torch.cat([lm_buffer, win_input], dim=1)
        if lm_input_ids.shape[1] > cap:
            lm_input_ids = lm_input_ids[:, -cap:]

        out = model.forward_window(
            lm_input_ids=lm_input_ids,
            prev_window_hiddens=cur_prev_hiddens,
            prev_states=states,
            target_mask=win_mask,
            hard_routing=hard_routing,
        )
        # Strip heavy tensors before appending — `outputs` is only used for
        # debug/test inspection downstream. `logits` is [BS, T_window, V]
        # (~500MB across a chunk at vocab=128K, BS=2, T_window=256).
        # `current_hiddens` is also redundant since we keep it as
        # `cur_prev_hiddens` and it surfaces as `final_hiddens`. Carry only
        # the small per-window metadata + the new_states tensor (needed by
        # tests).
        outputs.append({
            "new_states": out["new_states"],
            "read_visited": out["read_visited"],
            "write_visited": out["write_visited"],
            "surprise": out["surprise"],
        })
        losses.append(out["surprise"].sum())                       # NTP CE summed across batch
        surprises.append(out["surprise"])

        # Carry forward into next window.
        states = out["new_states"]
        cur_prev_hiddens = out["current_hiddens"]
        # Buffer for next window's LM input — at most (cap - T_window)
        # tokens, since each forward will append the next window of
        # T_window tokens before truncation.
        lm_buffer = lm_input_ids[:, -(cap - T):] if cap > T else lm_input_ids[:, :0]

    aggregate_loss = torch.stack(losses).sum()                     # scalar
    surprise_history = torch.stack(surprises, dim=1)               # [BS, D]

    return {
        "window_outputs": outputs,
        "aggregate_loss": aggregate_loss,
        "final_states": states,
        "final_hiddens": cur_prev_hiddens,
        "final_lm_context": lm_buffer,
        "surprise_history": surprise_history,
    }
