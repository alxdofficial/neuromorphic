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
    for chunk in chunks:
        result = run_chunk(integrated_lm, chunk, prev_states, prev_hiddens)
        loss = result["aggregate_loss"]
        loss.backward()
        prev_states = result["final_states"].detach()  # cut grad
        prev_hiddens = result["final_hiddens"].detach()
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
    *,
    target_mask: Tensor | None = None,
    hard_routing: bool = True,
) -> dict:
    """Run D consecutive windows with autograd kept alive across the chunk.

    Args:
        model:               IntegratedLM
        windows:             [BS, D, T_window] token IDs
        prev_states:         [BS, N, D_concept] state at start of chunk
        prev_window_hiddens: [BS, T_window, d_lm] from window before this
                             chunk, or None if chunk starts a new sequence.
        target_mask:         [BS, D, T_window] or None
        hard_routing:        Gumbel-STE if True

    Returns:
        dict with keys:
            window_outputs:   list of D forward_window outputs
            aggregate_loss:   scalar loss summed over windows (NTP CE)
            final_states:     [BS, N, D_concept] state after last window
            final_hiddens:    [BS, T_window, d_lm] hiddens of last window
            surprise_history: [BS, D] surprise per window
    """
    BS, D, T = windows.shape
    assert D == model.cfg.D, f"chunk has D={D}, expected {model.cfg.D}"

    states = prev_states
    cur_prev_hiddens = prev_window_hiddens

    outputs: list[dict] = []
    losses: list[Tensor] = []
    surprises: list[Tensor] = []

    for d in range(D):
        win_input = windows[:, d, :]                              # [BS, T_window]
        win_mask = target_mask[:, d, :] if target_mask is not None else None
        out = model.forward_window(
            input_ids=win_input,
            prev_window_hiddens=cur_prev_hiddens,
            prev_states=states,
            target_mask=win_mask,
            hard_routing=hard_routing,
        )
        outputs.append(out)
        losses.append(out["surprise"].sum())                      # NTP CE summed across batch
        surprises.append(out["surprise"])

        # Carry forward into next window.
        states = out["new_states"]
        cur_prev_hiddens = out["current_hiddens"]

    aggregate_loss = torch.stack(losses).sum()                    # scalar
    surprise_history = torch.stack(surprises, dim=1)              # [BS, D]

    return {
        "window_outputs": outputs,
        "aggregate_loss": aggregate_loss,
        "final_states": states,
        "final_hiddens": cur_prev_hiddens,
        "surprise_history": surprise_history,
    }
