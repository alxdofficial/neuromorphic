"""Phase-1 parallel training harness for graph_walker + frozen Llama.

Per step:
1. `begin_segment(bs)` — zero walker state.
2. Full Llama forward with memory read/write at layer L (via MemInjectLayer
   closure). Walker processes the T-long segment sequentially inside
   `walk_segment`, with TBPTT detach every `tbptt_block` tokens.
3. Loss: Llama's next-token CE on `target_ids`. The walker has no aux loss
   of its own; it learns purely via gradient flowing back through `W_out`.
4. `loss.backward()`, grad-clip, `opt.step()`.
5. Compute Llama's per-token CE (`reduction='none'`) and feed it to
   `model.memory.update_plasticity(per_token_ce)`. This folds Llama's
   next-token surprise into the walker's `surprise_ema`, fires the
   structural plasticity step, and builds the next segment's neuromod
   delta. Plasticity does NOT fire inside `walk_segment`.
6. `model.detach_memory()` to release the autograd graph.

Gumbel-soft STE routing is the default in `routing.py`; phase-1 τ/ε
schedules come from `GraphWalkerConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph_walker.pretrained.integrated_lm import IntegratedLM


@dataclass
class Phase1Batch:
    """Phase-1 training batch.

    Convention (teacher-forced, NOT pre-shifted):
        input_ids[:, t]   — token fed to the LM at position t.
        target_ids[:, t]  — token at position t (same length as input_ids).
        The trainer internally shifts: logits[:, t] is CE'd against
        target_ids[:, t+1], so the last position of each row is
        unsupervised. If your dataloader yields already-shifted targets,
        DO NOT pass them as `target_ids` — pass them as `input_ids` of
        length T+1 instead, or shift them back.

    For the standard "predict next token from input" case, set
    `target_ids = input_ids` (the trainer does the shift internally).
    """
    input_ids: torch.Tensor        # [BS, T]
    target_ids: torch.Tensor       # [BS, T] — NOT pre-shifted


@dataclass
class Phase1Stats:
    loss: float
    ce_loss: float
    ce_p50: float                        # per-token CE percentiles (catches hard examples
    ce_p90: float                        # that the scalar mean would hide).
    ce_p99: float
    ce_max: float
    grad_norm: float
    inject_residual_ratio: float         # ||scale·W_out(readout)|| / ||hidden_states||
                                         # (real injection SNR; the previous
                                         # `inject_residual_norm` was a static
                                         # parameter product that didn't reflect
                                         # whether the readout was actually doing
                                         # anything).
    tok_per_sec: float
    load_balance_loss: float = 0.0       # walker load-balance aux (pre-weight).
                                         # Watch this drift toward zero as
                                         # walkers spread; if it stays high or
                                         # grows over training, walkers are
                                         # collapsing onto a small col subset.


def phase1_pretrained_step(
    model: IntegratedLM,
    opt: torch.optim.Optimizer,
    batch: Phase1Batch,
    *,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
) -> Phase1Stats:
    """Run one parallel teacher-forced phase-1 step.

    Loss = Llama primary next-token CE. After backward + opt.step, Llama's
    per-token CE is fed back into the walker as the surprise signal.
    """
    import time

    cfg = model.config
    device = next(model.parameters()).device
    input_ids = batch.input_ids.to(device)
    target_ids = batch.target_ids.to(device)

    # Phase-1 must run in train mode. If model got switched to inference
    # mode by a leaked rollout/bench pass, routing inside `_walker_step`
    # flips to deterministic argmax (no Gumbel STE), so the walker would
    # produce optimizer steps but no routing-side learning signal.
    if not model.training:
        raise RuntimeError(
            "phase1_pretrained_step requires model.train() — inference-mode "
            "leak would silently disable Gumbel-STE routing."
        )

    BS, T = input_ids.shape
    if target_ids.shape != (BS, T):
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} must match input_ids "
            f"shape {(BS, T)}. The trainer internally shifts target_ids[:, 1:] "
            "vs logits[:, :-1] — DO NOT pre-shift. See Phase1Batch docstring."
        )
    model.begin_segment(bs=BS)
    opt.zero_grad(set_to_none=True)

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if (amp_dtype is not None and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    t0 = time.perf_counter()
    with ctx:
        out = model(input_ids)
        logits = out.logits                                        # [BS, T, vocab]
        # Next-token CE: logits[:, t] predicts target_ids[:, t+1]; last
        # position of each row is unsupervised.
        logits_shift = logits[:, :-1]                              # [BS, T-1, V]
        targets_shift = target_ids[:, 1:]                          # [BS, T-1]
        ce = F.cross_entropy(
            logits_shift.reshape(-1, logits_shift.size(-1)).float(),
            targets_shift.reshape(-1),
            ignore_index=-100,
        )
        loss = cfg.ce_weight * ce

        # Anti-collapse load-balance auxiliary loss (Switch-Transformer
        # style; minimized when walkers + columns are uniformly visited).
        # Without this, walkers tend to pile onto a few columns over
        # training and the rest of the graph becomes dead weight —
        # directly undermines the "graph as concept space" thesis. The
        # walker accumulates the per-step load-balance loss during its
        # forward and we add the segment mean to the total loss here,
        # weighted by the configured `lambda_balance` (default 0.01).
        lb_loss_value: float = 0.0
        if model.memory is not None and model.memory.cfg.lambda_balance > 0.0:
            lb = model.memory.consume_load_balance_loss()
            if lb is not None:
                loss = loss + model.memory.cfg.lambda_balance * lb.float()
                lb_loss_value = float(lb.detach())

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in model.trainable_parameters()],
        grad_clip,
    )
    opt.step()

    # Per-token surprise for the walker. Recompute CE with reduction='none'
    # on the shifted logits — the autograd graph has already been consumed
    # by backward, but the value tensors are still alive in this scope.
    # `update_plasticity` folds this [BS, T-1] tensor into `surprise_ema`,
    # fires the plasticity step once, and builds the next segment's
    # neuromod delta. The last position of each row has no supervised
    # target so we feed only T-1 positions.
    if model.memory is not None:
        with torch.no_grad():
            # CE with ignore_index=-100 returns 0 at masked positions.
            # Zero is NOT the same as "no surprise" for the EMA — it
            # would be folded in as a strong signal of confidence at
            # positions the LM didn't actually predict. Pass the
            # explicit mask through update_plasticity so masked
            # positions are skipped entirely from the EMA update.
            per_token_ce = F.cross_entropy(
                logits_shift.reshape(-1, logits_shift.size(-1)).float(),
                targets_shift.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(BS, T - 1)
            valid_mask = (targets_shift != -100)
        model.memory.update_plasticity(per_token_ce, valid_mask=valid_mask)

    model.detach_memory()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    tok_per_sec = BS * T / elapsed

    # Real injection signal-to-noise ratio. MemInjectLayer.forward records
    # the actual ||inj|| and ||hidden|| in detached buffers per call.
    # The earlier scalar (mean(|scale|) * ||W_out.weight||) was a static
    # parameter product — it didn't depend on whether the readout was
    # actually emitting useful signal.
    with torch.no_grad():
        h_norm = float(model.mem_inject._last_hidden_norm.item())
        i_norm = float(model.mem_inject._last_inj_norm.item())
        inject_ratio = i_norm / max(h_norm, 1e-12)
        # Per-token CE percentiles — catches hard examples that the
        # scalar mean would smooth over.
        if model.memory is not None:
            pt = per_token_ce.detach().float().reshape(-1)
            ce_p50 = float(pt.quantile(0.5).item())
            ce_p90 = float(pt.quantile(0.9).item())
            ce_p99 = float(pt.quantile(0.99).item())
            ce_max = float(pt.max().item())
        else:
            ce_p50 = ce_p90 = ce_p99 = ce_max = 0.0

    return Phase1Stats(
        loss=float(loss.detach()),
        ce_loss=float(ce.detach()),
        ce_p50=ce_p50, ce_p90=ce_p90, ce_p99=ce_p99, ce_max=ce_max,
        grad_norm=float(grad_norm) if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
        inject_residual_ratio=inject_ratio,
        tok_per_sec=tok_per_sec,
        load_balance_loss=lb_loss_value,
    )
