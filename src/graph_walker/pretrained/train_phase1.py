"""Phase-1 parallel training harness for graph_walker + frozen Llama.

Per step:
1. `reset_memory(bs)` — zero walker state.
2. Full Llama forward with memory read/write at layer L (via MemInjectLayer
   closure). Walker processes the T-long segment sequentially inside
   `forward_segment`, with TBPTT detach every `tbptt_block` tokens.
3. Loss: Llama's next-token CE on `target_ids`. The walker has no aux loss
   of its own; it learns purely via gradient flowing back through `W_out`.
4. `loss.backward()`, grad-clip, `opt.step()`.
5. Compute Llama's per-token CE (`reduction='none'`) and feed it to
   `wrapper.memory.update_plasticity(per_token_ce)`. This folds Llama's
   next-token surprise into the walker's `surprise_ema`, fires the
   structural plasticity step, and builds the next segment's neuromod
   delta. Plasticity does NOT fire inside `forward_segment`.
6. `wrapper.detach_memory()` to release the autograd graph.

Gumbel-soft STE routing is the default in `routing.py`; phase-1 τ/ε
schedules come from `GraphWalkerConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM


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
    prev_token: torch.Tensor | None = None    # reserved (not currently used)


@dataclass
class Phase1Stats:
    loss: float
    ce_loss: float
    grad_norm: float
    inject_residual_norm: float          # ||scale · W_out(readout)|| mean
    tok_per_sec: float


def phase1_pretrained_step(
    wrapper: GraphWalkerPretrainedLM,
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

    cfg = wrapper.config
    device = next(wrapper.parameters()).device
    input_ids = batch.input_ids.to(device)
    target_ids = batch.target_ids.to(device)

    # Phase-1 must run in train mode. If wrapper got switched to inference
    # mode by a leaked rollout/bench pass, routing inside `_step_core_pure`
    # flips to deterministic argmax (no Gumbel STE), so the walker would
    # produce optimizer steps but no routing-side learning signal.
    if not wrapper.training:
        raise RuntimeError(
            "phase1_pretrained_step requires wrapper.train() — inference-mode "
            "leak would silently disable Gumbel-STE routing."
        )

    BS, T = input_ids.shape
    if target_ids.shape != (BS, T):
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} must match input_ids "
            f"shape {(BS, T)}. The trainer internally shifts target_ids[:, 1:] "
            "vs logits[:, :-1] — DO NOT pre-shift. See Phase1Batch docstring."
        )
    wrapper.reset_memory(bs=BS)
    opt.zero_grad(set_to_none=True)

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if (amp_dtype is not None and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    t0 = time.perf_counter()
    with ctx:
        out = wrapper(input_ids)
        logits = out.logits                                        # [BS, T, vocab]
        # Next-token CE: logits[:, t] predicts target_ids[:, t+1]; last
        # position of each row is unsupervised.
        logits_shift = logits[:, :-1]                              # [BS, T-1, V]
        targets_shift = target_ids[:, 1:]                          # [BS, T-1]
        ce = F.cross_entropy(
            logits_shift.reshape(-1, logits_shift.size(-1)).float(),
            targets_shift.reshape(-1),
        )
        loss = cfg.ce_weight * ce

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in wrapper.trainable_parameters()],
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
    if wrapper.memory is not None:
        with torch.no_grad():
            per_token_ce = F.cross_entropy(
                logits_shift.reshape(-1, logits_shift.size(-1)).float(),
                targets_shift.reshape(-1),
                reduction="none",
            ).reshape(BS, T - 1)
        wrapper.memory.update_plasticity(per_token_ce)

    wrapper.detach_memory()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    tok_per_sec = BS * T / elapsed

    # Inject residual magnitude — useful diagnostic for "is memory
    # contributing anything". `scale` and `W_out.weight` are both on
    # wrapper.mem_inject.
    with torch.no_grad():
        inj_norm = (
            wrapper.mem_inject.scale.abs().mean()
            * wrapper.mem_inject.W_out.weight.norm()
        ).item()

    return Phase1Stats(
        loss=float(loss.detach()),
        ce_loss=float(ce.detach()),
        grad_norm=float(grad_norm) if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
        inject_residual_norm=inj_norm,
        tok_per_sec=tok_per_sec,
    )
