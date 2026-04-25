"""Phase-1 parallel training harness for graph_walker + frozen Llama.

Per step:
1. `reset_memory(bs)` — zero walker state.
2. Full Llama forward with memory read/write at layer L (via MemInjectLayer
   closure). Walker processes the T-long segment sequentially inside
   `forward_segment`, with TBPTT detach every `tbptt_block` tokens.
3. Primary loss: Llama's next-token CE on `target_ids`.
   Aux losses: walker's multi-horizon CE + adapter-side horizon-1 CE
   (both stashed on `wrapper._last_mem_loss`).
4. `loss = ce_weight · CE + aux_weight · aux`
5. Backward, grad-clip, opt.step, detach memory.

Gumbel-soft STE routing is already the default in `routing.py`; phase-1
τ/ε schedules come from `GraphWalkerConfig` (not per-step-step).
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
    aux_loss: float
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

    CE is computed over all T-1 shifted positions (teacher forced). Aux
    loss from memory is combined per `wrapper.config`'s weight knobs.
    """
    import time

    cfg = wrapper.config
    device = next(wrapper.parameters()).device
    input_ids = batch.input_ids.to(device)
    target_ids = batch.target_ids.to(device)

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
        # Next-token CE: shift targets left by 1; last position is unsupervised.
        # logits[:, t] predicts target_ids[:, t+1] equivalent → we treat
        # the caller-provided target_ids as "what to predict at each position".
        # Standard teacher-forced convention: targets are inputs shifted.
        # We accept whatever the caller passed and just CE against it,
        # masking the last position if lengths differ.
        logits_flat = logits[:, :-1].reshape(-1, logits.size(-1))
        targets_flat = target_ids[:, 1:].reshape(-1)
        ce = F.cross_entropy(logits_flat.float(), targets_flat)

        aux = wrapper._last_mem_loss
        if aux is not None:
            loss = cfg.ce_weight * ce + cfg.lm_aux_weight * aux.float()
        else:
            loss = cfg.ce_weight * ce

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in wrapper.trainable_parameters()],
        grad_clip,
    )
    opt.step()
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
        aux_loss=float(aux.detach()) if aux is not None else 0.0,
        grad_norm=float(grad_norm) if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
        inject_residual_norm=inj_norm,
        tok_per_sec=tok_per_sec,
    )
