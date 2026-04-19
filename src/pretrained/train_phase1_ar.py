"""Phase 1 — autoregressive Gumbel-softmax backprop unroll.

The base `run_phase1` trains memory via parallel teacher-forced NTP: all
T positions predict in one forward, CE on every position. That works but
asks memory only to improve predictions where the LM already sees a
ground-truth prefix — a frozen-coherent-Llama setting where memory's
marginal contribution is small and its long-horizon utility never gets
exercised.

This variant trains memory on a more directly useful signal: "given the
writes you made during the prefix, how well can the memory-augmented LM
predict the next T_cont tokens, one at a time, with each prediction's
gradient flowing back through memory's state evolution to the prefix
fires?"

Per step:

    1. Prefix pass (teacher-forced, parallel, Gumbel-soft). Memory fires
       T_pre/mod_interval times; each fire writes ΔW/Δdecay with gradient
       attached to the modulator logits. State lives on `memory.{h,msg,W,...}`
       graph-connected (not detached) because we're in
       `wrapper.preserve_memory_graph()` context.
    2. Autoregressive continuation unroll with KV cache. For each of
       T_cont continuation tokens:
         • Feed the GROUND-TRUTH previous token as input (teacher-forced).
         • Forward one token through Llama + memory. Memory does one LIF
           step; no modulator fire (clock never reaches mod_interval from
           a fresh start). State evolves on the carried, graph-connected
           tensors.
         • Compute CE of the step's logit against the ground-truth target.
    3. Loss = mean CE across continuation tokens.
    4. Backward flows: CE → step logit → Llama layers → readout at step →
       memory h at step → memory h at step-1 → … → memory h at prefix-end
       → prefix-fire modulator writes → modulator parameters.
    5. Clip, step, detach memory for the next iteration.

The unroll is an O(T_cont) sequence of single-token forwards. Peak memory
grows linearly in T_cont because the graph is held until backward. For
larger T_cont, wrap the per-step forward in `torch.utils.checkpoint` (not
done here to keep the smoke simple; production runs should checkpoint).
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from src.pretrained.llm_wrapper import PretrainedLMWithMemory
from src.pretrained.telemetry import (
    JsonlLogger, collect_codebook_stats, collect_inject_stats,
    collect_lr_stats, collect_memory_stats, collect_throughput_stats,
)


@contextlib.contextmanager
def _nullcontext():
    yield


@dataclass
class Phase1ARBatch:
    prefix_ids: Tensor          # [BS, T_pre]
    continuation_ids: Tensor    # [BS, T_cont] — teacher-forced inputs AND targets


@dataclass
class Phase1ARStepLog:
    step: int
    loss: float          # = continuation CE
    mem_pred_loss: float   # memory's aux NTP loss; populated if the
                            # prefix-pass forward produced one (i.e.,
                            # wrapper was in training mode, which the AR
                            # loop always is).
    gumbel_tau: float
    grad_norm: float
    extras: dict = field(default_factory=dict)


def anneal_gumbel_tau(step: int, total_steps: int, tau_start: float,
                      tau_end: float) -> float:
    if total_steps <= 0:
        return tau_end
    frac = min(1.0, step / total_steps)
    return tau_start + (tau_end - tau_start) * frac


def run_phase1_ar(
    wrapper: PretrainedLMWithMemory,
    optimizer: torch.optim.Optimizer,
    data_iter: Iterator[Phase1ARBatch],
    *,
    steps: int,
    max_grad_norm: float = 1.0,
    mem_pred_weight: float = 0.1,
    gumbel_tau_start: float = 1.0,
    gumbel_tau_end: float = 0.3,
    anneal_across_steps: int | None = None,
    on_step: Callable[[Phase1ARStepLog], None] | None = None,
    metrics_path: str | None = None,
    log_interval: int = 10,
):
    """Autoregressive Gumbel-softmax unroll training for phase 1.

    Expects each batch to carry a (prefix, continuation) pair. Memory
    state is carried graph-connected from the prefix pass through every
    continuation step; one backward pass at the end flows gradient to
    every modulator fire during the prefix.
    """
    anneal_across_steps = anneal_across_steps or steps
    wrapper.train()
    device = next(wrapper.parameters()).device
    device_type = device.type
    use_autocast = device_type == "cuda"
    logger = JsonlLogger(metrics_path)

    for step in range(steps):
        t_step_start = time.time()
        batch = next(data_iter)
        prefix_ids = batch.prefix_ids.to(device)
        cont_ids = batch.continuation_ids.to(device)
        BS, T_pre = prefix_ids.shape
        T_cont = cont_ids.shape[1]
        assert cont_ids.shape[0] == BS

        tau = anneal_gumbel_tau(step, anneal_across_steps,
                                gumbel_tau_start, gumbel_tau_end)
        wrapper.memory.gumbel_tau = tau
        wrapper.reset_memory(bs=BS)

        amp_ctx = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                   if use_autocast else _nullcontext())

        # The whole forward — prefix pass + continuation unroll — happens
        # inside preserve_memory_graph so memory state stays connected
        # across calls. autocast covers matmul dtypes on CUDA.
        with wrapper.preserve_memory_graph(), amp_ctx:
            # 1. Prefix pass. Aux memory-pred loss runs here (modulator
            # fires; readouts line up with input_ids). KV cache captured
            # for the unroll.
            out_prefix = wrapper(prefix_ids, use_cache=True)
            past = out_prefix.past_key_values
            prefix_mem_loss = wrapper._last_mem_pred_loss     # scalar or None
            # Logit at the LAST prefix position predicts cont_ids[:, 0].
            last_logit = out_prefix.logits[:, -1]     # [BS, vocab]
            ce_first = F.cross_entropy(last_logit.float(), cont_ids[:, 0])

            # 2. Unroll with aux loss DISABLED — each T=1 continuation
            # step's aux would be a near-zero-signal 128K-vocab matmul on
            # a single readout. Keep the prefix's aux signal (where all
            # the modulator fires happen).
            ce_terms: list[Tensor] = [ce_first]
            with wrapper.compute_aux_loss_override(False):
                for i in range(T_cont - 1):
                    new_tok = cont_ids[:, i:i + 1]        # [BS, 1]
                    out_i = wrapper(new_tok, past_key_values=past, use_cache=True)
                    past = out_i.past_key_values
                    logit_i = out_i.logits[:, -1]         # [BS, vocab]
                    target_i = cont_ids[:, i + 1]
                    ce_terms.append(F.cross_entropy(logit_i.float(), target_i))

            # Mean CE over all T_cont predicted positions + weighted aux.
            cont_loss = torch.stack(ce_terms).mean()
            if prefix_mem_loss is not None and torch.is_tensor(prefix_mem_loss):
                loss = cont_loss + mem_pred_weight * prefix_mem_loss
            else:
                loss = cont_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for _, p in wrapper.trainable_parameters()]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).item()
        optimizer.step()
        wrapper.detach_memory()

        mem_loss_val = (float(prefix_mem_loss.item())
                        if torch.is_tensor(prefix_mem_loss) else 0.0)

        is_log_step = step == 0 or (step + 1) % log_interval == 0
        extras: dict = {}
        if is_log_step:
            extras.update(collect_lr_stats(optimizer))
            extras.update(collect_inject_stats(wrapper))
            extras.update(collect_codebook_stats(wrapper))
            extras.update(collect_memory_stats(wrapper, include_slow=True))
            ms_step = (time.time() - t_step_start) * 1000
            tok_per_s = (prefix_ids.numel() + cont_ids.numel()) / max(
                1e-9, (time.time() - t_step_start))
            extras.update(collect_throughput_stats(
                tok_per_s=tok_per_s, ms_per_step=ms_step, device=device))

        log = Phase1ARStepLog(
            step=step,
            loss=float(loss.item()),
            mem_pred_loss=mem_loss_val,
            gumbel_tau=tau,
            grad_norm=float(grad_norm) if math.isfinite(grad_norm) else float("inf"),
            extras=extras,
        )
        if is_log_step:
            logger.write({
                "phase": "phase1_ar",
                "step": log.step,
                "loss": log.loss,
                "mem_pred_loss": log.mem_pred_loss,
                "gumbel_tau": log.gumbel_tau,
                "grad_norm": log.grad_norm,
                **log.extras,
            })
        if on_step is not None:
            on_step(log)
