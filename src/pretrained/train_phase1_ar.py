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
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

from src.pretrained.llm_wrapper import PretrainedLMWithMemory


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
    gumbel_tau: float
    grad_norm: float


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
    gumbel_tau_start: float = 1.0,
    gumbel_tau_end: float = 0.3,
    anneal_across_steps: int | None = None,
    on_step: Callable[[Phase1ARStepLog], None] | None = None,
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

    for step in range(steps):
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
            # 1. Prefix pass. KV cache captured for the unroll.
            out_prefix = wrapper(prefix_ids, use_cache=True)
            past = out_prefix.past_key_values
            # Logit at the LAST prefix position predicts cont_ids[:, 0].
            last_logit = out_prefix.logits[:, -1]     # [BS, vocab]
            ce_first = F.cross_entropy(last_logit.float(), cont_ids[:, 0])

            # 2. Unroll. Feed cont_ids[:, i] as the new input for step i+1;
            # its logit predicts cont_ids[:, i+1]. So after T_cont-1 steps
            # we have scored positions 1..T_cont-1.
            ce_terms: list[Tensor] = [ce_first]
            for i in range(T_cont - 1):
                new_tok = cont_ids[:, i:i + 1]        # [BS, 1]
                out_i = wrapper(new_tok, past_key_values=past, use_cache=True)
                past = out_i.past_key_values
                logit_i = out_i.logits[:, -1]         # [BS, vocab]
                target_i = cont_ids[:, i + 1]
                ce_terms.append(F.cross_entropy(logit_i.float(), target_i))

            # Mean CE over all T_cont predicted positions.
            loss = torch.stack(ce_terms).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for _, p in wrapper.trainable_parameters()]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).item()
        optimizer.step()
        wrapper.detach_memory()

        log = Phase1ARStepLog(
            step=step,
            loss=float(loss.item()),
            gumbel_tau=tau,
            grad_norm=float(grad_norm) if math.isfinite(grad_norm) else float("inf"),
        )
        if on_step is not None:
            on_step(log)
