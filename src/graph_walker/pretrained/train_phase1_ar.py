"""Phase-1 autoregressive unroll for graph_walker + frozen Llama.

Why AR (vs parallel teacher-forced):
    Parallel phase-1 trains memory on a distribution memory will never face
    at inference: predict the *next* token while the LM also sees the
    correct prefix tokens. The walker contributes a tiny CE delta in this
    regime. The AR unroll instead trains the walker to "use the writes
    you made during the prefix to shape the CONTINUATION's logits, one
    token at a time, with each prediction's gradient reaching back through
    memory's recurrent state to the prefix-pass writes."

Per step:
1. `reset_memory(bs=1)`. Replicate prefix to [BS, T_pre].
2. Enter `preserve_memory_graph()`: memory state stays graph-connected
   across the prefix pass and the per-token continuation forwards. No
   intra-segment detach.
3. Prefix forward: `wrapper(prefix_ids, use_cache=True)`. Walker fires
   plasticity inside `forward_segment` at every `mod_period`. Gradient
   from later continuation steps reaches every fire's neuromod write.
4. Continuation unroll: for `i = 0..T_cont - 1`:
     - Feed ground-truth token `cont_ids[:, i:i+1]` with the captured
       past_key_values → `out_i.logits[:, -1]`.
     - Walker runs one step on the carried graph-connected state.
     - CE of `prev_logits` (from the previous forward) against
       `cont_ids[:, i]`.
       For i=0, `prev_logits` comes from the PREFIX's last position.
5. Loss = mean of T_cont per-step CE losses.
6. `loss.backward()` → grad-clip → `opt.step()` → `detach_memory()`.

Invariants:
- `cfg.memory.tbptt_block >= T_pre`. The walker's intra-segment detach
  is bypassed inside `preserve_memory_graph()`, but the per-block
  surprise fold and plasticity fires still happen at `mod_period` cadence.
  No detach means activations live until backward — VRAM scales with
  `T_pre + T_cont`.
- Walker's slow plastic state (E_bias_flat) is updated by the prefix
  fires in-place (detached, additive). The continuation unroll READS
  e_bias_active but doesn't fire new plasticity (window not yet closed
  again).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rollout import _freeze_plasticity_ctx


@dataclass
class Phase1ARBatch:
    prefix_ids: torch.Tensor          # [BS, T_pre]
    continuation_ids: torch.Tensor    # [BS, T_cont]


@dataclass
class Phase1ARStats:
    loss: float
    ce_per_step: list[float]      # length T_cont
    grad_norm: float


def phase1_ar_pretrained_step(
    wrapper: GraphWalkerPretrainedLM,
    opt: torch.optim.Optimizer,
    batch: Phase1ARBatch,
    *,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
) -> Phase1ARStats:
    """One AR-unroll training step. Both prefix and continuation pass through
    the wrapper; the walker stays graph-connected across the boundary."""
    device = next(wrapper.parameters()).device
    prefix_ids = batch.prefix_ids.to(device)
    continuation_ids = batch.continuation_ids.to(device)
    BS, T_pre = prefix_ids.shape
    _, T_cont = continuation_ids.shape
    if T_cont < 1:
        raise ValueError(
            f"continuation_ids must have at least one token; got T_cont={T_cont}. "
            "AR unroll has nothing to predict."
        )

    if not wrapper.training:
        raise RuntimeError(
            "phase1_ar_pretrained_step requires wrapper.train() — inference-"
            "mode leak would silently disable aux loss and Gumbel-STE routing."
        )

    wrapper.reset_memory(bs=BS)
    opt.zero_grad(set_to_none=True)

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if (amp_dtype is not None and device.type == "cuda")
        else torch.autocast(device_type=device.type, enabled=False)
    )

    losses: list[torch.Tensor] = []

    with ctx, wrapper.preserve_memory_graph():
        # 1. Prefix pass with KV cache.
        prefix_out = wrapper(prefix_ids, use_cache=True)
        past_key_values = prefix_out.past_key_values
        # Logits for the FIRST continuation step come from the prefix's
        # last position.
        prev_logits = prefix_out.logits[:, -1, :]                  # [BS, vocab]

        # 2. Continuation unroll. The continuation forwards are T=1
        # "policy reads", not training tokens — they have no upcoming
        # targets within the segment, so any plasticity firing during
        # them would be driven by stale surprise + the generation walks'
        # co-visit footprint. Freeze plasticity here (mirroring what
        # phase-2 generation does in autoregressive_rollout). The walker
        # state itself stays graph-connected via preserve_memory_graph,
        # so gradient from the per-step CE still reaches the prefix's
        # plastic / neuromod fires.
        with _freeze_plasticity_ctx(wrapper.memory):
            for i in range(T_cont):
                # CE for step i: prev_logits should predict continuation_ids[:, i].
                ce_i = F.cross_entropy(
                    prev_logits.float(), continuation_ids[:, i],
                )
                losses.append(ce_i)

                if i + 1 == T_cont:
                    # Last step: no need for another forward.
                    break

                # Feed ground-truth token i forward to produce logits for token i+1.
                tok = continuation_ids[:, i:i+1]                   # [BS, 1]
                out = wrapper(
                    tok, past_key_values=past_key_values, use_cache=True,
                )
                past_key_values = out.past_key_values
                prev_logits = out.logits[:, -1, :]

    loss = torch.stack(losses).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in wrapper.trainable_parameters()],
        grad_clip,
    )
    opt.step()
    wrapper.detach_memory()

    return Phase1ARStats(
        loss=float(loss.detach()),
        ce_per_step=[float(c.detach()) for c in losses],
        grad_norm=float(grad_norm) if isinstance(grad_norm, torch.Tensor)
                 else float(grad_norm),
    )
