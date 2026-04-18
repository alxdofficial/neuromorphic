"""Autoregressive rollout primitive for phase-2 GRPO.

Setup:
    K parallel rollouts share a prefix, diverge through the memory graph's
    stochastic code sampling during prefix processing, and then generate
    `gen_length` tokens autoregressively. Because memory state differs per
    rollout and the memory injects into the LM residual, each rollout's
    per-step logits differ — so sampled tokens diverge and each rollout
    yields a distinct continuation.

Memory during generation:
    For T=1 per step, the modulator fires when (t+1) % mod_interval == 0
    with t = start_t + offset = 0 + 0 = 0 — so it NEVER fires during
    single-token forward calls. Memory's fast state (h, msg, prev_readout)
    still updates per-token, but W / decay / hebbian stay frozen at their
    post-prefix values. The discrete code sample that differentiated rollouts
    was drawn at the last modulator fire before gen start.

This is the phase-2 generation contract: differences in memory writes land
during PREFIX processing, and the divergent memory states drive divergent
token generation during gen. GRPO will train those prefix writes.

Scope of this module: primitive only. Reward computation and GRPO loss live
in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.pretrained.llm_wrapper import PretrainedLMWithMemory


@dataclass
class RolloutResult:
    generated_ids: Tensor    # [K, gen_length]
    prefix_ids: Tensor       # [K, T_prefix] (replicated across rollouts)
    final_logits: Tensor     # [K, vocab] — last-step logits for optional scoring


@torch.no_grad()
def autoregressive_rollout(
    wrapper: PretrainedLMWithMemory,
    prefix_ids: Tensor,             # [1, T_prefix]
    *,
    gen_length: int,
    num_rollouts: int,
    temperature: float = 1.0,
    seed: int | None = None,
) -> RolloutResult:
    """K parallel autoregressive rollouts from a shared prefix.

    Returns [K, gen_length] token ids. Uses multinomial sampling on the
    memory-augmented per-step logits; temperature scales the logits before
    softmax. Seed (if given) makes the sample sequence deterministic.
    """
    assert prefix_ids.dim() == 2 and prefix_ids.shape[0] == 1, (
        f"prefix_ids must be [1, T_prefix]; got {tuple(prefix_ids.shape)}")
    K = num_rollouts
    T_prefix = prefix_ids.shape[1]
    device = prefix_ids.device

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    # 1) Replicate prefix across K rollouts and reset memory for BS=K.
    prefix_rep = prefix_ids.expand(K, T_prefix).contiguous()
    wrapper.train(False)
    wrapper.reset_memory(bs=K)

    # 2) Process prefix. Memory samples stochastic codes per BS item (K
    # independent sample streams), so the end-of-prefix memory states
    # diverge across rollouts. Train-mode memory uses sample_discrete
    # when phase != "phase1"; eval-mode memory uses argmax which collapses
    # rollouts to identical behavior. For true divergence we need train
    # mode during the prefix pass.
    wrapper.train(True)
    with torch.no_grad():
        _ = wrapper(prefix_rep)
    wrapper.train(False)

    # 3) Autoregressive generation. At each step forward the full current
    # sequence (no KV cache yet — adds complexity that's not smoke-scope).
    current = prefix_rep
    generated = []
    last_logits: Tensor | None = None
    for _ in range(gen_length):
        out = wrapper(current)
        step_logits = out.logits[:, -1]          # [K, vocab]
        last_logits = step_logits
        probs = F.softmax(step_logits.float() / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=gen)  # [K, 1]
        generated.append(sampled)
        current = torch.cat([current, sampled], dim=1)

    gen_tensor = torch.cat(generated, dim=1)     # [K, gen_length]
    assert last_logits is not None
    return RolloutResult(
        generated_ids=gen_tensor,
        prefix_ids=prefix_rep,
        final_logits=last_logits,
    )
