"""Autoregressive rollout primitive for graph_walker + frozen Llama.

Used by phase-2 GRPO and inference. Given a prefix `[BS, T_pre]` and a
desired generation length, returns generated token ids `[BS, T_pre + L]`
and (optionally) the per-step logits.

Memory semantics during rollout:
- Plasticity does NOT fire inside `forward_segment` — the walker is
  vocab-agnostic and surprise is supplied by the trainer post-backward
  via `wrapper.memory.update_plasticity(per_token_ce)`.
- During AR rollout (no ground truth), the trainer typically passes
  `None` to `update_plasticity` (or skips the call entirely) and the
  walker's plastic state stays frozen for the duration of the rollout.
- In phase 2 mode, routing is hard Categorical and `log_pi_sum`
  accumulates over routing decisions for REINFORCE.

`_freeze_plasticity_ctx` is retained as a no-op-equivalent safety net
(sets mod_period ≈ ∞). Under the external-surprise design no plasticity
firing happens inside the rollout regardless, but back-compat callers
of `autoregressive_rollout` may still pass `freeze_plasticity=True`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM


@dataclass
class RolloutOutput:
    generated: torch.Tensor          # [BS, T_pre + L] full sequence
    new_tokens: torch.Tensor         # [BS, L] generated tail
    log_pi_sum: torch.Tensor | None  # [BS] log π over routing during prefix
    last_logits: torch.Tensor | None # [BS, vocab] for diagnostics


@contextmanager
def _freeze_plasticity_ctx(memory):
    """Pin the walker's plasticity off (mod_period → ∞) during the
    generation phase so each new token doesn't trigger a fresh fire that
    would re-randomize the policy mid-rollout."""
    if memory is None:
        yield
        return
    saved = memory.cfg.mod_period
    try:
        # Set to a huge number so window_len never reaches it.
        memory.cfg.mod_period = 10**9
        yield
    finally:
        memory.cfg.mod_period = saved


def _sample_next_token(
    logits: torch.Tensor,            # [BS, vocab]
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Temperature + nucleus (top-p) sampling. Returns [BS] long."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        keep = cum <= top_p
        # Always keep the top-1 token.
        keep[:, 0] = True
        # Mask out everything after the cutoff.
        sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, float("-inf")))
        # Scatter back.
        unsort_logits = torch.full_like(logits, float("-inf"))
        unsort_logits.scatter_(1, sorted_idx, sorted_logits)
        logits = unsort_logits
    probs = logits.softmax(dim=-1)
    probs = probs.clamp(min=1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def autoregressive_rollout(
    wrapper: GraphWalkerPretrainedLM,
    prefix_ids: torch.Tensor,        # [BS, T_pre]
    *,
    gen_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    phase: str = "phase2",           # "phase1" or "phase2" (sets walker.phase)
    grad_during_prefix: bool = True,
    grad_during_gen: bool = False,
) -> RolloutOutput:
    """Run a prefix + generation rollout.

    Args:
        wrapper: GraphWalkerPretrainedLM (memory must be attached).
        prefix_ids: [BS, T_pre] starting tokens; replicated by caller for
            K-rollout GRPO (caller is responsible for `prefix_ids = prefix.repeat(K, 1)`).
        gen_length: number of tokens to generate.
        temperature, top_p: sampling controls.
        phase: "phase2" for GRPO (hard Categorical routing + log_pi),
            "phase1" for diagnostic / forward-only inference.
        grad_during_prefix: whether to keep gradients for the prefix pass
            (True for GRPO; False for inference).
        grad_during_gen: whether to keep gradients during generation. GRPO
            uses False — the policy is captured in log_pi_sum from the
            prefix pass, generation is just sampling under it.
    """
    assert wrapper.memory is not None, "rollout requires attached memory"
    device = next(wrapper.parameters()).device
    prefix_ids = prefix_ids.to(device)
    BS, T_pre = prefix_ids.shape
    # `wrapper.forward()` propagates wrapper.current_phase → memory.phase
    # before the closure runs. We also save/restore `wrapper.training`:
    # the generation loop sets train(False) which would otherwise leak
    # into the next phase-1 cycle (compute_aux gates on `self.training`,
    # so a leaked False would silently skip aux loss).
    saved_phase = wrapper.current_phase
    saved_training = wrapper.training
    wrapper.current_phase = phase

    try:
        wrapper.reset_memory(bs=BS)

        # Prefix pass.
        prefix_ctx = (
            torch.enable_grad() if grad_during_prefix else torch.no_grad()
        )
        with prefix_ctx:
            wrapper.train(grad_during_prefix)
            out = wrapper(prefix_ids, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]
            log_pi_sum = wrapper.memory.consume_log_pi_sum()

        # Generation pass — freeze plasticity to keep the policy fixed.
        gen_ctx = torch.enable_grad() if grad_during_gen else torch.no_grad()
        new_tokens: list[torch.Tensor] = []
        with gen_ctx, _freeze_plasticity_ctx(wrapper.memory):
            wrapper.train(False)
            for _ in range(gen_length):
                tok = _sample_next_token(last_logits, temperature, top_p)
                new_tokens.append(tok)
                out = wrapper(
                    tok.unsqueeze(-1), past_key_values=past_kv, use_cache=True,
                )
                past_kv = out.past_key_values
                last_logits = out.logits[:, -1, :]
        new_tokens_t = torch.stack(new_tokens, dim=1) if new_tokens else \
                       torch.zeros(BS, 0, dtype=torch.long, device=device)
        full = torch.cat([prefix_ids, new_tokens_t], dim=1)

        return RolloutOutput(
            generated=full,
            new_tokens=new_tokens_t,
            log_pi_sum=log_pi_sum,
            last_logits=last_logits.detach(),
        )
    finally:
        wrapper.current_phase = saved_phase
        wrapper.train(saved_training)
        # memory.phase was last set by wrapper.forward() during the
        # generation loop. Sync it back so a follow-up direct memory
        # call (without going through wrapper.forward) doesn't see a
        # leaked phase-2 setting.
        if wrapper.memory is not None:
            wrapper.memory.phase = saved_phase
