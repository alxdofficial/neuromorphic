"""Phase 2: GRPO on autoregressive rollouts.

Per step:
    1. Replicate prefix K times so K parallel rollouts share the prefix.
    2. Run prefix through wrapper in phase 2 mode — memory samples K
       independent code streams (hard Categorical + log_pi tracked). The
       segment-total log π sum lands on memory._last_log_pi_sum with
       the graph still connected back to the modulator logits.
    3. Freeze memory state (train(False) for the LM but leave memory's
       log_pi graph attached) and generate gen_length tokens
       autoregressively per rollout. Generation is under no_grad; its
       purpose is to produce divergent continuations whose reward varies
       across K.
    4. Compute per-rollout reward = token-level match with the reference
       continuation. Richer rewards (BLEU, self-BLEU, task-specific
       metrics) can slot in here without changing the rest of the loop.
    5. Advantage = (reward - mean) / (std + eps).
    6. Loss = -(log_pi_sum * advantage.detach()).mean().
       Optionally add KL penalty vs a reference policy (future work).
    7. Backward, clip, step, detach memory.

The critical "break verify_01" property: step 3 GENERATES tokens under
memory-influenced logits, so the K rollouts produce different token
streams. Reward differs across rollouts in a way that memory-code
choice can drive, so the GRPO advantage has real signal — unlike the
teacher-forced verify_01 setup whose SNR was 1e-4.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.pretrained.llm_wrapper import PretrainedLMWithMemory


@contextlib.contextmanager
def _nullcontext():
    yield


@dataclass
class GrpoStepLog:
    loss: float
    reward_mean: float
    reward_std: float
    log_pi_mean: float
    advantage_max_abs: float


def token_match_reward(
    generated: Tensor,     # [K, gen_length]
    reference: Tensor,     # [gen_length]
) -> Tensor:
    """Fraction of generated tokens matching the reference continuation."""
    assert reference.dim() == 1 and reference.shape[0] == generated.shape[1]
    return (generated == reference.unsqueeze(0)).float().mean(dim=-1)  # [K]


def _generate_tokens(
    wrapper: PretrainedLMWithMemory,
    last_prefix_logits: Tensor,   # [K, vocab] — logits from position T_prefix-1
    past_key_values,              # HF Cache from the prefix pass
    gen_length: int,
    temperature: float,
    gen: torch.Generator | None,
) -> Tensor:
    """Sample `gen_length` tokens autoregressively using KV cache.

    The first token is sampled from `last_prefix_logits` (the logits at the
    last prefix position, which are the distribution over position T_prefix).
    Each subsequent step passes ONLY the newly sampled token plus the KV
    cache, so memory sees a T=1 segment per step and does one LIF update
    on its carried state — no re-processing of the prefix, no state drift.
    """
    out_tokens = []
    with torch.no_grad():
        # First sample from the prefix's last-position logits.
        probs = F.softmax(last_prefix_logits.float() / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=gen)  # [K, 1]
        out_tokens.append(sampled)
        past = past_key_values

        for _ in range(gen_length - 1):
            out = wrapper(sampled, past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits = out.logits[:, -1].float()
            probs = F.softmax(step_logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1, generator=gen)
            out_tokens.append(sampled)

    return torch.cat(out_tokens, dim=1)  # [K, gen_length]


def grpo_step(
    wrapper: PretrainedLMWithMemory,
    optimizer: torch.optim.Optimizer,
    *,
    prefix_ids: Tensor,         # [1, T_prefix]
    reference_cont: Tensor,     # [gen_length] — ground-truth continuation tokens
    num_rollouts: int,
    gen_length: int,
    temperature: float = 1.0,
    max_grad_norm: float = 1.0,
    entropy_coef: float = 0.0,
    reward_fn: Callable[[Tensor, Tensor], Tensor] = token_match_reward,
    seed: int | None = None,
) -> GrpoStepLog:
    assert prefix_ids.dim() == 2 and prefix_ids.shape[0] == 1
    K = num_rollouts
    T_prefix = prefix_ids.shape[1]
    device = prefix_ids.device

    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    # 1) Replicate prefix. 2) Reset memory with BS=K.
    prefix_rep = prefix_ids.expand(K, T_prefix).contiguous()
    wrapper.current_phase = "phase2"
    wrapper.train(True)
    wrapper.reset_memory(bs=K)

    device_type = next(wrapper.parameters()).device.type
    use_autocast = device_type == "cuda"
    amp_ctx = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
               if use_autocast else _nullcontext())

    # 3) Prefix pass with gradient tracking — memory samples hard codes and
    # log_pi accumulates. Request KV cache so the generation loop can skip
    # re-processing the prefix (which would otherwise re-run memory's LIF
    # over prefix tokens on every gen step, drifting the state).
    with amp_ctx:
        out = wrapper(prefix_rep, use_cache=True)
    log_pi_sum = wrapper.memory._last_log_pi_sum            # [K], graph-connected
    last_prefix_logits = out.logits[:, -1].detach()         # [K, vocab_lm]
    past_key_values = out.past_key_values

    # 4) Generate under no_grad; rewards. Memory's carried state [K, ...]
    # diverges per rollout thanks to the hard-Categorical prefix sampling,
    # and each gen step runs with a T=1 segment so memory updates its
    # fast state exactly once per generated token (no modulator fires —
    # the mod clock never reaches mod_interval from a fresh start).
    wrapper.train(False)
    generated = _generate_tokens(
        wrapper, last_prefix_logits, past_key_values,
        gen_length=gen_length, temperature=temperature, gen=gen)
    rewards = reward_fn(generated, reference_cont)           # [K]
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages = advantages.to(log_pi_sum.dtype)

    # 5) GRPO loss: -E[log_pi * A]. Minus because we minimize; sign gives
    # policy gradient in the right direction.
    loss = -(log_pi_sum * advantages.detach()).mean()

    # Optional entropy bonus — encourages exploration. Current log_pi_sum
    # is already a function of the sampled codes, so we'd need per-fire
    # logits to compute entropy properly. Skip for this smoke.
    _ = entropy_coef  # reserved

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [p for _, p in wrapper.trainable_parameters()]
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()

    # Reset memory phase for any subsequent phase-1 calls.
    wrapper.current_phase = "phase1"
    wrapper.train(True)
    wrapper.detach_memory()

    return GrpoStepLog(
        loss=float(loss.item()),
        reward_mean=float(rewards.mean().item()),
        reward_std=float(rewards.std().item()),
        log_pi_mean=float(log_pi_sum.detach().mean().item()),
        advantage_max_abs=float(advantages.abs().max().item()),
    )
