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
    5. Advantage = (reward - mean) / max(std, adv_std_floor).
    6. Loss = -(log_pi_sum * advantage.detach()).mean().
       KL / entropy regularizers are intentionally not wired yet.
    7. Backward, clip, step, detach memory.

The critical "break verify_01" property: step 3 GENERATES tokens under
memory-influenced logits, so the K rollouts produce different token
streams. Reward differs across rollouts in a way that memory-code
choice can drive, so the GRPO advantage has real signal — unlike the
teacher-forced verify_01 setup whose SNR was 1e-4.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.pretrained.llm_wrapper import PretrainedLMWithMemory
from src.pretrained.telemetry import (
    JsonlLogger, collect_codebook_stats, collect_inject_stats,
    collect_lr_stats, collect_memory_stats, collect_rollout_stats,
    collect_throughput_stats, summarize_tensor,
)


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
    extras: dict = field(default_factory=dict)


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
    reward_fn: Callable[[Tensor, Tensor], Tensor],
    temperature: float = 1.0,
    max_grad_norm: float = 1.0,
    adv_std_floor: float = 1e-3,
    seed: int | None = None,
    metrics_path: str | None = None,
    step_idx: int = 0,
    collect_heavy_telemetry: bool = True,
) -> GrpoStepLog:
    """Run one GRPO step. `reward_fn` is required (no default).

    **Caller responsibility**: this helper backpropagates through
    whatever parameters have `requires_grad=True`. The canonical
    phase-2 protocol (logit_head only) is set up by
    `run_cycle_loop` / `freeze_all_but_logit_head`; if you call
    `grpo_step` directly with the full training surface attached to
    the optimizer, all trainable memory params move. That is
    intentional flexibility — for production phase-2 runs, freeze the
    surface yourself before handing the optimizer in.

    **Reward warning**: `token_match_reward` (exported from this
    module) is near-zero-signal over a 128K-vocab continuation unless
    the task is already near exact-match. It's for smoke tests only.
    Real runs should pass a task-appropriate reward (log-prob of
    ground-truth continuation under the memory-augmented LM, BLEU,
    downstream-task score, etc.)."""
    assert prefix_ids.dim() == 2 and prefix_ids.shape[0] == 1
    device = next(wrapper.parameters()).device
    prefix_ids = prefix_ids.to(device)
    reference_cont = reference_cont.to(device)
    K = num_rollouts
    T_prefix = prefix_ids.shape[1]
    device = prefix_ids.device

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        # Also seed the default torch RNG because MemoryGraph.initialize_states
        # uses torch.randperm (no explicit generator) for the sparse-W init;
        # without this, two seeded grpo_step calls produce different initial
        # W across the same reset_memory sequence.
        _rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    else:
        gen = None
        _rng_state = None
    t_step_start = time.time()

    # 1) Replicate prefix. 2) Reset memory with BS=K.
    prefix_rep = prefix_ids.expand(K, T_prefix).contiguous()
    wrapper.reset_memory(bs=K)

    # 3) Prefix pass inside rollout_mode: eval (no dropout) + hard
    # Categorical sampling + phase2 + bf16 autocast on CUDA. Removes the
    # previous non-determinism where modulator dropout, turned on by
    # wrapper.train(True), added noise not represented in log_pi.
    # The REINFORCE gradient path through log_pi_sum is unchanged —
    # log_pi is still graph-connected through log_softmax to modulator
    # logits.
    with wrapper.rollout_mode():
        out = wrapper(prefix_rep, use_cache=True)
        log_pi_sum = wrapper.memory._last_log_pi_sum            # [K], graph-connected
        last_prefix_logits = out.logits[:, -1].detach()         # [K, vocab_lm]
        past_key_values = out.past_key_values

    # 4) Generate under no_grad; rewards. Memory's carried state [K, ...]
    # diverges per rollout thanks to the hard-Categorical prefix sampling,
    # and each gen step runs with a T=1 segment so memory updates its
    # fast state exactly once per generated token (no modulator fires —
    # the mod clock never reaches mod_interval from a fresh start).
    # Generation runs in eval mode too, so gen-step memory forwards don't
    # re-sample codes.
    generated = _generate_tokens(
        wrapper, last_prefix_logits, past_key_values,
        gen_length=gen_length, temperature=temperature, gen=gen)
    rewards = reward_fn(generated, reference_cont)           # [K]
    # Advantage normalization. `adv_std_floor` clamps the denominator so
    # early-training rollouts with near-uniform rewards don't produce
    # 1e4-scale advantages from noise-level std. Standard eps=1e-8 is
    # unstable — it turns small real variance into explosive gradients
    # through log_pi_sum before grad clipping.
    reward_std = rewards.std()
    denom = torch.clamp(reward_std, min=adv_std_floor)
    advantages = (rewards - rewards.mean()) / denom
    advantages = advantages.to(log_pi_sum.dtype)

    # 5) GRPO loss: -E[log_pi * A]. Minus because we minimize; sign gives
    # policy gradient in the right direction. KL-to-reference-policy and
    # entropy-bonus penalties intentionally omitted — adding them requires
    # snapshotting a reference modulator (KL) and plumbing per-fire logits
    # out of the memory graph (entropy). Tracked in docs as future work.
    loss = -(log_pi_sum * advantages.detach()).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [p for _, p in wrapper.trainable_parameters()]
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()

    # Reset memory phase for any subsequent phase-1 calls. `rollout_mode`
    # already restored train/phase to pre-rollout state, but be explicit:
    # phase-1 is the default outside this function.
    wrapper.current_phase = "phase1"
    wrapper.train(True)
    wrapper.detach_memory()

    # Rich telemetry — distributions and rollout diagnostics. Cheap on
    # K=8 rollout scale; gated on `collect_heavy_telemetry` so unit smokes
    # can skip the memory-health CUDA syncs.
    extras: dict = {}
    if collect_heavy_telemetry:
        extras.update(collect_lr_stats(optimizer))
        extras.update(collect_inject_stats(wrapper))
        extras.update(collect_codebook_stats(wrapper))
        extras.update(collect_memory_stats(wrapper, include_slow=False))
        extras.update(summarize_tensor(rewards, prefix="reward"))
        extras.update(summarize_tensor(advantages, prefix="advantage"))
        extras.update(summarize_tensor(log_pi_sum.detach(), prefix="log_pi_sum"))
        extras.update(collect_rollout_stats(generated, reference_cont))
        ms_step = (time.time() - t_step_start) * 1000
        toks = K * (T_prefix + gen_length)
        tok_per_s = toks / max(1e-9, time.time() - t_step_start)
        extras.update(collect_throughput_stats(
            tok_per_s=tok_per_s, ms_per_step=ms_step, device=device))

    log = GrpoStepLog(
        loss=float(loss.item()),
        reward_mean=float(rewards.mean().item()),
        reward_std=float(rewards.std().item()),
        log_pi_mean=float(log_pi_sum.detach().mean().item()),
        advantage_max_abs=float(advantages.abs().max().item()),
        extras=extras,
    )
    if metrics_path is not None:
        JsonlLogger(metrics_path).write({
            "phase": "phase2_grpo",
            "step": int(step_idx),
            "loss": log.loss,
            "reward_mean": log.reward_mean,
            "reward_std": log.reward_std,
            "log_pi_mean": log.log_pi_mean,
            "advantage_max_abs": log.advantage_max_abs,
            **log.extras,
        })
    if _rng_state is not None:
        torch.random.set_rng_state(_rng_state)
    return log
