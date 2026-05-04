"""Phase-2 GRPO for graph_walker + frozen Llama.

Per step:
1. Replicate `prefix` K times → [K, T_pre].
2. `wrapper.current_phase = "phase2"`, hard Categorical routing.
3. Prefix pass: `wrapper(prefix_rep, use_cache=True)`. Walker accumulates
   per-batch `log_pi_sum [K]` over every routing decision (anchor +
   per-token). KV cache captured for the unroll.
4. Generation: K rollouts diverge in token space. Plasticity frozen during
   gen (memory's slow state is fixed at post-prefix value).
5. Reward: `reward_fn(generated [K, T_pre+L], reference [L]) → rewards [K]`.
6. Advantage: `A = (r - r.mean()) / max(r.std(), adv_std_floor)`.
7. REINFORCE loss: `L = -(log_pi_sum · A.detach()).mean()`.
8. Backward → grad-clip → step → detach memory.

Trainable surface: whatever the caller put in their optimizer. The
`freeze_all_but_E_bias_and_neuromod()` helper on the wrapper sets a
minimal phase-2 surface — only the neuromod's parameters move under
REINFORCE, leaving the bulk of routing parameters fixed (E_bias is a
buffer, not a parameter, and gets updated via plasticity / detach).

Not yet wired (future work): KL penalty against a reference policy,
entropy bonus from the full routing distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.rollout import autoregressive_rollout


@dataclass
class GRPOStats:
    loss: float
    reward_mean: float
    reward_std: float
    reward_min: float            # min reward across K rollouts; floor
    reward_max: float            # max reward across K rollouts; ceiling
    log_pi_mean: float
    log_pi_max_abs: float        # max |log_pi_mean| across rollouts; magnitude regression detector
    advantage_max: float
    advantage_std: float         # std of advantages; learning-signal indicator
    gen_unique_count: int        # # unique generations among K rollouts; if 1, no signal
    grad_norm: float


def grpo_step(
    wrapper: GraphWalkerPretrainedLM,
    opt: torch.optim.Optimizer,
    *,
    prefix_ids: torch.Tensor,        # [1, T_pre] or [T_pre]
    reference_cont: torch.Tensor,    # [L] reference continuation tokens
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    num_rollouts: int = 8,
    gen_length: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    grad_clip: float = 1.0,
    adv_std_floor: float = 1e-3,
    gen_sample_routing: bool = False,
) -> GRPOStats:
    """One GRPO step on a single (prefix, reference) pair.

    Args:
        reward_fn: maps `(generated [K, T_pre+L], reference [L]) → rewards [K]`.
            Default is per-rollout fraction of generated tail tokens that
            match the reference (placeholder; real runs want BERT-cosine).
        gen_sample_routing: if True, walker routing during generation
            uses Categorical sampling (more reward variance across the K
            rollouts at the cost of noisier walker writes during gen).
            Defaults to False (argmax routing — matches the policy that
            inference would use). Try True if reward_std stays near zero
            for too long.
    """
    if reward_fn is None:
        reward_fn = _default_token_match_reward

    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    K = num_rollouts
    prefix_rep = prefix_ids.expand(K, -1).contiguous()

    opt.zero_grad(set_to_none=True)

    out = autoregressive_rollout(
        wrapper, prefix_rep,
        gen_length=gen_length,
        temperature=temperature,
        top_p=top_p,
        phase="phase2",
        grad_during_prefix=True,
        grad_during_gen=False,
        gen_sample_routing=gen_sample_routing,
    )

    log_pi_mean = out.log_pi_mean
    if log_pi_mean is None:
        raise RuntimeError(
            "GRPO step produced no log_pi — phase-2 routing didn't fire. "
            "Check that wrapper.memory.phase == 'phase2' during the prefix."
        )
    # In phase-2 the only trainable surface is neuromod (everything else
    # frozen by `freeze_all_but_E_bias_and_neuromod`). Routing scores then
    # only carry grad via `_active_delta_nm`. If the prefix doesn't span a
    # full plasticity window AND `reset_memory()` cleared neuromod carryover
    # at start-of-segment, the first window runs with `_active_delta_nm=None`
    # → routing scores have no grad → `log_pi_mean.requires_grad=False` →
    # `loss.backward()` raises an opaque "element 0 of tensors does not
    # require grad" error. Fail loudly with a fix-it message instead.
    if not log_pi_mean.requires_grad:
        raise RuntimeError(
            "log_pi_mean has no grad in phase-2 — typically means the prefix "
            "did not span a full plasticity window AND neuromod carryover was "
            "cleared, so the first window had _active_delta_nm=None. Make "
            f"prefix_len > memory.cfg.mod_period, OR use "
            "reset_memory(clear_neuromod_carryover=False)."
        )

    rewards = reward_fn(out.generated, reference_cont).to(log_pi_mean.device)
    if rewards.shape != (K,):
        raise ValueError(
            f"reward_fn returned shape {tuple(rewards.shape)}, expected ({K},)"
        )
    r_mean = rewards.mean()
    r_std = rewards.std().clamp(min=adv_std_floor)
    advantages = (rewards - r_mean) / r_std

    # REINFORCE on per-step-mean log π. The mean normalization is what
    # keeps gradient magnitudes bounded as T_pre changes — see the
    # consume_log_pi_mean docstring in graph_walker.py for the math.
    loss = -(log_pi_mean * advantages.detach()).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in wrapper.trainable_parameters()],
        grad_clip,
    )
    opt.step()
    wrapper.detach_memory()

    # Generation diversity: how many of the K rollouts produced unique
    # token sequences? If 1, all rollouts converged, advantages are 0,
    # no learning signal — early warning for "GRPO has nothing to do."
    gen_unique_count = int(out.new_tokens.unique(dim=0).shape[0])

    return GRPOStats(
        loss=float(loss.detach()),
        reward_mean=float(r_mean.detach()),
        reward_std=float(rewards.std().detach()),
        reward_min=float(rewards.detach().min()),
        reward_max=float(rewards.detach().max()),
        log_pi_mean=float(log_pi_mean.detach().mean()),
        log_pi_max_abs=float(log_pi_mean.detach().abs().max()),
        advantage_max=float(advantages.detach().abs().max()),
        advantage_std=float(advantages.detach().std()),
        gen_unique_count=gen_unique_count,
        grad_norm=float(grad_norm) if isinstance(grad_norm, torch.Tensor)
                  else float(grad_norm),
    )


def _default_token_match_reward(
    generated: torch.Tensor,         # [K, T_pre + L]
    reference: torch.Tensor,         # [L]
) -> torch.Tensor:
    """Placeholder reward: fraction of generated tail tokens that match
    the reference (hard token-match). Use as a smoke baseline only —
    real runs need a paraphrase-tolerant signal (PrefBERT, entity-F1).
    """
    K = generated.shape[0]
    L = reference.shape[0]
    tail = generated[:, -L:]
    matches = (tail == reference.unsqueeze(0).to(tail.device)).float().mean(dim=1)
    return matches
