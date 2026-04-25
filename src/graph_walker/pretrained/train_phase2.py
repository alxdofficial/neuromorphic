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
    log_pi_mean: float
    advantage_max: float
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
) -> GRPOStats:
    """One GRPO step on a single (prefix, reference) pair.

    Args:
        reward_fn: maps `(generated [K, T_pre+L], reference [L]) → rewards [K]`.
            Default is per-rollout fraction of generated tail tokens that
            match the reference (placeholder; real runs want PrefBERT or
            entity-F1).
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
    )

    log_pi_sum = out.log_pi_sum
    if log_pi_sum is None:
        raise RuntimeError(
            "GRPO step produced no log_pi — phase-2 routing didn't fire. "
            "Check that wrapper.memory.phase == 'phase2' during the prefix."
        )

    rewards = reward_fn(out.generated, reference_cont).to(log_pi_sum.device)
    if rewards.shape != (K,):
        raise ValueError(
            f"reward_fn returned shape {tuple(rewards.shape)}, expected ({K},)"
        )
    r_mean = rewards.mean()
    r_std = rewards.std().clamp(min=adv_std_floor)
    advantages = (rewards - r_mean) / r_std

    loss = -(log_pi_sum * advantages.detach()).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in wrapper.trainable_parameters()],
        grad_clip,
    )
    opt.step()
    wrapper.detach_memory()

    return GRPOStats(
        loss=float(loss.detach()),
        reward_mean=float(r_mean.detach()),
        reward_std=float(rewards.std().detach()),
        log_pi_mean=float(log_pi_sum.detach().mean()),
        advantage_max=float(advantages.detach().abs().max()),
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
