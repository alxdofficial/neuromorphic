"""Phase-2 GRPO for graph_walker + frozen Llama.

Per step:
1. ``prefix_ids[B, T_pre].repeat_interleave(K, dim=0)`` → [B*K, T_pre].
   B = BS_outer (independent prefixes per step, defaults to 1), K =
   num_rollouts (group size for advantage normalization).
2. Sample under no_grad: walker is reset to bs=B*K, captures per-step
   routing trace; LM AR-generates gen_length tokens.
3. Score: ``reward_fn(new_tokens [B*K, L], references [B] or [L]) →
   rewards [B*K]``. Each contiguous K-block is scored against its own
   reference (per-group reward, not shared).
4. Advantage: per-group normalize. Reshape rewards to [B, K], subtract
   per-row mean, divide by per-row std (clamped at adv_std_floor),
   flatten back to [B*K]. NO cross-prefix advantage mixing — REINFORCE
   variance reduction stays within each prompt's K rollouts.
5. Replay: teacher-forced re-forward [B*K, T_pre + gen_length - 1] with
   grad enabled. Walker consumes the captured trace (replay_choices),
   produces log_pi_mean [B*K] with grad attached AND per_token_ce
   [B*K, T_replay] no-grad (Llama's next-token CE against the replay
   sequence — used as the plasticity surprise signal post-opt.step).
6. Loss: ``-(log_pi_mean · advantages.detach()).mean()`` over B*K.
7. Backward → grad-clip → step.
8. ``model.memory.update_plasticity(per_token_ce)`` — fires the
   walker's plastic update (snapshot + commit + rebuild active delta).
   This is structurally identical to Phase 1's call site; the walker
   does not know which phase it is in, only what surprise signal it
   was handed. Without this call, ``E_bias_flat`` is frozen during
   Phase 2, which silently disables long-term plastic encoding.
9. ``model.detach_memory()``.

BS_outer back-compat: if the caller passes prefix_ids[1, T_pre] +
reference_cont as a Tensor (legacy single-prefix shape), B=1 is
inferred and behavior matches the pre-batched implementation.

BS_outer rationale: K is a variance-reduction lever (group size). B is
a parallelism lever. The K-sweep flatness observed at B=1 (K=4: 0.50,
K=8: 0.48, K=16: 0.45 steps/s) means GPU has compute headroom; growing
B fills that headroom with more independent prompts, near-linear
speedup until L2/HBM saturation. Walker memory state is sized [B*K, ...]
for the duration of the step — each of the B prefixes gets its own
K-block of memory state, no cross-talk.

Trainable surface: whatever the caller put in their optimizer. The
``freeze_all_but_E_bias_and_neuromod()`` helper on the model sets a
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

from src.graph_walker.pretrained.integrated_lm import IntegratedLM
from src.graph_walker.pretrained.rollout import (
    autoregressive_rollout,
    replay_grpo_rollout,
    sample_grpo_rollout,
)


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
    eos_fraction: float = 0.0
    load_balance_loss: float = 0.0       # walker load-balance aux (pre-weight)
    # Per-group breakdown — populated when B>1. For B=1, each list has length 1.
    # Useful for the multi-session uniform-batched path which needs per-prompt
    # mean/std for honest logging (the global reward_mean above averages across
    # ALL B*K rollouts and obscures per-prompt variation).
    per_group_reward_mean: list[float] | None = None    # [B] per-prompt mean reward
    per_group_reward_std: list[float] | None = None     # [B] per-prompt std
    # Per-group reward std clamping diagnostics — exposes the difference
    # between the unclamped std (what we report) and the clamped std
    # (what's actually used as the advantage divisor). When clamped > 0
    # but unclamped ≈ 0, advantages were dampened by `adv_std_floor` —
    # the actual learning signal is smaller than the reported std implies.
    per_group_reward_std_clamped: list[float] | None = None  # [B] clamped at adv_std_floor


def grpo_step(
    model: IntegratedLM,
    opt: torch.optim.Optimizer,
    *,
    prefix_ids: torch.Tensor,                                   # [B, T_pre], [1, T_pre], or [T_pre]
    reference_cont: torch.Tensor | list[torch.Tensor],          # [L] (B=1) or list of B tensors
    reward_fn: Callable[
        [torch.Tensor, torch.Tensor | list[torch.Tensor]], torch.Tensor,
    ] | None = None,
    num_rollouts: int = 8,
    gen_length: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    grad_clip: float = 1.0,
    adv_std_floor: float = 1e-3,
    eos_id: int | None = None,
    lm_context_window: int | None = None,
) -> GRPOStats:
    """One GRPO step on B independent (prefix, reference) pairs.

    Args:
        prefix_ids: [B, T_pre], [1, T_pre], or [T_pre]. B = BS_outer
            (independent prompts per step). All prefixes share T_pre.
        reference_cont: a single Tensor [L_ref] (B=1 back-compat path)
            or a list of B variable-length Tensors (each [L_b]).
        reward_fn: ``(generated [B*K, gen_length], reference) → rewards [B*K]``.
            Reference is forwarded as-is, so the reward function must
            handle the same Tensor-or-list polymorphism. Default is
            per-rollout fraction of generated head tokens that match
            the head of the corresponding reference.
        num_rollouts: K — group size for per-prompt advantage normalization.

    Layout convention:
        prefix b's K rollouts occupy slice [b*K : (b+1)*K] in all B*K
        tensors (sample, replay, rewards, advantages, log_pi). This is
        induced by ``repeat_interleave(K, dim=0)``.

    Notes:
        Routing during sampling is always Categorical (phase-2 hard
        sample) — the previous `gen_sample_routing=False` argmax mode
        was vestigial under the DeepSeek-style replay flow. Reward
        variance comes from the K rollouts diverging in the LM's token
        sampling and the walker's stochastic routing together; argmax
        routing during gen would collapse the latter source of variance.

        Per-group advantage normalization: rewards are reshaped to
        [B, K] and normalized per row (mean/std along dim=1). Cross-
        prefix advantage mixing would inject high-variance noise (one
        easy prompt's high reward would punish a hard prompt's mediocre
        reward), so each prompt's K rollouts only compete with each
        other.
    """
    if reward_fn is None:
        reward_fn = _default_token_match_reward

    # Normalize prefix shape → [B, T_pre].
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    if prefix_ids.dim() != 2:
        raise ValueError(
            f"prefix_ids must be [B, T_pre] or [T_pre]; got "
            f"{tuple(prefix_ids.shape)}"
    )
    B, T_pre = prefix_ids.shape
    K = num_rollouts
    if K < 2:
        raise ValueError(f"num_rollouts must be >= 2 for group-normalized GRPO; got {K}")

    # Normalize reference shape → either Tensor (passed through) or
    # list of length B. The reward function does its own polymorphism;
    # we mainly need to validate B matches.
    if isinstance(reference_cont, torch.Tensor):
        if B != 1:
            raise ValueError(
                f"prefix_ids has B={B} but reference_cont is a single "
                "Tensor (B=1 back-compat path). For B>1, pass "
                "reference_cont as a list[Tensor] of length B."
            )
    else:
        ref_list = list(reference_cont)
        if len(ref_list) != B:
            raise ValueError(
                f"prefix_ids has B={B} but reference_cont has "
                f"len={len(ref_list)}. They must match."
            )
        reference_cont = ref_list

    # Replicate each prefix K times in contiguous K-blocks → [B*K, T_pre].
    # repeat_interleave (NOT expand) so memory is independent across K.
    prefix_rep = prefix_ids.repeat_interleave(K, dim=0).contiguous()

    opt.zero_grad(set_to_none=True)

    # ---- Sample phase (no grad, capture routing trace) ----
    # sample_grpo_rollout transparently handles batch dim B*K — it just
    # begin_segment(bs=B*K) under the hood. EOS early-stop forces post-EOS
    # tokens to eos_id (preserves trace length); reward decoder strips.
    sampled = sample_grpo_rollout(
        model, prefix_rep,
        gen_length=gen_length,
        temperature=temperature,
        top_p=top_p,
        eos_id=eos_id,
        lm_context_window=lm_context_window,
    )

    # ---- Score phase ----
    # Reward contract: pass the GENERATION ONLY (sampled.new_tokens),
    # not the full prefix+gen. The reward function is responsible for
    # mapping K-block layout → per-block reference (BertCosineReward
    # does this automatically via repeat_interleave on B refs).
    rewards = reward_fn(sampled.new_tokens, reference_cont).to(
        next(model.parameters()).device,
    )
    BK = B * K
    if rewards.shape != (BK,):
        raise ValueError(
            f"reward_fn returned shape {tuple(rewards.shape)}, "
            f"expected ({BK},) for B={B}, K={K}"
        )

    # Per-group advantage normalization. Reshape [B*K] → [B, K], normalize
    # within each row, flatten back. dim=1 std uses unbiased=True (Bessel
    # correction); for K>=2 this matches the prior B=1 behavior exactly.
    r_grp = rewards.view(B, K)
    r_mean_grp = r_grp.mean(dim=1, keepdim=True)
    r_std_grp = r_grp.std(dim=1, keepdim=True).clamp(min=adv_std_floor)
    adv_grp = (r_grp - r_mean_grp) / r_std_grp                   # [B, K]
    advantages = adv_grp.view(BK)                                # [B*K]

    # ---- Replay phase (with grad, teacher-forced) ----
    # DeepSeek-style: re-forward the sampled trajectory with grad enabled,
    # using the captured routing trace as `replay_choices` per step. The
    # returned log_pi_mean carries grad and aggregates per-action log-π
    # over BOTH prefix and gen routing decisions — finer credit assignment
    # than the prior implementation, which only credited prefix routing.
    # `per_token_ce` is no-grad and feeds plasticity post-opt.step.
    replay = replay_grpo_rollout(
        model, sampled, lm_context_window=lm_context_window,
    )
    log_pi_mean = replay.log_pi                                   # [B*K] grad
    per_token_ce = replay.per_token_ce                            # [B*K, T_replay]
    if log_pi_mean.shape != (BK,):
        raise ValueError(
            f"replay log_pi_mean returned shape {tuple(log_pi_mean.shape)}, "
            f"expected ({BK},)"
        )

    # REINFORCE on the replay log-π × advantage. The replay log_pi_mean
    # is mean-normalized (sum / step_count) so gradient magnitudes stay
    # bounded as (T_pre + gen_length) scales. Mean over B*K reduces
    # variance proportionally to the parallelism — equivalent to the
    # prior K-only mean when B=1.
    loss = -(log_pi_mean * advantages.detach()).mean()

    # Anti-collapse load-balance auxiliary loss (same Switch-Transformer
    # pressure as Phase 1). The walker accumulates per-step lb-loss
    # across the replay forward (begin_segment in replay_grpo_rollout
    # resets the accumulator, so this is replay-only — sample-phase
    # accumulation under no_grad is correctly discarded). Without this
    # in Phase 2, any column-spreading achieved in Phase 1 can be
    # undone by REINFORCE collapsing onto a few high-reward routes.
    lb_loss_value: float = 0.0
    if model.memory is not None and model.memory.cfg.lambda_balance > 0.0:
        lb = model.memory.consume_load_balance_loss()
        if lb is not None:
            loss = loss + model.memory.cfg.lambda_balance * lb.float()
            lb_loss_value = float(lb.detach())

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for _, p in model.trainable_parameters()],
        grad_clip,
    )
    opt.step()

    # ---- Unified plasticity (mirrors Phase 1) ----
    # The walker's plasticity behavior is phase-agnostic: it always fires
    # once per training step. What varies is the surprise SOURCE — here,
    # CE against the replay sequence (prefix = ground-truth cumulative
    # prior; gen = the model's own sampled tokens, i.e. self-entropy at
    # sample time). Without this, E_bias_flat stays frozen during all of
    # Phase 2 and the long-term plastic encoding pathway is disabled.
    if model.memory is not None:
        model.memory.update_plasticity(per_token_ce)
    model.detach_memory()

    # Generation diversity: how many UNIQUE generations across all B*K
    # rollouts? Reported as a global indicator. (Per-group uniqueness
    # could collapse to 1 within a group while still being diverse
    # across groups — but for the early-warning use case the global
    # count is sufficient.)
    gen_unique_count = int(sampled.new_tokens.unique(dim=0).shape[0])

    # Per-group reward stats — same r_grp/r_mean_grp/r_std_grp computed
    # earlier for advantage normalization, just exposed in the return.
    # Both unclamped (true reward dispersion) AND clamped (the actual
    # divisor used in advantage normalization). When unclamped < clamped,
    # advantages were attenuated by `adv_std_floor` — the learning signal
    # is smaller than the unclamped std implies.
    per_group_reward_mean = r_mean_grp.detach().squeeze(-1).tolist()  # [B]
    per_group_reward_std = r_grp.std(dim=1).detach().tolist()         # [B] unclamped
    per_group_reward_std_clamped = (
        r_std_grp.detach().squeeze(-1).tolist()
    )                                                                  # [B] clamped
    eos_fraction = 0.0
    if sampled.eos_step is not None:
        eos_fraction = float((sampled.eos_step < gen_length).float().mean().detach())

    return GRPOStats(
        loss=float(loss.detach()),
        reward_mean=float(rewards.mean().detach()),
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
        eos_fraction=eos_fraction,
        load_balance_loss=lb_loss_value,
        per_group_reward_mean=per_group_reward_mean,
        per_group_reward_std=per_group_reward_std,
        per_group_reward_std_clamped=per_group_reward_std_clamped,
    )


def _default_token_match_reward(
    generated: torch.Tensor,                                     # [B*K, gen_length]
    reference: torch.Tensor | list[torch.Tensor],                # [L] (B=1) or list of B
) -> torch.Tensor:
    """Placeholder reward: fraction of generated head tokens that match
    the head of the corresponding reference (hard token-match). Use as
    a smoke baseline only — real runs need a paraphrase-tolerant
    signal (BERT-cosine).

    Contract (2026-05-06): caller passes the GENERATION ONLY (sampled.
    new_tokens) and either a single Tensor (B=1) or a list of B
    tensors (B>1, K-block layout). Each gen[b*K:(b+1)*K] is scored
    against reference[b] on the min(gen_length, L_b) HEAD of each.
    """
    if isinstance(reference, torch.Tensor):
        refs: list[torch.Tensor] = [reference]
    else:
        refs = list(reference)
    B = len(refs)
    BK = generated.shape[0]
    if BK % B != 0:
        raise ValueError(
            f"generated.shape[0]={BK} not divisible by B={B}"
        )
    K = BK // B
    rewards = torch.zeros(BK, device=generated.device)
    for b in range(B):
        ref_b = refs[b]
        L = min(int(generated.shape[1]), int(ref_b.shape[0]))
        if L == 0:
            continue
        head = generated[b * K:(b + 1) * K, :L]
        ref_head = ref_b[:L].unsqueeze(0).to(head.device)
        rewards[b * K:(b + 1) * K] = (head == ref_head).float().mean(dim=1)
    return rewards


# ----------------------------------------------------------------------
# Multi-turn GRPO (Wave 4): session-aware, per-turn updates
# ----------------------------------------------------------------------


@dataclass
class SessionGRPOStats:
    """Aggregate per-session GRPO step statistics.

    Single-turn ``GRPOStats`` reports per-step. Multi-turn yields one
    GRPO step PER ASSISTANT TURN, so we report aggregates across turns.
    """
    n_assistant_turns: int
    total_session_tokens: int            # sum of all turn lengths (input observed)
    per_turn_reward_mean: list[float]    # length n_assistant_turns
    per_turn_reward_std: list[float]
    per_turn_grad_norm: list[float]
    per_turn_loss: list[float]
    per_turn_unique_count: list[int]     # # unique gens among K
    eos_fraction: float                  # fraction of K rollouts that hit EOS at all


def grpo_session_step(
    model: IntegratedLM,
    opt: torch.optim.Optimizer,
    *,
    session=None,                                                # MultiTurnSession or None
    sessions=None,                                               # list[MultiTurnSession] | None
    reward_fn: Callable[
        [torch.Tensor, torch.Tensor | list[torch.Tensor]], torch.Tensor,
    ] | None = None,
    num_rollouts: int = 8,
    max_response_len: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    grad_clip: float = 1.0,
    adv_std_floor: float = 1e-3,
    eos_id: int | None = None,
    skip_first_user_only_turns: bool = True,
    max_prior_tokens: int | None = None,
    lm_context_window: int | None = None,
) -> SessionGRPOStats:
    """One multi-turn GRPO session: walks turn-by-turn through a chat
    session, doing per-assistant-turn GRPO updates.

    Per assistant turn:
    1. Build cumulative prior context = concat of all turn ids up to (not
       including) this turn. Replicate K times → [K, prior_len].
    2. ``sample_grpo_rollout(prefix=cumulative_prior, gen_length=max_response_len,
       eos_id=eos_id)`` → K sampled assistant continuations.
    3. ``reward_fn(sampled.new_tokens, ground_truth_assistant_ids) → rewards [K]``.
       Standard BERT-cosine on the assistant's reference for that turn.
    4. Group-relative advantage: ``(r - mean(r)) / max(std(r), adv_std_floor)``.
    5. Replay with grad → log_pi_mean [K].
    6. Loss = -(log_pi_mean × advantage.detach()).mean(). backward, grad-clip, step.
    7. Detach memory. Append GROUND-TRUTH assistant ids to cumulative
       prior (the K diverged samples are discarded — this is the
       aligned-trajectory protocol's re-converge step).
    8. Move to next turn.

    User and system turns are accumulated into the cumulative prior
    without any separate forward/loss step — they're consumed during
    the next assistant turn's prefix pass.

    Args:
        session: a ``MultiTurnSession`` (from ``src.data.wildchat_loader``).
        reward_fn: standard ``(generated [K, L], reference [L]) -> rewards [K]``.
            For MT, the reference is each assistant turn's ground-truth
            ids (passed as a single Tensor — internally B=1 per turn).
        num_rollouts: K group size (same as single-turn).
        max_response_len: cap on generated tokens per assistant turn.
            Each rollout stops earlier if it emits ``eos_id``.
        eos_id: if set, rollouts stop emitting at EOS (rest of trace is
            EOS-pad). Recommended: pass the LM's ``tokenizer.eos_token_id``.
        skip_first_user_only_turns: if True (default), if the session's
            first turn(s) before any assistant turn are user/system, the
            first assistant turn's prefix pass STILL forwards them — so
            this flag is mostly informational. Kept for explicitness.
        max_prior_tokens: if set, the cumulative prior is truncated to
            its last N tokens before each assistant turn's prefix pass.
            Useful as a memory cap when sessions have very long context;
            None = no truncation. Note: truncating breaks the walker's
            full-history-as-context invariant, so leave it off unless
            VRAM forces the issue.

    Returns:
        SessionGRPOStats with per-turn reward / grad / loss arrays.
    """
    if reward_fn is None:
        reward_fn = _default_token_match_reward
    K = num_rollouts
    if K < 2:
        raise ValueError(f"num_rollouts must be >= 2 for group-normalized GRPO; got {K}")
    device = next(model.parameters()).device

    # ---- Multi-session dispatch ----
    # Accept either `session=<one>` (legacy) or `sessions=[s1, s2, ...]`.
    if sessions is not None:
        if session is not None:
            raise ValueError("Pass `session` OR `sessions`, not both")
        sessions_list = list(sessions)
    elif session is not None:
        sessions_list = [session]
    else:
        raise ValueError("Must pass either `session` or `sessions`")

    if len(sessions_list) == 0:
        raise ValueError("`sessions` is empty")

    # Detect uniform Wave-3-shape batching: all sessions are 2-turn
    # (user, assistant) AND user-turn ids share the same length. This
    # is the case where we can batch B*K rollouts in one shot and
    # recover BS_outer's ~5× speedup. Otherwise (variable-shape Wave 4),
    # process sessions sequentially.
    if len(sessions_list) > 1 and _can_batch_as_single_turn(sessions_list):
        return _grpo_session_step_uniform_batched(
            model, opt, sessions_list,
            reward_fn=reward_fn,
            num_rollouts=K,
            max_response_len=max_response_len,
            temperature=temperature, top_p=top_p,
            grad_clip=grad_clip, adv_std_floor=adv_std_floor,
            eos_id=eos_id, lm_context_window=lm_context_window,
            max_prior_tokens=max_prior_tokens,
        )

    # Sequential per-session fallback (multi-turn Wave 4, or single session).
    if len(sessions_list) > 1:
        return _grpo_session_step_sequential(
            model, opt, sessions_list,
            reward_fn=reward_fn,
            num_rollouts=K,
            max_response_len=max_response_len,
            temperature=temperature, top_p=top_p,
            grad_clip=grad_clip, adv_std_floor=adv_std_floor,
            eos_id=eos_id, max_prior_tokens=max_prior_tokens,
            lm_context_window=lm_context_window,
        )

    # Single-session path (existing logic below).
    session = sessions_list[0]

    # Verify session has at least one assistant turn to score.
    has_assistant = any(t.role == "assistant" for t in session.turns)
    if not has_assistant:
        return SessionGRPOStats(
            n_assistant_turns=0,
            total_session_tokens=session.total_tokens,
            per_turn_reward_mean=[], per_turn_reward_std=[],
            per_turn_grad_norm=[], per_turn_loss=[],
            per_turn_unique_count=[], eos_fraction=0.0,
        )

    # Cumulative prior accumulator. Lives on CPU as a python list of ints
    # to avoid repeated tensor concatenation; converted to a [K, L] tensor
    # at each assistant-turn boundary.
    cumulative_prior: list[int] = []

    per_turn_reward_mean: list[float] = []
    per_turn_reward_std: list[float] = []
    per_turn_grad_norm: list[float] = []
    per_turn_loss: list[float] = []
    per_turn_unique_count: list[int] = []
    eos_hit_count = 0
    eos_total = 0

    for t, turn in enumerate(session.turns):
        if turn.role != "assistant":
            cumulative_prior.extend(turn.ids.cpu().tolist())
            continue

        if not cumulative_prior:
            # Degenerate: assistant-first session with no prior context.
            # Skip — there's nothing to predict from.
            cumulative_prior.extend(turn.ids.cpu().tolist())
            continue

        # Optional prior-truncation cap.
        prior = cumulative_prior
        if max_prior_tokens is not None and len(prior) > max_prior_tokens:
            prior = prior[-max_prior_tokens:]

        prefix_ids = torch.tensor(
            [prior], dtype=torch.long, device=device,
        ).expand(K, -1).contiguous()                              # [K, prior_len]
        reference_ids = turn.ids.to(device)                       # [L_ref]

        # ---- Sample (no grad, capture trace) ----
        opt.zero_grad(set_to_none=True)
        sampled = sample_grpo_rollout(
            model, prefix_ids,
            gen_length=max_response_len,
            temperature=temperature, top_p=top_p,
            eos_id=eos_id,
            lm_context_window=lm_context_window,
        )

        # ---- Score ----
        rewards = reward_fn(sampled.new_tokens, reference_ids).to(device)
        if rewards.shape != (K,):
            raise ValueError(
                f"reward_fn returned {tuple(rewards.shape)}, expected ({K},)"
            )
        r_mean = rewards.mean()
        r_std = rewards.std().clamp(min=adv_std_floor)
        advantages = (rewards - r_mean) / r_std                   # [K]

        # ---- Replay with grad ----
        replay = replay_grpo_rollout(
            model, sampled, lm_context_window=lm_context_window,
        )
        log_pi_mean = replay.log_pi                               # [K]
        per_token_ce = replay.per_token_ce                        # [K, T_replay]
        if log_pi_mean.shape != (K,):
            raise ValueError(
                f"replay log_pi_mean returned shape {tuple(log_pi_mean.shape)}, "
                f"expected ({K},)"
            )
        loss = -(log_pi_mean * advantages.detach()).mean()

        # Anti-collapse load-balance aux (same as grpo_step's loss).
        if model.memory is not None and model.memory.cfg.lambda_balance > 0.0:
            lb = model.memory.consume_load_balance_loss()
            if lb is not None:
                loss = loss + model.memory.cfg.lambda_balance * lb.float()

        # ---- Backward + step ----
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in model.trainable_parameters()],
            grad_clip,
        )
        opt.step()

        # Unified plasticity (mirrors Phase 1; see grpo_step docstring).
        # Per-turn surprise covers this turn's prefix + gen — walker's
        # plastic update commits the active neuromod delta into
        # E_bias_flat and rebuilds a fresh delta for the next turn.
        if model.memory is not None:
            model.memory.update_plasticity(per_token_ce)
        model.detach_memory()

        # ---- Stats ----
        per_turn_reward_mean.append(float(r_mean.detach()))
        per_turn_reward_std.append(float(rewards.std().detach()))
        per_turn_grad_norm.append(float(grad_norm) if isinstance(grad_norm, torch.Tensor)
                                   else float(grad_norm))
        per_turn_loss.append(float(loss.detach()))
        per_turn_unique_count.append(
            int(sampled.new_tokens.unique(dim=0).shape[0])
        )
        if sampled.eos_step is not None:
            eos_hit = int((sampled.eos_step < max_response_len).sum())
            eos_hit_count += eos_hit
            eos_total += K

        # ---- Re-converge: append GROUND-TRUTH assistant tokens ----
        # The K sampled completions are discarded; the next turn's prefix
        # pass will rebuild walker state from the new cumulative prior
        # (which contains the ground-truth assistant turn t).
        cumulative_prior.extend(turn.ids.cpu().tolist())

    eos_fraction = (eos_hit_count / eos_total) if eos_total > 0 else 0.0

    return SessionGRPOStats(
        n_assistant_turns=len(per_turn_reward_mean),
        total_session_tokens=session.total_tokens,
        per_turn_reward_mean=per_turn_reward_mean,
        per_turn_reward_std=per_turn_reward_std,
        per_turn_grad_norm=per_turn_grad_norm,
        per_turn_loss=per_turn_loss,
        per_turn_unique_count=per_turn_unique_count,
        eos_fraction=eos_fraction,
    )


# ----------------------------------------------------------------------
# Multi-session helpers (Wave 3 uniform batched + Wave 4 sequential)
# ----------------------------------------------------------------------


def _can_batch_as_single_turn(sessions) -> bool:
    """True if every session is structurally a single Q/A turn AND
    they all share the same user-prefix length. This is the Wave 3
    case: passphrase chat sessions are always 2-turn (user prefix +
    assistant ref), with the prefix padded/trimmed to a fixed T_pre.

    When True, we can use the BS_outer batched path (B*K parallel
    rollouts) for ~5× session-throughput speedup vs sequential.

    Wave 4 sessions have variable N_assistant_turns and variable
    per-turn lengths — they fall back to sequential processing.
    """
    if len(sessions) < 2:
        return False
    first = sessions[0]
    if len(first.turns) != 2:
        return False
    if first.turns[0].role != "user" or first.turns[1].role != "assistant":
        return False
    T_pre_ref = int(first.turns[0].ids.numel())
    for s in sessions[1:]:
        if len(s.turns) != 2:
            return False
        if s.turns[0].role != "user" or s.turns[1].role != "assistant":
            return False
        if int(s.turns[0].ids.numel()) != T_pre_ref:
            return False
    return True


def _grpo_session_step_uniform_batched(
    model, opt, sessions, *,
    reward_fn, num_rollouts, max_response_len,
    temperature, top_p, grad_clip, adv_std_floor,
    eos_id, lm_context_window, max_prior_tokens,
) -> SessionGRPOStats:
    """B sessions, each with one user prefix + one assistant ref.
    All prefixes share the same length, so we can stack into [B, T_pre]
    and run one BS_outer-shaped grpo_step internally for B*K parallel
    rollouts. This recovers Wave 3's pre-unification BS_outer speedup.
    """
    B = len(sessions)
    K = num_rollouts
    device = next(model.parameters()).device

    prefix_list = [s.turns[0].ids.to(device) for s in sessions]
    if max_prior_tokens is not None:
        prefix_list = [p[-max_prior_tokens:].contiguous() for p in prefix_list]
    prefix_ids = torch.stack(prefix_list, dim=0)             # [B, T_pre]
    references = [s.turns[1].ids.to(device) for s in sessions]

    gstats = grpo_step(
        model, opt,
        prefix_ids=prefix_ids,
        reference_cont=references,
        reward_fn=reward_fn,
        num_rollouts=K,
        gen_length=max_response_len,
        temperature=temperature,
        top_p=top_p,
        grad_clip=grad_clip,
        adv_std_floor=adv_std_floor,
        eos_id=eos_id,
        lm_context_window=lm_context_window,
    )

    total_tokens = sum(int(s.total_tokens) for s in sessions)
    # Per-group reward stats are populated by grpo_step (one entry per
    # session in the batch); fall back to global mean if missing
    # (shouldn't happen post-2026-05-06 update).
    pg_reward_mean = (
        gstats.per_group_reward_mean
        if gstats.per_group_reward_mean is not None
        else [gstats.reward_mean] * B
    )
    pg_reward_std = (
        gstats.per_group_reward_std
        if gstats.per_group_reward_std is not None
        else [gstats.reward_std] * B
    )
    # grad_norm and loss are computed once over the whole B*K batch (one
    # backward / opt.step per outer call), so they're shared across the
    # B "turns" we're reporting.
    return SessionGRPOStats(
        n_assistant_turns=B,
        total_session_tokens=total_tokens,
        per_turn_reward_mean=pg_reward_mean,
        per_turn_reward_std=pg_reward_std,
        per_turn_grad_norm=[gstats.grad_norm] * B,
        per_turn_loss=[gstats.loss] * B,
        per_turn_unique_count=[gstats.gen_unique_count] * B,
        eos_fraction=gstats.eos_fraction,
    )


def _grpo_session_step_sequential(
    model, opt, sessions, *,
    reward_fn, num_rollouts, max_response_len,
    temperature, top_p, grad_clip, adv_std_floor,
    eos_id, max_prior_tokens, lm_context_window,
) -> SessionGRPOStats:
    """Process B sessions sequentially. No cross-session batching, but
    keeps the unified API. Each session does its own per-assistant-turn
    GRPO updates internally. Used for variable-shape (Wave 4) sessions.
    """
    n_assistant_turns_total = 0
    total_session_tokens = 0
    per_turn_reward_mean: list[float] = []
    per_turn_reward_std: list[float] = []
    per_turn_grad_norm: list[float] = []
    per_turn_loss: list[float] = []
    per_turn_unique_count: list[int] = []
    eos_hit_total = 0.0
    eos_count_total = 0

    for s in sessions:
        sub = grpo_session_step(
            model, opt,
            session=s,
            reward_fn=reward_fn,
            num_rollouts=num_rollouts,
            max_response_len=max_response_len,
            temperature=temperature, top_p=top_p,
            grad_clip=grad_clip, adv_std_floor=adv_std_floor,
            eos_id=eos_id, max_prior_tokens=max_prior_tokens,
            lm_context_window=lm_context_window,
        )
        n_assistant_turns_total += sub.n_assistant_turns
        total_session_tokens += sub.total_session_tokens
        per_turn_reward_mean.extend(sub.per_turn_reward_mean)
        per_turn_reward_std.extend(sub.per_turn_reward_std)
        per_turn_grad_norm.extend(sub.per_turn_grad_norm)
        per_turn_loss.extend(sub.per_turn_loss)
        per_turn_unique_count.extend(sub.per_turn_unique_count)
        if sub.n_assistant_turns > 0:
            eos_hit_total += sub.eos_fraction * sub.n_assistant_turns
            eos_count_total += sub.n_assistant_turns

    eos_fraction = (
        eos_hit_total / eos_count_total if eos_count_total > 0 else 0.0
    )
    return SessionGRPOStats(
        n_assistant_turns=n_assistant_turns_total,
        total_session_tokens=total_session_tokens,
        per_turn_reward_mean=per_turn_reward_mean,
        per_turn_reward_std=per_turn_reward_std,
        per_turn_grad_norm=per_turn_grad_norm,
        per_turn_loss=per_turn_loss,
        per_turn_unique_count=per_turn_unique_count,
        eos_fraction=eos_fraction,
    )
