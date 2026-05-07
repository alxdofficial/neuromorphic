"""Autoregressive rollout primitive for graph_walker + frozen Llama.

Used by phase-2 GRPO and inference. Given a prefix `[BS, T_pre]` and a
desired generation length, returns generated token ids `[BS, T_pre + L]`
and (optionally) the per-step logits.

Memory semantics during rollout:
- Plasticity does NOT fire inside `walk_segment` — the walker is
  vocab-agnostic and surprise is supplied by the trainer post-backward
  via `model.memory.update_plasticity(per_token_ce)`.
- The walker's plasticity behavior is INDEPENDENT of training phase.
  It always fires once per training step at mod_period boundaries.
  What varies between phases is the SURPRISE TARGET:
    * Phase 1 (parallel teacher-forced):  CE against ground-truth tokens
    * Phase 2 (GRPO sample/replay):       CE against the replay sequence
                                           (= prefix ground-truth + the
                                           model's own sampled gen tokens)
    * AR free-generation / inference:     surprise=None; plasticity is
                                           skipped (see `update_plasticity`).
- In phase 2 mode, routing is hard Categorical and `log_pi_sum`
  accumulates over routing decisions for REINFORCE. The replay forward
  also captures logits for the per-token CE that drives plasticity post-
  opt.step (see `replay_grpo_rollout` → `ReplayResult`).

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

from src.graph_walker.pretrained.integrated_lm import IntegratedLM
from src.graph_walker.routing import StepRoutingChoices


@dataclass
class RolloutOutput:
    generated: torch.Tensor          # [BS, T_pre + L] full sequence
    new_tokens: torch.Tensor         # [BS, L] generated tail
    log_pi_mean: torch.Tensor | None # [BS] MEAN log π over routing during prefix
                                      # (= sum / step_count for proper REINFORCE
                                      # normalization — see consume_log_pi_mean)
    last_logits: torch.Tensor | None # [BS, vocab] for diagnostics


@dataclass
class ReplayResult:
    """Output of `replay_grpo_rollout` — DeepSeek-style replay phase.

    Carries both the REINFORCE objective (``log_pi`` with grad) and the
    plasticity surprise signal (``per_token_ce`` no_grad). Phase-2 GRPO's
    ``grpo_step`` uses the former for backward and the latter for an
    ``update_plasticity`` call after ``opt.step()`` — same pattern as
    Phase 1, just with a different surprise target.
    """
    log_pi: torch.Tensor          # [B*K] mean log-π over routing decisions; grad attached
    per_token_ce: torch.Tensor    # [B*K, T_replay] no-grad CE for plasticity surprise


def _per_token_ce_chunked(
    logits: torch.Tensor,            # [B*K, T, V]
    targets: torch.Tensor,           # [B*K, T] long
    chunk_size: int = 256,
    ignore_token_id: int | None = None,
) -> torch.Tensor:
    """Memory-bounded per-token CE.

    At production scales (V=128256 for Llama-3.2, T_replay~2K, B*K~64) a
    single-shot ``F.cross_entropy(reduction='none')`` materializes a
    float32 logits view that peaks ~30 GB. Chunking along T keeps the
    transient float-conversion + softmax bounded to roughly
    ``BK · chunk_size · V · 4`` bytes per chunk (~16 GB at chunk=256,
    BK=64, V=128k — adjustable downward if needed).

    ``ignore_token_id``: if set, positions whose target equals this id
    have their CE forced to 0 — used to mask Wave 3 PAD tokens (the
    silent stretch between filler and question; predicting a specific
    pad token isn't meaningful surprise) so they don't pollute the
    walker's plasticity signal. Pass the tokenizer's ``pad_token_id``.

    Returned tensor is detached and on the same device as logits.
    """
    BK, T, V = logits.shape
    assert targets.shape == (BK, T), (
        f"targets {tuple(targets.shape)} mismatches logits T={T}, BK={BK}"
    )
    if T == 0:
        return torch.zeros(BK, 0, device=logits.device, dtype=torch.float32)
    out_chunks = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        log_chunk = logits[:, s:e, :].float()
        tgt_chunk = targets[:, s:e]
        ce = F.cross_entropy(
            log_chunk.reshape(-1, V),
            tgt_chunk.reshape(-1),
            reduction="none",
        ).reshape(BK, e - s)
        if ignore_token_id is not None:
            ce = torch.where(
                tgt_chunk == ignore_token_id,
                torch.zeros_like(ce),
                ce,
            )
        out_chunks.append(ce.detach())
    return torch.cat(out_chunks, dim=1)


@dataclass
class GRPOSampledRollout:
    """Output of `sample_grpo_rollout` — DeepSeek-style sample phase.

    Captured under no_grad: the trajectory itself plus the routing trace
    that the walker took to produce it. The trace is consumed by
    `replay_grpo_rollout` for a teacher-forced re-forward with grad.
    """
    generated: torch.Tensor                   # [K, T_pre + L] full sequence
    new_tokens: torch.Tensor                  # [K, L] generated tail
    prefix_ids: torch.Tensor                  # [K, T_pre] prefix only (saved for replay)
    routing_trace: list[StepRoutingChoices]   # length T_pre + L (one per walker step)
    last_logits: torch.Tensor | None          # [K, vocab] for diagnostics
    eos_step: torch.Tensor | None = None      # [K] long; first step a rollout emitted EOS,
                                              # or gen_length if it never did. Useful for
                                              # reward-decoders that want to drop post-EOS
                                              # padding from the scored continuation.


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


def sample_grpo_rollout(
    model: IntegratedLM,
    prefix_ids: torch.Tensor,        # [BS, T_pre] where BS = B * K post-BS_outer
    *,
    gen_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    lm_context_window: int | None = None,
) -> GRPOSampledRollout:
    """DeepSeek-style sample phase: full prefix + AR gen under no_grad,
    capturing the walker's per-step routing trace.

    Batch convention: this function is batch-dim-agnostic. ``prefix_ids``
    has the outer batch already laid out by the caller. Pre-BS_outer the
    convention was [K, T_pre] (single prefix replicated K times); post-
    BS_outer it's [B*K, T_pre] (B distinct prefixes each replicated K
    times in contiguous K-blocks). The walker's per-batch routing trace
    is captured in whatever order the caller laid out.

    Returned `routing_trace` has length `T_pre + gen_length - 1` — one
    `StepRoutingChoices` per walker step that ACTUALLY contributes to a
    sampled token. The walker step processing the LAST sampled token is
    intentionally skipped: its readout would be used to compute logits
    at position L+1, which is not in the trajectory and not seen by
    `reward_fn`. Crediting that routing decision under REINFORCE would
    be non-causal noise, so `sample_grpo_rollout` stops forwarding after
    the L-1th gen token (the L-th token is sampled and included in
    `generated` for the reward function, but never forwarded through
    the walker).

    `_freeze_plasticity_ctx` is intentionally NOT used here. Under the
    external-surprise design, plasticity only fires via
    `update_plasticity` (called externally post-backward); forward-only
    paths never trigger plasticity. The old freeze-by-mutating-mod_period
    was a relic from an earlier design and silently corrupts the
    `_active_neuromod_delta` rebuild that runs at TBPTT block boundaries (the
    rebuild reads `cfg.mod_period` for co-visit normalization).

    Routing during sample is gated on `model.memory.training` only —
    NOT `model.training` — so the host LM stays in eval mode (host
    dropout disabled, batchnorm running stats untouched). This keeps the
    LM's hidden-state stream identical between sample and replay.

    EOS early-stop semantics (when ``eos_id`` is provided): once a rollout
    emits ``eos_id``, all subsequent sampled tokens for that rollout are
    forced to ``eos_id`` (post-EOS padding). The walker still forwards
    these padding tokens — so the routing trace length is preserved for
    constant-length replay — but the BERT reward decoder strips them via
    ``skip_special_tokens=True``. The ``eos_step`` field of the returned
    ``GRPOSampledRollout`` records the first EOS position per rollout.

    If ``eos_id`` is None, no early-stop is applied (legacy behavior:
    every rollout generates exactly ``gen_length`` tokens regardless).

    ``lm_context_window`` (when set, < T_pre): two-phase forward to
    decouple walker's effective context (full prefix) from the LM's
    (last ``lm_context_window`` tokens):
      1. Phase 1 — forward ``prefix_ids[:, :T_pre - lm_context_window]``
         through model with ``use_cache=False``. Walker advances state
         through the early portion. LM forwards too (necessary because
         walker.walk_segment requires LM hidden states), but its
         outputs / KV cache are discarded.
      2. Phase 2 — forward the last ``lm_context_window`` prefix tokens
         with ``use_cache=True``. Walker continues from its phase-1 state
         (memory now reflects the FULL prefix). LM caches only these
         recent tokens.
      3. AR gen continues from phase-2's KV cache.

    From the LM's perspective, the prefix is just ``lm_context_window``
    tokens long during AR gen — it cannot attend back to the early
    portion. The walker, however, has integrated information from the
    entire prefix into its memory state. This forces the walker to
    actually carry information that the LM can no longer see directly.

    If ``lm_context_window`` is None or >= T_pre, single-phase forward
    is used (legacy behavior).
    """
    assert model.memory is not None, "rollout requires attached memory"
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device)
    K, T_pre = prefix_ids.shape

    saved_phase = model.current_phase
    saved_walker_training = model.memory.training
    model.current_phase = "phase2"
    try:
        model.begin_segment(bs=K)
        model.memory.start_capturing_routes()

        # Toggle ONLY the walker's training mode for phase-2 routing's
        # hard-Categorical sampling. `gumbel_top1_softmax(phase=phase2)`
        # gates on `training=True` from `_walker_step`'s `is_training
        # = self.training` read (where `self` is the walker). The host
        # LM stays in whatever mode the caller had it in.
        model.memory.train(True)

        # Per-rollout done flags + first-EOS-step. After a rollout
        # emits eos_id, its subsequent tokens are forced to eos_id (so
        # the routing trace length stays uniform across the K-batch).
        done = torch.zeros(K, dtype=torch.bool, device=device)
        eos_step = torch.full((K,), gen_length, dtype=torch.long, device=device)

        with torch.no_grad():
            # Prefix pass — captures T_pre routing choices.
            # Optional two-phase decoupling: walker absorbs full prefix
            # but LM only retains a sliding window of the last
            # lm_context_window tokens. See docstring.
            if (
                lm_context_window is not None
                and lm_context_window > 0
                and lm_context_window < T_pre
            ):
                early_len = T_pre - lm_context_window
                early_ids = prefix_ids[:, :early_len]
                recent_ids = prefix_ids[:, early_len:]
                # Phase 1: walker advances through early prefix; LM cache
                # discarded (use_cache=False).
                model(early_ids, use_cache=False)
                # Phase 2: LM caches recent prefix; walker continues.
                out = model(recent_ids, use_cache=True)
            else:
                out = model(prefix_ids, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]

            new_tokens: list[torch.Tensor] = []
            # Generate gen_length tokens, but only forward the first
            # (gen_length - 1) of them through the walker. The last
            # sampled token is added to `generated` for the reward
            # function but never seen by the walker, so its routing
            # decision (which would only affect logits at position L+1)
            # is never made and never credited under REINFORCE.
            for i in range(gen_length):
                tok = _sample_next_token(
                    last_logits, temperature, top_p,
                )
                # EOS early-stop: rollouts already done get force-fed
                # eos_id from here on. Their trace contributions during
                # the post-EOS suffix are still captured for length
                # uniformity; the reward decoder strips them.
                if eos_id is not None:
                    if done.any():
                        tok = torch.where(
                            done, torch.full_like(tok, eos_id), tok,
                        )
                    just_emitted = (tok == eos_id) & (~done)
                    if just_emitted.any():
                        eos_step = torch.where(
                            just_emitted,
                            torch.full_like(eos_step, i),
                            eos_step,
                        )
                        done = done | just_emitted
                new_tokens.append(tok)
                if i == gen_length - 1:
                    break
                # Optional fast-exit: if every rollout has emitted EOS,
                # further forwarding only emits eos_id and accumulates
                # routing trace under post-EOS padding. We still need
                # the trace length to be exactly T_pre + (gen_length-1)
                # for the replay step's per-step alignment, so we MUST
                # keep forwarding. (No early-break on all-done; eat the
                # cost. K is small, gen_length is bounded.)
                out = model(
                    tok.unsqueeze(-1),
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out.past_key_values
                last_logits = out.logits[:, -1, :]

        new_tokens_t = (
            torch.stack(new_tokens, dim=1) if new_tokens
            else torch.zeros(K, 0, dtype=torch.long, device=device)
        )
        generated = torch.cat([prefix_ids, new_tokens_t], dim=1)

        trace = model.memory.consume_routing_trace()
        if trace is None:
            raise RuntimeError(
                "sample_grpo_rollout produced no routing trace — "
                "start_capturing_routes() must have failed to arm."
            )
        # Sanity: trace length = T_pre (prefix) + (gen_length - 1)
        # gen-token forwards. The L-th gen token is sampled but never
        # forwarded, so it contributes no walker step.
        expected = T_pre + max(gen_length - 1, 0)
        assert len(trace) == expected, (
            f"trace length {len(trace)} != T_pre + (gen_length-1) "
            f"({expected}) — did capture get armed/disarmed mid-rollout?"
        )

        return GRPOSampledRollout(
            generated=generated,
            new_tokens=new_tokens_t,
            prefix_ids=prefix_ids,
            routing_trace=trace,
            last_logits=last_logits.detach(),
            eos_step=eos_step if eos_id is not None else None,
        )
    finally:
        model.current_phase = saved_phase
        model.memory.train(saved_walker_training)
        if model.memory is not None:
            model.memory.phase = saved_phase
            # Defensive: ensure the capture buffer is cleared even if an
            # exception bypassed `consume_routing_trace`.
            if model.memory._captured_routes is not None:
                model.memory._captured_routes = None


def replay_grpo_rollout(
    model: IntegratedLM,
    sampled: GRPOSampledRollout,
    *,
    lm_context_window: int | None = None,
    ignore_token_id: int | None = None,
) -> ReplayResult:
    """DeepSeek-style replay phase: teacher-forced re-forward with grad.

    Concatenates prefix + sampled-tokens-EXCEPT-LAST → [K, T_pre + L_gen - 1]
    and runs one parallel forward through the model. The walker uses
    the saved routing trace as `replay_choices` per step, accumulating
    per-action log-π with grad.

    Returns ``ReplayResult(log_pi, per_token_ce)`` where:
      - ``log_pi``: [B*K] aggregated log-π over routing steps, with grad.
      - ``per_token_ce``: [B*K, T_pre + L_gen - 1] no-grad CE — Llama's
        next-token loss against the replay sequence at every position.
        For prefix positions the target is the ground-truth cumulative
        prior; for gen positions the target is the model's own sampled
        token at that position. The trainer feeds this to
        ``model.memory.update_plasticity(per_token_ce)`` post-opt.step
        to drive the walker's plastic update — exactly mirroring Phase 1.

    Why drop the last sampled token: its walker step's routing decision
    can only influence logits at position L+1, which the trajectory does
    not include and the reward function does not see. Sampling skipped
    that walker step; replay must too, or trace lengths would mismatch.

    `model.preserve_autograd_graph()` is held across the replay forward.
    Without it, `walk_segment` would call `detach_state()` at every
    `tbptt_block`-token boundary inside the replay; that detach also
    rebuilds `_active_neuromod_delta` via `_begin_plastic_window`, which reads
    `cfg.mod_period`. Mutating mod_period to "freeze plasticity" (the
    old `_freeze_plasticity_ctx` trick) would silently corrupt the
    rebuilt active delta after the first block — replay log-probs after
    that point would be gradients for a different policy than the
    sampler used. preserve_graph=True bypasses both detach and rebuild.

    Routing-mode toggle is on the walker only (NOT the model), so the
    host LM stays in caller-set mode and replay sees the same hidden-
    state stream as sampling did.

    ``lm_context_window``: must match the value used during the
    corresponding ``sample_grpo_rollout`` call. Replay does a two-phase
    walker advance — phase 1 forwards the EARLY prefix with
    ``use_cache=False`` (so the LM's KV cache only retains the recent
    window, matching sample-time AR-gen attention scope); phase 2
    forwards the recent prefix + gen tokens with the LM's KV cache
    building.

    BOTH phases run under ``enable_grad`` + ``preserve_autograd_graph``
    so walker routing decisions across the FULL prefix get gradient
    credit. The LM is windowed (`use_cache=False` drops phase-1's KV
    cache before AR-gen continuation), but the walker is NOT — its
    routing decisions during the early prefix DO get REINFORCE gradient.

    This matters because the walker's "write" decisions (early prefix:
    routing tokens at the start of the cumulative prior into walker
    memory) are exactly what determines whether useful information is
    accessible for the gen at the end. Crediting only "read" decisions
    (recent prefix + gen) under REINFORCE leaves the encoding side
    unsupervised. Walker context is unwindowed; LM context is windowed.
    That's the actual decoupling design.
    """
    assert model.memory is not None, "replay requires attached memory"
    device = next(model.parameters()).device
    # Drop the last sampled token: walker step processing it was skipped
    # during sampling (see sample_grpo_rollout) and the trace does not
    # include it. Replay input length must match trace length.
    full_seq = sampled.generated.to(device)
    if sampled.new_tokens.numel() > 0:
        # sampled.generated = cat(prefix, new_tokens). Drop last new_token.
        replay_seq = full_seq[:, :-1]
    else:
        replay_seq = full_seq
    K = replay_seq.shape[0]
    # Plasticity surprise targets: replay[:, t]'s logits predict
    # full_seq[:, t+1]. So the per-position target sequence is full_seq
    # shifted by one. Length = T_pre + L_gen - 1 = replay_seq.shape[1].
    targets_full = full_seq[:, 1:1 + replay_seq.shape[1]]

    saved_phase = model.current_phase
    saved_walker_training = model.memory.training
    model.current_phase = "phase2"

    try:
        model.begin_segment(bs=K)
        model.memory.train(True)

        T_pre = sampled.prefix_ids.shape[1]
        full_trace = sampled.routing_trace
        two_phase = (
            lm_context_window is not None
            and lm_context_window > 0
            and lm_context_window < T_pre
        )
        if two_phase:
            # Two-phase replay. The split is purely about the LM's KV
            # cache scope (phase 1 use_cache=False so AR-gen at sample
            # time only attended to the recent window). The walker is
            # NOT windowed — both phases run under enable_grad +
            # preserve_autograd_graph so walker routing decisions across
            # the full prefix get gradient credit and `log_pi` covers
            # ALL routing decisions (early prefix + recent prefix +
            # gen). The walker's walk_segment requires the armed
            # trace length to match the input length, so we re-arm
            # per phase but DON'T consume the accumulator between
            # phases — log_pi sums across both.
            early_len = T_pre - lm_context_window
            early_seq = replay_seq[:, :early_len]
            recent_seq = replay_seq[:, early_len:]
            with torch.enable_grad(), model.preserve_autograd_graph():
                model.memory.arm_replay_trace(full_trace[:early_len])
                out_early = model(early_seq, use_cache=False)
                # Compute early-phase per-token CE while logits are alive,
                # then drop them so memory peak only briefly carries the
                # large [BK, early_len, V] tensor. The CE compute is
                # no_grad — surprise signal does not need to backprop.
                with torch.no_grad():
                    ce_early = _per_token_ce_chunked(
                        out_early.logits, targets_full[:, :early_len],
                        ignore_token_id=ignore_token_id,
                    )
                del out_early
                model.memory.arm_replay_trace(full_trace[early_len:])
                out_recent = model(recent_seq)
                with torch.no_grad():
                    ce_recent = _per_token_ce_chunked(
                        out_recent.logits, targets_full[:, early_len:],
                        ignore_token_id=ignore_token_id,
                    )
                del out_recent
                log_pi = model.memory.consume_log_pi_mean()
            per_token_ce = torch.cat([ce_early, ce_recent], dim=1)
        else:
            model.memory.arm_replay_trace(full_trace)
            with torch.enable_grad(), model.preserve_autograd_graph():
                out = model(replay_seq)
                with torch.no_grad():
                    per_token_ce = _per_token_ce_chunked(
                        out.logits, targets_full,
                        ignore_token_id=ignore_token_id,
                    )
                del out
                log_pi = model.memory.consume_log_pi_mean()

        if log_pi is None:
            raise RuntimeError(
                "replay produced no log_pi — replay_choices were "
                "either not consumed or routing was skipped during "
                "the teacher-forced forward."
            )
        if not log_pi.requires_grad:
            raise RuntimeError(
                "replay log_pi has no grad — re-evaluation forward "
                "ran without grad enabled, or routing scores were "
                "detached. REINFORCE backward will fail."
            )
        return ReplayResult(log_pi=log_pi, per_token_ce=per_token_ce)
    finally:
        model.current_phase = saved_phase
        model.memory.train(saved_walker_training)
        if model.memory is not None:
            model.memory.phase = saved_phase


def autoregressive_rollout(
    model: IntegratedLM,
    prefix_ids: torch.Tensor,        # [BS, T_pre]
    *,
    gen_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    phase: str = "phase2",           # "phase1" or "phase2" (sets walker.phase)
    grad_during_prefix: bool = True,
    grad_during_gen: bool = False,
    gen_sample_routing: bool = False,
    update_plasticity: bool = True,
) -> RolloutOutput:
    """Run a prefix + generation rollout.

    Args:
        model: IntegratedLM (memory must be attached).
        prefix_ids: [BS, T_pre] starting tokens; replicated by caller for
            K-rollout GRPO (caller is responsible for `prefix_ids = prefix.repeat(K, 1)`).
        gen_length: number of tokens to generate.
        temperature, top_p: sampling controls.
        phase: "phase2" for GRPO (hard Categorical routing + log_pi),
            "phase1" for diagnostic / forward-only inference.
        grad_during_prefix: whether to keep gradients for the prefix pass
            (True for GRPO; False for inference).
        grad_during_gen: whether to keep gradients during generation. GRPO
            uses False — the policy is captured in log_pi_mean from the
            prefix pass, generation is just sampling under it.
        gen_sample_routing: if True, walker routing during generation
            uses Categorical sampling (more reward variance across the K
            rollouts at the cost of noisier walker writes during gen). If
            False (default), routing during gen is argmax (deterministic
            given walker state). Sampled-routing-during-gen is an
            experimental variance-improvement lever — try it if K=8
            rollouts produce indistinguishable rewards.

            Note: even with gen_sample_routing=True, log_pi from gen-time
            routing is consumed-and-discarded after prefix; only the
            prefix log_pi participates in REINFORCE.
        update_plasticity: if True (default), at the end of the rollout
            compute per-token CE against the actual next-token sequence
            (= teacher-forced prefix tokens for the prefix portion +
            sampled gen tokens for the gen portion) and call
            ``model.memory.update_plasticity(per_token_ce)``. This
            honors the unified surprise rule: the model is always
            predicting the next token, and the walker's plastic state
            updates from whatever surprise the actual next token
            delivered — regardless of whether that token came from
            data, a sampler, or a user. Set False for diagnostic /
            snapshot-stable runs that must not mutate plastic state.
    """
    assert model.memory is not None, "rollout requires attached memory"
    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device)
    BS, T_pre = prefix_ids.shape
    # `model.forward()` propagates model.current_phase → memory.phase
    # before the closure runs. We also save/restore `model.training`:
    # the generation loop sets train(False) which would otherwise leak
    # into the next phase-1 cycle (compute_aux gates on `self.training`,
    # so a leaked False would silently skip aux loss).
    saved_phase = model.current_phase
    saved_training = model.training
    model.current_phase = phase

    try:
        model.begin_segment(bs=BS)

        # Prefix pass.
        prefix_ctx = (
            torch.enable_grad() if grad_during_prefix else torch.no_grad()
        )
        with prefix_ctx:
            model.train(grad_during_prefix)
            out = model(prefix_ids, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]
            # Hold a reference to the prefix logits so we can compute
            # per-token CE against the prefix-shifted-by-1 targets at
            # the end of the rollout (for plasticity surprise). Stored
            # under no_grad to avoid bloating the autograd graph; this
            # is a value-only use (plasticity surprise has no grad path
            # in the unified-plasticity contract).
            prefix_logits_for_surprise = out.logits.detach() if update_plasticity else None
            # Consume the MEAN (sum / step_count). The raw sum scales
            # with T_pre × n_hops × n_heads ≈ thousands of decisions per
            # rollout, producing gradient magnitudes thousands of times
            # too large for a sane LR. See consume_log_pi_mean docstring.
            log_pi_mean = model.memory.consume_log_pi_mean()

        # Generation pass — freeze plasticity to keep the policy fixed.
        gen_ctx = torch.enable_grad() if grad_during_gen else torch.no_grad()
        new_tokens: list[torch.Tensor] = []
        gen_logits: list[torch.Tensor] = []
        # train(True) during gen with phase=phase2 makes routing Categorical-
        # sample (more rollout variance). The accumulated log_pi from gen
        # is discarded post-rollout via the begin_segment reset on the
        # next training step — only the consumed-after-prefix log_pi_mean
        # participates in REINFORCE.
        gen_train_mode = bool(gen_sample_routing)
        with gen_ctx, _freeze_plasticity_ctx(model.memory):
            model.train(gen_train_mode)
            for _ in range(gen_length):
                # `last_logits` here predicts the token we're ABOUT to
                # sample. Capture it BEFORE sampling so we can compute
                # CE against the sampled token after the loop.
                if update_plasticity:
                    gen_logits.append(last_logits.detach())
                tok = _sample_next_token(last_logits, temperature, top_p)
                new_tokens.append(tok)
                out = model(
                    tok.unsqueeze(-1), past_key_values=past_kv, use_cache=True,
                )
                past_kv = out.past_key_values
                last_logits = out.logits[:, -1, :]
        new_tokens_t = torch.stack(new_tokens, dim=1) if new_tokens else \
                       torch.zeros(BS, 0, dtype=torch.long, device=device)
        full = torch.cat([prefix_ids, new_tokens_t], dim=1)

        # Unified plasticity at inference. The surprise rule is:
        #   "the model is always predicting the next token; the walker
        #    updates from CE against whatever the actual next token was
        #    — data, sample, user input, or tool output."
        # Here the prefix's actual next-token sequence is prefix_ids
        # itself (teacher-forced), and the gen's actual next-token
        # sequence is the just-sampled new_tokens (self-prediction
        # surprise = the model's distribution entropy at the chosen
        # samples). Concatenate, compute per-token CE under no_grad,
        # call update_plasticity. The walker doesn't know — and
        # doesn't need to know — which was data vs sample.
        if update_plasticity and model.memory is not None:
            with torch.no_grad():
                # Prefix CE: logits at position t predict prefix_ids[t+1].
                # We have T_pre logits and need T_pre - 1 CE values
                # (positions 0..T_pre-2 predicting 1..T_pre-1). The
                # T_pre-1 logit is captured separately in gen_logits[0]
                # which predicts the FIRST gen token.
                prefix_logits_shift = prefix_logits_for_surprise[:, :-1, :]
                prefix_targets = prefix_ids[:, 1:]
                # Gen CE: gen_logits[i] predicts new_tokens[:, i].
                # Stack to [BS, L_gen, V].
                if gen_logits:
                    gen_logits_t = torch.stack(gen_logits, dim=1)
                    gen_targets = new_tokens_t
                    full_logits = torch.cat(
                        [prefix_logits_shift, gen_logits_t], dim=1,
                    )                                          # [BS, T_pre-1+L_gen, V]
                    full_targets = torch.cat(
                        [prefix_targets, gen_targets], dim=1,
                    )                                          # [BS, T_pre-1+L_gen]
                else:
                    full_logits = prefix_logits_shift
                    full_targets = prefix_targets
                per_token_ce = _per_token_ce_chunked(
                    full_logits, full_targets,
                )
            model.memory.update_plasticity(per_token_ce)

        return RolloutOutput(
            generated=full,
            new_tokens=new_tokens_t,
            log_pi_mean=log_pi_mean,
            last_logits=last_logits.detach(),
        )
    finally:
        model.current_phase = saved_phase
        model.train(saved_training)
        # memory.phase was last set by model.forward() during the
        # generation loop. Sync it back so a follow-up direct memory
        # call (without going through model.forward) doesn't see a
        # leaked phase-2 setting.
        if model.memory is not None:
            model.memory.phase = saved_phase
