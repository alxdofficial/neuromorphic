"""Phase 2 (AR GRPO) trainer for Wave 3 (verifiable-reward) and Wave 4
(long-session). Implements GRPO: sample J responses per prompt, score each,
compute group-relative advantage, policy gradient (plan §4.7).

**Two-pass GRPO** (the plan §4.6 fallback, made default 2026-05-09 to
fix three correctness issues that the single-pass path had):

  Pass 1 — sampling (`_ar_sample_one`, no_grad).
    Prefill the prompt through memory (forward_window over each T_window
    window of the prompt, state carried), then generate AR token-by-token
    with `forward_window`. No autograd graph; cheap memory.

  Pass 2 — TF logp for backward (`_tf_compute_sample_logp`).
    TF-forward the full (prompt + sampled-response) sequence in
    T_window-sized windows with state carrying — this is structurally
    identical to W1/W2 forward, including memory state passing through
    `write_module`. Per-sample-token logp is computed under the same
    tempered softmax used during sampling. Single backward at the end
    accumulates policy gradient with full graph including the writer.

Why two-pass over single-pass:
  - Single-pass detached `new_states / hiddens` every generated token to
    bound activation memory. That worked but cut the gradient chain
    from rollout reward to write_module → writer never learned from
    GRPO. Two-pass only holds activations alive within pass 2's single
    TF forward (linear in sequence length, not token count × samples).
  - Single-pass had no prefill: long prompts (NarrativeQA 8K, WildChat
    priors 4-8K) entered AR with empty memory state. Two-pass gets
    prefill for free in both passes — pass 1 explicit, pass 2 by
    structure.
  - KV cache is fundamentally hard with per-window memory injection
    (memory readout changes per window, not per token). Two-pass moves
    the dominant gradient cost out of AR (which still has no KV cache)
    into TF (which never needed one). AR is still slow but it's now
    pure inference, can be optimized later.

The `Phase2Trainer` class is the proper training harness with grad
clipping, LR scheduling, metrics dict, and checkpoint integration. The
legacy `grpo_step` free function is preserved for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.rewards import compute_reward


@dataclass
class Phase2Metrics:
    policy_loss: float
    grad_norm: float
    lr: list[float]
    rewards: list[float]
    advantages: list[float]
    decoded: list[str]
    mean_response_len: float
    # IS-clipping diagnostics (Phase B): fraction of tokens whose ratio
    # was clipped, mean ratio. Both useful for catching policy drift.
    clip_fraction: float = 0.0
    mean_ratio: float = 1.0
    # KL diagnostics (Phase B): mean per-token KL to reference policy.
    # Zero when no ref_state set (KL term off).
    kl_to_ref: float = 0.0


def compute_grpo_advantages(rewards: Tensor, *, eps: float = 1e-6) -> Tensor:
    """DeepSeek-style group-relative advantage: (r_i - mean(r)) / std(r).

    Centers around 0 by construction; variance-normalized so loss scale
    is consistent across prompts with different reward magnitudes.

    Args:
        rewards: [J] reward per sample.
        eps:     numerical floor on std to avoid div-by-zero when all
                 rewards are equal (degenerate group → zero advantage,
                 zero gradient — that's the desired behavior).

    Returns:
        [J] advantages.
    """
    if rewards.numel() == 0:
        return rewards
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    return (rewards - mean_r) / std_r.clamp_min(eps)


class Phase2Trainer:
    """Phase 2 GRPO trainer.

    Usage:
        trainer = Phase2Trainer(model, optimizer, scheduler=...)
        for example in dataset:
            metrics = trainer.step(
                prompt_ids, num_samples=4, max_new_tokens=256,
                reward_kind=example["reward_kind"], gold=...,
                meta=example["meta"], tokenizer=tokenizer,
            )
    """

    def __init__(
        self,
        model: IntegratedLM,
        optimizer: Optimizer,
        *,
        scheduler: object | None = None,
        grad_clip: float | None = 1.0,
        clip_eps: float = 0.2,
        clip_eps_higher: float | None = None,
        kl_coef: float = 0.001,
    ):
        """
        Args:
            clip_eps: PPO importance-sampling ratio clip width (symmetric).
                Default 0.2 matches TRL/verl. Set to a large number (e.g. 10)
                AND set `clip_eps_higher` to the same to disable clipping.
            clip_eps_higher: optional asymmetric upper clip (DeepSeek-R1's
                `clip_higher` mode). When None, uses symmetric `clip_eps`.
                When set, ratio is clamped to [1 - clip_eps, 1 + clip_eps_higher],
                allowing the policy to grow more freely while still bounding
                downside drift. R1 used clip_eps_higher=10.
            kl_coef: weight on the `KL(π_θ || π_ref)` regularization term.
                Default 0.001 matches verl. Set to 0 to disable. Reference
                policy must be set via `set_reference_state()` for KL to
                fire.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.clip_eps = clip_eps
        self.clip_eps_higher = clip_eps_higher
        self.kl_coef = kl_coef
        self._step_count = 0
        # Reference policy snapshot (Phase B): set ONCE via
        # `set_reference_state()` before training starts. Used for KL
        # regularization to keep the policy near its Phase-1-end state.
        self.ref_state: dict[str, Tensor] | None = None

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        # N5 fix — persist ref_state across resume so the KL anchor doesn't
        # drift to the resumed policy. Without this, every full resume calls
        # `set_reference_state()` again and snapshots the resumed (i.e.,
        # mid-Phase-2-trained) weights as π_ref, defeating the point of KL
        # regularization.
        d: dict = {"step_count": self._step_count}
        if self.ref_state is not None:
            # Move to CPU for portability of the saved blob.
            d["ref_state"] = {n: t.detach().cpu() for n, t in self.ref_state.items()}
        return d

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]
        if "ref_state" in state:
            # Restore ref_state. The checkpoint blob has CPU tensors;
            # move them to the model's device.
            device = next(self.model.parameters()).device
            self.ref_state = {n: t.to(device) for n, t in state["ref_state"].items()}

    def set_reference_state(self) -> None:
        """Snapshot current trainable params as the reference policy π_ref.

        Call ONCE after loading the Phase 1 checkpoint and BEFORE starting
        Phase 2 training. The snapshot is used by `_compute_ref_logps` to
        compute the KL regularization term.

        Memory cost: ~size of trainable params (16.5M for medium config
        in fp32 = ~66 MB on GPU).
        """
        self.ref_state = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def _compute_ref_logps_batch(
        self,
        prompt_ids: Tensor,
        sample_ids_list: list[Tensor],
        *,
        temperature: float,
        pad_id: int,
        device: torch.device,
    ) -> list[Tensor | None]:
        """No-grad: compute REFERENCE-policy logps for K samples in ONE
        swap-in/swap-out cycle. Returns list of [n_sample_tokens] tensors
        (or None for empty samples).

        Phase D1 optimization: shared no_grad ref-policy prefill across
        all K samples (instead of K full prompt+sample forwards each
        with their own ref prefill). Same param-swap pattern as the old
        `_compute_ref_logps` but amortized over K rather than per-sample.
        """
        K = len(sample_ids_list)
        if self.ref_state is None or self.kl_coef == 0:
            return [None] * K

        # Save current trainable params.
        saved = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        results: list[Tensor | None] = []
        try:
            # Swap in ref params for the whole batch.
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(self.ref_state[name])

            # Shared ref-policy prefill (encoded once under ref weights).
            # Test-mode (no real Llama) skips shared prefill.
            ref_prefill = None
            if self.model.llama is not None:
                # _prefill_prompt now returns 5-tuple (N4 fix); we only
                # need the first 3 elements for prefill_state passing.
                _ref = self._prefill_prompt(
                    prompt_ids, pad_id=pad_id, device=device,
                )
                ref_prefill = _ref[:3]  # (states, hiddens, kv_cache)

            import copy
            for sample_ids in sample_ids_list:
                if sample_ids.numel() == 0:
                    results.append(None)
                    continue
                # Per-sample ref prefill state (cloned so each sample's
                # ref forward extends independently).
                sample_prefill = None
                if ref_prefill is not None:
                    sample_prefill = (
                        ref_prefill[0].detach().clone(),
                        ref_prefill[1].detach().clone(),
                        copy.deepcopy(ref_prefill[2]),
                    )
                ref_logps = self._tf_compute_sample_logp(
                    prompt_ids=prompt_ids, sample_ids=sample_ids,
                    temperature=temperature, pad_id=pad_id, device=device,
                    prefill_state=sample_prefill,
                )
                results.append(
                    ref_logps.detach() if ref_logps.numel() > 0 else None
                )
        finally:
            # Restore current params.
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(saved[name])
        return results

    @torch.no_grad()
    def _compute_ref_logps(
        self,
        prompt_ids: Tensor,
        sample_ids: Tensor,
        *,
        temperature: float,
        pad_id: int,
        device: torch.device,
    ) -> Tensor | None:
        """Single-sample backwards-compat wrapper around
        `_compute_ref_logps_batch`. Used only by tests; production call
        site uses the batch version directly to share the param-swap
        across K samples."""
        results = self._compute_ref_logps_batch(
            prompt_ids, [sample_ids],
            temperature=temperature, pad_id=pad_id, device=device,
        )
        return results[0]

    def step(
        self,
        prompt_ids: Tensor,
        *,
        num_samples: int,
        max_new_tokens: int,
        reward_kind: str,
        gold: str | None,
        meta: dict | None,
        tokenizer,
        temperature: float = 1.0,
    ) -> Phase2Metrics:
        """One GRPO step on a single prompt."""
        # Llama-3 has two natural stop signals: `<|end_of_text|>` (eos,
        # 128001) used for math/code completions, and `<|eot_id|>` (128009)
        # used to terminate an assistant turn in the chat template. Wave 3
        # data uses the former; Wave 4 (TurnPair from preprocess_chat.py)
        # appends the latter. Passing both as stop_ids covers both reward
        # shapes without needing source-aware plumbing.
        stop_ids = {tokenizer.eos_token_id}
        # `<|eot_id|>` is Llama-3's chat-turn terminator. Add it if the
        # tokenizer recognizes it; some tokenizers (e.g. test fakes) won't.
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                unk = getattr(tokenizer, "unk_token_id", None)
                if eot_id is not None and eot_id != unk:
                    stop_ids.add(eot_id)
            except Exception:
                pass
        pad_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
        device = next(self.model.parameters()).device

        # ── PASS 1: AR sample each response (no_grad, with prefill).
        # Returns (sample_ids, sample_logp_old) per sample. logp_old is the
        # detached log-prob of each sampled token under the policy at
        # SAMPLING time — used by pass 2 for the PPO importance-sampling
        # ratio. Recording it here is essentially free (softmax already
        # computed for sampling).
        samples: list[Tensor] = []
        samples_logp_old: list[Tensor] = []
        for _ in range(num_samples):
            sample, logp_old = self._ar_sample_one(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_ids=stop_ids,
                pad_id=pad_id,
                device=device,
            )
            samples.append(sample)
            samples_logp_old.append(logp_old)

        # ── SCORE ────────────────────────────────────────────────────
        decoded = [tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in samples]
        rewards = [
            compute_reward(reward_kind, c, gold=gold, meta=meta) for c in decoded
        ]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards_t)

        # ── PASS 2 SHARED PREFILL (Phase D1, no_grad) ─────────────────
        # Encode the prompt ONCE under the current policy, no_grad, to
        # obtain a shared starting state for the K per-sample TF replays.
        # Without this, each sample re-encodes the prompt — at typical
        # T_pre=1024 + T_gen=64 + K=6 that's ~78% wasted compute.
        # Cost of sharing: prompt-position writes don't get gradient
        # signal (only sample-position writes do). Acceptable: GRPO is
        # optimizing the rollout policy, sample-position writes are the
        # ones that directly affect reward.
        # Test-mode (no real Llama) skips shared prefill — the synthetic
        # logits path doesn't benefit and the prefill machinery uses
        # forward_window directly.
        shared_prefill = None
        if self.model.llama is not None:
            with torch.no_grad():
                _sp = self._prefill_prompt(
                    prompt_ids, pad_id=pad_id, device=device,
                )
                # _prefill_prompt now returns 5-tuple (N4 fix); use first 3.
                shared_prefill = _sp[:3]  # (states, hiddens, kv_cache)

        # ── PASS 2: TF logp + PPO-clip surrogate + KL to reference ───
        # TF-forward through SAMPLE only (with grad), starting from the
        # shared prefill state. Compute IS ratio against pass-1 logp,
        # apply PPO clip, accumulate weighted advantage. Optionally add
        # KL(π_θ ‖ π_ref) regularization computed under a temporarily-
        # swapped ref-policy forward pass.
        self.optimizer.zero_grad()
        policy_loss = torch.zeros((), device=device)
        kl_loss = torch.zeros((), device=device)
        any_grad = False

        # Diagnostics
        n_clipped = 0
        n_total = 0
        ratio_sum = 0.0
        kl_sum = 0.0
        kl_count = 0

        clip_low = 1.0 - self.clip_eps
        clip_high = 1.0 + (self.clip_eps_higher if self.clip_eps_higher is not None
                           else self.clip_eps)

        # Pre-compute ref-policy logps for all K samples in ONE swap-in
        # cycle (Phase D1 + B4). When kl_coef=0 or no ref_state, returns
        # list of Nones.
        ref_logps_list = self._compute_ref_logps_batch(
            prompt_ids, [s.to(device) for s in samples],
            temperature=temperature, pad_id=pad_id, device=device,
        )

        for i, (sample, logp_old, adv) in enumerate(zip(samples, samples_logp_old, advantages)):
            if sample.numel() == 0:
                continue
            # Per-sample prefill state: deep-clone the shared prefill so
            # each sample's writes evolve the manifold + cache independently.
            if shared_prefill is not None:
                import copy
                prefill_state = (
                    shared_prefill[0].detach().clone(),
                    shared_prefill[1].detach().clone(),
                    copy.deepcopy(shared_prefill[2]),  # KV cache deep-clone
                )
            else:
                prefill_state = None
            sample_logps_new = self._tf_compute_sample_logp(
                prompt_ids=prompt_ids,
                sample_ids=sample.to(device),
                temperature=temperature,
                pad_id=pad_id,
                device=device,
                prefill_state=prefill_state,
            )
            if sample_logps_new.numel() == 0:
                continue

            # Align logp_old to logp_new — logp_old has one entry per
            # sampled token; logp_new may drop the very-first token of the
            # sequence if there's no in-graph predecessor (see
            # `_tf_compute_sample_logp` boundary handling). Take the tail.
            n_new = sample_logps_new.numel()
            logp_old_aligned = logp_old[-n_new:].to(device).to(sample_logps_new.dtype)

            # PPO importance-sampling ratio + clip surrogate.
            log_ratio = sample_logps_new - logp_old_aligned
            ratio = log_ratio.exp()
            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
            # Loss is -min(r·A, clip(r)·A) (we minimize → negate the
            # surrogate objective).
            surrogate = -torch.min(adv * ratio, adv * clipped_ratio)
            policy_loss = policy_loss + surrogate.sum()

            # Diagnostics.
            n_clipped += int(((ratio < clip_low) | (ratio > clip_high)).sum())
            n_total += int(n_new)
            ratio_sum += float(ratio.detach().sum())

            # KL term against reference policy (precomputed in batch above).
            ref_logps = ref_logps_list[i]
            if ref_logps is not None and ref_logps.numel() == n_new:
                # John Schulman's K3 estimator: KL ≈ exp(log r) - log r - 1
                # where log r = logp_new - logp_ref. Always non-negative,
                # low-variance, exact in expectation.
                log_r = sample_logps_new - ref_logps.to(sample_logps_new.dtype)
                kl_per_tok = log_r.exp() - log_r - 1.0
                kl_loss = kl_loss + kl_per_tok.sum()
                kl_sum += float(kl_per_tok.detach().sum())
                kl_count += int(n_new)

            any_grad = True

        # Audit fix (Phase A1, Dr.GRPO): normalize by `K * max_new_tokens`
        # rather than just `K`. The earlier `sum(logp) / K` per sample
        # gives long completions larger |loss| for the same advantage,
        # producing the documented length-bias pathologies in GRPO
        # (completion-length explosion in correct-bias mode; "long-wrong"
        # under negative advantage). See arxiv 2503.20783 (Sea AI Lab,
        # COLM 2025). Dividing by a constant `max_new_tokens` is
        # length-agnostic.
        denom = max(num_samples * max_new_tokens, 1)
        policy_loss = policy_loss / denom
        kl_loss = kl_loss / denom
        total_loss = policy_loss + self.kl_coef * kl_loss

        if any_grad:
            total_loss.backward()
            grad_norm = self._clip_and_step()
        else:
            grad_norm = 0.0
            if self.scheduler is not None:
                self.scheduler.step()
        self._step_count += 1

        return Phase2Metrics(
            policy_loss=float(total_loss.detach()),
            grad_norm=float(grad_norm),
            lr=[g["lr"] for g in self.optimizer.param_groups],
            rewards=rewards,
            advantages=advantages.tolist(),
            decoded=decoded,
            mean_response_len=sum(len(s) for s in samples) / max(len(samples), 1),
            clip_fraction=n_clipped / max(n_total, 1),
            mean_ratio=ratio_sum / max(n_total, 1),
            kl_to_ref=kl_sum / max(kl_count, 1),
        )

    # ── PASS 1: no-grad AR sampling (with prefill, KV-cached) ────────

    @torch.no_grad()
    def _prefill_prompt(
        self,
        prompt_ids: Tensor,
        *,
        pad_id: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor | None, object, Tensor | None, int]:
        """Walk forward_window over the prompt in T_window-sized windows,
        accumulating manifold state AND populating an HF KV cache.

        N4 fix — handles prompts whose length is NOT divisible by T_window
        without putting pad tokens in the KV cache:
          - Full T-aligned windows: encoded via forward_window (memory ops fire)
          - Trailing partial window (n_tail < T): encoded via direct
            llama.model call WITHOUT memory ops AND WITHOUT padding —
            real tokens only enter the KV cache.

        The trailing tail's last hidden is what AR sampling uses for the
        FIRST generated token's logit (instead of the prior version's
        pad-position hidden). Returns the tail's last hidden as
        `last_real_hidden`; `_ar_sample_one` uses this as the predecessor
        for the first sampled token.

        Returns (prev_states, prev_window_hiddens, kv_cache,
                 last_real_hidden, abs_pos).
            - prev_window_hiddens is the LAST FULL T-window's hiddens (used
              by the next memory window's read trajectory)
            - last_real_hidden is the hidden of the actual last prompt
              token (used by AR for first-token logit)
            - abs_pos is the absolute position counter after all prompt
              tokens have been encoded (= n)

        Long prompts (NarrativeQA 8K, WildChat 4K+) were previously not
        properly written into memory in the old per-token AR loop;
        prefill makes the entire prompt's T-aligned portion visible.

        no_grad — pure inference setup. Pass 2 redoes a TF forward with
        grad if memory writes need gradient signal.
        """
        from src.trajectory_memory.tbptt import _trim_kv_cache

        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window
        toks = prompt_ids.flatten().tolist()
        n = len(toks)

        prev_states = self.model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None
        kv_cache: object | None = None
        abs_pos = 0

        # Encode full T-aligned windows via forward_window (memory ops fire).
        n_aligned = (n // T) * T
        for win_start in range(0, n_aligned, T):
            win = toks[win_start : win_start + T]
            win_t = torch.tensor(win, dtype=torch.int64, device=device).unsqueeze(0)

            last_prev_logit = (
                prev_window_hiddens[:, -1:, :]
                if prev_window_hiddens is not None else None
            )
            out = self.model.forward_window(
                lm_input_ids=win_t,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None,
                hard_routing=False,  # N3 — deterministic routing in pass 1
                past_key_values=kv_cache,
                use_kv_cache=True,
                last_prev_logit_hidden=last_prev_logit,
                cache_abs_pos=abs_pos,
            )
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            kv_cache = out.get("new_past_key_values", kv_cache)
            abs_pos = out.get("new_cache_abs_pos", abs_pos + T)
            kv_cache = _trim_kv_cache(kv_cache, cap)

        # N4 — encode the trailing partial window (if any) via direct
        # llama.model call. NO padding, NO memory ops. Just extends the
        # KV cache with REAL tokens at correct absolute positions.
        last_real_hidden: Tensor | None = None
        n_tail = n - n_aligned
        if n_tail > 0:
            tail = toks[n_aligned:]
            tail_t = torch.tensor(tail, dtype=torch.int64, device=device).unsqueeze(0)
            cache_position = torch.arange(
                abs_pos, abs_pos + n_tail, device=device,
            )
            base_out = self.model.llama.model(
                input_ids=tail_t,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            kv_cache = base_out.past_key_values
            abs_pos += n_tail
            kv_cache = _trim_kv_cache(kv_cache, cap)
            # last hidden of the actual last prompt token — used by AR for
            # the first generated token's logit.
            last_real_hidden = base_out.last_hidden_state[:, -1:, :]
        elif prev_window_hiddens is not None:
            # No tail; last real hidden is the last position of the last
            # encoded full window.
            last_real_hidden = prev_window_hiddens[:, -1:, :]

        return prev_states, prev_window_hiddens, kv_cache, last_real_hidden, abs_pos

    @torch.no_grad()
    def _ar_sample_one(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        stop_ids: set[int] | None,
        pad_id: int,
        device: torch.device,
    ) -> Tensor:
        """One AR rollout, no_grad, with per-window memory + KV cache.

        Per-window structure (correct per design — fires read/write at
        memory-window boundaries, NOT per token):
          1. Prefill prompt → populates manifold state and KV cache
          2. For each generation window of up to T_window tokens:
               (a) read once from prev_window_hiddens
               (b) install memory_fn for this window
               (c) AR-generate token-by-token with KV cache (1-token
                   forward against cached prefix for each new token)
               (d) write once at end of window
               (e) trim cache to effective_lm_context

        Pre-KV-cache version padded every token to T_window and called
        forward_window per-token, which fired read+write per-token AND
        re-encoded the entire rolling buffer per-token. This version
        keeps memory ops at proper window boundaries AND each token's LM
        forward is a single 1-token forward against KV cache.

        Pass 2 (`_tf_compute_sample_logp`) recomputes per-token logp with
        gradient — pass 1 doesn't need to record it.
        """
        from src.trajectory_memory.tbptt import _trim_kv_cache

        # Test-mode fallback (no real Llama): the KV cache path requires
        # llama.model and llama.lm_head, which only exist with attach_lm=True.
        # Tests run with attach_lm=False using forward_window's synthetic
        # logits path — fall back to per-window forward_window calls.
        if self.model.llama is None:
            return self._ar_sample_one_test_mode(
                prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature,
                stop_ids=stop_ids, pad_id=pad_id, device=device,
            )

        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window

        # Prefill — populates manifold state and KV cache for the prompt.
        # N4 fix — `last_real_hidden` is the hidden of the actual last
        # prompt token (no pad), and `abs_pos` is the absolute position
        # counter after all prompt tokens have been encoded.
        (prev_states, prev_window_hiddens, kv_cache,
         last_real_hidden, abs_pos) = self._prefill_prompt(
            prompt_ids, pad_id=pad_id, device=device,
        )

        generated: list[int] = []
        # `logp_old` records logp_pi_old(token) at sampling time, used by
        # pass 2's PPO-clip importance-sampling ratio. Detached / no_grad
        # — pass 2 recomputes logp_new under autograd.
        logp_old_list: list[Tensor] = []

        # First generated token: predict from `last_real_hidden` (= the
        # actual last prompt position's hidden, NOT a pad token).
        # last_real_hidden is in Llama's native dtype (bf16).
        lm_head_dtype = next(self.model.llama.lm_head.parameters()).dtype
        last_hidden = (
            last_real_hidden if last_real_hidden is not None
            else prev_window_hiddens[:, -1:, :]
        ).to(lm_head_dtype)
        first_logit = self.model.llama.lm_head(last_hidden).float()
        scaled = first_logit[:, -1, :] / temperature
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        first_token = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(first_token)
        logp_old_list.append(log_probs[0, first_token].detach())

        if stop_ids is not None and first_token in stop_ids:
            return (
                torch.tensor(generated, dtype=torch.int64),
                torch.stack(logp_old_list).cpu(),
            )

        # Generation windows.
        stopped = False
        while len(generated) < max_new_tokens and not stopped:
            n_remaining = max_new_tokens - len(generated)
            # We'll generate up to T tokens this window; since `generated`
            # already has the FIRST token of this window, we actually need
            # T-1 more *forwards* to fill the window's hiddens (the first
            # token's hidden comes from forwarding it, plus T-1 subsequent
            # tokens get sampled from each forward's logit).
            #
            # Concretely: each iteration forwards `generated[-1]` to get
            # its hidden + sample the next token. So one iteration produces
            # one hidden and one new sampled token.

            # ── (a, b) READ for this window + install memory_fn ──
            # N3 fix — deterministic routing (argmax, no Gumbel noise)
            # in pass 1. Pass 2's TF replay also uses argmax → same
            # routing path on both sides → IS ratio numerator/denominator
            # condition on the SAME memory state. Tradeoff: lose Gumbel
            # exploration of routing during pass 1; routing decisions are
            # deterministic given the input. Accepted because the
            # alternative (record + replay) is a much bigger refactor.
            # Short-prompt edge case: if prompt < T_window, no full
            # window was encoded so prev_window_hiddens is None. Fall
            # back to zeros (matches forward_window's first-window logic).
            if prev_window_hiddens is None:
                prev_window_hiddens = torch.zeros(
                    1, T, cfg.d_lm,
                    dtype=prev_states.dtype, device=device,
                )
            prev_hid_mem = prev_window_hiddens.to(prev_states.dtype)
            read_visited, _ = self.model.read_module(
                prev_hid_mem, prev_states, self.model.manifold, hard=False,
            )
            mem_inject = self.model._mem_inject_layer()
            mem_inject.memory_fn = self.model._build_memory_fn(read_visited)

            window_hiddens_list: list[Tensor] = []
            try:
                # ── (c) AR-generate up to T tokens within this window ──
                for _ in range(min(T, n_remaining)):
                    # Forward last sampled token (1 token) against KV cache.
                    last_token = generated[-1]
                    input_ids = torch.tensor(
                        [[last_token]], dtype=torch.int64, device=device,
                    )
                    # N1 — pass cache_position for RoPE correctness.
                    cache_position = torch.tensor(
                        [abs_pos], dtype=torch.int64, device=device,
                    )
                    base_out = self.model.llama.model(
                        input_ids=input_ids,
                        past_key_values=kv_cache,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    abs_pos += 1
                    kv_cache = base_out.past_key_values
                    hidden = base_out.last_hidden_state          # [1, 1, d_lm]
                    window_hiddens_list.append(hidden)

                    # Sample next token from this hidden's logit.
                    logits = self.model.llama.lm_head(hidden).float()
                    scaled = logits[:, -1, :] / temperature
                    log_probs = F.log_softmax(scaled, dim=-1)
                    probs = log_probs.exp()
                    sampled = int(torch.multinomial(probs, num_samples=1).item())
                    generated.append(sampled)
                    logp_old_list.append(log_probs[0, sampled].detach())

                    if stop_ids is not None and sampled in stop_ids:
                        stopped = True
                        break
            finally:
                mem_inject.memory_fn = None

            if not window_hiddens_list:
                break

            # ── (d) WRITE at end of window ──
            cur_hiddens = torch.cat(window_hiddens_list, dim=1)  # [1, n_win, d_lm]
            n_win = cur_hiddens.shape[1]
            # Audit fix (Phase A2): partial windows pad with zero hiddens
            # which would scatter_mean *zero* into selected concept slots
            # (polluting them on every short completion). Two safeguards:
            #   1. If the window is mostly zeros (n_win < T/2), skip the
            #      write entirely. Sparse short writes wouldn't carry
            #      meaningful signal anyway.
            #   2. Otherwise, build prev_window_hiddens by repeating the
            #      LAST real hidden over the pad positions (instead of
            #      zeros) so subsequent reads aren't fed garbage.
            if n_win < T // 2:
                # Skip write — too few real tokens to carry useful signal.
                # Just propagate previous state forward unchanged. Carry
                # the real hiddens (no pad) as prev_window_hiddens so the
                # NEXT window's read isn't conditioned on zero pad.
                pad = cur_hiddens[:, -1:, :].expand(1, T - n_win, cfg.d_lm)
                prev_window_hiddens = torch.cat([cur_hiddens, pad], dim=1)
            else:
                # Pad by repeating the last real hidden (not zeros) so the
                # write module sees coherent content rather than zeros.
                if n_win < T:
                    pad = cur_hiddens[:, -1:, :].expand(1, T - n_win, cfg.d_lm)
                    cur_hiddens_padded = torch.cat([cur_hiddens, pad], dim=1)
                else:
                    cur_hiddens_padded = cur_hiddens

                cur_hiddens_mem = cur_hiddens_padded.to(prev_states.dtype)
                # Surprise=0 during AR (per plan §5.4 — generated tokens have
                # no NTP target).
                surprise = torch.zeros(1, dtype=prev_states.dtype, device=device)
                # N3 fix — deterministic routing here too (matches pass 2
                # replay's deterministic routing in the writer).
                new_states, _, _ = self.model.write_module(
                    cur_hiddens_mem, surprise, prev_states, self.model.manifold,
                    hard=False,
                )
                prev_states = new_states
                prev_window_hiddens = cur_hiddens_padded

            # ── (e) Trim cache ──
            kv_cache = _trim_kv_cache(kv_cache, cap)

        # Drop the very last generated token if it was sampled but not
        # part of any window's hiddens (i.e. window's last sample triggered
        # stop before the next iter could forward it). Actually no — we
        # KEEP it in `generated` since it's the stop token / completion.
        return (
            torch.tensor(generated, dtype=torch.int64),
            torch.stack(logp_old_list).cpu() if logp_old_list else torch.zeros(0),
        )

    @torch.no_grad()
    def _ar_sample_one_test_mode(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        stop_ids: set[int] | None,
        pad_id: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Test-mode AR loop — uses forward_window for each generated token.
        Slow but works without real Llama. Used by `test_phase2_trainer_*`.
        Returns (sample_ids, sample_logp_old)."""
        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window

        # Per-window prefill via forward_window (rolling-buffer mode).
        toks = prompt_ids.flatten().tolist()
        prev_states = self.model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None
        lm_buffer = torch.empty(1, 0, dtype=torch.int64, device=device)
        for win_start in range(0, len(toks), T):
            win = toks[win_start : win_start + T]
            if len(win) < T:
                win = win + [pad_id] * (T - len(win))
            win_t = torch.tensor(win, dtype=torch.int64, device=device).unsqueeze(0)
            lm_input = torch.cat([lm_buffer, win_t], dim=1)
            if lm_input.shape[1] > cap:
                lm_input = lm_input[:, -cap:]
            out = self.model.forward_window(
                lm_input_ids=lm_input,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None, hard_routing=True,
            )
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            lm_buffer = lm_input[:, -(cap - T):] if cap > T else lm_input[:, :0]

        cur_tokens = list(toks)
        generated: list[int] = []
        logp_old_list: list[Tensor] = []
        for _ in range(max_new_tokens):
            lm_input = torch.tensor(
                cur_tokens[-cap:], dtype=torch.int64, device=device,
            ).unsqueeze(0)
            if lm_input.shape[1] < T:
                pad_n = T - lm_input.shape[1]
                lm_input = F.pad(lm_input, (pad_n, 0), value=pad_id)
            out = self.model.forward_window(
                lm_input_ids=lm_input,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None, hard_routing=True,
            )
            logits = out["logits"][:, -1, :].float()
            scaled = logits / temperature
            log_probs = F.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
            sampled = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(sampled)
            logp_old_list.append(log_probs[0, sampled].detach())
            cur_tokens.append(sampled)
            if stop_ids is not None and sampled in stop_ids:
                break
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
        return (
            torch.tensor(generated, dtype=torch.int64),
            torch.stack(logp_old_list).cpu() if logp_old_list else torch.zeros(0),
        )

    # ── PASS 2: TF logp computation (with grad, single backward, KV-cached) ──

    def _tf_compute_sample_logp(
        self,
        prompt_ids: Tensor,
        sample_ids: Tensor,
        *,
        temperature: float,
        pad_id: int,
        device: torch.device,
        prefill_state: tuple | None = None,
    ) -> Tensor:
        """TF-forward through (prompt + sample) with KV cache, return
        per-sample-token logp under `softmax(logits / temperature)`.

        Two modes:

        - **Self-contained** (`prefill_state=None`, default): walks the
          full `prompt + sample` sequence with grad. Used by tests and
          by the legacy `grpo_step` shim.

        - **Shared-prefill** (Phase D1): caller passes
          `prefill_state = (prev_states, prev_window_hiddens, kv_cache)`
          from a prior no_grad prompt encoding (`_pass2_shared_prefill`).
          We skip the prompt windows and walk only the sample tokens
          starting from the prefill state. Per-step pass 2 cost drops
          ~3-4× when K > 1 samples share the same prompt — the prompt
          is encoded ONCE no_grad rather than K times with grad.

        Memory state carries through `write_module` across sample windows
        without any detach, so the autograd graph from each sample-token
        logp reaches all trainable params. KV cache also carries with
        grad. In shared-prefill mode the prefill's contribution is
        detached (no_grad) — write_module gets gradient signal only from
        sample-position writes, which is what GRPO actually optimizes.

        Returns: [n_sample_tokens] tensor of logps with grad.
        """
        from src.trajectory_memory.tbptt import _trim_kv_cache

        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window

        sample = sample_ids.flatten().tolist()
        n_sample = len(sample)

        if prefill_state is not None:
            # Shared-prefill mode: prompt already encoded no_grad upstream.
            prev_states, prev_window_hiddens, kv_cache = prefill_state
            seq_to_walk = sample
            n_walk = n_sample
            first_sample_offset = 0
        else:
            # Self-contained: walk full (prompt + sample).
            prompt = prompt_ids.flatten().tolist()
            n_prompt = len(prompt)
            prev_states = self.model.manifold.reset_states(batch_size=1)
            prev_window_hiddens: Tensor | None = None
            kv_cache: object | None = None
            seq_to_walk = prompt + sample
            n_walk = len(seq_to_walk)
            first_sample_offset = n_prompt

        sample_logps: list[Tensor] = []

        for win_start in range(0, n_walk, T):
            win_end = min(win_start + T, n_walk)
            win = seq_to_walk[win_start:win_end]
            n_real = len(win)
            if n_real < T:
                win = win + [pad_id] * (T - n_real)
            win_t = torch.tensor(win, dtype=torch.int64, device=device).unsqueeze(0)

            last_prev_logit = (
                prev_window_hiddens[:, -1:, :]
                if prev_window_hiddens is not None else None
            )
            # N2 fix — when in shared-prefill mode (sample-only TF replay),
            # this loop processes RESPONSE windows. Pass 1's `_ar_sample_one`
            # wrote response windows with `surprise=0` (per plan §5.4 —
            # generated tokens have no NTP target). Pass 2 must do the
            # same; otherwise the writer mutates differently than pass 1
            # did, the memory state diverges, and the IS ratio's numerator/
            # denominator condition on different states.
            # N3 fix — also use deterministic routing (`hard_routing=False`
            # = argmax, no Gumbel noise) in pass 2's sample-window TF
            # replay. Pass 1 used Gumbel-stochastic routing but didn't
            # record the routing decisions; if pass 2 resamples with
            # Gumbel, routing diverges → memory paths diverge. Argmax on
            # both sides gives a SINGLE consistent routing path that pass
            # 1 and pass 2 will follow (ignoring the Gumbel sampling
            # entirely is a tradeoff — we lose stochastic exploration of
            # routing in exchange for IS-ratio correctness).
            in_shared_prefill_mode = prefill_state is not None
            out = self.model.forward_window(
                lm_input_ids=win_t,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None,
                hard_routing=not in_shared_prefill_mode,
                past_key_values=kv_cache,
                use_kv_cache=True,
                last_prev_logit_hidden=last_prev_logit,
                force_surprise=0.0 if in_shared_prefill_mode else None,
            )
            # In KV-cache mode forward_window's `logits` is [1, T, V]
            # aligned to the new window's T positions. logit at index i
            # predicts win[i+1] (since position i's hidden contains the
            # context to predict the next token).
            #
            # For sample positions p in [n_prompt, n_full):
            #   - if p > win_start: predictor logit is logits[0, p - win_start - 1]
            #   - if p == win_start: predictor logit is the NTP-shift
            #     left predecessor — i.e., the hidden of position
            #     win_start-1 (last position of previous window). When
            #     we have prev_window_hiddens (always except very first
            #     window), this is `last_prev_logit` whose logit we
            #     compute below.
            logits = out["logits"]                               # [1, T, V]

            for p in range(max(win_start, first_sample_offset), win_end):
                if p == win_start:
                    # Predecessor is the previous window's last hidden.
                    if last_prev_logit is None:
                        continue  # very first token of sequence — skip
                    if self.model.llama is None:
                        # Test mode — synthesize via the same path
                        # forward_window uses for its synthetic logits.
                        logit_at_pred = (
                            last_prev_logit.to(self.model._test_proj.dtype)
                            @ self.model._test_lm_head
                        )                                        # [1, 1, V]
                    else:
                        # prev_window_hiddens is in manifold's fp32; cast
                        # to lm_head's native dtype (bf16).
                        lm_head_dtype = next(
                            self.model.llama.lm_head.parameters()
                        ).dtype
                        logit_at_pred = self.model.llama.lm_head(
                            last_prev_logit.to(lm_head_dtype)
                        )                                        # [1, 1, V]
                    logit_vec = logit_at_pred[0, 0]              # [V]
                else:
                    logit_idx = p - win_start - 1
                    logit_vec = logits[0, logit_idx]             # [V]
                scaled = logit_vec / temperature
                logp_p = F.log_softmax(scaled, dim=-1)[seq_to_walk[p]]
                sample_logps.append(logp_p)

            # Carry state — NO DETACH, autograd graph stays alive across
            # windows of a sample. KV cache likewise.
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            kv_cache = out.get("new_past_key_values", kv_cache)
            kv_cache = _trim_kv_cache(kv_cache, cap)

        if not sample_logps:
            return torch.zeros(0, device=device)
        return torch.stack(sample_logps)

    # ── helpers ───────────────────────────────────────────────────────

    def _clip_and_step(self) -> float:
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                max_norm=self.grad_clip,
            )
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(grad_norm)


# ── Legacy free functions ─────────────────────────────────────────────


def grpo_rollout(
    model: IntegratedLM,
    prompt_ids: Tensor,
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_id: int | None = None,
    pad_id: int = 0,
):
    """Compatibility shim — pure no-grad sampling. Returns
    `(samples, logps)` where `logps` is a list of empty tensors (the
    new two-pass design recomputes logps in pass 2 inside the trainer
    step, so there's no per-token logp here). Existing callers should
    migrate to `Phase2Trainer.step` for actual training."""
    trainer = Phase2Trainer(model, optimizer=torch.optim.SGD([torch.zeros(1)], lr=0))
    device = next(model.parameters()).device
    samples = []
    stop_ids = {eos_id} if eos_id is not None else None
    for _ in range(num_samples):
        sample, _logp_old = trainer._ar_sample_one(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_ids=stop_ids,
            pad_id=pad_id,
            device=device,
        )
        samples.append(sample)
    logps = [torch.zeros(0) for _ in samples]
    return samples, logps


def grpo_step(
    model: IntegratedLM,
    prompt_ids: Tensor,
    *,
    optimizer: Optimizer,
    num_samples: int,
    max_new_tokens: int,
    reward_kind: str,
    gold: str | None,
    meta: dict | None,
    tokenizer,
    kl_coef: float = 0.0,
) -> dict:
    """Compatibility shim — single GRPO step. Prefer `Phase2Trainer.step`."""
    trainer = Phase2Trainer(model, optimizer)
    m = trainer.step(
        prompt_ids,
        num_samples=num_samples, max_new_tokens=max_new_tokens,
        reward_kind=reward_kind, gold=gold, meta=meta, tokenizer=tokenizer,
    )
    return {
        "policy_loss": m.policy_loss,
        "rewards": m.rewards,
        "advantages": m.advantages,
        "decoded": m.decoded,
    }
