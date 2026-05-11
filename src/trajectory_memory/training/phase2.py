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
                # Keep full 5-tuple so per-sample ref TF replay can thread
                # cache_abs_pos to forward_window for RoPE correctness.
                ref_prefill = self._prefill_prompt(
                    prompt_ids, pad_id=pad_id, device=device,
                )

            import copy
            for sample_ids in sample_ids_list:
                if sample_ids.numel() == 0:
                    results.append(None)
                    continue
                # Per-sample ref prefill state (cloned so each sample's
                # ref forward extends independently).
                sample_prefill = None
                if ref_prefill is not None:
                    _rps, _rph, _rkv, _rlr, _rap = ref_prefill
                    sample_prefill = (
                        _rps.detach().clone(),
                        _rph.detach().clone() if _rph is not None else None,
                        copy.deepcopy(_rkv),
                        _rlr.detach().clone() if _rlr is not None else None,
                        _rap,
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
        #
        # N7 fix — shared pass-1 prefill across K samples (encode prompt
        # ONCE). For real Llama we then run all K AR rollouts IN PARALLEL
        # at BS=K via `_ar_sample_batch` — replaces K × T_gen sequential
        # single-token forwards with T_gen BS=K forwards (~K× launch
        # overhead reduction; rollout is launch-bound on a single 4090).
        # For test mode (no real Llama, attach_lm=False), we fall back to
        # the K-serial path since `_ar_sample_one_test_mode` uses
        # forward_window directly and doesn't share the BS=K KV cache
        # plumbing.
        pass1_prefill_shared = None
        if self.model.llama is not None:
            pass1_prefill_shared = self._prefill_prompt(
                prompt_ids, pad_id=pad_id, device=device,
            )

        if self.model.llama is not None:
            # Batched rollout — single call, K samples in parallel.
            samples, samples_logp_old = self._ar_sample_batch(
                prompt_ids,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_ids=stop_ids,
                pad_id=pad_id,
                device=device,
                prefill_state=pass1_prefill_shared,
            )
        else:
            # Test-mode fallback (no Llama) — K serial AR loops.
            samples = []
            samples_logp_old = []
            for _ in range(num_samples):
                sample, logp_old = self._ar_sample_one(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_ids=stop_ids,
                    pad_id=pad_id,
                    device=device,
                    prefill_state=None,
                )
                samples.append(sample)
                samples_logp_old.append(logp_old)

        # ── SCORE ────────────────────────────────────────────────────
        decoded = [tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in samples]
        rewards = [
            compute_reward(reward_kind, c, gold=gold, meta=meta) for c in decoded
        ]
        # Build advantages directly on the training device. Earlier code
        # built on CPU and relied on PyTorch's 0-dim CPU↔CUDA broadcast
        # to multiply with `ratio` later. That works on modern PyTorch but
        # is fragile (changes with version, breaks on non-zero-dim ops)
        # and silently slow if it ever falls back to host sync.
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = compute_grpo_advantages(rewards_t)

        # Phase 2 perf fix (#8) — early-exit on zero-advantage groups.
        # When all rewards in a GRPO group are equal, advantages are all
        # zero (group-relative center) → policy gradient is exactly zero
        # → entire pass 2 + backward is wasted compute. Common with
        # sparse exact-match rewards early in training. Only safe to skip
        # when there's also no KL term — KL would otherwise still pull
        # the policy toward π_ref independently of advantage.
        no_signal = (
            advantages.abs().max() < 1e-8
            and (self.kl_coef == 0 or self.ref_state is None)
        )
        if no_signal:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self._step_count += 1
            return Phase2Metrics(
                policy_loss=0.0, grad_norm=0.0,
                lr=[g["lr"] for g in self.optimizer.param_groups],
                rewards=rewards, advantages=advantages.tolist(),
                decoded=decoded,
                mean_response_len=sum(len(s) for s in samples) / max(len(samples), 1),
                clip_fraction=0.0, mean_ratio=1.0, kl_to_ref=0.0,
            )

        # ── PASS 2 SHARED PREFILL ─────────────────────────────────────
        # Reuses pass 1's prefill — same weights (no optimizer step yet),
        # deterministic routing in _prefill_prompt, and pass 1's
        # `pass1_prefill_shared` is only deep-cloned for samples (the
        # original is preserved). Re-encoding under no_grad here was
        # redundant compute (~78% saved at T_pre=1024 / K=6+).
        shared_prefill = pass1_prefill_shared

        # ── PASS 2: TF logp + PPO-clip surrogate + KL to reference ───
        # TF-forward through SAMPLE only (with grad), starting from the
        # shared prefill state. Compute IS ratio against pass-1 logp,
        # apply PPO clip, accumulate weighted advantage. Optionally add
        # KL(π_θ ‖ π_ref) regularization computed under a temporarily-
        # swapped ref-policy forward pass.
        # R3: per-sample backward — accumulate gradients sample-by-sample
        # rather than holding all K autograd graphs alive until a joint
        # backward. PyTorch's optimizer.zero_grad() at start + per-sample
        # backward() leaves grad buffers accumulating; this is
        # mathematically identical to a joint backward
        # (∇Σᵢ lᵢ = Σᵢ ∇lᵢ) but cuts peak VRAM by ~K× because each
        # sample's TF activation graph is freed after its backward fires.
        self.optimizer.zero_grad()
        any_grad = False
        total_loss_value = 0.0          # scalar for reporting only

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

        # Audit fix (Phase A1, Dr.GRPO): normalize by `K * max_new_tokens`
        # rather than just `K`. The earlier `sum(logp) / K` per sample
        # gives long completions larger |loss| for the same advantage,
        # producing the documented length-bias pathologies in GRPO
        # (completion-length explosion in correct-bias mode; "long-wrong"
        # under negative advantage). See arxiv 2503.20783 (Sea AI Lab,
        # COLM 2025). Dividing by a constant `max_new_tokens` is
        # length-agnostic. Applied per-sample so the sum-of-backwards
        # equals the joint-backward gradient.
        denom = max(num_samples * max_new_tokens, 1)

        for i, (sample, logp_old, adv) in enumerate(zip(samples, samples_logp_old, advantages)):
            if sample.numel() == 0:
                continue
            # Per-sample prefill state: deep-clone the shared prefill so
            # each sample's writes evolve the manifold + cache independently.
            if shared_prefill is not None:
                import copy
                _sps, _sph, _skv, _slr, _sap = shared_prefill
                prefill_state = (
                    _sps.detach().clone(),
                    _sph.detach().clone() if _sph is not None else None,
                    copy.deepcopy(_skv),  # KV cache deep-clone
                    _slr.detach().clone() if _slr is not None else None,
                    _sap,  # int, no clone needed
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
            # Edge case: if a sample is so short that its TF replay only
            # spans the very first sample-position (logp derived from the
            # detached last_prev_logit predecessor + frozen lm_head), the
            # resulting logp has no autograd graph and `.backward()` would
            # crash. This is rare (model emits EOS as the FIRST sampled
            # token, leaving a 1-token sample), but observed once per
            # ~hundred steps on an untrained backbone. Skip — no useful
            # gradient signal anyway.
            if not sample_logps_new.requires_grad:
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
            sample_policy_loss = surrogate.sum() / denom

            # Diagnostics.
            n_clipped += int(((ratio < clip_low) | (ratio > clip_high)).sum())
            n_total += int(n_new)
            ratio_sum += float(ratio.detach().sum())

            # KL term against reference policy (precomputed in batch above).
            ref_logps = ref_logps_list[i]
            sample_kl_loss = torch.zeros((), device=device)
            if ref_logps is not None and ref_logps.numel() == n_new:
                # John Schulman's K3 estimator: KL ≈ exp(log r) - log r - 1
                # where log r = logp_new - logp_ref. Always non-negative,
                # low-variance, exact in expectation.
                log_r = sample_logps_new - ref_logps.to(sample_logps_new.dtype)
                kl_per_tok = log_r.exp() - log_r - 1.0
                sample_kl_loss = kl_per_tok.sum() / denom
                kl_sum += float(kl_per_tok.detach().sum())
                kl_count += int(n_new)

            # Per-sample backward — frees this sample's activation graph
            # before next iteration builds its own.
            sample_total_loss = sample_policy_loss + self.kl_coef * sample_kl_loss
            sample_total_loss.backward()
            total_loss_value += float(sample_total_loss.detach())
            any_grad = True

        if any_grad:
            grad_norm = self._clip_and_step()
        else:
            grad_norm = 0.0
            if self.scheduler is not None:
                self.scheduler.step()
        self._step_count += 1

        return Phase2Metrics(
            policy_loss=total_loss_value,
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
        # CRUCIAL: MemInjectLayer's scale is non-zero (init=0.1) and its
        # forward() raises if memory_fn is None and scale != 0 (silent-
        # bypass footgun). For the tail-encode call we install a
        # zero-returning memory_fn so the bridge runs as a no-op and the
        # forward succeeds. Cleared after the call.
        last_real_hidden: Tensor | None = None
        n_tail = n - n_aligned
        if n_tail > 0:
            tail = toks[n_aligned:]
            tail_t = torch.tensor(tail, dtype=torch.int64, device=device).unsqueeze(0)
            n_cache_before = (
                kv_cache.get_seq_length() if kv_cache is not None else 0
            )
            cache_position = torch.arange(
                n_cache_before, n_cache_before + n_tail, device=device,
            )
            position_ids = torch.arange(
                abs_pos, abs_pos + n_tail, device=device,
            ).unsqueeze(0)

            mem_inject = self.model._mem_inject_layer()
            def _zero_memory(h_mem):
                return torch.zeros_like(h_mem)
            mem_inject.memory_fn = _zero_memory
            try:
                base_out = self.model.llama.model(
                    input_ids=tail_t,
                    past_key_values=kv_cache,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    use_cache=True,
                )
            finally:
                mem_inject.memory_fn = None
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
        prefill_state: tuple | None = None,
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

        N7 fix — `prefill_state` argument lets the caller share a
        pre-computed prefill (states, hiddens, kv_cache, last_real_hidden,
        abs_pos) across K samples instead of K separate prefills. Caller
        is responsible for cloning the state per sample (deep_copy of
        cache, .clone() of states/hiddens). When None, we prefill ourselves
        (legacy behavior, used by test mode and the legacy `grpo_rollout`
        shim).

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
        # N7 fix — accept a pre-computed prefill_state from the caller
        # (shared across K samples), or compute one ourselves (legacy
        # path, used by `grpo_rollout` shim). Caller-provided state has
        # already been cloned per-sample so this AR loop's mutations
        # don't affect other samples.
        if prefill_state is not None:
            (prev_states, prev_window_hiddens, kv_cache,
             last_real_hidden, abs_pos) = prefill_state
        else:
            # N4 — `last_real_hidden` is the actual last prompt token's
            # hidden (no pad), `abs_pos` is the abs position counter.
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
            # Audit fix #3 (post-Day-3) — pass 1 must use the SAME padding
            # strategy as pass 2's TF replay so memory state evolves
            # identically across passes (otherwise IS ratio is wrong on
            # the last partial window). Pass 2's TF replay encodes the
            # tail as `pad_id` tokens through Llama, getting Llama's
            # actual hidden representation for pad. Pass 1 here mimics
            # that by feeding pad_id tokens to llama.model() to get the
            # same pad-position hiddens.
            #
            # Audit fix (Phase A2) preserved — if window is mostly empty
            # (n_win < T/2), still skip the write entirely; sparse short
            # writes wouldn't carry useful signal anyway.
            if n_win < T // 2:
                pad = cur_hiddens[:, -1:, :].expand(1, T - n_win, cfg.d_lm)
                prev_window_hiddens = torch.cat([cur_hiddens, pad], dim=1)
            else:
                if n_win < T:
                    # Forward T - n_win pad_id tokens through Llama to get
                    # their hiddens — mirrors what pass 2 does when its
                    # last sample window is also partial.
                    n_pad = T - n_win
                    pad_t = torch.full(
                        (1, n_pad), pad_id,
                        dtype=torch.int64, device=device,
                    )
                    pad_cache_position = torch.arange(
                        kv_cache.get_seq_length() if kv_cache is not None else 0,
                        (kv_cache.get_seq_length() if kv_cache is not None else 0) + n_pad,
                        device=device,
                    )
                    pad_position_ids = torch.arange(
                        abs_pos, abs_pos + n_pad, device=device,
                    ).unsqueeze(0)
                    # CRASH FIX — the AR loop's `finally` block above
                    # cleared mem_inject.memory_fn, so the prior comment
                    # "memory_fn is still installed" was wrong. Re-install
                    # the SAME read-trajectory's memory_fn for pad encoding
                    # so it's consistent with pass 2 (which keeps memory_fn
                    # installed for the entire window's encoding).
                    # Without this re-install, MemInjectLayer raises
                    # "called without memory_fn but scale is not all-zero",
                    # crashing every Phase 2 step where AR generates
                    # T/2..T-1 tokens (very common at default
                    # --max-new-tokens 256).
                    mem_inject.memory_fn = self.model._build_memory_fn(read_visited)
                    try:
                        pad_out = self.model.llama.model(
                            input_ids=pad_t,
                            past_key_values=kv_cache,
                            cache_position=pad_cache_position,
                            position_ids=pad_position_ids,
                            use_cache=True,
                        )
                    finally:
                        mem_inject.memory_fn = None
                    kv_cache = pad_out.past_key_values
                    abs_pos += n_pad
                    pad_hiddens = pad_out.last_hidden_state
                    cur_hiddens_padded = torch.cat([cur_hiddens, pad_hiddens], dim=1)
                else:
                    cur_hiddens_padded = cur_hiddens

                cur_hiddens_mem = cur_hiddens_padded.to(prev_states.dtype)
                surprise = torch.zeros(1, dtype=prev_states.dtype, device=device)
                # N3 — deterministic argmax routing in pass 1. Honest note:
                # this means routing modules' weights (entry_mlp, step_mlp,
                # head_query) are FROZEN in Phase 2 (no gradient through
                # argmax). Refining them under GRPO would require recording
                # routing IDs in pass 1 and forcing them in pass 2 — real
                # refactor, deferred. Phase 2 trains writer mutate_mlp +
                # bridge + cross-attn; routing is locked at Phase-1-end values.
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

    # ── PASS 1 (BATCHED): K rollouts in parallel at BS=K ──

    def _expand_kv_cache_to_K(self, cache, K: int):
        """In-place: expand a BS=1 DynamicCache to BS=K by duplicating
        each layer's keys/values along the batch dim. Returns the same
        cache object after mutation. Caller is responsible for cloning
        the BS=1 cache first if they want to preserve it (we do this in
        `_ar_sample_batch`)."""
        if cache is None:
            return None
        for layer in cache.layers:
            if not layer.is_initialized:
                continue
            layer.keys = layer.keys.expand(K, -1, -1, -1).contiguous()
            layer.values = layer.values.expand(K, -1, -1, -1).contiguous()
        return cache

    @torch.no_grad()
    def _ar_sample_batch(
        self,
        prompt_ids: Tensor,
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        stop_ids: set[int] | None,
        pad_id: int,
        device: torch.device,
        prefill_state: tuple | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """K AR rollouts in parallel at BS=K. Replaces K serial
        `_ar_sample_one` calls.

        Same per-window structure (read at window start → AR within →
        write at window end → trim cache), but all K samples advance
        in lockstep via the BS dimension. Cuts Pass-1 launch count
        by ~K× — the dominant cost in our AR rollout (launch-bound
        single-token forwards).

        Per-sample EOS tracking via a `finished` mask. Once a sample
        emits a stop token, its subsequent token positions get force-
        written to pad_id and logp 0 (compute still runs at that slot
        but those positions are dropped on return — static padding).

        Returns:
            samples:    list of K [n_k] int64 tensors (truncated at first stop).
            logps_old:  list of K [n_k] float tensors aligned to `samples`.
        """
        from src.trajectory_memory.tbptt import _trim_kv_cache
        import copy as _copy

        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window
        K = num_samples

        # ── Prefill (BS=1, shared); clone internally so pass 2 can
        # safely reuse the caller's prefill_state unmodified.
        if prefill_state is not None:
            (ps_1, ph_1, kv_1, lr_1, abs_pos) = prefill_state
            prev_states_1 = ps_1.detach().clone()
            prev_window_hiddens_1 = ph_1.detach().clone() if ph_1 is not None else None
            kv_cache = _copy.deepcopy(kv_1)
            last_real_hidden_1 = lr_1.detach().clone() if lr_1 is not None else None
        else:
            (prev_states_1, prev_window_hiddens_1, kv_cache,
             last_real_hidden_1, abs_pos) = self._prefill_prompt(
                prompt_ids, pad_id=pad_id, device=device,
            )

        # ── Expand prefill state from BS=1 to BS=K.
        prev_states = prev_states_1.expand(K, -1, -1).contiguous()
        prev_window_hiddens = (
            prev_window_hiddens_1.expand(K, -1, -1).contiguous()
            if prev_window_hiddens_1 is not None else None
        )
        last_real_hidden = (
            last_real_hidden_1.expand(K, -1, -1).contiguous()
            if last_real_hidden_1 is not None else None
        )
        kv_cache = self._expand_kv_cache_to_K(kv_cache, K)

        # ── Output buffers (preallocated to max_new_tokens; truncated on return).
        generated = torch.full(
            (K, max_new_tokens), pad_id, dtype=torch.int64, device=device,
        )
        logp_old = torch.zeros((K, max_new_tokens), device=device)
        finished = torch.zeros(K, dtype=torch.bool, device=device)
        n_gen = 0

        stop_ids_t = (
            torch.tensor(list(stop_ids), dtype=torch.int64, device=device)
            if stop_ids else None
        )

        def _update_finished(sampled: Tensor) -> None:
            nonlocal finished
            if stop_ids_t is not None:
                is_stop = (sampled.unsqueeze(-1) == stop_ids_t).any(dim=-1)
                finished = finished | is_stop

        # ── First token: predict from last_real_hidden (per-sample).
        lm_head_dtype = next(self.model.llama.lm_head.parameters()).dtype
        last_hidden = (
            last_real_hidden if last_real_hidden is not None
            else prev_window_hiddens[:, -1:, :]
        ).to(lm_head_dtype)
        first_logits = self.model.llama.lm_head(last_hidden).float()
        scaled = first_logits[:, -1, :] / temperature                # [K, V]
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        first_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [K]
        generated[:, 0] = first_tokens
        logp_old[:, 0] = log_probs.gather(
            1, first_tokens.unsqueeze(-1),
        ).squeeze(-1).detach()
        n_gen = 1
        _update_finished(first_tokens)

        # ── Generation windows.
        while n_gen < max_new_tokens and not finished.all():
            # READ at window start (BS=K).
            if prev_window_hiddens is None:
                prev_window_hiddens = torch.zeros(
                    K, T, cfg.d_lm,
                    dtype=prev_states.dtype, device=device,
                )
            prev_hid_mem = prev_window_hiddens.to(prev_states.dtype)
            read_visited, _ = self.model.read_module(
                prev_hid_mem, prev_states, self.model.manifold, hard=False,
            )
            mem_inject = self.model._mem_inject_layer()
            mem_inject.memory_fn = self.model._build_memory_fn(read_visited)

            window_hiddens_list: list[Tensor] = []
            n_window = min(T, max_new_tokens - n_gen)
            try:
                # AR-generate up to T tokens within this window.
                for _ in range(n_window):
                    last_token_K = generated[:, n_gen - 1:n_gen]   # [K, 1]
                    cache_position = torch.tensor(
                        [abs_pos], dtype=torch.int64, device=device,
                    )
                    base_out = self.model.llama.model(
                        input_ids=last_token_K,
                        past_key_values=kv_cache,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    abs_pos += 1
                    kv_cache = base_out.past_key_values
                    hidden = base_out.last_hidden_state              # [K, 1, d_lm]
                    window_hiddens_list.append(hidden)

                    logits = self.model.llama.lm_head(hidden).float()
                    scaled = logits[:, -1, :] / temperature          # [K, V]
                    log_probs = F.log_softmax(scaled, dim=-1)
                    probs = log_probs.exp()
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # Force pad for already-finished samples; their logp stays 0.
                    sampled = torch.where(
                        finished, torch.full_like(sampled, pad_id), sampled,
                    )
                    logp_sample = log_probs.gather(
                        1, sampled.unsqueeze(-1),
                    ).squeeze(-1)
                    logp_sample = torch.where(
                        finished, torch.zeros_like(logp_sample), logp_sample,
                    )

                    generated[:, n_gen] = sampled
                    logp_old[:, n_gen] = logp_sample.detach()
                    n_gen += 1
                    _update_finished(sampled)

                    if finished.all():
                        break
            finally:
                mem_inject.memory_fn = None

            if not window_hiddens_list:
                break

            # WRITE at window end (BS=K). Mirrors _ar_sample_one's
            # pad-token-forward strategy for partial windows so pass-2
            # TF replay sees the same memory state evolution.
            cur_hiddens = torch.cat(window_hiddens_list, dim=1)      # [K, n_win, d_lm]
            n_win = cur_hiddens.shape[1]

            if n_win < T // 2:
                pad_h = cur_hiddens[:, -1:, :].expand(K, T - n_win, cfg.d_lm)
                prev_window_hiddens = torch.cat([cur_hiddens, pad_h], dim=1)
            else:
                if n_win < T:
                    n_pad = T - n_win
                    pad_t = torch.full(
                        (K, n_pad), pad_id, dtype=torch.int64, device=device,
                    )
                    cache_len_now = (
                        kv_cache.get_seq_length() if kv_cache is not None else 0
                    )
                    pad_cache_position = torch.arange(
                        cache_len_now, cache_len_now + n_pad, device=device,
                    )
                    pad_position_ids = torch.arange(
                        abs_pos, abs_pos + n_pad, device=device,
                    ).unsqueeze(0).expand(K, -1)
                    mem_inject.memory_fn = self.model._build_memory_fn(read_visited)
                    try:
                        pad_out = self.model.llama.model(
                            input_ids=pad_t,
                            past_key_values=kv_cache,
                            cache_position=pad_cache_position,
                            position_ids=pad_position_ids,
                            use_cache=True,
                        )
                    finally:
                        mem_inject.memory_fn = None
                    kv_cache = pad_out.past_key_values
                    abs_pos += n_pad
                    pad_hiddens = pad_out.last_hidden_state
                    cur_hiddens_padded = torch.cat([cur_hiddens, pad_hiddens], dim=1)
                else:
                    cur_hiddens_padded = cur_hiddens

                cur_hiddens_mem = cur_hiddens_padded.to(prev_states.dtype)
                surprise = torch.zeros(K, dtype=prev_states.dtype, device=device)
                new_states, _, _ = self.model.write_module(
                    cur_hiddens_mem, surprise, prev_states, self.model.manifold,
                    hard=False,
                )
                prev_states = new_states
                prev_window_hiddens = cur_hiddens_padded

            kv_cache = _trim_kv_cache(kv_cache, cap)

        # ── Truncate per-sample at first stop.
        samples: list[Tensor] = []
        logps: list[Tensor] = []
        for k in range(K):
            seq = generated[k, :n_gen]
            lp = logp_old[k, :n_gen]
            if stop_ids_t is not None and seq.numel() > 0:
                is_stop = (seq.unsqueeze(-1) == stop_ids_t).any(dim=-1)
                stop_pos = is_stop.nonzero(as_tuple=True)[0]
                if stop_pos.numel() > 0:
                    first_stop = int(stop_pos[0].item())
                    seq = seq[:first_stop + 1]
                    lp = lp[:first_stop + 1]
            samples.append(seq.cpu())
            logps.append(lp.cpu())
        return samples, logps

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
            # Now a 5-tuple — last_real_hidden + abs_pos let us thread
            # cache_abs_pos to forward_window for RoPE correctness on the
            # cached prompt KVs.
            (prev_states, prev_window_hiddens, kv_cache,
             last_real_hidden, cache_abs_pos) = prefill_state
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
            last_real_hidden: Tensor | None = None
            cache_abs_pos = 0
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

            # Predecessor for position win_start (used to compute the
            # first-token logit of this window). Default: last position of
            # the previous window's hiddens.
            #
            # CRITICAL — for the first sample window in shared-prefill
            # mode (win_start == 0 AND prefill_state is not None AND
            # last_real_hidden is provided), the predecessor MUST be
            # `last_real_hidden` (the actual final-prompt-token's hidden,
            # which may have come from a partial trailing window in
            # _prefill_prompt). pass 1's `_ar_sample_one` samples the
            # first generated token from `last_real_hidden`; if pass 2
            # uses `prev_window_hiddens[-1:]` instead (the last position
            # of the last FULL T-window), it conditions on a different
            # state. The IS-ratio numerator/denominator then compute logp
            # for the same action under different states — meaningless
            # surrogate. Only an issue when prompt length is not a
            # multiple of T_window.
            if (
                win_start == 0
                and prefill_state is not None
                and last_real_hidden is not None
            ):
                last_prev_logit = last_real_hidden
            else:
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
                cache_abs_pos=cache_abs_pos,
                force_surprise=0.0 if in_shared_prefill_mode else None,
            )
            # In KV-cache mode forward_window's `logits` is [1, T, V]
            # aligned to the new window's T positions. logit at index i
            # predicts win[i+1] (i's hidden contains the context to
            # predict the next token).
            #
            # For sample positions p in [first_sample_offset, n_full):
            #   - if p > win_start: predictor logit = logits[0, p - win_start - 1]
            #   - if p == win_start: predictor logit = lm_head(last_prev_logit)
            #     (the previous window's last hidden). If last_prev_logit
            #     is None (very first window of sequence), drop position 0.
            logits = out["logits"]                               # [1, T, V]

            # Phase 2 perf fix (#5) — vectorize: stack all predictor
            # logits for THIS window's sample positions into [n_pred, V],
            # one log_softmax + one gather per window instead of one per
            # token. Old loop was 1024 small softmaxes/step at K=16
            # T_gen=64 — now n_windows softmaxes/step.
            sample_lo = max(win_start, first_sample_offset)
            sample_hi = win_end
            if sample_lo < sample_hi:
                # Targets for sample positions [sample_lo, sample_hi).
                target_ids = torch.tensor(
                    seq_to_walk[sample_lo:sample_hi],
                    dtype=torch.int64, device=device,
                )                                                # [n_pred]
                if sample_lo == win_start:
                    # First sample position needs last_prev_logit
                    # (predecessor is from previous window).
                    if last_prev_logit is None:
                        # Very first token of sequence — drop it.
                        target_ids = target_ids[1:]
                        if target_ids.numel() == 0:
                            # carry state below; nothing to log this window
                            pred_logits = None
                        else:
                            pred_logits = logits[0, :target_ids.numel(), :]
                    else:
                        # Compute first-position logit from last_prev_logit.
                        if self.model.llama is None:
                            first_logit = (
                                last_prev_logit.to(self.model._test_proj.dtype)
                                @ self.model._test_lm_head
                            )[0, 0:1, :]                          # [1, V]
                        else:
                            lm_head_dtype = next(
                                self.model.llama.lm_head.parameters()
                            ).dtype
                            first_logit = self.model.llama.lm_head(
                                last_prev_logit.to(lm_head_dtype)
                            )[0, 0:1, :]                          # [1, V]
                        n_after_first = target_ids.numel() - 1
                        if n_after_first > 0:
                            rest_logits = logits[0, :n_after_first, :]
                            pred_logits = torch.cat(
                                [first_logit, rest_logits], dim=0,
                            )                                     # [n_pred, V]
                        else:
                            pred_logits = first_logit             # [1, V]
                else:
                    # All sample positions are p > win_start; predictor
                    # index = p - win_start - 1.
                    first_idx = sample_lo - win_start - 1
                    pred_logits = logits[0, first_idx:first_idx + target_ids.numel(), :]

                if pred_logits is not None and pred_logits.shape[0] > 0:
                    # R2: selective log-softmax — gather + logsumexp
                    # instead of full log_softmax + gather. Avoids
                    # transiently materializing the [n_pred, V] log_softmax
                    # output tensor (V=128k → ~25 MB per window of n_pred
                    # tokens). Math is identical:
                    #   logp(t) = scaled[t] - logsumexp(scaled, dim=-1)
                    scaled = (pred_logits / temperature).float()
                    target_logits = scaled.gather(
                        -1, target_ids.unsqueeze(-1),
                    ).squeeze(-1)                                  # [n_pred]
                    lse = torch.logsumexp(scaled, dim=-1)          # [n_pred]
                    logps = target_logits - lse                    # [n_pred]
                    sample_logps.append(logps)

            # Carry state — NO DETACH, autograd graph stays alive across
            # windows of a sample. KV cache likewise. Advance abs_pos by
            # window length so subsequent windows get correct RoPE.
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            kv_cache = out.get("new_past_key_values", kv_cache)
            cache_abs_pos = out.get("new_cache_abs_pos", cache_abs_pos + T)
            kv_cache = _trim_kv_cache(kv_cache, cap)

        if not sample_logps:
            return torch.zeros(0, device=device)
        # Each entry is now a [n_pred_in_window] tensor (post-#5
        # vectorization), not a scalar — use cat not stack.
        return torch.cat(sample_logps)

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
