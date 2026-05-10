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
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

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
    def _compute_ref_logps(
        self,
        prompt_ids: Tensor,
        sample_ids: Tensor,
        *,
        temperature: float,
        pad_id: int,
        device: torch.device,
    ) -> Tensor | None:
        """No-grad TF-forward through (prompt + sample) under the REFERENCE
        policy. Returns per-sample-token logp.

        Implementation: temporarily swap the trainable params to the ref
        snapshot, call `_tf_compute_sample_logp` (no_grad context), restore
        the current params. Two memcpy passes per call (~66 MB each at
        medium config — trivial).
        """
        if self.ref_state is None or self.kl_coef == 0:
            return None

        # Save current trainable params.
        saved = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        try:
            # Swap in ref params.
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(self.ref_state[name])
            # Compute under ref policy. _tf_compute_sample_logp returns
            # tensors with grad enabled by default; we're inside @no_grad
            # so the returned logp tensors are leaf-no-grad.
            ref_logps = self._tf_compute_sample_logp(
                prompt_ids=prompt_ids, sample_ids=sample_ids,
                temperature=temperature, pad_id=pad_id, device=device,
            )
            return ref_logps.detach() if ref_logps.numel() > 0 else None
        finally:
            # Restore current params.
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(saved[name])

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

        # ── PASS 2: TF logp + PPO-clip surrogate + KL to reference ───
        # TF-forward through (prompt + sample) under CURRENT policy with
        # grad → sample_logps_new. Compute IS ratio against pass-1 logp,
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

        for sample, logp_old, adv in zip(samples, samples_logp_old, advantages):
            if sample.numel() == 0:
                continue
            sample_logps_new = self._tf_compute_sample_logp(
                prompt_ids=prompt_ids,
                sample_ids=sample.to(device),
                temperature=temperature,
                pad_id=pad_id,
                device=device,
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

            # KL term against reference policy (no_grad ref forward).
            if self.ref_state is not None and self.kl_coef > 0:
                ref_logps = self._compute_ref_logps(
                    prompt_ids=prompt_ids,
                    sample_ids=sample.to(device),
                    temperature=temperature, pad_id=pad_id, device=device,
                )
                if ref_logps is not None and ref_logps.numel() == n_new:
                    # John Schulman's K3 estimator: KL ≈ exp(log r) - log r - 1
                    # where log r = logp_new - logp_ref. Always non-negative,
                    # low-variance, exact in expectation.
                    log_r = sample_logps_new - ref_logps
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
    ) -> tuple[Tensor, Tensor | None, object]:
        """Walk forward_window over the FULL prompt in T_window-sized
        windows, accumulating manifold state AND populating an HF KV cache.
        Returns (prev_states, prev_window_hiddens, kv_cache).

        Long prompts (NarrativeQA 8K, WildChat 4K+) were previously not
        being properly written into memory in the old per-token AR loop;
        prefill makes the entire prompt's content visible to the manifold
        before generation starts.

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

        for win_start in range(0, n, T):
            win = toks[win_start : win_start + T]
            if len(win) < T:
                # Right-pad partial last window — pad positions get
                # encoded but their hiddens are discarded later (we're
                # done with this prompt).
                win = win + [pad_id] * (T - len(win))
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
                hard_routing=True,
                past_key_values=kv_cache,
                use_kv_cache=True,
                last_prev_logit_hidden=last_prev_logit,
            )
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            kv_cache = out.get("new_past_key_values", kv_cache)
            kv_cache = _trim_kv_cache(kv_cache, cap)

        return prev_states, prev_window_hiddens, kv_cache

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
        prev_states, prev_window_hiddens, kv_cache = self._prefill_prompt(
            prompt_ids, pad_id=pad_id, device=device,
        )

        generated: list[int] = []
        # `logp_old` records logp_pi_old(token) at sampling time, used by
        # pass 2's PPO-clip importance-sampling ratio. Detached / no_grad
        # — pass 2 recomputes logp_new under autograd.
        logp_old_list: list[Tensor] = []

        # First generated token: predict from the LAST cached hidden
        # (= prev_window_hiddens[:, -1:, :], the prompt's last position).
        # prev_window_hiddens was cast to manifold's fp32 dtype inside
        # forward_window; lm_head expects Llama's native dtype (bf16).
        lm_head_dtype = next(self.model.llama.lm_head.parameters()).dtype
        last_hidden = prev_window_hiddens[:, -1:, :].to(lm_head_dtype)
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
            prev_hid_mem = prev_window_hiddens.to(prev_states.dtype)
            read_visited, _ = self.model.read_module(
                prev_hid_mem, prev_states, self.model.manifold, hard=True,
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
                    base_out = self.model.llama.model(
                        input_ids=input_ids,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
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
                new_states, _, _ = self.model.write_module(
                    cur_hiddens_mem, surprise, prev_states, self.model.manifold,
                    hard=True,
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
    ) -> Tensor:
        """TF-forward through (prompt + sample) with KV cache, return
        per-sample-token logp under `softmax(logits / temperature)` —
        matches pass 1's sampling distribution.

        Memory state carries through `write_module` across windows
        without any detach, so the autograd graph from each sample-token
        logp reaches all trainable params (bridge, read_module,
        write_module, manifold). KV cache also carries with grad —
        backward through cached KV tensors propagates to the prior
        window's bridge-after-layer-8 trainable params.

        Returns: [n_sample_tokens] tensor of logps with grad.
        """
        from src.trajectory_memory.tbptt import _trim_kv_cache

        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window

        prompt = prompt_ids.flatten().tolist()
        sample = sample_ids.flatten().tolist()
        full_seq = prompt + sample
        n_full = len(full_seq)
        n_prompt = len(prompt)

        prev_states = self.model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None
        kv_cache: object | None = None
        sample_logps: list[Tensor] = []

        for win_start in range(0, n_full, T):
            win_end = min(win_start + T, n_full)
            win = full_seq[win_start:win_end]
            n_real = len(win)
            if n_real < T:
                win = win + [pad_id] * (T - n_real)
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
                hard_routing=True,
                past_key_values=kv_cache,
                use_kv_cache=True,
                last_prev_logit_hidden=last_prev_logit,
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

            for p in range(max(win_start, n_prompt), win_end):
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
                logp_p = F.log_softmax(scaled, dim=-1)[full_seq[p]]
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
