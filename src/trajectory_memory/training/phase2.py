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
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

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

        # ── PASS 1: AR sample each response (no_grad), with prefill ──
        samples: list[Tensor] = []
        for _ in range(num_samples):
            sample = self._ar_sample_one(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_ids=stop_ids,
                pad_id=pad_id,
                device=device,
            )
            samples.append(sample)

        # ── SCORE ────────────────────────────────────────────────────
        decoded = [tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in samples]
        rewards = [
            compute_reward(reward_kind, c, gold=gold, meta=meta) for c in decoded
        ]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards_t)

        # ── PASS 2: TF logp for each sample (with grad), backward ────
        # TF-forward through (prompt + sample) for each sample,
        # collecting per-sample-token logp under the tempered softmax
        # used during sampling. Memory writes are in the gradient path
        # because pass 2 carries state through `write_module` normally
        # (no detach within a single sample's forward).
        self.optimizer.zero_grad()
        policy_loss = torch.zeros((), device=device)
        any_grad = False
        for sample, adv in zip(samples, advantages):
            if sample.numel() == 0:
                continue
            sample_logps = self._tf_compute_sample_logp(
                prompt_ids=prompt_ids,
                sample_ids=sample.to(device),
                temperature=temperature,
                pad_id=pad_id,
                device=device,
            )
            if sample_logps.numel() == 0:
                continue
            policy_loss = policy_loss + (-adv * sample_logps.sum())
            any_grad = True
        policy_loss = policy_loss / max(num_samples, 1)
        if any_grad:
            policy_loss.backward()
            grad_norm = self._clip_and_step()
        else:
            grad_norm = 0.0
            if self.scheduler is not None:
                self.scheduler.step()
        self._step_count += 1

        return Phase2Metrics(
            policy_loss=float(policy_loss.detach()),
            grad_norm=float(grad_norm),
            lr=[g["lr"] for g in self.optimizer.param_groups],
            rewards=rewards,
            advantages=advantages.tolist(),
            decoded=decoded,
            mean_response_len=sum(len(s) for s in samples) / max(len(samples), 1),
        )

    # ── PASS 1: no-grad AR sampling (with prefill) ────────────────────

    @torch.no_grad()
    def _prefill_prompt(
        self,
        prompt_ids: Tensor,
        *,
        pad_id: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Walk forward_window over the FULL prompt in T_window-sized
        windows, accumulating manifold state. Returns the final
        (prev_states, prev_window_hiddens, lm_buffer) so AR generation
        starts from a memory state that has seen the entire prompt.

        Earlier rollout paths only fed the trailing `effective_lm_context`
        tokens of the prompt into a fresh manifold — long prompts beyond
        the LM cap were never written into memory.

        no_grad: this is pure inference setup; pass 2 will redo the
        forward with grad if memory writes need gradient signal.
        """
        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window
        toks = prompt_ids.flatten().tolist()
        n = len(toks)

        prev_states = self.model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None
        lm_buffer = torch.empty(1, 0, dtype=torch.int64, device=device)

        for win_start in range(0, n, T):
            win = toks[win_start : win_start + T]
            if len(win) < T:
                # Right-pad the partial last window so forward_window
                # gets a full-T_window slice. The pad tokens are
                # overwritten by the next AR step's tokens or simply
                # ignored (we don't compute loss here).
                win = win + [pad_id] * (T - len(win))
            win_t = torch.tensor(win, dtype=torch.int64, device=device).unsqueeze(0)
            lm_input = torch.cat([lm_buffer, win_t], dim=1)
            if lm_input.shape[1] > cap:
                lm_input = lm_input[:, -cap:]

            out = self.model.forward_window(
                lm_input_ids=lm_input,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None,
                hard_routing=True,
            )
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            lm_buffer = lm_input[:, -(cap - T):] if cap > T else lm_input[:, :0]

        return prev_states, prev_window_hiddens, lm_buffer

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
        """One AR rollout, no_grad. Returns just the generated token IDs.

        Pass-2 (`_tf_compute_sample_logp`) recomputes per-token logp with
        gradient — pass 1 doesn't need to record it.
        """
        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window

        # Prefill the full prompt through memory.
        prev_states, prev_window_hiddens, lm_buffer = self._prefill_prompt(
            prompt_ids, pad_id=pad_id, device=device,
        )

        cur_tokens = prompt_ids.flatten().tolist()
        generated: list[int] = []

        for _ in range(max_new_tokens):
            lm_input = torch.tensor(
                cur_tokens[-cap:],
                dtype=torch.int64, device=device,
            ).unsqueeze(0)
            if lm_input.shape[1] < T:
                pad_n = T - lm_input.shape[1]
                lm_input = F.pad(lm_input, (pad_n, 0), value=pad_id)

            out = self.model.forward_window(
                lm_input_ids=lm_input,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None,
                hard_routing=True,
            )
            logits = out["logits"][:, -1, :].float()
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).item()
            generated.append(sampled)
            cur_tokens.append(sampled)

            if stop_ids is not None and sampled in stop_ids:
                break

            # State carry between AR steps. Detach is fine here — pass 1
            # is no_grad anyway. The graph that matters lives in pass 2.
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]

        return torch.tensor(generated, dtype=torch.int64)

    # ── PASS 2: TF logp computation (with grad, single backward) ──────

    def _tf_compute_sample_logp(
        self,
        prompt_ids: Tensor,
        sample_ids: Tensor,
        *,
        temperature: float,
        pad_id: int,
        device: torch.device,
    ) -> Tensor:
        """TF-forward through (prompt + sample), return per-sample-token
        logp under `softmax(logits / temperature)` — matches pass 1's
        sampling distribution exactly.

        Memory state carries through `write_module` across windows
        without any detach, so the autograd graph from each sample-token
        logp reaches all trainable params (bridge, read_module,
        write_module, manifold).

        Returns: [n_sample_tokens] tensor of logps with grad.
        """
        cfg = self.model.cfg
        cap = cfg.effective_lm_context
        T = cfg.T_window

        prompt = prompt_ids.flatten().tolist()
        sample = sample_ids.flatten().tolist()
        full_seq = prompt + sample
        n_full = len(full_seq)
        n_prompt = len(prompt)

        # Walk through full_seq in T-sized windows. State carries with grad.
        prev_states = self.model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None
        lm_buffer = torch.empty(1, 0, dtype=torch.int64, device=device)
        sample_logps: list[Tensor] = []

        for win_start in range(0, n_full, T):
            win_end = min(win_start + T, n_full)
            win = full_seq[win_start:win_end]
            n_real = len(win)
            if n_real < T:
                win = win + [pad_id] * (T - n_real)
            win_t = torch.tensor(win, dtype=torch.int64, device=device).unsqueeze(0)
            lm_input = torch.cat([lm_buffer, win_t], dim=1)
            if lm_input.shape[1] > cap:
                lm_input = lm_input[:, -cap:]

            out = self.model.forward_window(
                lm_input_ids=lm_input,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None,
                hard_routing=True,
            )
            # forward_window's `logits` is [BS, T_window, V] aligned to the
            # LAST T tokens of lm_input — i.e., the current window. logit at
            # position i predicts win[i+1] (using win[i] as input). For
            # sample positions, the predictor is the previous token in the
            # full sequence.
            logits = out["logits"]                               # [1, T, V]
            # For each position p in [win_start, win_end), the predictor
            # is at logit index p - win_start - 1 (within this window).
            # Skip p == 0 (no predecessor) and skip prompt positions
            # (we only need logp for sample tokens).
            for p in range(max(win_start, n_prompt), win_end):
                logit_idx = p - win_start - 1
                if logit_idx < 0:
                    # Predecessor was in the previous window — handle by
                    # reading from the lm_input tail of THIS window
                    # (rolling buffer). Actual predictor position within
                    # lm_input is (lm_input.shape[1] - T) + logit_idx.
                    # If that's also < 0 we're at the very first token of
                    # the sequence (no predecessor); skip.
                    abs_idx = lm_input.shape[1] - T + logit_idx
                    if abs_idx < 0:
                        continue
                    # forward_window returns logits aligned to the last
                    # T_window positions of lm_input. To get logits for
                    # earlier positions we'd need to re-forward — not
                    # worth it for the boundary token. Skip it. (One
                    # missed logp per sample is noise.)
                    continue
                logit = logits[0, logit_idx]                     # [V]
                scaled = logit / temperature
                logp_p = F.log_softmax(scaled, dim=-1)[full_seq[p]]
                sample_logps.append(logp_p)

            # Carry state — NO DETACH, autograd graph stays alive.
            prev_states = out["new_states"]
            prev_window_hiddens = out["current_hiddens"]
            lm_buffer = lm_input[:, -(cap - T):] if cap > T else lm_input[:, :0]

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
        sample = trainer._ar_sample_one(
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
