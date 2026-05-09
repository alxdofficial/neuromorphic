"""Phase 2 (AR GRPO) trainer for Wave 3 (verifiable-reward) and Wave 4
(long-session). Implements GRPO: sample J responses per prompt, score each,
compute group-relative advantage, policy gradient (plan §4.7).

The `Phase2Trainer` class is the proper training harness with grad
clipping, LR scheduling, metrics dict, and checkpoint integration. The
legacy `grpo_step` free function is preserved for back-compat.

NOTE on rollout efficiency: the current `grpo_rollout` does one
`forward_window` call per generated token — correct but slow. A real
training run should batch generation per-T_window (one read+predict+write
cycle per window of generated tokens, with an HF `generate()`-style KV
cache inside each window). Wired as a follow-up; see `_rollout_one`.
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
        samples, logps = self._rollout(
            prompt_ids,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_ids=stop_ids,
            pad_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
        )

        decoded = [tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in samples]
        rewards = [
            compute_reward(reward_kind, c, gold=gold, meta=meta) for c in decoded
        ]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards_t)

        self.optimizer.zero_grad()
        policy_loss = torch.zeros((), device=prompt_ids.device)
        any_grad = False
        for adv, lp in zip(advantages, logps):
            if lp.numel() == 0:
                continue
            policy_loss = policy_loss + (-adv * lp.sum())
            any_grad = True
        policy_loss = policy_loss / max(num_samples, 1)
        # If no samples produced any tokens (e.g., all hit eos immediately, or
        # test-mode rollout with V=1 always sampling index 0), skip the
        # optimizer update — there's nothing to learn from this step.
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

    # ── rollout (batched-per-token; correct but slow) ─────────────────

    def _rollout(
        self,
        prompt_ids: Tensor,
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        stop_ids: set[int] | None = None,
        pad_id: int = 0,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Sample `num_samples` AR responses from a single prompt.

        Returns (samples, token_logps). See module docstring on efficiency.
        """
        cfg = self.model.cfg
        device = next(self.model.parameters()).device

        samples: list[Tensor] = []
        logps: list[Tensor] = []

        for _ in range(num_samples):
            sample, lp = self._rollout_one(
                prompt_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, stop_ids=stop_ids,
                pad_id=pad_id, device=device,
            )
            samples.append(sample)
            logps.append(lp)
        return samples, logps

    def _rollout_one(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        stop_ids: set[int] | None,
        pad_id: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """One AR sample. Token-by-token (slow but correct)."""
        cfg = self.model.cfg
        prev_states = self.model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None

        cur_tokens = prompt_ids.tolist() if prompt_ids.dim() == 1 else prompt_ids[0].tolist()
        generated: list[int] = []
        per_tok_logp: list[Tensor] = []

        for _ in range(max_new_tokens):
            lm_input = torch.tensor(
                cur_tokens[-cfg.effective_lm_context:],
                dtype=torch.int64, device=device,
            ).unsqueeze(0)
            if lm_input.shape[1] < cfg.T_window:
                pad_n = cfg.T_window - lm_input.shape[1]
                # Left-pad with the tokenizer's pad token (= eos_token for
                # Llama-3, id 128001). Earlier `value=0` padded with `!`
                # (token 0), which corrupted the LM's leading context.
                lm_input = F.pad(lm_input, (pad_n, 0), value=pad_id)

            out = self.model.forward_window(
                lm_input_ids=lm_input,
                prev_window_hiddens=prev_window_hiddens,
                prev_states=prev_states,
                target_mask=None,
                hard_routing=True,
            )
            logits = out["logits"][:, -1, :].float()
            # Sample under the TEMPERED distribution and record logp under
            # the same distribution (policy-gradient correctness — when
            # temperature != 1.0, the sampling distribution and the recorded
            # logp must match, else the policy gradient is biased).
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).item()
            logp = F.log_softmax(scaled_logits, dim=-1)[0, sampled]
            per_tok_logp.append(logp)
            generated.append(sampled)
            cur_tokens.append(sampled)

            if stop_ids is not None and sampled in stop_ids:
                break

            prev_states = out["new_states"].detach()
            prev_window_hiddens = out["current_hiddens"].detach()

        return (
            torch.tensor(generated, dtype=torch.int64),
            torch.stack(per_tok_logp) if per_tok_logp else torch.zeros(0),
        )

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
):
    """Compatibility shim around Phase2Trainer._rollout."""
    trainer = Phase2Trainer(model, optimizer=torch.optim.SGD([torch.zeros(1)], lr=0))
    return trainer._rollout(
        prompt_ids, num_samples=num_samples,
        max_new_tokens=max_new_tokens, temperature=temperature, eos_id=eos_id,
    )


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
