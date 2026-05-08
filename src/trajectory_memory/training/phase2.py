"""Phase 2 (AR GRPO) trainer for Wave 3 (verifiable-reward) and Wave 4
(long-session). Implements GRPO: sample J responses per prompt, score each,
compute group-relative advantage, policy gradient through both the LM
token-sampling and the trajectory-routing decisions (plan §4.7).

This is the simplest viable single-pass GRPO trainer — no rewind-and-
re-forward (§4.6 fallback). For multi-turn long sessions (Wave 4), we
loop over assistant turns within a session, using TurnPair semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.rewards import compute_reward


def grpo_rollout(
    model: IntegratedLM,
    prompt_ids: Tensor,                  # [T_prompt] int64 — single example
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    eos_id: int | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """Sample `num_samples` AR responses from a single prompt.

    Returns:
        sampled_ids:  list of [T_response] tensors per sample
        token_logps:  list of [T_response] log-prob tensors (for policy gradient)

    Memory state is RESET per sample (per-rollout independence). The
    forward path is per-token AR generation with the trajectory-memory
    cycle active per window boundary.

    NOTE: This is a *minimal* implementation — it does AR generation by
    running the model token-by-token through forward_window. For
    production-scale GRPO, replace with batched HF `generate()` plus
    a custom hook that fires the per-window read/write cycle.
    """
    cfg = model.cfg
    device = next(model.parameters()).device
    eos_id = eos_id if eos_id is not None else cfg.T_window  # placeholder; replace with tokenizer.eos
    samples: list[Tensor] = []
    logps: list[Tensor] = []

    for _ in range(num_samples):
        # State for this sample.
        prev_states = model.manifold.reset_states(batch_size=1)
        prev_window_hiddens: Tensor | None = None
        prev_lm_context: Tensor | None = None

        # Generate AR. We accumulate tokens and run forward_window every
        # T_window steps. (Production should use KV-cache for efficiency.)
        # For this minimal version we tokenize-then-call once per
        # T_window-aligned chunk.

        generated: list[int] = []
        per_tok_logp: list[Tensor] = []
        prompt_list = prompt_ids.tolist()

        # Pad prompt to start at a T_window boundary so the rolling
        # context is well-defined. We just generate from the prompt
        # forward; the first window is the prompt's last T_window tokens.
        # This is a simplified rollout.
        cur_tokens = list(prompt_list)

        for _step in range(max_new_tokens):
            # Build the LM input: last min(len(cur_tokens), effective_lm_context)
            lm_input = torch.tensor(
                cur_tokens[-cfg.effective_lm_context:],
                dtype=torch.int64, device=device,
            ).unsqueeze(0)
            if lm_input.shape[1] < cfg.T_window:
                # Need at least T_window tokens for forward_window.
                # Pad on the left with pad_id (0 by convention).
                pad_n = cfg.T_window - lm_input.shape[1]
                lm_input = F.pad(lm_input, (pad_n, 0), value=0)

            with torch.set_grad_enabled(True):
                out = model.forward_window(
                    lm_input_ids=lm_input,
                    prev_window_hiddens=prev_window_hiddens,
                    prev_states=prev_states,
                    target_mask=None,
                    hard_routing=True,
                )
            logits = out["logits"][:, -1, :].float()              # [1, V] for the next token
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).item()
            logp = F.log_softmax(logits, dim=-1)[0, sampled]
            per_tok_logp.append(logp)
            generated.append(sampled)
            cur_tokens.append(sampled)

            if sampled == eos_id:
                break

            # Update carries (only when we cross a window boundary —
            # with this simple per-token loop we update every step,
            # which is O(N) windows for an N-token response. A real
            # implementation only crosses windows every T_window tokens).
            prev_states = out["new_states"].detach()
            prev_window_hiddens = out["current_hiddens"].detach()
            prev_lm_context = lm_input[:, -(cfg.effective_lm_context - cfg.T_window):]

        samples.append(torch.tensor(generated, dtype=torch.int64))
        logps.append(torch.stack(per_tok_logp) if per_tok_logp else torch.zeros(0))

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
    """One GRPO training step on a single prompt.

    Algorithm (plan §4.7):
      1. Sample J responses from the current model.
      2. Score each with `compute_reward(reward_kind, ...)`.
      3. Group-relative advantage: a_i = r_i - mean(r) (no value baseline).
      4. Policy loss = -mean_i (advantage_i * sum_t logp_i_t).
      5. Optional KL to a reference model (skipped here — kl_coef=0).
    """
    samples, logps = grpo_rollout(
        model, prompt_ids,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_token_id,
    )

    # Decode samples and score.
    decoded = [tokenizer.decode(s.tolist(), skip_special_tokens=True) for s in samples]
    rewards = [
        compute_reward(reward_kind, c, gold=gold, meta=meta) for c in decoded
    ]
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    # Group-relative advantage (DeepSeek-style).
    mean_r = rewards_t.mean()
    std_r = rewards_t.std(unbiased=False).clamp_min(1e-6)
    advantages = (rewards_t - mean_r) / std_r              # [J]

    # Policy loss: maximize advantage * sum(logp_per_token).
    # logps[i] has shape [T_response_i]; sum over T to get a scalar per sample.
    policy_loss = torch.zeros((), device=samples[0].device)
    for adv, lp in zip(advantages, logps):
        if lp.numel() == 0:
            continue
        policy_loss = policy_loss + (-adv * lp.sum())
    policy_loss = policy_loss / max(num_samples, 1)

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return {
        "policy_loss": float(policy_loss.detach()),
        "rewards": rewards,
        "advantages": advantages.tolist(),
        "decoded": decoded,
    }
