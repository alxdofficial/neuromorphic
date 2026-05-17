"""Wave3TrainerV2 — GRPO trainer for the v2 vocabulary-trajectory model.

Phase 2 RL training: the model rolls out J responses per prompt via
auto-regressive sampling, each response is scored against a gold
answer (rule-based / exact-match / BERT cosine — see `_rewards.py`),
and the policy is updated via Group-Relative Policy Optimization:

    advantage[j] = (reward[j] - mean(rewards)) / (std(rewards) + ε)
    L_pg = -E_j[ advantage[j] · sum_t log π_θ(token_t | prefix_t) ]
    L_kl = β · KL(π_θ || π_ref)            # ref = snapshot at run start
    L = L_pg + L_kl + load_balance_coef · aux_lb + z_loss_coef · aux_z

Streaming behavior parallels Wave 2: the prompt is processed as one
window, its hiddens go into mem_inject during generation. Manifold
state is snapshotted around each prompt's rollout so independent
samples don't pollute each other's memory.

This is a minimum-viable trainer: single-example rollout (no batched
prompts), no prompt-length bucketing, no AR cudagraph optimization.
Wire up incrementally as it proves itself.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from torch.optim import Optimizer

from src.trajectory_memory_v2.integrated_lm import IntegratedLMV2
from src.trajectory_memory_v2._rewards import compute_reward


@dataclass
class V2Wave3Metrics:
    loss: float = 0.0
    pg_loss: float = 0.0
    kl_loss: float = 0.0
    aux_load_balance: float = 0.0
    aux_z_loss: float = 0.0
    mean_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    reward_std: float = 0.0
    mean_response_len: float = 0.0
    grad_norm: float = 0.0
    n_active_edges: int = 0
    mean_effectiveness: float = 0.0
    mean_visit_count: float = 0.0
    per_source: dict[str, float] = field(default_factory=dict)


class Wave3TrainerV2:
    """GRPO trainer for prompt → J rollouts → group-relative advantage.

    Usage:
        trainer = Wave3TrainerV2(model, optimizer, pad_token_id=tok.pad_token_id,
                                  tokenizer=tok, n_rollouts=4, max_response_tokens=128)
        trainer.set_reference_state()    # snapshot current model as π_ref
        metrics = trainer.step(example)  # one prompt/rollout/update
    """

    def __init__(
        self,
        model: IntegratedLMV2,
        optimizer: Optimizer,
        *,
        pad_token_id: int,
        tokenizer,
        scheduler: Any | None = None,
        grad_clip: float | None = 1.0,
        n_rollouts: int = 4,
        max_response_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        kl_coef: float = 0.01,
        load_balance_coef: float | None = None,
        z_loss_coef: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.n_rollouts = n_rollouts
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kl_coef = kl_coef
        cfg = model.cfg
        self.load_balance_coef = load_balance_coef if load_balance_coef is not None else cfg.load_balance_coef
        self.z_loss_coef = z_loss_coef if z_loss_coef is not None else cfg.z_loss_coef
        self._step_count = 0
        self._ref_params: Optional[dict[str, Tensor]] = None

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

    @torch.no_grad()
    def set_reference_state(self) -> None:
        """Snapshot current model PARAMETERS as the reference policy π_ref
        for KL regularization. Buffers (manifold edge state, step_counter,
        etc.) are intentionally excluded so the reference forward sees the
        LIVE memory state — the policy is being regularized, not the
        memory contents.

        Call once at the start of training (the original GRPO recipe keeps
        the reference fixed; some variants refresh it periodically).
        """
        # Detached + cloned so the snapshot doesn't share storage with
        # live params (which would defeat the purpose under any in-place
        # update by the optimizer).
        self._ref_params: dict[str, Tensor] = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
        }

    def _truncate_and_pad(self, ids: list[int], target_len: int) -> list[int]:
        if len(ids) >= target_len:
            return ids[:target_len]
        return ids + [self.pad_token_id] * (target_len - len(ids))

    @torch.no_grad()
    def _rollout(
        self,
        prompt_ids: list[int],
    ) -> tuple[list[list[int]], Tensor]:
        """Generate `n_rollouts` responses via stochastic AR sampling.

        Returns:
            response_ids_per_rollout: list of J token-id lists (post-prompt)
            full_log_probs:           [J, max_resp] log-prob of each generated
                                       token under the current policy (for the
                                       reference-policy KL term and as a
                                       sanity check; the gradient-bearing
                                       log-probs are recomputed in `step`)
        """
        device = next(self.model.parameters()).device
        T = self.model.cfg.T_window
        prompt_t = torch.tensor(
            self._truncate_and_pad(prompt_ids, T),
            dtype=torch.long, device=device,
        ).unsqueeze(0)   # [1, T]
        prompt_mask = (prompt_t != self.pad_token_id)

        # 1. Forward the prompt window — populates memory via write trajectory
        #    and runs Llama. We need the final-token KV state to begin AR.
        #    For the v2 architecture, the simplest approach is to use the
        #    HF .generate() API on the underlying llama after configuring
        #    mem_inject's memory_fn from the read trajectory.
        prompt_len = min(len(prompt_ids), T)
        # Run forward_window in passage mode — populates writes + Llama hiddens
        # (memory_fn is set inside forward_window).
        _ = self.model.forward_window(
            lm_input_ids=prompt_t,
            attention_mask=prompt_mask,
            prev_window_hiddens=None,
            hard_routing=False,
            write_mode="passage",
        )

        # 2. AR generation: J independent samples via HF generate. With
        #    mem_inject's memory_fn already installed, the LM can read from
        #    memory at every generated token.
        gen_input = prompt_t[:, :prompt_len]  # [1, prompt_len]
        gen_attn = (gen_input != self.pad_token_id)
        responses: list[list[int]] = []
        for _ in range(self.n_rollouts):
            out = self.model.llama.generate(
                gen_input,
                attention_mask=gen_attn,
                max_new_tokens=self.max_response_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.pad_token_id,
            )
            # Strip the prompt prefix; keep only generated tokens.
            gen_only = out[0, prompt_len:].tolist()
            # Drop trailing pads.
            while gen_only and gen_only[-1] == self.pad_token_id:
                gen_only.pop()
            responses.append(gen_only)

        # Placeholder log-probs (recomputed differentiably in `step`).
        return responses, torch.zeros(self.n_rollouts, self.max_response_tokens)

    def _score_rollouts(
        self,
        responses: list[list[int]],
        gold_ids: list[int],
        reward_kind: str,
        meta: dict,
    ) -> list[float]:
        """Decode each response and compute its reward."""
        gold_text = self.tokenizer.decode(gold_ids, skip_special_tokens=True)
        rewards = []
        for resp_ids in responses:
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            r = compute_reward(reward_kind, text, gold=gold_text, meta=meta)
            rewards.append(float(r))
        return rewards

    def _prepare_lp_inputs(
        self,
        prompt_ids_tensor: Tensor,        # [1, prompt_len]
        response_ids: list[int],          # generated tokens
    ) -> tuple[Optional[Tensor], Optional[Tensor], int, int]:
        """Build the (full_ids, full_mask, prompt_len, resp_len_kept) tuple
        used by both current-policy and reference-policy log-prob compute.
        Returns (None, None, 0, 0) if there's nothing to score."""
        device = prompt_ids_tensor.device
        if not response_ids:
            return None, None, 0, 0
        T = self.model.cfg.T_window
        full_ids = torch.cat([
            prompt_ids_tensor[0],
            torch.tensor(response_ids, dtype=torch.long, device=device),
        ]).unsqueeze(0)
        prompt_len = prompt_ids_tensor.shape[1]
        if full_ids.shape[1] > T:
            full_ids = full_ids[:, :T]
            resp_len_kept = T - prompt_len
            if resp_len_kept <= 0:
                return None, None, 0, 0
        else:
            resp_len_kept = len(response_ids)
        full_mask = (full_ids != self.pad_token_id)
        return full_ids, full_mask, prompt_len, resp_len_kept

    def _gather_response_logprobs(
        self, logits: Tensor, full_ids: Tensor, prompt_len: int, resp_len_kept: int,
    ) -> Tensor:
        """Slice logits → log-probs of the response tokens."""
        resp_slice_logits = logits[0, prompt_len - 1 : prompt_len - 1 + resp_len_kept, :]
        resp_targets = full_ids[0, prompt_len : prompt_len + resp_len_kept]
        log_probs = F.log_softmax(resp_slice_logits.float(), dim=-1)
        return log_probs.gather(-1, resp_targets.unsqueeze(-1)).squeeze(-1)

    def _compute_log_probs(
        self,
        prompt_ids_tensor: Tensor,
        response_ids: list[int],
        prompt_hiddens: Optional[Tensor] = None,
        prompt_mask_for_read: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-token log π under the CURRENT policy. Gradient enabled.

        When `prompt_hiddens` is provided, they are used as
        `read_conditioning_hiddens` so the read trajectory fires and
        mem_inject's bridge MLPs see a non-zero readout — required for
        PG gradient to actually flow through the adapter weights. Without
        it, `mem_inject` falls back to zero_readout and the trainable
        params (walker MLPs, bridge weights) don't affect logits.
        """
        full_ids, full_mask, prompt_len, resp_len_kept = self._prepare_lp_inputs(
            prompt_ids_tensor, response_ids,
        )
        if full_ids is None:
            return torch.zeros(0, device=prompt_ids_tensor.device)
        out = self.model.forward_window(
            lm_input_ids=full_ids,
            attention_mask=full_mask,
            prev_window_hiddens=None,
            read_conditioning_hiddens=prompt_hiddens,
            read_conditioning_mask=prompt_mask_for_read,
            hard_routing=False,
            write_mode="passage",
        )
        return self._gather_response_logprobs(out["logits"], full_ids, prompt_len, resp_len_kept)

    @torch.no_grad()
    def _compute_log_probs_ref(
        self,
        prompt_ids_tensor: Tensor,
        response_ids: list[int],
        prompt_hiddens: Optional[Tensor] = None,
        prompt_mask_for_read: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-token log π under the REFERENCE policy.

        Uses `torch.func.functional_call` so the live model parameters
        are NOT modified — avoids the in-place-update autograd violation
        a state_dict swap would trigger. Live buffers (manifold edge
        state etc.) ARE used because we want the reference forward to
        see the same memory state as the current-policy forward; only
        the policy weights differ.
        """
        if self._ref_params is None:
            return torch.zeros(0, device=prompt_ids_tensor.device)
        full_ids, full_mask, prompt_len, resp_len_kept = self._prepare_lp_inputs(
            prompt_ids_tensor, response_ids,
        )
        if full_ids is None:
            return torch.zeros(0, device=prompt_ids_tensor.device)
        # functional_call dispatches through model.__call__ → model.forward,
        # which we aliased to forward_window in IntegratedLMV2.
        out = functional_call(
            self.model,
            self._ref_params,
            args=(full_ids,),
            kwargs=dict(
                attention_mask=full_mask,
                prev_window_hiddens=None,
                read_conditioning_hiddens=prompt_hiddens,
                read_conditioning_mask=prompt_mask_for_read,
                hard_routing=False,
                write_mode="passage",
            ),
            tie_weights=True,
            strict=False,
        )
        return self._gather_response_logprobs(
            out["logits"], full_ids, prompt_len, resp_len_kept,
        )

    def step(self, example: dict) -> V2Wave3Metrics:
        """One GRPO update step on a single prompt.

        `example` is one row from PromptResponseDataset:
            { prompt_ids, gold_ids, source, reward_kind, meta }
        """
        cfg = self.model.cfg
        device = next(self.model.parameters()).device
        T = cfg.T_window
        prompt_ids = example["prompt_ids"]
        gold_ids = example["gold_ids"]
        reward_kind = example["reward_kind"]
        meta = example.get("meta", {})
        source = example.get("source", "unknown")

        self.optimizer.zero_grad(set_to_none=True)
        self.model.manifold.advance_step()

        # Snapshot manifold so independent rollouts don't pollute each other's
        # memory state. Restore after rollouts; the GRPO update then operates
        # on the policy alone (manifold is no-grad anyway).
        snap = self.model.manifold.snapshot_edge_state()

        # 1. Rollout J responses (no grad — generation is stochastic AR).
        try:
            responses, _ = self._rollout(prompt_ids)
        finally:
            self.model.manifold.restore_edge_state(snap)

        # 2. Score each response → group-relative advantages.
        rewards = self._score_rollouts(responses, gold_ids, reward_kind, meta)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        if rewards_t.numel() <= 1 or rewards_t.std().item() < 1e-6:
            # All-zero or all-same rewards → no learning signal this step.
            # Skip the policy update but still advance the step counter so
            # the training loop progresses.
            self._step_count += 1
            return V2Wave3Metrics(
                loss=0.0,
                mean_reward=float(rewards_t.mean()),
                max_reward=float(rewards_t.max()) if rewards else 0.0,
                min_reward=float(rewards_t.min()) if rewards else 0.0,
                reward_std=0.0,
                mean_response_len=sum(len(r) for r in responses) / max(len(responses), 1),
                per_source={source: float(rewards_t.mean())},
            )
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        # 3. Differentiable log-probs of generated tokens under current policy
        #    AND reference policy. Precompute prompt hiddens once and pass
        #    as read_conditioning_hiddens so the read trajectory fires and
        #    mem_inject's bridge MLPs actually affect logits — otherwise
        #    PG has no learning signal (memory_fn would be zero_readout).
        prompt_t = torch.tensor(
            self._truncate_and_pad(prompt_ids, T)[:min(len(prompt_ids), T)],
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        prompt_mask = (prompt_t != self.pad_token_id)
        # Detached prompt-only forward to source the read conditioning.
        # No backward through this — its only role is to feed
        # read_conditioning_hiddens into the per-rollout log-prob call.
        snap_prompt = self.model.manifold.snapshot_edge_state()
        try:
            with torch.no_grad():
                prompt_out = self.model.forward_window(
                    lm_input_ids=prompt_t,
                    attention_mask=prompt_mask,
                    prev_window_hiddens=None,
                    hard_routing=False,
                    write_mode="passage",
                )
                prompt_hiddens = prompt_out["current_hiddens"].detach()
        finally:
            self.model.manifold.restore_edge_state(snap_prompt)

        pg_loss = torch.zeros((), device=device)
        kl_loss = torch.zeros((), device=device)
        aux_lb_acc = torch.zeros((), device=device)
        aux_z_acc = torch.zeros((), device=device)
        n_walker_calls = 0
        n_kept = 0
        for j, resp in enumerate(responses):
            if not resp:
                continue
            # Snapshot manifold again — each rollout's grad pass must see
            # the fresh post-prompt state, not the previous rollout's writes.
            snap_j = self.model.manifold.snapshot_edge_state()
            try:
                lp_cur = self._compute_log_probs(
                    prompt_t, resp,
                    prompt_hiddens=prompt_hiddens,
                    prompt_mask_for_read=prompt_mask,
                )
                if self._ref_params is not None and self.kl_coef > 0:
                    lp_ref = self._compute_log_probs_ref(
                        prompt_t, resp,
                        prompt_hiddens=prompt_hiddens,
                        prompt_mask_for_read=prompt_mask,
                    )
                else:
                    lp_ref = None
            finally:
                self.model.manifold.restore_edge_state(snap_j)
            if lp_cur.numel() == 0:
                continue
            # Policy-gradient: -advantage · sum_t log π(token_t)
            pg_loss = pg_loss - advantages[j] * lp_cur.sum()
            # KL via Schulman's k3 estimator: always-nonneg, unbiased.
            # k3 = exp(log_ratio) - 1 - log_ratio  where log_ratio = log π/π_ref
            if lp_ref is not None and lp_ref.numel() == lp_cur.numel():
                log_ratio = lp_cur - lp_ref.detach()
                k3 = log_ratio.exp() - 1.0 - log_ratio
                kl_loss = kl_loss + k3.mean()
            n_kept += 1
            n_walker_calls += 1  # forward_window aux losses
        if n_kept == 0:
            self.model.manifold.restore_edge_state(snap)
            return V2Wave3Metrics(
                loss=0.0,
                mean_reward=float(rewards_t.mean()),
                reward_std=float(rewards_t.std()),
                mean_response_len=0.0,
                per_source={source: float(rewards_t.mean())},
            )
        pg_loss = pg_loss / n_kept
        kl_loss = kl_loss / n_kept

        total_loss = pg_loss + self.kl_coef * kl_loss
        total_loss.backward()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._step_count += 1

        stats = self.model.manifold.edge_stats()
        return V2Wave3Metrics(
            loss=float(total_loss.detach()),
            pg_loss=float(pg_loss.detach()),
            kl_loss=float(kl_loss.detach()),
            aux_load_balance=float(aux_lb_acc.detach()) if isinstance(aux_lb_acc, Tensor) else 0.0,
            aux_z_loss=float(aux_z_acc.detach()) if isinstance(aux_z_acc, Tensor) else 0.0,
            mean_reward=float(rewards_t.mean()),
            max_reward=float(rewards_t.max()),
            min_reward=float(rewards_t.min()),
            reward_std=float(rewards_t.std()),
            mean_response_len=sum(len(r) for r in responses) / max(len(responses), 1),
            grad_norm=float(grad_norm.detach()) if isinstance(grad_norm, Tensor) else 0.0,
            n_active_edges=stats["n_active_edges"],
            mean_effectiveness=stats.get("mean_effectiveness", 0.0),
            mean_visit_count=stats["mean_visit_count"],
            per_source={source: float(rewards_t.mean())},
        )
