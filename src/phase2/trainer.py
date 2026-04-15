"""Phase 2 trainer: GRPO over the neuromodulator's factored categorical policy.

Architecture (new — see discrete_policy.py + chat log):
- Neuromodulator emits per-cell logits over K=256 codes.
- A frozen codebook + decoder (trained during bootstrap) maps codes to
  continuous memory updates.
- Phase 2 trains ONLY the logit head. Codebook, decoder, LM, memory
  dynamics all frozen. Gradient = factored categorical policy gradient
  with shared (per-(call, sample)) advantage broadcast across cells.

Reward: -CE over a window of W future tokens following each modulation
event, computed by running the frozen upper scan + LM head on
H_enriched = H_mid + mem_scale * readout. Curriculum ramps W from 512
to 4096 over stages.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model.config import Config
from ..model.model import Model


@dataclass
class CurriculumStage:
    """One stage of the phase 2 reward-window curriculum."""
    reward_window: int   # W: number of tokens in the reward window
    token_budget: int    # how many tokens to train this stage for


class Phase2Trainer:
    """GRPO trainer for the factored-categorical modulator."""

    def __init__(
        self,
        model: Model,
        dataloader,
        config: Config,
        device: torch.device,
        group_size: int = 8,
        lr: float = 1e-4,
        tau: float = 1.0,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 100,
        eval_loader_factory=None,
        eval_batches: int = 8,
        eval_warmup_batches: int = 4,
        metrics_path: str | None = None,
        train_loader_factory=None,
        sanity_check_interval: int = 50,
    ):
        self.model = model
        self.dataloader = dataloader
        self.train_loader_factory = train_loader_factory
        self.config = config
        self.device = device
        self.group_size = group_size
        self.tau = tau
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_loader_factory = eval_loader_factory
        self.eval_batches = eval_batches
        self.eval_warmup_batches = eval_warmup_batches
        self.sanity_check_interval = sanity_check_interval

        # Freeze everything. Unfreeze just the logit head below.
        # In the new architecture this is: discrete_policy.logit_{w1,b1,w2,b2}.
        # codebook + decoder are frozen (they define the action vocabulary
        # + code semantics; GRPO only adjusts the policy over them).
        dp = model.memory.discrete_policy
        for p in model.parameters():
            p.requires_grad = False
        self.trainable_params = [
            dp.logit_w1, dp.logit_b1, dp.logit_w2, dp.logit_b2,
        ]
        for p in self.trainable_params:
            p.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, betas=(0.9, 0.95),
            fused=(device.type == "cuda"))
        self.global_step = 0

        # Snapshot of initial logit-head weights for drift tracking.
        self._mod_w0_snapshot = [p.detach().clone() for p in self.trainable_params]

        self.reward_window: int = 0

        self.metrics_path = metrics_path
        if metrics_path is not None:
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

        # Quartile-reward side-channel (set inside _windowed_reward).
        self._last_window_quartile_rewards: list[float] | None = None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _append_metrics(self, metrics: dict):
        if self.metrics_path is None:
            return
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, eval_loader, n_batches: int = 8,
                 warmup_batches: int = 0, use_memory: bool = True,
                 key_prefix: str = "eval") -> dict:
        """Held-out eval pass using the continuous modulator path (phase 1 fwd)."""
        model = self.model
        memory = model.memory

        train_mem_state = memory.runtime_state_dict() if memory._initialized else None
        train_lm_carries = [
            h.clone() if h is not None else None for h in model.lm._carries]
        train_initialized = model._initialized
        was_training = model.training
        was_collecting = memory._collecting_actions
        saved_action_buffer = memory._action_buffer
        if was_collecting:
            memory._collecting_actions = False
            memory._action_buffer = []

        memory._initialized = False
        model._initialized = False
        model.lm._carries = [None] * self.config.L_total
        model.train(False)

        total_ce = 0.0
        total_aux = 0.0
        count = 0
        prev_token = None
        total_batches = warmup_batches + n_batches
        eval_chunk_T = self.config.T
        try:
            for i, batch in enumerate(eval_loader):
                if i >= total_batches:
                    break
                input_ids_full = batch.input_ids.to(self.device, non_blocking=True)
                target_ids_full = batch.target_ids.to(self.device, non_blocking=True)
                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(self.device, non_blocking=True)

                _, T_full = input_ids_full.shape
                ce_sum = 0.0
                ce_count = 0
                aux_sum = 0.0
                effective_prev = batch_prev if batch_prev is not None else prev_token
                for s in range(0, T_full, eval_chunk_T):
                    e = min(s + eval_chunk_T, T_full)
                    result = model.forward_chunk(
                        input_ids_full[:, s:e], target_ids=target_ids_full[:, s:e],
                        use_memory=use_memory,
                        prev_token=effective_prev,
                    )
                    chunk_len = e - s
                    ce_sum += result["ce_loss"].item() * chunk_len
                    aux_sum += result["aux_loss"].item() * chunk_len
                    ce_count += chunk_len
                    effective_prev = input_ids_full[:, e - 1]

                prev_token = input_ids_full[:, -1]
                if i < warmup_batches:
                    continue
                total_ce += ce_sum / ce_count
                total_aux += aux_sum / ce_count
                count += 1
        finally:
            if train_mem_state is not None:
                memory.load_runtime_state(train_mem_state)
            else:
                memory._initialized = False
            model.lm._carries = train_lm_carries
            model._initialized = train_initialized
            if was_training:
                model.train(True)
            if was_collecting:
                memory._collecting_actions = True
                memory._action_buffer = saved_action_buffer

        if count == 0:
            return {f"{key_prefix}_ce_loss": 0.0, f"{key_prefix}_aux_loss": 0.0,
                    f"{key_prefix}_ppl": 0.0, f"{key_prefix}_batches": 0}
        avg_ce = total_ce / count
        avg_aux = total_aux / count
        return {
            f"{key_prefix}_ce_loss": avg_ce,
            f"{key_prefix}_aux_loss": avg_aux,
            f"{key_prefix}_ppl": min(math.exp(avg_ce), 1e6),
            f"{key_prefix}_batches": count,
        }

    @torch.no_grad()
    def evaluate_quantized(self, eval_loader, n_batches: int = 4,
                           warmup_batches: int = 0) -> dict:
        """Eval using the deterministic (argmax) categorical policy.

        This is the policy phase-2 GRPO is optimizing toward. Divergence
        from evaluate() (stochastic Gumbel sample during phase-1 fwd)
        indicates the argmax policy is under-tuned.
        """
        model = self.model
        memory = model.memory
        lm = model.lm

        train_mem_state = memory.runtime_state_dict() if memory._initialized else None
        train_lm_carries = [
            h.clone() if h is not None else None for h in lm._carries]
        train_initialized = model._initialized
        was_training = model.training
        was_collecting = memory._collecting_actions
        saved_action_buffer = memory._action_buffer
        if was_collecting:
            memory._collecting_actions = False
            memory._action_buffer = []

        memory._initialized = False
        model._initialized = False
        lm._carries = [None] * self.config.L_total
        model.train(False)

        total_ce = 0.0
        count = 0
        prev_token = None
        total_batches = warmup_batches + n_batches
        try:
            for i, batch in enumerate(eval_loader):
                if i >= total_batches:
                    break
                input_ids = batch.input_ids.to(self.device, non_blocking=True)
                BS, T = input_ids.shape
                if not memory._initialized:
                    memory.initialize_states(BS, self.device)

                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(self.device, non_blocking=True)
                effective_prev = (batch_prev if batch_prev is not None
                                  else prev_token)

                H_mid = self._run_lower_scan(
                    input_ids, prev_token=effective_prev)

                result = memory.forward_segment_phase2(
                    H_mid, input_ids, lm,
                    tau=self.tau, sample=False)     # ← argmax policy
                readouts_q = result["readouts"]

                mem_scale = lm.mem_scale
                H_enriched = H_mid.to(readouts_q.dtype) + mem_scale * readouts_q
                eot = self.config.eot_id
                eos_positions = (input_ids == eot)
                reset_mask = torch.zeros_like(eos_positions)
                reset_mask[:, 1:] = eos_positions[:, :-1]
                if effective_prev is not None:
                    reset_mask[:, 0] = (effective_prev == eot)
                if not reset_mask.any():
                    reset_mask = None
                chunk_t = 128
                target_ids = batch.target_ids.to(self.device, non_blocking=True)
                total_ce_batch = 0.0
                total_valid_batch = 0.0
                for s in range(0, T, chunk_t):
                    e = min(s + chunk_t, T)
                    rm = reset_mask[:, s:e] if reset_mask is not None else None
                    H_up = lm.forward_scan_upper(H_enriched[:, s:e], reset_mask=rm)
                    chunk_logits = lm.forward_output(H_up)
                    chunk_targets = target_ids[:, s:e]
                    chunk_ce = F.cross_entropy(
                        chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                        chunk_targets.reshape(-1),
                        reduction="none",
                    ).reshape(BS, -1)
                    chunk_valid = (input_ids[:, s:e] != eot).float()
                    total_ce_batch += (chunk_ce * chunk_valid).sum().item()
                    total_valid_batch += chunk_valid.sum().item()

                prev_token = input_ids[:, -1]

                if i < warmup_batches:
                    continue

                valid_count = max(total_valid_batch, 1.0)
                batch_ce = total_ce_batch / valid_count
                total_ce += batch_ce
                count += 1
        finally:
            if train_mem_state is not None:
                memory.load_runtime_state(train_mem_state)
            else:
                memory._initialized = False
            lm._carries = train_lm_carries
            model._initialized = train_initialized
            if was_training:
                model.train(True)
            if was_collecting:
                memory._collecting_actions = True
                memory._action_buffer = saved_action_buffer

        if count == 0:
            return {"quant_eval_ce": 0.0, "quant_eval_ppl": 0.0}
        avg_ce = total_ce / count
        return {
            "quant_eval_ce": avg_ce,
            "quant_eval_ppl": min(math.exp(avg_ce), 1e6),
        }

    # ------------------------------------------------------------------
    # Lower-scan helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_lower_scan(
        self, input_ids: Tensor, prev_token: Tensor | None = None,
    ) -> Tensor:
        eot_id = self.config.eot_id
        BS, T_total = input_ids.shape
        seg_T = self.config.T
        chunks = []
        for start in range(0, T_total, seg_T):
            end = min(start + seg_T, T_total)
            seg_ids = input_ids[:, start:end]
            eos_positions = (seg_ids == eot_id)
            reset_mask = torch.zeros_like(eos_positions)
            reset_mask[:, 1:] = eos_positions[:, :-1]
            if start == 0 and prev_token is not None:
                reset_mask[:, 0] = (prev_token.to(seg_ids.device) == eot_id)
            elif start > 0:
                reset_mask[:, 0] = (input_ids[:, start - 1] == eot_id)
            if not reset_mask.any():
                reset_mask = None
            chunks.append(self.model.lm.forward_scan_lower(seg_ids, reset_mask=reset_mask))
        return torch.cat(chunks, dim=1)

    # ------------------------------------------------------------------
    # Per-token reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_per_token_reward_lm_ce(
        self, readouts: Tensor, H_mid: Tensor, input_ids: Tensor,
        valid_mask: Tensor,
        h_mid_batch_map: Tensor | None = None,
    ) -> Tensor:
        """Compute -LM CE as per-token reward (chunked to bound VRAM)."""
        BS, T, D = readouts.shape
        lm = self.model.lm
        mem_scale = lm.mem_scale
        dt = readouts.dtype

        eot = self.config.eot_id
        eos_positions = (input_ids == eot)
        reset_mask = torch.zeros_like(eos_positions)
        reset_mask[:, 1:] = eos_positions[:, :-1]

        saved_carries = [
            h.clone() if h is not None else None for h in lm._carries]
        split = self.config.scan_split_at

        sub_bs = min(BS, 8)
        chunk_t = 128
        all_ce = torch.zeros(BS, T - 1, device=readouts.device, dtype=torch.float32)

        try:
            for b_start in range(0, BS, sub_bs):
                b_end = min(b_start + sub_bs, BS)
                if h_mid_batch_map is not None:
                    sub_h_mid = H_mid[h_mid_batch_map[b_start:b_end]].to(dt)
                else:
                    sub_h_mid = H_mid[b_start:b_end].to(dt)
                sub_H = sub_h_mid + mem_scale * readouts[b_start:b_end]
                del sub_h_mid
                sub_ids = input_ids[b_start:b_end]
                sub_rm = reset_mask[b_start:b_end]

                for i in range(split, self.config.L_total):
                    lm._carries[i] = None

                prev_last_logit = None
                ce_pos = 0
                for s in range(0, T, chunk_t):
                    e = min(s + chunk_t, T)
                    rm_chunk = sub_rm[:, s:e] if sub_rm.any() else None
                    H_upper = lm.forward_scan_upper(sub_H[:, s:e], reset_mask=rm_chunk)
                    chunk_logits = lm.forward_output(H_upper)
                    chunk_targets = sub_ids[:, s:e]

                    if prev_last_logit is not None:
                        shifted = torch.cat([prev_last_logit, chunk_logits[:, :-1]], dim=1)
                    else:
                        shifted = chunk_logits[:, :-1]
                        chunk_targets = chunk_targets[:, 1:]

                    if shifted.shape[1] > 0:
                        chunk_ce = F.cross_entropy(
                            shifted.reshape(-1, shifted.shape[-1]).float(),
                            chunk_targets.reshape(-1),
                            reduction="none",
                        ).reshape(b_end - b_start, -1)
                        all_ce[b_start:b_end, ce_pos:ce_pos + chunk_ce.shape[1]] = chunk_ce
                        ce_pos += chunk_ce.shape[1]

                    prev_last_logit = chunk_logits[:, -1:]
                    del chunk_logits, H_upper
        finally:
            lm._carries = saved_carries

        rewards = torch.zeros(BS, T, device=readouts.device, dtype=torch.float32)
        rewards[:, 1:] = -all_ce
        rewards = rewards * valid_mask
        return rewards

    def _windowed_reward(
        self, per_token_reward: Tensor, call_positions: Tensor, window: int,
    ) -> tuple[Tensor, Tensor]:
        """For each modulator call at t, mean reward over [t+1, t+1+W].

        Returns:
            rewards: [K, n_calls, BS] (incomplete-window slots zeroed)
            complete: [n_calls] bool — True for calls whose window fits in T.
                      Same across K and BS at a given call index, since
                      completeness depends only on (call_position, T, window).
        """
        T = per_token_reward.shape[-1]
        positions = call_positions.to(per_token_reward.device)
        starts = positions + 1
        offsets = torch.arange(window, device=per_token_reward.device)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        complete = (starts + window <= T)
        indices = indices.clamp(max=T - 1)

        gathered = per_token_reward[..., indices]
        out = gathered.mean(dim=-1)
        out = out * complete.to(out.dtype)

        # Quartile diagnostic (side-effect stored on self)
        if window >= 4:
            q = window // 4
            quartile_means = torch.stack([
                gathered[..., 0:q].mean(dim=-1),
                gathered[..., q:2 * q].mean(dim=-1),
                gathered[..., 2 * q:3 * q].mean(dim=-1),
                gathered[..., 3 * q:].mean(dim=-1),
            ], dim=-1)
            complete_mask = complete.view(
                *(1,) * (quartile_means.ndim - 2), -1, 1)
            leading_slots = int(
                torch.tensor(quartile_means.shape[:-2]).prod().item())
            live_slots = leading_slots * int(complete.sum().item())
            if live_slots > 0:
                self._last_window_quartile_rewards = (
                    (quartile_means * complete_mask).sum(
                        dim=tuple(range(quartile_means.ndim - 1)))
                    / live_slots
                ).tolist()

        return out.transpose(-1, -2), complete

    # ------------------------------------------------------------------
    # Memory-state snapshot/restore
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_mem_state(self) -> dict:
        mem = self.model.memory
        return {
            "h": mem.h.clone(), "msg": mem.msg.clone(), "W": mem.W.clone(),
            "decay": mem.decay.clone(), "hebbian": mem.hebbian.clone(),
            "s_mem_live": mem.s_mem_live.clone(),
            "s_mem_ema_fast": mem.s_mem_ema_fast.clone(),
            "prev_readout": mem.prev_readout.clone(),
            "readout_drift": mem.readout_drift.clone(),
        }

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, batch) -> dict:
        """Run K trajectories in parallel via batch-expanded memory.

        Returns dict:
            mod_inputs: [K, n_calls, BS, NC, mod_in]
            codes:      [K, n_calls, BS, NC]  long
            rewards:    [K, n_calls, BS]  per-action reward
        """
        device = self.device
        K = self.group_size
        input_ids = batch.input_ids.to(device, non_blocking=True)
        prev_token = getattr(batch, "prev_token", None)
        BS, T = input_ids.shape

        self.model.lm.detach_carries()

        H_mid = self._run_lower_scan(input_ids, prev_token=prev_token)

        init_snapshot = self._save_mem_state()

        mem = self.model.memory
        for key, val in init_snapshot.items():
            expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
            setattr(mem, key, expanded.clone())

        ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])
        h_mid_batch_map = torch.arange(K * BS, device=device) % BS

        result = mem.forward_segment_phase2(
            H_mid, ids_exp, self.model.lm,
            tau=self.tau, sample=True,
            h_mid_batch_map=h_mid_batch_map)

        readouts = result["readouts"].view(K, BS, T, -1)
        call_positions = result["call_positions"]
        n_calls = result["mod_inputs"].shape[0]
        # mod_inputs shape: [n_calls, K*BS, NC, mod_in] → [K, n_calls, BS, NC, mod_in]
        mod_inputs = result["mod_inputs"].view(
            n_calls, K, BS, *result["mod_inputs"].shape[2:]).transpose(0, 1)
        # codes shape: [n_calls, K*BS, NC] → [K, n_calls, BS, NC]
        codes = result["codes"].view(
            n_calls, K, BS, *result["codes"].shape[2:]).transpose(0, 1)

        # Per-token reward via full LM CE
        prev_token_exp = None
        if prev_token is not None:
            prev_token_exp = prev_token.unsqueeze(0).expand(K, *prev_token.shape).reshape(K * BS)
        # Build valid mask (same as phase 1 CE masking — skip positions after EOT)
        eot = self.config.eot_id
        valid_mask = torch.ones(K * BS, T, device=device, dtype=torch.float32)
        if prev_token_exp is not None:
            valid_mask[:, 0] = (prev_token_exp != eot).float()
        eos_positions = (ids_exp == eot)
        if eos_positions.any():
            valid_mask[:, 1:] = valid_mask[:, 1:] * (1.0 - eos_positions[:, :-1].float())
        per_token_reward = self._compute_per_token_reward_lm_ce(
            result["readouts"], H_mid, ids_exp, valid_mask,
            h_mid_batch_map=h_mid_batch_map)
        per_token_reward = per_token_reward.reshape(K, BS, T)

        rewards, complete_mask = self._windowed_reward(
            per_token_reward, call_positions, self.reward_window)

        mean_per_k_sample = rewards.mean(dim=1)
        mean_reward = mean_per_k_sample.mean().item()
        best_mean_reward = mean_per_k_sample.max(dim=0).values.mean().item()

        # Pick ONE trajectory per sample to carry memory state forward.
        # Random choice → unbiased expected-state propagation.
        random_k_per_sample = torch.randint(
            0, K, (BS,), device=device, dtype=torch.long)
        gather_idx = random_k_per_sample * BS + torch.arange(BS, device=device)
        for key in init_snapshot:
            full = getattr(mem, key)
            setattr(mem, key, full[gather_idx].clone())

        return {
            "mod_inputs": mod_inputs,
            "codes": codes,
            "rewards": rewards,
            "complete_mask": complete_mask,  # [n_calls] bool
            "mean_reward": mean_reward,
            "best_mean_reward": best_mean_reward,
            "T": T,
            "BS": BS,
        }

    # ------------------------------------------------------------------
    # GRPO
    # ------------------------------------------------------------------

    def _compute_advantages(
        self, rewards: Tensor, complete_mask: Tensor | None = None,
    ) -> Tensor:
        """Group-relative advantages, per-(call, sample) normalized.

        For incomplete-window slots, rewards are zero across all K (same
        completeness for all K at a given call). Baseline is 0, std is 0
        (clamped to 1e-8), so advantage comes out as 0 / 1e-8 = 0 already.
        We still explicitly mask incomplete advantages to 0 for safety —
        if any code path ever produces non-zero reward on incomplete slots
        (e.g. a different zero-convention), the mask keeps GRPO honest.
        """
        baseline = rewards.mean(dim=0, keepdim=True)
        advantages = rewards - baseline
        std = advantages.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantages = advantages / std
        if complete_mask is not None:
            # complete_mask: [n_calls] → broadcast to [K, n_calls, BS]
            mask = complete_mask.view(1, -1, 1).to(advantages.dtype)
            advantages = advantages * mask
        return advantages

    def grpo_step(self, rollout_result: dict) -> dict:
        """Factored-categorical GRPO step.

        Policy: product over NC cells of Categorical(logits_i).
        Loss: -E[ A · Σ_i log π_i(code_i) ] - β · Σ_i H(π_i)
        (advantage broadcast across cells; credit assignment is "democratic").

        Re-runs the neuromod on stored mod_inputs to get current logits,
        scores the rollout-sampled codes, backprops through logit head only.
        """
        # Force the trainable logit head into train mode for the gradient
        # pass. train_phase2.py keeps the full model in eval mode so rollouts
        # are deterministic (no LM dropout); the gradient pass should see the
        # policy as training. Restore the original mode before returning so
        # grpo_step is neutral — no-op today (compute_logits has no
        # dropout/BN), but defensive against future additions.
        dp = self.model.memory.discrete_policy
        dp_was_training = dp.training
        dp.train(True)

        K, n_calls, BS, NC, mod_in_dim = rollout_result["mod_inputs"].shape
        mod_inputs = rollout_result["mod_inputs"]              # [K, n_calls, BS, NC, mod_in] f32
        codes = rollout_result["codes"]                        # [K, n_calls, BS, NC] long
        rewards = rollout_result["rewards"]                    # [K, n_calls, BS] f32
        complete_mask = rollout_result.get("complete_mask")    # [n_calls] bool or None

        advantages = self._compute_advantages(rewards, complete_mask)
        advantages = advantages.unsqueeze(-1).expand(-1, -1, -1, NC)  # [K, n_calls, BS, NC]

        # Flatten (K * n_calls * BS) into a single batch dim for the
        # modulator forward. NC stays as the cell dim (modulator is per-cell).
        M = K * n_calls * BS
        mod_inputs_flat = mod_inputs.reshape(M, NC, mod_in_dim)
        codes_flat = codes.reshape(M, NC)
        adv_flat = advantages.reshape(M, NC)

        # Chunk over M to bound VRAM (logits tensor is [C, NC, K_codes]).
        K_codes = self.model.memory.discrete_policy.K
        chunk_size = max(1024, 32768 // max(K_codes // 32, 1))

        total_loss = 0.0
        total_log_pi = 0.0
        total_entropy = 0.0
        per_cell_log_pi_sum = torch.zeros(NC, device=self.device)
        per_cell_count = 0

        do_sanity = (
            self.sanity_check_interval > 0
            and self.global_step > 0
            and self.global_step % self.sanity_check_interval == 0
        )
        log_pi_pre_cache = [] if do_sanity else None

        self.optimizer.zero_grad(set_to_none=True)
        policy = self.model.memory.discrete_policy

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            mi_chunk = mod_inputs_flat[start:end]         # [C, NC, mod_in]
            c_chunk = codes_flat[start:end]                # [C, NC]
            a_chunk = adv_flat[start:end]                  # [C, NC]
            C = mi_chunk.shape[0]

            logits = policy.compute_logits(mi_chunk)       # [C, NC, K_codes]
            log_probs = F.log_softmax(logits / self.tau, dim=-1)
            log_pi = log_probs.gather(-1, c_chunk.unsqueeze(-1)).squeeze(-1)  # [C, NC]

            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)     # [C, NC]

            # Loss normalized by full M (not chunk C) so backward accumulates
            # to the correct mean gradient.
            chunk_loss = (-(a_chunk * log_pi) - self.entropy_coeff * entropy).sum() / (M * NC)
            chunk_loss.backward()

            with torch.no_grad():
                total_loss += chunk_loss.item() * (M * NC)
                total_log_pi += log_pi.sum().item()
                total_entropy += entropy.sum().item()
                per_cell_log_pi_sum += log_pi.sum(dim=0)
                per_cell_count += C

                if log_pi_pre_cache is not None:
                    log_pi_pre_cache.append(log_pi.clone())

        grad_norm = nn.utils.clip_grad_norm_(
            self.trainable_params, self.max_grad_norm).item()
        # NaN guard: a -inf log_pi (from a near-zero-prob sampled code) makes
        # the gradient NaN; clip_grad_norm_ returns NaN which silently fails
        # the threshold comparison, and optimizer.step() would write NaN into
        # the logit head weights, poisoning every subsequent rollout.
        if math.isfinite(grad_norm):
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            print(f"[WARN p2 step {self.global_step}] skipping optimizer step: "
                  f"non-finite grad_norm ({grad_norm})")

        # -----------------------------------------------------
        # Sanity check (H5): Δlog_pi vs advantage correlation
        # -----------------------------------------------------
        sanity_corr = float("nan")
        sanity_delta_mean = float("nan")
        if log_pi_pre_cache is not None:
            with torch.no_grad():
                log_pi_post_parts = []
                for start in range(0, M, chunk_size):
                    end = min(start + chunk_size, M)
                    mi_chunk = mod_inputs_flat[start:end]
                    c_chunk = codes_flat[start:end]
                    logits2 = policy.compute_logits(mi_chunk)
                    lp2 = F.log_softmax(logits2 / self.tau, dim=-1).gather(
                        -1, c_chunk.unsqueeze(-1)).squeeze(-1)
                    log_pi_post_parts.append(lp2)
                log_pi_pre = torch.cat(log_pi_pre_cache, dim=0).reshape(-1)
                log_pi_post = torch.cat(log_pi_post_parts, dim=0).reshape(-1)
                delta = log_pi_post - log_pi_pre
                a_flat = adv_flat.reshape(-1)
                d = delta - delta.mean()
                a = a_flat - a_flat.mean()
                denom = d.pow(2).sum().sqrt() * a.pow(2).sum().sqrt()
                if denom > 0:
                    sanity_corr = (d * a).sum().item() / denom.item()
                sanity_delta_mean = delta.mean().item()

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------
        rewards_f = rewards.float()
        reward_mean = rewards_f.mean().item()
        reward_std = rewards_f.std().item()
        reward_min = rewards_f.min().item()
        reward_max = rewards_f.max().item()

        # Complete-window-corrected reward, using the real completeness
        # mask (not a `abs > 1e-8` proxy that would misfire if any
        # legitimate CE-reward happened to land near zero).
        if complete_mask is not None:
            # [n_calls] → [K, n_calls, BS]
            cm_b = complete_mask.view(1, -1, 1).expand_as(rewards_f).to(torch.bool)
        else:
            cm_b = rewards_f.abs() > 1e-8  # legacy fallback
        n_complete = int(cm_b.sum().item())
        if n_complete > 0:
            reward_mean_complete = rewards_f[cm_b].mean().item()
            complete_fraction = n_complete / rewards_f.numel()
        else:
            reward_mean_complete = 0.0
            complete_fraction = 0.0

        # K-diversity across trajectories, counted only on complete slots
        # (incomplete slots have rewards=0 across all K, std=0 — would pull
        # the mean spread down artificially).
        k_std_per_slot = rewards_f.std(dim=0, unbiased=False)
        if complete_mask is not None:
            live_mask = complete_mask.view(-1, 1).expand_as(k_std_per_slot).float()
        else:
            live_mask = (rewards_f.abs().sum(dim=0) > 1e-8).float()
        live_count = live_mask.sum().clamp(min=1.0)
        k_spread_mean = (k_std_per_slot * live_mask).sum().item() / live_count.item()

        # Advantage stats
        adv_abs_mean = adv_flat.abs().mean().item()
        adv_max = adv_flat.abs().max().item()
        adv_flat_frac = (adv_flat.abs() < 0.1).float().mean().item()

        # Per-K mean reward
        per_k_mean = rewards_f.mean(dim=(1, 2))
        per_k_fields = {f"k{i}_reward": per_k_mean[i].item()
                        for i in range(per_k_mean.shape[0])}

        # Code diversity per slot: fraction where all K picked the same code
        all_same_k = (codes == codes[0:1]).all(dim=0)  # [n_calls, BS, NC]
        frac_all_k_same = all_same_k.float().mean().item()

        # Modulator drift from init
        mod_drift_rel = 0.0
        if getattr(self, "_mod_w0_snapshot", None) is not None:
            with torch.no_grad():
                drift_sq = 0.0
                init_sq = 0.0
                for p, p0 in zip(self.trainable_params, self._mod_w0_snapshot):
                    drift_sq += (p - p0).pow(2).sum().item()
                    init_sq += p0.pow(2).sum().item()
                mod_drift_rel = math.sqrt(drift_sq) / max(math.sqrt(init_sq), 1e-8)

        # Per-cell log_pi variation (are all NC cells engaged?)
        per_cell_logpi = per_cell_log_pi_sum / max(per_cell_count, 1)
        per_cell_logpi_std = per_cell_logpi.std().item() if NC > 1 else 0.0

        # Codebook usage in rollout (for diagnostic; codebook is frozen)
        n_unique = int(torch.unique(codes.reshape(-1)).numel())

        dp.train(dp_was_training)

        return {
            "loss": total_loss / (M * NC),
            "log_pi_mean": total_log_pi / (M * NC),
            "entropy_mean": total_entropy / (M * NC),
            "reward_mean": reward_mean,
            "reward_mean_complete": reward_mean_complete,
            "complete_fraction": complete_fraction,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "k_spread_mean": k_spread_mean,
            "adv_abs_mean": adv_abs_mean,
            "adv_max": adv_max,
            "adv_flat_frac": adv_flat_frac,
            **per_k_fields,
            "mod_grad_norm": grad_norm,
            "M": M,
            "n_unique_codes": n_unique,
            "frac_all_k_same_code": frac_all_k_same,
            "mod_drift_rel": mod_drift_rel,
            "per_cell_logpi_std": per_cell_logpi_std,
            "sanity_logpi_adv_corr": sanity_corr,
            "sanity_logpi_delta_mean": sanity_delta_mean,
            **(
                {f"window_q{i}_reward": v
                 for i, v in enumerate(self._last_window_quartile_rewards)}
                if self._last_window_quartile_rewards else {}
            ),
        }

    # ------------------------------------------------------------------
    # Curriculum loop
    # ------------------------------------------------------------------

    def run_curriculum(self, stages: list[CurriculumStage]):
        for stage_idx, stage in enumerate(stages):
            self.reward_window = stage.reward_window
            print(f"\n=== Phase 2 stage {stage_idx+1}/{len(stages)}: "
                  f"W={stage.reward_window}, budget={stage.token_budget:,} tokens ===")
            if self.train_loader_factory is not None:
                import inspect
                sig = inspect.signature(self.train_loader_factory)
                accepts_stage_idx = (
                    "stage_idx" in sig.parameters
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in sig.parameters.values())
                )
                if accepts_stage_idx:
                    self.dataloader = self.train_loader_factory(
                        stage.reward_window, stage_idx=stage_idx)
                else:
                    self.dataloader = self.train_loader_factory(
                        stage.reward_window)

                stage_bs = getattr(self.dataloader, "batch_size", None)
                if stage_bs is None:
                    ds = getattr(self.dataloader, "dataset", None)
                    stage_bs = getattr(ds, "batch_size", None)
                if (stage_bs is not None
                        and self.model.memory._initialized
                        and self.model.memory.h.shape[0] != stage_bs):
                    print(f"  Resizing memory from BS={self.model.memory.h.shape[0]} "
                          f"to stage BS={stage_bs}")
                    self.model.memory.resize_to_bs(stage_bs)
                    self.model._initialized = self.model.memory._initialized
                    self.model.lm._carries = [None] * self.config.L_total
            self._run_stage(stage)

    def _run_stage(self, stage: CurriculumStage):
        tokens_seen = 0
        t_stage_start = time.time()
        for step_idx, batch in enumerate(self.dataloader):
            BS, T = batch.input_ids.shape
            step_tokens = BS * T
            if tokens_seen >= stage.token_budget:
                break

            t0 = time.time()
            rollout_result = self.rollout(batch)
            t_rollout = time.time() - t0

            t0 = time.time()
            step_metrics = self.grpo_step(rollout_result)
            t_grad = time.time() - t0

            tokens_seen += step_tokens
            self.global_step += 1

            diag_stats = {}
            if self.global_step % self.log_interval == 0:
                diag_stats.update(self.model.memory.compute_lane_divergence())
                diag_stats.update(self.model.memory.compute_memory_health())

            log_row = {
                "step": self.global_step,
                "stage_window": stage.reward_window,
                "tokens_seen": tokens_seen,
                "rollout_time": t_rollout,
                "grad_time": t_grad,
                **step_metrics,
                **diag_stats,
            }
            self._append_metrics(log_row)

            if self.global_step % self.log_interval == 0:
                print(f"[p2 step {self.global_step}] "
                      f"W={stage.reward_window} "
                      f"loss={step_metrics['loss']:.4f} "
                      f"r={step_metrics['reward_mean']:+.3f}"
                      f"±{step_metrics['reward_std']:.2f} "
                      f"log_pi={step_metrics['log_pi_mean']:+.2f} "
                      f"mod_gn={step_metrics['mod_grad_norm']:.3f} "
                      f"codes={step_metrics['n_unique_codes']} "
                      f"roll={t_rollout:.1f}s grad={t_grad:.1f}s "
                      f"tokens={tokens_seen:,}/{stage.token_budget:,}")

            if (
                self.eval_loader_factory is not None
                and self.eval_interval > 0
                and self.global_step > 0
                and self.global_step % self.eval_interval == 0
            ):
                eval_metrics = self.evaluate(
                    self.eval_loader_factory(),
                    n_batches=self.eval_batches,
                    warmup_batches=self.eval_warmup_batches)
                quant_metrics = self.evaluate_quantized(
                    self.eval_loader_factory(),
                    n_batches=self.eval_batches,
                    warmup_batches=self.eval_warmup_batches)
                no_mem_metrics = self.evaluate(
                    self.eval_loader_factory(),
                    n_batches=self.eval_batches,
                    warmup_batches=self.eval_warmup_batches,
                    use_memory=False,
                    key_prefix="eval_no_mem")
                print(f"[p2 eval {self.global_step}] "
                      f"ce={eval_metrics['eval_ce_loss']:.3f} "
                      f"ce_nomem={no_mem_metrics['eval_no_mem_ce_loss']:.3f} "
                      f"q_ce={quant_metrics['quant_eval_ce']:.3f} "
                      f"ppl={eval_metrics['eval_ppl']:.1f} "
                      f"({eval_metrics['eval_batches']} batches)")
                self._append_metrics({
                    "step": self.global_step,
                    "stage_window": stage.reward_window,
                    "event": "eval",
                    **eval_metrics,
                    **quant_metrics,
                    **no_mem_metrics,
                })

        elapsed = time.time() - t_stage_start
        print(f"  stage complete: {tokens_seen:,} tokens in {elapsed:.0f}s")

        if self.metrics_path is not None:
            try:
                from scripts.plot_training import (
                    load_metrics, plot_phase2_grpo, plot_phase2_diversity)
                plots_dir = os.path.join(os.path.dirname(self.metrics_path), "plots")
                os.makedirs(plots_dir, exist_ok=True)
                records = load_metrics(self.metrics_path)
                plot_phase2_grpo(records,
                    os.path.join(plots_dir, "phase2_grpo.png"))
                plot_phase2_diversity(records,
                    os.path.join(plots_dir, "phase2_diversity.png"))
            except Exception as e:
                print(f"  (plot regen failed: {e})")
