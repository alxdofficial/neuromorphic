"""Phase 2 trainer: GRPO over discrete RVQ codes for the neuromodulator.

Everything except the modulator's mod_w1/b1/w2/b2 is frozen. The modulator's
continuous output is encoded via a frozen RVQ-VAE, sampled into discrete codes,
and the quantized reconstruction is applied to memory state. The training
signal is a group-relative policy gradient on windowed mem_pred_loss reward.

See docs/training_strategy.md for the full design.
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
from tqdm import tqdm

from ..model.config import Config
from ..model.model import Model
from ..codebook import ActionVQVAE


@dataclass
class CurriculumStage:
    """One stage of the phase 2 reward-window curriculum."""
    reward_window: int   # W: number of tokens in the reward window
    token_budget: int    # how many tokens to train this stage for


class Phase2Trainer:
    """GRPO trainer for the modulator with frozen everything else.

    Usage:
        trainer = Phase2Trainer(model, vqvae, dataloader, config, device, ...)
        trainer.run_curriculum([
            CurriculumStage(reward_window=512,  token_budget=25_000_000),
            CurriculumStage(reward_window=2048, token_budget=15_000_000),
            CurriculumStage(reward_window=4096, token_budget=10_000_000),
        ])
    """

    def __init__(
        self,
        model: Model,
        vqvae: ActionVQVAE,
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
        reward_mode: str = "lm_ce",
    ):
        self.model = model
        self.vqvae = vqvae
        # Default (legacy): a single dataloader for all stages.
        # New: train_loader_factory(reward_window) makes a dataloader per
        # stage. The factory decides the actual seq_length (typically
        # 2 * reward_window so most actions get complete reward windows).
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
        assert reward_mode in ("mem_pred", "lm_ce"), \
            f"reward_mode must be 'mem_pred' or 'lm_ce', got {reward_mode}"
        self.reward_mode = reward_mode

        # Freeze everything except the modulator's MLP params
        for p in model.parameters():
            p.requires_grad = False
        self.trainable_params = [
            model.memory.mod_w1, model.memory.mod_b1,
            model.memory.mod_w2, model.memory.mod_b2,
        ]
        for p in self.trainable_params:
            p.requires_grad = True
        for p in vqvae.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, betas=(0.9, 0.95))
        self.global_step = 0

        # Current curriculum stage (set by run_curriculum)
        self.reward_window: int = 0
        self.segment_length: int = 0

        # JSONL metrics log
        self.metrics_path = metrics_path
        if metrics_path is not None:
            os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    def _append_metrics(self, metrics: dict):
        if self.metrics_path is None:
            return
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    @torch.no_grad()
    def evaluate(self, eval_loader, n_batches: int = 8,
                 warmup_batches: int = 0) -> dict:
        """Held-out eval pass using the CONTINUOUS modulator path.

        Evaluates the underlying continuous head (forward_chunk). Complements
        evaluate_quantized() below which uses the actual VQ-argmax policy
        that phase 2 GRPO is optimizing. Reporting both lets us detect
        divergence between the continuous head and the deployed discrete
        policy (proxy drift).

        warmup_batches: batches to warm memory state before scoring.

        Uses a fresh memory state (snapshot + restore around the call) so eval
        is independent of the current phase-2 rollout memory carry.
        """
        model = self.model
        memory = model.memory

        train_mem_state = (
            memory.runtime_state_dict() if memory._initialized else None
        )
        train_lm_carries = [
            h.clone() if h is not None else None for h in model.lm._carries
        ]
        train_initialized = model._initialized
        was_training = model.training

        # Pause action collection during eval (phase 2 shouldn't have it set,
        # but be safe to prevent eval actions from polluting any buffer).
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
        try:
            for i, batch in enumerate(eval_loader):
                if i >= total_batches:
                    break
                input_ids = batch.input_ids.to(self.device, non_blocking=True)
                target_ids = batch.target_ids.to(self.device, non_blocking=True)
                batch_prev = getattr(batch, "prev_token", None)
                if batch_prev is not None:
                    batch_prev = batch_prev.to(self.device, non_blocking=True)
                result = model.forward_chunk(
                    input_ids, target_ids=target_ids,
                    use_memory=True,
                    prev_token=batch_prev if batch_prev is not None else prev_token,
                )
                prev_token = input_ids[:, -1]
                if i < warmup_batches:
                    continue
                total_ce += result["ce_loss"].item()
                total_aux += result["aux_loss"].item()
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
            return {"eval_ce_loss": 0.0, "eval_aux_loss": 0.0,
                    "eval_ppl": 0.0, "eval_batches": 0}
        avg_ce = total_ce / count
        avg_aux = total_aux / count
        return {
            "eval_ce_loss": avg_ce,
            "eval_aux_loss": avg_aux,
            "eval_ppl": min(math.exp(avg_ce), 1e6),
            "eval_batches": count,
        }

    @torch.no_grad()
    def evaluate_quantized(self, eval_loader, n_batches: int = 4,
                           warmup_batches: int = 0) -> dict:
        """Eval using the VQ-ARGMAX (deterministic quantized) policy.

        This is the actual policy phase 2 GRPO is trying to optimize, as
        opposed to the continuous head that evaluate() measures. If
        evaluate_continuous and evaluate_quantized diverge, the VQ bottleneck
        is losing meaningful signal and the continuous head metric is
        over-reporting real phase-2 quality.

        warmup_batches: batches to warm memory state before scoring (same
        semantics as evaluate() — ensures apples-to-apples comparison).
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

                # 1) Lower scan (reuse the helper that chunks by config.T)
                H_mid = self._run_lower_scan(
                    input_ids, prev_token=effective_prev)

                # 2) VQ-argmax rollout to produce quantized readouts
                result = memory.forward_segment_phase2(
                    H_mid, input_ids, lm, self.vqvae,
                    tau=self.tau, sample=False,
                    prev_token=effective_prev)
                readouts_q = result["readouts"]

                # 3) Upper scan on H_enriched. Carries persist across eval
                # batches (matching phase 1 eval semantics) — they're reset
                # only at EOT via the reset_mask. Previous version zeroed
                # all upper carries every batch (codex finding #5).
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
                # Chunked upper scan + CE to avoid materializing [BS, T, V].
                chunk_t = 128
                target_ids = batch.target_ids.to(self.device, non_blocking=True)
                total_ce_batch = 0.0
                total_valid_batch = 0.0
                for s in range(0, T, chunk_t):
                    e = min(s + chunk_t, T)
                    rm = reset_mask[:, s:e] if reset_mask is not None else None
                    H_up = lm.forward_scan_upper(H_enriched[:, s:e], reset_mask=rm)
                    chunk_logits = lm.forward_output(H_up)  # [BS, chunk, V]
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

                # Warmup batches: run forward to warm memory state, don't score.
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
    # Rollout + reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_lower_scan(
        self, input_ids: Tensor, prev_token: Tensor | None = None,
    ) -> Tensor:
        """Run the frozen LM lower scan to produce H_mid.

        Chunks the input into segments of config.T to bound per-chunk VRAM.
        Mirrors the EOT reset logic in `Model.forward_chunk`: builds a
        reset_mask from in-batch EOT positions and the optional `prev_token`
        (last token of previous batch).
        """
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

    @torch.no_grad()
    def _compute_per_token_reward(
        self, readouts: Tensor, prev_readout_at_start: Tensor, input_ids: Tensor,
        H_mid: Tensor | None = None,
        prev_token: Tensor | None = None,
    ) -> Tensor:
        """Compute per-token reward for GRPO.

        Two modes (selected by self.reward_mode):
          - 'mem_pred': fast — uses the weight-tied memory-prediction head.
            Rewards the modulator for making readouts that predict the next
            token well directly. Cheap but a proxy for LM CE.
          - 'lm_ce': principled — runs the full LM path (upper scan + head)
            on H_enriched = H_mid + mem_scale * readout and uses -CE of the
            main LM output as reward. Measures the actual quantity we care
            about but ~2x more expensive per rollout.

        Masks positions where the previous token was EOT (same as phase 1
        main CE loss masking).

        Returns: [BS, T] float — per-token reward. Invalid positions get 0.
        """
        BS, T, D = readouts.shape
        eot = self.config.eot_id
        valid_mask = torch.ones(BS, T, device=readouts.device, dtype=torch.float32)
        if T > 1:
            valid_mask[:, 1:] = (input_ids[:, :-1] != eot).float()
        # Mask position 0 if the previous chunk ended with EOT (cross-document
        # boundary — same logic as phase 1's mem_pred_loss valid_mask).
        if prev_token is not None:
            valid_mask[:, 0] = (prev_token.to(input_ids.device) != eot).float()

        if self.reward_mode == "lm_ce" and H_mid is not None:
            return self._compute_per_token_reward_lm_ce(
                readouts, H_mid, input_ids, valid_mask)

        # mem_pred mode (default/legacy)
        # Shift: readout[t-1] predicts token at position t.
        shifted = torch.cat([
            prev_readout_at_start.unsqueeze(1).to(readouts.dtype),
            readouts[:, :-1],
        ], dim=1)  # [BS, T, D]

        rewards = torch.empty(BS, T, device=readouts.device, dtype=torch.float32)
        chunk = max(32, 4096 // max(BS, 1))
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            sub_readout = shifted[:, s:e]
            sub_target = input_ids[:, s:e]
            sub_logits = self.model.lm.mem_head_logits(sub_readout)   # [BS, chunk, V]
            ce = F.cross_entropy(
                sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                sub_target.reshape(-1),
                reduction="none",
            ).reshape(BS, e - s)
            rewards[:, s:e] = -ce * valid_mask[:, s:e]
        return rewards

    @torch.no_grad()
    def _compute_per_token_reward_lm_ce(
        self, readouts: Tensor, H_mid: Tensor, input_ids: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Compute -LM CE as per-token reward.

        Runs H_enriched = H_mid + mem_scale * readouts through the
        (frozen) upper scan and LM head, computes per-token CE on the
        NEXT-token prediction (logits[t] → input_ids[t+1] shifted).

        Upper scan carries are reset fresh inside this computation — we
        measure per-rollout LM quality, not cross-rollout sequence quality.
        """
        BS, T, D = readouts.shape
        lm = self.model.lm
        mem_scale = lm.mem_scale  # frozen in phase 2
        H_enriched = H_mid.to(readouts.dtype) + mem_scale * readouts

        # Build reset mask for upper scan from in-batch EOT positions
        eot = self.config.eot_id
        eos_positions = (input_ids == eot)
        reset_mask = torch.zeros_like(eos_positions)
        reset_mask[:, 1:] = eos_positions[:, :-1]

        # Save / reset upper-scan carries so the rollout's CE is independent
        # of outer-loop context. Only upper-scan carries matter here; lower
        # scan is already done (H_mid is cached).
        saved_carries = [
            h.clone() if h is not None else None for h in lm._carries]
        split = self.config.scan_split_at
        for i in range(split, self.config.L_total):
            lm._carries[i] = None  # fresh at K*BS shape

        try:
            # Compute per-token CE in chunks without materializing full
            # [K*BS, T, V] logits tensor (OOMs at K=8, BS=8, T=1024).
            # Each chunk runs upper scan → head → CE and discards logits.
            chunk_t = 128
            ce_chunks = []
            for s in range(0, T, chunk_t):
                e = min(s + chunk_t, T)
                H_chunk = H_enriched[:, s:e]
                rm_chunk = reset_mask[:, s:e] if reset_mask.any() else None
                H_upper = lm.forward_scan_upper(H_chunk, reset_mask=rm_chunk)
                chunk_logits = lm.forward_output(H_upper)  # [BS, chunk, V]
                ce_chunks.append(chunk_logits)
            # Compute shifted CE: logits[t-1] predicts input_ids[t].
            # Concatenate only to do the shift, then chunk the CE itself.
            # To avoid full concat, handle the shift across chunk boundaries:
            # prev_last holds the last logit from the previous chunk.
            all_ce = []
            prev_last_logit = None
            chunk_idx = 0
            for s in range(0, T, chunk_t):
                e = min(s + chunk_t, T)
                chunk_logits = ce_chunks[chunk_idx]  # [BS, chunk, V]
                chunk_targets = input_ids[:, s:e]     # [BS, chunk]

                # Build shifted pairs: logits[t-1] predicts target[t].
                # For positions within this chunk: logits[s..e-2] predict input_ids[s+1..e-1]
                # For the first position of this chunk: prev_last_logit predicts input_ids[s]
                if prev_last_logit is not None:
                    # Prepend last logit from previous chunk
                    shifted = torch.cat([prev_last_logit, chunk_logits[:, :-1]], dim=1)
                else:
                    # First chunk: no logit for position 0, use chunk_logits[:-1]
                    shifted = chunk_logits[:, :-1]
                    chunk_targets = chunk_targets[:, 1:]

                if shifted.shape[1] > 0:
                    chunk_ce = F.cross_entropy(
                        shifted.reshape(-1, shifted.shape[-1]).float(),
                        chunk_targets.reshape(-1),
                        reduction="none",
                    ).reshape(BS, -1)
                    all_ce.append(chunk_ce)

                prev_last_logit = chunk_logits[:, -1:].detach()
                chunk_idx += 1
            del ce_chunks  # free logits memory
            ce = torch.cat(all_ce, dim=1)  # [BS, T-1]
        finally:
            # Restore the outer-loop upper carries
            lm._carries = saved_carries

        # Align reward at position t (reward for predicting token t from
        # action context at t-1). The first position has no reward (no
        # prior token to predict from). Pad at position 0 with zero.
        rewards = torch.zeros(BS, T, device=readouts.device, dtype=torch.float32)
        rewards[:, 1:] = -ce
        rewards = rewards * valid_mask
        return rewards

    @torch.no_grad()
    def _windowed_reward(
        self, per_token_reward: Tensor, call_positions: Tensor, window: int,
    ) -> Tensor:
        """For each modulator call at token t, compute mean reward over [t+1, t+1+W].

        ONLY calls whose full window [t+1, t+1+W) fits entirely within the
        sequence contribute a meaningful reward — calls near the end of the
        sequence would otherwise get truncated windows that bias the reward
        signal toward early positions. Out-of-range calls get zero reward
        (effectively masked from advantages by centering).

        For the curriculum to produce meaningful rewards at window W, the
        rollout sequence length must exceed W by at least a few modulator
        intervals (train_phase2 sets seq_length = 2 * W as the default).

        Args:
            per_token_reward: [*, BS, T] on-device (leading dims broadcast, e.g. [K, BS, T])
            call_positions: [n_calls] cpu long tensor — absolute token index of each call
            window: reward window W

        Returns:
            rewards: [*, n_calls, BS] on-device — per-action reward averaged over
                     the full window (only for actions with complete windows)
        """
        T = per_token_reward.shape[-1]
        positions = call_positions.to(per_token_reward.device)

        # Build [n_calls, W] index tensor for vectorized gather
        starts = positions + 1                                        # [n_calls]
        offsets = torch.arange(window, device=per_token_reward.device)  # [W]
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)          # [n_calls, W]

        # An action's window is COMPLETE iff start + W <= T. Incomplete
        # windows contribute zero reward (they're discarded via the mask
        # below). Since group-relative advantages subtract per-sample
        # means, zero reward for invalid actions won't introduce phantom
        # gradient signal.
        complete = (starts + window <= T)                             # [n_calls]
        indices = indices.clamp(max=T - 1)

        gathered = per_token_reward[..., indices]                     # [*, BS, n_calls, W]
        # Mean over the full window (all W tokens valid when complete).
        out = gathered.mean(dim=-1)                                   # [*, BS, n_calls]
        # Zero out incomplete windows.
        out = out * complete.to(out.dtype)
        # Move n_calls before BS: [..., n_calls, BS]
        return out.transpose(-1, -2)

    @torch.no_grad()
    def _save_mem_state(self) -> dict:
        """Snapshot the memory runtime state so we can reset between rollouts."""
        mem = self.model.memory
        return {
            "h": mem.h.clone(), "msg": mem.msg.clone(), "W": mem.W.clone(),
            "decay": mem.decay.clone(), "hebbian": mem.hebbian.clone(),
            "s_mem_live": mem.s_mem_live.clone(),
            "s_mem_ema_fast": mem.s_mem_ema_fast.clone(),
            "prev_readout": mem.prev_readout.clone(),
            "readout_drift": mem.readout_drift.clone(),
        }

    @torch.no_grad()
    def _restore_mem_state(self, snapshot: dict):
        mem = self.model.memory
        mem.h = snapshot["h"].clone()
        mem.msg = snapshot["msg"].clone()
        mem.W = snapshot["W"].clone()
        mem.decay = snapshot["decay"].clone()
        mem.hebbian = snapshot["hebbian"].clone()
        mem.s_mem_live = snapshot["s_mem_live"].clone()
        mem.s_mem_ema_fast = snapshot["s_mem_ema_fast"].clone()
        mem.prev_readout = snapshot["prev_readout"].clone()
        mem.readout_drift = snapshot["readout_drift"].clone()

    @torch.no_grad()
    def rollout(self, batch) -> dict:
        """Run K trajectories on the same batch in parallel via batch-expanded memory.

        Instead of looping K times sequentially, we expand the memory state's
        batch dimension from BS to K*BS and run a single forward_segment_phase2.
        Each of the K "sub-batches" gets the same H_mid and input_ids but
        independent stochastic VQ sampling, producing K divergent trajectories
        in one pass.

        Returns dict:
            mod_inputs: [K, n_calls, BS, NC, mod_in]
            codes:      [K, n_calls, BS, NC, num_levels]
            rewards:    [K, n_calls, BS]  per-action reward
        """
        device = self.device
        K = self.group_size
        input_ids = batch.input_ids.to(device, non_blocking=True)
        prev_token = getattr(batch, "prev_token", None)
        BS, T = input_ids.shape

        self.model.lm.detach_carries()

        # Run lower scan once (LM is frozen and deterministic per batch).
        H_mid = self._run_lower_scan(input_ids, prev_token=prev_token)

        # Snapshot initial memory state at BS
        init_snapshot = self._save_mem_state()

        # Expand memory state: replicate each tensor K times along batch dim.
        # After this, memory operates on K*BS "samples" simultaneously.
        mem = self.model.memory
        for key, val in init_snapshot.items():
            expanded = val.unsqueeze(0).expand(K, *val.shape).reshape(K * BS, *val.shape[1:])
            setattr(mem, key, expanded.clone())

        # Expand H_mid and input_ids: [BS, T, D] -> [K*BS, T, D]
        H_mid_exp = H_mid.unsqueeze(0).expand(K, *H_mid.shape).reshape(K * BS, *H_mid.shape[1:])
        ids_exp = input_ids.unsqueeze(0).expand(K, *input_ids.shape).reshape(K * BS, *input_ids.shape[1:])

        # Also expand identity matrix cache

        # Single batched rollout — all K trajectories run in parallel
        result = mem.forward_segment_phase2(
            H_mid_exp, ids_exp, self.model.lm, self.vqvae,
            tau=self.tau, sample=True)

        # Unpack: split K*BS back to (K, BS) using .view() so any
        # non-contiguous layout bugs raise immediately instead of silently
        # scrambling trajectory assignments.
        readouts = result["readouts"].view(K, BS, T, -1)              # [K, BS, T, D]
        call_positions = result["call_positions"]                     # [n_calls] cpu
        n_calls = result["mod_inputs"].shape[0]
        mod_inputs = result["mod_inputs"].view(
            n_calls, K, BS, *result["mod_inputs"].shape[2:]).transpose(0, 1)  # [K, n_calls, BS, NC, mod_in]
        codes = result["codes"].view(
            n_calls, K, BS, *result["codes"].shape[2:]).transpose(0, 1)       # [K, n_calls, BS, NC, L]

        # Compute per-trajectory reward
        init_prev_readout = init_snapshot["prev_readout"]  # [BS, D]
        init_prev_exp = init_prev_readout.unsqueeze(0).expand(K, *init_prev_readout.shape).reshape(K * BS, -1)
        # Expand prev_token to K*BS for cross-doc masking at position 0
        prev_token_exp = None
        if prev_token is not None:
            prev_token_exp = prev_token.unsqueeze(0).expand(K, *prev_token.shape).reshape(K * BS)
        per_token_reward = self._compute_per_token_reward(
            result["readouts"], init_prev_exp, ids_exp,
            H_mid=H_mid_exp, prev_token=prev_token_exp)     # [K*BS, T]
        per_token_reward = per_token_reward.reshape(K, BS, T)

        # Windowed rewards — vectorized across K trajectories
        rewards = self._windowed_reward(
            per_token_reward, call_positions, self.reward_window)  # [K, n_calls, BS]

        # Per-sample reward for logging.
        # `mean_per_k_sample[k, b]` = mean reward across all action calls
        # for trajectory k of sample b. Shape [K, BS].
        mean_per_k_sample = rewards.mean(dim=1)  # [K, BS]
        # True mean across all K trajectories and all samples — this is
        # the "average reward" you'd expect from the field name `mean_reward`.
        mean_reward = mean_per_k_sample.mean().item()
        # Max-over-K then mean-over-BS: the average of "best trajectory
        # per sample" rewards, i.e. how lucky the best-of-K is. Logged
        # separately for diagnostics but systematically optimistic.
        best_mean_reward = mean_per_k_sample.max(dim=0).values.mean().item()

        # Choose ONE trajectory per sample uniformly at random to carry
        # its end-state forward. Picking the *best* trajectory would
        # introduce optimism bias — the starting state for the next batch
        # would be systematically "lucky" and diverge from the state
        # distribution the deployed policy actually produces. Random
        # selection gives unbiased expected-state propagation across
        # batches.
        random_k_per_sample = torch.randint(
            0, K, (BS,), device=device, dtype=torch.long)
        gather_idx = random_k_per_sample * BS + torch.arange(BS, device=device)
        for key in init_snapshot:
            full = getattr(mem, key)  # [K*BS, ...]
            setattr(mem, key, full[gather_idx].clone())

        return {
            "mod_inputs": mod_inputs,
            "codes": codes,
            "rewards": rewards,
            "mean_reward": mean_reward,             # true mean across K×BS
            "best_mean_reward": best_mean_reward,   # max-over-K then mean-over-BS
            "T": T,
            "BS": BS,
        }

    # ------------------------------------------------------------------
    # GRPO gradient step
    # ------------------------------------------------------------------

    def _compute_advantages(self, rewards: Tensor) -> Tensor:
        """Group-relative advantages.

        rewards: [K, n_calls, BS] — per-action reward per trajectory
        Returns: [K, n_calls, BS] — normalized advantages

        Normalization is PER-(call, sample): each (call_idx, sample_idx)
        slot is centered and scaled by its own K-variance. We used to use
        a global std, which biased the gradient toward low-variance
        early-sequence actions (late actions have higher reward variance
        because the memory state diverges more across K by then, so the
        global std was dominated by late positions and early ones got
        proportionally larger advantages). See audit #6.
        """
        # Per-action baseline: mean over K trajectories.
        baseline = rewards.mean(dim=0, keepdim=True)         # [1, n_calls, BS]
        advantages = rewards - baseline                      # [K, n_calls, BS]
        # Per-(call, sample) std over K. Use unbiased=False to avoid NaN when
        # K=1 (population variance of a single element is 0, not NaN).
        std = advantages.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-8)
        return advantages / std

    def grpo_step(self, rollout_result: dict) -> dict:
        """Compute GRPO loss from rollout records and update modulator.

        The gradient pass:
          1. Flatten all (mod_input, codes) records across K x n_calls x BS x NC
             into a single batch of modulator inputs.
          2. Run modulator forward -> raw action -> vqvae encoder -> latent z.
             This is the only on-graph computation.
          3. Compute log pi(codes | z) via distance-based categorical over each
             RVQ level.
          4. Loss = -(advantage * log_pi).mean(), backward, step.

        Memory dynamics, LM, mem_head, decoder — all frozen and off the graph.
        """
        K, n_calls, BS, NC, mod_in_dim = rollout_result["mod_inputs"].shape
        num_levels = rollout_result["codes"].shape[-1]

        # Records already live on device from forward_segment_phase2
        mod_inputs = rollout_result["mod_inputs"]    # [K, n_calls, BS, NC, mod_in] f32 gpu
        codes = rollout_result["codes"]              # [K, n_calls, BS, NC, L] long gpu
        rewards = rollout_result["rewards"]          # [K, n_calls, BS] f32 gpu

        # Advantages — broadcast over NC since actions are per-cell
        advantages = self._compute_advantages(rewards)                 # [K, n_calls, BS]
        advantages = advantages.unsqueeze(-1).expand(-1, -1, -1, NC)   # [K, n_calls, BS, NC]

        # Flatten all records into [M, ...] where M = K * n_calls * BS * NC
        mod_inputs_flat = mod_inputs.reshape(-1, NC, mod_in_dim)       # [K*n_calls*BS, NC, mod_in]
        codes_flat = codes.reshape(-1, NC, num_levels)                 # same
        advantages_flat = advantages.reshape(-1, NC)                   # same
        # Each per-cell action is its own "record"; reshape [M, mod_in] by folding NC
        # But the modulator forward is per-cell (einsum "bni,nih->bnh") and needs
        # the NC dim. So we'll process [M, NC, ...] as a big batch.

        # Chunk the gradient pass to bound VRAM
        M = mod_inputs_flat.shape[0]
        # Scale chunk size with codes_per_level — more codes = bigger distance
        # tensors in log_prob and entropy ([C*NC, codes, latent_dim])
        codes_per_level = self.vqvae.rvq.codes_per_level
        chunk_size = max(1024, 16384 // max(codes_per_level // 16, 1))
        total_loss = 0.0
        total_log_pi = 0.0
        total_entropy = 0.0
        # Quantization residual accumulator (continuous-vs-quantized gap).
        quant_residual_sq = 0.0
        raw_action_sq = 0.0
        n_chunks = 0

        self.optimizer.zero_grad(set_to_none=True)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            mi_chunk = mod_inputs_flat[start:end]          # [C, NC, mod_in]
            c_chunk = codes_flat[start:end]                # [C, NC, L]
            a_chunk = advantages_flat[start:end]           # [C, NC]
            C = mi_chunk.shape[0]

            # Modulator forward (on-graph). mi_chunk is already f32 from the rollout.
            raw_action = self.model.memory._modulator_forward(mi_chunk)  # [C, NC, mod_out]

            # Flatten [C, NC] -> [C*NC] for the VQ encoder
            action_flat = raw_action.reshape(C * NC, -1)
            action_norm = self.vqvae.normalize(action_flat)
            z = self.vqvae.encoder(action_norm)                # [C*NC, latent]

            # log pi(codes | z) summed across RVQ levels (on-graph through z)
            codes_flat_flat = c_chunk.reshape(C * NC, num_levels)
            log_pi = self.vqvae.rvq.log_prob(z, codes_flat_flat, tau=self.tau)  # [C*NC]
            log_pi = log_pi.reshape(C, NC)

            # Entropy bonus: encourage policy diversity to prevent code collapse
            entropy = self.vqvae.rvq.entropy(z, tau=self.tau).reshape(C, NC)

            # GRPO loss - entropy bonus (normalized by full M)
            chunk_loss = (-(a_chunk * log_pi) - self.entropy_coeff * entropy).sum() / (M * NC)
            chunk_loss.backward()

            total_loss += chunk_loss.item() * (M * NC)
            total_log_pi += log_pi.detach().sum().item()
            total_entropy += entropy.detach().sum().item()

            # Quantization residual: ||raw_action - decode(quantize(raw_action))||.
            # Measures how much information is lost through the VQ bottleneck —
            # a direct signal of continuous-vs-quantized policy divergence.
            with torch.no_grad():
                lvl_idx = torch.arange(num_levels, device=z.device)
                z_q = self.vqvae.rvq.codebooks[lvl_idx.unsqueeze(0), codes_flat_flat].sum(dim=1)
                decoded_norm = self.vqvae.decoder(z_q)
                decoded = self.vqvae.denormalize(decoded_norm)
                resid = (action_flat - decoded)
                quant_residual_sq += resid.pow(2).sum().item()
                raw_action_sq += action_flat.pow(2).sum().item()
            n_chunks += 1

        grad_norm = nn.utils.clip_grad_norm_(
            self.trainable_params, self.max_grad_norm).item()
        self.optimizer.step()

        # Reward distribution stats over the flattened records
        rewards_flat = rollout_result["rewards"].float()
        reward_mean = rewards_flat.mean().item()
        reward_std = rewards_flat.std().item()
        reward_min = rewards_flat.min().item()
        reward_max = rewards_flat.max().item()

        # Codebook usage over the sampled codes (fraction of unique tuples)
        codes_flat_all = rollout_result["codes"].reshape(-1, num_levels)
        n_unique = torch.unique(codes_flat_all, dim=0).shape[0]

        # Quantization residual metrics
        import math
        quant_resid_norm = math.sqrt(quant_residual_sq / max(M * NC, 1))
        raw_norm = math.sqrt(raw_action_sq / max(M * NC, 1))
        quant_relative = quant_resid_norm / max(raw_norm, 1e-8)

        return {
            "loss": total_loss / (M * NC),
            "log_pi_mean": total_log_pi / (M * NC),
            "entropy_mean": total_entropy / (M * NC),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "mod_grad_norm": grad_norm,
            "n_chunks": n_chunks,
            "M": M,
            "n_unique_codes": n_unique,
            # Continuous vs quantized: high residual = proxy drift risk.
            "quant_resid_rel": quant_relative,
            "quant_resid_abs": quant_resid_norm,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run_curriculum(self, stages: list[CurriculumStage]):
        """Run the full phase-2 curriculum, advancing through stages sequentially.

        If a train_loader_factory is provided, each stage gets its own
        dataloader at seq_length == stage.reward_window — so early stages
        actually process shorter sequences (cheaper per step) and the
        curriculum is a true easy-to-hard ramp. Without a factory, falls
        back to the single dataloader passed at init time.
        """
        for stage_idx, stage in enumerate(stages):
            self.reward_window = stage.reward_window
            self.segment_length = stage.reward_window
            print(f"\n=== Phase 2 stage {stage_idx+1}/{len(stages)}: "
                  f"W={stage.reward_window}, budget={stage.token_budget:,} tokens ===")
            if self.train_loader_factory is not None:
                # Pass stage_idx so the factory can offset the seed — without
                # this every stage starts at the same shard prefix. See the
                # comment in train_phase2.py train_loader_factory for details.
                try:
                    self.dataloader = self.train_loader_factory(
                        stage.reward_window, stage_idx=stage_idx)
                except TypeError:
                    # Back-compat for factories that only accept reward_window.
                    self.dataloader = self.train_loader_factory(
                        stage.reward_window)
            self._run_stage(stage)

    def _run_stage(self, stage: CurriculumStage):
        """Run one curriculum stage until token_budget is reached."""
        tokens_seen = 0
        t_stage_start = time.time()
        for step_idx, batch in enumerate(self.dataloader):
            # Assume batch.input_ids.shape == [BS, T]; token budget is on total
            # tokens seen across all steps.
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

            # Diagnostic: measure lane divergence (does not modify state).
            lane_stats = {}
            if self.global_step % self.log_interval == 0:
                lane_stats = self.model.memory.compute_lane_divergence()

            # Persistent jsonl logging for plotting
            log_row = {
                "step": self.global_step,
                "stage_window": stage.reward_window,
                "tokens_seen": tokens_seen,
                "rollout_time": t_rollout,
                "grad_time": t_grad,
                **step_metrics,
                **lane_stats,
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
                      f"records={step_metrics['M']} "
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
                # Also eval the VQ-argmax (deterministic quantized) policy
                # — this is the actual policy phase 2 is optimizing, and
                # divergence from the continuous eval indicates proxy drift.
                quant_metrics = self.evaluate_quantized(
                    self.eval_loader_factory(),
                    n_batches=self.eval_batches,
                    warmup_batches=self.eval_warmup_batches)
                print(f"[p2 eval {self.global_step}] "
                      f"ce={eval_metrics['eval_ce_loss']:.3f} "
                      f"q_ce={quant_metrics['quant_eval_ce']:.3f} "
                      f"ppl={eval_metrics['eval_ppl']:.1f} "
                      f"mem_pred={eval_metrics['eval_aux_loss']:.3f} "
                      f"({eval_metrics['eval_batches']} batches)")
                self._append_metrics({
                    "step": self.global_step,
                    "stage_window": stage.reward_window,
                    "event": "eval",
                    **eval_metrics,
                    **quant_metrics,
                })

        elapsed = time.time() - t_stage_start
        print(f"  stage complete: {tokens_seen:,} tokens in {elapsed:.0f}s")

        # Auto-regenerate plots at end of each stage
        if self.metrics_path is not None:
            try:
                from scripts.plot_training import load_metrics, plot_phase2_grpo
                plots_dir = os.path.join(os.path.dirname(self.metrics_path), "plots")
                os.makedirs(plots_dir, exist_ok=True)
                records = load_metrics(self.metrics_path)
                plot_phase2_grpo(records,
                    os.path.join(plots_dir, "phase2_grpo.png"))
            except Exception as e:
                print(f"  (plot regen failed: {e})")
