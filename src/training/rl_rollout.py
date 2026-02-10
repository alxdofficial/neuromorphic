"""
RL counterfactual rollout engine for neuromorphic LM neuromodulator training.

Extracted from trainer.py. Contains BoundarySnapshot, all rollout logic,
and neuromodulator update steps. Only active in Phase D+.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.config import ModelConfig
from ..model.model import NeuromorphicLM
from ..model.state import save_runtime_state, load_runtime_state
from . import span_ops


@dataclass
class BoundarySnapshot:
    """State at a span boundary for counterfactual rollouts.

    Captured BEFORE PM commit + EM write so the rollout can apply
    force_on/off and then measure its effect on the next span.
    """
    runtime_state: dict
    span_start: int              # position in chunk where rollout's NEXT span starts
    input_ids: Tensor            # full chunk input (reference, not copy)
    target_ids: Tensor           # full chunk targets (reference, not copy)
    span_surprise_mean: Tensor   # [BS] surprise for this span
    pm_elig_norms: dict = field(default_factory=dict)   # {(b_idx, l_idx): [BS]}
    pm_usages: dict = field(default_factory=dict)        # {(b_idx, l_idx): [BS]}
    em_novelties: dict = field(default_factory=dict)     # {b_idx: [BS]}
    em_usages: dict = field(default_factory=dict)         # {b_idx: [BS]}
    # EM candidate buffers for forced write in rollout
    em_cand_K: list = field(default_factory=list)         # [B] of [BS, P, D_em]
    em_cand_V: list = field(default_factory=list)         # [B] of [BS, P, D_em]
    em_cand_score: list = field(default_factory=list)     # [B] of [BS, P]
    em_cand_valid: list = field(default_factory=list)     # [B] of [BS, P]
    em_g_em_chosen: dict = field(default_factory=dict)    # {b_idx: [BS]}
    em_tau_chosen: dict = field(default_factory=dict)    # {b_idx: [BS]}
    em_ww_chosen: dict = field(default_factory=dict)     # {b_idx: [BS]}


def detached_runtime_state(model: NeuromorphicLM) -> dict:
    """Save runtime state with all tensors detached+cloned for safe deepcopy."""
    state = save_runtime_state(model)
    detached = {}
    for path, sub in state.items():
        detached[path] = {}
        for name, val in sub.items():
            if val is not None and isinstance(val, Tensor):
                detached[path][name] = val.detach().clone()
            else:
                detached[path][name] = val
    return detached


def select_rl_spans(num_spans: int, rl_events: int) -> list[int]:
    """Select evenly-spaced span indices for RL rollouts.

    Excludes the last span (index num_spans-1) because the trainer needs
    at least one future span after the snapshot for the counterfactual
    rollout to measure reward against.
    """
    # Last span can't be snapshot'd (no future tokens for reward), so exclude it
    usable = num_spans - 1
    if rl_events <= 0 or usable <= 0:
        return []
    rl_events = min(rl_events, usable)
    step = usable / rl_events
    return [int(i * step + step / 2) for i in range(rl_events)]


class RLRolloutEngine:
    """Manages counterfactual rollouts for RL neuromodulator training."""

    def __init__(
        self,
        model: NeuromorphicLM,
        config: ModelConfig,
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
        rl_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.rl_optimizer = rl_optimizer

        # RL warmup state
        self._rl_warmup_steps: int = 0
        self._rl_warmup_step: int = 0
        self._rl_base_lr: float = config.rl_lr if config.rl_enabled else 0.0

    def set_rl_warmup(self, warmup_steps: int):
        """Enable RL optimizer LR warmup for the next N steps."""
        self._rl_warmup_steps = warmup_steps
        self._rl_warmup_step = 0
        if self.rl_optimizer is not None and warmup_steps > 0:
            for pg in self.rl_optimizer.param_groups:
                pg["lr"] = 0.0

    def _tick_rl_warmup(self):
        """Advance RL warmup by one step, scaling LR linearly."""
        if self._rl_warmup_steps <= 0 or self.rl_optimizer is None:
            return
        self._rl_warmup_step += 1
        if self._rl_warmup_step >= self._rl_warmup_steps:
            for pg in self.rl_optimizer.param_groups:
                pg["lr"] = self._rl_base_lr
            self._rl_warmup_steps = 0
        else:
            scale = self._rl_warmup_step / self._rl_warmup_steps
            for pg in self.rl_optimizer.param_groups:
                pg["lr"] = self._rl_base_lr * scale

    def collect_neuromod_grad_norms(self) -> dict:
        """Capture neuromodulator gradient norms before optimizer zeros them."""
        norms = {}
        for b_idx, block in enumerate(self.model.blocks):
            nm_em = block.em_neuromodulator
            total_sq = 0.0
            for p in nm_em.parameters():
                if p.grad is not None:
                    total_sq += p.grad.detach().norm().item() ** 2
            if total_sq > 0:
                norms[f"gnorm_b{b_idx}_em_neuromod"] = math.sqrt(total_sq)
            for l_idx, layer in enumerate(block.layers):
                nm_pm = layer.pm_neuromodulator
                total_sq = 0.0
                for p in nm_pm.parameters():
                    if p.grad is not None:
                        total_sq += p.grad.detach().norm().item() ** 2
                if total_sq > 0:
                    norms[f"gnorm_b{b_idx}_l{l_idx}_pm_neuromod"] = math.sqrt(total_sq)
        return norms

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _select_pm_targets(self, snap: BoundarySnapshot) -> list[tuple[int, int]]:
        """Pick salient PM controllers for deconfounded rollouts."""
        if not snap.pm_elig_norms:
            return []
        k = int(getattr(self.config, "rl_pm_targets_per_event", 1))
        if k <= 0:
            return []
        ranked = sorted(
            snap.pm_elig_norms.items(),
            key=lambda kv: float(kv[1].mean().item()),
            reverse=True,
        )
        return [key for key, _ in ranked[:k]]

    def _select_em_targets(self, snap: BoundarySnapshot) -> list[int]:
        """Pick salient EM controllers for deconfounded rollouts."""
        if not snap.em_novelties:
            return []
        k = int(getattr(self.config, "rl_em_targets_per_event", 1))
        if k <= 0:
            return []
        ranked = sorted(
            snap.em_novelties.items(),
            key=lambda kv: float(kv[1].mean().item()),
            reverse=True,
        )
        return [key for key, _ in ranked[:k]]

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def _commit_pm_rollout_boundary(
        self,
        span_surprise: Tensor,
        force_mode: str = "normal",
        target: Optional[tuple[int, int]] = None,
    ):
        """PM commit helper for rollouts."""
        for b_idx, block in enumerate(self.model.blocks):
            for l_idx, layer in enumerate(block.layers):
                pm = layer.pm
                if pm.elig_K is None:
                    continue

                local_force = force_mode
                if target is not None and (b_idx, l_idx) != target:
                    local_force = "normal"

                if local_force == "force_off":
                    continue

                elig_norm = pm.elig_K.norm(dim=-1).mean(dim=-1)
                pm_usage = pm.pm_a.sum(dim=-1)

                if local_force == "force_on":
                    bs = elig_norm.shape[0]
                    commit_mask = torch.ones(bs, dtype=torch.bool, device=elig_norm.device)
                    lambda_vals = torch.full((bs,), pm.decay, device=elig_norm.device)
                    g = torch.full((bs,), 0.5, device=elig_norm.device)
                    pm.commit(commit_mask, lambda_vals, g, None)
                    continue

                surprise_input = span_surprise if span_surprise is not None else elig_norm
                commit_mask, lambda_vals, g, slot_logits, _ = layer.pm_neuromodulator.forward(
                    elig_norm,
                    pm_usage / self.config.budget_pm,
                    surprise_input,
                )
                pm.commit(commit_mask, lambda_vals, g, slot_logits)

    def _rollout_span(
        self, snap: BoundarySnapshot,
        pm_force: str, em_force: str,
        pm_target: Optional[tuple[int, int]] = None,
        em_target: Optional[int] = None,
    ) -> Tensor:
        """Run counterfactual rollout: apply forced commit/write, then measure
        loss over the next span of tokens using forward_span (parallel).
        """
        P = self.config.P
        input_ids = snap.input_ids
        target_ids = snap.target_ids
        span_start = snap.span_start
        span_end = min(span_start + P, input_ids.shape[1])
        span_P = span_end - span_start
        BS = input_ids.shape[0]
        eot_id = self.config.eot_id

        span_ids = input_ids[:, span_start:span_end]
        span_targets = target_ids[:, span_start:span_end]

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

        with torch.no_grad(), amp_ctx:
            # Step 1: Apply forced PM commit + EM write BEFORE measuring tokens.
            if self.config.pm_enabled:
                for block in self.model.blocks:
                    for layer in block.layers:
                        layer.pm.base_decay()
                self._commit_pm_rollout_boundary(
                    span_surprise=snap.span_surprise_mean,
                    force_mode=pm_force,
                    target=pm_target,
                )

            if self.config.em_enabled:
                for b, block in enumerate(self.model.blocks):
                    sK = snap.em_cand_K[b]
                    if sK is None:
                        continue
                    sV = snap.em_cand_V[b]
                    sScore = snap.em_cand_score[b]
                    sValid = snap.em_cand_valid[b]

                    local_mode = em_force
                    if em_target is not None and b != em_target:
                        local_mode = "normal"

                    neuromod = block.em_neuromodulator
                    default_g = getattr(neuromod, "default_g", 0.3)
                    default_tau = getattr(neuromod, "default_tau", self.config.tau_em)
                    default_ww = getattr(neuromod, "default_ww", self.config.weakness_weight_em)

                    if local_mode == "baseline":
                        write_mask = torch.ones(BS, dtype=torch.bool, device=self.device)
                        g_em = torch.full((BS,), default_g, device=self.device)
                        tau_em = torch.full((BS,), default_tau, device=self.device)
                        ww_em = torch.full((BS,), default_ww, device=self.device)
                    elif local_mode == "chosen":
                        write_mask = torch.ones(BS, dtype=torch.bool, device=self.device)
                        g_em = snap.em_g_em_chosen.get(
                            b, torch.full((BS,), default_g, device=self.device)
                        )
                        tau_em = snap.em_tau_chosen.get(
                            b, torch.full((BS,), default_tau, device=self.device)
                        )
                        ww_em = snap.em_ww_chosen.get(
                            b, torch.full((BS,), default_ww, device=self.device)
                        )
                    else:  # "normal"
                        em_usage = (
                            block.em.em_S.sum(dim=-1)
                            if block.em.em_S is not None
                            else torch.zeros(BS, device=self.device)
                        )
                        cvf = sValid.float()
                        cc = cvf.sum(dim=-1).clamp(min=1)
                        novelty = (sScore * cvf).sum(dim=-1) / cc
                        write_mask, g_em, tau_em, ww_em = block.em_neuromodulator.forward(
                            snap.span_surprise_mean,
                            em_usage / self.config.budget_em,
                            novelty,
                        )

                    block.em.write_at_boundary(
                        sK, sV, sScore,
                        write_mask, g_em,
                        tau=tau_em, weakness_weight=ww_em,
                        cand_valid=sValid,
                    )

            # Step 2: Set frozen surprise from snapshot for forward_span gates.
            self.model.surprise = snap.span_surprise_mean.unsqueeze(-1)

            # Step 3: Compute reset_mask_first for the span.
            if span_start == 0:
                reset_first = torch.zeros(BS, dtype=torch.bool, device=self.device)
            else:
                reset_first = (input_ids[:, span_start - 1] == eot_id)

            # Step 4: Single parallel forward pass.
            logits_all, _, _ = self.model.forward_span(span_ids, reset_first)

            # Step 5: Compute per-stream loss.
            _, loss_mask = span_ops.compute_loss_mask(
                span_ids, eot_id, self.config.reset_on_doc_boundary
            )
            per_token = F.cross_entropy(
                logits_all.reshape(BS * span_P, -1),
                span_targets.reshape(BS * span_P),
                reduction="none",
            ).reshape(BS, span_P)

            stream_loss = (per_token * loss_mask.float()).sum(dim=1)
            stream_count = loss_mask.float().sum(dim=1)

        return stream_loss / stream_count.clamp(min=1)

    # ------------------------------------------------------------------
    # Neuromodulator updates
    # ------------------------------------------------------------------

    def _update_pm_neuromodulators(
        self, snap: BoundarySnapshot, reward: Tensor,
        targets: Optional[set[tuple[int, int]]] = None,
    ) -> float:
        """Weighted BCE update for each PM neuromodulator's gate head."""
        total_loss_val = 0.0
        for (b_idx, l_idx), elig_norm in snap.pm_elig_norms.items():
            if targets is not None and (b_idx, l_idx) not in targets:
                continue
            neuromod = self.model.blocks[b_idx].layers[l_idx].pm_neuromodulator
            if not neuromod.rl_enabled:
                continue
            pm_usage_norm = snap.pm_usages.get((b_idx, l_idx))
            if pm_usage_norm is None:
                continue

            _, _, _, _, p_commit = neuromod(
                elig_norm, pm_usage_norm, snap.span_surprise_mean,
            )

            label = (reward > 0).float()
            credit = elig_norm / elig_norm.clamp(min=1e-6).max()
            weight = reward.abs() * credit

            # Clamp to avoid CUDA assert in BCE from fp16 rounding
            p_commit = p_commit.clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p_commit, label, weight=weight)
            total_loss_val += loss.item()
            loss.backward()
        return total_loss_val

    def _update_em_neuromodulators(
        self, snap: BoundarySnapshot, reward: Tensor,
        targets: Optional[set[int]] = None,
    ) -> float:
        """Continuous EM objective for g_em, tau, and weakness_weight."""
        total_loss_val = 0.0
        for b_idx in range(self.config.B):
            if targets is not None and b_idx not in targets:
                continue
            neuromod = self.model.blocks[b_idx].em_neuromodulator
            if not neuromod.rl_enabled:
                continue
            em_usage_norm = snap.em_usages.get(b_idx)
            if em_usage_norm is None:
                continue
            novelty = snap.em_novelties.get(b_idx)
            if novelty is None:
                continue
            chosen_g = snap.em_g_em_chosen.get(b_idx)
            if chosen_g is None:
                continue

            _, g_em, tau_em, ww_em = neuromod(
                snap.span_surprise_mean, em_usage_norm, novelty,
            )

            credit = novelty / novelty.clamp(min=1e-6).max()
            weight = reward.abs() * credit

            # -- g_em loss --
            baseline_g = torch.full_like(chosen_g, neuromod.default_g)
            target_g = torch.where(reward > 0, chosen_g, baseline_g)

            g_scale = max(neuromod.g_em_ceil - neuromod.g_em_floor, 1e-6)
            g_em_normalized = ((g_em - neuromod.g_em_floor) / g_scale).clamp(0.0, 1.0)
            target_g_normalized = ((target_g - neuromod.g_em_floor) / g_scale).clamp(0.0, 1.0)

            sq_err_g = (g_em_normalized - target_g_normalized).pow(2)
            loss_g = (sq_err_g * weight).sum() / weight.sum().clamp(min=1e-6)

            # -- tau loss --
            chosen_tau = snap.em_tau_chosen.get(b_idx)
            if chosen_tau is not None:
                baseline_tau = torch.full_like(chosen_tau, neuromod.default_tau)
                target_tau = torch.where(reward > 0, chosen_tau, baseline_tau)
                tau_scale = max(neuromod.tau_ceil - neuromod.tau_floor, 1e-6)
                tau_normalized = ((tau_em - neuromod.tau_floor) / tau_scale).clamp(0.0, 1.0)
                target_tau_normalized = ((target_tau - neuromod.tau_floor) / tau_scale).clamp(0.0, 1.0)
                sq_err_tau = (tau_normalized - target_tau_normalized).pow(2)
                loss_tau = (sq_err_tau * weight).sum() / weight.sum().clamp(min=1e-6)
            else:
                loss_tau = torch.tensor(0.0, device=g_em.device)

            # -- weakness_weight loss --
            chosen_ww = snap.em_ww_chosen.get(b_idx)
            if chosen_ww is not None:
                baseline_ww = torch.full_like(chosen_ww, neuromod.default_ww)
                target_ww = torch.where(reward > 0, chosen_ww, baseline_ww)
                ww_scale = max(neuromod.ww_ceil - neuromod.ww_floor, 1e-6)
                ww_normalized = ((ww_em - neuromod.ww_floor) / ww_scale).clamp(0.0, 1.0)
                target_ww_normalized = ((target_ww - neuromod.ww_floor) / ww_scale).clamp(0.0, 1.0)
                sq_err_ww = (ww_normalized - target_ww_normalized).pow(2)
                loss_ww = (sq_err_ww * weight).sum() / weight.sum().clamp(min=1e-6)
            else:
                loss_ww = torch.tensor(0.0, device=g_em.device)

            total_loss = loss_g + loss_tau + loss_ww
            total_loss_val += total_loss.item()
            total_loss.backward()
        return total_loss_val

    # ------------------------------------------------------------------
    # Main RL step
    # ------------------------------------------------------------------

    def rl_step(
        self,
        snapshots: list[BoundarySnapshot],
        final_runtime_state: dict,
    ) -> dict:
        """Full RL step: rollouts + neuromod updates. Returns rl_metrics dict."""
        if not snapshots:
            return {}

        rl_metrics = {
            "rl_pm_reward_mean": 0.0,
            "rl_em_reward_mean": 0.0,
            "rl_pm_gate_loss": 0.0,
            "rl_em_g_loss": 0.0,
            "rl_events": len(snapshots),
        }
        pm_reward_sum = 0.0
        em_reward_sum = 0.0
        pm_gate_loss_sum = 0.0
        em_gate_loss_sum = 0.0
        pm_count = 0
        em_count = 0

        for snap in snapshots:
            # -- PM rollout --
            if self.config.pm_enabled and self.config.rl_enabled:
                pm_targets = self._select_pm_targets(snap)
                for target in pm_targets:
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_off = self._rollout_span(
                        snap, pm_force="force_off", em_force="normal",
                        pm_target=target,
                    )
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_on = self._rollout_span(
                        snap, pm_force="force_on", em_force="normal",
                        pm_target=target,
                    )
                    reward = loss_off - loss_on
                    pm_reward_sum += reward.mean().item()
                    pm_count += 1
                    pm_gate_loss_sum += self._update_pm_neuromodulators(
                        snap, reward, targets={target}
                    )

            # -- EM rollout --
            if self.config.em_enabled and self.config.rl_enabled:
                em_targets = self._select_em_targets(snap)
                for b_idx in em_targets:
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_baseline = self._rollout_span(
                        snap, pm_force="normal", em_force="baseline",
                        em_target=b_idx,
                    )
                    load_runtime_state(self.model, copy.deepcopy(snap.runtime_state))
                    loss_chosen = self._rollout_span(
                        snap, pm_force="normal", em_force="chosen",
                        em_target=b_idx,
                    )
                    reward = loss_baseline - loss_chosen
                    em_reward_sum += reward.mean().item()
                    em_count += 1
                    em_gate_loss_sum += self._update_em_neuromodulators(
                        snap, reward, targets={b_idx}
                    )

        # Restore final real state
        load_runtime_state(self.model, final_runtime_state)

        # Capture neuromodulator grad norms BEFORE optimizer zeros them
        rl_metrics.update(self.collect_neuromod_grad_norms())

        # Single optimizer step for all neuromodulators
        if self.rl_optimizer is not None:
            self._tick_rl_warmup()
            self.rl_optimizer.step()
            self.rl_optimizer.zero_grad()

        # Collect RL-specific metrics
        if pm_count > 0:
            rl_metrics["rl_pm_reward_mean"] = pm_reward_sum / pm_count
            rl_metrics["rl_pm_gate_loss"] = pm_gate_loss_sum / pm_count
        if em_count > 0:
            rl_metrics["rl_em_reward_mean"] = em_reward_sum / em_count
            rl_metrics["rl_em_g_loss"] = em_gate_loss_sum / em_count

        # Neuromodulator output stats — use last snapshot's real inputs
        last_snap = snapshots[-1]
        commit_rates = []
        lambda_vals = []
        g_vals = []
        g_em_vals = []

        with torch.no_grad():
            for b_idx, block in enumerate(self.model.blocks):
                for l_idx, layer in enumerate(block.layers):
                    nm = layer.pm_neuromodulator
                    if nm.rl_enabled and hasattr(nm, "gate_head"):
                        elig_norm = last_snap.pm_elig_norms.get((b_idx, l_idx))
                        pm_usage = last_snap.pm_usages.get((b_idx, l_idx))
                        if elig_norm is not None and pm_usage is not None:
                            _, lv, gv, _, pc = nm(
                                elig_norm, pm_usage, last_snap.span_surprise_mean,
                            )
                            if pc is not None:
                                commit_rates.append((pc > 0.5).float().mean().item())
                                lambda_vals.append(lv.mean().item())
                                g_vals.append(gv.mean().item())

                nm_em = block.em_neuromodulator
                if nm_em.rl_enabled and hasattr(nm_em, "g_head"):
                    em_usage = last_snap.em_usages.get(b_idx)
                    em_nov = last_snap.em_novelties.get(b_idx)
                    if em_usage is not None and em_nov is not None:
                        _, gem, tau_out, ww_out = nm_em(
                            last_snap.span_surprise_mean, em_usage, em_nov,
                        )
                        g_em_vals.append(gem.mean().item())

        if commit_rates:
            rl_metrics["rl_pm_commit_rate"] = sum(commit_rates) / len(commit_rates)
        if g_em_vals:
            rl_metrics["rl_em_g_mean"] = sum(g_em_vals) / len(g_em_vals)
        if lambda_vals:
            rl_metrics["rl_pm_lambda_mean"] = sum(lambda_vals) / len(lambda_vals)
        if g_vals:
            rl_metrics["rl_pm_g_mean"] = sum(g_vals) / len(g_vals)

        return rl_metrics
