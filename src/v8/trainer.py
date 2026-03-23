"""V8 Trainer — joint LM training + RL for neuromodulator.

Each step:
1. Full scan over T tokens (once, shared)
2. Segmented memory processing: neuromod acts → graph dynamics (8 segments)
3. Per-segment CE loss → rewards collected
4. LM backward on logits (every chunk)
5. Every rl_collect_chunks chunks: compute returns across full window
   with learned value baseline, then neuromod backward
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .config import V8Config
from .model import V8Model


class V8Trainer:
    """Training loop for v8 model."""

    def __init__(
        self,
        model: V8Model,
        lm_optimizer: torch.optim.Optimizer,
        neuromod_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        dataloader,
        config: V8Config,
        device: torch.device,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        collector=None,
        use_memory: bool = True,
        neuromod_scheduler=None,
    ):
        self.model = model
        self.lm_optimizer = lm_optimizer
        self.neuromod_optimizer = neuromod_optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.collector = collector
        self.global_step = 0
        self.use_memory = use_memory
        self.neuromod_scheduler = neuromod_scheduler

        self._states_initialized = False
        self.use_amp = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # RL collection buffer: accumulate across chunks before updating neuromod
        self.rl_collect_chunks = config.rl_collect_chunks
        self._rl_buffer: list[dict] = []

    def train_chunk(self, batch) -> dict:
        """Process one T-token chunk.

        Returns dict with training metrics.
        """
        self.model.train()
        BS = batch.input_ids.shape[0]
        T = batch.input_ids.shape[1]

        if not self._states_initialized:
            self.model.initialize_states(BS, self.device)
            self._states_initialized = True

        input_ids = batch.input_ids.to(self.device, non_blocking=True)
        target_ids = batch.target_ids.to(self.device, non_blocking=True)
        prev_token = batch.prev_token.to(self.device, non_blocking=True)

        eot_id = self.config.eot_id
        reset_mask = (prev_token == eot_id)

        t_start = time.time()

        amp_ctx = torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

        # ==========================================
        # Forward: single scan + segmented memory
        # ==========================================
        with amp_ctx:
            result = self.model.forward_chunk(
                input_ids, target_ids=target_ids,
                reset_mask=reset_mask,
                use_memory=self.use_memory,
            )

        logits = result["logits"]
        aux_loss = result["aux_loss"]
        rl_data = result["rl_data"]

        # ==========================================
        # LM loss + backward
        # ==========================================
        is_eot = (input_ids == eot_id)
        loss_mask = ~is_eot

        ce_per_token = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target_ids.reshape(-1),
            reduction='none',
        ).reshape(BS, T)

        valid_mask = loss_mask.float()
        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = (ce_per_token * valid_mask).sum() / valid_count

        lm_loss = ce_loss + aux_loss

        self.lm_optimizer.zero_grad(set_to_none=True)
        lm_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.lm.parameters(), self.max_grad_norm
        ).item()
        self.lm_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # ==========================================
        # Neuromodulator: accumulate RL data, update every N chunks
        # ==========================================
        rl_metrics = {}
        if rl_data is not None:
            self._rl_buffer.append(rl_data)

            # Per-chunk logging (always available)
            rl_metrics = {
                "rl_loss": rl_data["loss"],
                "rl_seg_loss_first": rl_data["seg_losses"][:, 0].mean().item(),
                "rl_seg_loss_last": rl_data["seg_losses"][:, -1].mean().item(),
            }

            # Update neuromod when buffer is full
            if len(self._rl_buffer) >= self.rl_collect_chunks:
                self.neuromod_optimizer.zero_grad(set_to_none=True)

                # Compute returns + advantages across all collected chunks
                combined = self.model.compute_rl_advantages(self._rl_buffer)

                # Policy gradient + value function update
                rl_losses = self.model.replay_for_neuromod_grads(
                    combined,
                    amp_enabled=self.use_amp,
                    amp_dtype=self.amp_dtype,
                )

                nm_grad_norm = nn.utils.clip_grad_norm_(
                    self.model.neuromod.parameters(), self.max_grad_norm
                ).item()
                self.neuromod_optimizer.step()

                adv = combined["advantages"]
                returns = combined["returns"]

                # Value function explained variance: 1 - Var(returns-V) / Var(returns)
                # Higher = better baseline (1.0 = perfect, 0.0 = useless)
                with torch.no_grad():
                    global_obs_flat = combined["global_obs"].reshape(
                        -1, combined["global_obs"].shape[-1])
                    v_dtype = next(self.model.neuromod.value_net.parameters()).dtype
                    v_pred = self.model.neuromod.get_value(
                        global_obs_flat.to(v_dtype))
                    n_seg = combined["global_obs"].shape[0]
                    v_pred = v_pred.reshape(n_seg, -1).T  # [BS, n_seg]
                    var_returns = returns.var()
                    explained_var = (
                        1.0 - (returns - v_pred).var() / var_returns.clamp(min=1e-8)
                    ).item()

                rl_metrics.update({
                    "rl_policy_loss": rl_losses["policy_loss"],
                    "rl_value_loss": rl_losses["value_loss"],
                    "rl_entropy": rl_losses["entropy"],
                    "rl_adv_mean": adv.mean().item(),
                    "rl_adv_std": adv.std().item(),
                    "rl_returns_mean": returns.mean().item(),
                    "rl_returns_std": returns.std().item(),
                    "rl_explained_var": round(explained_var, 4),
                    "rl_nm_grad_norm": nm_grad_norm,
                })

                # Clear buffer
                self._rl_buffer = []

            # Step neuromod scheduler every chunk (not just on RL updates)
            # so the LR schedule tracks global training progress
            if self.neuromod_scheduler is not None:
                self.neuromod_scheduler.step()

        # ==========================================
        # TBPTT boundary
        # ==========================================
        self.model.detach_states()

        elapsed = time.time() - t_start
        tok_per_s = BS * T / elapsed

        loss_val = ce_loss.item()
        ppl = min(torch.exp(ce_loss.detach()).item(), 1e6)
        lr = self.lm_optimizer.param_groups[0]["lr"]

        nm_lr = (self.neuromod_optimizer.param_groups[0]["lr"]
                 if self.use_memory else 0.0)
        metrics = {
            "loss": loss_val,
            "ppl": ppl,
            "aux_loss": aux_loss.item(),
            "lr": lr,
            "nm_lr": nm_lr,
            "tok_s": tok_per_s,
            "grad_norm": grad_norm,
            "elapsed": elapsed,
            **rl_metrics,
        }

        self.global_step += 1
        return metrics

    def train_epoch(self, max_steps: int, step_callback=None) -> list[dict]:
        all_metrics = []
        pbar = tqdm(total=max_steps, desc="v8 train", unit="step")

        for step_idx, batch in enumerate(self.dataloader):
            if step_idx >= max_steps:
                break

            metrics = self.train_chunk(batch)
            all_metrics.append(metrics)

            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                ppl=f"{metrics['ppl']:.1f}",
                toks=f"{metrics['tok_s']/1e3:.1f}K",
            )
            pbar.update(1)

            if step_callback is not None:
                step_callback(metrics)

        pbar.close()
        return all_metrics
