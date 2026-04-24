"""Phase 1 training — teacher-forced TBPTT with blockwise lexical readout.

The graph core runs token-by-token because it is genuinely recurrent and
sparse. The expensive model-space readout is kept off that hot path: we buffer
motor_state over the TBPTT block, then do one batched lexical forward pass for
exact multi-horizon CE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph_walker.standalone import StandaloneLM


@dataclass
class StepStats:
    loss: float
    ce_loss: float
    load_balance_loss: float
    per_horizon_loss: list[float]
    grad_norm: float
    visit_entropy: float      # log(N) - H(visit_freq); 0 means uniform


def phase1_step(
    lm: StandaloneLM,
    opt: torch.optim.Optimizer,
    tokens: torch.Tensor,             # [B, T_seq]
    *,
    tbptt_block: int | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
    training_step: int = 0,
) -> StepStats:
    cfg = lm.cfg
    B, T_seq = tokens.shape
    device = tokens.device
    if (
        device.type == "cuda"
        and cfg.compile_on_train
        and lm.memory._compiled_step is None
    ):
        lm.memory.compile_step()
    opt.zero_grad(set_to_none=True)
    lm.set_training_step(training_step)

    lm.memory.begin_segment(B, device)
    tbptt = tbptt_block if tbptt_block is not None else cfg.tbptt_block

    K_h = cfg.K_horizons
    # Kept GPU-resident and accumulated in place to avoid per-flush .item()
    # syncs (each such sync stalls the CPU until pending GPU work completes).
    total_balance_tensor = torch.zeros((), device=device, dtype=torch.float32)
    per_horizon_accum = torch.zeros(K_h, device=device, dtype=torch.float32)
    per_horizon_counts = torch.zeros(K_h, device=device, dtype=torch.float32)
    n_segments = 0
    block_balance_sum = torch.zeros((), device=device, dtype=torch.float32)
    block_horizon_sum = torch.zeros(K_h, device=device, dtype=torch.float32)
    block_horizon_count = torch.zeros(K_h, device=device, dtype=torch.float32)
    ticks_since_backward = 0
    block_start_t = 0
    motor_state_block: list[torch.Tensor] = []
    horizon_weights = torch.full((K_h,), 0.2, device=device, dtype=torch.float32)
    horizon_weights[0] = 1.0

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def flush():
        """Backprop accumulated block loss + stream CE into surprise + fire
        plasticity if the block's last token closed a plasticity window +
        detach state."""
        nonlocal total_balance_tensor, n_segments
        nonlocal block_balance_sum, ticks_since_backward, block_start_t
        nonlocal block_horizon_sum, block_horizon_count
        nonlocal motor_state_block
        if ticks_since_backward == 0:
            return
        motor_state_bt = torch.stack(motor_state_block, dim=1)        # [B, T_block, D_s]
        with ctx:
            block_logits = lm.memory.readout_from_state_block(motor_state_bt)
        with torch.autocast(device_type=device.type, enabled=False):
            block_logits = block_logits.float()
            # Vectorized multi-horizon CE: collapse the T_block × K_h python
            # double-loop into a single F.cross_entropy call. The old loop
            # was generating 2-3 autograd `select` nodes per (i, k) pair,
            # and each `SelectBackward` allocated a full-shape zeros tensor
            # (~128MB) for gradient scatter. With T_block=16 and K_h=8 per
            # block that's ~380 select nodes per flush, dominating backward.
            T_block = ticks_since_backward
            i_idx = torch.arange(T_block, device=device)
            k_idx = torch.arange(1, K_h + 1, device=device)
            # t_idx[i, k-1] = block_start_t + i + k
            t_idx = block_start_t + i_idx.unsqueeze(1) + k_idx.unsqueeze(0)
            valid = t_idx < T_seq                                 # [T_block, K_h]
            t_idx_clamped = t_idx.clamp(max=T_seq - 1)            # safe gather
            # targets[b, i, k-1] = tokens[b, t_idx[i, k-1]]
            targets = tokens.index_select(1, t_idx_clamped.reshape(-1)).reshape(
                B, T_block, K_h,
            )
            # Flat CE: reduction='none' so we can mask invalid horizons.
            logits_flat = block_logits.reshape(-1, block_logits.shape[-1])  # [B*T_block*K_h, V]
            targets_flat = targets.reshape(-1)
            ce_flat = F.cross_entropy(
                logits_flat, targets_flat, reduction="none",
            )                                                     # [B*T_block*K_h]
            valid_flat = valid.unsqueeze(0).expand(B, -1, -1).reshape(-1).float()
            ce_masked = (ce_flat * valid_flat).reshape(B, T_block, K_h)
            block_horizon_sum = block_horizon_sum + ce_masked.sum(dim=(0, 1))
            block_horizon_count = (
                block_horizon_count + valid.float().sum(dim=0) * B
            )
        has_counts = block_horizon_count > 0
        per_h = block_horizon_sum / block_horizon_count.clamp(min=1)
        ce = (
            per_h * horizon_weights * has_counts.float()
        ).sum() / horizon_weights[has_counts].sum().clamp(min=1)
        loss = ce + block_balance_sum

        # Stream the flush's per-(position, horizon) CE into surprise_ema,
        # then fire plasticity if this block closed a window. Done BEFORE
        # backward so plasticity sees this window's surprise and any new
        # _active_delta_nm it produces is ready for the next block's step.
        # The detached CE tensor keeps the autograd graph decoupled —
        # surprise is a training diagnostic, not a gradient carrier.
        lm.memory.accumulate_block_ce(ce_masked.detach(), valid.detach())
        lm.memory._maybe_finalize_surprise_and_plasticity()

        loss.backward()
        # Accumulate on GPU; sync once at end-of-step (not per flush).
        total_balance_tensor = total_balance_tensor + block_balance_sum.detach()
        n_segments += 1
        per_horizon_accum.add_(block_horizon_sum.detach())
        per_horizon_counts.add_(block_horizon_count)
        block_balance_sum = torch.zeros((), device=device, dtype=torch.float32)
        block_horizon_sum = torch.zeros(K_h, device=device, dtype=torch.float32)
        block_horizon_count = torch.zeros(K_h, device=device, dtype=torch.float32)
        ticks_since_backward = 0
        block_start_t = 0
        motor_state_block = []
        lm.memory.detach_state()

    with ctx:
        for t in range(T_seq):
            if ticks_since_backward == 0:
                block_start_t = t
            r = lm.memory.step_core(tokens[:, t])
            # Plasticity + surprise are now fired inside flush (after CE is
            # computed), not per-token. With tbptt_block == mod_period (the
            # enforced config alignment), each flush closes exactly one
            # plasticity window.
            motor_state_block.append(r.motor_state)
            if cfg.lambda_balance > 0:
                # Switch-style load-balance aux loss (carries gradient to
                # routing params through soft_probs). r.load_balance_loss is
                # N · Σ P·f; we scale by lambda_balance.
                block_balance_sum = block_balance_sum + (
                    cfg.lambda_balance * r.load_balance_loss
                )
            ticks_since_backward += 1
            if ticks_since_backward >= tbptt and (t + 1) < T_seq:
                flush()
        flush()

    grad_norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), grad_clip)
    opt.step()

    # Visit entropy (higher = more uniform)
    if lm.memory.visit_count is not None:
        vc = lm.memory.visit_count.float()
        p = (vc / vc.sum().clamp(min=1)).clamp(min=1e-12)
        entropy = -(p * p.log()).sum().item()
        entropy_uniform = torch.log(torch.tensor(float(cfg.N))).item()
        visit_entropy_frac = entropy / entropy_uniform
    else:
        visit_entropy_frac = 1.0

    has_total_counts = per_horizon_counts > 0
    per_horizon_mean = per_horizon_accum / per_horizon_counts.clamp(min=1)
    ce_loss = (
        per_horizon_mean * horizon_weights * has_total_counts.float()
    ).sum().item() / horizon_weights[has_total_counts].sum().clamp(min=1).item()
    # Single end-of-step sync for total_balance.
    total_balance = total_balance_tensor.item()
    load_balance_loss = total_balance / max(n_segments, 1)

    return StepStats(
        loss=ce_loss + load_balance_loss,
        ce_loss=ce_loss,
        load_balance_loss=total_balance / max(n_segments, 1),
        per_horizon_loss=per_horizon_mean.cpu().tolist(),
        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        visit_entropy=visit_entropy_frac,
    )
