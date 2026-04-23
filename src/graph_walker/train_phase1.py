"""Phase 1 training — teacher-forced TBPTT with streaming multi-horizon CE.

Drives the GraphWalker forward, computes CE loss, adds load-balance aux
loss from visit frequency, backprops every tbptt_block tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


@dataclass
class StepStats:
    loss: float
    ce_loss: float
    load_balance_loss: float
    per_horizon_loss: list[float]
    grad_norm: float
    visit_entropy: float      # log(N) - H(visit_freq); 0 means uniform


def _multi_horizon_ce_streaming(
    motors: torch.Tensor,                  # [B, T_block, D_s]
    tokens: torch.Tensor,                  # [B, T_total]
    t_offset: int,
    horizon_emb: torch.Tensor,
    pred_head: torch.nn.Module,
    unembed_weight: torch.Tensor,
    t_chunk: int = 64,
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Streaming multi-horizon CE, never materialises [B, T_block, K_h, V]."""
    B, T_block, D_s = motors.shape
    K_h, _ = horizon_emb.shape
    V = unembed_weight.shape[0]
    device = motors.device
    T_total = tokens.shape[1]

    motor_flat = pred_head(motors.reshape(B * T_block, D_s)).reshape(B, T_block, D_s)

    per_horizon = torch.zeros(K_h, device=device, dtype=torch.float32)
    counts = torch.zeros(K_h, device=device, dtype=torch.float32)

    with torch.autocast(device_type=device.type, enabled=False):
        for k in range(1, K_h + 1):
            valid_end = min(T_block, T_total - t_offset - k)
            if valid_end <= 0:
                continue
            for t_start in range(0, valid_end, t_chunk):
                t_end = min(t_start + t_chunk, valid_end)
                motor_chunk = motor_flat[:, t_start:t_end].float()
                shifted = motor_chunk + horizon_emb[k - 1].to(motor_chunk.dtype)
                logits_chunk = torch.matmul(shifted, unembed_weight.t().float())
                tgt_chunk = tokens[:, t_offset + t_start + k: t_offset + t_end + k]
                ce = F.cross_entropy(
                    logits_chunk.reshape(-1, V), tgt_chunk.reshape(-1),
                    reduction="sum",
                )
                per_horizon[k - 1] += ce
                counts[k - 1] += B * (t_end - t_start)

    per_horizon_mean = per_horizon / counts.clamp(min=1)

    if horizon_weights is None:
        weights = torch.full((K_h,), 0.2, device=device, dtype=torch.float32)
        weights[0] = 1.0
    else:
        weights = horizon_weights.to(device).float()

    has_counts = counts > 0
    total = (per_horizon_mean * weights * has_counts.float()).sum() / weights[has_counts].sum().clamp(min=1)
    return total, per_horizon_mean


def phase1_step(
    lm: StandaloneLM,
    opt: torch.optim.Optimizer,
    tokens: torch.Tensor,             # [B, T_seq]
    *,
    tbptt_block: int | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
    t_chunk: int = 64,
    training_step: int = 0,
) -> StepStats:
    cfg = lm.cfg
    B, T_seq = tokens.shape
    device = tokens.device
    opt.zero_grad(set_to_none=True)
    lm.set_training_step(training_step)

    lm.memory.begin_segment(B, device)
    tbptt = tbptt_block if tbptt_block is not None else cfg.tbptt_block

    K_h = cfg.K_horizons
    total_loss = 0.0
    total_ce = 0.0
    total_balance = 0.0
    per_horizon_accum = [0.0] * K_h
    n_segments = 0
    block_ce_sum = torch.zeros((), device=device, dtype=torch.float32)
    block_balance_sum = torch.zeros((), device=device, dtype=torch.float32)

    motors_buf: list[torch.Tensor] = []
    block_start = 0
    ticks_since_backward = 0

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def flush(end: int):
        """Backprop accumulated block loss + detach."""
        nonlocal total_loss, total_ce, total_balance, n_segments
        nonlocal block_ce_sum, block_balance_sum, ticks_since_backward
        if not motors_buf:
            return
        motor_stack = torch.stack(motors_buf, dim=1)                     # [B, T_block, D_s]
        ce, per_h = _multi_horizon_ce_streaming(
            motor_stack, tokens, t_offset=block_start,
            horizon_emb=lm.memory.readout.horizon_emb,
            pred_head=lm.memory.readout.pred_head,
            unembed_weight=lm.token_emb.weight, t_chunk=t_chunk,
        )
        loss = ce + block_balance_sum
        loss.backward()
        total_loss += loss.item()
        total_ce += ce.item()
        total_balance += block_balance_sum.item()
        n_segments += 1
        for i in range(K_h):
            per_horizon_accum[i] += per_h[i].item()
        motors_buf.clear()
        block_ce_sum = torch.zeros((), device=device, dtype=torch.float32)
        block_balance_sum = torch.zeros((), device=device, dtype=torch.float32)
        ticks_since_backward = 0
        lm.memory.detach_state()

    with ctx:
        for t in range(T_seq):
            r = lm.memory.step(tokens[:, t])
            motors_buf.append(r.motor)
            if cfg.lambda_balance > 0 and r.visit_freq_step is not None:
                # Aux load-balance loss: KL(visit_freq, uniform)
                freq = r.visit_freq_step
                N = freq.shape[0]
                kl = (freq * torch.log(freq.clamp(min=1e-9) * N)).sum()
                block_balance_sum = block_balance_sum + cfg.lambda_balance * kl
            ticks_since_backward += 1
            if ticks_since_backward >= tbptt and (t + 1) < T_seq:
                flush(t + 1)
                block_start = t + 1
        flush(T_seq)

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

    return StepStats(
        loss=total_loss / max(n_segments, 1),
        ce_loss=total_ce / max(n_segments, 1),
        load_balance_loss=total_balance / max(n_segments, 1),
        per_horizon_loss=[p / max(n_segments, 1) for p in per_horizon_accum],
        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        visit_entropy=visit_entropy_frac,
    )
