"""Phase 1 training — teacher-forced TBPTT with multi-horizon CE.

Per step:
  - Segment of shape [B, T_seq]. Begin a fresh segment on the memory.
  - Stream tokens one at a time. At each tick, produce [B, K_h, V] logits.
  - Multi-horizon CE: logits at tick t scored against tokens at t+1..t+K_h.
  - Accumulate loss; every tbptt_block ticks, backward + detach state + continue.
  - One optimiser step per segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.column_graph.config import ColumnGraphConfig
from src.column_graph.standalone import StandaloneLM


@dataclass
class StepStats:
    loss: float
    per_horizon_loss: list[float]        # len = K_horizons
    grad_norm: float


def multi_horizon_loss(
    logits: torch.Tensor,          # [T_tick, B, K_h, V]
    tokens: torch.Tensor,          # [B, T_tick + K_h]  — padded right so that
                                   # tokens[b, t+k] is valid for t < T_tick, k in 1..K_h
    horizon_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Sum of per-horizon CE across all (t, k) pairs where the target exists.

    tokens shape must be at least [B, T_tick]; for horizons we need future
    tokens up to T_tick + K_h - 1. Callers pad with PAD_ID and mask those.
    For the Phase-1 streaming caller, the target for logits_at_tick_t,
    horizon_k is tokens[b, t + k]. We only supervise where t + k < T_total.
    """
    T_tick, B, K_h, V = logits.shape
    T_total = tokens.shape[1]

    per_horizon = torch.zeros(K_h, device=logits.device, dtype=torch.float32)
    counts = torch.zeros(K_h, device=logits.device, dtype=torch.float32)

    with torch.autocast(device_type=logits.device.type, enabled=False):
        for k in range(1, K_h + 1):
            # For each tick t, supervise if t + k < T_total.
            t_start, t_end = 0, T_tick  # all ticks
            valid_upper = T_total - k     # t must satisfy t + k < T_total → t < T_total - k
            valid_t_end = min(t_end, valid_upper)
            if valid_t_end <= t_start:
                continue
            # gather logits: [B, K_h, V] at each t → [valid_t, B, V] for horizon k-1
            logits_k = logits[t_start:valid_t_end, :, k - 1, :].float()  # [t, B, V]
            tgt_k = tokens[:, t_start + k: valid_t_end + k]               # [B, t]
            ce = F.cross_entropy(
                logits_k.reshape(-1, V),
                tgt_k.t().reshape(-1),
                reduction="mean",
            )
            per_horizon[k - 1] = ce
            counts[k - 1] = 1

    # Total loss: horizon-weighted mean. Default: primary k=1 weight 1.0,
    # k>1 weight 0.2.
    if horizon_weights is None:
        weights = torch.full((K_h,), 0.2, device=logits.device, dtype=torch.float32)
        weights[0] = 1.0
    else:
        weights = horizon_weights.to(per_horizon.device).float()
    total = (per_horizon * weights * counts).sum() / weights[counts > 0].sum().clamp(min=1)
    return total, per_horizon.tolist()


def phase1_step(
    lm: StandaloneLM,
    opt: torch.optim.Optimizer,
    tokens: torch.Tensor,           # [B, T_seq]
    *,
    tbptt_block: int,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
) -> StepStats:
    """One training step: segment of T_seq tokens with TBPTT backward."""
    cfg = lm.cfg
    B, T_seq = tokens.shape
    device = tokens.device
    opt.zero_grad(set_to_none=True)

    lm.memory.begin_segment(B, device)
    K_h = cfg.K_horizons

    # Track running loss across blocks
    total_loss = 0.0
    per_horizon_total = [0.0] * K_h
    num_blocks = 0

    block_logits: list[torch.Tensor] = []
    block_start = 0

    def flush_block(end: int):
        """Backward-pass the accumulated block, update totals, detach state."""
        nonlocal total_loss, num_blocks
        if not block_logits:
            return
        stacked = torch.stack(block_logits, dim=0)   # [T_tick, B, K_h, V]
        # For horizon lookup we pass full token tensor; the loss masks OOB targets.
        loss, per_h = multi_horizon_loss(stacked, tokens)
        loss.backward()
        total_loss += loss.item()
        for i in range(K_h):
            per_horizon_total[i] += per_h[i]
        num_blocks += 1
        block_logits.clear()
        lm.memory.detach_state()

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with ctx:
        for t in range(T_seq):
            r = lm.memory.step(tokens[:, t])
            block_logits.append(r.logits)
            # Flush at block boundary (but not at the very end — we flush below)
            if (t + 1) % tbptt_block == 0 and (t + 1) < T_seq:
                flush_block(t + 1)
                block_start = t + 1
        # Final block
        flush_block(T_seq)

    # Gradient clip, optimiser step
    grad_norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), grad_clip)
    opt.step()

    return StepStats(
        loss=total_loss / max(num_blocks, 1),
        per_horizon_loss=[p / max(num_blocks, 1) for p in per_horizon_total],
        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
    )
