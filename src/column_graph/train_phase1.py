"""Phase 1 training — block-compiled TBPTT with streaming multi-horizon CE.

Hot path structure:
  - segment = T_seq tokens, processed in chunks of `block_len` tokens
  - block_len = mod_period (default 16): one plasticity event per block end
  - each block: one call to `memory.run_block(tokens_block)` returning
    motor_stack [B, block_len, D_s]. This call is the torch.compile target.
  - readout + multi-horizon CE loss computed on motor_stack in chunks of
    `t_chunk` tokens (streaming — never materializes [B, T, K_h, V]).
  - plasticity runs every block end, after the compile region.
  - TBPTT: backward every `tbptt_block` tokens (configurable; defaults to
    `tbptt_block = mod_period` so each compiled block backs up in isolation).

Benefit: Inductor sees the full `block_len`-tick loop as one graph, fuses
across ticks, and generates a compiled forward + backward automatically.
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
    per_horizon_loss: list[float]
    grad_norm: float


def streaming_multi_horizon_ce(
    motor_stack: torch.Tensor,            # [B, T, D_s]
    tokens: torch.Tensor,                  # [B, T_total]
    t_offset: int,                          # absolute offset of motor_stack[:, 0] in tokens
    horizon_emb: torch.Tensor,              # [K_h, D_s]
    pred_head,                              # nn.Module applied to motor before unembed
    unembed_weight: torch.Tensor,           # [V, D_s]
    horizon_weights: torch.Tensor | None = None,
    t_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multi-horizon CE without materializing [B, T, K, V].

    For each horizon k, stream over T in chunks of `t_chunk`:
      logits_chunk = pred_head(motor_chunk + horizon_emb[k]) @ unembed.T
      ce_k += F.cross_entropy(logits_chunk, targets_chunk)

    Returns (total_loss, per_horizon_losses [K_h]).
    """
    B, T_block, D_s = motor_stack.shape
    K_h, _ = horizon_emb.shape
    V = unembed_weight.shape[0]
    device = motor_stack.device
    T_total = tokens.shape[1]

    # Apply pred_head once to the full motor_stack (cheap residual).
    motor_flat = motor_stack.reshape(B * T_block, D_s)
    motor_flat = pred_head(motor_flat).reshape(B, T_block, D_s)

    per_horizon = torch.zeros(K_h, device=device, dtype=torch.float32)
    counts = torch.zeros(K_h, device=device, dtype=torch.float32)

    with torch.autocast(device_type=device.type, enabled=False):
        for k in range(1, K_h + 1):
            # Targets: tokens[:, t + k] for t in [t_offset, t_offset + T_block),
            # valid only when t + k < T_total.
            valid_end = min(T_block, T_total - t_offset - k)
            if valid_end <= 0:
                continue

            # Stream T_block in chunks of t_chunk.
            for t_start in range(0, valid_end, t_chunk):
                t_end = min(t_start + t_chunk, valid_end)
                chunk_len = t_end - t_start

                motor_chunk = motor_flat[:, t_start:t_end].float()  # [B, chunk, D_s]
                shifted = motor_chunk + horizon_emb[k - 1].to(motor_chunk.dtype)
                # [B, chunk, V]
                logits_chunk = torch.matmul(shifted, unembed_weight.t().float())
                tgt_chunk = tokens[:, t_offset + t_start + k: t_offset + t_end + k]
                ce = F.cross_entropy(
                    logits_chunk.reshape(-1, V),
                    tgt_chunk.reshape(-1),
                    reduction="sum",
                )
                per_horizon[k - 1] += ce
                counts[k - 1] += B * chunk_len

    # Normalize by count per horizon (mean CE).
    per_horizon_mean = per_horizon / counts.clamp(min=1)

    if horizon_weights is None:
        weights = torch.full((K_h,), 0.2, device=device, dtype=torch.float32)
        weights[0] = 1.0
    else:
        weights = horizon_weights.to(device).float()

    # Total: horizon-weighted mean over horizons that have at least one target
    has_counts = counts > 0
    total = (per_horizon_mean * weights * has_counts.float()).sum() / weights[has_counts].sum().clamp(min=1)
    return total, per_horizon_mean


def phase1_step(
    lm: StandaloneLM,
    opt: torch.optim.Optimizer,
    tokens: torch.Tensor,           # [B, T_seq]
    *,
    tbptt_block: int | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    grad_clip: float = 1.0,
    t_chunk: int = 64,
) -> StepStats:
    """One training step: segment of T_seq tokens with block-compiled TBPTT."""
    cfg = lm.cfg
    B, T_seq = tokens.shape
    device = tokens.device
    opt.zero_grad(set_to_none=True)

    lm.memory.begin_segment(B, device)

    # Block size = mod_period so plasticity fires exactly once per block.
    block_len = cfg.mod_period
    tbptt = tbptt_block if tbptt_block is not None else block_len

    # Surprise/ring-buffer bookkeeping is done per-block AFTER the compiled
    # propagation returns. We need the motor stack to feed into the
    # ring buffer + surprise update.
    K_h = cfg.K_horizons
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    per_horizon_total = torch.zeros(K_h, device=device, dtype=torch.float32)
    per_horizon_count = 0
    block_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    ticks_since_backward = 0

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    with ctx:
        t = 0
        while t < T_seq:
            chunk_end = min(t + block_len, T_seq)
            block = tokens[:, t:chunk_end]
            actual_block_len = block.shape[1]

            # If this block is shorter than block_len (tail), pad with zeros
            # and mask — simpler: for v1, skip the tail block if shorter.
            # Actually, the streaming CE masks by valid target lookup, so
            # running a short block is fine. The compiled block might
            # recompile for each unique shape — to avoid that, pad to
            # block_len and ignore the tail in CE target range.
            if actual_block_len < block_len:
                pad_len = block_len - actual_block_len
                pad = torch.zeros(B, pad_len, dtype=block.dtype, device=device)
                block = torch.cat([block, pad], dim=1)

            motor_stack, last_m_out, last_incoming, last_w_out = lm.memory.run_block(block)

            # Streaming multi-horizon CE over this block's motor stack.
            # Targets live in tokens; valid range masked inside the helper.
            block_loss, per_h = streaming_multi_horizon_ce(
                motor_stack[:, :actual_block_len],
                tokens,
                t_offset=t,
                horizon_emb=lm.memory.readout.horizon_emb,
                pred_head=lm.memory.readout.pred_head,
                unembed_weight=lm.token_emb.weight,
                t_chunk=t_chunk,
            )
            block_loss_sum = block_loss_sum + block_loss
            per_horizon_total += per_h.detach()
            per_horizon_count += 1

            # Ring buffer + surprise bookkeeping per tick in this block.
            # Driven outside the compiled region (graph-broken).
            lm.memory._ringbuf_block_bookkeeping(
                motor_stack[:, :actual_block_len],
                tokens[:, t:t + actual_block_len],
            )

            # Plasticity step at end of block (fires every mod_period ticks).
            lm.memory._plasticity_step(last_m_out, last_incoming, last_w_out)

            ticks_since_backward += actual_block_len
            t += actual_block_len

            # TBPTT boundary: flush accumulated loss + detach state.
            if ticks_since_backward >= tbptt or t >= T_seq:
                block_loss_sum.backward()
                total_loss = total_loss + block_loss_sum.detach()
                block_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                lm.memory.detach_state()
                ticks_since_backward = 0

    grad_norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), grad_clip)
    opt.step()

    per_horizon_mean = (per_horizon_total / max(per_horizon_count, 1)).tolist()
    return StepStats(
        loss=total_loss.item() / max(per_horizon_count, 1),
        per_horizon_loss=per_horizon_mean,
        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
    )
