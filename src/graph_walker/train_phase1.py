"""Phase 1 training — teacher-forced TBPTT with whole-block compile.

Each TBPTT block is one call to `memory.block_forward(...)` (compiled),
which runs `mod_period` sequential steps in a single autograd graph.
Inductor fuses across step boundaries and produces one forward + one
backward graph per block — ~3.7× faster than the older per-step compile.

The graph core runs token-by-token *inside* the compiled block (genuine
recurrence), but Python only invokes the block once per mod_period tokens
instead of once per token. The expensive model-space readout stays off
that hot path: motor_state is buffered over the block, then one batched
lexical forward pass produces the multi-horizon CE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

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

    # Runtime clock-alignment checks (mirror config.__post_init__).
    tbptt = tbptt_block if tbptt_block is not None else cfg.tbptt_block
    if tbptt != cfg.mod_period:
        raise ValueError(
            f"tbptt_block ({tbptt}) must equal mod_period ({cfg.mod_period}). "
            "Each flush must close exactly one plasticity window."
        )
    if T_seq % cfg.mod_period != 0:
        raise ValueError(
            f"tokens shape[1] = {T_seq} must be a multiple of mod_period "
            f"({cfg.mod_period}) so the final partial window isn't dropped."
        )

    # Compile block_forward on first use.
    if (
        device.type == "cuda"
        and cfg.compile_on_train
        and lm.memory._compiled_block is None
    ):
        lm.memory.compile_block(mode=getattr(cfg, "compile_mode", "default"))

    opt.zero_grad(set_to_none=True)
    lm.set_training_step(training_step)
    lm.memory.begin_segment(B, device)
    # Block caches must be populated before any block_forward call (we no
    # longer go through step_core, which used to ensure them lazily).
    lm.memory._ensure_block_caches(lm.memory.tied_token_emb.weight)

    K_h = cfg.K_horizons

    # Schedule tensors are constants for the segment.
    from src.graph_walker.routing import gumbel_schedule
    tau = torch.tensor(
        gumbel_schedule(
            training_step, cfg.gumbel_tau_start, cfg.gumbel_tau_end,
            cfg.gumbel_anneal_steps,
        ),
        device=device, dtype=torch.float32,
    )
    epsilon = torch.tensor(
        gumbel_schedule(
            training_step, cfg.epsilon_start, cfg.epsilon_end,
            cfg.epsilon_anneal_steps,
        ),
        device=device, dtype=torch.float32,
    )

    horizon_weights = torch.full((K_h,), 0.2, device=device, dtype=torch.float32)
    horizon_weights[0] = 1.0

    # Cross-block accumulators.
    total_balance_tensor = torch.zeros((), device=device, dtype=torch.float32)
    per_horizon_accum = torch.zeros(K_h, device=device, dtype=torch.float32)
    per_horizon_counts = torch.zeros(K_h, device=device, dtype=torch.float32)
    n_blocks = 0

    n_blocks_total = T_seq // tbptt
    block_fn = (
        lm.memory._compiled_block
        if lm.memory._compiled_block is not None
        else lm.memory.block_forward
    )

    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    with ctx:
        for blk in range(n_blocks_total):
            block_start_t = blk * tbptt
            tokens_block = tokens[:, block_start_t:block_start_t + tbptt]

            # detach_state invalidates block-level caches at the previous
            # iteration's end so they can rebuild with fresh grad_fn refs.
            # Repopulate them before this block's compiled call.
            lm.memory._ensure_block_caches(lm.memory.tied_token_emb.weight)

            # Active E_bias = frozen persistent base + grad-carrying neuromod
            # delta (rebuilt fresh each block via _begin_plastic_window from
            # the previous detach_state).
            e_bias = lm.memory._active_e_bias()

            # Anchor only fires at the start of a plasticity window. With
            # tbptt == mod_period, every block IS a fresh window — so t=0 of
            # every block is an anchor step.
            out = block_fn(
                lm.memory.s, lm.memory.walker_pos, lm.memory.walker_state,
                lm.memory.prev_motor, e_bias,
                tokens_block, tau, epsilon, True,
            )

            # Non-grad accumulator add-into stable buffers — safe before
            # backward because they don't participate in the autograd graph.
            with torch.no_grad():
                lm.memory.co_visit_flat.add_(out.co_visit_total)
                if lm.memory.visit_count is not None:
                    lm.memory.visit_count.add_(out.visit_count_total)
            # Forward-state writebacks happen AFTER backward — copy_ bumps
            # the tensor version, and any saved-for-backward refs to these
            # tensors must complete before the bump.
            # window_len bookkeeping — block_forward processed mod_period
            # tokens, which closes exactly one plasticity window.
            lm.memory.window_len = tbptt
            lm.memory.tick_counter += tbptt

            # Build targets + validity mask for this block.
            with torch.autocast(device_type=device.type, enabled=False):
                T_block = tokens_block.shape[1]
                i_idx = torch.arange(T_block, device=device)
                k_idx = torch.arange(1, K_h + 1, device=device)
                t_idx = block_start_t + i_idx.unsqueeze(1) + k_idx.unsqueeze(0)
                valid_tk = t_idx < T_seq                                # [T_block, K_h]
                t_idx_clamped = t_idx.clamp(max=T_seq - 1)
                targets = tokens.index_select(1, t_idx_clamped.reshape(-1)).reshape(
                    B, T_block, K_h,
                )
                valid_btk = valid_tk.unsqueeze(0).expand(B, -1, -1)

                # Factorized CE — never materialises [B, T_block, K_h, V].
                ce_masked = lm.memory.readout_ce_block(
                    out.motor_states_bt, targets, valid_btk,
                )                                                       # [B, T_block, K_h]
                block_horizon_sum = ce_masked.sum(dim=(0, 1))            # [K_h]
                block_horizon_count = valid_tk.float().sum(dim=0) * B    # [K_h]

            has_counts = block_horizon_count > 0
            per_h = block_horizon_sum / block_horizon_count.clamp(min=1)
            ce = (
                per_h * horizon_weights * has_counts.float()
            ).sum() / horizon_weights[has_counts].sum().clamp(min=1)
            balance_term = cfg.lambda_balance * out.load_balance_loss
            loss = ce + balance_term

            # Stream surprise EMA + fire plasticity for the just-closed window.
            lm.memory.accumulate_block_ce(ce_masked.detach(), valid_tk.detach())
            lm.memory._maybe_finalize_surprise_and_plasticity()

            loss.backward()

            # Now safe to copy block_forward outputs into stable state buffers.
            # NOTE: out.* tensors live in cudagraph pool memory under
            # reduce-overhead and may be overwritten on the next block_fn
            # call; .copy_() takes ownership into stable storage.
            with torch.no_grad():
                lm.memory.s.copy_(out.s_new)
                lm.memory.walker_pos.copy_(out.walker_pos_new)
                lm.memory.walker_state.copy_(out.walker_state_new)
                lm.memory.prev_motor.copy_(out.prev_motor_new)

            # Cross-block accumulators (GPU-resident, single sync at end).
            total_balance_tensor = total_balance_tensor + balance_term.detach()
            per_horizon_accum.add_(block_horizon_sum.detach())
            per_horizon_counts.add_(block_horizon_count)
            n_blocks += 1

            # TBPTT detach.
            lm.memory.detach_state()

            # Signal end of this iteration's cudagraph window. Frees the
            # output buffers from the just-completed block_fn replay so
            # they can be reused on the next iteration's replay. Must come
            # AFTER the writebacks above (we need the values) but before
            # the next block_fn call. No-op under non-reduce-overhead modes.
            torch.compiler.cudagraph_mark_step_begin()

    grad_norm = torch.nn.utils.clip_grad_norm_(lm.parameters(), grad_clip)
    opt.step()

    # Visit entropy.
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
    total_balance = total_balance_tensor.item()
    load_balance_loss = total_balance / max(n_blocks, 1)

    return StepStats(
        loss=ce_loss + load_balance_loss,
        ce_loss=ce_loss,
        load_balance_loss=total_balance / max(n_blocks, 1),
        per_horizon_loss=per_horizon_mean.cpu().tolist(),
        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        visit_entropy=visit_entropy_frac,
    )
