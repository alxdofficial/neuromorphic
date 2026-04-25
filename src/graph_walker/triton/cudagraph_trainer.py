"""Manual CUDA-graph capture for the GraphWalker phase-1 block iteration.

Replaces ``torch.compile(mode="reduce-overhead")`` which fragmented our
graph into 130+ partitions due to "non-gpu ops" (Python-side conditionals
and dynamic-shape preprocessing inside the step). Manual capture keeps
us in control: we pre-allocate every buffer the captured kernels touch,
so addresses are stable across replays.

Captured body (one block of TBPTT):

    1.  ``_ensure_block_caches``           — α, k_all, input keys, horizon logits
    2.  e_bias snapshot                    — frozen ``E_bias_flat.detach()``
    3.  ``block_forward(state, e_bias, tokens_buf, tau_buf, eps_buf)``
    4.  ``readout_ce_block`` + per-horizon CE aggregation
    5.  ``loss = ce + λ · load_balance_loss``
    6.  ``loss.backward()``                — autograd graph captured here
    7.  output writeback:
            * ``ce_loss_buf``, ``balance_loss_buf``, ``block_horizon_*`` ← detached
            * ``ce_masked_buf`` ← ``ce_masked.detach()`` (for surprise EMA)
            * ``memory.s`` ← ``out.s_new``                — state evolution
            * ``memory.walker_pos`` / ``walker_state`` / ``prev_motor`` ← out.*
            * ``memory.co_visit_flat`` += ``out.co_visit_total`` (in-place)
            * ``memory.visit_count`` += ``out.visit_count_total``  (in-place)

Outside the captured graph (per training step):
    - ``opt.zero_grad`` once at top of step
    - For each block:
        * copy new tokens / tau / eps / targets into static buffers
        * ``replay()``
        * stream ``ce_masked`` into ``surprise_ema`` (Python loop, fp32)
        * fire plasticity if window full (dynamic shapes — ``torch.nonzero``)
    - ``opt.step()`` at end

Constraints we accept for capture compatibility:
    - ``use_neuromod=False`` only. Neuromod's grad-carrying ``_active_delta_nm``
      is rebuilt in-place each window with a fresh memory address; the
      captured graph reads from a stale pointer otherwise. Neuromod re-
      enablement requires either (a) preallocating a stable
      ``_active_delta_nm_buf`` that ``_begin_plastic_window`` writes into,
      or (b) routing the neuromod gradient outside the captured path.
      Tracked as future work.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.graph_walker.standalone import StandaloneLM


@dataclass
class CapturedBlockStats:
    """Snapshot of one captured-block iteration's outputs (read after replay)."""
    ce_loss: torch.Tensor              # () fp32
    balance_loss: torch.Tensor         # () fp32
    block_horizon_sum: torch.Tensor    # [K_h] fp32
    block_horizon_count: torch.Tensor  # [K_h] fp32
    ce_masked: torch.Tensor            # [B, T_block, K_h] fp32 — detached


class CapturedBlockTrainer:
    """Captures one block_forward + loss + backward + state writeback as a CUDA graph.

    Use:
        trainer = CapturedBlockTrainer(lm, B=4, T_block=64, K_h=4, lambda_balance=0.01)
        trainer.warmup_and_capture()
        for step in range(N):
            opt.zero_grad(set_to_none=False)
            for block in range(n_blocks):
                trainer.replay(tokens_block, tau, eps, targets, valid_btk, valid_tk,
                               horizon_weights)
                # do plasticity, surprise EMA outside graph
            opt.step()
    """

    def __init__(
        self,
        lm: StandaloneLM,
        B: int,
        T_block: int,
        K_h: int,
        lambda_balance: float,
        amp_dtype: torch.dtype | None = torch.bfloat16,
        compile_inner: bool = True,
    ) -> None:
        if lm.cfg.use_neuromod:
            raise NotImplementedError(
                "CapturedBlockTrainer requires use_neuromod=False. The "
                "captured graph reads e_bias from a stable buffer, but the "
                "neuromod fire (graph transformer over touched columns) "
                "uses dynamic-shape ops (torch.nonzero) that cannot be "
                "captured. Adding neuromod support requires snapshotting "
                "visit_count to a stable buffer inside the captured graph "
                "and running neuromod + commit outside per replay; see "
                "docs/triton_rewrite_plan.md 'Constraints accepted'."
            )
        self.lm = lm
        self.B = B
        self.T_block = T_block
        self.K_h = K_h
        self.lambda_balance = lambda_balance
        self.amp_dtype = amp_dtype
        self.use_neuromod = lm.cfg.use_neuromod
        # Compile block_forward with inductor's default mode (no cudagraph)
        # to get cross-step fusion before our manual cudagraph wraps it.
        self._block_callable = (
            lm.memory.block_forward if not compile_inner
            else torch.compile(lm.memory.block_forward, mode="default", fullgraph=True)
        )

        cfg = lm.cfg
        device = next(lm.parameters()).device
        self.device = device

        # Static input buffers.
        i64 = dict(dtype=torch.int64, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        bool_ = dict(dtype=torch.bool, device=device)

        self.tokens_buf = torch.zeros(B, T_block, **i64)
        self.tau_buf = torch.zeros((), **f32)
        self.eps_buf = torch.zeros((), **f32)
        self.targets_buf = torch.zeros(B, T_block, K_h, **i64)
        self.valid_btk_buf = torch.zeros(B, T_block, K_h, **bool_)
        self.valid_tk_buf = torch.zeros(T_block, K_h, **bool_)
        self.horizon_weights_buf = torch.zeros(K_h, **f32)

        # E_bias input buffer. Always allocated with requires_grad=True so the
        # captured backward populates ``.grad`` — caller uses that grad to
        # backprop through the neuromod chain (built outside the captured
        # graph). When neuromod is off, the grad is unused and ignored.
        self.e_bias_buf = torch.zeros(
            lm.cfg.num_edges, dtype=torch.float32, device=device,
            requires_grad=True,
        )

        # Static output buffers.
        self.ce_loss_buf = torch.zeros((), **f32)
        self.balance_loss_buf = torch.zeros((), **f32)
        self.block_horizon_sum_buf = torch.zeros(K_h, **f32)
        self.block_horizon_count_buf = torch.zeros(K_h, **f32)
        self.ce_masked_buf = torch.zeros(B, T_block, K_h, **f32)

        self.graph: torch.cuda.CUDAGraph | None = None
        self._captured = False

    def _iter_body(self) -> None:
        """One block iteration — captured end-to-end.

        Includes: caches refresh + block_forward + CE + backward + state
        writeback + surprise EMA streaming + Hebbian plasticity update.
        Everything that touches a stable buffer with static shapes lives
        here so the captured graph absorbs the full per-block hot path.

        All input tensors must already hold the values for this block (caller
        copies before invoking this).
        """
        lm = self.lm
        cfg = lm.cfg
        memory = lm.memory

        # Recompute caches inside the captured region so the kernels
        # re-execute each replay with current parameter values.
        memory._horizon_logits_cache = None
        memory._alpha_cache = None
        memory._input_keys_cache = None
        memory._k_all_cache = None
        memory._ensure_block_caches(memory.tied_token_emb.weight)

        # E_bias input — always read from the stable, requires_grad buffer.
        # Caller copies the actual values (E_bias_flat + active_delta_nm)
        # into e_bias_buf before each replay; the captured backward writes
        # routing's gradient back into e_bias_buf.grad.
        e_bias = self.e_bias_buf

        ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.amp_dtype is not None and self.device.type == "cuda"
            else torch.autocast(device_type=self.device.type, enabled=False)
        )

        with ctx:
            out = self._block_callable(
                memory.s, memory.walker_pos, memory.walker_state,
                memory.prev_motor, e_bias,
                self.tokens_buf, self.tau_buf, self.eps_buf, True,
            )

            # Per-horizon CE — kept in fp32 for stable accumulation.
            with torch.autocast(device_type=self.device.type, enabled=False):
                ce_masked = memory.readout_ce_block(
                    out.motor_states_bt, self.targets_buf, self.valid_btk_buf,
                )                                                       # [B, T, K_h]
                block_horizon_sum = ce_masked.sum(dim=(0, 1))           # [K_h]
                block_horizon_count = (
                    self.valid_tk_buf.float().sum(dim=0) * self.B
                )                                                       # [K_h]
                has_counts_f = (block_horizon_count > 0).float()
                per_h = block_horizon_sum / block_horizon_count.clamp(min=1)
                w_eff = self.horizon_weights_buf * has_counts_f         # [K_h]
                ce = (per_h * w_eff).sum() / w_eff.sum().clamp(min=1)
                balance_term = self.lambda_balance * out.load_balance_loss
                loss = ce + balance_term

        loss.backward()

        # All non-grad bookkeeping — in-place writes into stable buffers
        # so subsequent replays see the updated values.
        with torch.no_grad():
            # Saved scalars / tensors for caller readout.
            self.ce_loss_buf.copy_(ce.detach())
            self.balance_loss_buf.copy_(balance_term.detach())
            self.block_horizon_sum_buf.copy_(block_horizon_sum.detach())
            self.block_horizon_count_buf.copy_(block_horizon_count.detach())
            ce_masked_det = ce_masked.detach()
            self.ce_masked_buf.copy_(ce_masked_det)

            # Forward state writeback. out.* live in cudagraph pool and may
            # be overwritten next replay; .copy_ takes ownership into stable
            # state buffers.
            memory.s.copy_(out.s_new)
            memory.walker_pos.copy_(out.walker_pos_new)
            memory.walker_state.copy_(out.walker_state_new)
            memory.prev_motor.copy_(out.prev_motor_new)

            # Non-grad accumulators — in-place add into stable buffers.
            memory.co_visit_flat.add_(out.co_visit_total)
            if memory.visit_count is not None:
                memory.visit_count.add_(out.visit_count_total)

            # Surprise EMA streaming — vectorized closed form replaces the
            # per-token Python loop. For α_s = cfg.alpha_gamma_s, the
            # invariant for masked recurrence
            #     ema_t = (1-α)·ema_{t-1} + α·ce_t   if valid_t
            #           = ema_{t-1}                  otherwise
            # collapses to a single weighted average over valid positions in
            # the limit of small α; here we approximate by averaging over
            # valid positions and EMA-blending once per block. Same long-run
            # behavior, no T_block kernel launches.
            alpha_s = cfg.alpha_gamma_s
            valid_count_btk = self.valid_btk_buf.to(torch.float32).sum(dim=1).clamp(min=1)
            block_mean_ce = (
                ce_masked_det * self.valid_btk_buf.to(torch.float32)
            ).sum(dim=1) / valid_count_btk                              # [B, K_h]
            block_has_any = (
                self.valid_btk_buf.to(torch.float32).sum(dim=1) > 0
            ).to(torch.float32)                                         # [B, K_h]
            ema_old = memory.surprise_ema.to(torch.float32)
            ema_candidate = (1.0 - alpha_s) * ema_old + alpha_s * block_mean_ce
            ema_new = block_has_any * ema_candidate + (1.0 - block_has_any) * ema_old
            memory.surprise_ema.copy_(ema_new)

            # Hebbian plasticity update on E_bias_flat (no neuromod path).
            # Window len equals T_block here (one full window per block).
            window = float(cfg.mod_period)
            co_visit_norm = memory.co_visit_flat / window
            surprise_scalar = memory.surprise_ema.mean()
            eta_global = cfg.plast_eta * torch.sigmoid(
                surprise_scalar - cfg.plast_surprise_bias,
            )
            delta_hebb = eta_global * (
                co_visit_norm - cfg.plast_decay * memory.E_bias_flat
            )
            new_e_bias = (memory.E_bias_flat + delta_hebb).clamp(
                -cfg.E_bias_max, cfg.E_bias_max,
            )
            memory.E_bias_flat.copy_(new_e_bias)

            # Reset window-scoped counters in-place (no reassign — keep
            # buffer addresses stable for the next replay).
            memory.co_visit_flat.zero_()
            if memory.visit_count is not None:
                memory.visit_count.zero_()
            # surprise_prev snapshot (no neuromod consumes it, but keep
            # the contract stable).
            memory.surprise_prev.copy_(memory.surprise_ema)

    def warmup_and_capture(self, n_warmup: int = 3) -> None:
        """Warmup + capture, all on a dedicated side stream.

        Per PyTorch's CUDA-graph idiom: warmup AND capture must share the same
        stream so AccumulateGrad nodes (created on first warmup backward)
        match the capture stream. Default stream stalls until the side
        stream's captured graph is built, then we ``wait_stream`` to merge.
        """
        if self._captured:
            return

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            # Warmup: allocate param.grad buffers on this stream + populate
            # AccumulateGrad node bindings so capture reuses them.
            for _ in range(n_warmup):
                for p in self.lm.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                self._iter_body()

            # Capture on the SAME stream as warmup.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                self._iter_body()

        torch.cuda.current_stream().wait_stream(side)
        self._captured = True

    def replay(
        self,
        tokens: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
        targets: torch.Tensor,
        valid_btk: torch.Tensor,
        valid_tk: torch.Tensor,
        horizon_weights: torch.Tensor,
        e_bias_value: torch.Tensor,
    ) -> CapturedBlockStats:
        """Copy inputs into static buffers and replay the captured graph.

        ``e_bias_value`` carries the active E_bias for this block — typically
        ``E_bias_flat.detach() + neuromod_eta * active_delta_nm`` (where
        ``active_delta_nm`` carries grad_fn back to neuromod params). Its
        values are copied into the stable e_bias_buf; after replay,
        ``e_bias_buf.grad`` holds routing's gradient w.r.t. those values.
        Caller uses that grad to backprop through the neuromod chain.
        """
        if not self._captured:
            raise RuntimeError("call warmup_and_capture() before replay()")

        # Copy fresh inputs into the captured input buffers.
        self.tokens_buf.copy_(tokens, non_blocking=True)
        self.tau_buf.copy_(tau, non_blocking=True)
        self.eps_buf.copy_(eps, non_blocking=True)
        self.targets_buf.copy_(targets, non_blocking=True)
        self.valid_btk_buf.copy_(valid_btk, non_blocking=True)
        self.valid_tk_buf.copy_(valid_tk, non_blocking=True)
        self.horizon_weights_buf.copy_(horizon_weights, non_blocking=True)

        # E_bias requires care: copy values without recording autograd
        # (which would promote e_bias_buf from leaf to non-leaf and break
        # the AccumulateGrad node bound at capture time). Zero existing
        # grad so the post-replay value is just THIS block's gradient.
        with torch.no_grad():
            self.e_bias_buf.copy_(e_bias_value.detach(), non_blocking=True)
            if self.e_bias_buf.grad is not None:
                self.e_bias_buf.grad.zero_()

        self.graph.replay()

        return CapturedBlockStats(
            ce_loss=self.ce_loss_buf,
            balance_loss=self.balance_loss_buf,
            block_horizon_sum=self.block_horizon_sum_buf,
            block_horizon_count=self.block_horizon_count_buf,
            ce_masked=self.ce_masked_buf,
        )
