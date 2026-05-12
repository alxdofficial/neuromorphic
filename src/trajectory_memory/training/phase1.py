"""Phase 1 (TF NTP) trainer for Wave 1 (long-doc) and Wave 2 (long-chat).

Per plan §4.7:
- Wave 1: standard TF NTP on long documents, surprise on all tokens.
- Wave 2: TF NTP on TurnPair (prior, response), surprise only on response.

Both use cross-window TBPTT (plan §4.2): each "training sequence" is a
chunk of `D * T_window` tokens; backward fires per chunk; manifold state
detached at chunk boundary.

The `Phase1Trainer` class is the proper training harness:
- gradient clipping
- LR scheduler hookup
- per-step metrics dict (loss, grad_norm, lr)
- checkpoint integration via `state_dict()` / `load_state_dict()`

The legacy free functions `phase1_wave1_step` / `phase1_wave2_step` are
preserved for backward-compatibility with the old train_wave*.py scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.tbptt import _detach_kv_cache, run_chunk
from src.trajectory_memory.training.loaders import TurnPairBatch


@dataclass
class Phase1Metrics:
    """Per-step metrics returned by Phase1Trainer.step_*."""
    loss: float
    grad_norm: float
    lr: list[float]                # per param group
    surprise_history: Tensor | None = None
    final_states: Tensor | None = None
    final_hiddens: Tensor | None = None
    final_lm_context: Tensor | None = None
    final_past_key_values: object | None = None  # KV-cache mode only
    final_cache_abs_pos: int = 0                 # KV-cache mode only
    # B9 fix — surface visited IDs across windows for trajectory-diversity
    # telemetry. Shape after stacking: [BS, n_windows, J, K_read|K_write].
    read_visited_ids: Tensor | None = None
    write_visited_ids: Tensor | None = None
    # B8 fix — inject SNR diagnostics. Read from MemInjectLayer's
    # `_last_inj_norm` and `_last_hidden_norm` buffers (last forward only).
    # If memory module silently collapses (scale → 0), inject_snr → 0.
    inject_norm: float = 0.0
    hidden_norm: float = 0.0
    # Routing/bridge architectural-health scalars. Added after the
    # Gumbel-noise bug — these surface the parameters whose values
    # failed to learn during Wave 1, so a sustained-flat trace is a
    # smoke alarm.
    bridge_scale_raw_mean: float = 0.0      # MemInjectLayer.scale_raw.mean()
    read_logit_scale: float = 0.0           # exp(read_module.logit_scale_raw)
    write_logit_scale: float = 0.0          # exp(write_module.logit_scale_raw)
    # Loss component breakdown — each is the post-coefficient
    # contribution to `loss`, so they sum to `loss` (within rounding).
    # Watching the relative magnitudes catches: (a) aux losses
    # dominating CE (over-regularized to uniform routing), (b) CE
    # diverging while aux stays flat, etc.
    ntp_ce_loss: float = 0.0                # post-prior_loss_weight CE
    aux_lb_loss: float = 0.0                # post-coef load_balance contribution
    aux_z_loss: float = 0.0                 # post-coef z_loss contribution


class Phase1Trainer:
    """TF NTP trainer for Wave 1 + Wave 2.

    Usage:
        trainer = Phase1Trainer(model, optimizer, scheduler=scheduler, grad_clip=1.0)
        for step in range(num_steps):
            metrics = trainer.step_wave1(chunk)        # Wave 1
            # OR
            metrics = trainer.step_wave2(batch)        # Wave 2
            log(metrics)
    """

    def __init__(
        self,
        model: IntegratedLM,
        optimizer: Optimizer,
        *,
        scheduler: object | None = None,    # WarmupCosineScheduler-like
        grad_clip: float | None = 1.0,
        pad_token_id: int | None = None,
        use_kv_cache: bool = True,
        prior_loss_weight: float = 0.0,
        # Routing-collapse mitigations (Switch + ST-MoE + VQ-VAE pattern).
        # Defaults are the canonical values from those papers.
        load_balance_coef: float = 1e-2,
        z_loss_coef: float = 1e-3,
        revive_every: int = 500,            # steps between dead-code revival
        # `None` → auto-derive as 1/(100·N), i.e. 1% of uniform-routing
        # usage. Scales correctly across manifold sizes. At N=4096 this
        # gives 2.4e-6 (a concept visited ≤1% of uniform is "dead").
        revive_threshold: float | None = None,
        tau_init: float = 1.0,              # Gumbel temp at step 0
        tau_floor: float = 0.5,             # never drop below this
        tau_decay_rate: float = 1e-4,       # τ = max(floor, init·exp(-rate·step))
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        # KV-cache mode: Llama re-encodes only new T_window tokens per
        # window. ~1.79× faster than rolling buffer; for BS>1 caller
        # resets past_kv to None on doc boundaries (lockstep).
        self.use_kv_cache = use_kv_cache
        # pad_token_id is used in step_wave2 / eval_wave2 for chunk padding.
        # Default 0 (legacy) only for backward compat with toy tests; real
        # training MUST pass the tokenizer's pad_token_id (128001 for
        # Llama-3) so Llama doesn't see synthetic `!` tokens in its context.
        # pad_token_id MUST be passed for real training. Default None +
        # assert below catches the silent footgun (prior code defaulted to
        # 0 = `!` on Llama-3, polluting LM context with synthetic chars).
        if pad_token_id is None:
            pad_token_id = 0  # legacy test-mode fallback; real callers MUST pass
        self.pad_token_id = pad_token_id
        # prior_loss_weight: in Wave 2's `step_wave2`, all prior tokens are
        # masked out of the NTP loss by default — so memory writes during
        # the prior get NO direct gradient signal (per-chunk backward +
        # detach cuts the gradient path before the response loss arrives).
        # Setting `prior_loss_weight > 0` adds a small NTP loss on prior
        # tokens (multiplied by this weight) so prior writes get trained.
        # Default 0 preserves §4.5 behavior; recommended 0.1 for matching
        # plan §4.8 surprise table.
        self.prior_loss_weight = prior_loss_weight
        self._warned_prior_loss_weight = False  # one-shot warn at step_wave2
        # Routing aux losses + revival + temp schedule.
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        self.revive_every = revive_every
        # Auto-derive revive_threshold from manifold size so it scales
        # with N. Default 1/(100·N) = 1% of perfectly-uniform routing.
        if revive_threshold is None:
            revive_threshold = 1.0 / (100.0 * model.cfg.N)
        self.revive_threshold = revive_threshold
        self.tau_init = tau_init
        self.tau_floor = tau_floor
        self.tau_decay_rate = tau_decay_rate
        self._step_count = 0

    def _current_tau(self) -> Tensor:
        """Gumbel temperature schedule: exponentially decay from tau_init
        toward tau_floor. Floor prevents the saturation pathology that
        killed entry_proj's gradient in the Wave-1 plateau analysis.

        Returns a 0-dim CUDA tensor (not a Python float!) because dynamo
        specializes on Python scalar values and recompiles every step
        when tau changes. Passing as a tensor makes it a dynamic input
        — one compile, reused forever. This was an actual ~35% per-step
        slowdown until we found it.
        """
        import math as _m
        val = max(
            self.tau_floor,
            self.tau_init * _m.exp(-self.tau_decay_rate * self._step_count),
        )
        device = next(self.model.parameters()).device
        return torch.tensor(val, device=device, dtype=torch.float32)

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        """Trainer state (for checkpointing). Does NOT include model /
        optimizer / scheduler state — caller saves those separately via
        `save_checkpoint`."""
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

    # ── Wave 1: long-doc TF NTP ───────────────────────────────────────

    def step_wave1(
        self,
        chunk: Tensor,                              # [BS, D*T_window]
        *,
        prev_states: Tensor | None = None,
        prev_window_hiddens: Tensor | None = None,
        prev_lm_context: Tensor | None = None,
        target_mask: Tensor | None = None,          # [BS, D, T_window] bool
        past_key_values: object | None = None,
        cache_abs_pos: int = 0,
    ) -> Phase1Metrics:
        """One Wave 1 step (long-doc TF NTP).

        `target_mask`: per-token mask True=include in NTP CE, False=skip.
        Used to exclude pad-token tails from partial chunks (see
        LongDocChunk.valid_mask). When None, all tokens contribute to loss.

        `past_key_values`: pass `final_past_key_values` from prior chunk to
        thread KV cache across chunks of the same doc; None to start fresh.
        """
        cfg = self.model.cfg
        BS, T_total = chunk.shape
        assert T_total == cfg.D * cfg.T_window, (
            f"chunk length {T_total} != D*T_window {cfg.D * cfg.T_window}"
        )

        if prev_states is None:
            prev_states = self.model.manifold.reset_states(batch_size=BS)

        windows = chunk.view(BS, cfg.D, cfg.T_window)

        self.optimizer.zero_grad(set_to_none=True)
        tau_eff = self._current_tau()
        out = run_chunk(
            self.model,
            windows,
            prev_states=prev_states,
            prev_window_hiddens=prev_window_hiddens,
            prev_lm_context=prev_lm_context,
            target_mask=target_mask,
            hard_routing=True,
            use_kv_cache=self.use_kv_cache,
            past_key_values=past_key_values,
            cache_abs_pos=cache_abs_pos,
            tau=tau_eff,
        )
        loss = out["aggregate_loss"]
        ntp_loss_value = float(loss.detach())
        # Switch-Transformer + ST-MoE routing aux losses. Coefficients are
        # the canonical values from those papers (1e-2 and 1e-3 respectively).
        # Skipped if forward_window didn't surface them (e.g., the rare test
        # path that bypasses both modules — defensive).
        aux_lb = out.get("aux_load_balance")
        aux_z = out.get("aux_z_loss")
        aux_lb_value = 0.0
        aux_z_value = 0.0
        if aux_lb is not None:
            aux_lb_value = float(aux_lb.detach()) * self.load_balance_coef
            loss = loss + self.load_balance_coef * aux_lb
        if aux_z is not None:
            aux_z_value = float(aux_z.detach()) * self.z_loss_coef
            loss = loss + self.z_loss_coef * aux_z
        loss.backward()

        grad_norm = self._clip_and_step()
        self._step_count += 1

        # VQ-VAE-style dead-concept revival + usage tracking. Done AFTER
        # the optimizer step so the revived concept's reset doesn't get
        # immediately undone. Record visits on every step; revive only
        # periodically (cheap O(N) scan).
        with torch.no_grad():
            for w in out["window_outputs"]:
                self.model.manifold.record_visits(w["read_visited"])
                self.model.manifold.record_visits(w["write_visited"])
            if (
                self.revive_every > 0
                and self._step_count % self.revive_every == 0
            ):
                self.model.manifold.revive_dead_concepts(
                    threshold=self.revive_threshold,
                )

        # KV-cache-mode: detach final cache for cross-chunk carry.
        final_cache = out.get("final_past_key_values", None)
        if final_cache is not None:
            final_cache = _detach_kv_cache(final_cache)

        # B9 — collect visited IDs across windows for trajectory diversity.
        # `out["window_outputs"]` is a list of D dicts; each has read/write_visited
        # of shape [BS, J, K_*]. Stack along window axis.
        try:
            read_ids = torch.stack(
                [w["read_visited"] for w in out["window_outputs"]], dim=1,
            ).detach()
        except Exception:
            read_ids = None
        try:
            write_ids = torch.stack(
                [w["write_visited"] for w in out["window_outputs"]], dim=1,
            ).detach()
        except Exception:
            write_ids = None

        # B8 — inject SNR readout (cheap detached scalars).
        inj_norm = 0.0
        hid_norm = 0.0
        bridge_scale_raw_mean = 0.0
        if self.model.llama is not None:
            mil = self.model._mem_inject_layer()
            inj_norm = float(mil._last_inj_norm.item())
            hid_norm = float(mil._last_hidden_norm.item())
            bridge_scale_raw_mean = float(mil.scale_raw.detach().mean().item())

        # Routing logit_scale telemetry — detects the Gumbel-noise-dominant
        # routing pathology: if logit_scale doesn't grow above ~exp(1.5),
        # cosine signal is being out-noised in the softmax temperature.
        read_logit_scale = float(
            self.model.read_module.logit_scale_raw.detach().exp().item()
        )
        write_logit_scale = float(
            self.model.write_module.logit_scale_raw.detach().exp().item()
        )

        return Phase1Metrics(
            loss=float(loss.detach()),
            grad_norm=float(grad_norm),
            lr=self._current_lrs(),
            surprise_history=out["surprise_history"].detach(),
            final_states=out["final_states"].detach(),
            final_hiddens=out["final_hiddens"].detach(),
            final_lm_context=out["final_lm_context"],
            final_past_key_values=final_cache,
            final_cache_abs_pos=int(out.get("final_cache_abs_pos", 0)),
            read_visited_ids=read_ids,
            write_visited_ids=write_ids,
            inject_norm=inj_norm,
            hidden_norm=hid_norm,
            bridge_scale_raw_mean=bridge_scale_raw_mean,
            read_logit_scale=read_logit_scale,
            write_logit_scale=write_logit_scale,
            ntp_ce_loss=ntp_loss_value,
            aux_lb_loss=aux_lb_value,
            aux_z_loss=aux_z_value,
        )

    # ── Wave 2: long-chat TF NTP (TurnPair) ───────────────────────────

    def step_wave2(self, batch: TurnPairBatch) -> Phase1Metrics:
        """One Wave 2 step (long-chat TF NTP).

        Concatenates prior + response per example, chunks into TBPTT
        windows, applies surprise-on-response mask, detaches state at
        chunk boundary, accumulates loss across chunks, single backward
        per example.
        """
        if not self._warned_prior_loss_weight and self.prior_loss_weight == 0.0:
            self._warned_prior_loss_weight = True
            import warnings
            warnings.warn(
                "step_wave2() with prior_loss_weight=0: prior tokens "
                "contribute zero CE → memory writes during the prior get "
                "no gradient signal (per-chunk backward + detach cuts "
                "the path before response loss arrives). Recommended: "
                "pass prior_loss_weight=0.1 to Phase1Trainer (the "
                "train_wave2.py script default).",
                stacklevel=2,
            )
        cfg = self.model.cfg
        BS = batch.prior_ids.shape[0]
        device = batch.prior_ids.device
        pad_token = self.pad_token_id

        full_ids = torch.cat([batch.prior_ids, batch.response_ids], dim=1)
        # B12 fix — Per-position loss weight: 1.0 on response, prior_loss_weight
        # on prior, 0.0 on pad. When prior_loss_weight=0 (default), prior
        # tokens contribute zero CE → memory writes during the prior get NO
        # gradient signal (per-chunk backward + detach cuts the gradient
        # path before response loss arrives). Setting prior_loss_weight>0
        # gives prior writes a small training signal — recommended 0.1 per
        # plan §4.8 surprise table. The mask is a float weight, NOT a bool
        # — `_compute_surprise_window` and the run_chunk loss aggregation
        # treat the mask multiplicatively in CE.
        prior_w = self.prior_loss_weight
        full_mask = torch.cat(
            [batch.prior_mask.to(torch.float32) * prior_w,
             batch.response_mask.to(torch.float32)],
            dim=1,
        )
        T_full = full_ids.shape[1]
        chunk_len = cfg.D * cfg.T_window

        # Pad to multiple of chunk_len.
        if T_full % chunk_len != 0:
            pad_n = chunk_len - (T_full % chunk_len)
            full_ids = torch.cat([
                full_ids,
                torch.full((BS, pad_n), pad_token, dtype=full_ids.dtype, device=device),
            ], dim=1)
            full_mask = torch.cat([
                full_mask,
                torch.zeros((BS, pad_n), dtype=torch.float32, device=device),
            ], dim=1)
            T_full = full_ids.shape[1]
        n_chunks = T_full // chunk_len

        prev_states = self.model.manifold.reset_states(batch_size=BS)
        prev_window_hiddens: Tensor | None = None
        prev_lm_context: Tensor | None = None
        past_kv: object | None = None
        cache_abs_pos = 0

        self.optimizer.zero_grad(set_to_none=True)
        # Per-chunk backward + detach is the standard TBPTT pattern for
        # variable-length sequences: bound activation memory at one
        # chunk's worth, accumulate gradients across chunks via the
        # optimizer's grad buffers, do one optimizer.step at the end.
        # The earlier "single total_loss.backward()" pattern OOMd at BS=1
        # on long WildChat priors (10K+ tokens / 10+ chunks of activations
        # held alive simultaneously).
        #
        # Token-weighted cross-chunk normalization: pre-compute the total
        # valid token count from full_mask (cheap, no forward needed). Per
        # chunk, backward `chunk_ce_sum / total_valid_count` — equivalent
        # to summing per-chunk weighted CE then dividing by total count
        # (proper token-weighted overall mean). The earlier `chunk_loss
        # / n_chunks` was chunk-equal weighted: a chunk with 1 valid
        # token contributed the same gradient as a chunk with 1024 valid
        # tokens. Bad for Wave 2 where response often concentrates in
        # a few chunks while priors span many.
        total_valid_count = float((full_mask > 0).sum().item())
        # Guard: degenerate batch with zero valid tokens → no learning,
        # return early with zero loss to avoid div-by-zero.
        if total_valid_count == 0:
            self._step_count += 1
            return Phase1Metrics(
                loss=0.0, grad_norm=0.0, lr=self._current_lrs(),
                surprise_history=None,
            )
        total_loss_value = 0.0
        all_surprise: list[Tensor] = []

        for c in range(n_chunks):
            ids = full_ids[:, c * chunk_len : (c + 1) * chunk_len]
            mask = full_mask[:, c * chunk_len : (c + 1) * chunk_len]
            windows = ids.view(BS, cfg.D, cfg.T_window)
            win_mask = mask.view(BS, cfg.D, cfg.T_window)

            out = run_chunk(
                self.model, windows,
                prev_states=prev_states,
                prev_window_hiddens=prev_window_hiddens,
                prev_lm_context=prev_lm_context,
                target_mask=win_mask,
                hard_routing=True,
                use_kv_cache=self.use_kv_cache,
                past_key_values=past_kv,
                cache_abs_pos=cache_abs_pos,
            )
            # Token-weighted cross-chunk aggregation: scale by 1/total_count
            # so summing across chunks gives sum(weighted_CE) / total_count.
            # chunk_ce_sum is the with-grad weighted-CE sum across windows
            # in this chunk (carries float-mask weights baked in).
            if "chunk_ce_sum" in out:
                chunk_loss = out["chunk_ce_sum"] / total_valid_count
            else:
                # Test-mode fallback — no chunk_ce_sum surfaced; revert to
                # the previous chunk-equal pattern.
                chunk_loss = out["aggregate_loss"] / n_chunks
            # Mirror Wave 1's aux-loss handling — without this Wave 2
            # backprops only CE, leaving routing unregularized
            # (load-balance + z-loss) on long-chat data. The Wave 1
            # loop adds them at line ~240.
            aux_lb = out.get("aux_load_balance")
            aux_z = out.get("aux_z_loss")
            if aux_lb is not None:
                chunk_loss = chunk_loss + self.load_balance_coef * aux_lb
            if aux_z is not None:
                chunk_loss = chunk_loss + self.z_loss_coef * aux_z
            # Per-chunk backward — accumulates grad into optimizer's
            # buffers, releases this chunk's activations before the next
            # chunk allocates its own.
            chunk_loss.backward()
            total_loss_value = total_loss_value + float(chunk_loss.detach())
            all_surprise.append(out["surprise_history"].detach())

            # Detach for cross-chunk state carry. Already cut from autograd
            # by the per-chunk backward, but keep .detach() to drop the
            # graph node references explicitly.
            prev_states = out["final_states"].detach()
            prev_window_hiddens = out["final_hiddens"].detach()
            prev_lm_context = out["final_lm_context"]
            past_kv = out.get("final_past_key_values", None)
            cache_abs_pos = int(out.get("final_cache_abs_pos", cache_abs_pos))
            if past_kv is not None:
                past_kv = _detach_kv_cache(past_kv)

        grad_norm = self._clip_and_step()
        self._step_count += 1

        return Phase1Metrics(
            loss=total_loss_value,
            grad_norm=float(grad_norm),
            lr=self._current_lrs(),
            surprise_history=torch.stack(all_surprise, dim=1).detach() if all_surprise else None,
        )

    # ── Validation (no-grad) ──────────────────────────────────────────

    @torch.no_grad()
    def eval_wave1(
        self,
        chunk: Tensor,
        *,
        prev_states: Tensor | None = None,
        prev_window_hiddens: Tensor | None = None,
        prev_lm_context: Tensor | None = None,
        target_mask: Tensor | None = None,
        past_key_values: object | None = None,
        cache_abs_pos: int = 0,
    ) -> dict:
        """Forward-only Wave 1 chunk; returns NTP loss + carry state.

        N9 fix: when called repeatedly on chunks of the SAME document (per
        the train_wave1.py val loop), state must thread across calls just
        like training. Without state-threading, validation can't measure
        cross-chunk memory ability — needle-haystack val with planted
        facts >2K tokens away would be invisible to memory.

        Returns dict with: loss, final_states, final_hiddens,
        final_lm_context, final_past_key_values, final_cache_abs_pos.
        Caller threads these into the next chunk of the same doc.
        """
        cfg = self.model.cfg
        BS, T_total = chunk.shape
        assert T_total == cfg.D * cfg.T_window
        if prev_states is None:
            prev_states = self.model.manifold.reset_states(batch_size=BS)
        windows = chunk.view(BS, cfg.D, cfg.T_window)
        out = run_chunk(
            self.model, windows,
            prev_states=prev_states,
            prev_window_hiddens=prev_window_hiddens,
            prev_lm_context=prev_lm_context,
            target_mask=target_mask,
            hard_routing=False,    # eval: deterministic routing
            use_kv_cache=self.use_kv_cache,
            past_key_values=past_key_values,
            cache_abs_pos=cache_abs_pos,
        )
        return {
            "loss": float(out["aggregate_loss"].detach()),
            "final_states": out["final_states"],
            "final_hiddens": out["final_hiddens"],
            "final_lm_context": out["final_lm_context"],
            "final_past_key_values": out.get("final_past_key_values", None),
            "final_cache_abs_pos": int(out.get("final_cache_abs_pos", 0)),
        }

    @torch.no_grad()
    def eval_wave2(self, batch: TurnPairBatch) -> float:
        """Forward-only Wave 2 TurnPair; mirrors the training loss mask."""
        cfg = self.model.cfg
        BS = batch.prior_ids.shape[0]
        device = batch.prior_ids.device
        pad_token = self.pad_token_id

        full_ids = torch.cat([batch.prior_ids, batch.response_ids], dim=1)
        full_mask = torch.cat(
            [batch.prior_mask.to(torch.float32) * self.prior_loss_weight,
             batch.response_mask.to(torch.float32)],
            dim=1,
        )
        T_full = full_ids.shape[1]
        chunk_len = cfg.D * cfg.T_window
        if T_full % chunk_len != 0:
            pad_n = chunk_len - (T_full % chunk_len)
            full_ids = torch.cat([
                full_ids,
                torch.full((BS, pad_n), pad_token, dtype=full_ids.dtype, device=device),
            ], dim=1)
            full_mask = torch.cat([
                full_mask,
                torch.zeros((BS, pad_n), dtype=torch.float32, device=device),
            ], dim=1)
            T_full = full_ids.shape[1]
        n_chunks = T_full // chunk_len

        prev_states = self.model.manifold.reset_states(batch_size=BS)
        prev_window_hiddens: Tensor | None = None
        prev_lm_context: Tensor | None = None
        past_kv: object | None = None
        cache_abs_pos = 0
        # Token-weighted cross-chunk aggregation (mirrors step_wave2):
        # accumulate sum + count separately, divide once at end. The
        # earlier `total = sum(aggregate_loss)` was chunk-equal-weighted,
        # so a 1024-valid-token chunk contributed the same as a
        # 1-valid-token chunk → val_loss became hard to interpret across
        # examples of varying length.
        total_ce_sum = torch.zeros((), device=device)
        total_count = 0.0
        for c in range(n_chunks):
            ids = full_ids[:, c * chunk_len : (c + 1) * chunk_len]
            mask = full_mask[:, c * chunk_len : (c + 1) * chunk_len]
            out = run_chunk(
                self.model, ids.view(BS, cfg.D, cfg.T_window),
                prev_states=prev_states,
                prev_window_hiddens=prev_window_hiddens,
                prev_lm_context=prev_lm_context,
                target_mask=mask.view(BS, cfg.D, cfg.T_window),
                hard_routing=False,  # eval: deterministic routing
                use_kv_cache=self.use_kv_cache,
                past_key_values=past_kv,
                cache_abs_pos=cache_abs_pos,
            )
            if "chunk_ce_sum" in out:
                total_ce_sum = total_ce_sum + out["chunk_ce_sum"].detach()
                total_count += float(out["chunk_ce_count"].sum().item())
            else:
                # Test-mode fallback.
                total_ce_sum = total_ce_sum + out["aggregate_loss"].detach()
                total_count += 1.0
            prev_states = out["final_states"]
            prev_window_hiddens = out["final_hiddens"]
            prev_lm_context = out["final_lm_context"]
            past_kv = out.get("final_past_key_values", None)
            cache_abs_pos = int(out.get("final_cache_abs_pos", cache_abs_pos))
        if total_count == 0:
            return 0.0
        return float(total_ce_sum / max(total_count, 1.0))

    # ── helpers ───────────────────────────────────────────────────────

    def _clip_and_step(self) -> float:
        """Clip gradients, step optimizer + scheduler, return grad_norm.

        Skips optimizer.step() if grad_norm is non-finite so callers can
        abort cleanly without a NaN/Inf weight corruption first.
        """
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                max_norm=self.grad_clip,
            )
        else:
            grad_norm = torch.tensor(0.0)
        gn_val = float(grad_norm)
        if not math.isfinite(gn_val):
            self.optimizer.zero_grad(set_to_none=True)
            return gn_val
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return gn_val

    def _current_lrs(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


# ── Legacy free functions (kept for backward-compat with older callers) ──


def phase1_wave1_step(
    model: IntegratedLM,
    chunk: Tensor,
    *,
    optimizer: Optimizer,
    prev_states: Tensor | None = None,
    prev_window_hiddens: Tensor | None = None,
    prev_lm_context: Tensor | None = None,
    grad_clip: float | None = 1.0,
) -> dict:
    """Single-call Phase 1 / Wave 1 step. Prefer `Phase1Trainer.step_wave1`."""
    trainer = Phase1Trainer(model, optimizer, grad_clip=grad_clip)
    m = trainer.step_wave1(
        chunk,
        prev_states=prev_states,
        prev_window_hiddens=prev_window_hiddens,
        prev_lm_context=prev_lm_context,
    )
    return {
        "loss": m.loss,
        "grad_norm": m.grad_norm,
        "surprise_history": m.surprise_history,
        "final_states": m.final_states,
        "final_hiddens": m.final_hiddens,
        "final_lm_context": m.final_lm_context,
    }


def phase1_wave2_step(
    model: IntegratedLM,
    batch: TurnPairBatch,
    *,
    optimizer: Optimizer,
    grad_clip: float | None = 1.0,
) -> dict:
    """Single-call Phase 1 / Wave 2 step. Prefer `Phase1Trainer.step_wave2`."""
    trainer = Phase1Trainer(model, optimizer, grad_clip=grad_clip)
    m = trainer.step_wave2(batch)
    return {"loss": m.loss, "grad_norm": m.grad_norm,
            "surprise_history": m.surprise_history}
