"""IntegratedLM — Llama backbone + reused MemInjectLayer + per-window cycle.

Per-window cycle (see plan §2.5):
    1. READ  using prev_window_hiddens to autocomplete J trajectories
    2. PREDICT Llama tokens, with the read trajectory injected via cross-attn
    3. WRITE  using current_window_hiddens + surprise; persist via scatter_mean

The Llama-side cross-attention into the read trajectory uses a
TrajectoryReadAttn module (single-head, sufficient for the small KV
size of J·K_read = 32). MemInjectLayer's W_in/W_out handle the
d_lm ↔ D_concept bridge; TrajectoryReadAttn handles the actual cross
attention in concept space.

Surprise (per-window scalar) is computed as mean per-token NTP CE over
target-eligible positions. The data loader provides the target mask
(see plan §3.3, §4.8).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM

from src.pretrained.hosts import build_host
from src.pretrained.mem_inject_layer import MemInjectLayer
from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.manifold import Manifold
from src.trajectory_memory.read_module import ReadTrajectoryGenerator
from src.trajectory_memory.write_module import WriteTrajectoryGenerator


class TrajectoryReadAttn(nn.Module):
    """Cross-attention from Llama hidden (in D_concept space) to the read
    trajectory KV.

    This module is invoked inside MemInjectLayer's `memory_fn` closure.
    Both query and KV live in D_concept space — MemInjectLayer's W_in
    has already projected Llama's d_lm hidden down to D_concept.
    """

    def __init__(self, D_concept: int):
        super().__init__()
        self.D = D_concept
        self.W_q = nn.Linear(D_concept, D_concept, bias=False)
        self.W_k = nn.Linear(D_concept, D_concept, bias=False)
        self.W_v = nn.Linear(D_concept, D_concept, bias=False)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, h_mem: Tensor, read_trajectory: Tensor) -> Tensor:
        """
        Args:
            h_mem:           [BS, T, D_concept]    Llama hidden in memory space
            read_trajectory: [BS, J*K_read, D_concept]
        Returns:
            readout: [BS, T, D_concept]
        """
        Q = self.W_q(h_mem)                                       # [BS, T, D]
        K = self.W_k(read_trajectory)                             # [BS, M, D]
        V = self.W_v(read_trajectory)                             # [BS, M, D]
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.D)
        attn = F.softmax(scores, dim=-1)                          # [BS, T, M]
        return torch.bmm(attn, V)                                 # [BS, T, D]


class IntegratedLM(nn.Module):
    """Frozen Llama backbone + memory side-channel.

    Public API:
        wrapper = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B")
        out = wrapper.forward_window(input_ids, prev_window_hiddens, prev_states)
            → {logits, current_hiddens, new_states, read_visited_ids,
               write_visited_ids, surprise}

    `forward_window` is the per-window unit. Outer trainers chain D of
    these for cross-window TBPTT (see tbptt.py).
    """

    def __init__(
        self,
        cfg: TrajMemConfig,
        model_name: str = "meta-llama/Llama-3.2-1B",
        *,
        attach_lm: bool = True,
        llama_dtype: str = "bf16",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        # ── 1. Load LM (or skip in test mode) ─────────────────────────
        if attach_lm:
            hf_cfg = AutoConfig.from_pretrained(model_name)
            cfg.d_lm = hf_cfg.hidden_size                          # may differ from cfg.d_lm default
            cfg.validate()

            dt_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            self.llama = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dt_map[llama_dtype],
            )
            self.host = build_host(self.llama)
            if freeze_backbone:
                self.host.freeze_backbone()

            # ── 2. Replace inject layer with MemInjectLayer ───────────
            L = cfg.inject_layer
            assert L < hf_cfg.num_hidden_layers, (
                f"inject_layer={L} but model has {hf_cfg.num_hidden_layers} layers"
            )
            orig_layer = self.host.layer_list()[L]
            mem_inject = MemInjectLayer(
                orig_layer=orig_layer,
                d_lm=cfg.d_lm,
                d_mem=cfg.D_concept,
                scale_init=0.1,                   # small but non-zero so memory contributes from
                                                  # step 0; scale=0 zeros all downstream gradient
                                                  # to W_in/W_out/scale/read_attn (chicken-and-egg).
                memory_fn=None,
                bridge_hidden=cfg.bridge_hidden,
            )
            # Register under llama.model.layers[L] ONLY (via host.replace_layer).
            # We deliberately do NOT store as `self.mem_inject` because that
            # would alias it as a second submodule, duplicating its parameters
            # in state_dict() under two paths (mem_inject.* AND
            # llama.model.layers.{L}.*). PyTorch's state_dict walks all
            # module-tree paths without deduplicating. Access via
            # `self._mem_inject_layer()` instead.
            self.host.replace_layer(L, mem_inject)
        else:
            # Test mode: no Llama. Skeleton for unit tests of the cycle wiring.
            self.llama = None
            self.host = None

        # ── 3. Memory modules (always present) ───────────────────────
        self.manifold = Manifold(cfg)
        self.read_module = ReadTrajectoryGenerator(cfg)
        self.write_module = WriteTrajectoryGenerator(cfg)
        self.read_attn = TrajectoryReadAttn(cfg.D_concept)

    # ── memory wiring helpers ────────────────────────────────────────

    def _mem_inject_layer(self) -> MemInjectLayer:
        """Return the MemInjectLayer that lives at `cfg.inject_layer` in
        the Llama stack. Single registration path — see __init__ comment."""
        return self.host.layer_list()[self.cfg.inject_layer]

    def _build_memory_fn(self, read_trajectory: Tensor):
        """Return a closure suitable for MemInjectLayer.memory_fn.

        The closure captures `read_trajectory: [BS, J*K_read, D_concept]`
        and returns per-token readouts in D_concept space.
        """
        cfg = self.cfg
        flat_traj = read_trajectory.reshape(
            read_trajectory.shape[0],
            cfg.J * cfg.K_read,
            cfg.D_concept,
        )

        def memory_fn(h_mem: Tensor) -> Tensor:
            # h_mem: [BS, T, D_concept]; readout: [BS, T, D_concept]
            return self.read_attn(h_mem, flat_traj)

        return memory_fn

    # ── per-window cycle ─────────────────────────────────────────────

    def forward_window(
        self,
        lm_input_ids: Tensor,
        prev_window_hiddens: Tensor | None,
        prev_states: Tensor,
        *,
        target_mask: Tensor | None = None,
        hard_routing: bool = True,
        past_key_values: object | None = None,
        use_kv_cache: bool = False,
        last_prev_logit_hidden: Tensor | None = None,
        cache_abs_pos: int = 0,
        force_surprise: float | None = None,
    ) -> dict:
        """Run one window: read → predict → write.

        Two execution modes:

        - **Rolling-buffer mode** (`use_kv_cache=False`, default):
          `lm_input_ids` carries the full rolling LM context (last
          `effective_lm_context` tokens). Llama re-encodes the prefix
          every window. The current window is `lm_input_ids[:, -T_window:]`.

        - **KV-cache mode** (`use_kv_cache=True`): `lm_input_ids` is just
          the new T_window tokens. `past_key_values` carries cached KVs
          from prior windows; Llama only encodes the new tokens against
          the cache. Massively cheaper at long context. Returns the
          updated cache so the caller can carry it across windows of a
          chunk. See `docs/profile_analysis.md` for measured wins
          (rolling-buffer is ~70% of T1 vs V1.B slowdown).

        Args:
            lm_input_ids:        Rolling-buffer mode: [BS, L_lm] with
                                 L_lm in [T_window, effective_lm_context].
                                 KV-cache mode: [BS, T_window] (new tokens only).
            prev_window_hiddens: [BS, T_window, d_lm] from previous window.
                                 None at the very first window of a sequence.
            prev_states:         [BS, N, D_concept] manifold state going in.
            target_mask:         [BS, T_window] bool (True = include in
                                 surprise CE), aligned to the current window.
            hard_routing:        Gumbel-STE if True, argmax if False.
            past_key_values:     KV-cache mode only. HF DynamicCache (or
                                 None for first window of chunk). Tokens
                                 already in the cache are NOT re-encoded;
                                 their KVs were computed at their original
                                 encoding time and remain valid.
            use_kv_cache:        Switch between the two modes (above).
            last_prev_logit_hidden: KV-cache mode optional. The last hidden
                                 of the previous window (`prev_window_hiddens
                                 [:, -1:, :]` projected through the same path
                                 — but normally identical). Used to compute
                                 the predecessor logit for target 0 of the
                                 current window so we don't drop a CE token.
                                 None → first window of chunk → drop target 0.

        Returns:
            dict with keys:
                logits:               [BS, T_window, V] — logits for the current window
                current_hiddens:      [BS, T_window, d_lm]
                new_states:           [BS, N, D_concept]
                read_visited:         [BS, J, K_read]
                write_visited:        [BS, J, K_write]
                surprise:             [BS] mean per-token CE over the current window
                new_past_key_values:  KV-cache mode only — updated cache for next window
        """
        cfg = self.cfg
        T_window = cfg.T_window
        BS, L_lm = lm_input_ids.shape
        if use_kv_cache:
            assert L_lm == T_window, (
                f"KV-cache mode expects lm_input_ids of shape [BS, T_window]; "
                f"got L_lm={L_lm} != T_window={T_window}."
            )
        else:
            assert L_lm >= T_window, (
                f"lm_input_ids length {L_lm} < T_window {T_window}; the current "
                f"window must always be at the tail of lm_input_ids."
            )
            assert L_lm <= cfg.effective_lm_context, (
                f"lm_input_ids length {L_lm} > effective_lm_context "
                f"{cfg.effective_lm_context}; truncate before calling."
            )

        # ── 1. READ ──────────────────────────────────────────────────
        if prev_window_hiddens is None:
            # First window: no prior context. Use zeros — read trajectory
            # will be content-poor but the memory_fn will still have valid
            # shapes. Read module handles this gracefully (entry MLP just
            # gets a zero pooled vector).
            prev_window_hiddens = torch.zeros(
                BS, T_window, cfg.d_lm,
                dtype=prev_states.dtype, device=prev_states.device,
            )

        prev_hid_mem = prev_window_hiddens.to(prev_states.dtype)
        read_visited, read_visited_ids = self.read_module(
            prev_hid_mem, prev_states, self.manifold, hard=hard_routing,
        )                                                          # [BS, J, K_read, D]

        # ── 2. PREDICT ──────────────────────────────────────────────
        if self.llama is None:
            # Test-mode fallback: no Llama. We synthesize logits + hiddens
            # + surprise that ALL flow gradient back through the read
            # trajectory, so unit tests can exercise the full forward+backward
            # path. Real-LM mode overrides these from Llama's actual outputs.
            #
            # Buffers `_test_proj` (D_concept→d_lm) and `_test_lm_head`
            # (d_lm→fake_vocab) are registered lazily on first forward and
            # held as non-persistent buffers (don't show up in state_dict).
            traj_mean = read_visited.mean(dim=(1, 2))             # [BS, D_concept]
            fake_vocab = 64                                        # small vocab for test rollouts
            if not hasattr(self, "_test_proj"):
                proj = torch.randn(
                    cfg.D_concept, cfg.d_lm,
                    dtype=prev_states.dtype, device=prev_states.device,
                ) * 0.1
                self.register_buffer("_test_proj", proj, persistent=False)
                head = torch.randn(
                    cfg.d_lm, fake_vocab,
                    dtype=prev_states.dtype, device=prev_states.device,
                ) * 0.1
                self.register_buffer("_test_lm_head", head, persistent=False)
            current_hiddens = (
                traj_mean.unsqueeze(1).expand(BS, T_window, -1) @ self._test_proj
            )                                                      # [BS, T_window, d_lm]
            logits = current_hiddens @ self._test_lm_head          # [BS, T_window, fake_vocab]
            surprise = read_visited.mean(dim=(1, 2, 3)) * 0.01     # [BS]
        else:
            # Wire memory_fn for this forward call.
            mem_inject = self._mem_inject_layer()
            mem_inject.memory_fn = self._build_memory_fn(read_visited)
            # Call the base model (returns hidden states) instead of the
            # CausalLM wrapper. We then apply lm_head ONLY to the
            # T_window+1 positions we actually need.
            new_past_key_values = None
            try:
                if use_kv_cache:
                    # KV-cache mode: encode only the new T_window tokens
                    # against `past_key_values`. The cache transparently
                    # carries prior windows' KVs.
                    #
                    # TWO position-related concerns require passing two
                    # separate things to llama.model():
                    #
                    # (a) RoPE: cached KVs have rotations baked in at their
                    #     ORIGINAL abs positions. New Q must use absolute
                    #     positions so Q·K relative math is consistent.
                    #     → `position_ids = [abs_pos, abs_pos+1, ...]`
                    #
                    # (b) Causal mask: HF builds the mask via
                    #     `key_index <= cache_position[query]`. The mask
                    #     iterates over [0..target_length-1] where
                    #     target_length = cache.get_seq_length() + new_len.
                    #     If we pass absolute cache_position, ALL key
                    #     indices end up <= cache_position[q] → ALL allowed
                    #     → future tokens within the new window leak into
                    #     past queries. LABEL LEAKAGE.
                    #     → `cache_position = [cache_len, cache_len+1, ...]`
                    #     (cache-internal indices, NOT absolute)
                    #
                    # Decoupling these two is the only correct path with
                    # HF's API. Verified separately with a small smoke.
                    n_new = lm_input_ids.shape[1]
                    cache_len_before = (
                        past_key_values.get_seq_length()
                        if past_key_values is not None else 0
                    )
                    cache_position = torch.arange(
                        cache_len_before, cache_len_before + n_new,
                        device=lm_input_ids.device,
                    )
                    position_ids = torch.arange(
                        cache_abs_pos, cache_abs_pos + n_new,
                        device=lm_input_ids.device,
                    ).unsqueeze(0)
                    base_out = self.llama.model(
                        input_ids=lm_input_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_ids=position_ids,
                        use_cache=True,
                    )
                    new_past_key_values = base_out.past_key_values
                else:
                    base_out = self.llama.model(
                        input_ids=lm_input_ids,
                        use_cache=False,
                    )
            finally:
                # Always clear closure to avoid leaking refs across windows.
                mem_inject.memory_fn = None

            full_hiddens = base_out.last_hidden_state                # [BS, *, d_lm]
            # In KV-cache mode `*` is T_window (only new tokens encoded).
            # In rolling-buffer mode `*` is L_lm (full context).
            current_hiddens = full_hiddens[:, -T_window:, :].to(prev_states.dtype)

            # Phase 2 perf fix (#4) — when force_surprise is given (Phase 2
            # response-window TF replay sets it to 0.0 per N2), skip the
            # vocab-sized CE entirely. The earlier code computed a full
            # `_compute_surprise_window` (256 × 128K-vocab CE) and then
            # overwrote `surprise` with zero — pure waste at K samples × N
            # windows per step.
            #
            # We still need logits over T_window for the output (caller
            # uses them for AR sampling). KV-cache mode: full_hiddens IS
            # the T_window of new positions, so logits = lm_head(full_hiddens).
            # Rolling-buffer mode: take the last T_window hiddens. Skip the
            # +1 predecessor hidden since we don't need a CE target on
            # position 0.
            if force_surprise is not None:
                # Synthesize zero surprise + skip CE; still emit logits.
                if use_kv_cache:
                    needed_hidden = full_hiddens
                else:
                    needed_hidden = full_hiddens[:, -T_window:, :]
                needed_logits = self.llama.lm_head(needed_hidden)    # [BS, T_window, V]
                surprise_mean = torch.full(
                    (BS,), float(force_surprise),
                    dtype=full_hiddens.dtype, device=full_hiddens.device,
                )
                surprise_weighted_sum = torch.zeros_like(surprise_mean)
                surprise_count = torch.zeros_like(surprise_mean)
                surprise = surprise_mean.to(prev_states.dtype)
                logits = needed_logits[:, -T_window:, :]
            else:
                # Build `needed_hidden` for surprise CE: T_window+1 hiddens
                # (one predecessor + T_window targets) when available,
                # T_window otherwise (drop target 0 — first window of chunk).
                #
                # KV-cache mode: full_hiddens has shape [BS, T_window, d_lm].
                # The predecessor for target 0 is the last hidden of the
                # previous window — passed as `last_prev_logit_hidden`. If
                # None, we drop target 0 (matches first-window behavior).
                #
                # Rolling-buffer mode: full_hiddens has the predecessor at
                # position [-T_window-1] when L_lm > T_window. We slice
                # min(T_window+1, L_lm) tail positions.
                if use_kv_cache:
                    if last_prev_logit_hidden is not None:
                        needed_hidden = torch.cat(
                            [last_prev_logit_hidden.to(full_hiddens.dtype),
                             full_hiddens], dim=1,
                        )                                            # [BS, T_window+1, d_lm]
                    else:
                        needed_hidden = full_hiddens                 # [BS, T_window, d_lm]
                else:
                    n_needed = min(T_window + 1, full_hiddens.shape[1])
                    needed_hidden = full_hiddens[:, -n_needed:, :]

                needed_logits = self.llama.lm_head(needed_hidden)    # [BS, n_needed, V]

                target_ids = lm_input_ids[:, -T_window:]
                # N6 — _compute_surprise_window returns (mean, weighted_sum,
                # count). Mean is the per-window surprise that the writer
                # consumes (a magnitude per window). Weighted_sum keeps
                # float-mask weights (e.g. prior_loss_weight=0.1) baked in
                # — used by run_chunk for token-weighted chunk-level CE.
                surprise_mean, surprise_weighted_sum, surprise_count = (
                    self._compute_surprise_window(
                        needed_logits, target_ids, target_mask,
                    )
                )
                surprise = surprise_mean.to(prev_states.dtype)

                # Slice the final T_window logits for the output (caller may
                # use them for AR sampling in W3/W4 rollouts).
                logits = needed_logits[:, -T_window:, :]             # [BS, T_window, V]

        # ── 3. WRITE ─────────────────────────────────────────────────
        # Detach surprise: it's a loss-derived scalar from the same logits
        # we're trying to predict. Letting gradient flow back through it
        # would let the write module train W_in/W_out/scale to inflate or
        # deflate logits in service of write strength rather than NTP. The
        # write module sees surprise as a constant signal; its trainable
        # behavior is in mutate_mlp's response, not in shaping surprise.
        #
        # Pad-token contamination guard — when target_mask is provided and
        # any position has weight 0 (pad), replace those positions'
        # hiddens with the LAST REAL position's hidden before the write
        # module sees them. The write module pools current_hiddens by
        # mean (entry MLP) and uses them as cross-attn KV at every hop
        # (line 170 of write_module.py); pad-position hiddens carry
        # garbage signal from Llama (they're outputs of forwarding pad
        # tokens) and would dilute the mean and add noise to attention.
        # Mirrors what pass 1's _ar_sample_one does for partial-generation
        # windows. Only kicks in when target_mask is not None AND there's
        # actually some pad — otherwise no-op.
        write_hiddens = current_hiddens
        if target_mask is not None:
            real = (target_mask > 0)                          # [BS, T_window] bool
            # Per-batch last-real index (0 if no real token in this row).
            # `arange * real.long()` ranks positions; argmax picks the
            # highest-indexed real position. Falls back to 0 (idx 0) for
            # all-pad rows — guarded below.
            arange_t = torch.arange(
                T_window, device=current_hiddens.device,
            ).unsqueeze(0)
            last_real_idx = (arange_t * real.long()).argmax(dim=1)  # [BS]
            last_real_h = current_hiddens.gather(
                1, last_real_idx.view(BS, 1, 1).expand(BS, 1, cfg.d_lm),
            )                                                 # [BS, 1, d_lm]
            write_hiddens = torch.where(
                real.unsqueeze(-1), current_hiddens,
                last_real_h.expand_as(current_hiddens),
            )
            # Degenerate: row has zero real tokens (full-pad chunk).
            # Replace its hiddens with zeros to avoid scattering whatever
            # garbage the lookup happened to hit. Write_module still runs
            # but its pooling sees a zero signal → write is near-noop.
            no_real = (real.sum(dim=1) == 0).view(BS, 1, 1)
            if no_real.any():
                write_hiddens = torch.where(
                    no_real, torch.zeros_like(write_hiddens), write_hiddens,
                )
        new_states, write_visited_ids, _ = self.write_module(
            write_hiddens, surprise.detach(), prev_states, self.manifold,
            hard=hard_routing,
        )

        out = {
            "logits": logits,
            "current_hiddens": current_hiddens,
            "new_states": new_states,
            "read_visited": read_visited_ids,
            "write_visited": write_visited_ids,
            "surprise": surprise,
        }
        if self.llama is not None:
            # N6 — surface raw weighted sum + count per window so run_chunk
            # can aggregate token-weighted across the chunk.
            #
            # CRITICAL: `surprise_weighted_sum` MUST keep its autograd graph
            # — it carries the float-mask weights (e.g. prior_loss_weight=0.1)
            # baked into the scalar. The earlier code surfaced a DETACHED
            # `surprise_sum` and tbptt.run_chunk reconstructed
            # `surprise_mean * count` to recover grad. That reconstruction
            # equals `weighted_sum * count / weight_sum`, which silently
            # cancels float-mask weights when weight_sum != count (i.e.,
            # whenever any non-1.0 weight is in play). Verified numerically:
            # mask=[0.1,0.1,1,1] CE=[1,1,1,1] gave 4.0 instead of 2.2.
            # `surprise_count` stays detached — it's just an integer divisor.
            out["surprise_weighted_sum"] = surprise_weighted_sum
            out["surprise_count"] = surprise_count.detach()
        if use_kv_cache and self.llama is not None:
            out["new_past_key_values"] = new_past_key_values
            out["new_cache_abs_pos"] = cache_abs_pos + lm_input_ids.shape[1]
        return out

    @staticmethod
    def _compute_surprise_window(
        needed_logits: Tensor,        # [BS, T_window+1, V] — logits for the last T_window+1 positions
        target_ids: Tensor,           # [BS, T_window]      — current window's tokens
        target_mask: Tensor | None,   # [BS, T_window] bool — True=include in CE
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-window NTP CE statistics: returns (mean, sum, count) all
        shaped [BS]. Mean is what the writer module's surprise input
        consumes (a magnitude per window); (sum, count) are what
        run_chunk uses to aggregate a TOKEN-WEIGHTED chunk loss (N6 fix).

        Earlier this function returned only `mean`. tbptt.run_chunk then
        summed those means across windows. For W2 with response masking
        concentrated in the last chunk, a sparse last window (1 real
        token) got the same window-level loss weight as a full window
        (256 real tokens) — distorting training toward fragments. The
        new contract surfaces sum + count so tbptt can aggregate as
        `total_sum / total_count` (token-weighted) rather than
        `sum_of_means` (window-equal).
        """
        BS, n_pos, V = needed_logits.shape
        T_window = target_ids.shape[1]
        assert n_pos in (T_window, T_window + 1), (
            f"needed_logits has {n_pos} positions; expected T_window={T_window} "
            f"or T_window+1={T_window+1}"
        )

        # Standard NTP shift: needed_logits[:, :-1] predicts the targets
        # whose predecessors are present.
        shift_logits = needed_logits[:, :-1, :].float()    # [BS, n_pos-1, V] fp32
        if n_pos == T_window + 1:
            # Have predecessor for every target.
            used_target_ids = target_ids                                  # [BS, T_window]
            used_mask = target_mask
        else:
            # n_pos == T_window: very first window, no predecessor for
            # target 0. Predict only targets 1..T_window-1.
            used_target_ids = target_ids[:, 1:]                           # [BS, T_window-1]
            used_mask = target_mask[:, 1:] if target_mask is not None else None

        n_predicted = used_target_ids.shape[1]
        ce_per_tok = F.cross_entropy(
            shift_logits.reshape(-1, V),
            used_target_ids.reshape(-1),
            reduction="none",
        ).reshape(BS, n_predicted)

        if used_mask is not None:
            mask = used_mask.to(ce_per_tok.dtype)
            # Weighted sum: each token contributes weight × CE.
            ce_weighted_sum = (ce_per_tok * mask).sum(dim=1)     # [BS]
            # Sum of weights — used for the per-window MEAN (writer
            # surprise). Divides out the weights to give "typical CE per
            # unit of weight" — what the writer should care about as a
            # magnitude signal.
            weight_sum = mask.sum(dim=1)                         # [BS]
            # Count of nonzero-weight (=valid) positions — used as the
            # divisor for the LOSS aggregation in tbptt.run_chunk. This
            # is what makes prior_loss_weight=0.1 actually scale prior
            # tokens' gradient by 0.1×: the weighted sum carries the 0.1
            # factor, and dividing by COUNT (not weight_sum) preserves it.
            # If we divided by weight_sum the 0.1 would cancel out.
            valid_count = (mask > 0).to(ce_per_tok.dtype).sum(dim=1)
        else:
            ce_weighted_sum = ce_per_tok.sum(dim=1)
            weight_sum = torch.full(
                (BS,), float(n_predicted),
                dtype=ce_per_tok.dtype, device=ce_per_tok.device,
            )
            valid_count = weight_sum.clone()
        # Per-window mean: weighted average for writer surprise.
        ce_mean = ce_weighted_sum / weight_sum.clamp_min(1.0)
        # Return (mean_for_writer, weighted_sum_for_loss, valid_count_for_norm).
        return ce_mean, ce_weighted_sum, valid_count

    # ── parameter accounting ─────────────────────────────────────────

    def trainable_parameters(self) -> dict:
        """Return a dict {name: count} for telemetry."""
        out: dict = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                out[n] = p.numel()
        return out
