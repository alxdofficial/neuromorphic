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
            # memory_fn is set per forward via _set_memory_fn
            self.mem_inject = MemInjectLayer(
                orig_layer=orig_layer,
                d_lm=cfg.d_lm,
                d_mem=cfg.D_concept,
                scale_init=0.1,                   # small but non-zero so memory contributes from
                                                  # step 0; scale=0 zeros all downstream gradient
                                                  # to W_in/W_out/scale/read_attn (chicken-and-egg).
                memory_fn=None,
                bridge_hidden=cfg.bridge_hidden,
            )
            self.host.replace_layer(L, self.mem_inject)
        else:
            # Test mode: no Llama. Skeleton for unit tests of the cycle wiring.
            self.llama = None
            self.host = None
            self.mem_inject = None

        # ── 3. Memory modules (always present) ───────────────────────
        self.manifold = Manifold(cfg)
        self.read_module = ReadTrajectoryGenerator(cfg)
        self.write_module = WriteTrajectoryGenerator(cfg)
        self.read_attn = TrajectoryReadAttn(cfg.D_concept)

    # ── memory wiring helper ─────────────────────────────────────────

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
    ) -> dict:
        """Run one window: read → predict → write.

        The "current window" is the LAST `cfg.T_window` positions of
        `lm_input_ids`. Llama processes the full `lm_input_ids` (the
        rolling 2K-or-less LM context); we slice logits and hidden states
        for just the current window for surprise + write conditioning.
        See plan §4.1 — this is how the deliberate 2K LM context cap is
        implemented.

        Args:
            lm_input_ids:        [BS, L_lm] full LM context. The last
                                 T_window positions are the current window
                                 (whose tokens we predict and whose hidden
                                 states drive the write trajectory).
                                 `L_lm` must be ≥ T_window and ≤
                                 `cfg.effective_lm_context`. The caller
                                 (run_chunk) maintains the rolling buffer.
            prev_window_hiddens: [BS, T_window, d_lm] from previous window.
                                 None at the very first window of a sequence.
            prev_states:         [BS, N, D_concept] manifold state going in.
            target_mask:         [BS, T_window] bool (True = include in
                                 surprise CE), aligned to the current window
                                 (NOT the full LM context). None → all eligible.
            hard_routing:        Gumbel-STE if True, argmax if False.

        Returns:
            dict with keys:
                logits:           [BS, T_window, V] — logits for the current window only
                current_hiddens:  [BS, T_window, d_lm]
                new_states:       [BS, N, D_concept]
                read_visited:     [BS, J, K_read]
                write_visited:    [BS, J, K_write]
                surprise:         [BS] mean per-token CE over the current window
        """
        cfg = self.cfg
        T_window = cfg.T_window
        BS, L_lm = lm_input_ids.shape
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
            # Test-mode fallback: no Llama, no logits, just current_hiddens
            # = random so write has something to chew on. Surprise is 0.
            logits = torch.zeros(BS, T_window, 1, device=prev_states.device)
            current_hiddens = torch.randn(
                BS, T_window, cfg.d_lm,
                dtype=prev_states.dtype, device=prev_states.device,
            )
            surprise = torch.zeros(BS, device=prev_states.device, dtype=prev_states.dtype)
        else:
            # Wire memory_fn for this forward call.
            self.mem_inject.memory_fn = self._build_memory_fn(read_visited)
            try:
                lm_out = self.llama(
                    input_ids=lm_input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
            finally:
                # Always clear the closure to avoid leaking the trajectory tensor
                # into a stale reference across windows.
                self.mem_inject.memory_fn = None

            # Slice to the current-window positions (last T_window).
            full_logits = lm_out.logits                              # [BS, L_lm, V]
            full_hiddens = lm_out.hidden_states[-1]                  # [BS, L_lm, d_lm]
            logits = full_logits[:, -T_window:, :]                   # [BS, T_window, V]
            current_hiddens = full_hiddens[:, -T_window:, :].to(prev_states.dtype)

            # Compute surprise: mean per-token NTP CE on the current window.
            # Use FULL logits + lm_input_ids for the standard shift, then
            # slice to the current window's predictions. The first prediction
            # in the slice uses LM context that came BEFORE the current
            # window (good — that's the whole point of the rolling context).
            surprise = self._compute_surprise(
                full_logits, lm_input_ids, T_window, target_mask,
            ).to(prev_states.dtype)

        # ── 3. WRITE ─────────────────────────────────────────────────
        # Detach surprise: it's a loss-derived scalar from the same logits
        # we're trying to predict. Letting gradient flow back through it
        # would let the write module train W_in/W_out/scale to inflate or
        # deflate logits in service of write strength rather than NTP. The
        # write module sees surprise as a constant signal; its trainable
        # behavior is in mutate_mlp's response, not in shaping surprise.
        new_states, write_visited_ids, _ = self.write_module(
            current_hiddens, surprise.detach(), prev_states, self.manifold,
            hard=hard_routing,
        )

        return {
            "logits": logits,
            "current_hiddens": current_hiddens,
            "new_states": new_states,
            "read_visited": read_visited_ids,
            "write_visited": write_visited_ids,
            "surprise": surprise,
        }

    @staticmethod
    def _compute_surprise(
        full_logits: Tensor,
        lm_input_ids: Tensor,
        T_window: int,
        target_mask: Tensor | None,
    ) -> Tensor:
        """Mean per-token NTP CE over the *current window's* target positions.

        full_logits / lm_input_ids span the full LM context (up to
        effective_lm_context tokens). The current window is the last
        T_window tokens. This function takes the standard NTP shift on
        the full sequence, then slices to just the predictions for the
        current window.

        target_mask (if provided) is a [BS, T_window] mask for the current
        window's tokens; True = include in CE.
        """
        BS, L_lm, V = full_logits.shape
        if L_lm < 2:
            return torch.zeros(BS, device=full_logits.device)

        # Standard NTP shift on the full sequence.
        # logits[i] predicts input_ids[i+1].
        # Cast logits to fp32 for CE — bf16 softmax over V=32K can underflow.
        shift_logits = full_logits[:, :-1, :].contiguous().float()  # [BS, L_lm-1, V] fp32
        shift_targets = lm_input_ids[:, 1:].contiguous()            # [BS, L_lm-1]

        # Slice to the current window's predictions.
        # Current window targets are lm_input_ids[L_lm - T_window : L_lm];
        # in shift_targets these are at positions [L_lm - T_window - 1 : L_lm - 1].
        # That's the last T_window positions of shift_targets.
        # When L_lm == T_window we have only L_lm - 1 = T_window - 1 shift
        # positions — the very first token of the current window has no
        # predecessor to predict from. Take what we have.
        n_take = min(T_window, L_lm - 1)
        win_logits = shift_logits[:, -n_take:, :]                   # [BS, n_take, V]
        win_targets = shift_targets[:, -n_take:]                    # [BS, n_take]
        ce_per_tok = F.cross_entropy(
            win_logits.reshape(-1, V),
            win_targets.reshape(-1),
            reduction="none",
        ).reshape(BS, n_take)

        if target_mask is not None:
            # target_mask is [BS, T_window], aligned to current window.
            # We take its last n_take entries (drop first if we lost one).
            mask = target_mask[:, -n_take:].to(ce_per_tok.dtype)
            ce_sum = (ce_per_tok * mask).sum(dim=1)
            ce_count = mask.sum(dim=1).clamp_min(1.0)
            return ce_sum / ce_count
        else:
            return ce_per_tok.mean(dim=1)

    # ── parameter accounting ─────────────────────────────────────────

    def trainable_parameters(self) -> dict:
        """Return a dict {name: count} for telemetry."""
        out: dict = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                out[n] = p.numel()
        return out
