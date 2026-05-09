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
            # T_window+1 positions we actually need, instead of all L_lm
            # positions. At BS=2 / L_lm=2048 / V=128256 / bf16, the full
            # logit tensor would be ~1 GB; with T_window=256 we drop to
            # ~130 MB and skip an expensive 2048-position matmul through
            # lm_head on every forward.
            try:
                base_out = self.llama.model(
                    input_ids=lm_input_ids,
                    use_cache=False,
                )
            finally:
                # Always clear closure to avoid leaking refs across windows.
                mem_inject.memory_fn = None

            full_hiddens = base_out.last_hidden_state                # [BS, L_lm, d_lm]
            current_hiddens = full_hiddens[:, -T_window:, :].to(prev_states.dtype)

            # We want logits for the LAST T_window+1 hidden positions: the
            # final T_window for downstream sampling/output (positions
            # L_lm-T_window..L_lm-1), plus one extra to the LEFT (position
            # L_lm-T_window-1) so the standard NTP shift can produce
            # predictions for all T_window current-window targets.
            #
            # At the very first window of a sequence there's no rolling
            # context yet, so L_lm == T_window — we take all available
            # hidden states and `_compute_surprise_window` skips the first
            # target (which has no predecessor).
            n_needed = min(T_window + 1, full_hiddens.shape[1])
            needed_hidden = full_hiddens[:, -n_needed:, :]
            needed_logits = self.llama.lm_head(needed_hidden)        # [BS, n_needed, V]

            surprise = self._compute_surprise_window(
                needed_logits, lm_input_ids[:, -T_window:], target_mask,
            ).to(prev_states.dtype)

            # Slice the final T_window logits for the output (caller may
            # use them for AR sampling in W3/W4 rollouts).
            logits = needed_logits[:, -T_window:, :]                 # [BS, T_window, V]

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
    def _compute_surprise_window(
        needed_logits: Tensor,        # [BS, T_window+1, V] — logits for the last T_window+1 positions
        target_ids: Tensor,           # [BS, T_window]      — current window's tokens
        target_mask: Tensor | None,   # [BS, T_window] bool — True=include in CE
    ) -> Tensor:
        """Mean per-token NTP CE over the current window's target positions.

        Uses the standard NTP shift: position t in `needed_logits[:, :-1, :]`
        predicts `target_ids[:, t]`. So `needed_logits` must include one
        extra position to the left of the window (the "previous" token,
        whose prediction is the first target).

        Earlier we received the full L_lm-token logits and shifted +
        sliced inside this function, casting the entire sequence to fp32
        — at L_lm=2048/V=128256 that was ~2 GB of throwaway compute per
        forward. The new contract pre-slices, so we cast just T_window
        positions.
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
            ce_sum = (ce_per_tok * mask).sum(dim=1)
            ce_count = mask.sum(dim=1).clamp_min(1.0)
            return ce_sum / ce_count
        return ce_per_tok.mean(dim=1)

    # ── parameter accounting ─────────────────────────────────────────

    def trainable_parameters(self) -> dict:
        """Return a dict {name: count} for telemetry."""
        out: dict = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                out[n] = p.numel()
        return out
