"""IntegratedLM v2 — Llama backbone + vocabulary-trajectory memory.

Simpler than v1's IntegratedLM:
- No KV-cache / past_key_values plumbing (Wave 1 retrieval doesn't need it).
- No mutable cell-state buffer (VocabularyManifold has no `current_states`).
- No `new_states` return (manifold's edge buffer is the only stateful piece).
- mem_inject still injects via the existing MemInjectLayer machinery.

The injected memory is a sequence of `J * K_read` vocabulary embeddings
gathered along the read trajectory — frozen-vocab points the walker
visited. mem_inject's read_attn cross-attends from per-token query
vectors to this trajectory of vocab tokens.

Carryover from v1: `MemInjectLayer`, `host`, `EntryProjector`,
`TrajectoryReadAttn`. The architectural change is in the manifold +
read/write modules.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM

from src.pretrained.hosts import build_host
from src.pretrained.mem_inject_layer import MemInjectLayer
from src.trajectory_memory.integrated_lm import TrajectoryReadAttn
from src.trajectory_memory.read_module import EntryProjector
from src.trajectory_memory_v2.config import TrajMemV2Config
from src.trajectory_memory_v2.manifold import VocabularyManifold
from src.trajectory_memory_v2.read_module import ReadModule
from src.trajectory_memory_v2.walker import WalkerResult
from src.trajectory_memory_v2.write_module import WriteModule


# Default Llama inject layer (mid-stack). Can override via cfg or arg.
_DEFAULT_INJECT_LAYER_FRAC = 0.5


class IntegratedLMV2(nn.Module):
    """Llama + vocabulary-trajectory memory, wired for Wave 1 and streaming."""

    def __init__(
        self,
        cfg: TrajMemV2Config,
        model_name: str = "meta-llama/Llama-3.2-1B",
        *,
        attach_lm: bool = True,
        llama_dtype: str = "bf16",
        freeze_backbone: bool = True,
        inject_layer_frac: float = _DEFAULT_INJECT_LAYER_FRAC,
    ):
        super().__init__()
        self.cfg = cfg

        # ── 1. Load Llama (or skip in test mode) ────────────────────
        if attach_lm:
            hf_cfg = AutoConfig.from_pretrained(model_name)
            cfg.d_lm = hf_cfg.hidden_size
            cfg.validate()

            dt_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            self.llama = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dt_map[llama_dtype],
            )
            self.host = build_host(self.llama)
            if freeze_backbone:
                self.host.freeze_backbone()

            # Replace mid-stack layer with MemInjectLayer
            n_layers = hf_cfg.num_hidden_layers
            self.inject_layer = int(inject_layer_frac * n_layers)
            assert self.inject_layer < n_layers
            orig_layer = self.host.layer_list()[self.inject_layer]
            mem_inject = MemInjectLayer(
                orig_layer=orig_layer,
                d_lm=cfg.d_lm,
                d_mem=cfg.D_concept,
                scale_init=0.1,
                memory_fn=None,
                bridge_hidden=cfg.D_concept,
            )
            self.host.replace_layer(self.inject_layer, mem_inject)
        else:
            self.llama = None
            self.host = None
            self.inject_layer = None

        # ── 2. Memory modules (always present) ───────────────────────
        self.manifold = VocabularyManifold(cfg)
        # Shared entry projector (Hopfield-tied)
        self.entry_proj = EntryProjector(cfg)
        self.read_module = ReadModule(cfg, entry_proj=self.entry_proj)
        self.write_module = WriteModule(cfg, entry_proj=self.entry_proj)
        # TrajectoryReadAttn: per-token cross-attention from Llama hiddens
        # to the trajectory's J*K vocab embeddings. Carried over from v1.
        self.read_attn = TrajectoryReadAttn(cfg.D_concept)

    # ── memory wiring helpers ────────────────────────────────────────

    def _mem_inject_layer(self) -> MemInjectLayer:
        return self.host.layer_list()[self.inject_layer]

    def _build_memory_fn(self, trajectory_embeds: Tensor):
        """Return a closure for MemInjectLayer.memory_fn.

        trajectory_embeds: [BS, J, K_read, D_concept] — vocab embeddings
        along the read trajectory.
        """
        cfg = self.cfg
        flat_traj = trajectory_embeds.reshape(
            trajectory_embeds.shape[0],
            cfg.J * cfg.K_read,
            cfg.D_concept,
        )

        def memory_fn(h_mem: Tensor) -> Tensor:
            # h_mem: [BS, T, D_concept]; readout: [BS, T, D_concept]
            return self.read_attn(h_mem, flat_traj)

        return memory_fn

    # ── per-window forward ───────────────────────────────────────────

    def forward_window(
        self,
        lm_input_ids: Tensor,                            # [BS, T]
        prev_window_hiddens: Optional[Tensor] = None,    # [BS, T_prev, d_lm]
        *,
        hard_routing: bool = True,
        write_mode: str = "passage",                     # "passage" or "qa"
        read_conditioning_hiddens: Optional[Tensor] = None,
    ) -> dict:
        """One window's forward.

        Modes:
          - "passage" mode: the window is a passage/context to write.
            Runs WRITE trajectories conditioned on this window's Llama
            hiddens. Then runs Llama on this window (with read injection
            from prev_window_hiddens if available, for next-token loss).
          - "qa" mode: the window is a question+answer. Runs READ
            trajectory conditioned on `read_conditioning_hiddens` (the
            question's zero-memory hiddens). Llama predicts answer with
            memory inject from the read trajectory.

        Args:
            lm_input_ids: [BS, T] token ids
            prev_window_hiddens: [BS, T_prev, d_lm] — used for streaming
                read conditioning (Wave 2+) and as the read input for
                this window's mem_inject.
            hard_routing: STE (training) or argmax (eval).
            write_mode: "passage" or "qa" — see above.
            read_conditioning_hiddens: explicit override for read
                conditioning. Used in Wave 1 to pass the question's
                zero-memory hiddens. If None, falls back to
                prev_window_hiddens.

        Returns:
            dict with:
                logits: [BS, T, V] from Llama
                current_hiddens: [BS, T, d_lm] — this window's Llama hiddens
                read_visited_ids: [BS, J, K_read] long
                write_visited_ids: [BS, J, K_write] long (in passage mode)
                read_step_queries: [BS, J, K_read, D] — for per-step contrastive
                write_step_queries: [BS, J, K_write, D] (passage mode)
                aux_load_balance, aux_z_loss: scalar aux losses
        """
        cfg = self.cfg
        BS, T = lm_input_ids.shape
        device = lm_input_ids.device

        read_result: Optional[WalkerResult] = None
        write_result: Optional[WalkerResult] = None

        # ── 1. Run Llama on this window to get hiddens ──────────────
        # In v2 we always need this window's hiddens (either for writing
        # or as context for reads). We do NOT skip llama for writes.
        if self.llama is None:
            # Test mode — synthesize hiddens
            current_hiddens = torch.randn(
                BS, T, cfg.d_lm, device=device, dtype=torch.float32,
            )
            logits = torch.zeros(BS, T, 100, device=device)
            # In test mode we still want reads if prev/read_cond is given
            read_cond = (
                read_conditioning_hiddens
                if read_conditioning_hiddens is not None
                else prev_window_hiddens
            )
            if read_cond is not None:
                read_result = self.read_module(
                    read_cond, self.manifold, hard=hard_routing,
                )
        else:
            # Set memory_fn for this forward (controlled below)
            mem_inject = self._mem_inject_layer()

            # Build the read trajectory FIRST (for the read injection
            # during the Llama forward), then run Llama.
            read_cond = (
                read_conditioning_hiddens
                if read_conditioning_hiddens is not None
                else prev_window_hiddens
            )
            if read_cond is not None:
                read_result: WalkerResult = self.read_module(
                    read_cond, self.manifold, hard=hard_routing,
                )
                # Embed memory_fn
                mem_inject.memory_fn = self._build_memory_fn(read_result.visited_embeds)
            else:
                # No prior context — disable memory for this window
                mem_inject.memory_fn = self._zero_readout
                read_result = None

            # Run Llama
            base_out = self.llama.model(
                input_ids=lm_input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            full_hiddens = base_out.last_hidden_state              # [BS, T, d_lm]
            current_hiddens = full_hiddens
            logits = self.llama.lm_head(full_hiddens)             # [BS, T, V]

        # ── 2. Run write trajectory if requested ─────────────────────
        if write_mode == "passage":
            write_result = self.write_module(
                current_hiddens, self.manifold, hard=hard_routing,
            )

        # ── 3. Package output ───────────────────────────────────────
        out = {
            "logits": logits,
            "current_hiddens": current_hiddens,
        }
        if read_result is not None:
            out["read_visited_ids"] = read_result.visited_ids
            out["read_visited_embeds"] = read_result.visited_embeds
            out["read_step_queries"] = read_result.step_queries
            out["read_entry_logits_max"] = read_result.entry_logits_max
        if write_result is not None:
            out["write_visited_ids"] = write_result.visited_ids
            out["write_visited_embeds"] = write_result.visited_embeds
            out["write_step_queries"] = write_result.step_queries
            out["write_entry_logits_max"] = write_result.entry_logits_max

        # Aux losses
        aux_lb = torch.zeros((), device=device)
        aux_z = torch.zeros((), device=device)
        if read_result is not None:
            aux_lb = aux_lb + read_result.aux_lb
            aux_z = aux_z + read_result.aux_z
        if write_result is not None:
            aux_lb = aux_lb + write_result.aux_lb
            aux_z = aux_z + write_result.aux_z
        out["aux_load_balance"] = aux_lb
        out["aux_z_loss"] = aux_z

        return out

    def _zero_readout(self, h_mem: Tensor) -> Tensor:
        """Default no-op memory function (when no read trajectory is provided)."""
        return torch.zeros_like(h_mem)
