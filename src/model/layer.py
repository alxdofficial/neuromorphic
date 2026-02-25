"""
Layer — scan-friendly recurrence with procedural memory and FFN.

One Layer per (block, layer). Contains the affine recurrence
h_t = a_t * (carry * h_{t-1}) + b_t, a post-recurrence FFN for
nonlinear per-position processing, and a PM instance.

Gate inputs: concat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])
giving input_dim = 4*D_h + surprise_dim, where surprise_dim is D_pc
when PCM is enabled (vector surprise) or 1 (scalar surprise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig
from .procedural_memory import ProceduralMemory, PMNeuromodulator
from .scan import parallel_affine_scan, _HAS_FLA_HGRN, fla_hgrn_scan
from .utils import StateMixin, runtime_state_dtype


class Layer(nn.Module, StateMixin):
    _state_tensor_names = ["h"]

    def __init__(self, config: ModelConfig, block_idx: int, layer_idx: int):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.layer_idx = layer_idx
        D_h = config.D_h

        # Procedural memory for this layer
        self.pm = ProceduralMemory(config)
        self.pm_neuromodulator = PMNeuromodulator(config)

        # Gate input: x_block + y_pm + y_wm_proj + y_em_proj + surprise
        # surprise_dim is D_pc (vector δ) when PCM is enabled, else 1 (scalar)
        self.surprise_dim = config.D_pc if config.pcm_enabled else 1
        input_dim = 4 * D_h + self.surprise_dim

        # Fused gate: single GEMM produces both retention (sigmoid) and update (tanh)
        self.gate_ab = nn.Linear(input_dim, 2 * D_h)

        # Zero-init the surprise (δ) columns of gate weights so PCM surprise
        # has no effect at initialization. Prevents sudden distribution shift
        # in gate activations when z_hat first becomes nonzero after the first
        # span boundary. (Matches zero-init of W_gain in PCM.)
        if config.pcm_enabled:
            with torch.no_grad():
                self.gate_ab.weight[:, -self.surprise_dim:].zero_()

        # Per-layer output projection (spec §7.4)
        self.W_o = nn.Linear(D_h, D_h)

        self.norm = nn.LayerNorm(D_h)
        self.drop_resid = nn.Dropout(config.dropout)

        # Post-recurrence FFN for nonlinear per-position processing.
        # The recurrence mixes temporal info; the FFN adds reasoning depth.
        if config.ffn_expansion > 0:
            d_ff = D_h * config.ffn_expansion
            self.ffn_norm = nn.LayerNorm(D_h)
            self.ffn = nn.Sequential(
                nn.Linear(D_h, d_ff),
                nn.GELU(approximate="tanh"),
                nn.Dropout(config.dropout),
                nn.Linear(d_ff, D_h),
            )
        else:
            self.ffn_norm = None
            self.ffn = None

        # Recurrent hidden state (lazily initialized)
        self.h: Tensor = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Handle loading old checkpoints.

        Supports two migration paths:
        1. Old separate gate_a/gate_b → fused gate_ab
        2. Old gate_ab with surprise_dim=1 → new gate_ab with surprise_dim=D_pc
           (pads with zeros so PCM surprise columns start at zero)
        """
        wa = prefix + "gate_a.weight"
        wb = prefix + "gate_b.weight"
        ba = prefix + "gate_a.bias"
        bb = prefix + "gate_b.bias"
        wab = prefix + "gate_ab.weight"
        bab = prefix + "gate_ab.bias"
        if wa in state_dict and wab not in state_dict:
            state_dict[wab] = torch.cat(
                [state_dict.pop(wa), state_dict.pop(wb)], dim=0
            )
            state_dict[bab] = torch.cat(
                [state_dict.pop(ba), state_dict.pop(bb)], dim=0
            )

        # Pad gate_ab input dim if checkpoint has fewer columns than current
        # (e.g. loading surprise_dim=1 checkpoint into pcm_enabled model)
        if wab in state_dict:
            saved_in = state_dict[wab].shape[1]
            current_in = self.gate_ab.in_features
            if saved_in < current_in:
                pad_cols = current_in - saved_in
                state_dict[wab] = torch.cat([
                    state_dict[wab],
                    torch.zeros(state_dict[wab].shape[0], pad_cols,
                                device=state_dict[wab].device,
                                dtype=state_dict[wab].dtype),
                ], dim=1)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _lazy_init(self, BS: int, device: torch.device):
        self.h = torch.zeros(
            BS, self.config.D_h, device=device, dtype=runtime_state_dtype(device)
        )

    def step(self, x_block: Tensor, y_pm: Tensor, y_wm_proj: Tensor,
             y_em_proj: Tensor, surprise: Tensor, carry: Tensor,
             ffn_gain: Tensor = None, collect: bool = False):
        """Process one token through this layer.

        Args:
            x_block: [BS, D_h] — block input
            y_pm: [BS, D_h] — PM output (zeros if disabled)
            y_wm_proj: [BS, D_h] — WM output projected to D_h
            y_em_proj: [BS, D_h] — EM output projected to D_h (zeros if disabled)
            surprise: [BS, surprise_dim] — surprise signal (D_pc vector or scalar)
            carry: [BS, 1] — 0 at doc boundaries, 1 otherwise
            ffn_gain: [BS, D_h] — optional PCM multiplicative gain for FFN input
            collect: bool — if True, return (output, stats_dict)

        Returns:
            output: [BS, D_h] — layer output
            stats: dict (only when collect=True) — gate_a, gate_b, h_norm
        """
        BS = x_block.shape[0]
        device = x_block.device

        if self.h is None:
            self._lazy_init(BS, device)

        # Fuse inputs
        u = torch.cat([x_block, y_pm, y_wm_proj, y_em_proj, surprise], dim=-1)

        # Fused gate computation: one GEMM, then split + activate
        ab = self.gate_ab(u)                             # [BS, 2*D_h]
        a_raw, b_raw = ab.chunk(2, dim=-1)
        a = torch.sigmoid(a_raw)                         # [BS, D_h] retention
        b = torch.tanh(b_raw)                            # [BS, D_h] update

        # Affine recurrence with carry mask for doc boundaries
        self.h = a * (carry * self.h) + b

        # Output projection + residual + LayerNorm (spec §7.4)
        output = self.norm(self.drop_resid(self.W_o(self.h)) + x_block)

        # Post-recurrence FFN (pre-norm residual)
        if self.ffn is not None:
            ffn_input = self.ffn_norm(output)
            if ffn_gain is not None:
                ffn_input = ffn_input * ffn_gain
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                ffn_out = grad_checkpoint(
                    self.ffn, ffn_input, use_reentrant=False
                )
                output = output + self.drop_resid(ffn_out)
            else:
                output = output + self.drop_resid(self.ffn(ffn_input))

        if collect:
            stats = {
                "gate_a": a.detach(),
                "gate_b": b.detach(),
                "h_norm": self.h.detach().norm(dim=-1).mean().item(),
            }
            return output, stats
        return output

    def _forward_span_core(self, x_all: Tensor, y_pm_all: Tensor,
                           y_wm_proj_all: Tensor, y_em_proj_all: Tensor,
                           surprise_all: Tensor, carry_all: Tensor,
                           ffn_gain_all: Tensor = None) -> Tensor:
        """Compiled inner loop: gates + scan + output proj + FFN.

        No lazy init, no .item() calls, no data-dependent branches —
        safe for torch.compile(fullgraph=True).

        Args:
            surprise_all: [BS, P_b, surprise_dim] — pre-expanded surprise signal.
                When PCM is enabled: [BS, P_b, D_pc] (vector δ per token).
                When PCM is disabled: [BS, P, 1] (scalar surprise, expanded by caller).
            ffn_gain_all: [BS, P_b, D_h] — optional PCM multiplicative gain for FFN.
        """
        BS, P, D_h = x_all.shape

        # Fuse inputs: [BS, P, 4*D_h + surprise_dim]
        u = torch.cat([x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
                        surprise_all], dim=-1)

        # Fused gate computation: one GEMM, then split + activate
        ab = self.gate_ab(u)                         # [BS, P, 2*D_h]
        a_raw, b_raw = ab.chunk(2, dim=-1)
        b = torch.tanh(b_raw)                        # [BS, P, D_h]

        if self.config.use_fla_kernels and _HAS_FLA_HGRN and a_raw.is_cuda:
            # FLA HGRN path: log-space gates → fused Triton scan kernel
            g_log = F.logsigmoid(a_raw)              # [BS, P, D_h]

            # Apply carry mask in log space: where carry=0 (doc boundary),
            # set gate to -30 (exp(-30) ≈ 0, effectively zeroing retention)
            carry_bool = carry_all.to(torch.bool)    # [BS, P, 1]
            g_log = torch.where(carry_bool, g_log,
                                g_log.new_full((), -30.0))

            h_all, h_final = fla_hgrn_scan(
                g_log, b, self.h.to(g_log.dtype)
            )
            self.h = h_final.to(self.h.dtype)

            # Cache activated gates for collect path
            self._last_gate_a = torch.sigmoid(a_raw)
            self._last_gate_b = b
        else:
            # Pure PyTorch fallback (torch.compile fuses the loop)
            a = torch.sigmoid(a_raw)                 # [BS, P, D_h]

            # Cache gates for collect path
            self._last_gate_a = a
            self._last_gate_b = b

            # Apply carry mask (zero at doc boundaries)
            a_eff = a * carry_all.to(a.dtype)        # [BS, P, D_h]

            # Parallel affine scan
            h_all = parallel_affine_scan(a_eff, b, self.h.to(a_eff.dtype))
            self.h = h_all[:, -1].to(self.h.dtype)

        # Batched output projection + residual + LayerNorm
        output = self.norm(self.drop_resid(self.W_o(h_all)) + x_all)

        # Batched post-recurrence FFN
        if self.ffn is not None:
            ffn_input = self.ffn_norm(output)
            if ffn_gain_all is not None:
                ffn_input = ffn_input * ffn_gain_all
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                ffn_out = grad_checkpoint(
                    self.ffn, ffn_input, use_reentrant=False
                )
                output = output + self.drop_resid(ffn_out)
            else:
                output = output + self.drop_resid(self.ffn(ffn_input))

        return output

    def forward_span(self, x_all: Tensor, y_pm_all: Tensor,
                     y_wm_proj_all: Tensor, y_em_proj_all: Tensor,
                     surprise_all: Tensor, carry_all: Tensor,
                     ffn_gain_all: Tensor = None,
                     collect: bool = False) -> Tensor:
        """Process P tokens in parallel through this layer.

        Args:
            x_all: [BS, P_b, D_h] — block input for all tokens
            y_pm_all: [BS, P_b, D_h] — PM output for all tokens
            y_wm_proj_all: [BS, P_b, D_h] — WM output projected to D_h
            y_em_proj_all: [BS, P_b, D_h] — EM output projected to D_h
            surprise_all: [BS, P_b, surprise_dim] or [BS, surprise_dim] — surprise.
                If 2D, auto-expanded to [BS, P_b, surprise_dim] for backward compat.
            carry_all: [BS, P_b, 1] — 0 at doc boundaries, 1 otherwise
            ffn_gain_all: [BS, P_b, D_h] — optional PCM FFN gain modulation
            collect: if True, return (output, gate_stats) tuple

        Returns:
            output_all: [BS, P_b, D_h] — layer outputs for all tokens
            (if collect: also returns gate_stats dict)
        """
        # Auto-expand 2D surprise to 3D for backward compatibility
        if surprise_all.dim() == 2:
            P = x_all.shape[1]
            surprise_all = surprise_all.unsqueeze(1).expand(-1, P, -1)

        output = self._forward_span_core(
            x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
            surprise_all, carry_all, ffn_gain_all,
        )

        # Store full output sequence for post-forward eligibility/EM.
        self._last_h_all = output

        if collect:
            stats = {
                "gate_a": self._last_gate_a.detach().mean(dim=1),  # [BS, D_h]
                "gate_b": self._last_gate_b.detach().mean(dim=1),  # [BS, D_h]
                "h_norm": self.h.detach().norm(dim=-1).mean(),  # scalar tensor
            }
            return output, stats
        # Free gate tensors when not collecting (saves 2 * BS*P*D_h memory)
        self._last_gate_a = None
        self._last_gate_b = None
        return output
