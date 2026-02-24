"""
Layer — scan-friendly recurrence with procedural memory and FFN.

One Layer per (block, layer). Contains the affine recurrence
h_t = a_t * (carry * h_{t-1}) + b_t, a post-recurrence FFN for
nonlinear per-position processing, and a PM instance.

Gate inputs: concat([x_block, y_pm, y_wm_proj, y_em_proj, surprise])
giving input_dim = 4*D_h + 1.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig
from .procedural_memory import ProceduralMemory, PMNeuromodulator
from .scan import parallel_affine_scan
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
        input_dim = 4 * D_h + 1

        # Fused gate: single GEMM produces both retention (sigmoid) and update (tanh)
        self.gate_ab = nn.Linear(input_dim, 2 * D_h)

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
        """Handle loading old checkpoints with separate gate_a/gate_b."""
        wa = prefix + "gate_a.weight"
        wb = prefix + "gate_b.weight"
        ba = prefix + "gate_a.bias"
        bb = prefix + "gate_b.bias"
        wab = prefix + "gate_ab.weight"
        if wa in state_dict and wab not in state_dict:
            state_dict[wab] = torch.cat(
                [state_dict.pop(wa), state_dict.pop(wb)], dim=0
            )
            state_dict[prefix + "gate_ab.bias"] = torch.cat(
                [state_dict.pop(ba), state_dict.pop(bb)], dim=0
            )
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
             collect: bool = False):
        """Process one token through this layer.

        Args:
            x_block: [BS, D_h] — block input
            y_pm: [BS, D_h] — PM output (zeros if disabled)
            y_wm_proj: [BS, D_h] — WM output projected to D_h
            y_em_proj: [BS, D_h] — EM output projected to D_h (zeros if disabled)
            surprise: [BS, 1] — surprise signal
            carry: [BS, 1] — 0 at doc boundaries, 1 otherwise
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
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                ffn_out = grad_checkpoint(
                    self.ffn, self.ffn_norm(output), use_reentrant=False
                )
                output = output + self.drop_resid(ffn_out)
            else:
                output = output + self.drop_resid(self.ffn(self.ffn_norm(output)))

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
                           surprise_span: Tensor, carry_all: Tensor) -> Tensor:
        """Compiled inner loop: gates + scan + output proj + FFN.

        No lazy init, no .item() calls, no data-dependent branches —
        safe for torch.compile(fullgraph=True).
        """
        BS, P, D_h = x_all.shape

        # Expand frozen surprise to [BS, P, 1]
        surprise_all = surprise_span.unsqueeze(1).expand(BS, P, 1)

        # Fuse inputs: [BS, P, 4*D_h + 1]
        u = torch.cat([x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
                        surprise_all], dim=-1)

        # Fused gate computation: one GEMM, then split + activate
        ab = self.gate_ab(u)                         # [BS, P, 2*D_h]
        a_raw, b_raw = ab.chunk(2, dim=-1)
        a = torch.sigmoid(a_raw)                     # [BS, P, D_h]
        b = torch.tanh(b_raw)                        # [BS, P, D_h]

        # Cache raw gates for collect path (reference only, no compute)
        self._last_gate_a = a
        self._last_gate_b = b

        # Apply carry mask (zero at doc boundaries)
        # Cast carry to activation dtype (carry_all arrives as fp32 from bool→float)
        a_eff = a * carry_all.to(a.dtype)   # [BS, P, D_h]

        # Parallel affine scan (cast fp32 state to activation dtype for
        # memory-efficient scan; restore fp32 after for inter-span precision)
        h_all = parallel_affine_scan(a_eff, b, self.h.to(a_eff.dtype))  # [BS, P, D_h]

        # Update state to last token, preserving configured runtime precision.
        self.h = h_all[:, -1].to(self.h.dtype)

        # Batched output projection + residual + LayerNorm
        output = self.norm(self.drop_resid(self.W_o(h_all)) + x_all)

        # Batched post-recurrence FFN
        if self.ffn is not None:
            if self.config.gradient_checkpointing and torch.is_grad_enabled():
                ffn_out = grad_checkpoint(
                    self.ffn, self.ffn_norm(output), use_reentrant=False
                )
                output = output + self.drop_resid(ffn_out)
            else:
                output = output + self.drop_resid(self.ffn(self.ffn_norm(output)))

        return output

    def forward_span(self, x_all: Tensor, y_pm_all: Tensor,
                     y_wm_proj_all: Tensor, y_em_proj_all: Tensor,
                     surprise_span: Tensor, carry_all: Tensor,
                     collect: bool = False) -> Tensor:
        """Process P tokens in parallel through this layer.

        Args:
            x_all: [BS, P, D_h] — block input for all tokens
            y_pm_all: [BS, P, D_h] — PM output for all tokens
            y_wm_proj_all: [BS, P, D_h] — WM output projected to D_h
            y_em_proj_all: [BS, P, D_h] — EM output projected to D_h
            surprise_span: [BS, 1] — frozen surprise for this span
            carry_all: [BS, P, 1] — 0 at doc boundaries, 1 otherwise
            collect: if True, return (output, gate_stats) tuple

        Returns:
            output_all: [BS, P, D_h] — layer outputs for all tokens
            (if collect: also returns gate_stats dict)
        """
        output = self._forward_span_core(
            x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
            surprise_span, carry_all,
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
        return output
